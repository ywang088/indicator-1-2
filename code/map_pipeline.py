"""Sparse typed graph, isolated replay, independent reload, honest byte accounting."""
import argparse, hashlib, json, sqlite3, subprocess, sys, math, shutil
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from perception import sha, dump, pointmap

OBJECT_DEFAULTS={'position':None,'region_votes':{},'region_text_status':'no_readable_column_label','vote_view_baseline_m':0,'region':None,'region_assignment':'unknown_or_ambiguous','semantic_status':'unreviewed_detector_candidate','identity_status':'spatial_association_candidate'}

def edge(db,a,b,kind,attr=None):
 db.execute('INSERT OR IGNORE INTO edges VALUES (?,?,?,?)',(a,b,kind,json.dumps(attr or {},ensure_ascii=False,separators=(',',':'))))
def node(db,id,kind,attr):
 if kind=='object':attr={k:v for k,v in attr.items() if k not in OBJECT_DEFAULTS or v!=OBJECT_DEFAULTS[k]}
 db.execute('INSERT INTO nodes VALUES (?,?,?)',(id,kind,json.dumps(attr,ensure_ascii=False,allow_nan=False,separators=(',',':'))))
def inside(p,poly):
 x,y=p;ok=False
 for i,a in enumerate(poly):
  b=poly[i-1]
  if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:ok=not ok
 return ok

def make_graph(c,rows,path,trace,scope):
 db=sqlite3.connect(path);db.executescript('''PRAGMA foreign_keys=ON;
 CREATE TABLE nodes(id TEXT PRIMARY KEY,kind TEXT NOT NULL,attributes TEXT NOT NULL) WITHOUT ROWID;
 CREATE TABLE edges(source TEXT REFERENCES nodes(id),target TEXT REFERENCES nodes(id),kind TEXT,attributes TEXT,PRIMARY KEY(source,target,kind)) WITHOUT ROWID;
 CREATE INDEX edges_target ON edges(target);
 CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
 ''');db.execute('INSERT INTO metadata VALUES (?,?)',('scope',json.dumps(scope)));db.execute('INSERT INTO metadata VALUES (?,?)',('object_defaults',json.dumps(OBJECT_DEFAULTS)))
 # Runtime graph has exactly four permitted node kinds; scene/floor remain scope metadata.
 fmap={f['id']:f for f in c['frames']};objects=[];previous=None;events=[]
 from color_topology import make_topology
 topology=make_topology(c) if c.get('incremental_topology') else None
 for row in sorted(rows,key=lambda r:(fmap[r['frame_id']]['timestamp'],r['frame_id'])):
  fid=row['frame_id'];f=fmap[fid];fn='frame:'+fid;T=np.array(f['T_world_camera']);node(db,fn,'frame',dict(frame_id=fid,timestamp=f['timestamp'],T_world_camera=f['T_world_camera'],source_sha256=f['image_sha256']))
  if previous:edge(db,previous,fn,'temporal_next')
  previous=fn;seen=set();created=[];updates=[];pillar_observations=[]
  for d in row['detections']:
   if d['score']<c['min_confidence']:continue
   pos=d.get('position');match=None
   # Dynamic objects are not persistent identities; a nearby car may be another car.
   if pos is not None and d['label'] not in ('vehicle','person'):
    candidates=[o for o in objects if o['label']==d['label'] and o['position'] is not None and o['id'] not in seen and np.linalg.norm(np.array(o['position'])-pos)<=c['merge_radius_m']]
    if candidates:match=min(candidates,key=lambda o:np.linalg.norm(np.array(o['position'])-pos))
   obs=dict(frame_id=fid,detection_id=d['id'],score=d['score'],bbox=d['bbox'],texts=d.get('texts',[]),position=pos,geometry=d.get('geometry',{}))
   if d.get('pillar_color'):obs['pillar_color']=d['pillar_color']
   if d.get('visual_review'):obs.update(visual_review=d['visual_review'],semantic_status=d.get('semantic_status'))
   if match is None:
    match=dict(id='object:'+d['id'],label=d['label'],semantic_status=d.get('semantic_status','unreviewed_detector_candidate'),position=pos,observations=[],identity_status='single_observation' if d['label'] in ('vehicle','person') else 'spatial_association_candidate');objects.append(match);created.append(match['id'])
   else:updates.append(match['id'])
   match['observations'].append(obs)
   pillar_observations.append((match['id'],d))
   if d.get('semantic_status')=='assistant_visible_door_unverified_connection':match['semantic_status']=d['semantic_status']
   seen.add(match['id'])
   pp=[x['position'] for x in match['observations'] if x['position'] is not None]
   if pp:match['position']=np.median(pp,axis=0).tolist()
  topology_event=topology.update(f,pillar_observations) if topology else {}
  events.append(dict(frame_id=fid,timestamp=f['timestamp'],input_sha256=f['image_sha256'],new_objects=created,updated_objects=updates,objects_after=len(objects),**topology_event))
 # Region identities use causal pillar evidence; optional paint identities bind to later letters.
 regions={};floors={'unknown'}
 if topology:
  regions=topology.regions
  for o in objects:
   if o['id'] in topology.zone_by_object:o['sign_region']=topology.zone_by_object[o['id']];o['region_text_status']=regions[o['sign_region']].get('status','causal_column_text_evidence')
 else:
  for o in objects:
   votes=defaultdict(list);floor_votes=defaultdict(set)
   if o['label']=='column':
    for ob in o['observations']:
     for t in ob['texts']:
      v=t.get('parsed')
      if not v:continue
      if v.get('floor'):floor_votes[v['floor']].add(ob['frame_id'])
      if v.get('zone'):votes[v['zone']].append((ob['frame_id'],v,t['score']))
   scores={k:len({v[0] for v in vs}) for k,vs in votes.items()};o['region_votes']=scores
   winner=max(scores,key=scores.get) if scores else None
   supporting=sorted({v[0] for v in votes[winner]}) if winner else []
   camera_positions=[np.array(fmap[f]['T_world_camera'])[:3,3] for f in supporting]
   baseline=max((float(np.linalg.norm(a-b)) for a in camera_positions for b in camera_positions),default=0)
   accepted=winner and scores[winner]>=c['region_vote_min_frames'] and scores[winner]/max(1,sum(scores.values()))>=.75 and baseline>=c['region_vote_min_motion_m']
   o['region_text_status']='multiview_candidate' if accepted else ('conflicting_or_insufficient' if scores else 'no_readable_column_label')
   o['vote_view_baseline_m']=baseline
   if accepted:
    floor=next(iter(floor_votes)) if len(floor_votes)==1 and len(next(iter(floor_votes.values())))>=2 else 'unknown'
    rid=f'region:{floor}:{winner}';o['sign_region']=rid
    if rid not in regions:regions[rid]=dict(label=winner,floor=floor,status='column_text_multiview_candidate',anchors=[],positions=[],boundary_status='not_geofenced')
    regions[rid]['anchors'].append(o['id'])
    if o['position'] is not None:regions[rid]['positions'].append(o['position'])
    floors.add(floor)
 # Explicit frozen boundaries may only be supplied with source provenance.
 for reg in c.get('regions',[]):
  if not reg.get('source') or not reg.get('polygon_xz') or not reg.get('floor'):raise ValueError('frozen region needs source, polygon_xz and floor')
  rid=reg['id'];regions[rid]=dict(reg,status='frozen_boundary',positions=[],anchors=[],boundary_status='frozen_polygon');floors.add(reg['floor'])
 for rid,r in regions.items():node(db,rid,'region',r)
 def region_at(p,timestamp=None):
  if p is None:return None,None
  matches=[rid for rid,r in regions.items() if r.get('polygon_xz') and inside([p[0],p[2]],r['polygon_xz'])]
  if len(matches)==1:return matches[0],'frozen_boundary'
  if len(matches)>1:return None,'overlapping_boundaries'
  positions={rid:r['positions'] for rid,r in regions.items()}
  if topology and timestamp is not None:
   positions={rid:[e['position'] for e in topology.evidence[rid] if e['position'] is not None and e['timestamp']<=timestamp] for rid,r in regions.items() if fmap[r['established_frame']]['timestamp']<=timestamp}
  distances={rid:min(np.linalg.norm((np.array(q)-p)[[0,2]]) for q in ps) for rid,ps in positions.items() if ps}
  nearest=sorted(distances,key=distances.get)
  if nearest and distances[nearest[0]]<=c['region_radius_m'] and (len(nearest)==1 or distances[nearest[1]]-distances[nearest[0]]>2):return nearest[0],'nearest_column_candidate'
  return None,'unknown_or_ambiguous'
 for o in objects:
  rid,method=region_at(o['position']);o['region']=o.get('sign_region',rid);o['region_assignment']= 'column_label' if o.get('sign_region') else method
  if o.get('sign_region') and regions[o['sign_region']].get('color'):
   reg=regions[o['sign_region']];direct=any((t.get('parsed') or {}).get('zone')==reg.get('letter') and reg.get('letter') for ob in o['observations'] for t in ob['texts'])
   o['region_assignment']='column_label' if direct else 'column_color_binding' if reg.get('letter') else 'column_color_provisional'
  node(db,o['id'],'object',{k:v for k,v in o.items() if k not in ('id','observations')})
  for ob in o['observations']:edge(db,'frame:'+ob['frame_id'],o['id'],'observes',ob)
  if o['region']:edge(db,o['region'],o['id'],'contains' if method=='frozen_boundary' else 'candidate_contains',dict(method=o['region_assignment']))
 frame_regions={};frame_methods={}
 for row in rows:
  fid=row['frame_id'];rid,method=region_at(np.array(fmap[fid]['T_world_camera'])[:3,3],fmap[fid]['timestamp'])
  explicit=c.get('frame_region_evidence',{}).get(fid)
  if explicit:
   if explicit.get('region') not in regions or explicit.get('image_sha256')!=fmap[fid]['image_sha256'] or not explicit.get('reviewer') or not explicit.get('basis') or explicit.get('camera_in_region_confirmed') is not True:raise ValueError('invalid reviewed camera region evidence')
   rid,method=explicit['region'],'reviewed_camera_region'
  frame_regions[fid]=rid;frame_methods[fid]=method
  if rid:edge(db,rid,'frame:'+fid,'contains' if method=='frozen_boundary' else 'candidate_contains',dict(method=method))
 # Detection of a door does NOT create a portal. Only two-ended connections enter this layer.
 from portal_connections import trajectory_transitions,validate_reviewed_connection
 transitions,pending,trajectory_audit=trajectory_transitions(c,frame_regions,frame_methods,fmap)
 for t in transitions:
  node(db,t['id'],'portal',t)
  for region in t['regions']:edge(db,region,t['id'],'connects')
  for fid in t['evidence_frames']:edge(db,'frame:'+fid,t['id'],'supports_connection')
 if topology:
  for p in topology.portals.values():
   node(db,p['id'],'portal',p)
   for rid in p['regions']:edge(db,rid,p['id'],'connects',dict(status=p['status']))
   for fid in p['evidence_frames']:edge(db,'frame:'+fid,p['id'],'supports_connection')
   for obj in p['anchors']:edge(db,p['id'],obj,'anchored_by')
 seen_portals={t['id'] for t in transitions}
 for p in c.get('portals',[]):
  validate_reviewed_connection(p,c,fmap)
  if p['id'] in seen_portals:raise ValueError('duplicate portal ID')
  seen_portals.add(p['id'])
  if not set(p['regions'])<=set(regions) or not set(p['evidence_frames'])<=set(frame_regions):continue
  node(db,p['id'],'portal',dict(p,status='reviewed_connection'))
  for region in p['regions']:edge(db,region,p['id'],'connects')
  for fid in p['evidence_frames']:edge(db,'frame:'+fid,p['id'],'supports_connection')
  if p.get('object'):
   if not db.execute("SELECT 1 FROM nodes WHERE id=? AND kind='object'",(p['object'],)).fetchone():raise ValueError('portal object missing')
   edge(db,p['id'],p['object'],'represented_by')
 # Keep unconfirmed changes in the audit, never disguised as traversable graph edges.
 dump(Path(trace).with_name('connection_audit.json'),dict(trajectory_audit=trajectory_audit,unconfirmed_transitions=pending,verified_transitions=transitions,incremental_portals=list(topology.portals.values()) if topology else [],incremental_pending=topology.pending if topology else [],door_detection_is_portal=False,cross_trajectory_stitching=False,door_objects_without_verified_connection=sum(o['label'] in ('door','elevator') for o in objects)))
 db.commit();db.execute('VACUUM');portal_count=db.execute("SELECT count(*) FROM nodes WHERE kind='portal'").fetchone()[0];db.close()
 with open(trace,'w') as f:
  for e in events:f.write(json.dumps(e,ensure_ascii=False)+'\n')
 return dict(frames=len(rows),objects=len(objects),regions=len(regions),portals=portal_count,unlocalized_objects=sum(o['position'] is None for o in objects))

def load_query(dbpath,object_id=None):
 db=sqlite3.connect(f'file:{Path(dbpath).resolve()}?mode=ro',uri=True)
 integrity=db.execute('PRAGMA integrity_check').fetchone()[0]
 dangling=db.execute('PRAGMA foreign_key_check').fetchall()
 if integrity!='ok' or dangling:raise ValueError('graph integrity failure')
 counts=dict(db.execute('SELECT kind,count(*) FROM nodes GROUP BY kind').fetchall())
 oid=object_id or (db.execute("SELECT id FROM nodes WHERE kind='object' ORDER BY id LIMIT 1").fetchone() or [None])[0]
 result=dict(integrity=integrity,node_counts=counts,edges=db.execute('SELECT count(*) FROM edges').fetchone()[0],object_id=oid)
 if oid:
  row=db.execute('SELECT kind,attributes FROM nodes WHERE id=?',(oid,)).fetchone()
  if row is None:raise ValueError('object not found '+oid)
  attributes=json.loads(row[1])
  defaults=db.execute("SELECT value FROM metadata WHERE key='object_defaults'").fetchone()
  if row[0]=='object' and defaults:attributes={**json.loads(defaults[0]),**attributes}
  if row[0]=='object' and 'observations' not in attributes:attributes['observations']=[json.loads(x[0]) for x in db.execute("SELECT attributes FROM edges WHERE target=? AND kind='observes' ORDER BY source",(oid,))]
  result['node']=dict(kind=row[0],attributes=attributes);result['neighbors']=[dict(source=x[0],target=x[1],kind=x[2]) for x in db.execute('SELECT source,target,kind FROM edges WHERE source=? OR target=?',(oid,oid))]
 db.close();return result

def inventories(paths,root):
 seen=set();rows=[]
 for p,purpose in paths:
  p=Path(p).resolve()
  if p in seen:continue
  if not p.is_file():raise FileNotFoundError(p)
  seen.add(p);rows.append(dict(path=str(p.relative_to(root.resolve())),purpose=purpose,bytes=p.stat().st_size,sha256=sha(p)))
 return rows

def main():
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest='command',required=True)
 q=sp.add_parser('query');q.add_argument('--db',required=True);q.add_argument('--object-id')
 q=sp.add_parser('check-cloud');q.add_argument('--path',required=True)
 q=sp.add_parser('build');q.add_argument('--config',type=Path,required=True);q.add_argument('--predictions',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.command=='query':print(json.dumps(load_query(a.db,a.object_id),ensure_ascii=False));return
 if a.command=='check-cloud':
  x=np.load(a.path,allow_pickle=False)
  if x.ndim!=2 or x.shape[1]!=3 or not len(x) or not np.isfinite(x).all():raise ValueError('invalid pointcloud')
  print(json.dumps(dict(points=len(x),dtype=str(x.dtype))));return
 c=json.loads(a.config.read_text());rows=[json.loads(x) for x in a.predictions.read_text().splitlines() if x.strip()];fmap={f['id']:f for f in c['frames']}
 if len(fmap)!=len(c['frames']):raise ValueError('duplicate input frame ID')
 if len({r['frame_id'] for r in rows})!=len(rows):raise ValueError('duplicate predictions')
 detids=set()
 for r in rows:
  if r['frame_id'] not in fmap or r['image_sha256']!=fmap[r['frame_id']]['image_sha256']:raise ValueError('prediction/input mismatch')
  for d in r['detections']:
   if d['id'] in detids:raise ValueError('duplicate detection ID')
   detids.add(d['id'])
   if not math.isfinite(d['score']) or not 0<=d['score']<=1:raise ValueError('invalid score')
   if d.get('position') is not None and (np.asarray(d['position']).shape!=(3,) or not np.isfinite(d['position']).all()):raise ValueError('invalid location')
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'runtime').mkdir();(out/'baseline').mkdir();(out/'audit').mkdir();(out/'audit/replays').mkdir()
 shutil.copyfile(a.config,out/'audit/config.json');shutil.copyfile(a.predictions,out/'audit/detections.jsonl')
 scope=dict(scene=c['scene'],bounds_xyz=c['bounds_xyz'],trajectory_ids=[t['id'] for t in c['trajectories']],frame_ids=sorted(fmap),config_sha256=sha(a.config),predictions_sha256=sha(a.predictions))
 dump(out/'runtime/scope.json',scope);dump(out/'baseline/scope.json',dict(scope,pointcloud=c['pointcloud']))
 summary=make_graph(c,rows,out/'runtime/map.sqlite',out/'audit/updates.jsonl',scope)
 replays=[]
 for t in c['trajectories']:
  # New database = reset graph memory. Never initialize a replay from global map.
  rr=[r for r in rows if r['frame_id'] in t['frame_ids']];dest=out/'audit/replays'/t['id'];dest.mkdir()
  replays.append(dict(id=t['id'],expected=len(t['frame_ids']),**make_graph(c,rr,dest/'map.sqlite',dest/'updates.jsonl',dict(scope,trajectory_ids=[t['id']]))))
 for t in c['trajectories']:
  dest=out/'audit/replays'/t['id'];updates={x['frame_id']:x for x in [json.loads(line) for line in (dest/'updates.jsonl').read_text().splitlines()]}
  full=[f for f in c.get('replay_inputs',[]) if t.get('source_start',-1)<=int(f['id'])<=t.get('source_end',-1)]
  with (dest/'all_steps.jsonl').open('w') as log:
   for f in full:
    log.write(json.dumps(dict(frame_id=f['id'],timestamp=f['timestamp'],input_sha256=f['image_sha256'],input_available=bool(f['image_sha256']),keyframe_update=updates.get(f['id']),action='update' if f['id'] in updates else 'fixed_keyframe_policy_skip'))+'\n')
 dump(out/'audit/replays.json',replays);np.save(out/'baseline/points.npy',pointmap(c),allow_pickle=False)
 queries=[]
 for path in [out/'runtime/map.sqlite',*[out/'audit/replays'/t['id']/'map.sqlite' for t in c['trajectories']]]:
  result=subprocess.run([sys.executable,__file__,'query','--db',str(path)],capture_output=True,text=True,check=True);queries.append(dict(db=str(path.relative_to(out)),result=json.loads(result.stdout)))
 check=subprocess.run([sys.executable,__file__,'check-cloud','--path',str(out/'baseline/points.npy')],capture_output=True,text=True,check=True)
 dump(out/'audit/reload_queries.json',queries);dump(out/'audit/baseline_load.json',json.loads(check.stdout))
 gm=inventories([(out/'runtime/map.sqlite','all node/edge attributes and structured object observations; standalone query'),(out/'runtime/scope.json','sealed scope')],out)
 pc=inventories([(out/'baseline/points.npy','scoped XYZ voxel map'),(out/'baseline/scope.json','scope, units, fields, configuration')],out)
 gbytes=sum(x['bytes'] for x in gm);pbytes=sum(x['bytes'] for x in pc);reduction=100*(pbytes-gbytes)/pbytes
 reasons=[]
 if not summary['objects']:reasons.append('no structured object records')
 if any(r['frames']!=r['expected'] or r['frames']==0 for r in replays):reasons.append('one or more trajectory keyframe replays incomplete')
 if c.get('development_only'):reasons.append('development inputs, not fixed-stride acceptance replay')
 if set(fmap)!={r['frame_id'] for r in rows}:reasons.append('incomplete inference for sealed frames')
 if not c.get('replay_inputs') or any(not f.get('image_sha256') for f in c.get('replay_inputs',[])):reasons.append('full chronological replay inputs incomplete')
 if len(c['trajectories'])!=18:reasons.append('not 18 trajectories')
 if not c['pointcloud'].get('scope_verified') or not c['pointcloud'].get('scope_evidence'):reasons.append('same interval / effective coverage of point cloud baseline not verified')
 if c['pointcloud'].get('scope_evidence'):
  ep=Path(c['pointcloud']['scope_evidence'])
  if not ep.is_file():reasons.append('baseline scope evidence file missing')
  else:
   ev=json.loads(ep.read_text())
   if ev.get('scene')!=c['scene'] or ev.get('source_pointcloud_sha256')!=c['pointcloud']['sha256'] or ev.get('bounds_xyz')!=c['bounds_xyz'] or ev.get('trajectory_ids')!=[t['id'] for t in c['trajectories']] or not ev.get('reviewer') or not all(ev.get(k) is True for k in ['same_time_interval','same_effective_coverage','registration_verified']):reasons.append('baseline scope evidence does not verify this scene and inputs')
 if not c.get('regions'):reasons.append('column-derived regions remain candidates; frozen region boundaries required for acceptance')
 if 'not_official' in c['calibration_status'] and not c.get('calibration_review'):reasons.append('estimated calibration requires documented geometric review')
 review=c.get('human_review')
 if not review:reasons.append('visibility, instruction consistency, region and exit/elevator/junction spot checks require human review')
 else:
  rv=json.loads(Path(review).read_text())
  if rv.get('predictions_sha256')!=sha(a.predictions) or rv.get('config_scope_sha256')!=sha(a.config) or not rv.get('reviewer') or not all(rv.get(k) is True for k in ['visibility','instruction_consistency','regions','portals','exit_elevator_junction']):reasons.append('human review is incomplete or not bound to this run')
 # Predictions provenance is necessary to distinguish real SAM inference from fixtures.
 prov_path=a.predictions.parent/'provenance.json';prov=json.loads(prov_path.read_text()) if prov_path.exists() else {}
 if not prov.get('completed') or prov.get('config_sha256')!=sha(a.config) or not prov.get('checkpoint_sha256'):reasons.append('missing completed SAM3 provenance bound to config')
 result=dict(summary=summary,baseline=pc,cognitive=gm,baseline_bytes=pbytes,cognitive_bytes=gbytes,reduction_percent=reduction,baseline_load=json.loads(check.stdout),counting_note='Storage comparison counts the SQLite graph and scope file against the NPY point cloud and scope file. The graph stores frame IDs, timestamps, hashes, boxes, text, positions and adjacency.')
 dump(out/'report.json',result)
 from viewer import render
 render(c,rows,out,a.predictions.parent)
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
