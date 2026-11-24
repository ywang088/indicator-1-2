"""Two reproducible delivery entrypoints; SQLite runtime plus dictionary exports."""
import argparse,json,sqlite3,shutil,subprocess,sys,tempfile,csv,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def clean(v):
 if isinstance(v,dict):return {clean(k):clean(x) for k,x in v.items() if k!='method'}
 if isinstance(v,list):return [clean(x) for x in v]
 if isinstance(v,str) and v.startswith('region:unknown:'):return v.removeprefix('region:unknown:').replace('paint_','paint')
 return v
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def graph(dbfile):
 db=sqlite3.connect(dbfile);nodes={i:{'kind':k,'attributes':json.loads(a)} for i,k,a in db.execute('select * from nodes')};edges=[dict(source=s,target=t,kind=k,attributes=json.loads(a)) for s,t,k,a in db.execute('select * from edges')];metadata={k:json.loads(v) for k,v in db.execute('select key,value from metadata')};db.close();return clean(dict(nodes=nodes,edges=edges,metadata=metadata))
def normalize(src,dst):
 shutil.copy2(src,dst);db=sqlite3.connect(dst);nodes=db.execute('select * from nodes').fetchall();edges=db.execute('select * from edges').fetchall();db.execute('delete from edges');db.execute('delete from nodes')
 db.executemany('insert into nodes values(?,?,?)',[(clean(i),k,json.dumps(clean(json.loads(a)),ensure_ascii=False,separators=(',',':'))) for i,k,a in nodes]);db.executemany('insert into edges values(?,?,?,?)',[(clean(s),clean(t),k,json.dumps(clean(json.loads(a)),ensure_ascii=False,separators=(',',':'))) for s,t,k,a in edges]);db.commit();db.execute('VACUUM');assert not db.execute('pragma foreign_key_check').fetchall();db.close()
def export_run(run):
 c=json.loads((run/'audit/config.json').read_text());results=ROOT/'results';data=results/'online_full_map';data.mkdir(parents=True,exist_ok=True);rows={r['frame_id']:r for r in map(json.loads,(run/'audit/detections.jsonl').read_text().splitlines())};fs={f['id']:f for f in c['frames']}
 normalize(run/'runtime/map.sqlite',data/'graph.sqlite');save(data/'graph.json',graph(data/'graph.sqlite'));shutil.copy2(run/'runtime/scope.json',data/'graph_scope.json');shutil.copy2(run/'baseline/scope.json',data/'pointcloud_scope.json')
 shutil.copy2(run/'audit/config.json',results/'run_config.json')
 shutil.copy2(run/'audit/detections.jsonl',results/'detections.jsonl')
 from color_topology import make_topology
 summaries=[]
 for t in c['trajectories']:
  src=run/'audit/replays'/t['id'];out=results/'trajectories'/t['id'];out.mkdir(parents=True,exist_ok=True);normalize(src/'map.sqlite',out/'graph.sqlite');save(out/'graph.json',graph(out/'graph.sqlite'))
  db=sqlite3.connect(src/'map.sqlite');counts=dict(db.execute('select kind,count(*) from nodes group by kind'));unloc=sum(json.loads(a).get('position') is None for a, in db.execute("select attributes from nodes where kind='object'"));summary=dict(frames=counts.get('frame',0),objects=counts.get('object',0),regions=counts.get('region',0),portals=counts.get('portal',0),unlocalized_objects=unloc);save(out/'summary.json',{'summary':summary});summaries.append(dict(trajectory=t['id'],summary=summary))
  for name in ['updates.jsonl','all_steps.jsonl']:
   (out/name).write_text(''.join(json.dumps(clean(json.loads(s)),ensure_ascii=False)+'\n' for s in (src/name).read_text().splitlines()))
  state=make_topology(c);observed={json.loads(a)['detection_id']:oid for oid,a in db.execute("select target,attributes from edges where kind='observes'")}
  from PIL import Image,ImageDraw
  (out/'frames').mkdir(exist_ok=True)
  with (out/'query_log.jsonl').open('w') as log:
   for fid in t['frame_ids']:
    ds=rows[fid]['detections'];state.update(fs[fid],[(observed[d['id']],d) for d in ds if d['id'] in observed]);edges=[dict(source=s,target=tar,kind=k,attributes=json.loads(a)) for s,tar,k,a in db.execute("select * from edges where source=? and kind='observes'",('frame:'+fid,))]
    current={observed[d['id']] for d in ds if d['id'] in observed}
    for rid,r in state.regions.items():
     edges.extend(dict(source=rid,target=oid,kind='candidate_contains',attributes={}) for oid in r['anchors'] if oid in current)
    for portal in state.portals.values():
     if fid in portal['evidence_frames']:edges.extend(dict(source=rid,target=portal['id'],kind='connects',attributes={}) for rid in portal['regions'])
    for e in edges:log.write(json.dumps(clean(dict(frame_id=fid,**e)),ensure_ascii=False)+'\n')
    from render_detection_frames import render_frame
    render_frame(ROOT/'data/dataset/rgb'/f'{fid}.png',ds,observed,out/'frames'/f'{fid}.jpg',fid,t['id'])
  db.close()
 save(data/'summary.json',json.loads((run/'report.json').read_text())['summary'])
 from export_scene_review import main as export_examples
 export_examples()
 save(results/'report.json',{'trajectories':summaries});print(json.dumps(summaries,ensure_ascii=False))
def compare():
 data=ROOT/'data/full_map';r=ROOT/'results';r.mkdir(exist_ok=True);files=[]
 from final_delivery_check import verify_frozen_inputs
 verify_frozen_inputs()
 import numpy as np
 points=np.load(data/'points.npy',mmap_mode='r',allow_pickle=False);assert points.ndim==2 and points.shape[1]==3 and len(points)>0
 db=sqlite3.connect('file:'+str(data/'graph.sqlite')+'?mode=ro',uri=True);assert db.execute('pragma integrity_check').fetchone()[0]=='ok';assert not db.execute('pragma foreign_key_check').fetchall();db.close()
 for side,names in [('baseline',['points.npy','pointcloud_scope.json']),('cognitive',['graph.sqlite','graph_scope.json'])]:
  for name in names:
   p=data/name;files.append(dict(side=side,file='data/full_map/'+name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 baseline=sum(x['bytes'] for x in files if x['side']=='baseline');cognitive=sum(x['bytes'] for x in files if x['side']=='cognitive');v=dict(baseline_bytes=baseline,cognitive_bytes=cognitive,reduction_percent=100*(baseline-cognitive)/baseline)
 save(r/'storage_comparison.json',v)
 with (r/'storage_comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=['side','file','bytes','sha256']);w.writeheader();w.writerows(files)
 save(r/'storage_report.json',dict(**v,input_mode='pre_saved_frozen_full_map',input_graph='data/full_map/graph.sqlite',runtime_format='SQLite',graph_json='Dictionary export of the same graph, not an additional runtime dependency',dictionary_export_bytes=(data/'graph.json').stat().st_size))
 from check_scope_consistency import check
 check(ROOT)
 for note in (ROOT/'code/result_notes').glob('*.txt'):shutil.copy2(note,r/note.name)
 from final_delivery_check import validate_frozen
 validate_frozen()
 print(json.dumps(v,ensure_ascii=False))
def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['build','compare','export']);p.add_argument('--source',type=Path);a=p.parse_args()
 if a.command=='compare':compare();return
 if a.command=='export':export_run(a.source);return
 dataset=ROOT/'data/dataset';c=json.loads((dataset/'config.json').read_text())
 for f in c['replay_inputs']:
  image=dataset/'rgb'/(f['id']+'.png')
  if not image.exists() or hashlib.sha256(image.read_bytes()).hexdigest()!=f['image_sha256']:raise ValueError('Missing or changed original RGB: '+f['id'])
  f['image']=str(image)
 save(ROOT/'results/data_integrity.json',dict(raw_rgb_count=len(c['replay_inputs']),keyframe_count=len(c['frames']),all_source_image_hashes_verified=True))
 with tempfile.TemporaryDirectory(prefix='parking18-') as tmp:
  stage=Path(tmp);inputs=stage/'inputs';inputs.mkdir()
  for f in c['frames']:f['image']=str(dataset/'rgb'/(f['id']+'.png'))
  c['pointcloud']['path']=str(ROOT/'data/full_map/points.npy');c['pointcloud']['sha256']=hashlib.sha256(Path(c['pointcloud']['path']).read_bytes()).hexdigest();c['pointcloud'].pop('scope_evidence',None)
  save(inputs/'config.json',c)
  for name in ['detections.jsonl','images','masks']:(inputs/name).symlink_to(dataset/name,target_is_directory=(dataset/name).is_dir())
  prov=json.loads((dataset/'provenance.json').read_text());prov['parent_config_sha256']=prov.get('config_sha256');prov['config_sha256']=hashlib.sha256((inputs/'config.json').read_bytes()).hexdigest();save(inputs/'provenance.json',prov)
  subprocess.run([sys.executable,str(ROOT/'code/map_pipeline.py'),'build','--config',str(inputs/'config.json'),'--predictions',str(inputs/'detections.jsonl'),'--output',str(stage/'run')],check=True,stdout=subprocess.DEVNULL);export_run(stage/'run')
if __name__=='__main__':main()
