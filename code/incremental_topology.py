"""Causal zone evidence and shared-corridor connections from pillar observations.

Only current/past observations are consumed. Visibility of a zone is distinct from
camera membership in that zone. Letter identities require text; ColorTopology can supply provisional paint identities.
"""
from collections import defaultdict
import math
import numpy as np

class IncrementalTopology:
 def __init__(self,config):
  self.c=config;self.regions={};self.portals={};self.evidence=defaultdict(list);self.pending=[];self.last=None;self.zone_by_object={};self.recent=[];self.object_zones=defaultdict(set)
  self.members=defaultdict(set)
  for t in config.get('trajectories',[]):
   for fid in t['frame_ids']:self.members[fid].add(t['id'])
 def update(self,frame,observations):
  fid=frame['id'];now=frame['timestamp'];T=np.asarray(frame['T_world_camera']);camera=T[:3,3];forward=T[:3,2][[0,2]];width,height=self.c['image_size'];before=set(self.regions);before_p=set(self.portals)
  same=self.last is not None and bool(self.members[fid]&self.members[self.last['id']]) and 0<now-self.last['timestamp']<=3
  if not same:self.recent=[]
  seen=[]
  for obj,d in observations:
   if d['label']!='column':continue
   labels={}
   for text in d.get('texts',[]):
    p=text.get('parsed') or {};zone=p.get('zone');score=text.get('score');reviewed=text.get('review_status')=='assistant_visual_transcription_not_human_acceptance'
    if not zone or not (len(zone)==1 and zone in 'ABCDEFGH' or self.c.get('color_regions') and text.get('source')=='pillar_paint_region' and zone.startswith('paint_')):continue
    if reviewed and text.get('image_sha256')!=frame['image_sha256']:continue
    if not reviewed and (not isinstance(score,(int,float)) or score<.3):continue
    labels[zone]=dict(zone=zone,reviewed=reviewed,score=score,source=text.get('source','OCR'))
   if not labels and obj in self.zone_by_object:
    rid=self.zone_by_object[obj];v=next((e for e in reversed(self.evidence[rid]) if e['object']==obj),None)
    if v is not None:labels[v['zone']]=dict(zone=v['zone'],reviewed=False,score=1.0,source='previously_read_same_column_track',label_origin_frame_id=v.get('label_origin_frame_id',v['frame_id']),label_origin_detection_id=v.get('label_origin_detection_id',v['detection_id']))
   if len(labels)!=1:continue
   v=next(iter(labels.values()));v.setdefault('label_origin_frame_id',fid);v.setdefault('label_origin_detection_id',d['id']);self.object_zones[obj].add(v['zone'])
   if len(self.object_zones[obj])!=1:
    self.zone_by_object.pop(obj,None);continue
   rid='region:unknown:'+v['zone'];b=d['bbox'];cx=(b[0]+b[2])/2/width
   # Reject overhead text and tiny far-away fragments. The OCR importer also requires mask overlap.
   if b[3]<height*.52 or b[3]-b[1]<height*.13:continue
   ev=dict(region=rid,object=obj,detection_id=d['id'],frame_id=fid,image_sha256=frame['image_sha256'],timestamp=now,bbox=b,side='left' if cx<.46 else 'right' if cx>.54 else 'center',position=d.get('position'),camera=camera.tolist(),forward=forward.tolist(),**v)
   self.evidence[rid].append(ev);seen.append(ev)
   unique={e['frame_id'] for e in self.evidence[rid]};strong=v['reviewed'] or (v['score'] is not None and v['score']>=.85)
   # Several pillars with the same label can establish a zone even when instance tracking breaks.
   accepted=strong or len(unique)>=2
   if accepted:
    r=self.regions.setdefault(rid,dict(label=v['zone'],floor='unknown',status='causal_column_text_evidence',anchors=[],positions=[],boundary_status='not_geofenced',first_seen_frame=self.evidence[rid][0]['frame_id'],established_frame=fid,evidence_frames=[],evidence_sources=[]))
    for e in self.evidence[rid]:
     if e['object'] not in r['anchors']:r['anchors'].append(e['object'])
     if e['position'] is not None and e['position'] not in r['positions']:r['positions'].append(e['position'])
     if e['frame_id'] not in r['evidence_frames']:r['evidence_frames'].append(e['frame_id'])
     if e['source'] not in r['evidence_sources']:r['evidence_sources'].append(e['source'])
     self.zone_by_object[e['object']]=rid
  # Only recent text in the same traversal and approximately the same camera heading.
  retained=[]
  for e in self.recent:
   if not (0<=now-e['timestamp']<=6 and np.linalg.norm(camera-np.array(e['camera']))<=12 and np.dot(forward,e['forward'])>.85):continue
   # Old image left/right is not current left/right after moving past a pillar.
   # Reproject historical 3-D evidence; never pair a behind-camera sign with a new sign.
   if e['position'] is None:continue
   xyz=np.linalg.inv(T)@np.r_[e['position'],1.]
   if xyz[2]<=.5:continue
   K=np.asarray(self.c.get('K',[[width,0,width/2],[0,width,height/2],[0,0,1]]));uv=K@xyz[:3];u=float(uv[0]/uv[2]/width)
   side='left' if 0<=u<.46 else 'right' if .54<u<=1 else 'center'
   if side!=e['side']:continue
   retained.append(e)
  self.recent=retained+seen
  candidates=[]
  for a in self.recent:
   for b in self.recent:
    if a['side']!='left' or b['side']!='right' or a['region']==b['region'] or a['region'] not in self.regions or b['region'] not in self.regions:continue
    if fid not in (a['frame_id'],b['frame_id']):continue
    # Opposite labels must both be currently or recently observed; no stored global review is needed.
    candidates.append((a,b))
  for a,b in candidates:
   pair=sorted([a['region'],b['region']]);moving=False
   if same:
    prev=np.asarray(self.last['T_world_camera'])[:3,3];delta=(camera-prev)[[0,2]];longitudinal=float(np.dot(delta,forward));lateral=float(abs(forward[0]*delta[1]-forward[1]*delta[0]))
    moving=longitudinal>=.5 and lateral<=max(.5,longitudinal*.6) and np.dot(forward,np.asarray(self.last['T_world_camera'])[:3,2][[0,2]])>.85
   if not moving:
    self.pending.append(dict(frame_id=fid,regions=pair,reason='opposite_labels_without_supported_forward_travel'));continue
   old=[p for p in self.portals.values() if p['regions']==pair and (now-p['last_timestamp']<=6 and np.linalg.norm(camera-np.array(p['last_position']))<=12)]
   # A pair of zone names is NOT a globally unique doorway: reconnect only along a continuous traversal.
   if old:p=old[-1]
   else:
    pid='portal:online:'+fid+':'+pair[0].split(':')[-1]+'_'+pair[1].split(':')[-1]
    p=dict(id=pid,label=pair[0].split(':')[-1]+'↔'+pair[1].split(':')[-1]+' 共享通道',regions=pair,connection_type='shared_corridor',method='causal_opposite_pillar_labels_and_forward_travel',status='trajectory_supported_connection_candidate',created_frame=fid,evidence_frames=[],side_evidence=[],anchors=[],path_xyz=[],position=camera.tolist(),human_acceptance_completed=False,geometry_status='observed_camera_path_not_surveyed_boundary');self.portals[pid]=p
   for e in [a,b]:
    key=(e['frame_id'],e['detection_id'])
    if not any((v['frame_id'],v['detection_id'])==key for v in p['side_evidence']):p['side_evidence'].append(e)
    if e['object'] not in p['anchors']:p['anchors'].append(e['object'])
    for efid in [e['frame_id'],e['label_origin_frame_id']]:
     if efid not in p['evidence_frames']:p['evidence_frames'].append(efid)
   for efid in [fid,self.last['id']]:
    if efid not in p['evidence_frames']:p['evidence_frames'].append(efid)
   if camera.tolist() not in p['path_xyz']:p['path_xyz'].append(camera.tolist())
   p.update(last_timestamp=now,last_position=camera.tolist())
  self.last=frame
  return dict(regions_after=len(self.regions),region_ids=sorted(self.regions),new_regions=sorted(set(self.regions)-before),portals_after=len(self.portals),portal_ids=sorted(self.portals),new_portals=sorted(set(self.portals)-before_p),visible_zone_labels=sorted({e['zone'] for e in seen}),portal_creation_evidence=[dict(id=pid,created_frame=fid,regions=self.portals[pid]['regions'],evidence_frames=list(self.portals[pid]['evidence_frames']),side_detection_ids=[e['detection_id'] for e in self.portals[pid]['side_evidence']]) for pid in sorted(set(self.portals)-before_p)])
