"""Region transitions, not detector labels, establish topological connections."""
import numpy as np

def trajectory_transitions(config, frame_regions, frame_methods, frames):
 """Return accepted traversals and unresolved proposals, without joining different replays."""
 accepted=[];pending=[];audit=[]
 for trajectory in config.get('trajectories',[]):
  seq=sorted([frames[i] for i in trajectory['frame_ids'] if i in frame_regions],key=lambda f:f['timestamp'])
  runs=[]
  for j,f in enumerate(seq):
   region=frame_regions[f['id']]
   if not region:continue
   if runs and runs[-1]['region']==region:runs[-1]['indices'].append(j)
   else:runs.append(dict(region=region,indices=[j]))
  trajectory_audit=dict(trajectory_id=trajectory['id'],keyframes=len(seq),known_region_frames=sum(bool(frame_regions[f['id']]) for f in seq),stable_region_runs=[dict(region=v['region'],frames=len(v['indices'])) for v in runs],switches=0)
  for a,b in zip(runs,runs[1:]):
   trajectory_audit['switches']+=1
   ia,ib=a['indices'][-1],b['indices'][0];segment=seq[ia:ib+1];pts=np.array([np.array(f['T_world_camera'])[:3,3] for f in segment]);steps=np.linalg.norm(np.diff(pts[:,[0,2]],axis=0),axis=1);times=np.diff([f['timestamp'] for f in segment]);length=float(steps.sum());dt=float(times.sum());support=[seq[i]['id'] for i in a['indices'][-2:]+b['indices'][:2]];reasons=[]
   if len(a['indices'])<2 or len(b['indices'])<2:reasons.append('need_two_stable_observations_on_each_side')
   if len(a['indices'])>=2 and a['indices'][-1]-a['indices'][-2]>1:reasons.append('before_side_not_consecutive')
   if len(b['indices'])>=2 and b['indices'][1]-b['indices'][0]>1:reasons.append('after_side_not_consecutive')
   if dt<=0 or dt>6 or any(times<=0) or any(times>3):reasons.append('temporal_gap')
   if length<.5:reasons.append('insufficient_motion_possible_camera_rotation')
   if length>8:reasons.append('transition_interval_too_wide')
   if any(steps/np.maximum(times,.001)>8):reasons.append('pose_discontinuity')
   # Nearest-column labels are useful proposals but cannot prove the camera changed region.
   if any(frame_methods.get(fid) not in ('frozen_boundary','reviewed_camera_region') for fid in support):reasons.append('camera_region_assignment_not_confirmed')
   p=dict(id='portal:transition:'+trajectory['id']+':'+segment[0]['id']+':'+segment[-1]['id'],label='open passage',connection_type='open_passage',regions=[a['region'],b['region']],trajectory_id=trajectory['id'],evidence_frames=sorted(set(support+[f['id'] for f in segment])),transition_interval_frames=[segment[0]['id'],segment[-1]['id']],position=((pts[0]+pts[-1])/2).tolist(),position_uncertainty_m=length/2,path_length_m=length,elapsed_s=dt,requires_physical_door=False)
   if reasons:p.update(status='unconfirmed_region_transition',reasons=reasons);pending.append(p)
   else:p.update(status='trajectory_verified_connection',method='continuous_traversal_between_confirmed_region_states');accepted.append(p)
  audit.append(trajectory_audit)
 return accepted,pending,audit

def validate_reviewed_connection(p,config,all_frames):
 if not p.get('id','').startswith('portal:') or len(p.get('regions',[]))!=2 or len(set(p['regions']))!=2:raise ValueError('portal requires two distinct region IDs')
 if not p.get('reviewer') or not p.get('connection_confirmed') or p.get('connection_type') not in ('doorway','open_passage','shared_corridor'):raise ValueError('portal requires explicit confirmed connection review and type')
 evidence=p.get('side_evidence',[])
 if set(e.get('region_id') for e in evidence)!=set(p['regions']):raise ValueError('portal requires evidence for both incident regions')
 for e in evidence:
  fid=e.get('frame_id')
  if fid not in all_frames or e.get('image_sha256')!=all_frames[fid].get('image_sha256') or not e.get('basis'):raise ValueError('portal side evidence must identify actual frame hash and basis')
 if not p.get('evidence_frames') or not {e['frame_id'] for e in evidence}<=set(p['evidence_frames']) or not set(p['evidence_frames'])<=set(all_frames):raise ValueError('portal observation references invalid')
 if p.get('method') not in ('two_sided_visual_review','trajectory_crossing_review','shared_corridor_visual_review'):raise ValueError('portal needs connection method, not just a door score')
 if p['connection_type']=='shared_corridor':
  if p['method']!='shared_corridor_visual_review' or not p.get('opposite_sides_confirmed') or not p.get('open_access_confirmed'):raise ValueError('shared corridor requires opposite region sides and open access review')
  layouts=p.get('layout_evidence_frames',[])
  if not layouts or not set(layouts)<=set(p['evidence_frames']):raise ValueError('shared corridor needs layout evidence')
  segments=p.get('observed_segments',[])
  if not segments:raise ValueError('shared corridor requires observed extent')
  for segment in segments:
   t=next((t for t in config.get('trajectories',[]) if t['id']==segment.get('trajectory_id')),None)
   ids=segment.get('frame_ids',[])
   if t is None or len(ids)<2 or len(set(ids))!=len(ids) or not set(ids)<=set(p['evidence_frames']) or not set(ids)<=set(t['frame_ids']):raise ValueError('corridor segment must stay within a trajectory')
   indices=[t['frame_ids'].index(fid) for fid in ids]
   if any(b!=a+1 for a,b in zip(indices,indices[1:])):raise ValueError('corridor segment must be contiguous and ordered')
   expected=[np.asarray(all_frames[fid]['T_world_camera'])[:3,3].tolist() for fid in ids]
   if segment.get('camera_path_xyz')!=expected:raise ValueError('corridor geometry must retain observed camera positions')
  if p.get('human_acceptance_completed') is not False:raise ValueError('visual review must not imply human acceptance')
 if p['method']=='trajectory_crossing_review':
  t=next((t for t in config.get('trajectories',[]) if t['id']==p.get('trajectory_id')),None)
  if t is None or not set(p['evidence_frames'])<=set(t['frame_ids']):raise ValueError('crossing evidence must stay within one trajectory')
