"""Check shared inputs AND actual geometry. Coverage is not scene completeness."""
import json,hashlib,sqlite3
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(root=ROOT):
 data=root/'data/full_map';res=root/'results';c=read(data/'audit/config.json');gs=read(data/'graph_scope.json');ps=read(data/'pointcloud_scope.json');lo,hi=np.array(c['bounds_xyz']);pts=np.load(data/'points.npy');checks={}
 for k in ['scene','bounds_xyz','trajectory_ids','frame_ids','config_sha256','predictions_sha256']:checks['scope_equal_'+k]=gs[k]==ps[k]
 checks['config_hash']=sha(data/'audit/config.json')==gs['config_sha256'];checks['detections_hash']=sha(data/'audit/detections.jsonl')==gs['predictions_sha256']
 frames={f['id']:f for f in c['frames']};checks['frame_set']=set(frames)==set(gs['frame_ids']);checks['cloud_finite']=bool(np.isfinite(pts).all());checks['cloud_in_bounds']=bool(((pts>=lo)&(pts<=hi)).all())
 db=sqlite3.connect(data/'graph.sqlite');nodes=[(i,k,json.loads(a)) for i,k,a in db.execute('select * from nodes')];db.close();gn={a['frame_id']:a for _,k,a in nodes if k=='frame'}
 checks['graph_frame_set']=set(gn)==set(frames)
 checks['graph_poses_and_images']=all(gn[i]['T_world_camera']==f['T_world_camera'] and gn[i]['source_sha256']==f['image_sha256'] and sha(root/'data/dataset/rgb'/f'{i}.png')==f['image_sha256'] for i,f in frames.items())
 positions=np.array([a['position'] for _,k,a in nodes if k=='object' and a.get('position') is not None]);unlocated=sum(k=='object' and a.get('position') is None for _,k,a in nodes)
 checks['localized_objects_in_bounds']=bool(((positions>=lo)&(positions<=hi)).all())
 evidence=data/'scope_evidence';audit=read(evidence/'geometry_audit.json');cache=np.load(evidence/'frame_points.npz');checks['geometry_frame_set']=set(cache.files)=={'f'+i for i in frames};checks['scan_frame_set']={s['frame_id'] for s in audit['scans']}==set(frames);checks['sensor_pair_within_50ms']=all(abs(s['delta_s'])<=.05 for s in audit['scans'])
 outside=0;parts=[]
 for fid in sorted(frames):
  p=cache['f'+fid];parts.append(p);T=np.array(frames[fid]['T_world_camera']);q=np.einsum('ni,ij->nj',p-T[:3,3],T[:3,:3]);uv=np.einsum('ni,ji->nj',q,np.array(c['K']));uv=np.rint(uv[:,:2]/uv[:,2:]);w,h=c['image_size'];ok=(q[:,2]>.3)&(q[:,2]<40)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);outside+=int((~ok).sum())
 checks['source_geometry_in_rgb_frustums']=outside==0
 p=np.concatenate(parts);voxel=c['pointcloud']['voxel_m'];_,ix=np.unique(np.floor(p/voxel).astype('i4'),axis=0,return_index=True);p=p[np.sort(ix)].astype('float64');p=p[((p>=lo)&(p<=hi)).all(1)];_,ix=np.unique(np.floor(p/voxel).astype('int64'),axis=0,return_index=True);p=p[np.sort(ix)].astype('float32')
 checks['cloud_exactly_rebuilt_from_frame_geometry']=np.array_equal(p,pts)
 report=dict(scope_consistent=all(checks.values()),checks=checks,trajectory_count=len(c['trajectories']),keyframe_count=len(frames),raw_rgb_count=len(c['replay_inputs']),cloud_points=len(pts),cloud_sha256=sha(data/'points.npy'),roi_bounds_xyz=c['bounds_xyz'],cloud_actual_bounds_xyz=[pts.min(0).tolist(),pts.max(0).tolist()],localized_object_count=len(positions),unlocalized_object_count=unlocated,objects_actual_bounds_xyz=[positions.min(0).tolist(),positions.max(0).tolist()],frustum_outside_points=outside,max_sensor_delta_ms=audit['max_sensor_delta_s']*1000,evidence_sha256={p.name:sha(p) for p in evidence.iterdir() if p.is_file()})
 (res/'scope_consistency.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import Rectangle
 fig,ax=plt.subplots(figsize=(9,10));sample=pts[::10];ax.scatter(sample[:,0],sample[:,2],s=.15,c='#a6b2bd',label='Point cloud (display sample)');ax.scatter(positions[:,0],positions[:,2],s=2,c='#e48029',alpha=.5,label='Localized graph objects')
 for j,t in enumerate(c['trajectories']):
  path=np.array([np.array(frames[i]['T_world_camera'])[:3,3] for i in t['frame_ids']]);ax.plot(path[:,0],path[:,2],lw=1,label='18 RGB trajectories' if j==0 else None,color='#2575bb');ax.text(path[0,0],path[0,2],t['id'],fontsize=7,color='#173f70')
 ax.add_patch(Rectangle((lo[0],lo[2]),hi[0]-lo[0],hi[2]-lo[2],fill=False,ls='--',ec='black',label='Shared comparison ROI'))
 ax.set(xlabel='Scene X (m)',ylabel='Scene Z (m)',title='18-trajectory coverage: graph vs point cloud');ax.set_aspect('equal');ax.legend(loc='upper left',fontsize=8);fig.tight_layout();fig.savefig(res/'scope_consistency.png',dpi=160);plt.close(fig)
 print(json.dumps({k:v for k,v in report.items() if k in ['scope_consistent','checks','cloud_points','localized_object_count','unlocalized_object_count','frustum_outside_points','cloud_actual_bounds_xyz']},ensure_ascii=False))
 if not report['scope_consistent']:raise ValueError('Scope consistency checks failed; inspect results/scope_consistency.json')
 return report
if __name__=='__main__':check()
