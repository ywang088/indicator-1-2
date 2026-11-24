"""Build both map geometry and per-keyframe projection from the SAME synchronized scans."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from scipy.spatial import cKDTree
from perception import dump,sha,projection
from geometry_io import read_pcd,geometry_scope_sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--scans',type=Path,required=True);p.add_argument('--trajectory',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 c=json.loads(a.config.read_text());pairs={v['frame_id']:v for v in json.loads(a.pairs.read_text())};tr=np.loadtxt(a.trajectory);interp=Slerp(tr[:,0],Rotation.from_quat(tr[:,4:8]));world_source=np.array(c['pointcloud']['T_world_source']);perm=np.array([[0,1,0],[0,0,1],[1,0,0]])
 a.output.mkdir(parents=True,exist_ok=False);parts=[];arrays={};records=[];lo,hi=np.array(c['bounds_xyz'])
 for f in c['frames']:
  v=pairs[f['id']];t=v['scan_timestamp'];path=a.scans/f'{v["scan_id"]:06}.pcd'
  if not tr[0,0]<=t<=tr[-1,0] or abs(t-f['timestamp'])>.05:raise ValueError('unmatched timestamp')
  q=read_pcd(path)[0];distance=np.linalg.norm(q,axis=1);q=q[(distance>=1.5)&(distance<=40)]
  R=interp(t).as_matrix();P=np.array([np.interp(t,tr[:,0],tr[:,j]) for j in [1,2,3]])
  xyz=np.einsum('ni,ji->nj',q,world_source[:3,:3]@R@perm)+world_source[:3,:3]@P+world_source[:3,3]
  xyz=xyz[((xyz>=lo)&(xyz<=hi)).all(1)]
  ids,uv,z=projection(xyz,f,c['K'],c['image_size']);xyz=xyz[ids]
  _,ix=np.unique(np.floor(xyz/c['pointcloud']['voxel_m']).astype('i4'),axis=0,return_index=True);xyz=xyz[np.sort(ix)].astype('f4')
  arrays['f'+f['id']]=xyz;parts.append(xyz);records.append(dict(**v,source_sha256=sha(path),visible_points=len(xyz)))
  if len(records)%50==0:print('matched',len(records),flush=True)
 allpts=np.concatenate(parts);_,idx=np.unique(np.floor(allpts/c['pointcloud']['voxel_m']).astype('i4'),axis=0,return_index=True);baseline=allpts[np.sort(idx)]
 np.save(a.output/'points.npy',baseline,allow_pickle=False);np.savez_compressed(a.output/'frame_points.npz',**arrays)
 metadata=dict(geometry_scope_sha256=geometry_scope_sha256(c),scene=c['scene'],config_parent_sha256=sha(a.config),source_archive='IPS-3D 2022-3-24-11-57-11.zip / velodyne.zip; user 4090 source',trajectory_sha256=sha(a.trajectory),bounds_xyz=c['bounds_xyz'],selected_frames=len(records),max_sensor_delta_s=max(abs(v['delta_s']) for v in records),scans=records,points=len(baseline),voxel_m=c['pointcloud']['voxel_m'],source_fields='XYZ metres',format='NPY float32 uncompressed',registration='original LeGO-LOAM poses; linear translation and quaternion SLERP at actual LiDAR timestamps; provided Velodyne-to-LOAM axis permutation; original scene rigid transform',coverage='same selected RGB keyframe frustums, projected nearest sensor point, same spatial bounds, 0.3<camera Z<40m, LiDAR 1.5..40m',limitations=['Camera calibration is estimated, not surveyed ground truth.','Within-scan deskew is not added.','No surface completion, unseen-region points, point duplication or extra archive compression.'])
 dump(a.output/'geometry_audit.json',metadata);print('baseline',len(baseline),'points',baseline.nbytes,'bytes')
if __name__=='__main__':main()
