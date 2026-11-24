"""Freeze T01-T18 before inference; index only real, available images."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parent.parent);p.add_argument('--output',type=Path,required=True);p.add_argument('--rgb-root',type=Path);p.add_argument('--stride',type=int,default=20);a=p.parse_args()
 if a.stride<1:raise ValueError('stride must be positive')
 w=a.workspace.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 source=w/'work/replay20/manifest.json';m=json.loads(source.read_text());es=m['episodes'][:18]
 assert len(es)==18 and [e['id'] for e in es]==[f'T{i:02}' for i in range(1,19)]
 cam=np.load(w/'work/replay20/poses.npz');lookup={int(f):i for i,f in enumerate(cam['ids'])}
 roots=[a.rgb_root] if a.rgb_root else [w/'work/red_zone_v31/raw',w/'work/data/images',w/'work/full_markings/images']
 images={}
 for r in roots:
  if r and r.exists():
   for f in sorted(r.iterdir()):
    if f.suffix.lower() in ('.png','.jpg','.jpeg') and f.stem.isdigit():images.setdefault(int(f.stem),f.resolve())
 # Fixed stride, always include endpoints. Missing frames stay explicitly missing.
 wanted=sorted({f for e in es for f in [*range(e['start'],e['end']+1,a.stride),e['end']]})
 frames=[]
 for f in wanted:
  i=lookup[f];T=np.eye(4);T[:3,:3]=cam['R'][i];T[:3,3]=cam['C'][i]
  frames.append(dict(id=f'{f:06}',timestamp=float(cam['timestamps'][i]),image=str(images[f]) if f in images else None,image_sha256=sha(images[f]) if f in images else None,T_world_camera=T.tolist()))
 selected={int(f['id']) for f in frames if f['image']}
 available=[f for f in sorted(images) if any(e['start']<=f<=e['end'] for e in es)]
 # Development set uses all available frames, NEVER replaces missing sealed test frames.
 dev=[]
 for f in available:
  i=lookup[f];T=np.eye(4);T[:3,:3]=cam['R'][i];T[:3,3]=cam['C'][i]
  dev.append(dict(id=f'{f:06}',timestamp=float(cam['timestamps'][i]),image=str(images[f]),image_sha256=sha(images[f]),T_world_camera=T.tolist()))
 pc=sorted((w/'work/data/pcd').glob('*.pcd'))[-1]
 c=m['calibration'];K=[[c['fx'],0,c['cx']],[0,c['fy'],c['cy']],[0,0,1]]
 T=np.eye(4);T[:3,:3]=m['pose_check']['T_scene_loam_rotation'];T[:3,3]=m['pose_check']['T_scene_loam_translation']
 xyz=cam['C'][[lookup[f] for f in wanted]]
 bounds=[(xyz.min(0)-[15,5,15]).tolist(),(xyz.max(0)+[15,5,15]).tolist()]
 cfg=dict(schema=1,scene=m['sequence'],coordinate_system='scene XYZ metres, Y up; camera OpenCV x-right y-down z-forward',calibration_status=c['status'],K=K,image_size=[1280,720],frames=frames,replay_inputs=[dict(id=f'{f:06}',timestamp=float(cam['timestamps'][lookup[f]]),image=str(images[f]) if f in images else None,image_sha256=sha(images[f]) if f in images else None) for f in sorted({f for e in es for f in e['frame_ids']})],trajectories=[dict(id=e['id'],source_start=e['start'],source_end=e['end'],source_frame_count=e['count'],frame_ids=[f'{f:06}' for f in wanted if e['start']<=f<=e['end']]) for e in es],selection='T01-T18 chosen by original ID before detector execution, user authorized 2026-11-22',source_manifest_sha256=sha(source),pointcloud=dict(path=str(pc),sha256=sha(pc),T_world_source=T.tolist(),source='LeGO-LOAM cumulative snapshot',scope_verified=False,scope_note='Cumulative snapshot can contain scans outside selected intervals; crop is spatial only; not yet accepted as same-time baseline',voxel_m=0.10,fields='XYZ float32',format='NPY uncompressed',compression='none'),bounds_xyz=bounds,frame_stride=a.stride,min_confidence=0.3,merge_radius_m=0.8,region_radius_m=12.0,region_vote_min_frames=2,region_vote_min_motion_m=0.5,checkpoint='/data/user/ywang088/models/sam3/sam3.pt',bpe='/data/user/ywang088/projects/GGAP_exploration/r018_sam3_doors/vendor/bpe_simple_vocab_16e6.txt.gz',prompts={'column':['concrete pillar','parking garage column'],'vehicle':['car'],'door':['door','doorway'],'elevator':['elevator door'],'exit_sign':['exit sign'],'direction_sign':['direction sign'],'fire_equipment':['fire extinguisher','fire hydrant cabinet'],'parking_space':['parking space'],'traffic_cone':['traffic cone']},regions=[],portals=[],human_review=None)
 for name,obj in [('config.json',cfg),('development.json',dict(cfg,frames=dev,development_only=True)),('data_audit.json',dict(scene=m['sequence'],trajectories=18,source_frames=sum(e['count'] for e in es),selected_frames=len(wanted),selected_available=len(selected),missing_frames=[f for f in wanted if f not in selected],development_available=len(dev),calibration=c['status'],baseline_same_interval_verified=False,verdict='Offline prototype feasible; formal acceptance needs complete sealed inputs, baseline scope verification and human review'))]:
  (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
 print(json.dumps(json.loads((out/'data_audit.json').read_text()),ensure_ascii=False))
if __name__=='__main__':main()
