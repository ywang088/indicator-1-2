"""Deterministic post-processing of actual SAM3 masks, with synchronized raw LiDAR."""
import argparse,json,shutil
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
from perception import dump,sha,projection,zone_text

def surface(mask,proj,xyz,label):
 ids,uv,z=proj;ok=binary_erosion(mask,iterations=2)[uv[:,1],uv[:,0]];ids=ids[ok];z=z[ok]
 if len(ids)<8:return None,dict(points=len(ids),status='insufficient_synchronous_returns')
 order=np.argsort(z);ids=ids[order];z=z[order];groups=np.split(np.arange(len(z)),np.flatnonzero(np.diff(z)>.45)+1)
 groups=[g for g in groups if len(g)>=max(8,round(.05*len(z)))]
 if not groups:return None,dict(points=len(ids),status='no_supported_depth_component')
 # Foreground component removes background rays that pass beside a sparse column surface.
 g=groups[0];pts=xyz[ids[g]];center=np.median(pts,axis=0);spread=float(np.percentile(np.linalg.norm(pts-center,axis=1),80));extent=np.percentile(pts,95,axis=0)-np.percentile(pts,5,axis=0)
 limit=2.0 if label=='column' else 4.0
 if spread>limit:return None,dict(points=len(pts),status='mixed_or_extended_surface',spread_m=spread)
 return center.tolist(),dict(points=len(pts),status='synchronized_visible_surface',spread_m=spread,extent_xyz_m=extent.tolist(),source='matched_raw_velodyne',depth_component_count=len(groups))

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--readings',type=Path);p.add_argument('--rgb-root',type=Path);a=p.parse_args()
 c=json.loads(a.config.read_text())
 if a.rgb_root:
  for f in c['frames']:
   image=a.rgb_root/(f['id']+'.png')
   if not image.is_file() or sha(image)!=f['image_sha256']:raise ValueError('local RGB alias does not match sealed image '+f['id'])
   f['image']=str(image.resolve())
 fmap={f['id']:f for f in c['frames']};rows=[json.loads(l) for l in a.predictions.read_text().splitlines()];gp=np.load(a.geometry/'frame_points.npz');readings={}
 if a.readings:
  for l in a.readings.read_text().splitlines():
   r=json.loads(l)
   if r['labels'] is not None:
    for ob,label in zip(r['observations'],r['labels']):
     if label:
      if ob.get('image_sha256')!=fmap[ob['frame_id']]['image_sha256']:raise ValueError('column reading RGB hash mismatch')
      readings[ob['detection_id']]=dict(text=label,score=None,parsed=zone_text(label),source=r.get('source','unverified_model_crop_transcription'),source_job=r['id'],model_suggestion=True,review_status='assistant_visual_transcription_not_human_acceptance',image_sha256=ob['image_sha256'])
 a.output.mkdir(parents=True,exist_ok=False);(a.output/'images').mkdir();(a.output/'masks').mkdir();audit=json.loads((a.geometry/'geometry_audit.json').read_text());deltas={v['frame_id']:v['delta_s'] for v in audit['scans']}
 rejected=[];counts_before=Counter();counts_after=Counter()
 for row_index,row in enumerate(rows):
  fid=row['frame_id'];f=fmap[fid];xyz=gp['f'+fid];pr=projection(xyz,f,c['K'],c['image_size']);dets=row['detections'];masks={d['id']:np.array(Image.open(a.predictions.parent/d['mask']))>0 for d in dets};vehicles=[masks[d['id']] for d in dets if d['label']=='vehicle'];kept=[]
  for d in dets:
   counts_before[d['label']]+=1;m=masks[d['id']];reason=None
   if d['label'] in ('door','elevator') and vehicles:
    overlap=max(float(np.logical_and(m,v).sum()/max(1,m.sum())) for v in vehicles)
    if overlap>.55:reason='vehicle_door_not_building_portal'
   if d['label']=='parking_space' and m.mean()>.30:reason='whole_scene_parking_mask_not_an_individual_space'
   if reason:rejected.append(dict(frame_id=fid,detection_id=d['id'],label=d['label'],reason=reason));continue
   d['position'],d['geometry']=surface(m,pr,xyz,d['label']);d['geometry']['sensor_pair_delta_s']=deltas[fid]
   if d['id'] in readings:d['texts'].append(readings[d['id']])
   if d['label']=='parking_space':d['semantic_status']='parking_surface_candidate_not_verified_individual_bay'
   shutil.copyfile(a.predictions.parent/d['mask'],a.output/d['mask']);kept.append(d);counts_after[d['label']]+=1
  row['detections']=kept
  if (row_index+1)%25==0:print('refined',row_index+1,'/',len(rows),flush=True)
  # Native frame copied by the caller or original server path; never substitute generated imagery.
  im=Image.open(f['image']).convert('RGB');draw=ImageDraw.Draw(im)
  colors={'column':'#00bfaa','vehicle':'#e8a62c','door':'#bd66e5','elevator':'#bd66e5','exit_sign':'#e44951','direction_sign':'#39a4ff','parking_space':'#66997c'}
  for d in kept:
   if d['score']<.5:continue
   col=colors.get(d['label'],'#cccc33');draw.rectangle(d['bbox'],outline=col,width=2);label=d['label']+' '+f'{d["score"]:.2f}';codes=[t['parsed']['zone'] for t in d.get('texts',[]) if t.get('parsed') and t['parsed'].get('zone')]
   if codes:label+=' '+','.join(sorted(set(codes)))
   draw.text(d['bbox'][:2],label,fill=col)
  im.thumbnail((1280,720));im.save(a.output/'images'/f'{fid}.jpg',quality=90)
 (a.output/'detections.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
 shutil.copyfile(a.geometry/'points.npy',a.output/'matched_points.npy')
 c['parent_config_sha256']=sha(a.config);c['pointcloud']=dict(path=str((a.output/'matched_points.npy').resolve()),sha256=sha(a.output/'matched_points.npy'),T_world_source=np.eye(4).tolist(),voxel_m=c['pointcloud']['voxel_m'],fields='XYZ float32',format='NPY uncompressed',compression='none',source=f'synchronous raw Velodyne: same {len(c["frames"])} selected RGB frustums',scope_verified=True,scope_evidence=str((a.output/'scope_evidence.json').resolve()))
 dump(a.output/'config.json',c)
 dump(a.output/'scope_evidence.json',dict(scene=c['scene'],source_pointcloud_sha256=c['pointcloud']['sha256'],bounds_xyz=c['bounds_xyz'],trajectory_ids=[t['id'] for t in c['trajectories']],reviewer='deterministic same-input geometry audit, not human semantic acceptance',same_time_interval=True,same_effective_coverage=True,registration_verified=True,geometry_audit_sha256=sha(a.geometry/'geometry_audit.json'),limitation='Registration uses source odometry and estimated calibration; not externally measured absolute accuracy.'))
 prov=json.loads((a.predictions.parent/'provenance.json').read_text());prov.update(config_sha256=sha(a.output/'config.json'),parent_config_sha256=sha(a.config),parent_predictions_sha256=sha(a.predictions),postprocess_code_sha256=sha(Path(__file__)),synchronized_geometry_sha256=sha(a.geometry/'geometry_audit.json'),column_readings_sha256=sha(a.readings) if a.readings else None);dump(a.output/'provenance.json',prov)
 dump(a.output/'refinement_report.json',dict(before=dict(counts_before),after=dict(counts_after),rejected=rejected,localized=sum(d['position'] is not None for r in rows for d in r['detections']),observations=sum(len(r['detections']) for r in rows),column_text_added=len(readings)))
 print(json.dumps(dict(before=dict(counts_before),after=dict(counts_after),rejected=len(rejected)),ensure_ascii=False))
if __name__=='__main__':main()
