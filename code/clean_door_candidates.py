"""Apply hash-bound assistant visual audit; never equate door visibility with connectivity."""
import argparse,json,shutil
from pathlib import Path
from PIL import Image,ImageDraw
from perception import sha,dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 review=json.loads(a.review.read_text());rows=[json.loads(l) for l in a.predictions.read_text().splitlines()];c=json.loads(a.config.read_text());frames={f['id']:f for f in c['frames']};dec={v['detection_id']:v for v in review['decisions']}
 if review['predictions_sha256']!=sha(a.predictions) or len(dec)!=len(review['decisions']):raise ValueError('review predictions mismatch or duplicate decisions')
 expected={d['id'] for row in rows for d in row['detections'] if d['label'] in ('door','elevator')}
 if set(dec)!=expected:raise ValueError('review must cover every door/elevator observation, including uncertainty')
 a.output.mkdir(parents=True,exist_ok=False);(a.output/'masks').mkdir();(a.output/'images').mkdir();rejected=[]
 for row in rows:
  kept=[]
  for d in row['detections']:
   if d['id'] in dec:
    v=dec[d['id']]
    if v['frame_id']!=row['frame_id'] or v['image_sha256']!=row['image_sha256']:raise ValueError('review frame binding mismatch')
    if v['decision']=='reject_non_door':rejected.append(v);continue
    d['label']=v.get('corrected_label') or d['label'];d['semantic_status']='assistant_visible_door_unverified_connection' if v['decision']=='visible_door' else 'uncertain_door_candidate_unverified_connection';d['visual_review']=dict(source=review['reviewer'],decision=v['decision'],human_acceptance=False)
   shutil.copyfile(a.predictions.parent/d['mask'],a.output/d['mask']);kept.append(d)
  row['detections']=kept
  frame=frames[row['frame_id']];path=Path(frame['image'])
  if sha(path)!=row['image_sha256']:raise ValueError('RGB hash mismatch')
  im=Image.open(path).convert('RGB');draw=ImageDraw.Draw(im);colors={'column':'#00bfaa','vehicle':'#e8a62c','door':'#bd66e5','exit_sign':'#e44951','direction_sign':'#39a4ff','parking_space':'#66997c'}
  for d in kept:
   if d['score']<.5:continue
   col=colors.get(d['label'],'#cccc33');draw.rectangle(d['bbox'],outline=col,width=2);draw.text(d['bbox'][:2],d['label']+' '+d['id'],fill=col)
  im.thumbnail((1280,720));im.save(a.output/'images'/f'{row["frame_id"]}.jpg',quality=90)
 (a.output/'detections.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
 shutil.copyfile(a.config,a.output/'config.json')
 prov=json.loads((a.predictions.parent/'provenance.json').read_text());prov.update(parent_predictions_sha256=sha(a.predictions),door_audit_sha256=sha(a.review),door_cleanup_code_sha256=sha(Path(__file__)),human_review_completed=False);dump(a.output/'provenance.json',prov);dump(a.output/'door_cleanup_report.json',dict(reviewed=len(dec),removed_observations=len(rejected),remaining_door_observations=len(dec)-len(rejected),rejected=rejected,reviewer=review['reviewer'],limitation='Assistant visual audit, not human acceptance. Door visibility never establishes both incident regions.'))
 print('reviewed',len(dec),'removed',len(rejected),'kept door candidates',len(dec)-len(rejected))
if __name__=='__main__':main()
