"""Bind frame-local text evidence; remove global reviewed portals before causal replay."""
import argparse,json,shutil,re
from pathlib import Path
import numpy as np
from PIL import Image
from perception import sha,dump,zone_text
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--readings',type=Path);p.add_argument('--crop-jobs',type=Path);p.add_argument('--crop-ocr',type=Path);a=p.parse_args()
c=json.loads((a.source/'config.json').read_text());rows=[json.loads(l) for l in (a.source/'detections.jsonl').read_text().splitlines()];ds={d['id']:(r,d) for r in rows for d in r['detections']};count=0
if a.crop_ocr:
 jobs={j['frame_id']:j for j in json.loads(a.crop_jobs.read_text())}
 for raw in a.crop_ocr.read_text().splitlines():
  o=json.loads(raw);fid=Path(o['file']).stem;job=jobs[fid]
  for t in o['rows']:
   text=t['text'].strip().upper()
   if not re.fullmatch('[A-H]',text) or t['confidence']<.3:continue
   bx,by,bw,bh=t['box'];bb=np.array([bx*job['width'],(1-by-bh)*job['height'],(bx+bw)*job['width'],(1-by)*job['height']]);j=int((bb[0]+bb[2])/2//320)
   if not 0<=j<len(job['columns']):continue
   ob=job['columns'][j];r,d=ds[ob['detection_id']]
   if r['image_sha256']!=job['image_sha256']:raise ValueError('crop OCR input hash mismatch')
   x,y=ob['offset'];x0,y0,_,_=ob['bbox'];box=np.rint((bb-[x,y,x,y])/ob['scale']+[x0,y0,x0,y0]).astype(int);mask=np.asarray(Image.open(a.source/d['mask']))>0;u0,v0,u1,v1=box;u0=max(0,u0);v0=max(0,v0);u1=min(mask.shape[1],u1);v1=min(mask.shape[0],v1)
   if u1<=u0 or v1<=v0 or mask[v0:v1,u0:u1].mean()<.7:continue
   d.setdefault('texts',[]).append(dict(text=text,score=t['confidence'],parsed=zone_text(text),source='Apple_Vision_current_frame_mask_bound_crop',bbox=box.tolist(),image_sha256=r['image_sha256'],mask_overlap=float(mask[v0:v1,u0:u1].mean())));count+=1
review_count=0
if a.readings:
 for e in json.loads(a.readings.read_text()):
  r,d=ds[e['detection_id']]
  if r['image_sha256']!=e['image_sha256'] or r['frame_id']!=e['frame_id'] or d['label']!='column' or d['bbox']!=e['bbox']:raise ValueError('text review is not bound to original column')
  d.setdefault('texts',[]).append(dict(e,score=None,parsed=zone_text(e['text'])));review_count+=1
c.update(incremental_topology=True,portals=[],human_review=None,parent_config_sha256=sha(a.source/'config.json'))
a.output.mkdir(parents=True,exist_ok=False);dump(a.output/'config.json',c);(a.output/'detections.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
for name in ['images','masks']:(a.output/name).symlink_to((a.source/name).resolve(),target_is_directory=True)
prov=json.loads((a.source/'provenance.json').read_text());prov.update(config_sha256=sha(a.output/'config.json'),parent_predictions_sha256=sha(a.source/'detections.jsonl'),incremental_text_review_sha256=sha(a.readings) if a.readings else None,native_crop_ocr_sha256=sha(a.crop_ocr) if a.crop_ocr else None,incremental_topology=True,human_review_completed=False);dump(a.output/'provenance.json',prov)
dump(a.output/'input_changes.json',dict(added_ocr_readings=count,added_assistant_readings=review_count,global_reviewed_portals_removed=True));print(count,review_count)
