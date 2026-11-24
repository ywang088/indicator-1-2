"""Map native OCR crop boxes back to source pixels and require SAM3 mask support."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
from perception import zone_text,dump
p=argparse.ArgumentParser();p.add_argument('--groups',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
jobs=json.loads((a.groups/'jobs.json').read_text())['jobs'];rows=[json.loads(l) for l in (a.groups/'vision.jsonl').read_text().splitlines()];ds={d['id']:d for l in a.predictions.read_text().splitlines() for d in json.loads(l)['detections']};results=[];counts=Counter()
for r in rows:
 idx=int(Path(r['file']).stem);job=jobs[idx];n=len(job['observations']);candidates=[[] for _ in range(n)]
 for t in r['rows']:
  parsed=zone_text(t['text'])
  if not parsed or not parsed.get('zone') or t['confidence']<.65:continue
  bx,by,bw,bh=t['box'];box=np.array([bx*n*320,(1-by-bh)*480,(bx+bw)*n*320,(1-by)*480]);j=int((box[0]+box[2])/2//320)
  if not 0<=j<n or box[1]<30:continue
  ob=job['observations'][j];x0,y0,x1,y1=ob['bbox'];w=x1-x0;h=y1-y0;scale=min(300/w,440/h);nw,nh=round(w*scale),round(h*scale);px=j*320+(320-nw)//2;py=30+(440-nh)//2
  raw=np.rint((box-[px,py,px,py])/scale+[x0,y0,x0,y0]).astype(int);mask=np.array(Image.open(a.predictions.parent/ds[ob['detection_id']]['mask']))>0;rh,rw=mask.shape;u0,v0,u1,v1=raw;u0=max(0,u0);v0=max(0,v0);u1=min(rw,u1);v1=min(rh,v1)
  if u1<=u0 or v1<=v0 or mask[v0:v1,u0:u1].mean()<.7:continue
  candidates[j].append(dict(label=parsed['zone'],text=t['text'],confidence=t['confidence'],bbox=raw.tolist()))
 labels=[];scores=[]
 for cs in candidates:
  codes={v['label'] for v in cs};label=next(iter(codes)) if len(codes)==1 else None;labels.append(label);scores.append(max([v['confidence'] for v in cs],default=0));counts[label]+=1
 results.append(dict(**job,labels=labels,confidences=scores,ocr_evidence=candidates,engine='Apple_Vision_upscaled_column_crop_mask_bound',status='OCR_candidate_not_human_verified'))
a.output.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in results)+'\n');print(dict(counts))
valid=[r for r in results if any(r['labels'])];boards=[]
for i in range(0,len(valid),12):
 group=valid[i:i+12];out=Image.new('RGB',(1200,4*360),'white');draw=ImageDraw.Draw(out)
 for j,r in enumerate(group):
  im=Image.open(r['image']).convert('RGB');im.thumbnail((290,320));x=(j%4)*300;y=(j//4)*360;out.paste(im,(x,y+24));draw.text((x+3,y+3),f'{r["id"]} {r["labels"]}',fill='black')
 out=out.crop((0,0,1200,((len(group)+3)//4)*360));out.save(a.groups/f'review_{i//12:02}.jpg',quality=92)
dump(a.groups/'vision_summary.json',dict(jobs=len(jobs),views=sum(len(r['observations']) for r in results),counts={str(k):v for k,v in counts.items()},groups_with_text=len(valid)))
