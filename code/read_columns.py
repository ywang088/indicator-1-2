"""Offline VLM transcription of SAM3 column crops; suggestions, never human truth."""
import argparse,json,re,time,os
from pathlib import Path
from PIL import Image,ImageDraw
from perception import sha,dump,zone_text

def prepare(config,predictions,output,objects=None):
 c=json.loads(config.read_text());frames={f['id']:f for f in c['frames']};rows=[json.loads(l) for l in predictions.read_text().splitlines()];byid={d['id']:(r,d) for r in rows for d in r['detections']}
 output.mkdir(parents=True,exist_ok=False);(output/'crops').mkdir();jobs=[]
 if objects:
  import sqlite3
  db=sqlite3.connect(objects);groups=[dict(json.loads(a),id=i) for i,a in db.execute("select id,attributes from nodes where kind='object'")];groups=[g for g in groups if g['label']=='column']
  for g in groups:
   if 'observations' not in g:g['observations']=[json.loads(a) for a, in db.execute("select attributes from edges where target=? and kind='observes'",(g['id'],))]
  db.close()
 else:groups=[dict(id=d['id'],observations=[dict(detection_id=d['id'],frame_id=r['frame_id'])]) for r in rows for d in r['detections'] if d['label']=='column']
 for group in groups:
  eligible=[]
  for ob in group['observations']:
   r,d=byid[ob['detection_id']];x0,y0,x1,y1=d['bbox'];area=max(0,x1-x0)*max(0,y1-y0)
   if x1-x0>=28 and y1-y0>=80 and d['score']>=.5:eligible.append((area,r,d))
  selected=[]
  for _,r,d in sorted(eligible,key=lambda t:-t[0]):
   f=frames[r['frame_id']];pos=[row[3] for row in f['T_world_camera'][:3]]
   if selected:
    import numpy as np
    if min(np.linalg.norm(np.array(pos)-x[2]) for x in selected)<.5:continue
   selected.append((r,d,pos))
   if len(selected)>=3:break
  if not selected:continue
  board=Image.new('RGB',(len(selected)*320,480),'#eeeeee');draw=ImageDraw.Draw(board);observations=[]
  for j,(r,d,pos) in enumerate(selected):
   f=frames[r['frame_id']];im=Image.open(f['image']).convert('RGB');x0,y0,x1,y1=d['bbox'];b=(max(0,int(x0)),max(0,int(y0)),min(im.width,int(x1)),min(im.height,int(y1)))
   crop=im.crop(b);scale=min(300/crop.width,440/crop.height);crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS);board.paste(crop,(j*320+(320-crop.width)//2,30+(440-crop.height)//2));draw.text((j*320+10,10),f'[{j}] frame {r["frame_id"]}',fill='black')
   observations.append(dict(frame_id=r['frame_id'],detection_id=d['id'],bbox=list(b),image_sha256=f['image_sha256']))
  name=f'{len(jobs):04}.jpg';board.save(output/'crops'/name,quality=95)
  jobs.append(dict(id=group['id'],image=str((output/'crops'/name).resolve()),observations=observations))
 dump(output/'jobs.json',dict(parent_predictions_sha256=sha(predictions),jobs=jobs));print('column jobs',len(jobs))
def infer(jobs_path,output,model_path):
 import torch
 from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
 torch.set_num_threads(4)
 processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,min_pixels=256*28*28,max_pixels=768*28*28)
 model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
 payload=json.loads(jobs_path.read_text());output.mkdir(parents=True,exist_ok=False);done=[]
 for job in payload['jobs']:
  t=time.monotonic();im=Image.open(job['image']).convert('RGB');count=len(job['observations'])
  prompt=f'These are {count} numbered photographic crops of structural pillars in a parking garage. For each crop, read ONLY the LARGE parking ZONE LETTER actually painted on the PILLAR. Ignore car plates, overhead signs, wall arrows, logos and floor labels such as B2. A zone is usually one Latin letter A to H. Do NOT infer from color or copy another crop. If no clear letter is visible, use null. Return ONLY a JSON array of {count} entries in index order, for example ["F", null]. No explanation.'
  messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}];text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);inputs=processor(text=[text],images=[im],padding=True,return_tensors='pt').to('cuda')
  with torch.inference_mode():gen=model.generate(**inputs,max_new_tokens=48,do_sample=False)
  answer=processor.batch_decode(gen[:,inputs.input_ids.shape[-1]:],skip_special_tokens=True)[0];labels=None
  try:
   parsed=json.loads(re.search(r'\[[\s\S]*\]',answer)[0])
   if len(parsed)==count and all(v is None or (isinstance(v,str) and re.fullmatch('[A-H]',v.strip().upper())) for v in parsed):labels=[v.strip().upper() if v else None for v in parsed]
  except (ValueError,TypeError,AttributeError):pass
  row=dict(**job,answer=answer,labels=labels,seconds=time.monotonic()-t,status='model_suggestion_not_human_verified')
  with (output/'readings.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
  done.append(row);print(job['id'],labels,round(row['seconds'],2),flush=True)
 dump(output/'provenance.json',dict(model=str(model_path),source_jobs_sha256=sha(jobs_path),parent_predictions_sha256=payload['parent_predictions_sha256'],completed=True,jobs=len(done),method='local Qwen2.5-VL deterministic crop transcription; uncalibrated model suggestions'))
if __name__=='__main__':
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest='mode',required=True);q=sp.add_parser('prepare');q.add_argument('--config',type=Path,required=True);q.add_argument('--predictions',type=Path,required=True);q.add_argument('--objects');q.add_argument('--output',type=Path,required=True);q=sp.add_parser('infer');q.add_argument('--jobs',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--model',default='/data/user/ywang088/models/Qwen2.5-VL-3B-Instruct');a=p.parse_args()
 if a.mode=='prepare':prepare(a.config,a.predictions,a.output,a.objects)
 else:infer(a.jobs,a.output,a.model)
