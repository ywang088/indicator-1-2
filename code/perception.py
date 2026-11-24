"""SAM3 native image grounding + OCR bound to actual column masks."""
import argparse, json, hashlib, re, time, subprocess, sys, os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
from geometry_io import read_pcd

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def pointmap(c):
 p=Path(c['pointcloud']['path'])
 if sha(p)!=c['pointcloud']['sha256']:raise ValueError('point cloud hash changed')
 xyz=read_pcd(p)[0] if p.suffix=='.pcd' else np.load(p,allow_pickle=False)
 T=np.array(c['pointcloud']['T_world_source']);xyz=np.einsum('ni,ji->nj',xyz,T[:3,:3])+T[:3,3]
 lo,hi=np.array(c['bounds_xyz']);xyz=xyz[((xyz>=lo)&(xyz<=hi)).all(1)]
 if not len(xyz):raise ValueError('empty scoped point cloud')
 _,idx=np.unique(np.floor(xyz/c['pointcloud']['voxel_m']).astype(np.int64),axis=0,return_index=True)
 return xyz[np.sort(idx)].astype(np.float32)
def projection(xyz,f,K,size):
 T=np.array(f['T_world_camera']);q=np.einsum('ni,ij->nj',xyz-T[:3,3],T[:3,:3]);z=q[:,2];keep=(z>0.3)&(z<40)
 ids=np.flatnonzero(keep);q=q[keep];uv=np.einsum('ni,ji->nj',q,np.array(K));uv=np.rint(uv[:,:2]/uv[:,2:3]).astype(int)
 w,h=size;ok=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);ids=ids[ok];uv=uv[ok]
 # Nearest projected point in each pixel; no artificial hole filling.
 order=np.argsort(z[ids]);ids=ids[order];uv=uv[order];_,ix=np.unique(uv[:,1]*w+uv[:,0],return_index=True)
 return ids[ix],uv[ix],z[ids[ix]]
def locate(mask,proj,xyz):
 ids,uv,z=proj;inside=binary_erosion(mask,iterations=2)[uv[:,1],uv[:,0]];ids=ids[inside];z=z[inside]
 if len(ids)<8:return None,dict(points=len(ids),status='insufficient_surface_support')
 med=np.median(z);mad=np.median(abs(z-med));ids=ids[abs(z-med)<=max(.25,2.5*mad)]
 if len(ids)<8:return None,dict(points=len(ids),status='insufficient_inliers')
 pts=xyz[ids];center=np.median(pts,axis=0);spread=np.percentile(np.linalg.norm(pts-center,axis=1),80)
 if spread>4:return None,dict(points=len(ids),status='mixed_or_extended_surface',spread_m=float(spread))
 return center.tolist(),dict(points=len(ids),status='estimated_visible_surface',spread_m=float(spread),source='projected_cumulative_lidar_not_synchronous_depth')
def zone_text(text):
 # Single letters permitted ONLY after geometric column-mask association.
 s=re.sub(r'\s+','',text.upper()).replace('区域','区')
 floor_only=re.fullmatch(r'([BPL]\d{1,2})',s)
 if floor_only:return dict(floor=floor_only[1],zone=None,pillar_number=None,raw=text)
 m=re.fullmatch(r'(?:([BPL]\d{1,2})[-·]?)?([A-H])(?:区)?(?:[-·]?(\d{1,3}))?',s)
 if m:return dict(floor=m[1],zone=m[2],pillar_number=m[3],raw=text)
 m=re.fullmatch(r'([BPL]\d{1,2})',s)
 return dict(floor=m[1],zone=None,pillar_number=None,raw=text) if m else None

def iou(a,b):
 inter=np.logical_and(a,b).sum();return inter/max(1,np.logical_or(a,b).sum())
class Sam:
 def __init__(self,c,device):
  import torch
  from sam3.model_builder import build_sam3_image_model
  from sam3.model.sam3_image_processor import Sam3Processor
  self.torch=torch
  self.model=build_sam3_image_model(checkpoint_path=c['checkpoint'],load_from_HF=False,device=device,bpe_path=c['bpe']);self.model.eval()
  self.processor=Sam3Processor(self.model,device=device,confidence_threshold=c['min_confidence']);self.c=c
 def __call__(self,im):
  rows=[]
  with self.torch.inference_mode():
   state=self.processor.set_image(im)
   for label,prompts in self.c['prompts'].items():
    candidates=[]
    for prompt in prompts:
     self.processor.reset_all_prompts(state);out=self.processor.set_text_prompt(state=state,prompt=prompt)
     for mask,box,score in zip(out['masks'],out['boxes'],out['scores']):
      mask=mask.detach().cpu().numpy().squeeze().astype(bool)
      if mask.shape!=(im.height,im.width):raise ValueError('SAM3 mask/image size mismatch')
      if mask.sum()<30:continue
      candidates.append(dict(label=label,prompt=prompt,score=float(score),bbox=box.detach().cpu().tolist(),mask=mask))
    kept=[]
    for d in sorted(candidates,key=lambda d:-d['score']):
     if all(iou(d['mask'],k['mask'])<.6 for k in kept):kept.append(d)
    rows.extend(kept)
  return rows
class OCR:
 def __init__(self,engine,cache,device="cpu"):
  self.engine=engine;self.cache={}
  if engine=='json':
   for line in Path(cache).read_text().splitlines():
    x=json.loads(line);self.cache[Path(x['file']).stem]=x
  elif engine=='easyocr':
   import easyocr
   model_dir=os.environ.get('EASYOCR_MODEL_DIR')
   self.reader=easyocr.Reader(['ch_sim','en'],gpu=device if device.startswith('cuda') else False,model_storage_directory=model_dir,download_enabled=not bool(model_dir))
 def __call__(self,im,f):
  if self.engine=='easyocr':
   rows=[]
   for poly,text,score in self.reader.readtext(np.array(im)):
    a=np.array(poly);rows.append(dict(text=text,score=float(score),bbox=[*a.min(0).tolist(),*a.max(0).tolist()]))
   return rows
  x=self.cache.get(f['id'])
  if x is None:raise ValueError('OCR cache missing frame '+f['id'])
  if x.get('image_sha256')!=f['image_sha256']:raise ValueError('OCR cache image hash mismatch')
  rows=[]
  for v in x['rows']:
   if 'bbox' in v:rows.append(v);continue
   x0,y0,w,h=v['box'];rows.append(dict(text=v['text'],score=v['confidence'],bbox=[x0*im.width,(1-y0-h)*im.height,(x0+w)*im.width,(1-y0)*im.height]))
  return rows

def attach_text(d,ocr):
 rows=[];mask=d['mask'];h,w=mask.shape
 for t in ocr:
  b=np.rint(t['bbox']).astype(int);x0,y0,x1,y1=b;x0=max(0,x0);y0=max(0,y0);x1=min(w,x1);y1=min(h,y1)
  if x1<=x0 or y1<=y0:continue
  ratio=float(mask[y0:y1,x0:x1].mean())
  if ratio>=.7:
   row=dict(t,mask_overlap=ratio);row['parsed']=zone_text(t['text']) if d['label']=='column' and t['score']>=.65 else None;rows.append(row)
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--ocr',choices=['easyocr','json'],default='easyocr');p.add_argument('--ocr-json');p.add_argument('--checkpoint');p.add_argument('--bpe');p.add_argument('--limit',type=int);a=p.parse_args()
 c=json.loads(a.config.read_text());c['checkpoint']=a.checkpoint or c['checkpoint'];c['bpe']=a.bpe or c['bpe']
 frames=c['frames'][:a.limit] if a.limit else c['frames']
 if any(not f['image'] or not Path(f['image']).is_file() for f in frames):raise ValueError('Missing frozen RGB inputs; run prepare with complete --rgb-root or explicit development config')
 for key in ['checkpoint','bpe']:
  if not Path(c[key]).is_file():raise FileNotFoundError(c[key])
 a.output.mkdir(parents=True,exist_ok=False);(a.output/'masks').mkdir();(a.output/'images').mkdir()
 xyz=pointmap(c);model=Sam(c,a.device);ocr=OCR(a.ocr,a.ocr_json,a.device)
 import importlib.metadata
 provenance=dict(code_sha256={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},config_sha256=sha(a.config),checkpoint_sha256=sha(c['checkpoint']),sam3_version=importlib.metadata.version('sam3'),python=sys.version,torch=importlib.metadata.version('torch'),numpy=np.__version__,pillow=importlib.metadata.version('Pillow'),threshold=c['min_confidence'],limit=a.limit,device=a.device,ocr=a.ocr,completed=False)
 dump(a.output/'provenance.json',provenance)
 with (a.output/'detections.jsonl').open('w') as log:
  for f in frames:
   if sha(f['image'])!=f['image_sha256']:raise ValueError('RGB hash mismatch '+f['id'])
   started=time.monotonic();im=Image.open(f['image']).convert('RGB')
   if list(im.size)!=c['image_size']:raise ValueError('image size does not match calibration')
   dets=model(im);texts=ocr(im,f)
   # Small painted column lettering is often missed at full-image scale.
   if a.ocr=='easyocr':
    for d in dets:
     if d['label']!='column':continue
     x0,y0,x1,y1=[int(v) for v in d['bbox']];x0=max(0,x0);y0=max(0,y0);x1=min(im.width,x1);y1=min(im.height,y1)
     if x1-x0<12 or y1-y0<20:continue
     crop=im.crop((x0,y0,x1,y1));factor=min(4.,1200/max(crop.size));crop=crop.resize((round(crop.width*factor),round(crop.height*factor)))
     for t in ocr(crop,f):
      b=t['bbox'];t['bbox']=[b[0]/factor+x0,b[1]/factor+y0,b[2]/factor+x0,b[3]/factor+y0];t['source']='enlarged_column_crop';texts.append(t)
   proj=projection(xyz,f,c['K'],c['image_size']);vis=im.copy();draw=ImageDraw.Draw(vis)
   for i,d in enumerate(dets):
    d['id']=f'{f["id"]}:{i:03}';d['texts']=attach_text(d,texts);d['position'],d['geometry']=locate(d['mask'],proj,xyz)
    maskname=f'masks/{f["id"]}_{i:03}.png';Image.fromarray(d.pop('mask').astype(np.uint8)*255).save(a.output/maskname);d['mask']=maskname
    draw.rectangle(d['bbox'],outline='#00ffb0',width=3);draw.text(d['bbox'][:2],f'{i} {d["label"]} {d["score"]:.2f}',fill='#00ffb0')
   vis.thumbnail((960,540));vis.save(a.output/'images'/f'{f["id"]}.jpg',quality=88)
   row=dict(frame_id=f['id'],timestamp=f['timestamp'],image_sha256=f['image_sha256'],detections=dets,ocr=texts,seconds=time.monotonic()-started)
   log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');log.flush();print(f['id'],len(dets),round(row['seconds'],2),flush=True)
 provenance.update(completed=True,frames=len(frames));dump(a.output/'provenance.json',provenance)
if __name__=='__main__':main()
