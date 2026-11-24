"""Measure pillar paint from hash-bound RGB and SAM3 masks; no preassigned letters."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from perception import sha,dump

def pillar_paint(hsv,mask,bbox,score):
 x0,y0,x1,y1=map(int,bbox);h=y1-y0;w=x1-x0
 if score<.6 or h<94 or h<1.4*w or y1<mask.shape[0]*.52:return None
 crop=hsv[max(0,int(y0+.08*h)):int(y0+.60*h),max(0,int(x0+.12*w)):int(x1-.12*w)]
 roi=mask[max(0,int(y0+.08*h)):int(y0+.60*h),max(0,int(x0+.12*w)):int(x1-.12*w)]
 H,S,V=crop.transpose(2,0,1);valid=roi&(V>.18)&(V<.98);n=int(valid.sum())
 if n<150:return None
 chromatic=valid&(S>.28)
 colors={'red':((H<.055)|(H>.94)),'yellow':((H>=.055)&(H<.20)),'green':((H>=.20)&(H<.47)),'blue':((H>=.47)&(H<.77)),'purple':((H>=.77)&(H<=.94))}
 counts={k:int((chromatic&m).sum()) for k,m in colors.items()};winner=max(counts,key=counts.get);fraction=counts[winner]/n;purity=counts[winner]/max(1,sum(counts.values()))
 if fraction<.35 or purity<.75 or counts[winner]<100:return None
 return dict(color=winner,paint_fraction=round(fraction,3),purity=round(purity,3),pixels=counts[winner],method='SAM3_column_mask_upper_paint_HSV',neutral_colors_excluded=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 c=json.loads((a.source/'config.json').read_text());fs={f['id']:f for f in c['frames']};rows=[];counts={};bound=[]
 for r in map(json.loads,(a.source/'detections.jsonl').read_text().splitlines()):
  image=Path(fs[r['frame_id']]['image']);assert sha(image)==r['image_sha256'];hsv=np.asarray(Image.open(image).convert('HSV'),dtype=float)/255
  for d in r['detections']:
   if d['label']!='column':continue
   paint=pillar_paint(hsv,np.asarray(Image.open(a.source/d['mask']))>0,d['bbox'],d['score'])
   if paint:
    paint['image_sha256']=r['image_sha256'];d['pillar_color']=paint;counts[paint['color']]=counts.get(paint['color'],0)+1
    texts=[t['parsed']['zone'] for t in d.get('texts',[]) if t.get('parsed') and t['parsed'].get('zone')]
    if texts:bound.append(dict(id=d['id'],color=paint['color'],letters=texts))
  rows.append(r)
 c['color_regions']=True;c['sealed_version']='18-v3-color-v1-20261124';c['parent_config_sha256']=sha(a.source/'config.json')
 dump(out/'config.json',c);(out/'detections.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 for name in ['images','masks']:(out/name).symlink_to((a.source/name).resolve(),target_is_directory=True)
 for name in ['selection_manifest.json','scope_evidence.json','geometry_audit.json','scan_extraction_manifest.json']:
  if (a.source/name).exists():shutil.copy2(a.source/name,out/name)
 prov=json.loads((a.source/'provenance.json').read_text());prov.update(config_sha256=sha(out/'config.json'),parent_predictions_sha256=sha(a.source/'detections.jsonl'),paint_extractor_sha256=sha(Path(__file__)));dump(out/'provenance.json',prov)
 dump(out/'color_measurements.json',dict(counts=counts,text_color_observations=bound,letters_not_preassigned=True));dump(out/'frozen_files.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()});print(counts);print(bound)
if __name__=='__main__':main()
