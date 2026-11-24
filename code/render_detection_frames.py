"""Render recorded detections with replay graph IDs; no inference is fabricated."""
import json, sqlite3, colorsys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]
def font(size):
 for p in ['/System/Library/Fonts/Menlo.ttc','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
  if Path(p).exists():return ImageFont.truetype(p,size)
 return ImageFont.load_default()
def render_frame(rgb, ds, observed, dest, fid, trajectory):
 im=Image.open(rgb).convert('RGB');w,h=im.size; f=font(15);small=font(12)
 rows=[]
 canvas=Image.new('RGB',(w+500,max(h+50,len(ds)*29+90)), '#121b2a');canvas.paste(im,(0,50));draw=ImageDraw.Draw(canvas)
 draw.text((12,14),f'{trajectory} / RGB {fid} / {len(ds)} detections',font=f,fill='white')
 draw.text((w+12,14),'Box # / label / replay graph object ID',font=f,fill='white')
 for i,d in enumerate(ds,1):
  oid=observed.get(d['id']);color=tuple(int(v*255) for v in colorsys.hsv_to_rgb((i*.618)%1,.65,1));x1,y1,x2,y2=d['bbox'];y1+=50;y2+=50
  draw.rectangle((x1,y1,x2,y2),outline=color,width=2)
  label=f"{i} | {oid.removeprefix('object:') if oid else 'not mapped'}"
  tw=draw.textbbox((0,0),label,font=small)[2];x=max(0,min(x1,w-tw-6));y=max(50,y1-17)
  draw.rectangle((x,y,x+tw+6,y+17),fill='#121b2a');draw.text((x+3,y+1),label,font=small,fill=color)
  yy=55+(i-1)*29;draw.text((w+12,yy),f"{i:02d} {d['label']}  {oid or 'not mapped'}",font=small,fill=color)
  rows.append(dict(box_number=i,detection_id=d['id'],object_id=oid,label=d['label'],bbox=d['bbox']))
 draw.text((w+12,55+len(ds)*29),'Box label omits the common object: prefix.',font=small,fill='white')
 dest.parent.mkdir(parents=True,exist_ok=True);canvas.save(dest,quality=92)
 dest.with_suffix('.json').write_text(json.dumps(dict(trajectory=trajectory,frame_id=fid,graph='../graph.json',detections=rows),ensure_ascii=False,indent=2))
def main():
 c=json.loads((ROOT/'results/run_config.json').read_text());rows={r['frame_id']:r['detections'] for r in map(json.loads,(ROOT/'data/dataset/detections.jsonl').read_text().splitlines())};count=0
 for t in c['trajectories']:
  out=ROOT/'results/trajectories'/t['id'];db=sqlite3.connect(out/'graph.sqlite');observed={json.loads(a)['detection_id']:oid for oid,a in db.execute("select target,attributes from edges where kind='observes'")};nodes={i for i, in db.execute('select id from nodes')}
  for fid in t['frame_ids']:
   assert all(observed.get(d['id']) in nodes for d in rows[fid]),fid
   render_frame(ROOT/'data/dataset/rgb'/f'{fid}.png',rows[fid],observed,out/'frames'/f'{fid}.jpg',fid,t['id']);count+=1
  db.close()
 print(json.dumps({'rendered_frames':count,'all_detection_ids_resolve_to_replay_graph':True}))
if __name__=='__main__':main()
