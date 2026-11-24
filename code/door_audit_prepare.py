import json,math
from pathlib import Path
from PIL import Image,ImageDraw
from functools import lru_cache
r=Path(__file__).resolve().parent;out=r/'door_audit_20261124';out.mkdir(exist_ok=True);(out/'crops').mkdir(exist_ok=True)
rows=[json.loads(l) for l in (r/'refined_final/detections.jsonl').read_text().splitlines()];items=[]
@lru_cache(maxsize=6)
def load(fid):return Image.open(r/'source_rgb'/f'{fid}.png').convert('RGB')
for row in rows:
 for d in row['detections']:
  if d['label'] not in ('door','elevator'):continue
  im=load(row['frame_id']);x0,y0,x1,y1=d['bbox'];w=x1-x0;h=y1-y0;cx=(x0+x1)/2;cy=(y0+y1)/2;cw=max(w*1.6,65);ch=max(h*1.4,75);b=(max(0,int(cx-cw/2)),max(0,int(cy-ch/2)),min(im.width,int(cx+cw/2)),min(im.height,int(cy+ch/2)))
  crop=im.crop(b);dr=ImageDraw.Draw(crop);dr.rectangle((x0-b[0],y0-b[1],x1-b[0],y1-b[1]),outline='#ffcc00',width=1);scale=min(190/crop.width,195/crop.height);crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS)
  tile=Image.new('RGB',(200,235),'#eef2f6');tile.paste(crop,((200-crop.width)//2,35));dr=ImageDraw.Draw(tile);dr.text((4,4),f'{len(items):03} {d["id"]} {d["score"]:.2f}',fill='black');dr.text((4,19),f'{w:.0f}x{h:.0f}px {d["label"]}',fill='black');tile.save(out/'crops'/f'{len(items):03}.jpg',quality=92);items.append(dict(index=len(items),frame_id=row['frame_id'],detection_id=d['id'],score=d['score'],bbox=d['bbox'],image_sha256=row['image_sha256']))
for page in range(math.ceil(len(items)/24)):
 board=Image.new('RGB',(1200,940),'white')
 for j,it in enumerate(items[page*24:(page+1)*24]):board.paste(Image.open(out/'crops'/f'{it["index"]:03}.jpg'),((j%6)*200,(j//6)*235))
 board.save(out/f'page_{page:02}.jpg',quality=94)
(out/'items.json').write_text(json.dumps(items,indent=2));print(len(items),'door/elevator observations',math.ceil(len(items)/24),'sheets')
