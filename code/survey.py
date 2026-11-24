"""Read-only real-image OCR survey; not SAM3 results, not accepted zone labels."""
import argparse,json,subprocess,tempfile,shutil,sys,html
from pathlib import Path
from PIL import Image,ImageDraw
from perception import sha,zone_text
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();c=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
frames=[f for f in c['frames'] if f['image']];lookup={f['id']:f for f in frames}
with tempfile.TemporaryDirectory(prefix='parking-ocr-') as temp:
 for f in frames:Path(temp,f['id']+Path(f['image']).suffix).symlink_to(Path(f['image']).resolve())
 r=subprocess.run(['swift',str(Path(__file__).with_name('ocr_macos.swift')),temp],capture_output=True,text=True,check=True)
 rows=[]
 for line in r.stdout.splitlines():
  x=json.loads(line);fid=Path(x['file']).stem;x['image_sha256']=sha(lookup[fid]['image']);x['file']=lookup[fid]['image'];rows.append(x)
 (a.output/'ocr.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n')
candidates=[];cards=[]
for x in rows:
 fid=Path(x['file']).stem;parsed=[dict(v,parsed=zone_text(v['text'])) for v in x['rows'] if zone_text(v['text']) and v['confidence']>=.65]
 if not parsed:continue
 candidates.append(dict(frame_id=fid,texts=parsed,status='OCR only; needs SAM3 column-mask association and multiview review'))
 im=Image.open(x['file']).convert('RGB');draw=ImageDraw.Draw(im)
 for v in parsed:
  px,py,w,h=v['box'];b=[px*im.width,(1-py-h)*im.height,(px+w)*im.width,(1-py)*im.height];draw.rectangle(b,outline='#00ff00',width=4)
 im.thumbnail((640,360));im.save(a.output/f'{fid}.jpg',quality=90);cards.append(f'<article><h3>{fid}</h3><img src="{fid}.jpg"><pre>{html.escape(json.dumps(parsed,ensure_ascii=False,indent=2))}</pre></article>')
(a.output/'candidates.json').write_text(json.dumps(candidates,ensure_ascii=False,indent=2))
(a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>柱标 OCR 可行性检查</title><style>body{font:16px system-ui;background:#eef2f6;padding:24px}article{background:white;padding:16px;margin:16px;display:inline-block;vertical-align:top;width:44%}img{width:100%}pre{white-space:pre-wrap}</style><h1>真实离线图像 · 柱标识可读性</h1><p>这些是 OCR 文本候选，尚未经过 SAM3 柱体关联，不作为确定区域。</p>'+''.join(cards))
print(json.dumps(dict(frames=len(frames),frames_with_parseable_text=len(candidates),output=str(a.output)),ensure_ascii=False))
