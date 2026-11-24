"""Render actual SAM3 mask evidence, not synthesized images."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
r=Path(__file__).resolve().parent
rows={v['frame_id']:v for v in map(json.loads,(r/'refined_final/detections.jsonl').read_text().splitlines())}
frames=['000039','004661','007399','009390'];font=ImageFont.truetype('/System/Library/Fonts/STHeiti Light.ttc',22)
colors={'column':(0,185,150),'vehicle':(237,168,30),'door':(167,82,200),'elevator':(167,82,200),'exit_sign':(47,130,240),'direction_sign':(47,130,240)}
board=Image.new('RGB',(1920,1210),'#edf2f6');draw=ImageDraw.Draw(board)
for idx,fid in enumerate(frames):
 im=Image.open(r/'source_rgb'/f'{fid}.png').convert('RGB');pix=np.array(im,dtype=float);ds=[d for d in rows[fid]['detections'] if d['label'] in colors and d['score']>=.5]
 for d in ds:
  m=np.array(Image.open(r/'refined_final'/d['mask']))>0;pix[m]=.72*pix[m]+.28*np.array(colors[d['label']])
 im=Image.fromarray(np.uint8(np.clip(pix,0,255)));dr=ImageDraw.Draw(im)
 for d in ds:
  color=colors[d['label']];dr.rectangle(d['bbox'],outline=color,width=2);dr.text(d['bbox'][:2],d['label']+' '+f'{d["score"]:.2f}',fill=color)
 im.thumbnail((940,530));x=(idx%2)*960+10;y=(idx//2)*580+45;board.paste(im,(x,y));draw.text((x,y-32),f'真实帧 {fid} · 显示置信度 ≥ 0.5 的主要类别',font=font,fill='#173c58')
draw.text((20,1170),'绿：柱子  黄：车辆  紫：门候选  蓝：标牌；原始候选阈值 0.3，图示阈值 0.5。位置与类别仍需人工核验。',font=font,fill='#173c58');board.save(r/'final_results/SAM3识别抽查.jpg',quality=92)
