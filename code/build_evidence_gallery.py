import json,shutil,html
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
r=Path(__file__).resolve().parent;out=r/'final_results';dest=out/'column_evidence';dest.mkdir(exist_ok=True)
labels={34:'F：三视角清晰',199:'E：视角 0、2 清晰，中间视角不补猜',203:'E：三视角清晰',300:'E：三视角清晰',352:'E：三视角清晰'}
font=ImageFont.truetype('/System/Library/Fonts/STHeiti Light.ttc',24)
board=Image.new('RGB',(1000,len(labels)*310),'#edf2f6');d=ImageDraw.Draw(board);cards=[]
for j,(i,text) in enumerate(labels.items()):
 src=r/'column_groups/crops'/f'{i:04}.jpg';shutil.copyfile(src,dest/src.name);im=Image.open(src);im.thumbnail((960,270));board.paste(im,(20,j*310+35));d.text((20,j*310+5),text,fill='#173c58',font=font)
 cards.append(f'<section><h2>{html.escape(text)}</h2><img src="column_evidence/{src.name}"></section>')
board.save(out/'柱标识别抽查.jpg',quality=93)
body='''<!doctype html><meta charset="utf-8"><title>柱标识别证据</title><style>body{max-width:1100px;margin:40px auto;background:#edf2f6;color:#173c58;font:17px system-ui}section{padding:20px;background:white;margin:20px 0;border-radius:12px}img{max-width:100%}p{line-height:1.8}</style><h1>柱子标识 · 原始图像裁剪核对</h1><p>照片来自本次实际轨迹；只放大原始像素，没有生成或修复文字。下列 5 组补录 14 个清晰字母读数，与原始 OCR 的两个锚点一起形成 E / F 共 7 个多视角柱标锚点。补录由助手目视核对，保存原帧哈希，不能代替文档要求的人工验收。</p><p>其余抽查柱面多为无字、背面或遮挡，保持未知。小模型误读结果未采用。E / F 后面的编号仅是总览图中的锚点序号，不是柱上识别出的数字。</p><a href="index.html">返回分层图</a>'''+''.join(cards)
(out/'柱标证据.html').write_text(body)
p=out/'index.html';s=p.read_text().replace('<h1>停车场 · 可追溯分层认知地图</h1>','<h1>停车场 · 可追溯分层认知地图</h1><a style="color:#b8e7df" href="柱标证据.html">查看柱标照片核对</a>');p.write_text(s)
