"""Export fixed image-review examples and graph IDs from the current replay."""
import json, shutil, sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'results/scene_review';out.mkdir(parents=True,exist_ok=True);records=[]
 samples=[('T02','000299','出口导向牌，不能据此确认实际出场口。'),('T06','001753','通道交汇位置，可作路口抽查。'),('T17','008976','画面含墙面门，原图不足以确认电梯；本次检测见相关记录。'),('T17','009016','车辆横向汇入主通道，可作通道汇合抽查。')]
 for tid,fid,note in samples:
  shutil.copy2(ROOT/'data/dataset/rgb'/f'{fid}.png',out/f'{fid}.png');db=sqlite3.connect(ROOT/'results/trajectories'/tid/'graph.sqlite');found=[]
  for oid,attrs in db.execute("select target,attributes from edges where source=? and kind='observes'",('frame:'+fid,)):
   a=json.loads(attrs);label=json.loads(db.execute('select attributes from nodes where id=?',(oid,)).fetchone()[0])['label']
   if label in ['elevator','exit_sign','direction_sign']:found.append(dict(object_id=oid,label=label,detection_id=a['detection_id'],bbox=a['bbox'],score=a['score']))
  db.close();records.append(dict(trajectory=tid,frame_id=fid,image=f'{fid}.png',visual_review=note,related_detections=found))
 (out/'scene_examples.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
 (out/'说明.txt').write_text('固定抽查候选图像，非人工签署结论。\n'+'\n'.join(f'{t}/{f}: {n}' for t,f,n in samples))
if __name__=='__main__':main()
