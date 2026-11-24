"""Reconstruct historical object positions only from observations available at each step."""
import json,sqlite3,argparse,base64,io
from PIL import Image
from pathlib import Path
from collections import defaultdict
import numpy as np
b=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--result',type=Path,default=b/'results_20261124_connected');p.add_argument('--output',type=Path,default=Path('/Users/yixuanwang/.codex/visualizations/2026/11/22/01a20366-f049-7cc3-abdb-f630f96b0bd5/parking-eighteen-replays.html'));a=p.parse_args();r=a.result;c=json.load(open(r/'audit/config.json'));frames={f['id']:f for f in c['frames']};summaries=json.load(open(r/'audit/replays.json'));output=[]
for summary in summaries:
 t=summary['id'];p=r/'audit/replays'/t;db=sqlite3.connect(p/'map.sqlite');updates=[json.loads(l) for l in (p/'updates.jsonl').read_text().splitlines()];steps=[json.loads(l) for l in (p/'all_steps.jsonl').read_text().splitlines()];obs=defaultdict(list)
 for target,raw in db.execute("select target,attributes from edges where kind='observes'"):
  ob=json.loads(raw);obs[ob['frame_id']].append((target,ob['position']))
 ids={};history=defaultdict(list);positions={};records=[]
 for u in updates:
  fid=u['frame_id'];delta=[]
  for obj in u['new_objects']:
   assert obj not in ids
   ids[obj]=len(ids)
  for obj,pos in obs[fid]:
   assert obj in ids
   if pos is not None:history[obj].append(pos)
  for obj in set(u['new_objects']+u['updated_objects']):
   if history[obj]:
    pos=np.median(history[obj],axis=0);positions[obj]=pos;delta.append([ids[obj],round(float(pos[0]),3),round(float(pos[2]),3)])
  assert len(ids)==u['objects_after']
  f=frames[fid];records.append(dict(fid=fid,seconds=round(f['timestamp']-updates[0]['timestamp'],2),camera=[round(f['T_world_camera'][j][3],3) for j in [0,2]],objects=len(ids),localized=len(positions),new=len(u['new_objects']),updated=len(u['updated_objects']),delta=delta,regions=u.get('region_ids'),portals=u.get('portals_after'),new_portals=u.get('new_portals',[])))
 assert len(ids)==summary['objects'];assert len(ids)-len(positions)==summary['unlocalized_objects']
 regions=[i.split(':')[-1] for i, in db.execute("select id from nodes where kind='region'")]
 output.append(dict(id=t,source_steps=len(steps),keyframes=len(updates),regions=regions,portals=summary['portals'],records=records))
payload=dict(trajectories=output,source_steps=sum(t['source_steps'] for t in output),keyframe_updates=sum(t['keyframes'] for t in output),unique_keyframes=len({s['fid'] for t in output for s in t['records']}))
payload['evidence_images']=[]
for fid in ['004325','004345']:
 im=Image.open(b/'source_rgb'/f'{fid}.png');im.thumbnail((720,405));buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=78);payload['evidence_images'].append('data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode())
(b/'replay_inspection_20261124/data.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')))
s=(b/'replay_progress.template.html').read_text().replace('__REPLAY_DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')))
p=a.output;p.write_text(s);assert p.stat().st_size<1000000
(b/'replay_inspection_20261124/preview.js').write_text(s.split('<script>')[1].split('</script>')[0])
print(json.dumps(dict(path=str(p),bytes=p.stat().st_size,source_steps=payload['source_steps'],updates=payload['keyframe_updates'],unique=payload['unique_keyframes'],trajectories_with_regions=sum(bool(t['regions']) for t in output),all_replay_portals=sum(t['portals'] for t in output))))
