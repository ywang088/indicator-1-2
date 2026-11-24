"""Per-keyframe RGB, exact retained boxes and causal graph evidence, all 18 replays."""
import json,sqlite3,base64,io,hashlib,argparse,zipfile
from pathlib import Path
from PIL import Image
from color_topology import make_topology
b=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--result',type=Path,default=b/'incremental_20261124/final_verified/results');p.add_argument('--output',type=Path,default=b/'rgb_boxes_graph_20261124');p.add_argument('--inline-dir',type=Path,default=Path('/Users/yixuanwang/.codex/visualizations/2026/11/22/01a20366-f049-7cc3-abdb-f630f96b0bd5'));p.add_argument('--skip-inline',action='store_true');p.add_argument('--initial-trajectory');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
c=json.load(open(a.result/'audit/config.json'));fs={f['id']:f for f in c['frames']};rs={r['frame_id']:r for r in map(json.loads,(a.result/'audit/detections.jsonl').read_text().splitlines())};labels=sorted({d['label'] for r in rs.values() for d in r['detections']});tracks=[];total=0
for t in c['trajectories']:
 db=sqlite3.connect(a.result/'audit/replays'/t['id']/'map.sqlite');obs={json.loads(v)['detection_id']:oid for oid,v in db.execute("select target,attributes from edges where kind='observes'")};events=[json.loads(l) for l in (a.result/'audit/replays'/t['id']/'updates.jsonl').read_text().splitlines()];ids={};state=make_topology(c);records=[];seen_frames={}
 for j,event in enumerate(events):
  fid=event['frame_id'];seen_frames[fid]=j
  for oid in event['new_objects']:ids[oid]=len(ids)
  ds=[d for d in rs[fid]['detections'] if d['id'] in obs];step=state.update(fs[fid],[(obs[d['id']],d) for d in ds])
  assert step['region_ids']==event['region_ids'] and step['portal_ids']==event['portal_ids']
  assert len(ids)==event['objects_after']
  boxes=[[ids[obs[d['id']]],labels.index(d['label']),round(d['score'],3),*[round(float(v),1) for v in d['bbox']],','.join(sorted({t['parsed']['zone'] for t in d.get('texts',[]) if t.get('parsed') and t['parsed'].get('zone')}))] for d in ds]
  assert len({x[0] for x in boxes})==len(boxes),('two detections merged within one frame',fid)
  regions=[[r['label'],[ids[i] for i in r['anchors']],rid,r.get('binding_status','text'),r.get('color')] for rid,r in state.regions.items()]
  portals=[[[state.regions[r]['label'] for r in v['regions']],[ids[i] for i in v['anchors']],[seen_frames[f] for f in v['evidence_frames']],v['status']] for v in state.portals.values()]
  records.append(dict(fid=fid,seconds=round(fs[fid]['timestamp']-fs[events[0]['frame_id']]['timestamp'],2),count=len(ids),boxes=boxes,regions=regions,portals=portals));total+=len(boxes)
 tracks.append(dict(id=t['id'],ids=[i.removeprefix('object:') for i in ids],frames=records))
images={}
for fid in rs:
 path=b/'source_rgb'/f'{fid}.png';assert hashlib.sha256(path.read_bytes()).hexdigest()==fs[fid]['image_sha256'];images[fid]=Image.open(path).convert('RGB')
template=(b/'rgb_boxes_graph.template.html').read_text();wrapper=(b/'rgb_boxes_graph_wrapper.html').read_text()
def create(selected,width,quality,root,title):
 unique={f['fid'] for t in selected for f in t['frames']};encoded={}
 for fid in sorted(unique):
  im=images[fid].copy();im.thumbnail((width,width*9//16));buf=io.BytesIO();im.save(buf,format='WEBP',quality=quality);encoded[fid]='data:image/webp;base64,'+base64.b64encode(buf.getvalue()).decode()
 data=json.dumps(dict(labels=labels,tracks=selected,images=encoded,initialTrack=next((i for i,t in enumerate(selected) if t['id']==a.initial_trajectory),0)),ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
 markup=template
 if root!='parking-rgb-graph-full':markup=markup.replace('max-height:720px;overflow:auto','max-height:none;overflow:visible')
 return markup.replace('__ROOT_ID__',root).replace('__TITLE__',title).replace('__PAYLOAD__',data)
manifest=[]
for part in range(0 if a.skip_inline else 3):
 selected=tracks[part*6:(part+1)*6]
 for width,quality in [(400,45),(400,35),(360,35),(320,35),(320,25),(280,30)]:
  fragment=create(selected,width,quality,f'parking-rgb-graph-{c.get("sealed_version", "original")}-{part+1}',selected[0]['id']+'–'+selected[-1]['id']+' · RGB / 检测框 / graph')
  if len(fragment.encode())<980000:break
 assert len(fragment.encode())<1000000
 name=f'parking-rgb-graph-{part+1}.html';(a.inline_dir/name).write_text(fragment);(a.output/f'preview_{part+1}.html').write_text(wrapper.replace('__FRAGMENT__',fragment));manifest.append(dict(path=str(a.inline_dir/name),bytes=len(fragment.encode()),preview_width=width,keyframes=sum(len(t['frames']) for t in selected)))
fragment=create(tracks,1280,82,'parking-rgb-graph-full','18 条轨迹 · 逐帧 RGB / 检测框 / graph')
(a.output/'index.html').write_text(wrapper.replace('__FRAGMENT__',fragment));(a.output/'ui_check.js').write_text(fragment.split('<script>')[1].split('</script>')[0]);(a.output/'manifest.json').write_text(json.dumps(dict(tracks=18,unique_frames=len(images),replay_updates=sum(len(t['frames']) for t in tracks),boxes=total,causal_topology_matches_logs=True,graph_relations=['observes','column_label_support','connects','anchored_by','supports_connection'],omitted_relations=['nearest_column_candidate_contains','temporal_next'],inline=manifest),ensure_ascii=False,indent=2))
with zipfile.ZipFile(a.output/'逐帧_RGB_检测框_graph.zip','w',zipfile.ZIP_DEFLATED) as z:
 for n in ['index.html','manifest.json']:z.write(a.output/n,n)
print(json.dumps(dict(inline=manifest,full_bytes=(a.output/'index.html').stat().st_size,boxes=total),ensure_ascii=False))
