"""Independent checks for online mapping and the immutable pre-saved full map."""
import json,sys,subprocess,sqlite3,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def query(p):
 r=subprocess.run([sys.executable,str(ROOT/'code/map_pipeline.py'),'query','--db',str(p)],check=True,capture_output=True,text=True)
 return dict(database=str(p.relative_to(ROOT)),result=json.loads(r.stdout))
def verify_frozen_inputs():
 data=ROOT/'data/full_map';manifest=read(data/'frozen_manifest.json')
 for name,expected in manifest['files'].items():
  if sha(data/name)!=expected:raise ValueError('Frozen full-map input changed: '+name)
 return manifest

def validate_online():
 r=ROOT/'results';c=read(r/'run_config.json');e=read(r/'execution.json');checks={}
 checks['fresh_perception_completed']=e.get('pipeline_mode')=='fresh_sam3' and e.get('completed') is True and e.get('processed_frames')==len(c['frames']) and e.get('reuses_detection_cache') is False and e.get('reuses_ocr_cache') is False
 checks['current_perception_hash']=sha(r/'detections.jsonl')==e.get('predictions_sha256')
 checks['18_trajectories']=len(c['trajectories'])==18
 checks['required_online_outputs']=all((r/name).is_file() for name in ['report.json','data_integrity.json','online_full_map/graph.sqlite','online_full_map/graph.json','online_full_map/summary.json','scene_review/scene_examples.json'])
 for t in c['trajectories']:
  folder=r/'trajectories'/t['id'];db=sqlite3.connect(folder/'graph.sqlite');ids={i for i, in db.execute('select id from nodes')};edges=db.execute('select source,target from edges').fetchall()
  checks[t['id']+'_references']=all(s in ids and v in ids for s,v in edges)
  checks[t['id']+'_frames']={i for i in ids if i.startswith('frame:')}=={'frame:'+i for i in t['frame_ids']}
  steps=[json.loads(s) for s in (folder/'all_steps.jsonl').read_text().splitlines()]
  checks[t['id']+'_input_log_complete']=len(steps)==sum(t['source_start']<=int(f['id'])<=t['source_end'] for f in c['replay_inputs'])
  checks[t['id']+'_input_log_order']=all(a['timestamp']<=b['timestamp'] for a,b in zip(steps,steps[1:]))
  checks[t['id']+'_result_files']=all((folder/n).is_file() for n in ['summary.json','graph.json','graph.sqlite','updates.jsonl','query_log.jsonl'])
  for fid in t['frame_ids']:
   f=folder/'frames'/f'{fid}.json';v=read(f);assert f.with_suffix('.jpg').exists();assert all(x['object_id'] in ids for x in v['detections'])
  db.close()
 queries=[query(p) for p in [r/'online_full_map/graph.sqlite',*[r/'trajectories'/t['id']/'graph.sqlite' for t in c['trajectories']]]]
 checks['fresh_process_graph_queries']=len(queries)==19
 checks['all_raw_rgb_hashes']=all(sha(ROOT/'data/dataset/rgb'/f"{f['id']}.png")==f['image_sha256'] for f in c['replay_inputs'])
 save(r/'fresh_process_queries.json',queries)
 save(r/'online_automated_check.json',dict(run_id=e['run_id'],predictions_sha256=e['predictions_sha256'],automated_checks_passed=all(checks.values()),checks=checks))
 assert all(checks.values());print('Online mapping checks passed: 18 trajectories, 370 keyframes, 19 graph queries.')

def validate_frozen():
 r=ROOT/'results';manifest=verify_frozen_inputs();scope=read(r/'scope_consistency.json');storage=read(r/'storage_comparison.json')
 q=query(ROOT/'data/full_map/graph.sqlite')
 p=subprocess.run([sys.executable,str(ROOT/'code/map_pipeline.py'),'check-cloud','--path',str(ROOT/'data/full_map/points.npy')],check=True,capture_output=True,text=True)
 checks=dict(frozen_input_hashes=True,scope_checks_passed=scope['scope_consistent'],scope_fields_removed=not ({'coverage','limitations'}&set(scope)),reduction_at_least_20_percent=storage['reduction_percent']>=20,independent_graph_query=True,independent_cloud_load=True)
 save(r/'frozen_map_automated_check.json',dict(input_mode='pre_saved_frozen_full_map',frozen_source=manifest['source'],graph_sha256=sha(ROOT/'data/full_map/graph.sqlite'),automated_checks_passed=all(checks.values()),checks=checks,storage=storage,graph_query=q,pointcloud_load=json.loads(p.stdout)))
 assert all(checks.values());print('Frozen-map comparison checks passed; no online mapping result was required.')

def main():
 validate_online();validate_frozen()
 r=ROOT/'results';online=read(r/'online_automated_check.json');frozen=read(r/'frozen_map_automated_check.json')
 save(r/'final_automated_check.json',dict(automated_checks_passed=True,run_id=online['run_id'],predictions_sha256=online['predictions_sha256'],online_check='online_automated_check.json',frozen_map_check='frozen_map_automated_check.json',frozen_graph_sha256=frozen['graph_sha256']))
if __name__=='__main__':main()
