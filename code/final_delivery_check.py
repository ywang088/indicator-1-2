"""Repeatable artifact checks; human acceptance is not inferred from these checks."""
import json,sys,subprocess,sqlite3,hashlib,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 r=ROOT/'results';c=json.loads((r/'run_config.json').read_text());checks={};queries=[]
 required=['report.json','run_config.json','data_integrity.json','storage_comparison.json','storage_comparison.csv','storage_report.json','scope_consistency.json','scope_consistency.png','scene_review/scene_examples.json','范围对照图说明.txt','指标1.2文档核对.txt']
 checks['required_outputs_present']=all((r/name).is_file() for name in required)
 scope=json.loads((r/'scope_consistency.json').read_text());checks['scope_checks_passed']=scope['scope_consistent']
 checks['requested_scope_fields_removed']=not ({'coverage','limitations'} & set(scope))
 for t in c['trajectories']:
  folder=r/'trajectories'/t['id'];db=sqlite3.connect(folder/'graph.sqlite');ids={i for i, in db.execute('select id from nodes')};edges=db.execute('select source,target from edges').fetchall()
  checks[t['id']+'_references']=all(s in ids and v in ids for s,v in edges)
  checks[t['id']+'_frames']=set(i for i in ids if i.startswith('frame:'))=={'frame:'+i for i in t['frame_ids']}
  steps=[json.loads(s) for s in (folder/'all_steps.jsonl').read_text().splitlines()];checks[t['id']+'_input_log_complete']=len(steps)==sum(t['source_start']<=int(f['id'])<=t['source_end'] for f in c['replay_inputs'])
  checks[t['id']+'_input_log_order']=all(a['timestamp']<=b['timestamp'] for a,b in zip(steps,steps[1:]))
  checks[t['id']+'_result_files']=all((folder/n).is_file() for n in ['summary.json','graph.json','graph.sqlite','updates.jsonl','query_log.jsonl'])
  for fid in t['frame_ids']:
   f=folder/'frames'/f'{fid}.json';v=json.loads(f.read_text());assert (f.with_suffix('.jpg')).exists();assert all(x['object_id'] in ids for x in v['detections'])
  db.close()
 paths=[ROOT/'data/full_map/graph.sqlite']+[r/'trajectories'/t['id']/'graph.sqlite' for t in c['trajectories']]
 for p in paths:
  result=subprocess.run([sys.executable,str(ROOT/'code/map_pipeline.py'),'query','--db',str(p)],check=True,capture_output=True,text=True)
  queries.append(dict(database=str(p.relative_to(ROOT)),result=json.loads(result.stdout)))
 result=subprocess.run([sys.executable,'-c',"import numpy as np,json,sys;p=np.load(sys.argv[1],allow_pickle=False);print(json.dumps({'points':len(p),'shape':list(p.shape),'finite':bool(np.isfinite(p).all())}))",str(ROOT/'data/full_map/points.npy')],check=True,capture_output=True,text=True)
 checks['fresh_process_graph_queries']=len(queries)==19;checks['fresh_process_cloud_load']=json.loads(result.stdout)['finite'];checks['18_trajectories']=len(c['trajectories'])==18
 checks['all_raw_rgb_hashes']=all(hashlib.sha256((ROOT/'data/dataset/rgb'/f"{f['id']}.png").read_bytes()).hexdigest()==f['image_sha256'] for f in c['replay_inputs'])
 storage=json.loads((r/'storage_comparison.json').read_text());checks['reduction_at_least_20_percent']=storage['reduction_percent']>=20
 (r/'fresh_process_queries.json').write_text(json.dumps(queries,ensure_ascii=False,indent=2))
 (r/'final_automated_check.json').write_text(json.dumps(dict(automated_checks_passed=all(checks.values()),checks=checks,pointcloud_load=json.loads(result.stdout),environment=dict(python=sys.version,platform=platform.platform()),storage=storage),ensure_ascii=False,indent=2));assert all(checks.values());print('All artifact checks passed; 19 graph loads/queries and 1 point-cloud load in fresh processes.')
if __name__=='__main__':main()
