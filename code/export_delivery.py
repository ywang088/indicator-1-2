"""Export existing measured results without upgrading their acceptance status."""
import argparse,csv,hashlib,json,shutil,sqlite3,subprocess,sys
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 b=Path(__file__).resolve().parent
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,default=b/'incremental_20261124/final_verified/results');p.add_argument('--viewer',type=Path,default=b/'rgb_boxes_graph_local_20261124/index.html');p.add_argument('--output',type=Path,required=True);p.add_argument('--input-stage',type=Path);a=p.parse_args()
 result=a.result.resolve();out=a.output.resolve()
 if out.exists():raise ValueError('Use a new output directory')
 report=json.load(open(result/'report.json'))
 for side in ['baseline','cognitive']:
  for record in report[side]:
   f=result/record['path']
   if f.stat().st_size!=record['bytes'] or sha(f)!=record['sha256']:raise ValueError('Recorded map inventory no longer matches '+str(f))
 # Actually reload both runtime formats in separate processes before copying.
 subprocess.run([sys.executable,str(b/'map_pipeline.py'),'query','--db',str(result/'runtime/map.sqlite')],check=True,capture_output=True)
 subprocess.run([sys.executable,str(b/'map_pipeline.py'),'check-cloud','--path',str(result/'baseline/points.npy')],check=True,capture_output=True)
 out.mkdir(parents=True);shutil.copytree(result,out/'results');(out/'visualization').mkdir();shutil.copy2(a.viewer,out/'visualization/index.html')
 (out/'exports').mkdir();db=sqlite3.connect(result/'runtime/map.sqlite');defaults=json.loads(db.execute("select value from metadata where key='object_defaults'").fetchone()[0]);counts={}
 for name,query,keys in [('nodes','select id,kind,attributes from nodes',['id','kind','attributes']),('edges','select source,target,kind,attributes from edges',['source','target','kind','attributes'])]:
  n=0
  with (out/'exports'/f'{name}.jsonl').open('w') as f:
   for row in db.execute(query):
    item=dict(zip(keys,row));item['attributes']=json.loads(item['attributes']);f.write(json.dumps(item,ensure_ascii=False)+'\n');n+=1
  counts[name]=n
 with (out/'exports/objects.jsonl').open('w') as f:
  for oid,attrs in db.execute("select id,attributes from nodes where kind='object'"):
   observations=[json.loads(v) for v, in db.execute("select attributes from edges where target=? and kind='observes'",(oid,))]
   f.write(json.dumps({**defaults,**json.loads(attrs),'id':oid,'observations':observations},ensure_ascii=False)+'\n')
 with (out/'exports/storage_comparison.csv').open('w') as f:
  writer=csv.writer(f);writer.writerow(['side','path','purpose','bytes','sha256'])
  for side in ['baseline','cognitive']:
   for x in report[side]:writer.writerow([side,x['path'],x['purpose'],x['bytes'],x['sha256']])
 (out/'code').mkdir()
 for f in b.iterdir():
  if f.is_file() and (f.suffix in ['.py','.sh','.swift','.sbatch','.js'] or f.name.endswith('.template.html') or f.name=='rgb_boxes_graph_wrapper.html'):shutil.copy2(f,out/'code'/f.name)
 (out/'evidence').mkdir()
 for source in [b/'incremental_20261124/tests.txt',b/'incremental_20261124/verification.json',b/'incremental_20261124/label_review.json',b/'incremental_20261124/crop_jobs.json',b/'incremental_20261124/ocr_crops.jsonl',b/'incremental_20261124/final_verified/inputs/provenance.json',b/'door_audit_20261124/review.json',b/'rgb_boxes_graph_local_20261124/verification.json']:
  target=out/'evidence'/source.relative_to(b);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 docs=[b/'DELIVERY_CHECKLIST_1_2.md',b/'RESULTS_INCREMENTAL_20261124.md',b/'original_outline_extracted.txt']
 if a.input_stage:
  # Keep parent evidence identified as parent; seal and metrics belong to this run.
  docs=[b/'original_outline_extracted.txt']
  for source in a.input_stage.iterdir():
   if source.is_file() and source.suffix=='.json':
    target=out/'evidence/current_seal'/source.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
  for name in ['verification.json','tests.txt','SEAL_README.txt','EXTENSION_RECORD.txt','execution.json','color_validation.json']:
   source=a.input_stage.parent/name
   if source.exists():shutil.copy2(source,out/'evidence/current_seal'/name)
  source=a.viewer.parent/'verification.json'
  if source.exists():shutil.copy2(source,out/'evidence/current_seal/browser_verification.json')
 for f in docs:shutil.copy2(f,out/f.name)
 status=dict(reduction_percent=report['reduction_percent'],counts=counts,exported_tables_are_audit_copies_not_extra_runtime_dependencies=True,raw_data_and_weights_included=False)
 (out/'delivery_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
 (out/'README.txt').write_text('指标 1.2 当前阶段交付包（待验收）\n\nvisualization/index.html：离线 RGB、检测框、局部/完整 graph。\nresults/runtime：可独立加载的认知地图与范围元数据。\nresults/baseline：同范围点云及范围元数据。\nresults/audit：18 条轨迹、每步输入/更新、独立回放图和加载查询日志。\nexports：便于审阅的物体、节点、边与存储清单；它们是从运行数据库导出的重复审计数据，不用于增加或减少运行地图字节。\nevidence：辅助文字/门审核、测试和来源记录。\ncode：源代码；不是自带依赖、原始数据和权重的可移植训练环境。\n\n独立加载示例（安装 numpy、Pillow 等原项目依赖后）：\npython code/map_pipeline.py query --db results/runtime/map.sqlite\npython code/map_pipeline.py check-cloud --path results/baseline/points.npy\n\n原主机复跑：在原 indicator_1_2 工作目录执行 ./run_incremental.sh 新输出目录，再执行 build_rgb_boxes_graph.py --result 新输出目录/results --skip-inline --output 新可视化目录。原始缓存、RGB、几何及文字证据需保留。包内 code/run_incremental.sh 的默认数据路径不代表这些数据已随包提供。\n\n最终是否合格以 DELIVERY_CHECKLIST_1_2.md、delivery_status.json 和原文档要求为准；当前不能仅凭存储降低率作验收合格结论。\n')
 if a.input_stage:
  (out/'README.txt').write_text('指标 1.2 · 封存 18 条轨迹\n\nresults/runtime/map.sqlite：认知地图数据库。\nresults/baseline：同一封存范围的点云基线。\nresults/audit/replays：18 次从空地图开始的独立回放及逐帧日志。\nvisualization/index.html：RGB、目标框、四层 graph。\nexports：物体、节点、连接、存储字节清单。\nevidence/current_seal：本次封存清单、配置、来源及文件校验值。其他 evidence 子目录是沿用的上游证据。\n\n在原主机 indicator_1_2 目录运行：\nbash run_sealed_18.sh 新输出目录 '+str(a.input_stage.resolve())+'\n\n输入 RGB、SAM3 检测和几何均绑定原数据；新增推理与缓存复用的范围见 provenance.json。原始 RGB、模型权重和完整缓存不包含在此包内。逐轨迹清空地图，先冻结范围再重算图和同范围点云；当前版本的裁剪排序依据此前图像与结果，不追溯声称旧测试前预先指定。\n')
 files=[]
 for f in sorted(out.rglob('*')):
  if f.is_file():files.append(dict(path=str(f.relative_to(out)),bytes=f.stat().st_size,sha256=sha(f)))
 (out/'package_manifest.json').write_text(json.dumps(files,ensure_ascii=False,indent=2));archive=shutil.make_archive(str(out),'zip',root_dir=out)
 print(json.dumps(dict(output=str(out),archive=archive,files=len(files),counts=counts),ensure_ascii=False))
if __name__=='__main__':main()
