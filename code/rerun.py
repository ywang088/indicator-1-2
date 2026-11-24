"""Run the verified SAM3 -> synchronized geometry -> graph pipeline without overwriting outputs."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from perception import sha
from geometry_io import geometry_scope_sha256

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--config',type=Path,required=True)
 p.add_argument('--geometry',type=Path,required=True,help='Output of build_matched_geometry.py')
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--predictions',type=Path,help='Reuse completed SAM3 JSONL; omit to perform fresh GPU inference')
 p.add_argument('--rgb-root',type=Path,help='Optional hash-verified local keyframe aliases')
 p.add_argument('--door-review',type=Path,help='Exact-predictions-bound door audit; never reused on different detections')
 p.add_argument('--readings',type=Path,help='Optional crop transcription manifest; never treated as human acceptance')
 a=p.parse_args();a.output=a.output.resolve()
 audit=json.loads((a.geometry/'geometry_audit.json').read_text())
 if audit['config_parent_sha256']!=sha(a.config) and audit.get('geometry_scope_sha256')!=geometry_scope_sha256(json.loads(a.config.read_text())):raise ValueError('Geometry belongs to a different sealed source config')
 a.output.mkdir(parents=True,exist_ok=False)
 script=Path(__file__).resolve().parent
 def run(name,*args):subprocess.run([sys.executable,str(script/name),*map(str,args)],check=True)
 pred=a.predictions
 if pred is None:
  args=['--config',a.config,'--output',a.output/'perception','--device',os.environ.get('DEVICE','cuda')]
  for env,flag in [('SAM3_CHECKPOINT','--checkpoint'),('SAM3_BPE','--bpe')]:
   if os.environ.get(env):args.extend([flag,os.environ[env]])
  run('perception.py',*args);pred=a.output/'perception/detections.jsonl'
 else:
  prov=json.loads((pred.parent/'provenance.json').read_text())
  if not prov.get('completed') or prov.get('config_sha256')!=sha(a.config):raise ValueError('Cached perception is incomplete or uses another config')
 args=['--config',a.config,'--predictions',pred,'--geometry',a.geometry,'--output',a.output/'refined']
 if a.rgb_root:args.extend(['--rgb-root',a.rgb_root])
 if a.readings:args.extend(['--readings',a.readings])
 run('refine.py',*args)
 stage=a.output/'refined'
 if a.door_review:
  run('clean_door_candidates.py','--config',stage/'config.json','--predictions',stage/'detections.jsonl','--review',a.door_review,'--output',a.output/'cleaned')
  stage=a.output/'cleaned'
 run('map_pipeline.py','build','--config',stage/'config.json','--predictions',stage/'detections.jsonl','--output',a.output/'map')
 print('Result:',a.output/'map/index.html')
if __name__=='__main__':main()
