"""Default entrypoint: fresh GPU perception, then independent 18-track mapping."""
import argparse,datetime,hashlib,json,os,subprocess,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def prepare_config(root,checkpoint,bpe):
 data=root/'data';c=json.loads((data/'dataset/config.json').read_text())
 for group in ['frames','replay_inputs']:
  for f in c[group]:
   image=data/'dataset/rgb'/f"{f['id']}.png"
   if not image.is_file() or sha(image)!=f['image_sha256']:raise ValueError('Missing or changed original RGB: '+f['id'])
   f['image']=str(image.resolve())
 c['checkpoint']=str(checkpoint.resolve());c['bpe']=str(bpe.resolve());c['pointcloud']['path']=str((data/'full_map/points.npy').resolve());c['pointcloud']['sha256']=sha(data/'full_map/points.npy');c['pointcloud'].pop('scope_evidence',None)
 return c
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke-frames',type=int,default=0,help='GPU preflight only; does not overwrite final 18-track results');a=p.parse_args()
 if a.smoke_frames<0:raise ValueError('smoke-frames must be nonnegative')
 if not a.smoke_frames:save(ROOT/'results/execution.json',dict(pipeline_mode='fresh_sam3',completed=False,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),reuses_detection_cache=False))
 checkpoint=Path(os.environ.get('SAM3_CHECKPOINT',ROOT/'data/models/sam3.pt'));bpe=Path(os.environ.get('SAM3_BPE',ROOT/'data/models/bpe_simple_vocab_16e6.txt.gz'))
 for f in [checkpoint,bpe]:
  if not f.is_file():raise FileNotFoundError(f'Model file missing: {f}; no cached-perception fallback is allowed')
 os.environ.setdefault('EASYOCR_MODEL_DIR',str(ROOT/'data/models/ocr'))
 for name in ['craft_mlt_25k.pth','zh_sim_g2.pth']:
  if not (Path(os.environ['EASYOCR_MODEL_DIR'])/name).is_file():raise FileNotFoundError('OCR weight missing: '+name)
 import torch
 if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for fresh SAM3 inference; no cached-perception fallback is allowed')
 print('Verifying fixed RGB inputs and preparing fresh inference...',flush=True)
 c=prepare_config(ROOT,checkpoint,bpe)
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6];run=ROOT/'results/runs'/stamp;run.mkdir(parents=True)
 save(run/'input_config.json',c)
 evidence=ROOT/'data/full_map/scope_evidence';geometry=run/'geometry';geometry.mkdir()
 for name in ['frame_points.npz','geometry_audit.json']:(geometry/name).symlink_to((evidence/name).resolve())
 (geometry/'points.npy').symlink_to((ROOT/'data/full_map/points.npy').resolve())
 audit=dict(run_id=stamp,pipeline_mode='fresh_sam3',completed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,input_frames=len(c['frames']),raw_rgb_count=len(c['replay_inputs']),checkpoint_sha256=sha(checkpoint),bpe_sha256=sha(bpe),reuses_detection_cache=False,reuses_ocr_cache=False,geometry_input='fixed synchronized LiDAR XYZ with fixed calibration and poses',smoke_frames=a.smoke_frames)
 save(run/'execution.json',audit)
 def call(script,*args):
  command=[sys.executable,str(ROOT/'code'/script),*map(str,args)];print('Running',script,flush=True)
  with (run/(script+'.log')).open('w') as log:
   proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
   for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
   if proc.wait():raise RuntimeError(f'{script} failed; see {log.name}')
 args=['--config',run/'input_config.json','--output',run/'perception','--device','cuda','--ocr','easyocr','--geometry',geometry]
 if a.smoke_frames:args+=['--limit',a.smoke_frames]
 call('perception.py',*args)
 if a.smoke_frames:
  audit.update(completed=True,processed_frames=min(a.smoke_frames,len(c['frames'])));save(run/'execution.json',audit);print('GPU smoke run complete:',run);return
 call('refine.py','--config',run/'input_config.json','--predictions',run/'perception/detections.jsonl','--geometry',geometry,'--output',run/'refined')
 call('prepare_color_inputs.py','--source',run/'refined','--output',run/'colored')
 call('map_pipeline.py','build','--config',run/'colored/config.json','--predictions',run/'colored/detections.jsonl','--output',run/'map')
 from delivery_runner import export_run
 export_run(run/'map')
 audit.update(completed=True,processed_frames=len(c['frames']),predictions_sha256=sha(run/'colored/detections.jsonl'),provenance=str((run/'colored/provenance.json').relative_to(ROOT)))
 save(run/'execution.json',audit);save(ROOT/'results/execution.json',audit)
 save(ROOT/'results/data_integrity.json',dict(raw_rgb_count=len(c['replay_inputs']),keyframe_count=len(c['frames']),all_source_image_hashes_verified=True))
 from final_delivery_check import validate_online
 try:validate_online()
 except Exception:
  audit['completed']=False;save(ROOT/'results/execution.json',audit);save(run/'execution.json',audit);raise
 print('Fresh SAM3 + OCR + geometry + 18-track mapping completed:',run,flush=True)
if __name__=='__main__':main()
