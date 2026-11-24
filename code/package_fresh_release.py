"""Supplement fixed v1 data with fresh-inference models and outputs."""
import argparse,gzip,hashlib,json,tarfile,subprocess
from pathlib import Path
from package_release import Parts,digest,ROOT
class ModelParts(Parts):
 def write(self,b):
  total=len(b)
  while b:
   if self.f is None:
    p=self.out/f'models.tar.gz.part{self.i:04d}';self.f=p.open('wb');self.paths.append(p);self.i+=1;self.size=0
   n=min(len(b),64*1024*1024-self.size);self.f.write(b[:n]);b=b[n:];self.size+=n
   if self.size==64*1024*1024:self.f.close();self.f=None
  return total
def main():
 p=argparse.ArgumentParser();p.add_argument('--models-only',action='store_true');a=p.parse_args();out=ROOT/'release_fresh_assets';out.mkdir(exist_ok=True)
 if not list(out.glob('models.tar.gz.part*')):
  parts=ModelParts(out)
  with gzip.GzipFile(fileobj=parts,mode='wb',compresslevel=1,mtime=0) as gz:
   with tarfile.open(fileobj=gz,mode='w|') as tar:tar.add(ROOT/'data/models',arcname='data/models')
  parts.close()
 if not a.models_only:
  execution=json.loads((ROOT/'results/execution.json').read_text())
  checks=json.loads((ROOT/'results/final_automated_check.json').read_text())
  assert execution.get('pipeline_mode')=='fresh_sam3' and execution.get('completed') is True
  assert checks.get('run_id')==execution.get('run_id') and checks.get('predictions_sha256')==execution.get('predictions_sha256'), 'Validation must match the current run'
  assert checks.get('automated_checks_passed') is True, 'Fresh output validation must pass before packaging'
  with tarfile.open(out/'fresh-results.tar.gz','w:gz',compresslevel=1,dereference=True) as tar:
   tar.add(ROOT/'results',arcname='results')
  with tarfile.open(out/'frozen-map-metadata.tar.gz','w:gz',compresslevel=1) as tar:
   for name in ['audit','frozen_manifest.json']:tar.add(ROOT/'data/full_map'/name,arcname='data/full_map/'+name)
  subprocess.run(['git','archive','--format=zip','--prefix=indicator_docker/','--output='+str(out/'fresh-code.zip'),'HEAD'],cwd=ROOT,check=True)
 paths=sorted(p for p in out.iterdir() if p.name not in ['SHA256SUMS','release_manifest.json']);v=[dict(file=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in paths];(out/'release_manifest.json').write_text(json.dumps(v,indent=2));(out/'SHA256SUMS').write_text(''.join(f"{x['sha256']}  {x['file']}\n" for x in v));print('Packaged',len(v),'files')
if __name__=='__main__':main()
