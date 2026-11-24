"""Export a model-bundled Docker image as checksummed release parts."""
import argparse,gzip,json,subprocess
from pathlib import Path
from package_release import ROOT,Parts,digest
class ImageParts(Parts):
 def write(self,b):
  total=len(b)
  while b:
   if self.f is None:
    p=self.out/f'parking-gpu-with-models.tar.gz.part{self.i:04d}';self.f=p.open('wb');self.paths.append(p);self.i+=1;self.size=0
   n=min(len(b),64*1024*1024-self.size);self.f.write(b[:n]);b=b[n:];self.size+=n
   if self.size==64*1024*1024:self.f.close();self.f=None
  return total
def main():
 p=argparse.ArgumentParser();p.add_argument('--image',default='parking-indicator-1-2:bundled-gpu');p.add_argument('--output',type=Path,default=ROOT/'docker_image_assets');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 if list(a.output.glob('parking-gpu-with-models.tar.gz.part*')):raise SystemExit('Choose an empty output directory for a new image export.')
 metadata=json.loads(subprocess.check_output(['docker','image','inspect',a.image],text=True))[0]
 verification=json.loads(subprocess.check_output(['docker','run','--rm','--platform','linux/amd64','--network','none','--mount','type=tmpfs,destination=/workspace/data',a.image,'python3','code/verify_bundled_models.py'],text=True))
 parts=ImageParts(a.output);proc=subprocess.Popen(['docker','save',a.image],stdout=subprocess.PIPE)
 try:
  with gzip.GzipFile(fileobj=parts,mode='wb',compresslevel=1,mtime=0) as gz:
   for b in iter(lambda:proc.stdout.read(4*1024*1024),b''):gz.write(b)
  if proc.wait():raise RuntimeError('docker save failed')
 finally:
  parts.close()
  if proc.poll() is None:proc.terminate();proc.wait()
 info=dict(image=a.image,image_id=metadata['Id'],architecture=metadata['Architecture'],os=metadata['Os'],uncompressed_image_bytes=metadata['Size'],models_path='/opt/parking/models',model_verification=verification,model_sha256=json.loads((ROOT/'data/models/manifest.json').read_text()),load_command='cat parking-gpu-with-models.tar.gz.part* | gzip -dc | docker load')
 meta=a.output/'image_manifest.json';meta.write_text(json.dumps(info,indent=2))
 rows=[dict(file=f.name,bytes=f.stat().st_size,sha256=digest(f)) for f in parts.paths+[meta]]
 (a.output/'release_manifest.json').write_text(json.dumps(rows,indent=2));(a.output/'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['file']}\n" for r in rows));print(json.dumps(dict(parts=len(parts.paths),total_bytes=sum(r['bytes'] for r in rows),image_id=metadata['Id'])))
if __name__=='__main__':main()
