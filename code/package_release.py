"""Create private-release data parts without adding large datasets to Git history."""
import gzip,hashlib,json,tarfile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Parts:
 def __init__(self,out):self.out=out;self.i=0;self.size=0;self.f=None;self.paths=[]
 def write(self,b):
  total=len(b)
  while b:
   if self.f is None:
    p=self.out/f'data.tar.gz.part{self.i:04d}';self.f=p.open('wb');self.paths.append(p);self.i+=1;self.size=0
   n=min(len(b),64*1024*1024-self.size);self.f.write(b[:n]);b=b[n:];self.size+=n
   if self.size==64*1024*1024:self.f.close();self.f=None
  return total
 def flush(self):
  if self.f:self.f.flush()
 def close(self):
  if self.f:self.f.close()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 out=ROOT/'release_assets';out.mkdir(exist_ok=True)
 if list(out.glob('data.tar.gz.part*')):raise SystemExit('Existing data parts: use a new release directory or move them before rebuilding.')
 parts=Parts(out)
 with gzip.GzipFile(fileobj=parts,mode='wb',compresslevel=1,mtime=0) as gz:
  with tarfile.open(fileobj=gz,mode='w|') as tar:tar.add(ROOT/'data',arcname='data')
 parts.close()
 with tarfile.open(out/'results.tar.gz','w:gz',compresslevel=1) as tar:tar.add(ROOT/'results',arcname='results')
 subprocess.run(['git','archive','--format=zip','--prefix=indicator_docker/','--output='+str(out/'docker-code.zip'),'HEAD'],cwd=ROOT,check=True)
 paths=parts.paths+[out/'results.tar.gz',out/'docker-code.zip'];manifest=[dict(file=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in paths]
 (out/'SHA256SUMS').write_text(''.join(f"{x['sha256']}  {x['file']}\n" for x in manifest));(out/'release_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
