"""Download public release files with SHA256 verification; no GitHub login."""
import argparse,fnmatch,hashlib,json,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO='ywang088/indicator-1-2'
API='https://api.github.com/repos/'+REPO

def open_url(url):
 return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'indicator-1-2-downloader','Accept':'application/vnd.github+json'}),timeout=120)
def read_json(url):
 for attempt in range(4):
  try:
   with open_url(url) as f:return json.load(f)
  except Exception:
   if attempt==3:raise
   time.sleep(2**attempt)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def get_assets(tag):
 release=read_json(API+'/releases/tags/'+tag);assets=[];page=1
 while True:
  chunk=read_json(API+f"/releases/{release['id']}/assets?per_page=100&page={page}");assets.extend(chunk)
  if len(chunk)<100:return assets
  page+=1

def download(asset,destination):
 digest=asset.get('digest','')
 if not digest.startswith('sha256:'):raise ValueError('Release asset has no SHA256: '+asset['name'])
 expected=digest.split(':',1)[1];destination.parent.mkdir(parents=True,exist_ok=True)
 if destination.exists() and destination.stat().st_size==asset['size'] and sha(destination)==expected:
  print('Verified existing:',destination.name,flush=True);return
 part=destination.with_name(destination.name+'.partial')
 for attempt in range(4):
  try:
   with open_url(asset['browser_download_url']) as source,part.open('wb') as target:
    for block in iter(lambda:source.read(4*1024*1024),b''):target.write(block)
   if part.stat().st_size!=asset['size'] or sha(part)!=expected:raise ValueError('Checksum mismatch: '+destination.name)
   part.replace(destination);print('Downloaded:',destination.name,flush=True);return
  except Exception:
   if attempt==3:raise
   time.sleep(2**attempt)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('group',choices=['data','image','models','results']);p.add_argument('--output',type=Path);a=p.parse_args()
 groups={
  'data':[('sealed-18-v1',['data.tar.gz.part*','SHA256SUMS'],{'SHA256SUMS':'data-SHA256SUMS'}),('fresh-18-v2',['frozen-map-metadata.tar.gz','SHA256SUMS','release_manifest.json'],{})],
  'image':[('docker-bundled-v1',['gpu-with-models.tar.gz.part*','SHA256SUMS','image_manifest.json'],{})],
  'models':[('fresh-18-v2',['models.tar.gz.part*','SHA256SUMS','release_manifest.json'],{})],
  'results':[('fresh-18-v2',['fresh-results.tar.gz','SHA256SUMS','release_manifest.json'],{})],
 }
 output=a.output or ROOT/('image_assets' if a.group=='image' else 'release_assets')
 for tag,patterns,renames in groups[a.group]:
  assets=get_assets(tag)
  for pattern in patterns:
   if not any(fnmatch.fnmatch(x['name'],pattern) for x in assets):raise ValueError('Missing release files: '+tag+'/'+pattern)
  selected=sorted([x for x in assets if any(fnmatch.fnmatch(x['name'],pattern) for pattern in patterns)],key=lambda x:x['name'])
  for n,asset in enumerate(selected,1):
   print(f'{tag} [{n}/{len(selected)}]',flush=True);download(asset,output/renames.get(asset['name'],asset['name']))
 print('Download and SHA256 verification complete:',output)
if __name__=='__main__':main()
