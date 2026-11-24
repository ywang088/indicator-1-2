"""Check model files and configured paths inside the delivery image."""
import os,json,hashlib
from pathlib import Path
def main():
 root=Path('/opt/indicator/models');expected=json.loads((root/'manifest.json').read_text());verified={}
 for key,name in {'SAM3_CHECKPOINT':'sam3.pt','SAM3_BPE':'bpe_simple_vocab_16e6.txt.gz','EASYOCR_MODEL_DIR':'ocr'}.items():
  if Path(os.environ[key]).resolve()!=root/name:raise ValueError('Unexpected model path: '+key)
 for name,digest in expected.items():
  h=hashlib.sha256()
  with (root/name).open('rb') as f:
   for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
  if h.hexdigest()!=digest:raise ValueError('Model checksum mismatch: '+name)
  verified[name]=digest
 print(json.dumps(dict(passed=True,model_directory=str(root),files=verified)))
if __name__=='__main__':main()
