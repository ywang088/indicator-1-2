"""Apply hash-bound visual connection review to cached predictions; no GPU required."""
import argparse,json,shutil,hashlib
from pathlib import Path
from portal_connections import validate_reviewed_connection

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 c=json.loads((a.source/'config.json').read_text());f={x['id']:x for x in c['frames']};review=json.loads(a.review.read_text())
 for portal in review['portals']:validate_reviewed_connection(portal,c,f)
 if a.output.exists():raise ValueError('use a new output directory')
 a.output.mkdir(parents=True);c['portals']=review['portals'];c['parent_config_sha256']=sha(a.source/'config.json')
 (a.output/'config.json').write_text(json.dumps(c,ensure_ascii=False,indent=2))
 shutil.copy2(a.source/'detections.jsonl',a.output/'detections.jsonl')
 for name in ['images','masks']:(a.output/name).symlink_to((a.source/name).resolve(),target_is_directory=True)
 prov=json.loads((a.source/'provenance.json').read_text());prov.update(parent_config_sha256=c['parent_config_sha256'],config_sha256=sha(a.output/'config.json'),connection_review_sha256=sha(a.review),connection_review=str(a.review.resolve()),human_review_completed=False)
 (a.output/'provenance.json').write_text(json.dumps(prov,ensure_ascii=False,indent=2))
 print(json.dumps({'portals':len(c['portals']),'output':str(a.output)}))
if __name__=='__main__':main()
