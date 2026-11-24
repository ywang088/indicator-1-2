"""Human corrections without rerunning GPU; originals are preserved."""
import argparse,json,shutil
from pathlib import Path
from perception import sha,dump,zone_text
p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--review',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[json.loads(x) for x in a.predictions.read_text().splitlines()]
if not a.review:
 dump(a.output,dict(predictions_sha256=sha(a.predictions),reviewer='',decisions=[dict(detection_id=d['id'],frame_id=r['frame_id'],label=d['label'],accept=None,column_text=None,note='') for r in rows for d in r['detections']]));raise SystemExit(0)
rv=json.loads(a.review.read_text())
if not rv.get('reviewer') or rv['predictions_sha256']!=sha(a.predictions):raise ValueError('review must identify reviewer and exact predictions')
ids={d['id'] for r in rows for d in r['detections']};dec={d['detection_id']:d for d in rv['decisions']}
if len(dec)!=len(rv['decisions']) or not set(dec)<=ids:raise ValueError('duplicate or unknown reviewed detection')
a.output.mkdir(parents=True,exist_ok=False)
for r in rows:
 kept=[]
 for d in r['detections']:
  v=dec.get(d['id'])
  if v:
   if v['frame_id']!=r['frame_id']:raise ValueError('review frame mismatch')
   if v.get('accept') is False:continue
   if v.get('accept') is True:
    d['human_review']=dict(reviewer=rv['reviewer'],note=v.get('note',''))
    if v.get('column_text'):
     if d['label']!='column':raise ValueError('column text may only be assigned to detected columns')
     parsed=zone_text(v['column_text'])
     if not parsed:raise ValueError('unrecognized column text grammar')
     d['texts']=[dict(text=v['column_text'],score=1.,parsed=parsed,source='human_transcription',reviewer=rv['reviewer'])]
  kept.append(d)
 r['detections']=kept
(a.output/'detections.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
prov=json.loads((a.predictions.parent/'provenance.json').read_text());prov.update(parent_predictions_sha256=sha(a.predictions),human_corrections_sha256=sha(a.review));dump(a.output/'provenance.json',prov)
for name in ['images','masks']:
 if (a.predictions.parent/name).exists():shutil.copytree(a.predictions.parent/name,a.output/name)
print(a.output/'detections.jsonl')
