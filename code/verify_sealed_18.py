"""Verify real replay outputs against sealed continuous source intervals."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=json.loads((a.result/'audit/config.json').read_text())
    frames={f['id']:f for f in c['frames']};used=set();checks=[]
    assert len(c['trajectories'])==18
    for t in c['trajectories']:
        ids=t['frame_ids'];assert len(ids)==len(set(ids)) and ids==sorted(ids)
        assert int(ids[0])==t['source_start'] and int(ids[-1])==t['source_end']
        source=set(range(t['source_start'],t['source_end']+1));assert not used&source
        used|=source
        folder=a.result/'audit/replays'/t['id']
        events=[json.loads(s) for s in (folder/'updates.jsonl').read_text().splitlines()]
        steps=[json.loads(s) for s in (folder/'all_steps.jsonl').read_text().splitlines()]
        assert [e['frame_id'] for e in events]==ids
        assert {int(e['frame_id']) for e in steps}==source
        assert all(e['input_sha256'] and e['input_available'] for e in steps)
        assert events[0]['regions_after']>=1,(t['id'],'missing first-frame region')
        duration=frames[ids[-1]]['timestamp']-frames[ids[0]]['timestamp']
        checks.append(dict(id=t['id'],original_id=t.get('original_id'),
            start=ids[0],end=ids[-1],keyframes=len(ids),raw_frames=len(steps),
            duration_s=round(duration,3),first_regions=events[0]['region_ids'],
            final_regions=events[-1]['region_ids'],final_portals=events[-1]['portal_ids']))
    assert len(checks[0]['final_regions'])>=2 and len(checks[0]['final_portals'])>=1
    report=json.loads((a.result/'report.json').read_text())
    assert 'status' not in report and 'evidence_gaps' not in report
    a.output.write_text(json.dumps(dict(trajectories=checks,
        all_first_frames_have_region=True,source_intervals_do_not_overlap=True,
        chronological_contiguous_source_intervals=True,
        total_keyframes=sum(t['keyframes'] for t in checks),
        total_raw_frames=len(used),reduction_percent=report['reduction_percent']),ensure_ascii=False,indent=2))
    print('18 replay scopes and first-frame regions verified')

if __name__=='__main__':main()
