"""Freeze the revised 18 source intervals before replay; reuse hash-bound perception."""
import argparse, copy, datetime, json
from pathlib import Path
import numpy as np
from perception import dump, sha, zone_text
from geometry_io import geometry_scope_sha256


def main():
    b = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = b / 'incremental_20261124/final_verified/inputs'
    c = json.loads((source / 'config.json').read_text())
    rows = [json.loads(s) for s in (source / 'detections.jsonl').read_text().splitlines()]
    ds = {d['id']: (r, d) for r in rows for d in r['detections']}
    readings = []
    # Read from original pillar crops, not inferred from pillar colour.
    for did in ['004005:000', '005724:002', '006260:000']:
        r, d = ds[did]
        assert d['label'] == 'column'
        e = dict(frame_id=r['frame_id'], detection_id=did,
                 image_sha256=r['image_sha256'], bbox=d['bbox'], text='E',
                 review_status='assistant_visual_transcription_not_human_acceptance',
                 source='assistant_current_frame_pillar_reading')
        d.setdefault('texts', []).append(dict(e, score=None, parsed=zone_text('E')))
        readings.append(e)
    starts = [119, 550, 849, 1269, 1701, 2042, 2504, 3527, 4005,
              4641, 5139, 5724, 6260, 6931, 7299, 7869, 8676, 9350]
    order = [8] + [i for i in range(18) if i != 8]
    old = copy.deepcopy(c['trajectories'])
    tracks, edits = [], []
    for j, i in enumerate(order, 1):
        t = copy.deepcopy(old[i])
        t.update(id=f'T{j:02}', original_id=old[i]['id'], source_start=starts[i])
        t['frame_ids'] = [f for f in t['frame_ids'] if int(f) >= starts[i]]
        assert t['frame_ids'] and int(t['frame_ids'][0]) == starts[i]
        t['source_frame_count'] = sum(starts[i] <= int(f['id']) <= t['source_end'] for f in c['replay_inputs'])
        tracks.append(t)
        edits.append(dict(id=t['id'], original_id=t['original_id'],
                          original_interval=[old[i]['source_start'], old[i]['source_end']],
                          sealed_interval=[t['source_start'], t['source_end']],
                          trimmed_leading_keyframes=len(old[i]['frame_ids'])-len(t['frame_ids']),
                          added_keyframes=[], keyframes=len(t['frame_ids'])))
    ids = {fid for t in tracks for fid in t['frame_ids']}
    c['trajectories'] = tracks
    c['frames'] = [f for f in c['frames'] if f['id'] in ids]
    for f in c['frames']:
        assert sha(Path(f['image'])) == f['image_sha256']
    c['replay_inputs'] = [f for f in c['replay_inputs'] if any(t['source_start'] <= int(f['id']) <= t['source_end'] for t in tracks)]
    rows = [r for r in rows if r['frame_id'] in ids]
    assert {r['frame_id'] for r in rows} == ids
    c['selection'] = '18 trajectories revised at user request after prior results; intervals frozen before this replay; chronological frames retained within each trajectory'
    c['sealed_version'] = '18-v2-r02-20261124'
    c['parent_config_sha256'] = sha(source / 'config.json')
    geometry = b / 'matched_geometry_verified'
    audit = json.loads((geometry / 'geometry_audit.json').read_text())
    original_geometry = json.loads((b/'frozen/config.json').read_text())
    assert geometry_scope_sha256(original_geometry) == audit['geometry_scope_sha256']
    for key in ['scene', 'K', 'image_size', 'bounds_xyz']:
        assert original_geometry[key] == c[key], key
    original_frames = {f['id']:f for f in original_geometry['frames']}
    for f in c['frames']:
        for key in ['id', 'timestamp', 'T_world_camera']:
            assert original_frames[f['id']][key] == f[key], (f['id'],key)
    # The cache is already in scene coordinates; do not apply the raw LiDAR
    # transform a second time. Verify against the previously used world cloud.
    assert np.array_equal(np.load(geometry/'points.npy', allow_pickle=False),
                          np.load(b/'refined_final/matched_points.npy', allow_pickle=False))
    cache = np.load(geometry / 'frame_points.npz', allow_pickle=False)
    xyz = np.concatenate([cache['f'+f['id']] for f in c['frames']])
    _, ix = np.unique(np.floor(xyz/c['pointcloud']['voxel_m']).astype('i4'), axis=0, return_index=True)
    np.save(out / 'points.npy', xyz[np.sort(ix)], allow_pickle=False)
    c['pointcloud'].update(path=str(out/'points.npy'), sha256=sha(out/'points.npy'),
                           source=f'same {len(ids)} selected RGB frustums, synchronized Velodyne',
                           scope_evidence=str(out/'scope_evidence.json'))
    audit.update(geometry_scope_sha256=geometry_scope_sha256(c),
                 parent_geometry_audit_sha256=sha(geometry/'geometry_audit.json'),
                 selected_frames=len(ids), scans=[v for v in audit['scans'] if v['frame_id'] in ids],
                 points=len(ix), config_parent_sha256=c['parent_config_sha256'])
    dump(out/'geometry_audit.json', audit)
    dump(out/'scope_evidence.json', dict(scene=c['scene'], source_pointcloud_sha256=c['pointcloud']['sha256'],
        bounds_xyz=c['bounds_xyz'], trajectory_ids=[t['id'] for t in tracks], frame_ids=sorted(ids),
        reviewer='deterministic subset of synchronized original geometry; not human acceptance',
        same_time_interval=True, same_effective_coverage=True, registration_verified=True,
        geometry_audit_sha256=sha(out/'geometry_audit.json'),
        limitation='Original estimated camera calibration and source odometry retained.'))
    dump(out/'additional_readings.json', readings)
    dump(out/'selection_manifest.json', dict(version=c['sealed_version'],
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parent_config_sha256=c['parent_config_sha256'], trajectories=edits,
        selection_basis='User requested first trajectory with two regions and connection, other starts with readable region pillars; chosen using existing images/results.',
        chronology='No frame permutation inside a trajectory; no synthetic or duplicated added frames.',
        perception='Existing SAM3 detections reused with identical RGB hashes; three new assistant pillar transcriptions recorded separately.'))
    c['selection_manifest_sha256'] = sha(out/'selection_manifest.json')
    dump(out/'config.json', c)
    (out/'detections.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    for name in ['images', 'masks']:
        (out/name).symlink_to((source/name).resolve(), target_is_directory=True)
    prov = json.loads((source/'provenance.json').read_text())
    prov.update(config_sha256=sha(out/'config.json'), frames=len(ids),
        parent_predictions_sha256=sha(source/'detections.jsonl'),
        parent_provenance_sha256=sha(source/'provenance.json'),
        additional_readings_sha256=sha(out/'additional_readings.json'),
        synchronized_geometry_sha256=sha(out/'geometry_audit.json'),
        perception_execution='reuse unchanged-image SAM3 cache; no new GPU inference',
        selection_manifest_sha256=c['selection_manifest_sha256'])
    dump(out/'provenance.json', prov)
    dump(out/'frozen_files.json', {f.name:sha(f) for f in out.iterdir() if f.is_file()})
    print(json.dumps(dict(output=str(out), trajectories=len(tracks), keyframes=len(ids), raw_frames=len(c['replay_inputs']))))


if __name__ == '__main__':
    main()
