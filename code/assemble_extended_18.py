"""Bind new SAM3 observations and existing audited observations to extended intervals."""
import argparse, json, shutil
from pathlib import Path
import numpy as np
from perception import dump, sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args();stage=a.stage.resolve();base=Path(__file__).resolve().parent
    out=stage/'inputs';out.mkdir(exist_ok=False)
    c=json.loads((stage/'bound_config.json').read_text())
    plan=json.loads((stage/'extension_plan.json').read_text())
    cache=base/'incremental_20261124/final_verified/inputs'
    latest=base/'sealed_18_v2_20261124/revision_02/inputs'
    old={r['frame_id']:r for r in map(json.loads,(cache/'detections.jsonl').read_text().splitlines())}
    old.update({r['frame_id']:r for r in map(json.loads,(latest/'detections.jsonl').read_text().splitlines())})
    fresh={r['frame_id']:r for r in map(json.loads,(stage/'refined/detections.jsonl').read_text().splitlines())}
    assert set(fresh)==set(plan['new_sam_frames'])
    oldgeom=np.load(base/'matched_geometry_verified/frame_points.npz',allow_pickle=False)
    newgeom=np.load(stage/'geometry/frame_points.npz',allow_pickle=False)
    for fid in plan['cached_frames']:
        assert np.array_equal(oldgeom['f'+fid],newgeom['f'+fid]),fid
    (out/'masks').mkdir();(out/'images').mkdir();rows=[]
    for f in c['frames']:
        fid=f['id'];r=fresh[fid] if fid in fresh else old[fid]
        assert r['image_sha256']==f['image_sha256']==sha(Path(f['image']))
        source=stage/'refined' if fid in fresh else cache
        for d in r['detections']:
            if fid in fresh and d['label'] in ('door','elevator'):
                d['semantic_status']='uncertain_door_candidate_unverified_connection'
            shutil.copy2(source/d['mask'],out/d['mask'])
        shutil.copy2(source/'images'/f'{fid}.jpg',out/'images'/f'{fid}.jpg')
        rows.append(r)
    c.update(pointcloud=dict(path=str(stage/'geometry/points.npy'),sha256=sha(stage/'geometry/points.npy'),
        T_world_source=np.eye(4).tolist(),voxel_m=c['pointcloud']['voxel_m'],fields='XYZ float32',
        format='NPY uncompressed',compression='none',source='same 370 synchronized RGB frustums',
        scope_verified=True,scope_evidence=str(out/'scope_evidence.json')),
        incremental_topology=True,portals=[],regions=[],human_review=None)
    dump(out/'selection_manifest.json',dict(version=c['sealed_version'],frozen_at=c['frozen_at'],
        parent_selection_sha256=sha(latest/'selection_manifest.json'),
        selection=c['selection'],sampling='stride 20 plus previously retained endpoints and new endpoints',
        original_source_frames=len(c['replay_inputs']),**plan))
    c['selection_manifest_sha256']=sha(out/'selection_manifest.json')
    dump(out/'scope_evidence.json',dict(scene=c['scene'],source_pointcloud_sha256=c['pointcloud']['sha256'],
        bounds_xyz=c['bounds_xyz'],trajectory_ids=[t['id'] for t in c['trajectories']],
        frame_ids=[f['id'] for f in c['frames']],reviewer='deterministic synchronized geometry verification',
        same_time_interval=True,same_effective_coverage=True,registration_verified=True,
        geometry_audit_sha256=sha(stage/'geometry/geometry_audit.json'),
        limitation='Source odometry and estimated calibration retained; no surveyed boundary claim.'))
    dump(out/'config.json',c)
    shutil.copy2(stage/'geometry/geometry_audit.json',out/'geometry_audit.json')
    shutil.copy2(stage/'extraction_manifest.json',out/'scan_extraction_manifest.json')
    shutil.copy2(stage/'extend.sbatch',out/'inference_job.sbatch')
    (out/'detections.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    newprov=json.loads((stage/'perception/provenance.json').read_text())
    oldprov=json.loads((cache/'provenance.json').read_text())
    assert newprov['completed'] and newprov['frames']==len(fresh)
    assert newprov['config_sha256']==sha(stage/'inference_config.json')
    shutil.copy2(stage/'inference_config.json',out/'new_sam3_config.json')
    assert oldprov['checkpoint_sha256']==newprov['checkpoint_sha256']
    dump(out/'provenance.json',dict(config_sha256=sha(out/'config.json'),completed=True,frames=len(rows),
        checkpoint_sha256=newprov['checkpoint_sha256'],cached_frames=plan['cached_frames'],
        new_sam_frames=plan['new_sam_frames'],new_inference_provenance=newprov,
        gpu_execution=dict(host='HPC',job_id='875817',gpu='NVIDIA H100 80GB HBM3'),
        cached_provenance_sha256=sha(cache/'provenance.json'),
        latest_provenance_sha256=sha(latest/'provenance.json'),
        new_predictions_sha256=sha(stage/'perception/detections.jsonl'),
        geometry_audit_sha256=sha(stage/'geometry/geometry_audit.json'),
        selection_manifest_sha256=c['selection_manifest_sha256']))
    dump(out/'frozen_files.json',{f.name:sha(f) for f in out.iterdir() if f.is_file()})
    print('assembled',len(rows),'frames; new inference',len(fresh))


if __name__=='__main__':main()
