"""Resolve already-selected frame IDs to server originals; freeze hashes before inference."""
import argparse,json,copy
from pathlib import Path
from perception import sha,dump
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path(__file__).parent/'frozen/config.json');p.add_argument('--dataset',type=Path,default=Path('/data/user/ywang088/datasets/地下停车场数据集/2022-3-24-11-57-11'));p.add_argument('--pointcloud',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
c=json.loads(a.config.read_text());c['parent_config_sha256']=sha(a.config)
if a.output.exists():raise FileExistsError(a.output)
if a.dataset.name!=c['scene']:raise ValueError('dataset sequence name must match sealed scene')
cache={}
for f in c['frames']+c.get('replay_inputs',[]):
 path=(a.dataset/'image_color'/f'{int(f["id"]):06}.png').resolve()
 if not path.is_file():raise FileNotFoundError(path)
 if path not in cache:cache[path]=sha(path)
 f.update(image=str(path),image_sha256=cache[path])
cloud=a.pointcloud or a.dataset/'traj/pcd'/Path(c['pointcloud']['path']).name
if not cloud.is_file():raise FileNotFoundError(cloud)
c['pointcloud'].update(path=str(cloud.resolve()),sha256=sha(cloud))
dump(a.output,c);print(json.dumps(dict(config=str(a.output),unique_replay_inputs=len(cache),keyframes=len(c['frames']),sha256=sha(a.output))))
