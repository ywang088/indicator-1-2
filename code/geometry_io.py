"""Minimal binary/ASCII PCD reader; preserves fields and respects POINTS, not EOF padding."""
from pathlib import Path
import numpy as np

def read_pcd(path):
    with Path(path).open('rb') as f:
        header={}
        while True:
            line=f.readline()
            if not line:raise ValueError('PCD missing DATA header')
            words=line.decode('ascii').strip().split()
            if not words or words[0].startswith('#'):continue
            header[words[0]]=words[1:]
            if words[0]=='DATA':break
        fields=header['FIELDS'];sizes=list(map(int,header['SIZE']));counts=list(map(int,header.get('COUNT',['1']*len(fields))))
        dtype=[]
        for i,(name,size,kind,count) in enumerate(zip(fields,sizes,header['TYPE'],counts)):
            name=f'_padding_{i}' if name=='_' else name
            typ={('F',4):'<f4',('F',8):'<f8',('U',1):'u1',('U',2):'<u2',('U',4):'<u4',('I',1):'i1',('I',2):'<i2',('I',4):'<i4'}[(kind,size)]
            dtype.append((name,typ) if count==1 else (name,typ,(count,)))
        n=int(header['POINTS'][0]);dt=np.dtype(dtype)
        if header['DATA'][0]=='binary':
            raw=f.read(n*dt.itemsize)
            if len(raw)!=n*dt.itemsize:raise ValueError(f'Truncated PCD {path}')
            a=np.frombuffer(raw,dtype=dt).copy()
        else:raise ValueError('Only uncompressed binary PCD is supported')
    xyz=np.column_stack([a[k] for k in ('x','y','z')]).astype(np.float64)
    good=np.isfinite(xyz).all(axis=1)
    return xyz[good],a[good],header

def write_ply(path,xyz,rgb=None):
    xyz=np.asarray(xyz,dtype='<f4')
    fields=[('x','<f4'),('y','<f4'),('z','<f4')]
    if rgb is not None:fields += [('red','u1'),('green','u1'),('blue','u1')]
    data=np.empty(len(xyz),dtype=fields)
    for i,k in enumerate(['x','y','z']):data[k]=xyz[:,i]
    header=f'ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n'
    if rgb is not None:
        for i,k in enumerate(['red','green','blue']):data[k]=np.asarray(rgb,dtype=np.uint8)[:,i];header+=f'property uchar {k}\n'
    with Path(path).open('wb') as f:f.write((header+'end_header\n').encode());f.write(data.tobytes())

def geometry_scope_sha256(config):
 """Hash geometry-defining values; permit only verified filesystem rebinding."""
 import json,hashlib
 scope={k:config[k] for k in ('scene','K','image_size','bounds_xyz')}
 scope['frames']=[{k:f[k] for k in ('id','timestamp','T_world_camera')} for f in config['frames']]
 scope['pointcloud']={k:config['pointcloud'][k] for k in ('T_world_source','voxel_m')}
 scope['trajectories']=[{k:t[k] for k in ('id','frame_ids')} for t in config['trajectories']]
 return hashlib.sha256(json.dumps(scope,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
