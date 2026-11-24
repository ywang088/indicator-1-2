import json,sqlite3,base64,io
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
r=Path(__file__).resolve().parent;db=sqlite3.connect(r/'final_results/runtime/map.sqlite')
nodes={i:dict(json.loads(a),id=i,kind=k) for i,k,a in db.execute('select * from nodes')};edges=[dict(source=s,target=t,kind=k) for s,t,k in db.execute('select source,target,kind from edges')]
observations={}
for s,t,a in db.execute("select source,target,attributes from edges where kind='observes'"):observations.setdefault(t,[]).append(s)
for n in nodes.values():
 if n['kind']=='frame':n['position']=[v[3] for v in n['T_world_camera'][:3]]
portal=next(e['target'] for e in edges if e['source']=='region:unknown:E' and e['kind']=='candidate_incident')
unknown=next(i for i,n in nodes.items() if n['kind']=='object' and n.get('position') is None and n['label']=='column')
choices=[('门 000099:021 · 原图可见门','portal:object:000099:021'),('门 000099:025 · 疑似墙边误检','portal:object:000099:025'),('F 区柱子 · 多视角观测','object:000410:004'),('E 区柱子 · 多视角观测','object:004561:011'),('E 区门户候选',portal),('未定位柱子 · 未知归属',unknown)]
views=[];union=set()
for name,focus in choices:
 keep={focus}
 for e in edges:
  if e['source']==focus or e['target']==focus:
   other=e['target'] if e['source']==focus else e['source']
   if nodes[other]['kind']!='frame':keep.add(other)
 objects=[i for i in keep if nodes[i]['kind']=='object']
 for obj in objects:
  fs=observations.get(obj,[]);step=max(1,len(fs)//8);keep.update(fs[::step][:10])
 regions=[i for i in keep if nodes[i]['kind']=='region']
 for region in regions:
  keep.update(nodes[region].get('anchors',[]))
  portals=[e['target'] for e in edges if e['source']==region and e['kind']=='candidate_incident'][:2]
  keep.update(portals)
  for pid in portals:
   keep.update(e['target'] for e in edges if e['source']==pid and e['kind']=='represented_by')
 keep={i for i in keep if nodes[i]['kind'] not in ('scene','floor')}
 visible=[e for e in edges if e['source'] in keep and e['target'] in keep]
 views.append(dict(name=name,focus=focus,nodes=sorted(keep),edges=visible));union.update(keep)
compact=[]
for i in union:
 n=nodes[i];compact.append(dict(id=i,kind=n['kind'],label=n.get('label',n.get('name',n.get('frame_id',i))),position=n.get('position'),region=n.get('region'),observations=len(observations.get(n.get('object',i),[])),note=('原图可见门，尚未核验连通' if i=='portal:object:000099:021' else '疑似墙边误检，待复核' if i=='portal:object:000099:025' else None),neighbors=sum(e['source']==i or e['target']==i for e in edges)))
pts=np.load(r/'final_results/baseline/points.npy')[::100][:, [0,2]]
c=json.loads((r/'final_results/audit/config.json').read_text());frames={f['id']:f for f in c['frames']}
paths=[[[frames[i]['T_world_camera'][0][3],frames[i]['T_world_camera'][2][3]] for i in t['frame_ids']] for t in c['trajectories']]
anchors=[]
for n in nodes.values():
 if n['kind']=='region':
  for j,p in enumerate(n['positions']):anchors.append(dict(label=n['label']+str(j+1),position=p,id=n['anchors'][j]))
payload=dict(nodes=compact,views=views,counts=dict(Counter(n['kind'] for n in nodes.values())),edge_counts=dict(Counter(e['kind'] for e in edges)),points=pts.round(2).tolist(),objects=[[round(n['position'][0],2),round(n['position'][2],2)] for n in nodes.values() if n['kind']=='object' and n.get('position') is not None],paths=paths,anchors=anchors)
p=Path('/Users/yixuanwang/.codex/visualizations/2026/11/22/01a20366-f049-7cc3-abdb-f630f96b0bd5/parking-graph-current.html')
s=(r/'graph_preview.template.html').read_text().replace('__GRAPH_DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'));p.write_text(s);assert p.stat().st_size<1_000_000
for v in views:
 assert all(e['source'] in v['nodes'] and e['target'] in v['nodes'] for e in v['edges'])
 print(v['name'],len(v['nodes']),len(v['edges']))
print('bytes',p.stat().st_size)
Path('/tmp/parking-graph-check.js').write_text(s.split('<script>')[1].split('</script>')[0])
