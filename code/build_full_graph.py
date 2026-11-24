"""Full four-layer graph, compact topology with no node/edge sampling."""
import argparse,sqlite3,json,base64,io
from pathlib import Path
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();base=Path(__file__).resolve().parent
c=sqlite3.connect(a.result/'runtime/map.sqlite');nodes=[];index={};layers={'region':0,'portal':1,'object':2,'frame':3}
for id,kind,attrs in c.execute('SELECT * FROM nodes ORDER BY kind,id'):
 d=json.loads(attrs);index[id]=len(nodes);nodes.append([id,layers[kind],d.get('label','')])
kinds=[r[0] for r in c.execute('select distinct kind from edges order by kind')];edges=[[index[s],index[t],kinds.index(k)] for s,t,k in c.execute('select source,target,kind from edges')];photos=[]
for f in ['004305','004345']:
 im=Image.open(base/'source_rgb'/f'{f}.png');im.thumbnail((800,450));b=io.BytesIO();im.save(b,format='JPEG',quality=77);photos.append('data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode())
data=json.dumps(dict(nodes=nodes,edges=edges,kinds=kinds,photos=photos),ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
html=(base/'full_graph.template.html').read_text().replace('__GRAPH_DATA__',data)
assert len(html.encode())<1000000,(len(html.encode()),'inline limit')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(html)
print(json.dumps({'nodes':len(nodes),'edges':len(edges),'bytes':len(html.encode()),'path':str(a.output)},ensure_ascii=False))
