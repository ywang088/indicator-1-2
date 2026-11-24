"""Create a portable full viewer; retain graph records, embed display images."""
import base64,io,json,re
from pathlib import Path
from PIL import Image
root=Path(__file__).resolve().parent/'final_results'
def data(path,width=960,quality=75):
 im=Image.open(path).convert('RGB');im.thumbnail((width,width*3));b=io.BytesIO();im.save(b,format='JPEG',quality=quality,optimize=True)
 return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()
s=(root/'index.html').read_text();images={p.stem:data(p) for p in sorted((root/'audit/images').glob('*.jpg'))}
assert len(images)==267
s=s.replace('<script>const DATA=', '<script>const EMBEDDED_IMAGES='+json.dumps(images,separators=(',',':'))+';const DATA=',1)
old="$('image').src='audit/images/'+id+'.jpg'";assert old in s;s=s.replace(old,"$('image').src=EMBEDDED_IMAGES[id]")
s=s.replace('href="柱标证据.html"','href="#pillar-evidence"')
gallery=(root/'柱标证据.html').read_text();sections=''.join(re.findall(r'<section>.*?</section>',gallery,re.S))
for path in sorted((root/'column_evidence').glob('*.jpg')):sections=sections.replace('column_evidence/'+path.name,data(path))
s=s.replace('</main>','<section class="card" id="pillar-evidence"><h2>柱标原图核对</h2><p>助手目视转写，不是人工验收真值。此单文件版内嵌缩小后的展示图片；完整原图保持在原结果包。</p>'+sections+'</section></main>')
output=root/'停车场结果_单文件版.html';output.write_text(s)
fragment=Path('/Users/yixuanwang/.codex/visualizations/2026/11/22/01a20366-f049-7cc3-abdb-f630f96b0bd5/parking-results-remote.html')
f=fragment.read_text()
for key,name in [('OVERVIEW','结果总览.png'),('DETECTION','SAM3识别抽查.jpg'),('COLUMN','柱标识别抽查.jpg')]:f=f.replace('__'+key+'__',data(root/name,1440,78))
fragment.write_text(f);assert fragment.stat().st_size<1_000_000
assert "audit/images/'+id" not in s
print('full viewer',output.stat().st_size,'bytes; inline preview',fragment.stat().st_size,'bytes; embedded frames',len(images))
Path('/tmp/parking-single-check.js').write_text(s.split('<script>')[1].split('</script>')[0])
