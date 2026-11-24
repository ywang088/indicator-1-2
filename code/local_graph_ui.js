const modeSelect=root.querySelector('[data-graph-mode]'),localGraph=root.querySelector('.local-graph');
modeSelect.onchange=()=>{graphMode=modeSelect.value;drawGraph();save()};
function drawGraph(){
 const local=graphMode==='local';graph.hidden=local;root.querySelector('.layers').hidden=local;root.querySelector('.local-scroll').hidden=!local;
 if(!local){localVersion++;drawFullGraph();root.querySelector('[data-graph-summary]').textContent='完整 graph：截至当前帧的全部节点与已记录连接；当前目标高亮。';return}
 drawLocalGraph();
}
function drawLocalGraph(){
 const version=++localVersion,t=D.tracks[ti],f=t.frames[si],current=new Set(f.boxes.map(b=>b[0]));
 const portals=f.portals.map((p,i)=>({p,i})).filter(({p})=>p[2].includes(si)||p[1].some(id=>current.has(id)));
 const relatedZones=new Set(portals.flatMap(({p})=>p[0]));const regions=f.regions.filter(r=>r[1].some(id=>current.has(id))||relatedZones.has(r[0]));
 root.querySelector('[data-graph-summary]').textContent='当前帧局部图：'+f.boxes.length+' 个物体 · '+regions.length+' 个相关区域 · '+portals.length+' 个相关门户 · 1 个观测帧';
 localGraph.replaceChildren();const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.classList.add('local-edges');svg.setAttribute('aria-hidden','true');localGraph.append(svg);
 const top=document.createElement('div');top.className='local-top';localGraph.append(top);const nodes=new Map(),edges=[];
 function group(title){const g=document.createElement('div');g.className='local-group';const h=document.createElement('div');h.className='node-title';h.textContent=title;g.append(h);top.append(g);return g}
 function node(parent,id,title,sub,button=false){const e=document.createElement(button?'button':'div');e.className='local-node'+(button?' object-node':'');e.dataset.node=id;if(button)e.type='button';const name=document.createElement('span');name.className='node-title';name.textContent=title;const detail=document.createElement('span');detail.className='node-sub';detail.textContent=sub;if(button)e.append(name,detail);else e.append(name,document.createElement('br'),detail);parent.append(e);nodes.set(id,e);return e}
 const rg=group('相关区域');regions.forEach(r=>node(rg,'r'+r[0],r[0]+' 区',r[3]==='awaiting_letter'?'柱色临时区域 · 等待读取字母':r[3]==='conflict'?'颜色与字母冲突 · 暂停绑定':r[3]==='bound'?({red:'红色',yellow:'黄色',blue:'蓝色',green:'绿色',purple:'紫色'}[r[4]]||r[4])+' → '+r[0]+' · 字母已绑定':'柱子标识证据；非精确边界'));if(!regions.length){const e=document.createElement('span');e.textContent='当前帧没有相关区域证据';rg.append(e)}
 const pg=group('相关门户');portals.forEach(({p,i})=>node(pg,'p'+i,p[0].join(' ↔ ')+' 共享通道',p[3]==='color_supported_connection_candidate'?'颜色分区连接候选 · 字母待确认':'连接候选 · 非实体门要求'));if(!portals.length){const e=document.createElement('span');e.textContent='当前帧无相关门户';pg.append(e)}
 const fg=group('当前观测');node(fg,'frame','观测帧 '+f.fid,t.id+' · '+f.boxes.length+' 个目标');
 const list=document.createElement('div');list.className='object-grid';localGraph.append(list);
 f.boxes.forEach((b,j)=>{const zones=f.regions.filter(r=>r[1].includes(b[0])).map(r=>r[0]),links=portals.filter(({p})=>p[1].includes(b[0])).map(({p})=>p[0].join('↔'));
  const name=(j+1)+' · '+(words[D.labels[b[1]]]||D.labels[b[1]])+(b[7]?' ['+b[7]+']':'');const sub=(zones.length?f.regions.filter(r=>r[1].includes(b[0])).map(r=>r[0]+' 区 · '+(r[3]==='awaiting_letter'?'颜色暂分':r[3]==='conflict'?'绑定待核对':r[3]==='bound'?(b[7].split(',').includes(r[0])?'柱标确认':'颜色关联'):'柱标支持')).join('、'):'区域未确认')+(links.length?' · 支撑 '+links.join('、')+' 门户':'')+' · '+b[2].toFixed(2)+'\n'+t.ids[b[0]];
  const e=node(list,'o'+b[0],name,sub,true);e.setAttribute('aria-pressed',String(selected===b[0]));e.onclick=()=>{selected=b[0];obj.value=selected;detail();save()};edges.push(['frame','o'+b[0],'观测']);zones.forEach(z=>edges.push(['r'+z,'o'+b[0],'柱标']));
 });
 portals.forEach(({p,i})=>{p[0].forEach(z=>edges.push(['r'+z,'p'+i,'连接']));p[1].filter(id=>current.has(id)).forEach(id=>edges.push(['p'+i,'o'+id,'支撑']));if(p[2].includes(si))edges.push(['frame','p'+i,'支持连接'])});
 (()=>{if(version!==localVersion||graphMode!=='local')return;const box=localGraph.getBoundingClientRect();svg.setAttribute('viewBox','0 0 '+box.width+' '+box.height);
  edges.forEach(([a,b,kind])=>{if(!nodes.has(a)||!nodes.has(b))return;const aa=nodes.get(a).getBoundingClientRect(),bb=nodes.get(b).getBoundingClientRect(),x1=aa.left+aa.width/2-box.left,y1=aa.top+aa.height/2-box.top,x2=bb.left+bb.width/2-box.left,y2=bb.top+bb.height/2-box.top,path=document.createElementNS(ns,'path');path.setAttribute('d',`M ${x1} ${y1} C ${x1} ${(y1+y2)/2}, ${x2} ${(y1+y2)/2}, ${x2} ${y2}`);path.setAttribute('fill','none');const hot=a==='o'+selected||b==='o'+selected;path.setAttribute('stroke',color(hot?'--viz-series-3':'--foreground'));path.setAttribute('stroke-opacity',hot?'.85':'.12');path.setAttribute('stroke-width',hot?'2':'1');const title=document.createElementNS(ns,'title');title.textContent=kind;path.append(title);svg.append(path)})
 })();
}
