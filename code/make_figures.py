"""Render measured results, preserving metric map scale and unknown regions."""
import argparse,json,sqlite3
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

def main():
 p=argparse.ArgumentParser();p.add_argument('--map',type=Path,required=True);a=p.parse_args();root=a.map
 font=Path('/System/Library/Fonts/STHeiti Light.ttc')
 if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
 plt.rcParams['axes.unicode_minus']=False
 r=json.loads((root/'report.json').read_text());c=json.loads((root/'audit/config.json').read_text());db=sqlite3.connect(root/'runtime/map.sqlite');nodes=[dict(json.loads(v),id=i,kind=k) for i,k,v in db.execute('select * from nodes')];db.close();counts={k:sum(n['kind']==k for n in nodes) for k in ['scene','floor','region','portal','object','frame']};pts=np.load(root/'baseline/points.npy');pts=pts[::8]
 fig=plt.figure(figsize=(16,11),facecolor='#eef3f7');gs=fig.add_gridspec(2,2,width_ratios=[1.45,1],height_ratios=[1,1],left=.05,right=.97,top=.90,bottom=.10,hspace=.28,wspace=.20)
 ax=fig.add_subplot(gs[:,0]);ax.set_facecolor('#fafcfd');ax.scatter(pts[:,0],pts[:,2],s=.15,c='#9daeb9',alpha=.45,rasterized=True)
 objects=[n for n in nodes if n['kind']=='object' and n.get('position') is not None];pp=np.array([n['position'] for n in objects]);ax.scatter(pp[:,0],pp[:,2],s=3,c='#d99438',alpha=.30,label='可定位的候选物体')
 frames={f['id']:f for f in c['frames']};palette=plt.get_cmap('tab20')
 for i,t in enumerate(c['trajectories']):
  q=np.array([[frames[k]['T_world_camera'][0][3],frames[k]['T_world_camera'][2][3]] for k in t['frame_ids']]);ax.plot(q[:,0],q[:,1],color=palette(i),lw=1.2,alpha=.85)
 for region in [n for n in nodes if n['kind']=='region']:
  col='#00877d' if region['label']=='E' else '#2767b2'
  for i,pos in enumerate(region['positions']):
   ax.scatter(pos[0],pos[2],s=70,c=col,marker='D',edgecolors='white',zorder=5);ax.annotate(region['label']+str(i+1),(pos[0],pos[2]),xytext=(6,5),textcoords='offset points',fontsize=11,color=col,fontweight='bold',bbox=dict(fc='white',ec='none',alpha=.85,pad=1))
 ax.set_aspect('equal');ax.set_xlabel('场景 X / 米');ax.set_ylabel('场景 Z / 米');ax.set_title('18 条轨迹 · 实测几何与柱标锚点',loc='left',fontsize=17,pad=14);ax.grid(alpha=.12);ax.legend(loc='lower left',fontsize=9)
 ax.text(.02,.98,'灰：同步点云观测支持（不是可通行区域）\n彩线：18 条固定轨迹；菱形：多视角柱标\nE / F 是候选分区；未绘制未经核验的边界',transform=ax.transAxes,va='top',fontsize=10,bbox=dict(fc='white',ec='#d5dfe5',alpha=.95,pad=8))
 h=fig.add_subplot(gs[0,1]);h.axis('off');h.set_title('分层认知图 · 可追溯到观测帧',loc='left',fontsize=17,pad=14)
 labels=[('场景','scene','#173c58'),('楼层待核验','floor','#496d88'),('柱标区域候选','region','#00877d'),('门户候选','portal','#8060ac'),('物体候选记录','object','#d48c2f'),('关键帧','frame','#2880ba')]
 for j,(label,key,color) in enumerate(labels):
  y=.90-j*.16;h.text(.06,y,f'{label}   {counts[key]:,}',transform=h.transAxes,color='white',fontsize=14,va='center',bbox=dict(boxstyle='round,pad=.5',facecolor=color,edgecolor='none'))
  if j<5:h.annotate('',xy=(.24,y-.11),xytext=(.24,y-.045),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#819aaa',lw=1.8))
 h.text(.63,.88,'E：5 个柱标锚点\nF：2 个柱标锚点\n\n门户仅为候选开口\n未强行连接两侧区域\n\n未定位记录保留观测\n不生成虚假坐标',transform=h.transAxes,va='top',fontsize=11,linespacing=1.8)
 b=fig.add_subplot(gs[1,1]);values=[r['baseline_bytes']/1e6,r['cognitive_bytes']/1e6];bars=b.barh(['同范围点云','认知图 + 物体记录'],values,color=['#9baebc','#00877d'],height=.48);b.set_ylim(1.8,-.5);b.set_xlim(0,13);b.set_xlabel('未打包压缩文件 / MB（1 MB = 1,000,000 字节）');b.set_title('实际存储对比',loc='left',fontsize=17,pad=16);b.spines[['top','right','left']].set_visible(False);b.grid(axis='x',alpha=.13);b.set_axisbelow(True)
 for bar,val in zip(bars,values):b.text(val+.15,bar.get_y()+bar.get_height()/2,f'{val:.3f} MB',va='center',fontsize=12)
 b.text(.02,.14,f'节省 {r["reduction_percent"]:.2f}%',transform=b.transAxes,fontsize=26,color='#00877d',fontweight='bold');b.text(.02,.03,'19 份图独立进程加载成功（18 条回放 + 汇总图）',transform=b.transAxes,fontsize=10)
 fig.suptitle('指标 1.2｜停车场真实离线运行结果',x=.05,ha='left',fontsize=25,color='#173c58',fontweight='bold');fig.text(.05,.045,'HPC SAM3 实跑 267 帧；原回放输入 4,851 个唯一帧。存储数值超过 20% 门槛；区域边界、标定与人工语义核验尚未完成，不能据此宣称正式验收通过。',fontsize=11,color='#715520');fig.savefig(root/'结果总览.png',dpi=150);plt.close(fig)
 print(root/'结果总览.png')
if __name__=='__main__':main()
