from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'out'/'figures'
OUT.mkdir(parents=True, exist_ok=True)
d=json.loads((ROOT/'data/current_results.json').read_text())
scenes=['Expedia30','Expedia Head','Expedia Regular','Expedia Cold','KuaiSearch Overall','Q-Head','Q-Regular','Q-Cold','Q-Unseen']
short=['30 hotels','Hotel Head','Hotel Regular','Hotel Cold','Kuai Overall','Q-Head','Q-Regular','Q-Cold','Q-Unseen']
parse=lambda x:tuple(float(i.strip().replace(',','')) for i in x.split('±'))
p={(r['method'],r['scene']):parse(r['reported']) for r in d['performance']}
ppo={(r['method'],r['scene']):parse(r['reported']) for r in d['ppo']}
emb={row[0]:[parse(x) for x in row[1:]] for row in d['embedding_rows']['Transformer-Pointer']}
C={'base':'#6D625C','qwen':'#2F7882','bge':'#B29E61','mbert':'#99B66D','st':'#8AAAC5','e5':'#A688AF'}
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,'grid.color':'#C8CDD1','grid.linestyle':'--','grid.linewidth':.55,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=.08)
    fig.savefig(OUT/f'{name}.png',dpi=200,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
def gains(method,reference='A2C Transformer'):
    return [(p[method,s][0]/p[reference,s][0]-1)*100 for s in scenes]

fig,axs=plt.subplots(1,2,figsize=(9,3),layout='constrained')
for ax,idx,title in [(axs[0],range(5),'Hotel and Overall settings'),(axs[1],range(5,9),'Query-support settings')]:
    vals=gains('Qwen + S/R');xx=np.arange(len(idx));ax.bar(xx,[vals[i] for i in idx],color=C['qwen'],width=.6)
    ax.set_xticks(xx,[short[i] for i in idx],rotation=25,ha='right');ax.set_title(title);ax.set_ylabel('Gain over A2C Transformer (%)');ax.grid(axis='y')
    for x,i in zip(xx,idx):ax.annotate(f'{vals[i]:.2f}',(x,vals[i]),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
    ax.set_ylim(0,max(vals[i] for i in idx)*1.17)
save(fig,'04_main_gains')

fig,axs=plt.subplots(1,2,figsize=(8.5,3.1),layout='constrained')
x=np.arange(3)
for method,label,color,marker in [('A2C Transformer','A2C Transformer',C['base'],'o'),('Myopic','Myopic',C['bge'],'D'),('Qwen + S/R','Qwen S/R',C['qwen'],'s')]:
    axs[0].plot(x,[p[method,s][0] for s in scenes[1:4]],label=label,color=color,marker=marker,lw=1.7)
for method,label,color,marker in [('A2C Transformer','Over A2C Transformer',C['base'],'o'),('Myopic','Over Myopic',C['bge'],'D')]:
    axs[1].plot(x,[(p['Qwen + S/R',s][0]/p[method,s][0]-1)*100 for s in scenes[1:4]],label=label,color=color,marker=marker,lw=1.7)
for ax,title in zip(axs,['Revenue levels','Comparator-dependent Qwen gains']):
    ax.set_xticks(x,['Head','Regular','Cold']);ax.set_title(title);ax.grid();ax.legend(frameon=False,fontsize=7)
axs[0].set_ylabel('Revenue / search');axs[1].set_ylabel('Qwen relative gain (%)')
save(fig,'05_market_support')

fig,axs=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
for ax,idx,title in [(axs[0],list(range(5)),'Hotel and Overall settings'),(axs[1],list(range(5,9)),'Query-support settings')]:
    xx=np.arange(len(idx));w=.14
    methods=[('mBERT + mean pooling','mBERT','mbert'),('Sentence-Transformer','Sentence-Transformer','st'),('BGE','BGE-M3','bge'),('Multilingual E5','E5','e5'),(None,'Qwen S/R','qwen')]
    for j,(key,label,col) in enumerate(methods):
        vals=[((emb[key][i][0] if key else p['Qwen + S/R',scenes[i]][0])/p['A2C Transformer',scenes[i]][0]-1)*100 for i in idx]
        ax.bar(xx+(j-2)*w,vals,w,label=label,color=C[col])
    ax.axhline(0,color='#444',lw=.7);ax.set_xticks(xx,[short[i] for i in idx],rotation=25,ha='right');ax.set_title(title);ax.set_ylabel('Gain over A2C Transformer (%)');ax.grid(axis='y')
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='outside upper center',frameon=False)
save(fig,'08_augmentation')

fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
for ax,idx,title in [(axs[0],list(range(5)),'Hotel and Overall settings'),(axs[1],list(range(5,9)),'Query-support settings')]:
    methods=['Qwen + S/R','DeepSeek + S/R','GPT-4o-mini + S/R','GPT-5.4-mini + S/R']
    mat=np.array([gains(m) for m in methods])[:,idx];im=ax.imshow(mat,aspect='auto',cmap='YlGnBu',vmin=0,vmax=25)
    ax.set_xticks(range(len(idx)),[short[i] for i in idx],rotation=30,ha='right');ax.set_yticks(range(4),[m.replace(' + S/R','') for m in methods]);ax.set_title(title)
    for y in range(4):
        for x in range(len(idx)):ax.text(x,y,f'{mat[y,x]:.2f}%',ha='center',va='center',fontsize=8,color='white' if mat[y,x]>14 else '#172F41')
save(fig,'09_model_sources')

fig,axs=plt.subplots(1,2,figsize=(9,3),layout='constrained')
for ax,idx,title in [(axs[0],list(range(5)),'Hotel and Overall settings'),(axs[1],list(range(5,9)),'Query-support settings')]:
    vals=[(ppo['PPO + Qwen S/R',s][0]/ppo['PPO',s][0]-1)*100 for s in scenes];xx=np.arange(len(idx))
    ax.bar(xx,[vals[i] for i in idx],color=[C['qwen'] if vals[i]>=0 else '#AE6A60' for i in idx],width=.6);ax.axhline(0,color='#444',lw=.7)
    ax.set_xticks(xx,[short[i] for i in idx],rotation=25,ha='right');ax.set_title(title);ax.set_ylabel('Gain over PPO (%)');ax.grid(axis='y')
    for x,i in zip(xx,idx):ax.annotate(f'{vals[i]:.2f}',(x,vals[i]),xytext=(0,3 if vals[i]>0 else -10),textcoords='offset points',ha='center',fontsize=8)
    lo=min(0,min(vals[i] for i in idx));hi=max(vals[i] for i in idx);ax.set_ylim(lo-.15*(hi-lo),hi+.2*(hi-lo))
save(fig,'10_ppo_transfer')

for name,scene_list,shape,size in [('06_resource_results',scenes[:4],(2,2),(9,4.6)),('07_query_budget',scenes[5:],(2,2),(9,4.6))]:
    fig,axs=plt.subplots(*shape,figsize=size,layout='constrained')
    for ax,s in zip(axs.flat,scene_list):
        rows=[r for r in d['resources'] if r['scene']==s];x=[float(r['level'].split('=')[1]) for r in rows]
        for key,label,col,marker,style in [('base','A2C Transformer','base','o','--'),('full','Qwen S/R','qwen','s','-')]:
            vals=[parse(r[key])[0] for r in rows];sd=[parse(r[key])[1] for r in rows]
            ax.errorbar(x,vals,yerr=sd,label=label,color=C[col],marker=marker,linestyle=style,lw=1.5,ms=4,capsize=2)
        ax.set_title(s);ax.set_xlabel('Inventory per hotel' if name=='06_resource_results' else 'Exposure multiplier');ax.set_ylabel('Daily revenue' if s=='Expedia30' else ('Revenue / search' if name=='06_resource_results' else 'GMV / 1,000 requests'))
        ax.grid();ax.set_xticks(x)
    handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False)
    save(fig,name)
