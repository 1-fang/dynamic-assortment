"""Analytical planning displays from reported means; no new experimental data."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'data/current_results.json').read_text())
OUT=ROOT/'out'/'figures';OUT.mkdir(parents=True, exist_ok=True)
parse=lambda s:tuple(float(x.strip().replace(',','')) for x in s.split('±'))
SCENES=['Expedia30','Expedia Head','Expedia Regular','Expedia Cold','KuaiSearch Overall','Q-Head','Q-Regular','Q-Cold','Q-Unseen']
COL={'base':'#786B60','sr':'#287D8C'}
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,'grid.linestyle':'--','grid.linewidth':.55,'grid.color':'#CED3D6','pdf.fonttype':42,'savefig.facecolor':'white'})

def arrays(scene):
    rr=[r for r in DATA['resources'] if r['scene']==scene]
    return (np.array([float(r['level'].split('=')[1]) for r in rr]),np.array([parse(r['base'])[0] for r in rr]),np.array([parse(r['full'])[0] for r in rr]))
def capacity(y,levels,revenues):
    eligible=levels[revenues>=y-1e-9]
    return float(np.min(eligible)) if len(eligible) else float('inf')
def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=.08)
    fig.savefig(OUT/f'{name}.png',dpi=220,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)

derived={}
fig,axs=plt.subplots(3,3,figsize=(9.1,8.5),layout='constrained')
for i,(ax,scene) in enumerate(zip(axs.flat,SCENES)):
    levels,base,sr=arrays(scene)
    ceiling=max(base);floor=.8*min(min(base),min(sr))
    thresholds=np.unique(np.r_[floor,base,sr,ceiling]);thresholds=thresholds[(thresholds>=floor)&(thresholds<=ceiling)]
    # These intervals are exact inverse sets over the finite resource grid.
    segments=[]
    for left,right in zip(thresholds[:-1],thresholds[1:]):
        midpoint=(left+right)/2
        qb=capacity(midpoint,levels,base);qs=capacity(midpoint,levels,sr)
        segments.append({'target_left':float(left),'target_right':float(right),'base_capacity':qb,'sr_capacity':qs})
    for key,label,color,style in [('base_capacity','A2C-Transformer',COL['base'],'--'),('sr_capacity','S/R policy',COL['sr'],'-')]:
        xs=[];ys=[]
        for seg in segments:
            xs.extend([seg['target_left']/ceiling*100,seg['target_right']/ceiling*100]);ys.extend([seg[key],seg[key]])
        ax.plot(xs,ys,color=color,lw=1.7,ls=style,label=label)
    for seg in segments:
        if seg['sr_capacity']<seg['base_capacity']:
            ax.fill_between([seg['target_left']/ceiling*100,seg['target_right']/ceiling*100],seg['sr_capacity'],seg['base_capacity'],color=COL['sr'],alpha=.13,lw=0)
    ax.set_title(f'({chr(97+i)}) {scene}')
    ax.set_xlim(floor/ceiling*100,100);ax.set_yticks(levels)
    ax.set_ylim(levels[0]-.1*(levels[-1]-levels[0]),levels[-1]+.1*(levels[-1]-levels[0]));ax.grid()
    ax.set_xticks([x for x in [40,60,80,100] if x>=floor/ceiling*100])
    ax.set_ylabel('Inventory / hotel' if i<4 else 'Exposure multiplier')
    ax.set_xlabel('')
    derived[scene]={'levels':levels.tolist(),'base_revenue_means':base.tolist(),'sr_revenue_means':sr.tolist(),'target_denominator':float(ceiling),'exact_inverse_segments':segments,'minimum_sr_capacity_at_base_grid_targets':[capacity(float(y),levels,sr) for y in base]}
handles,labels=axs.flat[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False)
fig.supxlabel('Target revenue (% of Base maximum on the tested grid)',fontsize=10)
save(fig,'14_capacity_requirements')

# Fixed-catalog planning scenario: 30 hotels, q initial units per hotel.
levels,base,sr=arrays('Expedia30')
def optimum(charge,revenue):
    v=revenue-charge*30*levels
    k=int(np.argmax(v))
    return float(v[k]),float(levels[k])
breaks={0.,140.}
for revenue in [base,sr]:
    for i in range(len(levels)):
        for j in range(i+1,len(levels)):
            c=(revenue[j]-revenue[i])/(30*(levels[j]-levels[i]))
            if 0<c<140:breaks.add(float(c))
breaks=sorted(breaks)
active=[]
for lo,hi in zip(breaks[:-1],breaks[1:]):
    mid=(lo+hi)/2
    vb,qb=optimum(mid,base);vs,qs=optimum(mid,sr)
    if active and active[-1]['base_q']==qb and active[-1]['sr_q']==qs:
        active[-1]['charge_right']=hi
    else:active.append({'charge_left':lo,'charge_right':hi,'base_q':qb,'sr_q':qs})
fig,axs=plt.subplots(1,2,figsize=(9.2,3.9),layout='constrained')
for key,label,color,style in [('base_q','A2C-Transformer',COL['base'],'--'),('sr_q','S/R policy',COL['sr'],'-')]:
    xx=[];yy=[]
    for seg in active:xx.extend([seg['charge_left'],seg['charge_right']]);yy.extend([seg[key],seg[key]])
    axs[0].plot(xx,yy,label=label,color=color,ls=style,lw=1.9)
axs[0].set_yticks(levels);axs[0].set_ylim(.8,4.2);axs[0].set_xlim(0,140)
axs[0].set_title('(a) Inventory choice among tested policies');axs[0].set_ylabel('Chosen initial inventory / hotel');axs[0].grid();axs[0].legend(frameon=False,loc='upper right')
charges=np.unique(np.r_[np.linspace(0,140,500),breaks]);gain=[]
for c in charges:gain.append(optimum(float(c),sr)[0]-optimum(float(c),base)[0])
gain=np.array(gain)
axs[1].fill_between(charges,0,gain,color=COL['sr'],alpha=.3,label='S/R preferred')
axs[1].fill_between(charges,gain,1100,color=COL['base'],alpha=.18,label='Base preferred')
axs[1].plot(charges,gain,color=COL['sr'],lw=1.6)
axs[1].text(11,120,'S/R preferred',color='#145564',fontsize=9)
axs[1].text(87,870,'Base preferred',color='#5F5146',fontsize=9)
axs[1].set_ylim(0,1100);axs[1].set_xlim(0,140);axs[1].set_title('(b) Adoption with a daily S/R charge');axs[1].set_ylabel('Additional S/R charge / day');axs[1].grid()
for ax in axs:ax.set_xlabel('Inventory charge / initial unit / day')
save(fig,'15_capacity_adoption')

checks=[]
for c in [0,60,90,120,140]:
    vb,qb=optimum(c,base);vs,qs=optimum(c,sr)
    checks.append({'inventory_charge':c,'base_q':qb,'sr_q':qs,'base_net_mean':vb,'sr_net_mean_before_SR_charge':vs,'break_even_daily_SR_charge':vs-vb})
result={'status':'derived_planning_analysis','new_experiments':False,'empirical_source':'data/current_results.json','frontier_definition':'min tested resource r such that reported mean J_a(r)>=target','frontiers':derived,'cost_scenario':{'setting':'Expedia30','hotels':30,'candidate_inventory_per_hotel':[1,2,3,4],'net_objective':'reported_revenue_mean(q) - 30*q*inventory_charge - optional_SR_charge','observed_cost_data':False,'cost_range_is_hypothetical':[0,140],'analytic_optimal_inventory_intervals':active,'examples':checks},'uncertainty':'point-estimate planning calculations; no new confidence intervals or profit-training claims'}
(OUT/'planning-analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
print('Two analytical decision figures written; no experimental outcomes simulated.')
print(json.dumps(checks,indent=2))
