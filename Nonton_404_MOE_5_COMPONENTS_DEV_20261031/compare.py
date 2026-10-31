"""Actual CSV comparison; no fabricated weekly numbers."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def compare_files(reference,ensemble,image,table):
    a=pd.read_csv(reference)
    b=pd.read_csv(ensemble)
    keys=['week','warehouse_name','article']
    assert set(map(tuple,a[keys].values))==set(map(tuple,b[keys].values))
    x=a.groupby('week',as_index=False).quantity.sum().rename(columns={'quantity':'Official forecast 5'})
    y=b.groupby('week',as_index=False).quantity.sum().rename(columns={'quantity':'Rebuilt MOE'})
    c=x.merge(y,on='week').sort_values('week')
    c['Difference']=c['Rebuilt MOE']-c['Official forecast 5']
    c['Difference %']=100*(c['Rebuilt MOE']/c['Official forecast 5']-1)
    Path(table).parent.mkdir(parents=True,exist_ok=True)
    c.to_csv(table,index=False)
    fig,ax=plt.subplots(figsize=(12,6))
    ax.plot(c.week,c['Official forecast 5'],marker='o',linewidth=2,label='Official forecast 5 (CatBoost)')
    ax.plot(c.week,c['Rebuilt MOE'],marker='o',linewidth=2,label='Rebuilt Mixture of Experts')
    ax.set(title='Nine-week Nonton demand forecast comparison',xlabel='Week start',ylabel='Forecast units (all warehouses / articles)')
    ax.grid(alpha=.3);ax.legend();plt.xticks(rotation=35);fig.tight_layout()
    fig.savefig(image,dpi=170);plt.close(fig)
    print('COMPARED weekly totals:',c[['week','Official forecast 5','Rebuilt MOE','Difference %']].round(2).to_string(index=False))
    return c

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--ensemble',required=True)
    p.add_argument('--image',default='comparison.png');p.add_argument('--table',default='comparison_weekly.csv');a=p.parse_args()
    compare_files(a.reference,a.ensemble,a.image,a.table)
