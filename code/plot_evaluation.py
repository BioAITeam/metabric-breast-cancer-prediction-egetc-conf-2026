"""Export calibration and explanation figures from saved prediction artifacts."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from evaluation_utils import keyed, require

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'outputs'
ANALYSIS = ROOT / 'temp' / 'evaluation'
OUT = REF
plt.rcParams.update({'pdf.fonttype':42, 'ps.fonttype':42, 'font.family':'DejaVu Sans', 'font.size':9})

def calibration():
    colors = ['#264653','#e9c46a','#457b9d','#b56576','#6a994e']
    markers = ['o','s','^','D','v']
    fig, axes = plt.subplots(1,2,figsize=(11.5,4.5))
    for ax,target,panel in zip(axes,['Overall_Survival','NPI_High_Risk'],['A','B']):
        data = pd.read_csv(REF / f'oof_trad_{target}.csv')
        ax.plot([0,1],[0,1],'--',color='#333333',linewidth=1,label='Perfect calibration')
        for model,color,marker in zip(data.columns[2:],colors,markers):
            observed,predicted=calibration_curve(data.y_true,data[model],n_bins=10,strategy='uniform')
            score=brier_score_loss(data.y_true,data[model])
            value=f'{score:.2e}' if 0<score<.001 else f'{score:.3f}'
            ax.plot(predicted,observed,marker=marker,color=color,linewidth=1.3,markersize=3,label=f'{model} (Brier={value})')
        ax.set(xlabel='Mean predicted probability',ylabel='Observed event fraction',xlim=(0,1),ylim=(0,1))
        ax.text(.5,1.03,panel,transform=ax.transAxes,fontweight='bold',ha='center')
        ax.legend(fontsize=7,loc='best')
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/'Figure-5.pdf',dpi=300,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)

def explanations():
    files=[ANALYSIS/f'shap_Overall_Survival_seed{s}.csv' for s in (42,43,44)]
    if not all(f.exists() for f in files):
        raise FileNotFoundError('Three Overall Survival SHAP tables are required')
    matrix=pd.concat([keyed(f, 'feature').mean_absolute_shap for f in files],axis=1)
    require(np.isfinite(matrix.to_numpy()).all() and (matrix.to_numpy() >= 0).all(),
            'SHAP tables must contain matching feature sets and finite nonnegative values')
    mean=matrix.mean(axis=1).sort_values().tail(12)
    sd=matrix.std(axis=1,ddof=1).loc[mean.index]
    fig,ax=plt.subplots(figsize=(8,5))
    ax.barh(mean.index,mean,xerr=sd,color='#264653',capsize=2,error_kw={'linewidth':1})
    ax.set_xlabel('Mean absolute SHAP value (probability scale)')
    ax.set_xlim(left=0)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x',alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/'Figure-4.pdf',dpi=300,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Export vector calibration and SHAP figures.')
    parser.add_argument('--reference-dir',type=Path,default=REF)
    parser.add_argument('--analysis-dir',type=Path,default=ANALYSIS)
    parser.add_argument('--outdir',type=Path,default=OUT)
    args=parser.parse_args()
    REF=args.reference_dir.expanduser().resolve()
    ANALYSIS=args.analysis_dir.expanduser().resolve()
    OUT=args.outdir.expanduser().resolve()
    if (OUT == ROOT or OUT in ROOT.parents
            or any(OUT.is_relative_to(ROOT / name) for name in ('code', 'data', 'tests'))):
        parser.error('Output directory must not be a source directory')
    OUT.mkdir(parents=True,exist_ok=True)
    calibration()
    explanations()
