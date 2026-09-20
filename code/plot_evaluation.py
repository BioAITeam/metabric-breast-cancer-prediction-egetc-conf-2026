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
plt.rcParams.update({
    'pdf.fonttype':42, 'ps.fonttype':42, 'font.family':'DejaVu Sans',
    'font.size':10.5, 'axes.labelsize':10.5, 'xtick.labelsize':10.5,
    'ytick.labelsize':10.5, 'axes.linewidth':0.8,
})

def calibration():
    colors = ['#264653','#a97c13','#457b9d','#b56576','#6a994e']
    markers = ['o','s','^','D','v']
    names = {'Logistic Regression':'LR', 'Random Forest':'RF',
             'SVM (RBF)':'SVM', 'Stacking Ensemble':'Stacking'}
    fig, axes = plt.subplots(1,2,figsize=(7.2,3.2))
    for ax,target,panel in zip(axes,['Overall_Survival','NPI_High_Risk'],['A','B']):
        data = pd.read_csv(REF / f'oof_trad_{target}.csv')
        ax.plot([0,1],[0,1],'--',color='#333333',linewidth=1,zorder=1)
        for model,color,marker in zip(data.columns[2:],colors,markers):
            observed,predicted=calibration_curve(data.y_true,data[model],n_bins=10,strategy='uniform')
            score=brier_score_loss(data.y_true,data[model])
            value=f'{score:.2e}' if 0<score<.001 else f'{score:.3f}'
            ax.plot(predicted,observed,marker=marker,color=color,linewidth=1.3,
                    markersize=3.8,label=f'{names.get(model,model)} ({value})',zorder=2)
        ax.set(xlabel='Mean predicted probability',ylabel='Observed event fraction',xlim=(0,1),ylim=(0,1))
        ax.set_xticks(np.linspace(0,1,6))
        ax.set_yticks(np.linspace(0,1,6))
        ax.text(.5,1.06,panel,transform=ax.transAxes,fontsize=12,
                fontweight='bold',ha='center',va='bottom')
        ax.legend(fontsize=10.5,loc='upper center',bbox_to_anchor=(.5,-.34),
                  ncol=2,title='Brier score',title_fontsize=10.5,frameon=False,
                  handlelength=1.0,handletextpad=.4,columnspacing=.6,
                  labelspacing=.35,borderaxespad=0)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.2,linewidth=.6)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=.075,right=.99,top=.87,bottom=.38,wspace=.29)
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
    fig,ax=plt.subplots(figsize=(5.2,3.25))
    labels=['ln(1 + mutations)' if feature == 'log(Mutation Count)' else feature
            for feature in mean.index]
    ax.barh(labels,mean,xerr=sd,color='#264653',height=.68,
            capsize=2,error_kw={'linewidth':1})
    ax.set_xlabel('Mean absolute SHAP value\n(probability scale)')
    ax.set_xlim(left=0)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x',alpha=.2,linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y',length=0)
    fig.tight_layout(pad=.6)
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
