"""Nested model selection and sensitivity analyses for METABRIC classification."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import metabric_ml_pipeline as base
from evaluation_utils import MODELS, keyed, predictions, require

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "outputs"
OUT = ROOT / "temp" / "evaluation"
SEED = 42
DATA = base.default_data_path()


def save_csv(frame, destination):
    temporary = destination.with_suffix(f'.{os.getpid()}.tmp')
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def file_lock(destination):
    lock = destination.with_suffix('.lock')
    try:
        with lock.open('x') as stream:
            stream.write(str(os.getpid()))
    except FileExistsError as exc:
        raise RuntimeError(f'Lock exists: {lock}. Check for an active process before removing it.') from exc
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def completed_fold(destination, inner_path, ids, y, fold):
    if not destination.exists() and not inner_path.exists():
        return False
    data = keyed(destination, 'Patient ID', ids).loc[ids]
    predictions(data, destination.name)
    require(np.array_equal(data.y_true, y) and (data.outer_fold == fold).all(),
            f'{destination.name}: saved outcomes or folds differ')
    inner = keyed(inner_path, ['model', 'inner_fold'], itertools.product(MODELS, (1, 2, 3)))
    require((inner.outer_fold == fold).all() and inner.AUC.between(0, 1).all(),
            f'{inner_path.name}: invalid inner scores')
    scores = inner.groupby(level='model', sort=False).AUC.mean()
    require(data.model.nunique() == 1 and data.model.iloc[0] in scores.index,
            f'{destination.name}: invalid selected family')
    require(np.isclose(scores[data.model.iloc[0]], scores.max(), rtol=1e-9, atol=1e-12),
            f'{destination.name}: selected family disagrees with inner scores')
    require(data.youden_threshold.nunique() == 1 and data.youden_threshold.between(0, 1).all(),
            f'{destination.name}: invalid threshold')
    require(np.array_equal(data.youden_prediction, (data.y_prob >= data.youden_threshold).astype(int)),
            f'{destination.name}: invalid threshold predictions')
    return True


def models(smote=False):
    """Keep learned preprocessing inside calibration and stacking splits."""
    old = base.build_models(smote)
    def pipe(est):
        steps = [("scale", StandardScaler())]
        if smote:
            steps.append(("smote", SMOTE(random_state=SEED)))
        return Pipeline(steps + [("clf", est)])
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    svm = CalibratedClassifierCV(pipe(SVC(C=1, kernel="rbf", random_state=SEED)), cv=cv)
    estimators = old["Stacking Ensemble"].named_steps["clf"].estimators
    stack = StackingClassifier(
        estimators=[("rf", pipe(clone(estimators[0][1]))),
                    ("xgb", pipe(clone(estimators[1][1]))), ("svm", clone(svm))],
        final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        cv=cv, n_jobs=1)
    return {"Logistic Regression": old["Logistic Regression"],
            "Random Forest": old["Random Forest"], "XGBoost": old["XGBoost"],
            "SVM (RBF)": svm, "Stacking Ensemble": stack}


def cohort(config):
    data = base.load_and_engineer(DATA)
    X, y = base.preprocess(data, base.FEATURE_COLS, config["target_col"], config["exclude_cols"])
    ids = data.loc[X.index, "Patient ID"].to_numpy()
    saved = pd.read_csv(REFERENCE / f"cv_splits_{config['name']}.csv")
    require(saved['Patient ID'].is_unique and np.array_equal(ids, saved["Patient ID"]),
            f"{config['name']}: saved cohort differs from input data")
    require(np.array_equal(y, saved.y_true) and set(saved.outer_fold) == set(range(1, 6)),
            f"{config['name']}: saved outcomes or folds are invalid")
    return data, X, y, ids, saved.outer_fold.to_numpy()


def nested(config):
    name = config["name"]
    data, X, y, ids, folds = cohort(config)
    candidates = models(config["use_smote"])
    for outer in range(1, 6):
        destination = OUT / f"nested_{name}_fold{outer}.csv"
        inner_path = OUT / f"inner_{name}_fold{outer}.csv"
        tr, te = np.where(folds != outer)[0], np.where(folds == outer)[0]
        if completed_fold(destination, inner_path, ids[te], y[te], outer):
            continue
        inner = list(StratifiedKFold(3, shuffle=True, random_state=42 + outer).split(X.iloc[tr], y[tr]))
        scores, inner_prob, records = {}, {}, []
        for label, template in candidates.items():
            prob = np.full(len(tr), np.nan)
            aucs = []
            for k, (a, b) in enumerate(inner, 1):
                estimator = clone(template).fit(X.iloc[tr[a]], y[tr[a]])
                prob[b] = estimator.predict_proba(X.iloc[tr[b]])[:, 1]
                value = roc_auc_score(y[tr[b]], prob[b])
                aucs.append(value)
                records.append(dict(outer_fold=outer, inner_fold=k, model=label, AUC=value))
            scores[label] = float(np.mean(aucs))
            inner_prob[label] = prob
            print(name, outer, label, scores[label], flush=True)
        best = max(scores, key=scores.get)
        fpr, tpr, thresholds = roc_curve(y[tr], inner_prob[best])
        valid = np.isfinite(thresholds)
        threshold = float(thresholds[valid][np.argmax((tpr-fpr)[valid])])
        estimator = clone(candidates[best]).fit(X.iloc[tr], y[tr])
        prob = estimator.predict_proba(X.iloc[te])[:, 1]
        save_csv(pd.DataFrame(records), inner_path)
        save_csv(pd.DataFrame({"Patient ID": ids[te], "y_true": y[te], "outer_fold": outer,
                      "y_prob": prob, "model": best, "youden_threshold": threshold,
                      "youden_prediction": (prob >= threshold).astype(int)}), destination)
        print("OUTER COMPLETE", name, outer, best, roc_auc_score(y[te], prob), flush=True)


def sensitivity(config):
    data, X, y, ids, folds = cohort(config)
    target = config["target_col"]
    cols = [c for c in base.FEATURE_COLS if c != target and c not in config["exclude_cols"]]
    valid = data[target].notna()
    complete = data[cols].notna().all(axis=1) & valid
    rows = []
    for col in ["Age at Diagnosis", "Tumor Size", "Neoplasm Histologic Grade", "Lymph nodes examined positive", target]:
        a, b = data.loc[complete, col].dropna(), data.loc[valid & ~complete, col].dropna()
        den = np.sqrt((a.var(ddof=1) + b.var(ddof=1))/2)
        rows.append(dict(variable=col, included_n=len(a), excluded_n=len(b),
                         included_mean=a.mean(), excluded_mean=b.mean(), smd=(a.mean()-b.mean())/den))
    pd.DataFrame(rows).to_csv(OUT / f"attrition_{config['name']}.csv", index=False)
    numeric = [c for c in cols if c in ["Age at Diagnosis", "Lymph nodes examined positive", "Mutation Count", "Nottingham prognostic index", "Tumor Size", "log_mutation", "tumor_size_cm"]]
    categorical = [c for c in cols if c not in numeric]
    imp = ColumnTransformer([("numeric", SimpleImputer(strategy="median"), numeric),
                             ("categorical", SimpleImputer(strategy="most_frequent"), categorical)])
    steps = [("impute", imp), ("scale", StandardScaler())]
    if config["use_smote"]:
        steps.append(("smote", SMOTE(random_state=SEED)))
    model = Pipeline(steps + [("clf", LogisticRegression(max_iter=1000, random_state=SEED))])
    Xall = data.loc[valid, cols]
    yall = data.loc[valid, target].astype(int).to_numpy()
    prob = np.full(len(yall), np.nan)
    assignment = np.zeros(len(yall), dtype=int)
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xall, yall), 1):
        prob[te] = clone(model).fit(Xall.iloc[tr], yall[tr]).predict_proba(Xall.iloc[te])[:,1]
        assignment[te] = k
    pd.DataFrame({"Patient ID":data.loc[valid,"Patient ID"],"y_true":yall,"y_prob":prob,"outer_fold":assignment}).to_csv(OUT / f"imputation_{config['name']}.csv",index=False)


def explanations(config):
    import shap
    data, X, y, ids, folds = cohort(config)
    name = config["name"]
    label = json.loads((REFERENCE / f"trad_best_{name}_schema.json").read_text())["best_model"]
    for seed in (42, 43, 44):
        dest = OUT / f"shap_{name}_seed{seed}.csv"
        if dest.exists():
            saved = keyed(dest, 'feature', X.columns)
            require(np.isfinite(saved.mean_absolute_shap).all() and (saved.mean_absolute_shap >= 0).all(),
                    f'{dest.name}: invalid saved SHAP values')
            continue
        with file_lock(dest):
            base.seed_everything(seed)
            tr, te = train_test_split(np.arange(len(y)), test_size=.3, stratify=y, random_state=seed)
            model = base.build_models(config["use_smote"])[label].fit(X.iloc[tr], y[tr])
            scale = model.named_steps["scaler"]
            Xtr, Xte = scale.transform(X.iloc[tr]), scale.transform(X.iloc[te])
            clf = model.named_steps["clf"]
            if label == "Stacking Ensemble":
                background = shap.sample(Xtr, 75, random_state=seed)
                explained = shap.sample(Xte, 120, random_state=seed)
                explainer = shap.KernelExplainer(lambda a: clf.predict_proba(a)[:,1], background)
                values = explainer.shap_values(explained, nsamples=160, silent=True)
            elif label == "Logistic Regression":
                values = shap.LinearExplainer(clf, shap.sample(Xtr, 100, random_state=seed)).shap_values(Xte)
            else:
                values = shap.TreeExplainer(clf).shap_values(Xte)
            values = np.asarray(values)
            if values.ndim == 3:
                values = values[:,:,1]
            require(np.isfinite(values).all(), f'{name}/{seed}: nonfinite SHAP values')
            frame = pd.DataFrame({"feature":X.columns,"mean_absolute_shap":np.abs(values).mean(axis=0)})
            save_csv(frame.sort_values("mean_absolute_shap", ascending=False), dest)
            print("SHAP COMPLETE", name, seed, flush=True)


def summarize():
    for config in base.TARGET_CONFIGS:
        name = config['name']
        required = [OUT / f'nested_{name}_fold{fold}.csv' for fold in range(1, 6)]
        required += [OUT / f'inner_{name}_fold{fold}.csv' for fold in range(1, 6)]
        required += [OUT / f'shap_{name}_seed{seed}.csv' for seed in (42, 43, 44)]
        required += [OUT / f'imputation_{name}.csv']
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError('Complete all fitting stages before summarizing: ' + ', '.join(missing))
    summaries, tests, operating, fixed, imputed, shap_rows = [], [], [], [], [], []
    for config in base.TARGET_CONFIGS:
        name = config["name"]
        saved = pd.read_csv(REFERENCE / f"cv_splits_{name}.csv")
        traditional = pd.read_csv(REFERENCE / f"oof_trad_{name}.csv")
        require(saved["Patient ID"].equals(traditional["Patient ID"]), f'{name}: original cohort mismatch')
        folds, y = saved.outer_fold.to_numpy(), saved.y_true.to_numpy()
        for model in traditional.columns[2:]:
            pred = traditional[model].to_numpy()
            aucs = np.array([roc_auc_score(y[folds==f],pred[folds==f]) for f in range(1,6)])
            fixed.append(dict(target=name,model=model,AUC=aucs.mean(),SD=aucs.std(ddof=1),**{k:v for k,v in base.compute_metrics(y,pred).items() if k!='AUC'}))
        for fold in range(1, 6):
            subset = saved.loc[saved.outer_fold == fold]
            completed_fold(OUT / f'nested_{name}_fold{fold}.csv', OUT / f'inner_{name}_fold{fold}.csv',
                           subset['Patient ID'].to_numpy(), subset.y_true.to_numpy(), fold)
        frames=[pd.read_csv(OUT / f"nested_{name}_fold{f}.csv") for f in range(1,6)]
        nested_oof=pd.concat(frames).set_index("Patient ID").loc[saved["Patient ID"]].reset_index()
        nested_oof.to_csv(OUT / f"nested_{name}.csv",index=False)
        methods={"Nested Traditional":nested_oof.y_prob.to_numpy()}
        for family in ("mljar","tabnet"):
            f=pd.read_csv(REFERENCE/f"oof_{family}_{name}.csv").set_index("Patient ID").loc[saved["Patient ID"]]
            require(np.array_equal(f.y_true,y), f'{name}/{family}: outcome mismatch')
            methods[family]=f.y_prob.to_numpy()
        fold_aucs={}
        for method,prob in methods.items():
            aucs=np.array([roc_auc_score(y[folds==f],prob[folds==f]) for f in range(1,6)])
            fold_aucs[method]=aucs
            se=np.sqrt((1/5+1/4)*np.var(aucs,ddof=1))
            half=stats.t.ppf(.975,4)*se
            summaries.append(dict(target=name,method=method,AUC=aucs.mean(),SD=aucs.std(ddof=1),CI_low=max(0,aucs.mean()-half),CI_high=min(1,aucs.mean()+half),**{k:v for k,v in base.compute_metrics(y,prob).items() if k!='AUC'}))
        if name!='NPI_High_Risk':
            for other in ("mljar","tabnet"):
                delta=fold_aucs["Nested Traditional"]-fold_aucs[other]
                se=np.sqrt((.2+.25)*delta.var(ddof=1))
                p=2*stats.t.sf(abs(delta.mean()/se),4) if se else float(delta.mean()==0)
                tests.append(dict(target=name,comparison=other,delta=delta.mean(),p=p))
        for threshold in ("0.5","Youden"):
            prob=nested_oof.y_prob.to_numpy()
            adjusted=prob if threshold=='0.5' else nested_oof.youden_prediction.to_numpy()
            metric=base.compute_metrics(y,adjusted)
            probability_metrics=base.compute_metrics(y,prob)
            for key in ('AUC','AP','Brier'):
                metric[key]=probability_metrics[key]
            operating.append(dict(target=name,threshold=threshold,**metric))
        imp=pd.read_csv(OUT/f"imputation_{name}.csv")
        imp_auc=[roc_auc_score(d.y_true,d.y_prob) for _,d in imp.groupby('outer_fold')]
        imputed.append(dict(target=name,n=len(imp),positive=int(imp.y_true.sum()),AUC=np.mean(imp_auc),SD=np.std(imp_auc,ddof=1)))
        series=[pd.read_csv(OUT/f"shap_{name}_seed{s}.csv").set_index('feature').mean_absolute_shap for s in (42,43,44)]
        matrix=pd.concat(series,axis=1)
        if matrix.isna().any().any():
            raise ValueError(f'Inconsistent SHAP feature sets for {name}')
        correlations=[stats.spearmanr(matrix.iloc[:,a],matrix.iloc[:,b]).statistic for a,b in itertools.combinations(range(3),2)]
        overlaps=[len(set(matrix.iloc[:,a].nlargest(5).index)&set(matrix.iloc[:,b].nlargest(5).index))/5 for a,b in itertools.combinations(range(3),2)]
        shap_rows.append(dict(target=name,top_feature=matrix.mean(axis=1).idxmax(),rho_min=min(correlations),rho_max=max(correlations),top5_overlap=np.mean(overlaps)))
    table=pd.DataFrame(tests)
    order=np.argsort(table.p.to_numpy())
    adjusted=np.minimum(1,np.maximum.accumulate(table.p.to_numpy()[order]*np.arange(len(table),0,-1)))
    table.loc[order,'p_holm']=adjusted
    for filename,rows in [('method_summary',summaries),('fixed_models',fixed),('operating_points',operating),('imputation_summary',imputed),('shap_stability',shap_rows)]:
        pd.DataFrame(rows).to_csv(OUT/f'{filename}.csv',index=False)
    table.to_csv(OUT/'paired_tests.csv',index=False)
    print(pd.DataFrame(summaries)[['target','method','AUC','SD','CI_low','CI_high']].to_string(index=False))
    print(table.to_string(index=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['nested','shap','sensitivity','summarize'])
    parser.add_argument('--targets',nargs='+',choices=[c['name'] for c in base.TARGET_CONFIGS])
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--reference-dir',type=Path,default=REFERENCE)
    parser.add_argument('--outdir',type=Path,default=OUT)
    args=parser.parse_args()
    if args.stage == 'summarize' and args.targets:
        parser.error('--targets applies to fitting stages only; summarize requires all six targets')
    DATA=args.data.expanduser().resolve()
    REFERENCE=args.reference_dir.expanduser().resolve()
    OUT=args.outdir.expanduser().resolve()
    if OUT==REFERENCE:
        parser.error('--outdir must differ from --reference-dir')
    base._assert_replaceable_directory(OUT)
    if not REFERENCE.is_dir():
        parser.error(f'Reference directory not found: {REFERENCE}')
    if args.stage!='summarize' and not DATA.is_file():
        parser.error(f'Data file not found: {DATA}')
    base.seed_everything(SEED)
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='summarize':
        summarize()
    else:
        for config in base.TARGET_CONFIGS:
            if not args.targets or config['name'] in args.targets:
                {'nested':nested,'shap':explanations,'sensitivity':sensitivity}[args.stage](config)
