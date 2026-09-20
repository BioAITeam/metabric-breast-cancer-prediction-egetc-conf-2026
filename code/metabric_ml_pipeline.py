"""Train and evaluate machine-learning models on METABRIC clinical data.

The pipeline preserves missing outcomes, excludes target-derived predictors,
and performs preprocessing within each evaluation fold.
"""

# Imports and configuration
import argparse
import json
import os
import random
import shutil
import sys
import time
import warnings
from pathlib import Path

RANDOM_STATE = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# Logging

class _Tee:
    """Write every print() to both the terminal and a persistent log file."""
    def __init__(self, path: str):
        self._file   = open(path, "w", buffering=1, encoding="utf-8")
        self._stdout = sys.__stdout__

    def write(self, data: str):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass

    def fileno(self):
        return self._stdout.fileno()

    def isatty(self):
        return False

warnings.simplefilter("default")

from sklearn.base            import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import (
    RandomForestClassifier, StackingClassifier
)
from sklearn.svm             import SVC
from sklearn.metrics         import (
    roc_auc_score, roc_curve, confusion_matrix,
    brier_score_loss, average_precision_score, precision_recall_curve,
)
from sklearn.calibration      import CalibratedClassifierCV, calibration_curve

from imblearn.over_sampling   import SMOTE
from imblearn.pipeline        import Pipeline as ImbPipeline

import shap
from xgboost import XGBClassifier

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "code" else SCRIPT_DIR
DATA_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_DATA_PATH = DATA_DIR / "Breast_Cancer_METABRIC.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_WORK_DIR = REPO_ROOT / "temp" / "work"

OUTPUT_DIR = DEFAULT_OUTPUT_DIR
WORK_DIR = DEFAULT_WORK_DIR


def configure_paths(*, output_dir: Path | str | None = None,
                    work_dir: Path | str | None = None) -> None:
    """Configure artifact directories without creating them."""
    global OUTPUT_DIR, WORK_DIR
    if output_dir is not None:
        OUTPUT_DIR = Path(output_dir).expanduser().resolve()
    if work_dir is not None:
        WORK_DIR = Path(work_dir).expanduser().resolve()


def default_data_path() -> Path:
    """Return the organized data path."""
    return DEFAULT_DATA_PATH


def resolve_cli_path(value: str | Path) -> Path:
    """Resolve a command-line path from the current working directory."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _assert_replaceable_directory(path: Path) -> None:
    resolved = path.resolve()
    protected = {
        REPO_ROOT.resolve(),
        SCRIPT_DIR.resolve(),
        DATA_DIR.resolve(),
        Path(resolved.anchor).resolve(),
    }
    source_dirs = [SCRIPT_DIR.resolve(), (REPO_ROOT / 'data').resolve(),
                   (REPO_ROOT / 'tests').resolve()]
    if (resolved in protected or resolved in REPO_ROOT.resolve().parents
            or any(resolved.is_relative_to(p) for p in source_dirs)):
        raise ValueError(f"Refusing to replace protected directory: {resolved}")


def validate_directory_state(*, path: Path, replace: bool,
                             replacement_flag: str) -> None:
    """Validate a generated-artifact directory before any deletion occurs."""
    _assert_replaceable_directory(path)
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")
    if path.exists() and any(path.iterdir()) and not replace:
        raise FileExistsError(
            f"Directory is not empty: {path}. "
            f"Use {replacement_flag} to replace it explicitly.")


def prepare_output_directory(*, replace: bool = False) -> None:
    """Create an output directory without implicit artifact overwrites."""
    if (OUTPUT_DIR / 'method_summary.csv').exists():
        raise ValueError('This directory contains nested-analysis artifacts; use a separate --outdir.')
    validate_directory_state(
        path=OUTPUT_DIR, replace=replace,
        replacement_flag="--replace-outputs")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        _assert_replaceable_directory(OUTPUT_DIR)
        for child in OUTPUT_DIR.iterdir():
            if child.name == "README.md":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_work_directory(*, replace: bool = False) -> None:
    """Create a disposable work directory without implicit deletion."""
    validate_directory_state(
        path=WORK_DIR, replace=replace,
        replacement_flag="--replace-work")
    if WORK_DIR.exists() and any(WORK_DIR.iterdir()):
        _assert_replaceable_directory(WORK_DIR)
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = RANDOM_STATE) -> None:
    """Seed Python, NumPy and, when loaded, PyTorch deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.set_num_threads(1)
        except Exception:
            pass


def save_figure(path, fig=None, *, bbox_inches="tight") -> None:
    """Save a vector PDF with stable metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    target = fig if fig is not None else plt.gcf()
    target.savefig(
        stem.with_suffix(".pdf"),
        format="pdf",
        dpi=300,
        bbox_inches=bbox_inches,
        metadata={"CreationDate": None, "ModDate": None},
    )

PAL = {
    "LR":       "#1f77b4",
    "RF":       "#2ca02c",
    "Boosting": "#ff7f0e",
    "SVM":      "#9467bd",
    "Stacking": "#d62728",
}


# Data loading and feature engineering

def load_and_engineer(filepath: str) -> pd.DataFrame:
    """
    Load METABRIC CSV and engineer all features + binary target labels.

    New binary targets created:
        os_binary        : 1 = Deceased, 0 = Living
        rfs_binary       : 1 = Recurred, 0 = Not Recurred
        npi_high         : 1 = NPI > 5.4 (High Risk), 0 = Low/Moderate
        vital_disease    : 1 = Died of Disease, 0 = Living or Died of Other Causes
        is_luminal       : 1 = LumA or LumB, 0 = other subtype
        her2_pos         : 1 = HER2 Positive, 0 = Negative

    Engineered features:
        ln_pos_flag      : 1 if any lymph nodes positive
        high_grade       : 1 if Neoplasm Histologic Grade == 3
        age_group        : 0=<50, 1=50-65, 2=>65
        npi_group        : 0=Low, 1=Moderate, 2=High (ordinal)
        er_x_her2        : ER x HER2 interaction (both binary)
        log_mutation     : log1p(Mutation Count)
        tumor_size_cm    : Tumor Size / 10  (mm -> cm)
    """
    df = pd.read_csv(filepath)
    print(f"[LOAD] {df.shape[0]} patients x {df.shape[1]} columns")

    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    required = {
        "Patient ID", "Overall Survival Status", "Relapse Free Status",
        "Nottingham prognostic index", "Patient's Vital Status",
        "Pam50 + Claudin-low subtype", "HER2 Status", "ER Status", "PR Status",
        "Chemotherapy", "Hormone Therapy", "Radio Therapy", "Type of Breast Surgery",
        "Inferred Menopausal State", "Lymph nodes examined positive",
        "Neoplasm Histologic Grade", "Mutation Count", "Tumor Size",
        "Age at Diagnosis", "Tumor Stage",
    }
    missing = sorted(required.difference(df.columns))
    if missing or df.empty or not df.columns.is_unique:
        raise ValueError(f"Invalid clinical table; missing fields: {missing}")
    if df["Patient ID"].isna().any() or not df["Patient ID"].is_unique:
        raise ValueError("Patient ID must be present and unique for every row")
    numeric = ["Age at Diagnosis", "Tumor Size", "Tumor Stage", "Mutation Count",
               "Nottingham prognostic index", "Lymph nodes examined positive",
               "Neoplasm Histologic Grade"]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if np.isinf(df[column].to_numpy(dtype=float)).any():
            raise ValueError(f"Nonfinite values in clinical field: {column}")
    if (df["Mutation Count"].dropna() < 0).any():
        raise ValueError("Mutation Count must be nonnegative")

    # Binary targets
    # Explicit mappings keep unrecorded targets and features missing.
    df["os_binary"] = df["Overall Survival Status"].map(
        {"Deceased": 1.0, "Living": 0.0})
    df["rfs_binary"] = df["Relapse Free Status"].map(
        {"Recurred": 1.0, "Not Recurred": 0.0})
    npi = df["Nottingham prognostic index"]
    df["npi_high"] = np.where(npi.isna(), np.nan, (npi > 5.4).astype(float))
    df["vital_disease"] = df["Patient's Vital Status"].map({
        "Died of Disease": 1.0,
        "Living": 0.0,
        "Died of Other Causes": 0.0,
    })

    pam = df["Pam50 + Claudin-low subtype"]
    df["is_luminal"] = np.where(
        pam.isna(), np.nan, pam.isin(["LumA", "LumB"]).astype(float))
    df["her2_pos"] = df["HER2 Status"].map(
        {"Positive": 1.0, "Negative": 0.0})

    # Encode binary categoricals while preserving missingness
    df["er_binary"] = df["ER Status"].map({"Positive": 1.0, "Negative": 0.0})
    df["pr_binary"] = df["PR Status"].map({"Positive": 1.0, "Negative": 0.0})
    df["chemo"] = df["Chemotherapy"].map({"Yes": 1.0, "No": 0.0})
    df["hormone_rx"] = df["Hormone Therapy"].map({"Yes": 1.0, "No": 0.0})
    df["radio_rx"] = df["Radio Therapy"].map({"Yes": 1.0, "No": 0.0})
    df["mastectomy"] = df["Type of Breast Surgery"].map(
        {"Mastectomy": 1.0, "Breast Conserving": 0.0})
    df["postmeno"] = df["Inferred Menopausal State"].map(
        {"Post": 1.0, "Pre": 0.0})

    # Engineered numeric features
    lymph_nodes = df["Lymph nodes examined positive"]
    grade = df["Neoplasm Histologic Grade"]
    df["ln_pos_flag"] = np.where(
        lymph_nodes.isna(), np.nan, (lymph_nodes > 0).astype(float))
    df["high_grade"] = np.where(
        grade.isna(), np.nan, (grade == 3).astype(float))
    df["log_mutation"]        = np.log1p(df["Mutation Count"])
    df["tumor_size_cm"]       = df["Tumor Size"] / 10.0
    df["npi_score"]           = df["Nottingham prognostic index"]

    # NPI group: 0=Low(≤3.4), 1=Moderate(3.4-5.4), 2=High(>5.4)
    df["npi_group"] = np.select(
        [npi <= 3.4, npi <= 5.4], [0.0, 1.0], default=2.0)
    df.loc[npi.isna(), "npi_group"] = np.nan

    # Age groups: 0=<50, 1=50-65, 2=>65
    age = df["Age at Diagnosis"]
    df["age_group"] = np.select(
        [age < 50, age <= 65], [0.0, 1.0], default=2.0)
    df.loc[age.isna(), "age_group"] = np.nan
    
    # ER x HER2 interaction    
    df["er_x_her2"] = df["er_binary"] * df["her2_pos"]

    print(f"[ENGINEER] Targets + engineered features ready.")
    print(f"  os_binary:     {df['os_binary'].sum():.0f} / {df['os_binary'].notna().sum()} recorded Deceased")
    print(f"  rfs_binary:    {df['rfs_binary'].sum():.0f} / {df['rfs_binary'].notna().sum()} recorded Recurred")
    print(f"  npi_high:      {df['npi_high'].sum()} High-NPI patients")
    print(f"  vital_disease: {df['vital_disease'].sum()} Died of Disease")
    print(f"  is_luminal:    {df['is_luminal'].sum()} Luminal subtypes")
    print(f"  her2_pos:      {df['her2_pos'].sum()} HER2+ patients")
    return df


FEATURE_COLS = [
    "Age at Diagnosis", 
    "Neoplasm Histologic Grade",
    "Lymph nodes examined positive", 
    "Mutation Count",
    "Nottingham prognostic index", 
    "Tumor Size", 
    "Tumor Stage",
    "er_binary",
    "pr_binary", 
    "her2_pos",
    "chemo", 
    "hormone_rx", 
    "radio_rx", 
    "mastectomy", 
    "postmeno",
    "ln_pos_flag", 
    "high_grade", 
    "log_mutation",
    "tumor_size_cm", 
    "npi_group", 
    "age_group", 
    "er_x_her2",
]

FEATURE_LABELS = {
    "Age at Diagnosis":              "Age (yrs)",
    "Neoplasm Histologic Grade":     "Histologic Grade",
    "Lymph nodes examined positive": "LN Positive (count)",
    "Mutation Count":                "Mutation Count",
    "Nottingham prognostic index":   "NPI Score",
    "Tumor Size":                    "Tumor Size (mm)",
    "Tumor Stage":                   "Tumor Stage",
    "er_binary":                     "ER Status",
    "pr_binary":                     "PR Status",
    "her2_pos":                      "HER2 Status",
    "chemo":                         "Chemotherapy",
    "hormone_rx":                    "Hormone Therapy",
    "radio_rx":                      "Radio Therapy",
    "mastectomy":                    "Mastectomy",
    "postmeno":                      "Post-Menopausal",
    "ln_pos_flag":                   "LN Positive (flag)",
    "high_grade":                    "High Grade (G3)",
    "log_mutation":                  "log(Mutation Count)",
    "tumor_size_cm":                 "Tumor Size (cm)",
    "npi_group":                     "NPI Group",
    "age_group":                     "Age Group",
    "er_x_her2":                     "ER x HER2",
}

TARGET_CONFIGS = [
    {"target_col": "os_binary",     "name": "Overall_Survival",
     "exclude_cols": [], "use_smote": False},
    {"target_col": "rfs_binary",    "name": "Relapse_Free",
     "exclude_cols": [], "use_smote": False},
    {"target_col": "npi_high",      "name": "NPI_High_Risk",
     # NPI score and group directly determine this outcome and are excluded.
     "exclude_cols": ["Nottingham prognostic index", "npi_group"],
     "use_smote": True},
    {"target_col": "vital_disease", "name": "Vital_Disease",
     "exclude_cols": [], "use_smote": False},
    {"target_col": "is_luminal",    "name": "Luminal_Subtype",
     "exclude_cols": [], "use_smote": False},
    {"target_col": "her2_pos",      "name": "HER2_Positive",
     # er_x_her2 = er_binary * her2_pos and therefore contains the label.
     "exclude_cols": ["er_x_her2"], "use_smote": True},
]

def preprocess(df, feature_cols, target_col, exclude_cols=None):
    """
    Select complete cases and return a numeric feature matrix and outcome.

    Outcome-specific exclusions:
      1. target_col is always excluded from features (e.g. her2_pos appearing
         in both FEATURE_COLS and as a target).
      2. exclude_cols allows per-target exclusion of columns that mathematically
         encode the target (e.g. npi_group and Nottingham prognostic index
         when the target is npi_high).
    """
    if exclude_cols is None:
        exclude_cols = []

    all_excluded = set([target_col] + exclude_cols)

    if exclude_cols:
        print(f"  [PREP] Excluding leaky cols for '{target_col}': {exclude_cols}")

    valid_cols = [c for c in feature_cols if c not in all_excluded]
    missing = sorted(set(valid_cols + [target_col]).difference(df.columns))
    if missing:
        raise ValueError(f"Missing modeling columns: {missing}")
    sub = df[valid_cols + [target_col]].dropna().copy()
    if sub.empty or set(sub[target_col]) != {0, 1}:
        raise ValueError(f"Target {target_col} requires complete cases in both binary classes")
    X   = sub[valid_cols].copy()
    y   = sub[target_col].astype(int).values

    non_numeric = list(X.select_dtypes(exclude=[np.number]).columns)
    if non_numeric:
        raise TypeError(
            f"Non-numeric modeling columns require an explicit encoder: "
            f"{non_numeric}")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Modeling predictors must contain finite numeric values")

    X.columns = [FEATURE_LABELS.get(c, c) for c in X.columns]
    pos = int(y.sum()); n = len(y)
    print(f"  [PREP] '{target_col}': {pos}/{n} positive ({100*pos/n:.1f}%) - {n} complete cases, {len(valid_cols)} features")
    return X, y


# Descriptive analysis

def _absolute_cohen_d(negative: pd.Series, positive: pd.Series) -> float:
    """Absolute Cohen's d with the unweighted two-group SD convention."""
    negative = pd.to_numeric(negative, errors="coerce").dropna()
    positive = pd.to_numeric(positive, errors="coerce").dropna()
    denominator = np.sqrt((negative.var(ddof=1) + positive.var(ddof=1)) / 2)
    return float(abs(positive.mean() - negative.mean()) / denominator)


def generate_analysis_figures(df: pd.DataFrame) -> None:
    """Generate descriptive figures and marginal-statistics tables."""
    print("[FIGURES] Generating descriptive figures ...")
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    # Figure 1: missingness in the 34 source columns.
    raw_cols = list(df.columns[:34])
    missing_frame = df[raw_cols].copy()
    missing_pct = (100 * missing_frame.isna().mean())
    missing_pct = missing_pct[missing_pct > 0].sort_values(
        ascending=True, kind="stable")
    colors = np.where(missing_pct > 20, "#d62728",
                      np.where(missing_pct > 5, "#ff7f0e", "#1f77b4"))
    missing_labels = {
        "Cancer Type Detailed": "Detailed cancer type",
        "Cellularity": "Cellularity",
        "ER status measured by IHC": "ER status (IHC)",
        "HER2 status measured by SNP6": "HER2 status (SNP6)",
        "Neoplasm Histologic Grade": "Histologic grade",
        "Inferred Menopausal State": "Menopausal state",
        "Lymph nodes examined positive": "Positive lymph nodes",
        "Nottingham prognostic index": "Nottingham prognostic index",
        "Overall Survival (Months)": "Overall survival (months)",
        "Overall Survival Status": "Overall survival status",
        "Pam50 + Claudin-low subtype": "PAM50 + Claudin-low subtype",
        "Patient's Vital Status": "Vital status",
        "Primary Tumor Laterality": "Primary tumor laterality",
        "Relapse Free Status (Months)": "Relapse-free survival (months)",
        "Relapse Free Status": "Relapse-free status",
        "Type of Breast Surgery": "Breast surgery",
        "Tumor Other Histologic Subtype": "Other histologic subtype",
        "3-Gene classifier subtype": "3-gene classifier subtype",
    }
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    ax.barh(missing_pct.index, missing_pct.values, color=colors)
    ax.set_yticks(range(len(missing_pct)),
                  [missing_labels.get(c, c) for c in missing_pct.index])
    ax.tick_params(axis="y", labelsize=10.5, length=0, pad=4)
    ax.tick_params(axis="x", labelsize=10.5)
    ax.margins(y=0.015)
    ax.axvline(5, color="#ff7f0e", ls="--", lw=1.1,
               label="5% reference")
    ax.axvline(20, color="#d62728", ls="--", lw=1.1,
               label="20% reference")
    ax.set_xlabel("Missing (%)", fontsize=11)
    ax.legend(loc="lower right", fontsize=10.5, frameon=False)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.35)
    save_figure(OUTPUT_DIR / "Figure-1", fig=fig)
    plt.close(fig)

    # Figure 2: target-valid class counts. Unknown outcomes are not class 0.
    target_labels = [
        ("os_binary", "Overall Survival", "Living", "Deceased"),
        ("rfs_binary", "Recurrence Event", "Not Recurred", "Recurred"),
        ("npi_high", "NPI High Risk", "Low/Moderate", "High Risk"),
        ("vital_disease", "Disease-specific Death", "Living/Other", "Died of Disease"),
        ("is_luminal", "Luminal Subtype", "Non-Luminal", "Luminal"),
        ("her2_pos", "HER2 Status", "HER2 Negative", "HER2 Positive"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.15), sharey=False)
    for panel, (ax, (col, _, neg_label, pos_label)) in enumerate(zip(axes.flat, target_labels)):
        observed = df[col].dropna().astype(int)
        counts = observed.value_counts().reindex([0, 1], fill_value=0)
        bars = ax.bar([0, 1], counts.values,
                      color=["#1f77b4", "#d62728"], alpha=0.85)
        wrapped_labels = [label.replace("/", "/\n") if "/" in label
                          else label.replace(" ", "\n", 1)
                          if len(label) > 10 else label
                          for label in (neg_label, pos_label)]
        ax.set_xticks([0, 1], wrapped_labels)
        ax.tick_params(axis="x", labelsize=10.5, length=0, pad=4)
        ax.tick_params(axis="y", labelsize=10.5)
        ax.set_yticks([0, 1000, 2000])
        ax.text(0.5, 1.03, chr(65 + panel), transform=ax.transAxes,
                fontsize=12, fontweight="bold", ha="center")
        ax.set_ylabel("Recorded patients" if panel % 3 == 0 else "",
                      fontsize=10.5)
        for bar, value in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 35,
                    f"n={value:,}\n{100 * value / counts.sum():.1f}%",
                    ha="center", va="bottom", fontsize=10.5)
        ax.set_ylim(0, 2900)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=0.45, w_pad=0.7, h_pad=1.3)
    save_figure(OUTPUT_DIR / "Figure-2", fig=fig)
    plt.close(fig)

    # Figure 3: the 22-feature complete-case correlation matrix.
    correlation_data = df[FEATURE_COLS].dropna().copy()
    correlation_data.columns = [
        FEATURE_LABELS.get(c, c) for c in correlation_data.columns]
    correlation = correlation_data.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    correlation_labels = {
        "Age (yrs)": "Age (years)",
        "Histologic Grade": "Histologic grade",
        "LN Positive (count)": "Positive nodes",
        "Mutation Count": "Mutation count",
        "NPI Score": "NPI score",
        "Tumor Size (mm)": "Tumor size (mm)",
        "Tumor Stage": "Tumor stage",
        "Post-Menopausal": "Postmenopausal",
        "LN Positive (flag)": "Node-positive",
        "High Grade (G3)": "Grade 3",
        "log(Mutation Count)": "ln(1 + mutations)",
        "Tumor Size (cm)": "Tumor size (cm)",
        "NPI Group": "NPI group",
        "Age Group": "Age group",
    }
    labels = [correlation_labels.get(c, c) for c in correlation.columns]
    annotations = correlation.map(
        lambda value: f"{value:.2f}".replace("-0.", "-.").replace("0.", "."))
    annotations = annotations.replace("-.00", ".00")
    fig = plt.figure(figsize=(7.4, 6.6))
    ax = fig.add_axes([0.21, 0.21, 0.775, 0.775])
    cbar_ax = ax.inset_axes([0.59, 0.90, 0.36, 0.025])
    sns.heatmap(correlation, mask=mask, annot=annotations, fmt="",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.25,
                annot_kws={"size": 8.4, "family": "DejaVu Sans",
                           "fontstretch": "condensed"}, ax=ax,
                cbar_ax=cbar_ax,
                cbar_kws={"label": "Pearson r", "orientation": "horizontal",
                          "ticks": [-1, 0, 1]})
    for annotation, value in zip(ax.texts, correlation.to_numpy()[~mask]):
        annotation.set_color("white" if abs(value) >= 0.65 else "#171717")
    # Matplotlib rasterizes continuous colorbar strips by default in some
    # releases; force its solids to remain vector paths in the PDF artifact.
    colorbar = ax.collections[0].colorbar
    if colorbar is not None and colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    cbar_ax.tick_params(labelsize=10, length=3, pad=2)
    cbar_ax.set_xlabel("Pearson r", fontsize=11, labelpad=3)
    ax.set_xticklabels(labels, rotation=65, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels, rotation=0)
    ax.tick_params(axis="x", labelsize=10, length=0, pad=4)
    ax.tick_params(axis="y", labelsize=10, length=0, pad=4)
    save_figure(OUTPUT_DIR / "Figure-3", fig=fig)
    plt.close(fig)

    # NPI target-valid, pairwise-complete numerical distributions.
    numerical_features = [
        "Age at Diagnosis", "Tumor Size", "Nottingham prognostic index",
        "Mutation Count", "Lymph nodes examined positive"]
    npi_valid = df[df["npi_high"].notna()].copy()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8))
    for ax, feature in zip(axes.flat, numerical_features):
        pair = npi_valid[[feature, "npi_high"]].dropna()
        negative = pair.loc[pair["npi_high"] == 0, feature]
        positive = pair.loc[pair["npi_high"] == 1, feature]
        effect = _absolute_cohen_d(negative, positive)
        bin_edges = np.histogram_bin_edges(pair[feature], bins=30)
        ax.hist(negative, bins=bin_edges, density=True, alpha=0.38,
                color="#1f77b4", label=f"Low/Moderate (n={len(negative)})")
        ax.hist(positive, bins=bin_edges, density=True, alpha=0.38,
                color="#d62728", label=f"High (n={len(positive)})")
        if len(negative) > 2:
            negative.plot.kde(ax=ax, color="#1f77b4", lw=1.3,
                              label="_nolegend_")
        if len(positive) > 2:
            positive.plot.kde(ax=ax, color="#d62728", lw=1.3,
                              label="_nolegend_")
        ax.axvline(negative.mean(), color="#1f77b4", ls="--", lw=1)
        ax.axvline(positive.mean(), color="#d62728", ls="--", lw=1)
        ax.set_xlabel(FEATURE_LABELS.get(feature, feature))
        ax.text(0.97, 0.97, f"|d|={effect:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8)
        ax.set_ylabel("Density")
        ax.set_xlim(left=0)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    save_figure(WORK_DIR / "figures" / "npi_distributions", fig=fig)
    plt.close(fig)

    # Export marginal statistics for every target.
    target_columns = {config["name"]: config["target_col"]
                      for config in TARGET_CONFIGS}
    numeric_records = []
    binary_records = []
    binary_features = [
        "er_binary", "pr_binary", "her2_pos", "chemo", "hormone_rx",
        "radio_rx", "mastectomy", "postmeno", "ln_pos_flag", "high_grade"]
    for target_name, target_col in target_columns.items():
        for feature in numerical_features:
            pair = df[[feature, target_col]].dropna()
            negative = pair.loc[pair[target_col] == 0, feature]
            positive = pair.loc[pair[target_col] == 1, feature]
            numeric_records.append({
                "Target": target_name, "Feature": feature,
                "N_negative": len(negative), "N_positive": len(positive),
                "Mean_negative": negative.mean(),
                "Mean_positive": positive.mean(),
                "Absolute_Cohen_d": _absolute_cohen_d(negative, positive),
            })
        for feature in binary_features:
            if feature == target_col:
                continue
            pair = df[[feature, target_col]].dropna()
            phi = abs(pair[feature].corr(pair[target_col]))
            binary_records.append({
                "Target": target_name,
                "Feature": FEATURE_LABELS.get(feature, feature),
                "N": len(pair), "Phi_absolute": phi,
                "Rate_negative_target": pair.loc[pair[target_col] == 0, feature].mean(),
                "Rate_positive_target": pair.loc[pair[target_col] == 1, feature].mean(),
            })
    pd.DataFrame(numeric_records).to_csv(
        OUTPUT_DIR / "marginal_numeric_statistics.csv", index=False)
    pd.DataFrame(binary_records).to_csv(
        OUTPUT_DIR / "marginal_binary_statistics.csv", index=False)


# Model definitions

def build_models(use_smote: bool = False) -> dict:
    inner_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    xgb_clf = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=1)
    stacking_estimators = [
        ("rf",  RandomForestClassifier(n_estimators=200, max_depth=6,
                                        random_state=RANDOM_STATE, n_jobs=1)),
        ("xgb", xgb_clf),
        ("svm", CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE), cv=inner_cv)),
    ]
    stacking_clf = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=inner_cv, passthrough=False, n_jobs=1,
    )
    def pipe(clf):
        steps = [("scaler", StandardScaler())]
        if use_smote:
            # Distance-based SMOTE must operate in the scaled feature space.
            steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
        steps.append(("clf", clf))
        return ImbPipeline(steps)

    return {
        "Logistic Regression": pipe(
            LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
        "Random Forest": pipe(
            RandomForestClassifier(n_estimators=500, max_depth=8,
                                   min_samples_leaf=5, random_state=RANDOM_STATE,
                                   n_jobs=1)),
        "XGBoost": pipe(xgb_clf),
        "SVM (RBF)": pipe(
            CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, gamma="scale",
                                       random_state=RANDOM_STATE), cv=inner_cv)),
        "Stacking Ensemble": pipe(stacking_clf),
    }


# Evaluation metrics

def compute_metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        "AUC":         roc_auc_score(y_true, y_prob),
        "AP":          average_precision_score(y_true, y_prob),
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV":         ppv,
        "NPV":         npv,
        "Brier":       brier_score_loss(y_true, y_prob),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def cross_validate_models(
        models, X, y, n_splits=5
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(X, y))
    results = []
    oof_probabilities = {}
    for name, pipeline in models.items():
        print(f"    [CV] {name} ...", end=" ", flush=True)
        try:
            y_prob = np.full(len(y), np.nan, dtype=float)
            fold_aucs = []
            for fold_idx, (tr_idx, te_idx) in enumerate(splits):
                seed_everything(RANDOM_STATE + fold_idx)
                estimator = clone(pipeline)
                estimator.fit(X.iloc[tr_idx], y[tr_idx])
                fold_prob = estimator.predict_proba(X.iloc[te_idx])[:, 1]
                y_prob[te_idx] = fold_prob
                fold_aucs.append(roc_auc_score(y[te_idx], fold_prob))
            m = compute_metrics(y, y_prob)
            m["AUC_pooled"] = m["AUC"]
            m["AUC"] = float(np.mean(fold_aucs))
            m["AUC_std"] = float(np.std(fold_aucs))
            m["Model"] = name
            results.append(m)
            oof_probabilities[name] = y_prob
            print(f"AUC={m['AUC']:.3f}±{m['AUC_std']:.3f}  "
                  f"Sens={m['Sensitivity']:.3f}  NPV={m['NPV']:.3f}")
        except Exception as e:
            raise RuntimeError(f"Cross-validation failed for {name}") from e
    if not results:
        raise RuntimeError("All models failed. Check target column type and class balance.")
    return pd.DataFrame(results).set_index("Model"), oof_probabilities


# Roc and precision-recall curves

def plot_roc_pr(models, X, y, target_name, output_dir, oof_probabilities=None):
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    colors = list(PAL.values())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for (name, pipeline), col in zip(models.items(), colors):
        try:
            y_prob = (oof_probabilities.get(name)
                      if oof_probabilities is not None
                      else cross_val_predict(
                          pipeline, X, y, cv=cv,
                          method="predict_proba")[:, 1])
            fpr, tpr, _ = roc_curve(y, y_prob)
            auc_val = roc_auc_score(y, y_prob)
            axes[0].plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC={auc_val:.3f})")
            prec, rec, _ = precision_recall_curve(y, y_prob)
            ap_val = average_precision_score(y, y_prob)
            axes[1].plot(rec, prec, color=col, lw=2, label=f"{name} (AP={ap_val:.3f})")
        except Exception:
            pass

    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("1 - Specificity (FPR)"); axes[0].set_ylabel("Sensitivity (TPR)")
    axes[0].legend(loc="lower right", fontsize=8); axes[0].grid(alpha=0.3)

    base_rate = y.mean()
    axes[1].axhline(base_rate, color="navy", ls="--", lw=1.2, label=f"Baseline ({base_rate:.2f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right", fontsize=8); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fname = Path(output_dir) / f"roc_pr_{target_name}"
    save_figure(fname); plt.close()


# Calibration curves

def plot_calibration(models, X, y, target_name, output_dir,
                     oof_probabilities=None):
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect", lw=1.5)
    for (name, pipeline), col in zip(models.items(), PAL.values()):
        try:
            y_prob = (oof_probabilities.get(name)
                      if oof_probabilities is not None
                      else cross_val_predict(
                          pipeline, X, y, cv=cv,
                          method="predict_proba")[:, 1])
            fp, mp = calibration_curve(y, y_prob, n_bins=10)
            ax.plot(mp, fp, "s-", color=col, lw=2, label=name)
        except Exception:
            pass
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.tight_layout()
    fname = Path(output_dir) / f"calibration_{target_name}"
    save_figure(fname); plt.close()


# Shap explainability

def generate_shap_importance_figure(importance: pd.DataFrame) -> None:
    """Generate a diagnostic figure from a validated SHAP importance table."""
    required = {"feature", "mean_absolute_shap"}
    if not required.issubset(importance.columns) or importance.empty:
        raise ValueError("Invalid SHAP importance table")
    ordered = importance.sort_values(
        "mean_absolute_shap", ascending=True, kind="stable")
    fig_height = max(5.5, 0.30 * len(ordered))
    fig, ax = plt.subplots(figsize=(9.0, fig_height))
    ax.barh(
        ordered["feature"], ordered["mean_absolute_shap"],
        color="#1f77b4", edgecolor="white")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(WORK_DIR / "figures" / "shap_importance", fig=fig)
    plt.close(fig)

def compute_shap(pipeline, X_train, X_test, feature_names, model_name,
                 target_name, diagnostics_dir):
    clf    = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps.get("scaler", None)
    Xtr    = pd.DataFrame(scaler.transform(X_train) if scaler else X_train.values,
                           columns=feature_names)
    Xts    = pd.DataFrame(scaler.transform(X_test) if scaler else X_test.values,
                           columns=feature_names)
    print(f"    [SHAP] {model_name} / {target_name} ...")

    try:
        if isinstance(clf, (RandomForestClassifier, XGBClassifier)):
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(Xts)
            if isinstance(shap_vals, list): shap_vals = shap_vals[1]
        elif isinstance(clf, LogisticRegression):
            background = shap.sample(
                Xtr, min(100, len(Xtr)), random_state=RANDOM_STATE)
            explainer = shap.LinearExplainer(clf, background)
            shap_vals = explainer.shap_values(Xts)
        elif isinstance(clf, StackingClassifier):
            bg = shap.sample(Xtr, min(50, len(Xtr)), random_state=RANDOM_STATE)
            explainer = shap.KernelExplainer(lambda d: clf.predict_proba(d)[:, 1], bg)
            shap_vals = explainer.shap_values(Xts[:80], nsamples=80)
            Xts = Xts[:80]
        else:
            bg = shap.sample(Xtr, min(50, len(Xtr)), random_state=RANDOM_STATE)
            explainer = shap.KernelExplainer(lambda d: clf.predict_proba(d)[:, 1], bg)
            shap_vals = explainer.shap_values(Xts[:80], nsamples=80)
            Xts = Xts[:80]
    except Exception as e:
        print(f"    [SHAP] Failed: {e}"); return None, None

    # SHAP >=0.45 may return binary-class explanations as
    # (samples, features, classes) instead of the legacy list-of-classes.
    shap_vals = np.asarray(shap_vals)
    if shap_vals.ndim == 3 and shap_vals.shape[-1] == 2:
        shap_vals = shap_vals[:, :, 1]
    elif shap_vals.ndim == 3 and shap_vals.shape[0] == 2:
        shap_vals = shap_vals[1]
    if shap_vals.ndim != 2:
        print(f"    [SHAP] Unsupported explanation shape: {shap_vals.shape}")
        return None, None

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": np.abs(shap_vals).mean(axis=0),
    }).sort_values("mean_absolute_shap", ascending=False)
    importance.to_csv(
        OUTPUT_DIR
        / f"shap_importance_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_{target_name}.csv",
        index=False)

    safe = model_name.replace(" ", "_").replace("(", "").replace(")", "")

    # Plot 1: Beeswarm
    fig, _ = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals, Xts, feature_names=feature_names, show=False)
    plt.tight_layout()
    save_figure(Path(diagnostics_dir) / f"shap_summary_{safe}_{target_name}"); plt.close()

    # Plot 2: Bar importance
    fig, _ = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_vals, Xts, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.tight_layout()
    save_figure(Path(diagnostics_dir) / f"shap_bar_{safe}_{target_name}")
    plt.close(fig)
    if target_name == "Overall_Survival":
        generate_shap_importance_figure(importance)

    # Plot 3: Dependence - top feature
    top_idx  = int(np.argmax(np.abs(shap_vals).mean(axis=0)))
    top_feat = feature_names[top_idx]
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.dependence_plot(
            top_idx, shap_vals, Xts, feature_names=feature_names,
            interaction_index=None, show=False, ax=ax)
        plt.tight_layout()
        save_figure(Path(diagnostics_dir) /
                    f"shap_dep_{top_feat.replace(' ','_')}_{safe}_{target_name}")
        plt.close()
    except Exception as exc:
        plt.close("all")
        print(f"    [SHAP] Dependence plot skipped: {exc}")

    # Plot 4 & 5: Waterfall for high-risk and low-risk patient
    for label, idx in [
        ("high_risk", int(np.argmax(shap_vals.sum(axis=1)))),
        ("low_risk",  int(np.argmin(shap_vals.sum(axis=1)))),
    ]:
        try:
            base = (explainer.expected_value if not isinstance(explainer.expected_value, list)
                    else explainer.expected_value[1])
            sv = shap.Explanation(
                values=shap_vals[idx], base_values=base,
                data=Xts.iloc[idx].values, feature_names=feature_names)
            fig, _ = plt.subplots(figsize=(9, 5))
            shap.waterfall_plot(sv, show=False)
            plt.tight_layout()
            save_figure(Path(diagnostics_dir) /
                        f"shap_waterfall_{label}_{safe}_{target_name}"); plt.close()
        except Exception:
            pass

    print(f"    [SHAP] All plots saved for {model_name} / {target_name}")
    return shap_vals, explainer


# Subgroup analysis

def subgroup_analysis(y_prob, X, y, df_sub, target_name, output_dir):
    """Evaluate clinically relevant subgroups with held-out OOF predictions."""
    subgroups = {
        "LN Negative":   df_sub["Lymph nodes examined positive"] == 0,
        "LN Positive":   df_sub["Lymph nodes examined positive"] > 0,
        "Age < 50":      df_sub["Age at Diagnosis"] < 50,
        "Age ≥ 65":      df_sub["Age at Diagnosis"] >= 65,
        "Grade 1-2":     df_sub["Neoplasm Histologic Grade"] <= 2,
        "Grade 3":       df_sub["Neoplasm Histologic Grade"] == 3,
        "ER Positive":   df_sub["er_binary"] == 1,
        "ER Negative":   df_sub["er_binary"] == 0,
        "HER2 Positive": df_sub["her2_pos"] == 1,
        "NPI Low":       df_sub["npi_group"] == 0,
        "NPI High":      df_sub["npi_group"] == 2,
    }
    records = []
    for group_name, mask in subgroups.items():
        idx = mask[mask].index.intersection(X.index)
        if len(idx) < 30: continue
        positions = X.index.get_indexer(idx)
        y_sub = y[positions]
        prob_sub = y_prob[positions]
        if y_sub.sum() < 5 or y_sub.sum() == len(y_sub): continue
        try:
            m = compute_metrics(y_sub, prob_sub)
            m["Subgroup"] = group_name; m["N"] = len(y_sub); m["Events"] = int(y_sub.sum())
            records.append(m)
        except Exception as e:
            print(f"  [SUBGROUP] {group_name}: {e}")

    if not records: return pd.DataFrame()
    df_sg = pd.DataFrame(records).set_index("Subgroup")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in [
        (axes[0], "AUC"),
        (axes[1], "Sensitivity"),
    ]:
        if metric not in df_sg.columns: continue
        colors_sg = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df_sg)))
        bars = ax.barh(range(len(df_sg)), df_sg[metric], color=colors_sg, edgecolor="white")
        ax.axvline(0.5, color="red", ls="--", lw=1.5)
        ax.set_yticks(range(len(df_sg))); ax.set_yticklabels(df_sg.index, fontsize=9)
        ax.set_xlabel(metric)
        for b, v in zip(bars, df_sg[metric]):
            ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    save_figure(Path(output_dir) / f"subgroup_{target_name}")
    plt.close()
    return df_sg


# Traditional ml pipeline

def run_traditional_pipeline(df: pd.DataFrame, run_shap=True, run_subgroup=True):
    print("\n" + "=" * 62)
    print("  PHASE 2 - TRADITIONAL ML PIPELINE")
    print("=" * 62)

    all_results = {}
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    diagnostics_dir = WORK_DIR / "diagnostics" / "traditional"
    model_dir = WORK_DIR / "models" / "traditional"

    for config in TARGET_CONFIGS:
        target_col  = config["target_col"]
        target_name = config["name"]
        use_smote = config.get("use_smote", False)
        models = build_models(use_smote=use_smote)

        print(f"\n{'='*62}\n  TARGET: {target_name}  ({target_col})\n{'='*62}")
        X, y = preprocess(df, feat_cols, target_col,
                          exclude_cols=config.get("exclude_cols", []))
        feature_names = list(X.columns)
        print(f"  [PREP] SMOTE: {'training folds only' if use_smote else 'not used'}")
        fold_assignment = np.zeros(len(y), dtype=int)
        split_cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        for fold_idx, (_, te_idx) in enumerate(split_cv.split(X, y), start=1):
            fold_assignment[te_idx] = fold_idx
        pd.DataFrame({
            "Patient ID": df.loc[X.index, "Patient ID"].values,
            "y_true": y,
            "outer_fold": fold_assignment,
        }).to_csv(OUTPUT_DIR / f"cv_splits_{target_name}.csv", index=False)

        print("\n  [1/4] Cross-validated evaluation ...")
        cv_results, oof_probabilities = cross_validate_models(models, X, y)
        all_results[target_name] = cv_results
        print(cv_results[["AUC", "NPV", "PPV", "Sensitivity", "Specificity", "Brier"]].round(3).to_string())
        cv_results.to_csv(f"{OUTPUT_DIR}/cv_trad_{target_name}.csv")
        oof_df = pd.DataFrame({
            "Patient ID": df.loc[X.index, "Patient ID"].values,
            "y_true": y,
            **oof_probabilities,
        })
        oof_df.to_csv(OUTPUT_DIR / f"oof_trad_{target_name}.csv", index=False)

        print(f"\n  [2/4] ROC + PR curves ...")
        plot_roc_pr(
            models, X, y, target_name, diagnostics_dir, oof_probabilities)

        print(f"  [3/4] Calibration curves ...")
        plot_calibration(
            models, X, y, target_name, diagnostics_dir, oof_probabilities)

        best_name     = cv_results["AUC"].idxmax()
        best_pipeline = models[best_name]
        print(f"\n  Best model: {best_name}  (AUC={cv_results.loc[best_name,'AUC']:.3f})")

        X_tr, X_te, y_tr, _ = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
        best_pipeline.fit(X_tr, y_tr)

        print(f"  [4/4] SHAP ...")
        if run_shap:
            shap_values, _ = compute_shap(
                best_pipeline, X_tr, X_te, feature_names,
                best_name, target_name, diagnostics_dir)
            if shap_values is None:
                raise RuntimeError(
                    f"SHAP generation failed for {target_name}")

        if run_subgroup:
            sub_cols = ["Lymph nodes examined positive", "Age at Diagnosis",
                        "Neoplasm Histologic Grade", "er_binary", "her2_pos", "npi_group"]
            sub_df = df.loc[X.index, [c for c in sub_cols if c in df.columns]]
            subgroup_analysis(oof_probabilities[best_name], X, y, sub_df,
                              target_name, diagnostics_dir)

        # Explicit final refit after all held-out evaluation is complete.
        seed_everything(RANDOM_STATE)
        final_pipeline = clone(models[best_name]).fit(X, y)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_pipeline, model_dir / f"trad_best_{target_name}.pkl")
        with open(OUTPUT_DIR / f"trad_best_{target_name}_schema.json", "w",
                  encoding="utf-8") as schema_file:
            json.dump({
                "target": target_name,
                "best_model": best_name,
                "features": feature_names,
                "n": len(y),
                "positives": int(y.sum()),
                "seed": RANDOM_STATE,
                "smote_training_only": use_smote,
            }, schema_file, indent=2)

    return all_results


# Mljar automl

def run_mljar(df: pd.DataFrame, n_splits: int = 5,
              fold_time_limit: int = 300, quick: bool = False):
    print("\n" + "=" * 62)
    print("  PHASE 4: MLJAR AutoML (5-fold stratified CV)")
    print("=" * 62)

    if quick:
        print("  [MLJAR] quick mode: single 80/20 split")

    try:
        from supervised.automl import AutoML
    except ImportError as exc:
        raise RuntimeError(
            "MLJAR is unavailable; install the pinned requirements") from exc

    feat_cols     = [c for c in FEATURE_COLS if c in df.columns]
    mljar_results = {}
    cv            = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                    random_state=RANDOM_STATE)

    for config in TARGET_CONFIGS:
        target_col  = config["target_col"]
        target_name = config["name"]

        print(f"\n  [MLJAR] Target: {target_name}  ({target_col})")

        X, y = preprocess(df, feat_cols, target_col,
                          exclude_cols=config.get("exclude_cols", []))

        fold_aucs, fold_probs, fold_trues = [], [], []
        oof_prob = np.full(len(y), np.nan, dtype=float)
        total_t0 = time.time()

        if quick:
            splits = [(train_test_split(
                np.arange(len(y)), test_size=0.2,
                stratify=y, random_state=RANDOM_STATE
            ))]
        else:
            splits = cv.split(X, y)

        for fold_idx, (tr_idx, te_idx) in enumerate(splits):
            fold_label = "split" if quick else f"fold {fold_idx + 1}/{n_splits}"
            print(f"    [{fold_label}] training ...", end=" ", flush=True)

            seed_everything(RANDOM_STATE + fold_idx)
            X_tr = X.iloc[tr_idx].copy(); X_te = X.iloc[te_idx].copy()
            y_tr = y[tr_idx];      y_te = y[te_idx]

            # Fit scaling on the outer-training fold only. For the two
            # prespecified imbalanced endpoints, apply SMOTE after scaling and
            # only to outer-training observations.
            scaler = StandardScaler()
            X_tr = pd.DataFrame(
                scaler.fit_transform(X_tr), columns=X.columns)
            X_te = pd.DataFrame(
                scaler.transform(X_te), columns=X.columns)
            if config.get("use_smote", False):
                smote = SMOTE(random_state=RANDOM_STATE + fold_idx)
                X_res, y_tr = smote.fit_resample(X_tr, y_tr)
                X_tr = pd.DataFrame(X_res, columns=X.columns)

            fold_path = (
                WORK_DIR
                / "mljar"
                / target_name
                / ("split" if quick else f"fold_{fold_idx + 1}")
            )
            fold_path.mkdir(parents=True, exist_ok=True)

            t0 = time.time()
            automl = AutoML(
                mode="Explain",
                ml_task="binary_classification",
                total_time_limit=fold_time_limit,
                results_path=str(fold_path),
                explain_level=1,
                algorithms=["Baseline", "Linear", "Decision Tree",
                            "Random Forest", "Xgboost"],
                train_ensemble=True,
                stack_models=False,
                eval_metric="logloss",
                golden_features=False,
                features_selection=False,
                start_random_models=1,
                hill_climbing_steps=0,
                top_models_to_improve=0,
                boost_on_errors=False,
                kmeans_features=False,
                mix_encoding=False,
                n_jobs=1,
                random_state=RANDOM_STATE + fold_idx,
            )
            automl.fit(X_tr, y_tr)
            elapsed = time.time() - t0

            y_prob = automl.predict_proba(X_te)[:, 1]
            fold_auc = roc_auc_score(y_te, y_prob)
            fold_aucs.append(fold_auc)
            fold_probs.append(y_prob)
            fold_trues.append(y_te)
            oof_prob[te_idx] = y_prob
            print(f"AUC={fold_auc:.3f}  ({elapsed:.0f}s)")

            # Save fold leaderboard
            try:
                lb = automl.get_leaderboard()
                lb.to_csv(fold_path / "leaderboard.csv", index=False)
            except Exception:
                pass

        total_time = time.time() - total_t0

        evaluated = np.isfinite(oof_prob)
        m = compute_metrics(y[evaluated], oof_prob[evaluated])
        m["AUC_pooled"] = m["AUC"]
        m["AUC_mean"]  = float(np.mean(fold_aucs))
        m["AUC_std"]   = float(np.std(fold_aucs))
        mljar_results[target_name] = m
        pd.DataFrame({
            "Patient ID": df.loc[X.index[evaluated], "Patient ID"].values,
            "y_true": y[evaluated],
            "y_prob": oof_prob[evaluated],
        }).to_csv(OUTPUT_DIR / f"oof_mljar_{target_name}.csv", index=False)
        pd.DataFrame({
            "fold": np.arange(1, len(fold_aucs) + 1),
            "AUC": fold_aucs,
        }).to_csv(OUTPUT_DIR / f"folds_mljar_{target_name}.csv", index=False)

        print(f"  Mean AUC={m['AUC_mean']:.3f} ± {m['AUC_std']:.3f}"
              f"  Sens={m['Sensitivity']:.3f}  NPV={m['NPV']:.3f}"
              f"  Total={total_time:.0f}s")

        plt.figure(figsize=(8, 6))
        for i, (y_p, y_t) in enumerate(zip(fold_probs, fold_trues)):
            fpr, tpr, _ = roc_curve(y_t, y_p)
            plt.plot(fpr, tpr, alpha=0.4, lw=1.5,
                     label=f"Fold {i+1} (AUC={fold_aucs[i]:.3f})")

        # Pooled OOF ROC, displayed alongside fold-specific curves.
        fpr_all, tpr_all, _ = roc_curve(y[evaluated], oof_prob[evaluated])
        plt.plot(fpr_all, tpr_all, color="darkorange", lw=2.5,
                 label=f"Pooled OOF (AUC={m['AUC_pooled']:.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
        save_figure(WORK_DIR / "mljar" / "figures" / f"roc_{target_name}")
        plt.close()

    summary_df = pd.DataFrame({
        t: {
            "AUC_mean":    round(v["AUC_mean"], 3),
            "AUC_std":     round(v["AUC_std"], 3),
            "Sensitivity": round(v["Sensitivity"], 3),
            "Specificity": round(v["Specificity"], 3),
            "NPV":         round(v["NPV"], 3),
            "PPV":         round(v["PPV"], 3),
            "Brier":       round(v["Brier"], 3),
        }
        for t, v in mljar_results.items()
    }).T
    print(f"\n[MLJAR SUMMARY: {n_splits}-fold CV]\n{summary_df.to_string()}")
    print(f"  [MLJAR] Work files: {WORK_DIR / 'mljar'}")

    return mljar_results


# Tabnet

def run_tabnet(df: pd.DataFrame, n_splits: int = 5, quick: bool = False):
    """Run leakage-free TabNet evaluation with fold-local preprocessing.

    Each outer fold fits its scaler, unsupervised pretrainer, SMOTE (when
    prespecified), and supervised early stopping exclusively from outer-training
    data. The untouched outer-test fold is used once for evaluation.
    """
    print("\n" + "=" * 62)
    print("  PHASE 5: TABNET (LEAKAGE-FREE STRATIFIED CV)")
    print("=" * 62)

    try:
        import torch
        from pytorch_tabnet.pretraining import TabNetPretrainer
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError as exc:
        raise RuntimeError(
            "TabNet is unavailable; install the pinned requirements") from exc

    device = "cpu"
    seed_everything(RANDOM_STATE)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
    print("  [TabNet] Using deterministic CPU execution")

    pretrain_epochs = 30 if quick else 100
    finetune_epochs = 30 if quick else 100
    pretrain_patience = 8 if quick else 15
    finetune_patience = 8 if quick else 15

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    tabnet_results = {}
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for config in TARGET_CONFIGS:
        target_col = config["target_col"]
        target_name = config["name"]
        use_smote = config.get("use_smote", False)
        X_df, y = preprocess(
            df, feat_cols, target_col,
            exclude_cols=config.get("exclude_cols", []))
        X_np = X_df.values.astype(np.float32)
        splits = list(cv.split(X_np, y))

        oof_prob = np.full(len(y), np.nan, dtype=float)
        fold_aucs, fold_times, fold_roc, fold_importances = [], [], [], []

        print(f"\n  [TabNet] {target_name}: {len(y)} cases, "
              f"{X_np.shape[1]} features, SMOTE={use_smote}")

        for fold_idx, (outer_tr_idx, outer_te_idx) in enumerate(splits):
            fold_seed = RANDOM_STATE + fold_idx
            seed_everything(fold_seed)
            t0 = time.time()

            # An inner validation set controls pretraining and supervised early
            # stopping; the outer-test labels never enter model selection.
            inner_tr_idx, inner_val_idx = train_test_split(
                outer_tr_idx, test_size=0.20, stratify=y[outer_tr_idx],
                random_state=fold_seed)

            scaler = StandardScaler()
            X_fit = scaler.fit_transform(X_np[inner_tr_idx]).astype(np.float32)
            X_val = scaler.transform(X_np[inner_val_idx]).astype(np.float32)
            X_te = scaler.transform(X_np[outer_te_idx]).astype(np.float32)
            y_fit = y[inner_tr_idx]
            y_val = y[inner_val_idx]
            y_te = y[outer_te_idx]

            pretrainer = TabNetPretrainer(
                optimizer_fn=torch.optim.Adam,
                optimizer_params={"lr": 2e-3},
                mask_type="entmax",
                n_shared_decoder=1,
                n_indep_decoder=1,
                verbose=0,
                device_name=device,
                seed=fold_seed,
            )
            pretrainer.fit(
                X_train=X_fit,
                eval_set=[X_val],
                pretraining_ratio=0.8,
                max_epochs=pretrain_epochs,
                patience=pretrain_patience,
                batch_size=256,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False,
            )

            if use_smote:
                smote = SMOTE(random_state=fold_seed)
                X_fit, y_fit = smote.fit_resample(X_fit, y_fit)
                X_fit = X_fit.astype(np.float32)

            clf = TabNetClassifier(
                optimizer_fn=torch.optim.Adam,
                optimizer_params={"lr": 2e-3},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                scheduler_params={"step_size": 10, "gamma": 0.9},
                mask_type="sparsemax",
                verbose=0,
                device_name=device,
                seed=fold_seed,
            )
            clf.fit(
                X_train=X_fit,
                y_train=y_fit,
                eval_set=[(X_val, y_val)],
                eval_metric=["auc"],
                max_epochs=finetune_epochs,
                patience=finetune_patience,
                batch_size=256,
                virtual_batch_size=128,
                num_workers=0,
                from_unsupervised=pretrainer,
                drop_last=False,
            )

            fold_prob = clf.predict_proba(X_te)[:, 1]
            oof_prob[outer_te_idx] = fold_prob
            fold_auc = roc_auc_score(y_te, fold_prob)
            fpr, tpr, _ = roc_curve(y_te, fold_prob)
            fold_aucs.append(fold_auc)
            fold_times.append(time.time() - t0)
            fold_roc.append((fpr, tpr, fold_auc))
            try:
                fold_importances.append(clf.feature_importances_)
            except Exception:
                pass

            model_dir = WORK_DIR / "models" / "tabnet" / target_name
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"classifier": clf, "pretrainer": pretrainer, "scaler": scaler,
                 "features": list(X_df.columns), "outer_test_indices": outer_te_idx,
                 "seed": fold_seed},
                model_dir / f"fold_{fold_idx + 1}.pkl",
            )
            print(f"    fold {fold_idx + 1}/{n_splits}: "
                  f"AUC={fold_auc:.4f}, time={fold_times[-1]:.0f}s")

        metrics = compute_metrics(y, oof_prob)
        metrics["AUC_pooled"] = metrics["AUC"]
        metrics["AUC_mean"] = float(np.mean(fold_aucs))
        metrics["AUC_std"] = float(np.std(fold_aucs))
        metrics["AUC"] = metrics["AUC_mean"]
        tabnet_results[target_name] = metrics

        pd.DataFrame({
            "Patient ID": df.loc[X_df.index, "Patient ID"].values,
            "y_true": y,
            "y_prob": oof_prob,
        }).to_csv(OUTPUT_DIR / f"oof_tabnet_{target_name}.csv", index=False)
        pd.DataFrame({
            "fold": np.arange(1, n_splits + 1),
            "AUC": fold_aucs,
        }).to_csv(OUTPUT_DIR / f"folds_tabnet_{target_name}.csv", index=False)

        palette = plt.cm.Purples(np.linspace(0.4, 0.95, n_splits))
        fig, ax = plt.subplots(figsize=(7, 5))
        for fold_idx, (fpr, tpr, fold_auc) in enumerate(fold_roc):
            ax.plot(fpr, tpr, color=palette[fold_idx], lw=1.5,
                    label=f"Fold {fold_idx + 1} (AUC={fold_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        save_figure(
            WORK_DIR / "tabnet" / "figures" / f"roc_{target_name}", fig=fig)
        plt.close(fig)

        if fold_importances:
            mean_imp = np.mean(fold_importances, axis=0)
            fi_df = pd.DataFrame({
                "feature": list(X_df.columns),
                "importance": mean_imp,
            }).sort_values("importance").tail(15)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(fi_df["feature"], fi_df["importance"],
                    color="mediumpurple", edgecolor="white")
            ax.set_xlabel("Attention-based Importance")
            fig.tight_layout()
            save_figure(
                WORK_DIR / "tabnet" / "figures" / f"importance_{target_name}",
                fig=fig,
            )
            plt.close(fig)

        print(f"  [TabNet] {target_name}: "
              f"AUC={metrics['AUC_mean']:.4f} ± {metrics['AUC_std']:.4f}")

    summary_cols = [
        "AUC", "AUC_std", "AUC_pooled", "Sensitivity", "Specificity",
        "NPV", "PPV", "Brier"]
    summary_df = pd.DataFrame(tabnet_results).T[summary_cols].round(4)
    print(f"\n[TABNET CV SUMMARY]\n{summary_df.to_string()}")
    return tabnet_results

# Comparison summary

def run_comparison(trad_results: dict, mljar_results: dict,
                   tabnet_results: dict, *, diagnostics: bool = True):
    """Compare cross-validated AUC values across model families."""
    width = 76
    separator = "-" * width
    print("\nPHASE 6: COMPARISON SUMMARY")
    print(separator)

    if diagnostics and trad_results:
        trad_auc = pd.DataFrame(
            {t: res["AUC"] for t, res in trad_results.items()}
        )
        fig, ax = plt.subplots(figsize=(max(10, len(trad_results) * 1.8), 5))
        sns.heatmap(trad_auc.round(3), annot=True, fmt=".3f",
                    cmap="RdYlGn", vmin=0.5, vmax=1.0, ax=ax,
                    linewidths=0.5, cbar_kws={"label": "AUC-ROC"})
        plt.tight_layout()
        save_figure(
            WORK_DIR / "diagnostics" / "comparison" / "traditional_auc_heatmap")
        plt.close()

    rows = []
    for cfg in TARGET_CONFIGS:
        t   = cfg["name"]
        row = {"Target": t}

        if t in trad_results:
            best_auc   = trad_results[t]["AUC"].max()
            best_model = trad_results[t]["AUC"].idxmax()
            row["Trad_AUC"]   = float(best_auc)
            row["Trad_Model"] = best_model
            try:
                row["Trad_Std"] = float(
                    trad_results[t].loc[best_model, "AUC_std"])
            except Exception:
                row["Trad_Std"] = None

        if t in mljar_results:
            row["MLJAR_AUC"] = float(mljar_results[t].get(
                "AUC_mean", mljar_results[t].get("AUC", 0)))
            row["MLJAR_Std"] = float(mljar_results[t].get("AUC_std", 0))

        if t in tabnet_results:
            row["TabNet_AUC"] = float(tabnet_results[t].get(
                "AUC_mean", tabnet_results[t].get("AUC", 0)))
            row["TabNet_Std"] = float(tabnet_results[t].get("AUC_std", 0))

        rows.append(row)

    comp_df = pd.DataFrame(rows).set_index("Target")

    C1, C2, C3, C4, C5 = 20, 26, 18, 18, 22
    hdr = (f"  {'Target':<{C1}}"
           f"{'Trad ML (best)':<{C2}}"
           f"{'MLJAR AutoML':<{C3}}"
           f"{'TabNet':<{C4}}"
           f"{'Winner':<{C5}}")
    sub = (f"  {'':<{C1}}"
           f"{'(model | AUC ± std)':<{C2}}"
           f"{'(AUC ± std)':<{C3}}"
           f"{'(AUC ± std)':<{C4}}"
           f"{'':<{C5}}")

    print("\n  AUC-ROC COMPARISON TABLE (5-fold StratifiedKFold)")
    print(f"  {separator}")
    print(hdr)
    print(sub)
    print(f"  {separator}")

    winner_counts = {
        "Traditional ML": 0, "MLJAR AutoML": 0, "TabNet": 0, "Tie": 0}

    for _, row in comp_df.iterrows():
        trad_auc   = row.get("Trad_AUC")
        mljar_auc  = row.get("MLJAR_AUC")
        tabnet_auc = row.get("TabNet_AUC")

        scores = {}
        if pd.notna(trad_auc):   scores["Traditional ML"] = trad_auc
        if pd.notna(mljar_auc):  scores["MLJAR AutoML"]   = mljar_auc
        if pd.notna(tabnet_auc): scores["TabNet"]         = tabnet_auc

        if scores:
            winner_auc = max(scores.values())
            leaders = [method for method, score in scores.items()
                       if np.isclose(score, winner_auc, rtol=0, atol=1e-12)]
            winner = (leaders[0] if len(leaders) == 1
                      else "Tie: " + " / ".join(leaders))
            winner_counts["Tie" if len(leaders) > 1 else leaders[0]] += 1
        else:
            winner = "NA"

        trad_cell = "NA"
        if pd.notna(trad_auc):
            model_short = str(row.get("Trad_Model", ""))[:12]
            trad_std    = row.get("Trad_Std")
            if trad_std is not None:
                trad_cell = f"{model_short}  {trad_auc:.3f} ± {trad_std:.3f}"
            else:
                trad_cell = f"{model_short}  {trad_auc:.3f}"

        mljar_cell = "NA"
        if pd.notna(mljar_auc):
            std = row.get("MLJAR_Std") or 0
            mljar_cell = f"{mljar_auc:.3f} ± {std:.3f}"

        tabnet_cell = "NA"
        if pd.notna(tabnet_auc):
            std = row.get("TabNet_Std") or 0
            tabnet_cell = f"{tabnet_auc:.3f} ± {std:.3f}"

        win_cell = winner

        print(f"  {row.name:<{C1}}"
              f"{trad_cell:<{C2}}"
              f"{mljar_cell:<{C3}}"
              f"{tabnet_cell:<{C4}}"
              f"{win_cell:<{C5}}")

    print(f"  {separator}")

    print(f"\n  WINS PER METHOD  (out of {len(rows)} targets)")
    print(f"  {separator}")
    for method, wins in sorted(winner_counts.items(), key=lambda x: -x[1]):
        print(f"  {method:<20}  {wins}/{len(rows)}")
    print(f"  {separator}")

    export_rows = []
    for _, row in comp_df.iterrows():
        trad_auc  = row.get("Trad_AUC")
        mljar_auc = row.get("MLJAR_AUC")
        tnet_auc  = row.get("TabNet_AUC")
        scores    = {}
        if pd.notna(trad_auc):  scores["Traditional ML"] = trad_auc
        if pd.notna(mljar_auc): scores["MLJAR AutoML"]   = mljar_auc
        if pd.notna(tnet_auc):  scores["TabNet"]         = tnet_auc
        if scores:
            best_score = max(scores.values())
            leaders = [method for method, score in scores.items()
                       if np.isclose(score, best_score, rtol=0, atol=1e-12)]
            winner = (leaders[0] if len(leaders) == 1
                      else "Tie: " + " / ".join(leaders))
        else:
            winner = "NA"
        export_rows.append({
            "Target":          row.name,
            "Trad_AUC_mean":   trad_auc,
            "Trad_AUC_std":    row.get("Trad_Std"),
            "Trad_Best_Model": row.get("Trad_Model", ""),
            "MLJAR_AUC_mean":  mljar_auc,
            "MLJAR_AUC_std":   row.get("MLJAR_Std"),
            "TabNet_AUC_mean": tnet_auc,
            "TabNet_AUC_std":  row.get("TabNet_Std"),
            "Winner":          winner,
        })

    export_df = pd.DataFrame(export_rows).set_index("Target")
    export_df.to_csv(OUTPUT_DIR / "compare_02_auc_summary.csv")

    exact_rows = []
    metric_columns = [
        "AUC_mean", "AUC_std", "AUC_pooled", "Sensitivity",
        "Specificity", "NPV", "PPV", "Brier",
    ]
    for cfg in TARGET_CONFIGS:
        target = cfg["name"]
        for method_name, result_set in (
            ("MLJAR AutoML", mljar_results),
            ("TabNet", tabnet_results),
        ):
            if target not in result_set:
                continue
            metrics = result_set[target]
            exact_rows.append({
                "Target": target,
                "Method": method_name,
                **{column: metrics.get(column) for column in metric_columns},
            })
    pd.DataFrame(exact_rows).to_csv(
        OUTPUT_DIR / "method_metrics_exact.csv", index=False)

    # Bar chart comparison
    auc_plot_data = {}
    for _, row in comp_df.iterrows():
        if pd.notna(row.get("Trad_AUC")):
            auc_plot_data.setdefault("Traditional ML", {})[row.name] = row["Trad_AUC"]
        if pd.notna(row.get("MLJAR_AUC")):
            auc_plot_data.setdefault("MLJAR AutoML",   {})[row.name] = row["MLJAR_AUC"]
        if pd.notna(row.get("TabNet_AUC")):
            auc_plot_data.setdefault("TabNet",         {})[row.name] = row["TabNet_AUC"]

    if auc_plot_data:
        plot_df   = pd.DataFrame(auc_plot_data)
        n_methods = len(plot_df.columns)
        colors    = ["steelblue", "darkorange", "mediumpurple"][:n_methods]

        fig, ax = plt.subplots(figsize=(14, 6))
        x       = np.arange(len(plot_df))
        width   = 0.22
        offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods)

        for i, (col, color) in enumerate(zip(plot_df.columns, colors)):
            std_col = {"Traditional ML": "Trad_Std",
                       "MLJAR AutoML":   "MLJAR_Std",
                       "TabNet":         "TabNet_Std"}.get(col)
            errs = ([comp_df.loc[t, std_col] or 0 for t in plot_df.index]
                    if std_col and std_col in comp_df.columns else None)
            bars = ax.bar(x + offsets[i] * width, plot_df[col],
                          width=width, label=col, color=color,
                          edgecolor="white", zorder=3,
                          yerr=errs, capsize=3, error_kw={"elinewidth": 1.2})
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom",
                        fontsize=7, color="black")

        ax.axhline(0.5, color="red", ls="--", lw=1, label="Chance (0.5)", zorder=2)
        display_targets = {
            "Overall_Survival": "Overall survival",
            "Relapse_Free": "Recurrence event",
            "NPI_High_Risk": "NPI high risk",
            "Vital_Disease": "Disease-specific death",
            "Luminal_Subtype": "Luminal subtype",
            "HER2_Positive": "HER2 positive",
        }
        ax.set_xticks(x)
        ax.set_xticklabels(
            [display_targets.get(target, target.replace("_", " "))
             for target in plot_df.index],
            rotation=25, ha="right", fontsize=10)
        ax.set_ylim(0.45, 1.06)
        ax.set_ylabel("AUC-ROC", fontsize=11)
        ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3, zorder=0)
        fig.tight_layout()
        save_figure(WORK_DIR / "figures" / "method_comparison", fig=fig)
        plt.close(fig)
        print(f"  [PLOT] Method comparison saved to {WORK_DIR / 'figures' / 'method_comparison.pdf'}")

    heat_data = {}
    for _, row in comp_df.iterrows():
        heat_data[row.name] = {
            "Traditional ML": row.get("Trad_AUC"),
            "MLJAR AutoML":   row.get("MLJAR_AUC"),
            "TabNet":         row.get("TabNet_AUC"),
        }
    heat_df = pd.DataFrame(heat_data).T.astype(float)
    if diagnostics and not heat_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=0.5, vmax=1.0, ax=ax, linewidths=0.6,
                    cbar_kws={"label": "AUC-ROC"})
        ax.set_xlabel(""); ax.set_ylabel("")
        plt.yticks(rotation=0); plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        save_figure(
            WORK_DIR / "diagnostics" / "comparison" / "all_methods_heatmap")
        plt.close(fig)

    print(f"\n  Summary: {OUTPUT_DIR / 'compare_02_auc_summary.csv'}")
    print(f"  Figure: {WORK_DIR / 'figures' / 'method_comparison.pdf'}\n")

    return comp_df


def generate_calibration_figure() -> None:
    """Generate a diagnostic figure from saved traditional-model OOF scores."""
    panels = ["Overall_Survival", "NPI_High_Risk"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for panel, (ax, target_name) in enumerate(zip(axes, panels)):
        source = OUTPUT_DIR / f"oof_trad_{target_name}.csv"
        oof = pd.read_csv(source)
        y_true = oof["y_true"].to_numpy(dtype=int)
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect")
        model_columns = [
            column for column in oof.columns
            if column not in {"Patient ID", "y_true"}]
        for (model_name, color) in zip(model_columns, PAL.values()):
            probability = oof[model_name].to_numpy(dtype=float)
            fraction_positive, mean_predicted = calibration_curve(
                y_true, probability, n_bins=10, strategy="uniform")
            brier = brier_score_loss(y_true, probability)
            ax.plot(mean_predicted, fraction_positive, "o-", lw=1.5,
                    ms=3.5, color=color,
                    label=f"{model_name} (Brier={brier:.3f})")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed event fraction")
        ax.text(0.5, 1.03, chr(65 + panel), transform=ax.transAxes,
                fontweight="bold", ha="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5, loc="best")
    fig.tight_layout()
    save_figure(WORK_DIR / "figures" / "calibration", fig=fig)
    plt.close(fig)


# Command-line entry point

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate models on METABRIC clinical data."
    )
    parser.add_argument(
        "--data",
        default=str(default_data_path()),
        help="Path to Breast_Cancer_METABRIC.csv.",
    )
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for canonical CSV, JSON, and PDF artifacts.",
    )
    parser.add_argument(
        "--workdir",
        default=str(DEFAULT_WORK_DIR),
        help="Directory for logs, models, and diagnostic artifacts.",
    )
    parser.add_argument(
        "--replace-outputs",
        action="store_true",
        help="Replace a non-empty output directory explicitly.",
    )
    parser.add_argument(
        "--replace-work",
        action="store_true",
        help="Replace a non-empty disposable work directory explicitly.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    data_path = resolve_cli_path(args.data)
    if not data_path.is_file():
        parser.error(f"Data file not found: {data_path}")

    configure_paths(
        output_dir=resolve_cli_path(args.outdir),
        work_dir=resolve_cli_path(args.workdir),
    )
    if (OUTPUT_DIR == WORK_DIR
            or OUTPUT_DIR.is_relative_to(WORK_DIR)
            or WORK_DIR.is_relative_to(OUTPUT_DIR)):
        parser.error("Output and work directories must be separate paths.")
    if data_path.is_relative_to(OUTPUT_DIR) or data_path.is_relative_to(WORK_DIR):
        parser.error("The input data file must be outside output and work directories.")
    try:
        if (OUTPUT_DIR / 'method_summary.csv').exists():
            raise ValueError('This directory contains nested-analysis artifacts; use a separate --outdir.')
        validate_directory_state(
            path=OUTPUT_DIR, replace=args.replace_outputs,
            replacement_flag="--replace-outputs")
        validate_directory_state(
            path=WORK_DIR, replace=args.replace_work,
            replacement_flag="--replace-work")
        data = load_and_engineer(data_path)
        for config in TARGET_CONFIGS:
            preprocess(data, FEATURE_COLS, config["target_col"], config["exclude_cols"])
        prepare_work_directory(replace=args.replace_work)
        prepare_output_directory(replace=args.replace_outputs)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))

    seed_everything()
    log_path = WORK_DIR / "logs" / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tee = _Tee(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee
    try:
        print("\n" + "=" * 62)
        print("  METABRIC MACHINE-LEARNING PIPELINE")
        print("  Traditional ML + MLJAR + TabNet")
        print("=" * 62)

        print("\n" + "=" * 62 + "\n  PHASE 1: DESCRIPTIVE ANALYSIS\n" + "=" * 62)
        generate_analysis_figures(data)

        traditional_results = run_traditional_pipeline(
            data, run_shap=True, run_subgroup=True)
        mljar_results = run_mljar(
            data, n_splits=5, fold_time_limit=300, quick=False)
        tabnet_results = run_tabnet(data, n_splits=5, quick=False)
        run_comparison(traditional_results, mljar_results, tabnet_results)
        generate_calibration_figure()

        print(f"\n{'=' * 62}")
        print(f"  Outputs: {OUTPUT_DIR}")
        print(f"  Work files: {WORK_DIR}")
        print(f"{'=' * 62}")
    finally:
        sys.stdout = original_stdout
        tee.close()

    print(f"[LOG] {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
