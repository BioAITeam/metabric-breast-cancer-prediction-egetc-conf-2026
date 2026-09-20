# Multi-Target Machine Learning for Breast Cancer Prognosis and Molecular Classification Using METABRIC

A reproducible Python workflow for classification of recorded survival status, recurrence, disease-specific death, luminal subtype, and HER2 status in METABRIC. NPI high risk is evaluated separately as a target-reconstruction diagnostic.

The current comparison uses nested selection among five conventional model families and patient-aligned MLJAR and fixed-configuration TabNet predictions. Missing-data sensitivity, corrected statistical comparisons, calibration, operating points, and explanation stability accompany the discrimination estimates.

## Structure

```text
repo/
|-- code/
|   |-- evaluation_utils.py
|   |-- finalize_outputs.py
|   |-- metabric_ml_pipeline.py
|   |-- model_evaluation.py
|   |-- plot_evaluation.py
|   `-- regenerate_explanations.py
|-- data/
|   |-- README.md
|   `-- raw/
|-- outputs/
|   `-- predictions, final tables, and figures
|-- .gitattributes
|-- .gitignore
|-- .python-version
|-- LICENSE
|-- README.md
`-- requirements.txt
```

The fixed-model outputs and the nested analysis have different purposes; their estimates must not be interchanged. The commands and protocols for each are described below.

## License and citation

The original code and documentation are available under the [Noncommercial Research and Education License](LICENSE). Use, modification, and redistribution are permitted solely for noncommercial research or educational purposes, with the required citation and preservation of the license notices. Commercial use requires separate written permission from the relevant copyright holders.

Publications, reports, presentations, teaching materials, and other publicly shared works using the software or results obtained with it must cite Nham et al. (2026), *Multi-Target Machine Learning for Breast Cancer Prognosis and Molecular Classification Using METABRIC*. The full author list is provided in [LICENSE](LICENSE); use the final published bibliographic details and DOI when available.

Third-party libraries, METABRIC data, and applicable rights in data-derived artifacts remain governed by their own terms.

## Installation

The reference environment is CPython 3.12.6 on Windows. From the repository root:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

An existing compatible environment may be used instead. Keep the versions in `requirements.txt` when reproducing the results.

## Data

Training requires `data/raw/Breast_Cancer_METABRIC.csv`, the 2,509-row, 34-column clinical table. Raw data are supplied locally and excluded from Git.

See [data requirements](data/README.md) for source information, expected fields, and cohort definitions. Patient identifiers in prediction files are dataset identifiers used for alignment.

## Reproduce the nested analysis

Set process controls before starting Python:

```powershell
$env:PYTHONHASHSEED = "42"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
```

Fitting intermediates are written to the Git-ignored `temp/evaluation/` directory:

```powershell
.\venv\Scripts\python.exe -B .\code\model_evaluation.py nested
.\venv\Scripts\python.exe -B .\code\model_evaluation.py sensitivity
.\venv\Scripts\python.exe -B .\code\model_evaluation.py shap
.\venv\Scripts\python.exe -B .\code\model_evaluation.py summarize
.\venv\Scripts\python.exe -B .\code\plot_evaluation.py
```

This workflow reuses the original outer assignments and MLJAR/TabNet predictions in `outputs/` and refits conventional and explanatory models. For a new run or different inputs or settings, choose a new `--outdir` under `temp/`. Summarization requires all six targets and all three explanation seeds. These stages can take several hours on a CPU.

Generated analysis tables are stored in `temp/evaluation/`; `outputs/` contains the reference results. `plot_evaluation.py` reads the generated SHAP tables and saved predictions, then exports `Figure-4.pdf` and `Figure-5.pdf` to `outputs/`. Use `--analysis-dir`, `--reference-dir`, and `--outdir` to select other directories.

Each analysis command accepts `--data` and `--reference-dir`; `--targets` can restrict a fitting stage to named targets. Run `--help` for accepted names and paths.

## Reproduce the original comparison

The original fixed-configuration workflow, including MLJAR and TabNet training, remains executable:

```powershell
.\venv\Scripts\python.exe -B -u .\code\metabric_ml_pipeline.py --outdir .\temp\original --workdir .\temp\work
```

To use those newly generated outer predictions in the nested analysis, pass `--reference-dir .\temp\original` to the analysis and figure commands and use a fresh `--outdir` under `temp/`. `finalize_outputs.py` rebuilds original figures; `regenerate_explanations.py` reproduces the original single-split explanations. Neither replaces the current three-seed explanation analysis.

## Interpretation and reproducibility

- The outer protocol uses five seeded stratified folds; conventional family selection uses three inner folds.
- The current analysis fits preprocessing inside the corresponding training partitions, including conventional calibration and stacking.
- Outcomes are never imputed. The imputation sensitivity changes both eligibility and missing-data handling, so it is not a controlled estimate of an imputation effect.
- NPI and its ordinal group are excluded from T3; HER2 and its ER interaction are excluded from T6. Retained NPI components still make T3 reconstructive rather than independent prognosis.
- Status endpoints do not model censoring, competing risks, or a fixed prediction horizon. Treatment associations are not causal effects.
- Threshold 0.5 and training-selected Youden operating points are reported separately.
- Performance estimates come from resampling this cohort; no independent external cohort was evaluated.
- Pinned versions, seeded components, and single-threaded execution support reproducibility. Hardware, library builds, and MLJAR's time budget can still affect refitted results.
