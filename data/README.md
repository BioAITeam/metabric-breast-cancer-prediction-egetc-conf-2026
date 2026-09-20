# METABRIC input data

Place the locally supplied clinical table at `raw/Breast_Cancer_METABRIC.csv`. The raw directory is excluded from Git; the scripts do not download data automatically.

The analysis uses a fixed snapshot with 2,509 rows and 34 columns. METABRIC clinical data are available through [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric). A current portal export may use different field names or a different schema and is not automatically interchangeable with this CSV.

Required fields include `Patient ID`, `Age at Diagnosis`, `Tumor Size`, `Neoplasm Histologic Grade`, `Lymph nodes examined positive`, `Mutation Count`, `Nottingham prognostic index`, `Tumor Stage`, receptor status, treatment variables, `Overall Survival Status`, `Relapse Free Status`, `Patient's Vital Status`, and `Pam50 + Claudin-low subtype`. Exact field names and mappings are defined in `code/metabric_ml_pipeline.py`.

Unknown outcomes remain missing. Complete-case analysis includes 1,297 patients for overall survival, NPI high risk, luminal subtype, and HER2, and 1,296 for recurrence and disease-specific death. Imputation sensitivity retains every recorded outcome and imputes predictors within training folds.

METABRIC source publications:

- Curtis et al. (2012), [Nature](https://doi.org/10.1038/nature10983).
- Pereira et al. (2016), [Nature Communications](https://doi.org/10.1038/ncomms11479).

Dataset access and use remain governed by the source terms.
