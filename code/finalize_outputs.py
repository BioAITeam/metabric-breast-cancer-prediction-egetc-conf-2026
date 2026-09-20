"""Rebuild descriptive figures and local diagnostics from saved artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import metabric_ml_pipeline as pipeline


def _saved_method(target: str, method: str) -> dict[str, float]:
    oof = pd.read_csv(pipeline.OUTPUT_DIR / f"oof_{method}_{target}.csv")
    folds = pd.read_csv(pipeline.OUTPUT_DIR / f"folds_{method}_{target}.csv")
    metrics = pipeline.compute_metrics(
        oof["y_true"].to_numpy(dtype=int),
        oof["y_prob"].to_numpy(dtype=float),
    )
    metrics["AUC_pooled"] = metrics["AUC"]
    metrics["AUC_mean"] = float(folds["AUC"].mean())
    metrics["AUC_std"] = float(np.std(folds["AUC"].to_numpy(dtype=float)))
    metrics["AUC"] = metrics["AUC_mean"]
    return metrics


def _generate_shap_figure() -> None:
    source = (
        pipeline.OUTPUT_DIR
        / "shap_importance_Stacking_Ensemble_Overall_Survival.csv"
    )
    importance = pd.read_csv(source)
    pipeline.generate_shap_importance_figure(importance)


def _required_inputs() -> list[Path]:
    paths = [pipeline.OUTPUT_DIR / "shap_importance_Stacking_Ensemble_Overall_Survival.csv"]
    for config in pipeline.TARGET_CONFIGS:
        target = config["name"]
        paths.extend([
            pipeline.OUTPUT_DIR / f"cv_trad_{target}.csv",
            pipeline.OUTPUT_DIR / f"oof_mljar_{target}.csv",
            pipeline.OUTPUT_DIR / f"folds_mljar_{target}.csv",
            pipeline.OUTPUT_DIR / f"oof_tabnet_{target}.csv",
            pipeline.OUTPUT_DIR / f"folds_tabnet_{target}.csv",
        ])
    paths.extend([
        pipeline.OUTPUT_DIR / "oof_trad_Overall_Survival.csv",
        pipeline.OUTPUT_DIR / "oof_trad_NPI_High_Risk.csv",
    ])
    return paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild descriptive figures and fixed-model diagnostics from saved artifacts."
    )
    parser.add_argument("--data", default=str(pipeline.default_data_path()))
    parser.add_argument("--outdir", default=str(pipeline.DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--replace-outputs",
        action="store_true",
        help="Allow replacement of existing figures and tables.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    data_path = pipeline.resolve_cli_path(args.data)
    if not data_path.is_file():
        parser.error(f"Data file not found: {data_path}")
    pipeline.configure_paths(output_dir=pipeline.resolve_cli_path(args.outdir))
    try:
        pipeline._assert_replaceable_directory(pipeline.OUTPUT_DIR)
    except ValueError as exc:
        parser.error(str(exc))
    missing = [path for path in _required_inputs() if not path.is_file()]
    if missing:
        parser.error("Missing required artifacts: " + ", ".join(map(str, missing)))

    destinations = [
        pipeline.OUTPUT_DIR / f"Figure-{number}.pdf" for number in range(1, 4)
    ]
    destinations.extend(pipeline.WORK_DIR / "figures" / f"{name}.pdf" for name in (
        "npi_distributions", "method_comparison", "shap_importance", "calibration"))
    destinations.extend([
        pipeline.OUTPUT_DIR / "marginal_numeric_statistics.csv",
        pipeline.OUTPUT_DIR / "marginal_binary_statistics.csv",
        pipeline.OUTPUT_DIR / "compare_02_auc_summary.csv",
        pipeline.OUTPUT_DIR / "method_metrics_exact.csv",
    ])
    existing = [path for path in destinations if path.exists()]
    if existing and not args.replace_outputs:
        parser.error(
            "Canonical outputs already exist. Use --replace-outputs to refresh them."
        )

    pipeline.seed_everything()
    data = pipeline.load_and_engineer(data_path)
    pipeline.generate_analysis_figures(data)

    traditional: dict[str, pd.DataFrame] = {}
    mljar: dict[str, dict[str, float]] = {}
    tabnet: dict[str, dict[str, float]] = {}
    for config in pipeline.TARGET_CONFIGS:
        target = config["name"]
        traditional[target] = pd.read_csv(
            pipeline.OUTPUT_DIR / f"cv_trad_{target}.csv", index_col=0
        )
        mljar[target] = _saved_method(target, "mljar")
        tabnet[target] = _saved_method(target, "tabnet")

    pipeline.run_comparison(traditional, mljar, tabnet, diagnostics=False)
    _generate_shap_figure()
    pipeline.generate_calibration_figure()
    print(f"Canonical figures refreshed in {pipeline.OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
