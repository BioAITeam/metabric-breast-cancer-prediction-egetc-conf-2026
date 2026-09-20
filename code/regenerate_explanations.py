"""Regenerate SHAP artifacts from seeded explanatory fits."""

from __future__ import annotations

import argparse
import sys

from sklearn.model_selection import train_test_split

import metabric_ml_pipeline as pipeline


SELECTED_MODELS = {
    "Overall_Survival": "Stacking Ensemble",
    "Relapse_Free": "Stacking Ensemble",
    "NPI_High_Risk": "Random Forest",
    "Vital_Disease": "Stacking Ensemble",
    "Luminal_Subtype": "Stacking Ensemble",
    "HER2_Positive": "Logistic Regression",
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate seeded SHAP artifacts.")
    parser.add_argument("--data", default=str(pipeline.default_data_path()))
    parser.add_argument("--outdir", default=str(pipeline.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--workdir", default=str(pipeline.DEFAULT_WORK_DIR))
    parser.add_argument(
        "--replace-outputs",
        action="store_true",
        help="Allow replacement of existing SHAP tables and diagnostic figures.",
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
    data_path = pipeline.resolve_cli_path(args.data)
    if not data_path.is_file():
        parser.error(f"Data file not found: {data_path}")
    pipeline.configure_paths(
        output_dir=pipeline.resolve_cli_path(args.outdir),
        work_dir=pipeline.resolve_cli_path(args.workdir),
    )
    if (pipeline.OUTPUT_DIR == pipeline.WORK_DIR
            or pipeline.OUTPUT_DIR.is_relative_to(pipeline.WORK_DIR)
            or pipeline.WORK_DIR.is_relative_to(pipeline.OUTPUT_DIR)):
        parser.error("Output and work directories must be separate paths.")
    if data_path.is_relative_to(pipeline.OUTPUT_DIR) or data_path.is_relative_to(pipeline.WORK_DIR):
        parser.error("The input data file must be outside output and work directories.")
    try:
        pipeline._assert_replaceable_directory(pipeline.OUTPUT_DIR)
    except ValueError as exc:
        parser.error(str(exc))
    destinations = [pipeline.WORK_DIR / "figures" / "shap_importance.pdf"]
    destinations.extend(
        pipeline.OUTPUT_DIR
        / f"shap_importance_{model.replace(' ', '_')}_{target}.csv"
        for target, model in SELECTED_MODELS.items()
    )
    if any(path.exists() for path in destinations) and not args.replace_outputs:
        parser.error(
            "SHAP outputs already exist. Use --replace-outputs to refresh them."
        )

    try:
        pipeline.validate_directory_state(
            path=pipeline.WORK_DIR,
            replace=args.replace_work,
            replacement_flag="--replace-work",
        )
        data = pipeline.load_and_engineer(data_path)
        for config in pipeline.TARGET_CONFIGS:
            pipeline.preprocess(data, pipeline.FEATURE_COLS, config["target_col"],
                                config["exclude_cols"])
        pipeline.prepare_work_directory(replace=args.replace_work)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    pipeline.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = pipeline.WORK_DIR / "logs" / "shap_regeneration.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tee = pipeline._Tee(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee
    try:
        pipeline.seed_everything(pipeline.RANDOM_STATE)
        available_features = [
            column for column in pipeline.FEATURE_COLS if column in data.columns
        ]
        diagnostics_dir = pipeline.WORK_DIR / "diagnostics" / "shap"

        for config in pipeline.TARGET_CONFIGS:
            target_name = config["name"]
            model_name = SELECTED_MODELS[target_name]
            pipeline.seed_everything(pipeline.RANDOM_STATE)
            X, y = pipeline.preprocess(
                data,
                available_features,
                config["target_col"],
                exclude_cols=config.get("exclude_cols", []),
            )
            X_train, X_test, y_train, _ = train_test_split(
                X,
                y,
                test_size=0.3,
                stratify=y,
                random_state=pipeline.RANDOM_STATE,
            )
            model = pipeline.build_models(
                use_smote=config.get("use_smote", False)
            )[model_name]
            model.fit(X_train, y_train)
            values, _ = pipeline.compute_shap(
                model,
                X_train,
                X_test,
                list(X.columns),
                model_name,
                target_name,
                diagnostics_dir,
            )
            if values is None:
                raise RuntimeError(f"SHAP generation failed for {target_name}")
        print(f"SHAP artifacts refreshed in {pipeline.OUTPUT_DIR}")
    finally:
        sys.stdout = original_stdout
        tee.close()
    print(f"[LOG] {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
