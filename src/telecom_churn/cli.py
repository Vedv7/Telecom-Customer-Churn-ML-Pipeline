from __future__ import annotations

import argparse
from pathlib import Path

from telecom_churn.config import ARTIFACTS_DIR, DEFAULT_DATA_PATH, DEFAULT_OPTUNA_TRIALS
from telecom_churn.train import train


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Train telecom churn models (RFM segmentation, RFECV, XGBoost + Optuna) and write artifacts.",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to churn dataset (.xlsx / .xls / .csv). Defaults to data/mobile-churn-data.xlsx if present.",
    )
    p.add_argument("--out", type=Path, default=ARTIFACTS_DIR, help="Directory for churn_bundle.joblib and metrics.")
    p.add_argument("--trials", type=int, default=DEFAULT_OPTUNA_TRIALS, help="Optuna tuning trials.")
    p.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP plots (faster runs, e.g. smoke tests).",
    )
    args = p.parse_args(argv)
    data_path = args.data
    if data_path is None:
        data_path = DEFAULT_DATA_PATH if DEFAULT_DATA_PATH.is_file() else None
    if data_path is None:
        p.error("Pass --data PATH or place your workbook at data/mobile-churn-data.xlsx")

    summary = train(
        data_path,
        args.out,
        tune_trials=args.trials,
        skip_shap=args.skip_shap,
    )
    print(f"Wrote artifacts to {args.out.resolve()}")
    print(f"Selected features: {summary['metrics']['n_selected_features']}")


if __name__ == "__main__":
    main()
