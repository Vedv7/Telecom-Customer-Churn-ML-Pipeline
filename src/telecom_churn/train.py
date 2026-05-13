from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from telecom_churn.config import (
    ARTIFACTS_DIR,
    DEFAULT_OPTUNA_TRIALS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_FRACTION_OF_FIT,
    REPO_ROOT,
)
from telecom_churn.end_to_end import ChurnEndToEndModel
from telecom_churn.evaluation import (
    print_classification_metrics,
    save_confusion_matrix,
    save_roc_pr_curves,
)
from telecom_churn.explainability import save_shap_bar, save_shap_summary
from telecom_churn.feature_selection import select_features_rfecv
from telecom_churn.model_training import (
    benchmark_models,
    optuna_params_to_sklearn_xgb,
    train_xgboost,
    tune_xgboost_with_optuna,
)
from telecom_churn.preprocessing import (
    apply_correlation_drop,
    clean_dataset,
    encode_churn_target,
    fit_correlated_columns_to_drop,
    load_dataset,
    scale_and_balance_train_only,
)
from telecom_churn.segmentation import (
    assign_cluster_column,
    fit_rfm_kmeans,
    remove_low_value_cluster,
)


def train(
    data_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    tune_trials: int = DEFAULT_OPTUNA_TRIALS,
    target_col: str = "churn",
    images_dir: str | Path | None = None,
    skip_shap: bool = False,
    low_value_cluster: int = 1,
    corr_threshold: float = 0.85,
) -> dict[str, Any]:
    """Run the full churn pipeline and persist ``ChurnEndToEndModel`` + metrics.

    Split policy (leakage-aware):
    - 80% fit / 20% held-out test (stratified).
    - Correlation pruning + RFM KMeans fit **only** on the fit split; applied to test.
    - Low-value cluster filter applied per split (same rule as training).
    - From the fit split: 75% train / 25% validation (stratified) for RFECV + Optuna.
    - Optuna optimizes recall on **validation** only; test is used once for final metrics.
    """
    out_dir = Path(out_dir or ARTIFACTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(images_dir or (REPO_ROOT / "images"))

    df = load_dataset(str(data_path))
    df = clean_dataset(df)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    strat0 = encode_churn_target(df[target_col])
    df_fit, df_test = train_test_split(
        df,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=strat0,
    )

    corr_drop = fit_correlated_columns_to_drop(df_fit, target_col=target_col, threshold=corr_threshold)
    df_fit = apply_correlation_drop(df_fit, corr_drop)
    df_test = apply_correlation_drop(df_test, corr_drop)

    rfm_scaler, kmeans = fit_rfm_kmeans(df_fit, n_clusters=3, random_state=0)
    df_fit = assign_cluster_column(df_fit, rfm_scaler, kmeans)
    df_test = assign_cluster_column(df_test, rfm_scaler, kmeans)

    df_fit = remove_low_value_cluster(df_fit, low_value_cluster=low_value_cluster)
    df_test = remove_low_value_cluster(df_test, low_value_cluster=low_value_cluster)

    strat_fit = encode_churn_target(df_fit[target_col])
    df_train, df_val = train_test_split(
        df_fit,
        test_size=DEFAULT_VAL_FRACTION_OF_FIT,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=strat_fit,
    )

    X_train = df_train.drop(columns=[target_col])
    y_train = encode_churn_target(df_train[target_col])
    X_val = df_val.drop(columns=[target_col])
    y_val = encode_churn_target(df_val[target_col])
    X_test = df_test.drop(columns=[target_col])
    y_test = encode_churn_target(df_test[target_col])

    X_bal, y_bal, X_val_s, X_test_s, scaler = scale_and_balance_train_only(
        X_train,
        y_train,
        X_val,
        X_test,
        random_state=DEFAULT_RANDOM_STATE,
    )

    selected_features, _ = select_features_rfecv(X_bal, y_bal)
    X_bal = X_bal[selected_features]
    X_val_s = X_val_s[selected_features]
    X_test_s = X_test_s[selected_features]

    benchmark = benchmark_models(X_bal, y_bal)
    benchmark_path = out_dir / "model_benchmark.csv"
    benchmark.to_csv(benchmark_path, index=False)

    native_params, best_recall_val = tune_xgboost_with_optuna(
        X_bal,
        y_bal,
        X_val_s,
        y_val,
        n_trials=tune_trials,
    )
    sklearn_params = optuna_params_to_sklearn_xgb(native_params)
    model = train_xgboost(X_bal, y_bal, sklearn_params)

    print_classification_metrics(model, X_test_s, y_test)
    save_confusion_matrix(model, X_test_s, y_test, output_path=str(images_dir / "confusion_matrix.png"))
    save_roc_pr_curves(model, X_test_s, y_test, output_path=str(images_dir / "roc_pr_curves.png"))

    if not skip_shap:
        save_shap_summary(model, X_bal, output_path=str(images_dir / "shap_summary.png"))
        save_shap_bar(model, X_bal, output_path=str(images_dir / "shap_bar.png"))

    pipeline = ChurnEndToEndModel(
        target_col=target_col,
        corr_drop_columns=tuple(corr_drop),
        rfm_scaler=rfm_scaler,
        kmeans=kmeans,
        low_value_cluster=low_value_cluster,
        x_scaler=scaler,
        selected_features=tuple(selected_features),
        classifier=model,
    )

    bundle = {
        "version": 2,
        "pipeline": pipeline,
    }
    joblib.dump(bundle, out_dir / "churn_bundle.joblib")

    metrics = {
        "bundle_version": 2,
        "split_policy": {
            "outer_test_size": DEFAULT_TEST_SIZE,
            "val_fraction_of_fit": DEFAULT_VAL_FRACTION_OF_FIT,
            "random_state": DEFAULT_RANDOM_STATE,
            "note": "Correlation + KMeans fit on fit split only; Optuna uses validation only.",
        },
        "benchmark_csv": str(benchmark_path.resolve()),
        "best_recall_validation": float(best_recall_val),
        "n_selected_features": len(selected_features),
        "selected_features": selected_features,
        "expected_input_columns": pipeline.raw_feature_names(),
        "low_value_cluster_excluded": low_value_cluster,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {"metrics": metrics, "out_dir": str(out_dir.resolve())}


def predict_row(bundle: dict[str, Any], features: dict[str, float]) -> dict[str, Any]:
    """Dispatch v2 end-to-end pipeline or legacy v1 keys."""
    if bundle.get("version") == 2 and bundle.get("pipeline") is not None:
        return bundle["pipeline"].predict_one(features)

    model = bundle["model"]
    scaler = bundle["scaler"]
    selected = list(bundle["selected_features"])
    cols = list(bundle["scaler_columns"])
    missing = [c for c in cols if c not in features]
    if missing:
        raise ValueError(f"Missing keys required for scaling: {missing[:12]}{'...' if len(missing) > 12 else ''}")

    row = pd.DataFrame([{c: float(features[c]) for c in cols}])
    Xs = pd.DataFrame(scaler.transform(row), columns=cols)
    Xp = Xs[selected]
    proba = model.predict_proba(Xp)[0, 1]
    label = int(model.predict(Xp)[0])
    return {"eligible": True, "churn_probability": float(proba), "churn_predicted": label}
