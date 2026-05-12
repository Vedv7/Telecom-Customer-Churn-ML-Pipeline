import argparse
import pandas as pd

from src.preprocessing import (
    load_dataset,
    clean_dataset,
    remove_highly_correlated_features,
    prepare_train_test_data,
)
from src.segmentation import segment_customers_with_kmeans, remove_low_value_cluster
from src.feature_selection import select_features_rfecv
from src.model_training import benchmark_models, tune_xgboost_with_optuna, train_xgboost
from src.evaluation import print_classification_metrics, save_confusion_matrix, save_roc_pr_curves
from src.explainability import save_shap_summary, save_shap_bar


def run_pipeline(data_path: str, tune_trials: int = 100):
    print("Loading dataset...")
    df = load_dataset(data_path)

    print("Cleaning dataset...")
    df = clean_dataset(df)
    df = remove_highly_correlated_features(df, target_col="churn", threshold=0.85)

    print("Running RFM segmentation...")
    df = segment_customers_with_kmeans(df, n_clusters=3)
    df = remove_low_value_cluster(df, low_value_cluster=1)

    print("Preparing train/test data...")
    X_train, X_test, y_train, y_test, _ = prepare_train_test_data(df, target_col="churn")

    print("Selecting features with RFECV...")
    selected_features, _ = select_features_rfecv(X_train, y_train)
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    print(f"Selected {len(selected_features)} features")

    print("Benchmarking models...")
    benchmark = benchmark_models(X_train, y_train)
    benchmark.to_csv("model_benchmark.csv", index=False)
    print(benchmark)

    print("Tuning XGBoost...")
    best_params, best_recall = tune_xgboost_with_optuna(
        X_train,
        y_train,
        X_test,
        y_test,
        n_trials=tune_trials,
    )
    print("Best params:", best_params)
    print("Best recall:", best_recall)

    print("Training final XGBoost model...")
    model = train_xgboost(X_train, y_train, best_params)

    print("Evaluating model...")
    print_classification_metrics(model, X_test, y_test)
    save_confusion_matrix(model, X_test, y_test)
    save_roc_pr_curves(model, X_test, y_test)

    print("Generating SHAP explainability plots...")
    save_shap_summary(model, X_train)
    save_shap_bar(model, X_train)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telecom Customer Churn ML Pipeline")
    parser.add_argument("--data", required=True, help="Path to churn dataset CSV/XLSX")
    parser.add_argument("--trials", type=int, default=100, help="Optuna tuning trials")
    args = parser.parse_args()

    run_pipeline(args.data, args.trials)
