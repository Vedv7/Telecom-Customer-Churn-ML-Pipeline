import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from telecom_churn.end_to_end import ChurnEndToEndModel
from telecom_churn.segmentation import assign_cluster_column, fit_rfm_kmeans


def test_churn_end_to_end_predict_one_low_cluster_ineligible():
    """When cluster matches low_value_cluster, model is not applied."""
    df = pd.DataFrame(
        {
            "reloads_inactive_days": [1.0, 2.0, 3.0],
            "reloads_count": [10.0, 20.0, 30.0],
            "reloads_sum": [100.0, 200.0, 300.0],
            "extra": [0.0, 1.0, 2.0],
            "churn": [0, 1, 0],
        }
    )
    rfm_scaler, kmeans = fit_rfm_kmeans(df, n_clusters=3, random_state=0)

    df_tagged = assign_cluster_column(df, rfm_scaler, kmeans)
    X = df_tagged.drop(columns=["churn"])
    x_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X)
    clf = XGBClassifier(n_estimators=5, max_depth=2, verbosity=0, eval_metric="logloss")
    clf.fit(Xs, df_tagged["churn"])

    row0_cluster = int(df_tagged.iloc[0]["cluster"])
    model = ChurnEndToEndModel(
        target_col="churn",
        corr_drop_columns=(),
        rfm_scaler=rfm_scaler,
        kmeans=kmeans,
        low_value_cluster=row0_cluster,
        x_scaler=x_scaler,
        selected_features=tuple(X.columns),
        classifier=clf,
    )
    raw = df.iloc[0].to_dict()
    out = model.predict_one(raw)
    assert out["eligible"] is False
    assert out["reason"] == "low_value_segment"


def test_churn_end_to_end_predict_one_eligible():
    df = pd.DataFrame(
        {
            "reloads_inactive_days": [1.0, 2.0, 3.0, 4.0],
            "reloads_count": [10.0, 20.0, 30.0, 40.0],
            "reloads_sum": [100.0, 200.0, 300.0, 400.0],
            "extra": [0.0, 1.0, 2.0, 3.0],
            "churn": [0, 1, 0, 1],
        }
    )
    rfm_scaler, kmeans = fit_rfm_kmeans(df, n_clusters=3, random_state=1)

    df_tagged = assign_cluster_column(df, rfm_scaler, kmeans)
    X = df_tagged.drop(columns=["churn"])
    x_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X)
    clf = XGBClassifier(n_estimators=20, max_depth=2, verbosity=0, eval_metric="logloss")
    clf.fit(Xs, df_tagged["churn"])

    model = ChurnEndToEndModel(
        target_col="churn",
        corr_drop_columns=(),
        rfm_scaler=rfm_scaler,
        kmeans=kmeans,
        low_value_cluster=99,
        x_scaler=x_scaler,
        selected_features=tuple(X.columns),
        classifier=clf,
    )
    raw = df.iloc[0].to_dict()
    out = model.predict_one(raw)
    assert out["eligible"] is True
    assert "churn_probability" in out
    assert 0.0 <= out["churn_probability"] <= 1.0
