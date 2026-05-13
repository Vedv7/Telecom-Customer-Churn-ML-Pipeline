import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from telecom_churn.train import predict_row


def test_predict_row_minimal_bundle():
    df = pd.DataFrame({"f1": [0.0, 2.0, 1.0], "f2": [1.0, 0.0, 1.0]})
    y = [0, 1, 0]
    scaler = StandardScaler()
    scaler.fit(df)
    Xs = pd.DataFrame(scaler.transform(df), columns=["f1", "f2"])
    model = LogisticRegression()
    model.fit(Xs, y)
    bundle = {
        "model": model,
        "scaler": scaler,
        "scaler_columns": ["f1", "f2"],
        "selected_features": ["f1", "f2"],
    }
    out = predict_row(bundle, {"f1": 0.5, "f2": 0.5})
    assert out.get("eligible") is True
    assert "churn_probability" in out
    assert "churn_predicted" in out
    assert 0.0 <= out["churn_probability"] <= 1.0
