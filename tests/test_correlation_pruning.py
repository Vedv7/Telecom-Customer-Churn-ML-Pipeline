import pandas as pd

from telecom_churn.preprocessing import (
    apply_correlation_drop,
    fit_correlated_columns_to_drop,
)


def test_fit_correlated_columns_to_drop_learns_from_train_only():
    train = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [1.01, 2.01, 3.01, 4.01],
            "noise": [0.0, 1.0, 0.0, 1.0],
            "churn": [0, 1, 0, 1],
        }
    )
    drops = fit_correlated_columns_to_drop(train, target_col="churn", threshold=0.99)
    assert len(drops) >= 1
    assert "churn" not in drops

    out = apply_correlation_drop(train, drops)
    assert out.shape[1] == train.shape[1] - len(drops)
