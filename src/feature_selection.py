import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold


def select_features_rfecv(
    X: pd.DataFrame,
    y,
    scoring: str = "recall",
    random_state: int = 23,
):
    """Select important features using RFECV with Random Forest."""
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    clf = RandomForestClassifier(n_jobs=-1, random_state=random_state)

    selector = RFECV(
        estimator=clf,
        step=1,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    selector.fit(X, y)

    selected_features = X.columns[selector.support_].tolist()
    return selected_features, selector
