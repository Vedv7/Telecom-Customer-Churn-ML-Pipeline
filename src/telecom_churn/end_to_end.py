from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from telecom_churn.preprocessing import apply_correlation_drop, clean_dataset
from telecom_churn.segmentation import assign_cluster_column


@dataclass
class ChurnEndToEndModel:
    """Single artifact: raw feature row(s) → churn score (matches training-time preprocessing)."""

    target_col: str
    corr_drop_columns: tuple[str, ...]
    rfm_scaler: StandardScaler
    kmeans: KMeans
    low_value_cluster: int
    x_scaler: StandardScaler
    selected_features: tuple[str, ...]
    classifier: XGBClassifier

    def raw_feature_names(self) -> list[str]:
        """Feature columns expected after clean + correlation + clustering (excluding target)."""
        return list(self.x_scaler.feature_names_in_)

    def _preprocess_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = clean_dataset(df.copy())
        out = apply_correlation_drop(out, list(self.corr_drop_columns))
        return assign_cluster_column(out, self.rfm_scaler, self.kmeans)

    def _score_preprocessed_single(self, row_df: pd.DataFrame) -> dict[str, Any]:
        """``row_df`` must be a single-row frame after :meth:`_preprocess_frame`."""
        cluster = int(row_df.iloc[0]["cluster"])
        if cluster == self.low_value_cluster:
            return {
                "eligible": False,
                "reason": "low_value_segment",
                "cluster": cluster,
            }

        X = row_df.drop(columns=[self.target_col], errors="ignore")
        required = list(self.x_scaler.feature_names_in_)
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise ValueError(
                f"Missing keys after preprocessing: {missing[:20]}{'...' if len(missing) > 20 else ''}"
            )

        Xa = X[required].astype(float)
        Xs = self.x_scaler.transform(Xa)
        Xscaled = pd.DataFrame(Xs, columns=required, index=Xa.index)
        Xp = Xscaled[list(self.selected_features)]
        proba = float(self.classifier.predict_proba(Xp)[0, 1])
        label = int(self.classifier.predict(Xp)[0])
        return {
            "eligible": True,
            "cluster": cluster,
            "churn_probability": proba,
            "churn_predicted": label,
        }

    def predict_one(self, features: dict[str, Any]) -> dict[str, Any]:
        """Score one customer from **raw** workbook-like fields (``churn`` optional, ignored)."""
        proc = self._preprocess_frame(pd.DataFrame([features]))
        return self._score_preprocessed_single(proc)

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch score; returns one row per input with ``eligible`` / scores."""
        proc = self._preprocess_frame(df)
        rows = [self._score_preprocessed_single(proc.iloc[[i]]) for i in range(len(proc))]
        return pd.DataFrame(rows)
