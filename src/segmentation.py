import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build RFM features from telecom reload behavior."""
    required_cols = ["reloads_inactive_days", "reloads_count", "reloads_sum"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required RFM columns: {missing}")

    rfm = df[required_cols].copy()
    rfm.columns = ["recency", "frequency", "monetary"]
    return rfm


def segment_customers_with_kmeans(
    df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 0,
) -> pd.DataFrame:
    """Create RFM-based customer segments using K-Means clustering."""
    result = df.copy()
    rfm = build_rfm_features(result)

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    result["cluster"] = kmeans.fit_predict(rfm_scaled)
    return result


def remove_low_value_cluster(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    low_value_cluster: int = 1,
) -> pd.DataFrame:
    """Remove low-value customer segment identified from RFM analysis."""
    if cluster_col not in df.columns:
        raise ValueError(f"Cluster column '{cluster_col}' not found")
    return df[df[cluster_col] != low_value_cluster].copy()
