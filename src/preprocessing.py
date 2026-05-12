import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce dataframe memory usage by downcasting numeric columns."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).startswith("int"):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")
    return df


def load_dataset(path: str) -> pd.DataFrame:
    """Load churn dataset from Excel or CSV."""
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file format. Use .xlsx, .xls, or .csv")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleanup for customer churn dataset."""
    df = df.copy()
    drop_cols = [col for col in ["year", "user_account_id"] if col in df.columns]
    df.drop(drop_cols, axis=1, inplace=True)
    df = reduce_mem_usage(df)
    return df


def remove_highly_correlated_features(
    df: pd.DataFrame,
    target_col: str = "churn",
    threshold: float = 0.85,
) -> pd.DataFrame:
    """Remove highly correlated numeric features while preserving target column."""
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] >= threshold)]
    to_drop = [col for col in to_drop if col != target_col]
    return df.drop(columns=to_drop, errors="ignore")


def prepare_train_test_data(
    df: pd.DataFrame,
    target_col: str = "churn",
    test_size: float = 0.2,
    random_state: int = 23,
):
    """Split, scale, and balance the dataset using SMOTE and random undersampling."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.dtype == object or str(y.dtype) == "category":
        y = y.map({"No": 0, "Yes": 1, "no": 0, "yes": 1}).fillna(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    smote = SMOTE(sampling_strategy=0.6, random_state=random_state)
    X_smote, y_smote = smote.fit_resample(X_train_scaled, y_train)

    under = RandomUnderSampler(random_state=random_state)
    X_balanced, y_balanced = under.fit_resample(X_smote, y_smote)

    return X_balanced, X_test_scaled, y_balanced, y_test, scaler
