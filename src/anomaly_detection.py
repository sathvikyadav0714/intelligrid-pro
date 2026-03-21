import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df, feature_cols, contamination=0.01):

    df_model = df[feature_cols].copy()

    # force numeric
    df_model = df_model.apply(pd.to_numeric, errors='coerce')

    # remove inf
    df_model = df_model.replace([np.inf, -np.inf], np.nan)

    # 🔥 FIX 1: fill with median
    df_model = df_model.fillna(df_model.median())

    # 🔥 FIX 2: if column still NaN → fill 0
    df_model = df_model.fillna(0)

    # 🔥 FIX 3: final safety
    df_model = df_model.astype(float)

    # 🔥 CRITICAL: check again
    if df_model.isnull().sum().sum() > 0:
        df_model = df_model.fillna(0)

    if len(df_model) == 0:
        df["anomaly"] = 0
        return df

    sample_size = min(200000, len(df_model))
    df_sample = df_model.sample(sample_size, random_state=42)

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    model.fit(df_sample)

    preds = model.predict(df_model)

    df["anomaly"] = np.where(preds == -1, 1, 0)

    return df


def get_anomaly_summary(df):
    total = len(df)
    anomalies = df["anomaly"].sum()

    return {
        "total_points": total,
        "anomalies": int(anomalies),
        "percentage": round((anomalies / total) * 100, 2)
    }