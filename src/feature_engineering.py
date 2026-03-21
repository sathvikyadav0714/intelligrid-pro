import numpy as np


def add_time_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    
    return df


def add_log_target(df):
    df["log_meter_reading"] = np.log1p(df["meter_reading"])
    return df


def select_features(df):
    # safe feature selection (only existing columns)
    possible_features = [
        "square_feet",
        "air_temperature",
        "cloud_coverage",
        "dew_temperature",
        "hour",
        "dayofweek",
        "month"
    ]

    features = [col for col in possible_features if col in df.columns]

    X = df[features]
    y = df["log_meter_reading"]

    return X, y