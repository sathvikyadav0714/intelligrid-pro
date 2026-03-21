import pandas as pd


def load_data(train_path, weather_path, building_path):
    train = pd.read_csv(train_path)
    weather = pd.read_csv(weather_path)
    building = pd.read_csv(building_path)

    return train, weather, building


def merge_data(train, weather, building):
    # Step 1: merge building → adds site_id
    df = train.merge(
        building,
        how="left",
        on="building_id"
    )

    # Step 2: merge weather
    df = df.merge(
        weather,
        how="left",
        on=["site_id", "timestamp"]
    )

    return df


def clean_data(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    weather_cols = [
        "air_temperature",
        "dew_temperature",
        "sea_level_pressure",
        "wind_speed",
        "cloud_coverage",
        "precip_depth_1_hr"
    ]

    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if "year_built" in df.columns:
        df["year_built"] = df["year_built"].fillna(df["year_built"].median())

    if "floor_count" in df.columns:
        df["floor_count"] = df["floor_count"].fillna(1)

    return df