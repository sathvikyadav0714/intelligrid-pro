import pandas as pd


def get_peak_hours(df):
    peak = (
        df.groupby("hour")["meter_reading"]
        .mean()
        .sort_values(ascending=False)
        .head(3)
    )
    return peak


def get_high_usage_buildings(df, n=5):
    high_usage = (
        df.groupby("building_id")["meter_reading"]
        .mean()
        .sort_values(ascending=False)
        .head(n)
    )
    return high_usage


def get_anomaly_buildings(df, n=5):
    anomaly_counts = (
        df.groupby("building_id")["anomaly"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )
    return anomaly_counts


def generate_recommendations(df):
    recommendations = []

    # peak hours
    peak_hours = df.groupby("hour")["meter_reading"].mean().idxmax()
    recommendations.append(
        f"Peak energy usage occurs at hour {peak_hours}. Consider load shifting."
    )

    # anomaly heavy buildings
    anomaly_counts = df.groupby("building_id")["anomaly"].sum()
    worst_building = anomaly_counts.idxmax()

    recommendations.append(
        f"Building {worst_building} shows highest anomalies. Needs inspection."
    )

    # high usage building
    high_usage = df.groupby("building_id")["meter_reading"].mean().idxmax()

    recommendations.append(
        f"Building {high_usage} consumes highest average energy. Optimize systems."
    )

    return recommendations