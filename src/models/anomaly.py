import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy import stats


def zscore_detection(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    z = np.abs(stats.zscore(series.fillna(series.median())))
    return pd.Series(z > threshold, index=series.index, name="anomaly_zscore")


def iqr_detection(series: pd.Series, k: float = 2.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return ((series < lower) | (series > upper)).rename("anomaly_iqr")


def rolling_zscore_detection(series: pd.Series, window: int = 28, threshold: float = 3.0) -> pd.Series:
    rolling_mean = series.rolling(window, min_periods=7).mean()
    rolling_std = series.rolling(window, min_periods=7).std()
    z = (series - rolling_mean) / rolling_std.clip(lower=1e-6)
    return (z.abs() > threshold).rename("anomaly_rolling_z")


def isolation_forest_detection(df: pd.DataFrame, features: list, contamination: float = 0.04) -> pd.Series:
    X = df[features].fillna(0).values
    clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    preds = clf.fit_predict(X)
    return pd.Series(preds == -1, index=df.index, name="anomaly_iforest")


def detect_all(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df.copy().sort_values("date").reset_index(drop=True)
    series = df["total_claims"]

    df["anomaly_zscore"] = zscore_detection(series)
    df["anomaly_iqr"] = iqr_detection(series)
    df["anomaly_rolling_z"] = rolling_zscore_detection(series)

    iforest_features = ["total_claims", "total_handle_hours", "weather_event"]
    available = [f for f in iforest_features if f in df.columns]
    df["anomaly_iforest"] = isolation_forest_detection(df, available)

    anomaly_cols = ["anomaly_zscore", "anomaly_iqr", "anomaly_rolling_z", "anomaly_iforest"]
    df["anomaly_votes"] = df[anomaly_cols].sum(axis=1)
    df["is_anomaly"] = df["anomaly_votes"] >= 2

    return df


def anomaly_report(df: pd.DataFrame) -> pd.DataFrame:
    anomalies = df[df["is_anomaly"]].copy()
    mean_claims = df["total_claims"].mean()
    anomalies["deviation_pct"] = ((anomalies["total_claims"] - mean_claims) / mean_claims * 100).round(1)
    anomalies["direction"] = np.where(anomalies["total_claims"] > mean_claims, "spike", "drop")

    cols = ["date", "total_claims", "deviation_pct", "direction", "weather_event", "anomaly_votes"]
    available = [c for c in cols if c in anomalies.columns]
    return anomalies[available].sort_values("anomaly_votes", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from src.etl import load
    datasets = load()
    daily = datasets["daily_totals"]
    result = detect_all(daily)
    report = anomaly_report(result)

    print(f"Total anomalies detected: {result['is_anomaly'].sum()} / {len(result)} days")
    print(f"Weather-correlated: {(report['weather_event'] == 1).sum()}")
    print("\nTop anomalies:")
    print(report.head(10).to_string(index=False))
