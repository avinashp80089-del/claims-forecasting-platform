import sqlite3
import pandas as pd
import numpy as np


QUERIES = {
    "daily_totals": """
        SELECT
            date,
            SUM(claim_count)                                 AS total_claims,
            SUM(claim_count * avg_handle_time_min) / 60.0   AS total_handle_hours,
            AVG(avg_handle_time_min)                         AS avg_handle_time,
            MAX(weather_event)                               AS weather_event,
            day_of_week,
            month,
            week_of_year
        FROM claims
        GROUP BY date, day_of_week, month, week_of_year
        ORDER BY date
    """,
    "weekly_by_type": """
        SELECT
            strftime('%Y-%W', date)  AS year_week,
            claim_type,
            SUM(claim_count)         AS weekly_claims,
            AVG(avg_handle_time_min) AS avg_handle_time,
            SUM(weather_event)       AS weather_days
        FROM claims
        GROUP BY year_week, claim_type
        ORDER BY year_week
    """,
    "regional_summary": """
        SELECT
            region,
            strftime('%Y-%m', date)  AS year_month,
            claim_type,
            SUM(claim_count)         AS monthly_claims,
            AVG(avg_handle_time_min) AS avg_handle_time
        FROM claims
        GROUP BY region, year_month, claim_type
        ORDER BY region, year_month
    """,
    "capacity_gap": """
        WITH daily AS (
            SELECT
                c.date,
                SUM(c.claim_count * c.avg_handle_time_min) / 60.0 AS required_hours,
                s.staff_available,
                s.shrinkage_pct
            FROM claims c
            JOIN staffing s ON c.date = s.date
            GROUP BY c.date, s.staff_available, s.shrinkage_pct
        )
        SELECT
            date,
            required_hours,
            staff_available,
            shrinkage_pct,
            ROUND(staff_available * (1 - shrinkage_pct) * 8.0, 2) AS available_hours,
            ROUND(required_hours - staff_available * (1 - shrinkage_pct) * 8.0, 2) AS capacity_gap
        FROM daily
        ORDER BY date
    """,
    "rolling_7d": """
        SELECT
            date,
            total_claims,
            AVG(total_claims) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7d_avg,
            AVG(total_claims) OVER (ORDER BY date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS rolling_28d_avg,
            MAX(total_claims) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7d_max,
            MIN(total_claims) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7d_min
        FROM (
            SELECT date, SUM(claim_count) AS total_claims
            FROM claims GROUP BY date
        )
        ORDER BY date
    """,
}


def load(db_path="data/claims.db") -> dict[str, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    result = {name: pd.read_sql(sql, conn, parse_dates=["date"]) for name, sql in QUERIES.items()}
    conn.close()
    return result


def validate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    issues = []

    null_counts = df.isnull().sum()
    if null_counts.any():
        issues.append(f"Nulls: {null_counts[null_counts > 0].to_dict()}")

    if "total_claims" in df.columns:
        neg = (df["total_claims"] < 0).sum()
        if neg:
            issues.append(f"{neg} negative claim counts")

        zscore = (df["total_claims"] - df["total_claims"].mean()) / df["total_claims"].std()
        outliers = (zscore.abs() > 4).sum()
        if outliers:
            issues.append(f"{outliers} extreme outliers (|z| > 4)")

    if "date" in df.columns and df["date"].duplicated().any():
        issues.append(f"{df['date'].duplicated().sum()} duplicate dates")

    status = "PASS" if not issues else "WARN"
    print(f"[{status}] {name}: {', '.join(issues) if issues else 'all checks passed'}")
    return df


def build_model_dataset(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df.copy().sort_values("date").reset_index(drop=True)

    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df["total_claims"].shift(lag)

    for window in [7, 14, 28]:
        df[f"roll_mean_{window}"] = df["total_claims"].shift(1).rolling(window).mean()
        df[f"roll_std_{window}"] = df["total_claims"].shift(1).rolling(window).std()

    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["quarter"] = pd.to_datetime(df["date"]).dt.quarter
    df["day_of_year"] = pd.to_datetime(df["date"]).dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    df["same_dow_last_week"] = df["total_claims"].shift(7)
    df["same_dow_last_year"] = df["total_claims"].shift(364)

    return df.dropna()


if __name__ == "__main__":
    datasets = load()
    for name, df in datasets.items():
        validate(df, name)
    model_df = build_model_dataset(datasets["daily_totals"])
    print(f"\nModel dataset: {model_df.shape[0]} rows, {model_df.shape[1]} features")
    model_df.to_csv("data/model_dataset.csv", index=False)
