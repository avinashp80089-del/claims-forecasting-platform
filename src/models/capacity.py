import numpy as np
import pandas as pd


WORK_HOURS_PER_DAY = 8.0
DEFAULT_SHRINKAGE = 0.15
DEFAULT_HANDLE_TIME_MIN = 52.0
OCCUPANCY_TARGET = 0.85


def claims_to_fte(
    claim_count: float,
    handle_time_min: float = DEFAULT_HANDLE_TIME_MIN,
    shrinkage: float = DEFAULT_SHRINKAGE,
    occupancy: float = OCCUPANCY_TARGET,
    hours_per_day: float = WORK_HOURS_PER_DAY,
) -> float:
    required_hours = (claim_count * handle_time_min) / 60.0
    productive_hours_per_fte = hours_per_day * (1 - shrinkage) * occupancy
    return required_hours / productive_hours_per_fte if productive_hours_per_fte > 0 else 0.0


def build_capacity_plan(
    forecast_df: pd.DataFrame,
    current_staff: int = 85,
    handle_time_min: float = DEFAULT_HANDLE_TIME_MIN,
    shrinkage: float = DEFAULT_SHRINKAGE,
) -> pd.DataFrame:
    df = forecast_df.copy()

    df["required_fte"] = df["forecast"].apply(
        lambda x: claims_to_fte(x, handle_time_min, shrinkage)
    )
    df["required_fte"] = df["required_fte"].round(1)

    df["staffing_gap"] = df["required_fte"] - current_staff
    df["overstaffed"] = (df["staffing_gap"] < -5).astype(int)
    df["understaffed"] = (df["staffing_gap"] > 5).astype(int)

    df["recommended_staff"] = np.ceil(df["required_fte"] * 1.05).astype(int)

    df["utilization_pct"] = (
        (current_staff * WORK_HOURS_PER_DAY * (1 - shrinkage))
        / ((df["forecast"] * handle_time_min / 60.0).clip(lower=0.1))
        * 100
    ).round(1).clip(upper=200)

    return df


def sensitivity_analysis(
    base_forecast: float,
    handle_times: list = [45, 52, 60, 75],
    shrinkage_rates: list = [0.10, 0.15, 0.20],
) -> pd.DataFrame:
    rows = []
    for ht in handle_times:
        for sr in shrinkage_rates:
            fte = claims_to_fte(base_forecast, ht, sr)
            rows.append({
                "handle_time_min": ht,
                "shrinkage_pct": f"{int(sr*100)}%",
                "required_fte": round(fte, 1),
            })
    return pd.DataFrame(rows).pivot(
        index="handle_time_min", columns="shrinkage_pct", values="required_fte"
    )


def weekly_staffing_summary(capacity_df: pd.DataFrame) -> pd.DataFrame:
    df = capacity_df.copy()
    df["week"] = pd.to_datetime(df["date"]).dt.to_period("W")
    return (
        df.groupby("week")
        .agg(
            avg_forecast=("forecast", "mean"),
            avg_required_fte=("required_fte", "mean"),
            max_required_fte=("required_fte", "max"),
            understaffed_days=("understaffed", "sum"),
            overstaffed_days=("overstaffed", "sum"),
        )
        .round(1)
        .reset_index()
    )


if __name__ == "__main__":
    from src.etl import load, build_model_dataset
    from src.models.forecaster import evaluate_all

    datasets = load()
    model_df = build_model_dataset(datasets["daily_totals"])
    _, preds, test_df, _ = evaluate_all(model_df)

    forecast_df = test_df[["date"]].copy()
    forecast_df["forecast"] = preds["ensemble"][1]

    capacity = build_capacity_plan(forecast_df)
    print(capacity[["date", "forecast", "required_fte", "staffing_gap", "utilization_pct"]].head(14).to_string())

    print("\nSensitivity Analysis (FTE needed for 350 claims/day):")
    print(sensitivity_analysis(350))
