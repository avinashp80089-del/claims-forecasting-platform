"""
End-to-end pipeline: generate → ETL → forecast → capacity → anomaly → report
Run:  python -m src.pipeline
      python -m src.pipeline --backfill 2024-01-01 2024-06-30
"""
import argparse
import sys
import time
from datetime import datetime

import pandas as pd

from data.generate import generate_claims, generate_staffing, save_to_sqlite
from src.etl import load, validate, build_model_dataset
from src.models.forecaster import evaluate_all, SARIMAForecaster
from src.models.capacity import build_capacity_plan, sensitivity_analysis
from src.models.anomaly import detect_all, anomaly_report
from src.reporting import (
    plot_forecast_vs_actual,
    plot_capacity_dashboard,
    plot_anomaly_report,
    plot_model_leaderboard,
    generate_executive_summary,
)


def run_pipeline(backfill_start=None, backfill_end=None):
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f" CLAIMS FORECASTING PIPELINE  |  {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{'='*60}\n")

    print("[1/6] Generating / loading data...")
    if backfill_start and backfill_end:
        print(f"      Backfill mode: {backfill_start} → {backfill_end}")
        claims = generate_claims(backfill_start, backfill_end)
        staffing = generate_staffing(backfill_start, backfill_end)
    else:
        claims = generate_claims()
        staffing = generate_staffing()

    save_to_sqlite(claims, staffing)

    print("[2/6] Running ETL & validation...")
    datasets = load()
    for name, df in datasets.items():
        validate(df, name)

    model_df = build_model_dataset(datasets["daily_totals"])
    print(f"      Model dataset: {model_df.shape[0]} rows × {model_df.shape[1]} features")

    print("[3/6] Training & evaluating models...")
    leaderboard, preds, test_df, xgb_model = evaluate_all(model_df)
    print(leaderboard.to_string(index=False))

    print("\n[4/6] Building capacity plan...")
    forecast_df = test_df[["date"]].copy()
    forecast_df["forecast"] = preds["ensemble"][1]
    capacity_df = build_capacity_plan(forecast_df)
    print(sensitivity_analysis(350))

    print("\n[5/6] Running anomaly detection...")
    anomaly_df = detect_all(datasets["daily_totals"])
    report_df = anomaly_report(anomaly_df)
    print(f"      Detected {len(report_df)} anomalies")
    if not report_df.empty:
        print(report_df.head(5).to_string(index=False))

    print("\n[6/6] Generating reports...")
    sarima = SARIMAForecaster()
    sarima.fit(model_df.set_index("date")["total_claims"].iloc[:-90])

    plot_forecast_vs_actual(test_df, preds, sarima)
    plot_capacity_dashboard(capacity_df)
    plot_anomaly_report(anomaly_df)
    plot_model_leaderboard(leaderboard)
    generate_executive_summary(leaderboard, capacity_df, anomaly_df)

    elapsed = time.time() - t0
    print(f"\nPipeline complete in {elapsed:.1f}s. Reports in ./reports/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claims Forecasting Pipeline")
    parser.add_argument("--backfill", nargs=2, metavar=("START", "END"),
                        help="Backfill date range: YYYY-MM-DD YYYY-MM-DD")
    args = parser.parse_args()

    if args.backfill:
        run_pipeline(backfill_start=args.backfill[0], backfill_end=args.backfill[1])
    else:
        run_pipeline()
