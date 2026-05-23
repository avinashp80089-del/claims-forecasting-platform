import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from datetime import datetime

REPORT_DIR = "reports"
COLORS = {
    "primary": "#1f4e79",
    "accent": "#2e75b6",
    "warn": "#c55a11",
    "good": "#375623",
    "anomaly": "#FF4444",
    "forecast": "#2e75b6",
    "actual": "#1f4e79",
    "ci": "#bdd7ee",
}


def _save(fig, name):
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_forecast_vs_actual(test_df, preds_dict, sarima_model=None):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("Claims Demand Forecast — Model Comparison", fontsize=14, fontweight="bold", color=COLORS["primary"])

    actual = test_df["total_claims"].values
    dates = pd.to_datetime(test_df["date"])

    model_names = ["baseline", "ets", "xgboost", "ensemble"]
    model_labels = ["Baseline MA28", "ETS", "XGBoost", "Ensemble (SARIMA+XGB)"]
    model_colors = ["#aaa", "#e07b39", "#4caf50", COLORS["accent"]]

    ax = axes[0]
    ax.plot(dates, actual, color=COLORS["actual"], lw=2, label="Actual", zorder=5)

    if sarima_model:
        lo, hi = sarima_model.confidence_intervals(len(actual))
        ax.fill_between(dates, lo, hi, alpha=0.2, color=COLORS["ci"], label="SARIMA 95% CI")

    for name, label, color in zip(model_names, model_labels, model_colors):
        if name in preds_dict:
            ax.plot(dates, preds_dict[name][1][:len(actual)], lw=1.5, linestyle="--",
                    color=color, label=label, alpha=0.85)

    ax.set_title("All Models vs Actual", fontweight="bold")
    ax.set_ylabel("Daily Claims")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ensemble_pred = preds_dict["ensemble"][1][:len(actual)]
    residuals = actual - ensemble_pred
    ax.bar(dates, residuals, color=np.where(residuals > 0, COLORS["warn"], COLORS["good"]), alpha=0.7, width=0.8)
    ax.axhline(0, color="black", lw=1)
    ax.axhline(residuals.std() * 2, color="red", lw=1, linestyle=":", label="+2σ")
    ax.axhline(-residuals.std() * 2, color="red", lw=1, linestyle=":", label="-2σ")
    ax.set_title("Ensemble Residuals (Actual − Forecast)", fontweight="bold")
    ax.set_ylabel("Error (claims)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    cum_actual = np.cumsum(actual)
    cum_forecast = np.cumsum(ensemble_pred)
    ax.plot(dates, cum_actual, lw=2, color=COLORS["actual"], label="Cumulative Actual")
    ax.plot(dates, cum_forecast, lw=2, linestyle="--", color=COLORS["accent"], label="Cumulative Forecast")
    ax.fill_between(dates, cum_actual, cum_forecast, alpha=0.15, color=COLORS["warn"])
    ax.set_title("Cumulative Claims Tracking", fontweight="bold")
    ax.set_ylabel("Cumulative Claims")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save(fig, "forecast_vs_actual.png")


def plot_capacity_dashboard(capacity_df):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Claims Capacity Planning Dashboard", fontsize=14, fontweight="bold", color=COLORS["primary"])
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    dates = pd.to_datetime(capacity_df["date"])

    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(dates, capacity_df["required_fte"], alpha=0.3, color=COLORS["accent"], label="Required FTE")
    ax1.plot(dates, capacity_df["required_fte"], lw=2, color=COLORS["accent"])
    ax1.axhline(85, color=COLORS["warn"], lw=2, linestyle="--", label="Current Staff (85 FTE)")
    ax1.fill_between(dates, capacity_df["required_fte"], 85,
                     where=capacity_df["required_fte"] > 85,
                     alpha=0.3, color="red", label="Understaffed")
    ax1.fill_between(dates, capacity_df["required_fte"], 85,
                     where=capacity_df["required_fte"] < 85,
                     alpha=0.2, color="green", label="Overstaffed")
    ax1.set_title("FTE Demand vs Capacity", fontweight="bold")
    ax1.set_ylabel("Full-Time Equivalents")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    gap = capacity_df["staffing_gap"]
    ax2.hist(gap, bins=30, color=COLORS["accent"], edgecolor="white", alpha=0.8)
    ax2.axvline(0, color="black", lw=2)
    ax2.axvline(gap.mean(), color=COLORS["warn"], lw=2, linestyle="--", label=f"Mean: {gap.mean():.1f}")
    ax2.set_title("Staffing Gap Distribution", fontweight="bold")
    ax2.set_xlabel("Gap (Required − Available FTE)")
    ax2.set_ylabel("Days")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, 1])
    util = capacity_df["utilization_pct"].clip(upper=150)
    ax3.plot(dates, util, lw=1.5, color=COLORS["primary"])
    ax3.axhline(85, color="green", lw=2, linestyle="--", label="Target 85%")
    ax3.axhline(100, color="red", lw=2, linestyle="--", label="100% (overload)")
    ax3.fill_between(dates, util, 85, where=util > 100, alpha=0.2, color="red")
    ax3.set_title("Staff Utilization %", fontweight="bold")
    ax3.set_ylabel("Utilization (%)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    return _save(fig, "capacity_dashboard.png")


def plot_anomaly_report(anomaly_df):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Anomaly Detection — Claims Volume", fontsize=14, fontweight="bold", color=COLORS["primary"])

    dates = pd.to_datetime(anomaly_df["date"])
    series = anomaly_df["total_claims"]
    anomalies = anomaly_df[anomaly_df["is_anomaly"]]

    ax = axes[0]
    ax.plot(dates, series, lw=1.5, color=COLORS["primary"], alpha=0.8, label="Daily Claims")
    ax.scatter(pd.to_datetime(anomalies["date"]), anomalies["total_claims"],
               color=COLORS["anomaly"], s=60, zorder=5, label=f"Anomaly ({len(anomalies)})")
    rolling_mean = series.rolling(28).mean()
    rolling_std = series.rolling(28).std()
    ax.fill_between(dates, rolling_mean - 2 * rolling_std, rolling_mean + 2 * rolling_std,
                    alpha=0.15, color=COLORS["accent"], label="±2σ band")
    ax.plot(dates, rolling_mean, lw=1, linestyle="--", color=COLORS["accent"], alpha=0.6)
    ax.set_title("Claims Volume with Anomalies Flagged", fontweight="bold")
    ax.set_ylabel("Daily Claims")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    method_counts = {
        "Z-Score": anomaly_df["anomaly_zscore"].sum(),
        "IQR": anomaly_df["anomaly_iqr"].sum(),
        "Rolling Z": anomaly_df["anomaly_rolling_z"].sum(),
        "Isolation Forest": anomaly_df["anomaly_iforest"].sum(),
        "Consensus (≥2)": anomaly_df["is_anomaly"].sum(),
    }
    colors = [COLORS["accent"]] * 4 + [COLORS["warn"]]
    bars = ax.barh(list(method_counts.keys()), list(method_counts.values()), color=colors)
    for bar, val in zip(bars, method_counts.values()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, str(val),
                va="center", fontsize=9)
    ax.set_title("Anomalies Detected by Method", fontweight="bold")
    ax.set_xlabel("Number of Anomalous Days")
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    return _save(fig, "anomaly_report.png")


def plot_model_leaderboard(leaderboard_df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Performance Leaderboard", fontsize=13, fontweight="bold", color=COLORS["primary"])

    metrics = ["MAE", "RMSE", "MAPE%"]
    colors = [COLORS["accent"], COLORS["primary"], COLORS["warn"]]

    for ax, metric, color in zip(axes, metrics, colors):
        sorted_df = leaderboard_df.sort_values(metric)
        bars = ax.barh(sorted_df["model"], sorted_df[metric], color=color, alpha=0.8, edgecolor="white")
        for bar, val in zip(bars, sorted_df[metric]):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=8)
        ax.set_title(metric, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    return _save(fig, "model_leaderboard.png")


def generate_executive_summary(leaderboard_df, capacity_df, anomaly_df):
    best_model = leaderboard_df.iloc[0]
    understaffed_days = capacity_df["understaffed"].sum()
    total_days = len(capacity_df)
    anomaly_count = anomaly_df["is_anomaly"].sum()
    weather_anomalies = anomaly_df[anomaly_df["is_anomaly"] & (anomaly_df["weather_event"] == 1)].shape[0]

    summary = f"""
CLAIMS FORECASTING PLATFORM — EXECUTIVE SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*60}

BEST MODEL: {best_model['model']}
  MAE  : {best_model['MAE']:.1f} claims/day
  RMSE : {best_model['RMSE']:.1f} claims/day
  MAPE : {best_model['MAPE%']:.1f}%

CAPACITY ANALYSIS ({total_days} days evaluated)
  Understaffed days : {understaffed_days} ({understaffed_days/total_days*100:.1f}%)
  Avg utilization   : {capacity_df['utilization_pct'].mean():.1f}%
  Avg staffing gap  : {capacity_df['staffing_gap'].mean():.1f} FTE

ANOMALY DETECTION
  Total anomalies   : {anomaly_count} days ({anomaly_count/len(anomaly_df)*100:.1f}%)
  Weather-driven    : {weather_anomalies} ({weather_anomalies/max(anomaly_count,1)*100:.0f}% of anomalies)

REPORTS SAVED TO: {os.path.abspath(REPORT_DIR)}/
{'='*60}
"""
    print(summary)
    with open(os.path.join(REPORT_DIR, "executive_summary.txt"), "w") as f:
        f.write(summary)
    return summary
