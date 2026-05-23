# Claims Demand & Capacity Forecasting Platform

End-to-end pipeline for insurance claims demand forecasting, staffing optimization, anomaly detection, and automated reporting — built to production standards with modular Python, SQL-driven ETL, and multi-model benchmarking.

---

## What It Does

| Module | Description |
|---|---|
| **Data Pipeline (ETL)** | Generates synthetic claims data, loads into SQLite, transforms with SQL window functions and CTEs |
| **Demand Forecasting** | Benchmarks 5 models: Baseline MA, ETS, SARIMA(1,1,1)(1,1,1)7, XGBoost, and SARIMA+XGB Ensemble |
| **Capacity Planning** | Converts forecasted claim volume to FTE requirements accounting for handle time, shrinkage, and occupancy targets |
| **Anomaly Detection** | Consensus voting across Z-Score, IQR, Rolling Z-Score, and Isolation Forest |
| **Automated Reporting** | Generates 4 charts + executive summary text report on every pipeline run |

---

## Project Structure

```
claims-forecasting-platform/
├── data/
│   └── generate.py          # Synthetic claims + staffing data with seasonal patterns
├── src/
│   ├── etl.py               # SQL queries, data validation, feature engineering
│   ├── reporting.py         # Matplotlib dashboards, executive summary
│   ├── pipeline.py          # End-to-end runner (supports --backfill)
│   └── models/
│       ├── forecaster.py    # Baseline, ETS, SARIMA, XGBoost, Ensemble
│       ├── capacity.py      # FTE calculator, staffing gap, sensitivity analysis
│       └── anomaly.py       # Multi-method anomaly detection + consensus voting
├── tests/
│   └── test_forecasting.py  # 20 unit tests across all modules
├── reports/                 # Auto-generated on each run
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Generate data and run full pipeline
python -m src.pipeline

# Backfill a specific date range
python -m src.pipeline --backfill 2024-01-01 2024-06-30

# Run tests
pytest tests/ -v --cov=src
```

---

## Models & Performance

Five models are benchmarked on a 90-day hold-out set:

| Model | MAE | RMSE | MAPE% |
|---|---|---|---|
| Baseline MA28 | ~42 | ~55 | ~14% |
| ETS (Holt-Winters) | ~28 | ~38 | ~9% |
| SARIMA(1,1,1)(1,1,1)7 | ~22 | ~31 | ~7% |
| XGBoost (lag + calendar features) | ~20 | ~28 | ~6.5% |
| **Ensemble (SARIMA + XGBoost)** | **~18** | **~25** | **~5.8%** |

XGBoost top features: `lag_7`, `same_dow_last_week`, `roll_mean_7`, `sin_doy`, `weather_event`

---

## Capacity Planning

The staffing model converts forecasted daily volume to FTE headcount:

```
Required FTE = (Claims × Avg Handle Time) / (Hours/Day × (1 − Shrinkage) × Occupancy Target)
```

Outputs per forecast horizon:
- Required FTE vs current headcount
- Understaffed / overstaffed day flags
- Utilization % tracking
- Sensitivity table across handle times and shrinkage rates

---

## Anomaly Detection

Four methods run in parallel with consensus voting (≥2 votes = anomaly):

- **Z-Score** — global distribution outliers
- **IQR** — robust to skewed distributions
- **Rolling Z-Score** — detects regime shifts over 28-day windows
- **Isolation Forest** — multivariate, captures interactions between volume, handle time, and weather

---

## SQL Patterns Used

- Window functions (`ROWS BETWEEN ... PRECEDING AND CURRENT ROW`)
- CTEs for multi-step transformations
- JOIN across claims and staffing tables
- Aggregations with GROUP BY + HAVING
- SQLite views for reusable daily summaries

---

## Tech Stack

Python · pandas · NumPy · statsmodels · scikit-learn · XGBoost · SQLite · Matplotlib · pytest
