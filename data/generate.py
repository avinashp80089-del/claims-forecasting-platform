import pandas as pd
import numpy as np
from datetime import date, timedelta
import sqlite3
import os

np.random.seed(42)

CLAIM_TYPES = ["auto_collision", "auto_liability", "property_damage", "weather_event"]
HANDLE_TIMES = {"auto_collision": 45, "auto_liability": 60, "property_damage": 55, "weather_event": 75}
REGIONS = ["midwest", "south", "northeast", "west"]


def _seasonal(dt):
    doy = dt.timetuple().tm_yday
    return 1.0 + 0.25 * np.sin(2 * np.pi * (doy - 80) / 365)


def _dow_factor(dt):
    factors = {0: 1.15, 1: 1.05, 2: 1.0, 3: 1.0, 4: 1.1, 5: 0.7, 6: 0.5}
    return factors[dt.weekday()]


def _holiday_factor(dt):
    holidays = {
        date(dt.year, 1, 1), date(dt.year, 7, 4),
        date(dt.year, 11, 25), date(dt.year, 12, 25),
    }
    return 0.4 if dt.date() in holidays else 1.0


def _weather_spike(dt, spikes):
    return 2.5 if dt.date() in spikes else 1.0


def _generate_spikes(start, end):
    dates = pd.date_range(start, end, freq="D")
    n = int(len(dates) * 0.04)
    return set(pd.to_datetime(np.random.choice(dates, n, replace=False)).date)


def generate_claims(start="2022-01-01", end="2024-12-31"):
    dates = pd.date_range(start, end, freq="D")
    spikes = _generate_spikes(start, end)
    rows = []

    for dt in dates:
        base = 320
        mu = base * _seasonal(dt) * _dow_factor(dt) * _holiday_factor(dt) * _weather_spike(dt, spikes)
        total = int(np.random.poisson(mu))

        proportions = np.random.dirichlet([3.5, 2.5, 2.0, 1.0])
        for ctype, prop in zip(CLAIM_TYPES, proportions):
            count = max(0, int(total * prop + np.random.normal(0, 3)))
            handle_time = HANDLE_TIMES[ctype] * (1 + np.random.normal(0, 0.1))
            is_spike = dt.date() in spikes

            rows.append({
                "date": dt.date(),
                "claim_type": ctype,
                "region": np.random.choice(REGIONS),
                "claim_count": count,
                "avg_handle_time_min": round(handle_time, 1),
                "weather_event": int(is_spike),
                "day_of_week": dt.day_name(),
                "month": dt.month,
                "week_of_year": dt.isocalendar().week,
            })

    return pd.DataFrame(rows)


def generate_staffing(start="2022-01-01", end="2024-12-31"):
    dates = pd.date_range(start, end, freq="D")
    rows = []
    for dt in dates:
        is_weekend = dt.weekday() >= 5
        base_staff = 85 if not is_weekend else 40
        rows.append({
            "date": dt.date(),
            "staff_available": base_staff + np.random.randint(-5, 6),
            "staff_scheduled": base_staff + np.random.randint(0, 10),
            "shrinkage_pct": round(np.random.uniform(0.10, 0.20), 3),
        })
    return pd.DataFrame(rows)


def save_to_sqlite(claims_df, staffing_df, db_path="data/claims.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    claims_df.to_sql("claims", conn, if_exists="replace", index=False)
    staffing_df.to_sql("staffing", conn, if_exists="replace", index=False)

    conn.execute("""
        CREATE VIEW IF NOT EXISTS daily_claims AS
        SELECT
            date,
            SUM(claim_count)                                      AS total_claims,
            SUM(claim_count * avg_handle_time_min) / 3600.0      AS total_handle_hours,
            MAX(weather_event)                                    AS weather_event,
            day_of_week,
            month,
            week_of_year
        FROM claims
        GROUP BY date, day_of_week, month, week_of_year
    """)
    conn.commit()
    conn.close()
    print(f"Saved {len(claims_df)} claim rows and {len(staffing_df)} staffing rows to {db_path}")


if __name__ == "__main__":
    claims = generate_claims()
    staffing = generate_staffing()
    save_to_sqlite(claims, staffing)
    claims.to_csv("data/claims_raw.csv", index=False)
    staffing.to_csv("data/staffing_raw.csv", index=False)
    print("Data generation complete.")
