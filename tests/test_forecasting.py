import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.forecaster import BaselineMA, ETSForecaster, XGBoostForecaster
from src.models.capacity import claims_to_fte, build_capacity_plan, sensitivity_analysis
from src.models.anomaly import zscore_detection, iqr_detection, rolling_zscore_detection, detect_all
from src.etl import build_model_dataset


@pytest.fixture
def sample_series():
    np.random.seed(0)
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    values = 300 + 50 * np.sin(2 * np.pi * np.arange(365) / 7) + np.random.normal(0, 15, 365)
    return pd.Series(values.clip(min=0), index=dates)


@pytest.fixture
def sample_daily_df(sample_series):
    df = pd.DataFrame({
        "date": sample_series.index,
        "total_claims": sample_series.values,
        "total_handle_hours": sample_series.values * 52 / 60,
        "weather_event": np.random.binomial(1, 0.03, len(sample_series)),
        "month": sample_series.index.month,
        "week_of_year": sample_series.index.isocalendar().week.values,
        "day_of_week": sample_series.index.day_name(),
    })
    return df


class TestBaselineMA:
    def test_predict_length(self, sample_series):
        model = BaselineMA(window=14).fit(sample_series)
        preds = model.predict(horizon=28)
        assert len(preds) == 28

    def test_predict_reasonable_range(self, sample_series):
        model = BaselineMA(window=28).fit(sample_series)
        preds = model.predict(horizon=7)
        assert preds.mean() > 0
        assert preds.mean() < 1000

    def test_all_same_value(self, sample_series):
        model = BaselineMA(window=7).fit(sample_series)
        preds = model.predict(7)
        assert np.allclose(preds, preds[0])


class TestETSForecaster:
    def test_predict_length(self, sample_series):
        model = ETSForecaster().fit(sample_series)
        preds = model.predict(28)
        assert len(preds) == 28

    def test_no_negatives(self, sample_series):
        model = ETSForecaster().fit(sample_series)
        preds = model.predict(14)
        assert all(p >= 0 for p in preds)


class TestXGBoostForecaster:
    def test_predict_shape(self, sample_daily_df):
        df = build_model_dataset(sample_daily_df)
        split = int(len(df) * 0.8)
        train, test = df.iloc[:split], df.iloc[split:]
        model = XGBoostForecaster().fit(train)
        preds = model.predict(test)
        assert len(preds) == len(test)

    def test_no_negatives(self, sample_daily_df):
        df = build_model_dataset(sample_daily_df)
        split = int(len(df) * 0.8)
        train, test = df.iloc[:split], df.iloc[split:]
        model = XGBoostForecaster().fit(train)
        preds = model.predict(test)
        assert all(p >= 0 for p in preds)

    def test_feature_importance_exists(self, sample_daily_df):
        df = build_model_dataset(sample_daily_df)
        model = XGBoostForecaster().fit(df)
        assert model.feature_importance_ is not None
        assert len(model.feature_importance_) > 0


class TestCapacityPlanning:
    def test_fte_positive(self):
        fte = claims_to_fte(350, handle_time_min=52)
        assert fte > 0

    def test_fte_scales_with_volume(self):
        fte_low = claims_to_fte(200)
        fte_high = claims_to_fte(400)
        assert fte_high > fte_low

    def test_zero_claims(self):
        assert claims_to_fte(0) == 0.0

    def test_capacity_plan_columns(self, sample_daily_df):
        forecast_df = sample_daily_df[["date"]].copy().iloc[:30]
        forecast_df["forecast"] = 350.0
        result = build_capacity_plan(forecast_df)
        assert "required_fte" in result.columns
        assert "staffing_gap" in result.columns
        assert "utilization_pct" in result.columns

    def test_sensitivity_shape(self):
        result = sensitivity_analysis(350)
        assert result.shape == (4, 3)


class TestAnomalyDetection:
    def test_zscore_flags_extremes(self):
        series = pd.Series([300.0] * 100)
        series.iloc[50] = 900
        flags = zscore_detection(series)
        assert flags.iloc[50]
        assert not flags.iloc[0]

    def test_iqr_flags_outlier(self):
        series = pd.Series([300.0] * 100)
        series.iloc[30] = 1500
        flags = iqr_detection(series)
        assert flags.iloc[30]

    def test_rolling_zscore_length(self, sample_series):
        flags = rolling_zscore_detection(sample_series)
        assert len(flags) == len(sample_series)

    def test_detect_all_columns(self, sample_daily_df):
        result = detect_all(sample_daily_df)
        assert "is_anomaly" in result.columns
        assert "anomaly_votes" in result.columns

    def test_anomaly_rate_reasonable(self, sample_daily_df):
        result = detect_all(sample_daily_df)
        rate = result["is_anomaly"].mean()
        assert 0.0 <= rate <= 0.15


class TestETL:
    def test_model_dataset_features(self, sample_daily_df):
        result = build_model_dataset(sample_daily_df)
        assert "lag_1" in result.columns
        assert "lag_7" in result.columns
        assert "roll_mean_7" in result.columns
        assert "sin_doy" in result.columns

    def test_no_nulls_after_dropna(self, sample_daily_df):
        result = build_model_dataset(sample_daily_df)
        assert result.isnull().sum().sum() == 0
