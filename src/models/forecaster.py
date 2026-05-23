import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

HORIZON = 28


def _split(df: pd.DataFrame, test_days=90):
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    return df[df["date"] <= cutoff], df[df["date"] > cutoff]


def _metrics(actual, predicted, name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual.clip(lower=1))) * 100
    return {"model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE%": round(mape, 2)}


class BaselineMA:
    def __init__(self, window=28):
        self.window = window
        self.last_values = None

    def fit(self, series: pd.Series):
        self.last_values = series.values[-self.window:]
        return self

    def predict(self, horizon=HORIZON):
        base = self.last_values.mean()
        return np.full(horizon, base)


class ETSForecaster:
    def __init__(self):
        self.model = None

    def fit(self, series: pd.Series):
        self.model = ExponentialSmoothing(
            series.values,
            trend="add",
            seasonal="add",
            seasonal_periods=7,
            damped_trend=True,
        ).fit(optimized=True)
        return self

    def predict(self, horizon=HORIZON):
        return self.model.forecast(horizon)


class SARIMAForecaster:
    def __init__(self):
        self.model = None
        self.result = None

    def fit(self, series: pd.Series):
        self.model = SARIMAX(
            series.values,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.result = self.model.fit(disp=False)
        return self

    def predict(self, horizon=HORIZON):
        forecast = self.result.get_forecast(steps=horizon)
        return forecast.predicted_mean.clip(min=0)

    def confidence_intervals(self, horizon=HORIZON, alpha=0.05):
        forecast = self.result.get_forecast(steps=horizon)
        ci = forecast.conf_int(alpha=alpha)
        return ci[:, 0].clip(min=0), ci[:, 1]


class XGBoostForecaster:
    FEATURES = [
        "lag_1", "lag_7", "lag_14", "lag_28",
        "roll_mean_7", "roll_mean_14", "roll_mean_28",
        "roll_std_7", "roll_std_28",
        "dow", "is_weekend", "month", "quarter",
        "sin_doy", "cos_doy",
        "same_dow_last_week", "same_dow_last_year",
        "weather_event",
    ]

    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        self.scaler = StandardScaler()
        self.feature_importance_ = None

    def fit(self, df: pd.DataFrame):
        cols = [c for c in self.FEATURES if c in df.columns]
        X = self.scaler.fit_transform(df[cols])
        y = df["total_claims"].values
        self.model.fit(X, y)
        self.feature_cols = cols
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_, index=cols
        ).sort_values(ascending=False)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(df[self.feature_cols])
        return self.model.predict(X).clip(min=0)


def evaluate_all(model_df: pd.DataFrame):
    daily = model_df[["date", "total_claims", "weather_event"]].copy()
    train_df, test_df = _split(model_df)
    train_series = train_df.set_index("date")["total_claims"]
    test_series = test_df["total_claims"].values

    results = []

    ma = BaselineMA().fit(train_series)
    pred_ma = ma.predict(len(test_series))
    results.append(_metrics(test_series, pred_ma, "Baseline_MA28"))

    ets = ETSForecaster().fit(train_series)
    pred_ets = ets.predict(len(test_series))
    results.append(_metrics(test_series, pred_ets, "ETS"))

    sarima = SARIMAForecaster().fit(train_series)
    pred_sarima = sarima.predict(len(test_series))
    results.append(_metrics(test_series, pred_sarima, "SARIMA"))

    xgb = XGBoostForecaster().fit(train_df)
    pred_xgb = xgb.predict(test_df)
    results.append(_metrics(test_series, pred_xgb, "XGBoost"))

    ensemble = (pred_sarima + pred_xgb) / 2
    results.append(_metrics(test_series, ensemble, "Ensemble_SARIMA_XGB"))

    leaderboard = pd.DataFrame(results).sort_values("MAPE%")
    return leaderboard, {
        "baseline": (ma, pred_ma),
        "ets": (ets, pred_ets),
        "sarima": (sarima, pred_sarima),
        "xgboost": (xgb, pred_xgb),
        "ensemble": (None, ensemble),
    }, test_df, xgb


if __name__ == "__main__":
    from src.etl import load, build_model_dataset
    datasets = load()
    model_df = build_model_dataset(datasets["daily_totals"])
    leaderboard, preds, test_df, xgb = evaluate_all(model_df)
    print("\nModel Leaderboard:")
    print(leaderboard.to_string(index=False))
    print("\nTop 5 XGBoost Features:")
    print(xgb.feature_importance_.head())
