"""
VaxAlert - Tool 2: Prediction Tool
Uses LightGBM to forecast 30-day vaccination coverage trend
for a selected country. Flags declining or stagnating trends.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os


def create_lag_features(df: pd.DataFrame, target_col: str, lags: int = 7) -> pd.DataFrame:
    """
    Create lag and rolling features for time-series forecasting.
    Similar approach to ROGII competition pipeline.
    """
    df = df.copy()

    # Day-of-year and week features
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month

    # Lag features
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    # Rolling mean features
    df["rolling_7d_mean"] = df[target_col].shift(1).rolling(7).mean()
    df["rolling_14d_mean"] = df[target_col].shift(1).rolling(14).mean()

    # Rolling std (volatility)
    df["rolling_7d_std"] = df[target_col].shift(1).rolling(7).std()

    # Drop rows with NaN from lag creation
    df = df.dropna()

    return df


def train_lgbm_model(df_country: pd.DataFrame, target_col: str = "people_vaccinated_per_hundred"):
    """
    Train a LightGBM model on a country's vaccination time series.
    Returns trained model and feature list.
    """
    df_feat = create_lag_features(df_country, target_col)

    feature_cols = [
        "day_of_year", "week", "month",
        "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6", "lag_7",
        "rolling_7d_mean", "rolling_14d_mean", "rolling_7d_std"
    ]
    feature_cols = [f for f in feature_cols if f in df_feat.columns]

    X = df_feat[feature_cols]
    y = df_feat[target_col]

    # Train/val split (no shuffle — time series)
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 5,
        "verbose": -1,
        "n_estimators": 200,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(period=-1)]
    )

    return model, feature_cols


def forecast_next_30_days(
    country: str,
    filepath: str = "data/vaccinations.csv",
    forecast_days: int = 30
) -> dict:
    """
    VaxAlert Prediction Tool.
    Forecasts vaccination coverage for the next 30 days for a given country.

    Args:
        country: Country name (must match dataset)
        filepath: Path to vaccination CSV
        forecast_days: Number of days to forecast ahead (default: 30)

    Returns:
        dict with keys:
            - country: str
            - current_coverage: float
            - forecasted_coverage: float (end of forecast period)
            - trend: str ("improving" | "declining" | "stagnating")
            - trend_change_pct: float
            - forecast_summary: str
    """
    try:
        df = pd.read_csv(filepath, parse_dates=["date"])
        df_country = df[df["country"] == country][["date", "people_vaccinated_per_hundred"]].copy()
        df_country = df_country.sort_values("date").dropna()

        if len(df_country) < 30:
            return {
                "error": f"Not enough data for {country} (need at least 30 rows, got {len(df_country)})",
                "country": country,
                "trend": "unknown"
            }

        # Train model
        model, feature_cols = train_lgbm_model(df_country)

        # Rolling forecast: predict one day at a time, append to history
        current_coverage = df_country["people_vaccinated_per_hundred"].iloc[-1]
        history = df_country.copy()

        forecast_values = []
        last_date = history["date"].max()

        for i in range(1, forecast_days + 1):
            next_date = last_date + pd.Timedelta(days=i)
            temp_df = history.copy()

            # Build feature row for prediction
            feat_row = create_lag_features(temp_df, "people_vaccinated_per_hundred")
            if feat_row.empty:
                break

            last_feat = feat_row[feature_cols].iloc[[-1]]
            pred = model.predict(last_feat)[0]

            # Clamp to [0, 100]
            pred = min(max(pred, 0), 100)
            forecast_values.append(pred)

            # Append prediction to history for next iteration
            new_row = pd.DataFrame({
                "date": [next_date],
                "people_vaccinated_per_hundred": [pred]
            })
            history = pd.concat([history, new_row], ignore_index=True)

        forecasted_coverage = round(forecast_values[-1], 2) if forecast_values else current_coverage
        trend_change = forecasted_coverage - current_coverage

        # Classify trend
        if trend_change > 1.0:
            trend = "improving"
        elif trend_change < -1.0:
            trend = "declining"
        else:
            trend = "stagnating"

        trend_emoji = {"improving": "📈", "declining": "📉", "stagnating": "➡️"}[trend]

        summary = (
            f"Country: {country}\n"
            f"Current coverage: {round(current_coverage, 2)}%\n"
            f"Forecasted coverage (30 days): {forecasted_coverage}%\n"
            f"Trend: {trend_emoji} {trend.upper()} ({round(trend_change, 2)}% change)\n"
        )

        if trend in ["declining", "stagnating"]:
            summary += "⚠️ Action needed: This country needs urgent campaign attention."

        return {
            "country": country,
            "current_coverage": round(current_coverage, 2),
            "forecasted_coverage": forecasted_coverage,
            "trend": trend,
            "trend_change_pct": round(trend_change, 2),
            "forecast_summary": summary
        }

    except FileNotFoundError:
        return {
            "error": "Dataset not found. Place vaccinations.csv in the data/ folder.",
            "country": country,
            "trend": "unknown"
        }
    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}",
            "country": country,
            "trend": "unknown"
        }
