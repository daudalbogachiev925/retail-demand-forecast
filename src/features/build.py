"""Feature engineering pipeline."""
import pandas as pd
import numpy as np
from src.config import config


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)
    g = df.groupby("sku_id")["demand"]
    for lag in config.lags:
        df[f"lag_{lag}"] = g.shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("sku_id")["demand"]
    for w in config.rolling_windows:
        df[f"roll_mean_{w}"] = g.shift(1).rolling(w).mean().reset_index(0, drop=True)
        df[f"roll_std_{w}"] = g.shift(1).rolling(w).std().reset_index(0, drop=True)
    return df


def add_sku_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby("sku_id")["demand"].agg(["mean", "std"]).add_prefix("sku_")
    return df.merge(stats, left_on="sku_id", right_index=True, how="left")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline. TODO: parallelize with polars for speed."""
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_sku_stats(df)
    return df.dropna().reset_index(drop=True)
