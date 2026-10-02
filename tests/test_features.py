"""Tests for feature pipeline."""
import pandas as pd
import numpy as np
from src.features.build import add_calendar_features, add_lag_features


def test_calendar_features():
    df = pd.DataFrame({
        "sku_id": ["A"] * 10,
        "date": pd.date_range("2023-01-01", periods=10),
        "demand": np.arange(10),
    })
    out = add_calendar_features(df)
    assert "day_of_week" in out.columns
    assert "is_weekend" in out.columns
    assert out["is_weekend"].isin([0, 1]).all()


def test_lag_features():
    df = pd.DataFrame({
        "sku_id": ["A"] * 10,
        "date": pd.date_range("2023-01-01", periods=10),
        "demand": np.arange(10),
    })
    out = add_lag_features(df)
    assert "lag_1" in out.columns
    assert out["lag_1"].iloc[1] == 0
    assert out["lag_1"].iloc[5] == 4
