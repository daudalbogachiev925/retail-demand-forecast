"""Data loading. Uses M5 format, falls back to synthetic."""
import pandas as pd
import numpy as np
from src.config import config, DATA_DIR


def load_data() -> pd.DataFrame:
    """
    Load M5 dataset from local cache, else generate synthetic.

    Real dataset: https://www.kaggle.com/competitions/m5-forecasting-accuracy
    Place sales_train.csv in data/raw/
    """
    real_path = DATA_DIR / "raw" / "sales_train.csv"
    if real_path.exists():
        return pd.read_csv(real_path, parse_dates=["date"])

    print("Real data not found, generating synthetic.")
    return _generate_synthetic()


def _generate_synthetic(n_skus: int = 300) -> pd.DataFrame:
    """Generate synthetic demand with trend, seasonality, weekend effect."""
    rng = np.random.default_rng(config.random_seed)
    n_days = config.n_days_history
    dates = pd.date_range("2022-01-01", periods=n_days)
    rows = []

    for sku in range(n_skus):
        base = rng.lognormal(2, 0.8)
        trend = np.linspace(0, rng.uniform(-0.3, 0.5), n_days)
        season = 0.3 * np.sin(2 * np.pi * np.arange(n_days) / 365 + rng.random() * 6)
        dow = np.where(np.arange(n_days) % 7 >= 5, 1.3, 1.0)

        demand = base * (1 + trend + season) * dow
        demand = rng.poisson(np.clip(demand, 0, None))

        for i, d in enumerate(dates):
            rows.append({
                "sku_id": f"SKU_{sku:04d}",
                "date": d,
                "demand": int(demand[i]),
            })
    return pd.DataFrame(rows)
