"""Training script. Run: python scripts/train.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split
from src.data.loader import load_data
from src.features.build import build_features
from src.models.lgbm_model import DemandLGBM
from src.config import config, MODELS_DIR


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} rows, {df['sku_id'].nunique()} SKUs")

    print("Building features...")
    df = build_features(df)

    FEATURES = [c for c in df.columns if c not in ["sku_id", "date", "demand"]]
    X = df[FEATURES].values
    y = df["demand"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.test_fraction, random_state=config.random_seed,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=config.random_seed,
    )

    print(f"  Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_te.shape}")

    print("Training LightGBM...")
    model = DemandLGBM()
    model.fit(X_tr, y_tr, X_val, y_val)

    metrics = model.evaluate(X_te, y_te)
    print(f"  Test MAE:  {metrics['mae']:.4f}")
    print(f"  Test MAPE: {metrics['mape']:.2f}%")

    out_path = MODELS_DIR / "lgbm_demand.txt"
    model.save(out_path)
    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    main()
