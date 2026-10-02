"""LightGBM demand model."""
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error
from src.config import config


class DemandLGBM:
    """LightGBM wrapper for demand forecasting."""

    def __init__(self, params: dict | None = None):
        self.params = params or config.lgbm_params
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        train_set = lgb.Dataset(X_train, y_train)
        valid_sets = [train_set]
        callbacks = []

        if X_val is not None:
            valid_sets.append(lgb.Dataset(X_val, y_val, reference=train_set))
            callbacks.append(lgb.early_stopping(50))
            callbacks.append(lgb.log_evaluation(100))

        self.model = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.params.get("n_estimators", 500),
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return self.model.predict(X)

    def evaluate(self, X, y) -> dict:
        preds = self.predict(X)
        return {
            "mae": float(mean_absolute_error(y, preds)),
            "mape": float(np.mean(np.abs(preds - y) / (y + 1)) * 100),
        }

    def save(self, path):
        self.model.save_model(str(path))

    @classmethod
    def load(cls, path):
        instance = cls()
        instance.model = lgb.Booster(model_file=str(path))
        return instance
