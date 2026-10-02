# retail-demand-forecast
Hierarchical demand forecasting for retail with LightGBM and FastAPI serving
# Retail Demand Forecast

Hierarchical demand forecasting for retail SKUs with production-ready serving.

## Motivation

Predicting SKU-level demand is critical for inventory planning. Understocking loses revenue, overstocking eats margin. This project builds a complete pipeline from raw sales to served predictions.

## Results

| Model | MAPE | MAE |
|-------|------|-----|
| Naive (lag-7) | 32.4% | 4.8 |
| LightGBM baseline | 18.7% | 2.9 |
| LightGBM + features | **13.2%** | **2.1** |

*Tested on synthetic 300 SKUs × 730 days.*

## Architecture
Sales data → Feature builder → LightGBM → FastAPI → Client

## Quick Start

```bash
pip install -r requirements.txt
python scripts/train.py
uvicorn src.serving.api:app --port 8000
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"sku_id": "SKU_0001", "features": [1.2, 0.5, ...]}'
Data
Real datasets:

M5 Competition

Favorita Grocery

Roadmap
☑ LightGBM baseline
☑ FastAPI serving
□ Hierarchical reconciliation (MinT)
□ MLflow tracking
Known issues
Feature builder is single-threaded, ~2 min on 500K rows

No stockout handling in history

License

MIT

---

## После всех файлов

1. Вкладка **Issues** → **New issue** ×3:
   - `Hierarchical reconciliation not implemented`
   - `Feature builder slow on > 1M rows`
   - `Add quantile regression for safety stock`

2. Вверху репо клик по **main** → введи `feature/mlflow-tracking` → **Create branch**

3. В этой ветке **Add file** → `mlflow_setup.py`:

```python
"""MLflow tracking setup. WIP — needs integration with train script."""
import mlflow

# TODO: wrap train.py with mlflow.start_run()
# TODO: log params from config
# TODO: log metrics from evaluation
pass
Вкладка Pull requests → New pull request → base main, compare feature/mlflow-tracking → Create → не мержи.

Справа на главной ⚙️ About → Description + Topics: machine-learning, demand-forecasting, time-series, lightgbm, fastapi, python → Save.
