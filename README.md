# Predictive Maintenance for Industrial Equipment

Binary classification ML pipeline to predict equipment failure within 24 hours from operational sensor data — with rolling feature engineering, class imbalance handling, and threshold optimisation.

## Background

Unplanned equipment failure is one of the costliest operational risks in energy and manufacturing. This project builds a full predictive maintenance (PdM) pipeline, inspired directly by sensor data analysis work conducted on refinery control systems at Creas Consulting (step tests, 900k+ data points, process optimisation).

Built by [Janusshan Sivanesakanthan](https://github.com/janusshan24) — a data scientist with domain experience in energy systems (Shell, Creas) and automotive (Ford Motors).

## What's in this project

- **Synthetic industrial sensor data** — 18 months of hourly readings (temperature, pressure, vibration, RPM, oil viscosity) with realistic pre-failure degradation signatures
- **Feature engineering** — rolling means/std (4h, 24h), rate-of-change, physics-informed cross-sensor ratios
- **Class imbalance handling** — balanced class weights
- **Three models compared** — Logistic Regression, Random Forest, Gradient Boosting
- **Full evaluation** — confusion matrix, ROC-AUC, Precision-Recall curves, feature importance
- **Threshold optimisation** — tuned for recall vs precision trade-off in a maintenance context

## Results Summary

| Model | ROC-AUC | Recall (failure) | F1 (failure) |
|-------|---------|-----------------|--------------|
| Logistic Regression | ~0.91 | ~0.72 | ~0.58 |
| **Random Forest** | **~0.97** | **~0.82** | **~0.72** |
| Gradient Boosting | ~0.96 | ~0.79 | ~0.69 |

At the optimal threshold, Random Forest catches ~82% of failures — 4 in 5 — while keeping false alarms at manageable levels.

## Setup

```bash
git clone https://github.com/janusshan24/predictive-maintenance-industrial
cd predictive-maintenance-industrial
pip install -r requirements.txt
jupyter notebook predictive_maintenance.ipynb
```

## Requirements

See `requirements.txt`. Python 3.8+ recommended.

## Data

Synthetic sensor data is generated in-notebook with realistic degradation signatures. To extend with real data, the [NASA CMAPSS Turbofan Engine Dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) is a well-established benchmark for remaining useful life (RUL) prediction.

## Next steps

- Apply to NASA CMAPSS dataset for benchmarkable RUL prediction
- Add SHAP values for model explainability — essential for operator trust
- Build real-time scoring pipeline with configurable alert thresholds per asset type
- Extend to multi-class failure type classification (bearing, seal, lubrication faults)
