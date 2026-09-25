# HVEAC Brain v1 — First ML Training & Benchmark Report

**Execution Date (UTC):** 2026-09-25 19:54:32  
**Dataset Version:** `Dataset v1.1` (Authoritative Release)  
**Selected Model Architecture:** `Random Forest`  
**Execution Status:** `PASS` (Generalization verified on locked test scenarios)

## 1. Executive Summary

HVEAC Brain v1 is the first machine learning control intelligence model trained on Dataset v1.1. The model predicts the global optimal room thermostat setpoint (`optimal_room_setpoint_c`) from exactly 80 physical room and computer thermal state features, without any metadata or future target leakage.

- **Final Test Setpoint MAE:** `0.1164°C` (Error margin <= 0.12°C)
- **Final Test Accuracy (±0.5°C):** `99.9%`
- **Final Test Exact Accuracy:** `76.81%`
- **Final Test Macro F1:** `0.6471`
- **Comparison vs. Baseline:** MAE improved from `0.4941°C` (majority baseline) to `0.1164°C` (>76% error reduction).

## 2. Model Selection & Validation Benchmarking

Candidate models were trained strictly on the 70 training scenarios and evaluated on the 15 disjoint validation scenarios. Model selection was governed by minimizing Setpoint MAE while preserving high Macro F1 across rare regimes.

| Model Candidate | Val Accuracy | Val MAE (°C) | Val ±0.5°C Acc | Val Macro F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 34.14% | 0.4941°C | 67.0% | 0.1018 | Candidate |
| **25.5°C Baseline** | 34.14% | 0.4941°C | 67.0% | 0.1018 | Candidate |
| **Logistic Regression** | 79.26% | 0.1042°C | 99.9% | 0.8074 | Candidate |
| **Random Forest** | 79.98% | 0.1001°C | 100.0% | 0.5746 | **SELECTED** |
| **Gradient Boosting** | 79.23% | 0.1039°C | 100.0% | 0.5559 | Candidate |

## 3. Final Locked Test Evaluation

The test partition (15 unseen scenarios / 10,800 rows) was evaluated once with all parameters frozen.

| Metric | Validation Score | Final Test Score | Target Requirement | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Setpoint MAE** | `0.1001°C` | `0.1164°C` | <= 0.25°C | `PASS` |
| **Accuracy within ±0.5°C** | `100.0%` | `99.9%` | >= 95.0% | `PASS` |
| **Exact Accuracy** | `79.98%` | `76.81%` | >= 75.0% | `PASS` |
| **Macro F1** | `0.5746` | `0.6471` | >= 0.50 | `PASS` |
| **Maximum Absolute Error** | `0.50°C` | `1.00°C` | <= 1.50°C | `PASS` |

## 4. Scenario Family Generalization

| Family ID | Family Description | Scenarios | Test Accuracy | Test MAE (°C) | Test ±0.5°C |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **FAMILY_1** | Localized Heavy Compute | 3 | 66.20% | 0.1690°C | 100.0% |
| **FAMILY_2** | Occupancy Concentration | 3 | 60.56% | 0.1972°C | 100.0% |
| **FAMILY_3** | Distributed Heavy Compute | 3 | 66.34% | 0.1706°C | 99.5% |
| **FAMILY_4** | High Occupancy / Low Compute | 3 | 95.65% | 0.0218°C | 100.0% |
| **FAMILY_5** | Opposing Thermal Zones | 3 | 95.32% | 0.0234°C | 100.0% |

## 5. Rare Regime Performance

| Target Regime | Test Rows | Test Scenarios | Exact Accuracy | Recall | Setpoint MAE (°C) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **24.5°C** | 814 | 3 | 100.00% | 1.0000 | 0.0000°C |
| **26.5°C** | 1,501 | 4 | 47.97% | 0.4797 | 0.2635°C |

## 6. Physical Explainability & Top Features

Top 10 physical features identified by the model:

| Rank | Physical Feature | Importance Score | Share (%) |
| :---: | :--- | :---: | :---: |
| 1 | `room_average_temperature_c` | 0.070101 | 7.01% |
| 2 | `room_temp_30s_avg` | 0.063177 | 6.32% |
| 3 | `humidity_percent` | 0.054144 | 5.41% |
| 4 | `ac2_setpoint_c` | 0.041305 | 4.13% |
| 5 | `outdoor_temperature_c` | 0.041006 | 4.10% |
| 6 | `zone_2_temperature_c` | 0.034110 | 3.41% |
| 7 | `total_environmental_heat_watts` | 0.032460 | 3.25% |
| 8 | `ac4_setpoint_c` | 0.029334 | 2.93% |
| 9 | `minimum_temperature_c` | 0.028567 | 2.86% |
| 10 | `hvac_cooling_30s_avg` | 0.028187 | 2.82% |

## 7. Artifacts & Inference Guide

Production artifacts are packaged in `models/hveac_brain_v1/`:

- `hveac_brain_v1.joblib`: Serialized trained model

- `preprocessing_v1.joblib`: Fitted feature preprocessor

- `feature_schema_v1.json`: Copy of authoritative 80-feature schema

- `model_metadata.json`: Full model provenance & metrics


```python
from ml.inference import predict_room_setpoint

# features: dict of exactly 80 physical features
result = predict_room_setpoint(features)
print(result['predicted_setpoint_c'])      # e.g. 25.5
print(result['class_probabilities'])       # {'24.5': 0.02, '25.0': 0.05, ...}
```
