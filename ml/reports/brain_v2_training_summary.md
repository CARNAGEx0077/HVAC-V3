# HVEAC Brain V2 — Final Machine Learning Training Summary

## Overview

- **Model Version:** `hveac_brain_v2`
- **Architecture:** Random Forest Classifier (150 trees, max_depth=16, min_samples_leaf=2)
- **Dataset:** `dataset_v2` (70 Train, 15 Validation, 15 Test scenarios)
- **Target:** `optimal_room_setpoint_c` (Operative classes: 22.0°C to 25.0°C)
- **Physical Features:** 80 input features (134 encoded)
- **AI HVAC Control:** DISABLED (Model is offline / shadow benchmark only)

## Locked Test Performance

- **Accuracy:** `0.7414`
- **Balanced Accuracy:** `0.4162`
- **Macro F1:** `0.3824`
- **Setpoint MAE:** `0.1558°C`
- **±0.5°C Accuracy:** `94.69%`
- **±1.0°C Accuracy:** `100.00%`

## Model Comparison (Validation Split)

| Model | Accuracy | Balanced Acc | Macro F1 | MAE (°C) | ±0.5°C (%) | Selection |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Baseline (Majority Target 24.0°C) | `0.3857` | `0.1667` | `0.0928` | `0.3723` | `87.0%` |  |
| Baseline (Constant Target 24.0°C) | `0.3857` | `0.1667` | `0.0928` | `0.3723` | `87.0%` |  |
| Logistic Regression | `0.6700` | `0.4479` | `0.4197` | `0.3424` | `82.0%` |  |
| Random Forest | `0.7981` | `0.5442` | `0.5380` | `0.1009` | `100.0%` | **SELECTED** |
| HistGradientBoosting | `0.7545` | `0.5425` | `0.5451` | `0.1274` | `99.5%` |  |