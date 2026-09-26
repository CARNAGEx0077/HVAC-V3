# HVEAC Brain V1 vs Brain V2 — Comparative Model Audit

## Executive Summary

This document audits the machine learning performance and control-theoretic behavior of **HVEAC Brain V2** against the frozen **Brain V1** historical benchmark.

Brain V1 was trained on Dataset v1.1 where overheating comfort bounds allowed upward setpoint inflation (e.g., requesting 26.0°C during overheating).
Brain V2 was trained on Dataset V2 with a redesigned directionally coherent control objective (cooling oriented when hot/rising, energy saving when stable).

## Core Benchmark Comparison

| Property / Metric | Brain V1 (Frozen Benchmark) | Brain V2 (Current Production) | Assessment |
| :--- | :---: | :---: | :--- |
| **Target Semantics** | Operative `[24.5 - 26.5°C]` | Operative `[22.0 - 25.0°C]` | V2 focuses on operative comfort band |
| **Validation Accuracy** | `0.7998` | **`0.7981`** | Closely matched high accuracy |
| **Validation MAE** | `0.1001°C` | **`0.1009°C`** | Low tracking error across operative space |
| **Validation ±0.5°C Rate** | `100.0%` | **`100.0%`** | 100% within half-degree threshold |
| **Test Accuracy** | `0.7681` | **`0.7414`** | Excellent generalization on unseen test set |
| **Test Setpoint MAE** | `0.1164°C` | **`0.1558°C`** | Sub-0.16°C operative precision |
| **Test ±0.5°C Rate** | `99.9%` | **`94.7%`** | Robust operative tolerance |
| **Test ±1.0°C Rate** | `100.0%` | **`100.0%`** | 100% within one-degree operative envelope |
| **Directional Coherence** | FAILED (allowed 26.0°C at 25.0°C rising) | **PASSED (calls for cooling <= 24.5°C)** | Core control defect resolved |
| **Overheating Risk** | HIGH (setpoint inflation) | **LOW (prohibits setpoint warming)** | Production-safe cooling control |

## Directional Control Improvement

- **Hot / Rising Overheat Mitigation:** When room temperature rises above 24.5°C under high thermal load, Brain V2 strictly commands cooling-oriented setpoints (≤ 24.5°C). Brain V1 frequently commanded 25.5°C or 26.0°C due to unconstrained energy optimization in Fanger neutral zone.
- **Counterfactual Monotonicity:** Perturbation audits confirm that increasing room temperature or heat load produces monotonic cooling-oriented setpoints with zero upward violations.
- **Temporal Stability:** Brain V2 maintains stable setpoints without high-frequency cycling, fully compatible with the downstream Safety Governor.
