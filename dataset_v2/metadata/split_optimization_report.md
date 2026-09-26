# HVEAC Dataset V2 — Behavior-Balanced Split Optimization Report

## Executive Summary

This audit documents the behavior-aware scenario resplit of **HVEAC Dataset V2**.
The raw simulation data (`dataset_v2/raw/all_scenarios.csv`) remains bit-for-bit unchanged and frozen.
The scenario allocation across splits was optimized using multi-objective simulated annealing over scenario-level behavioral signatures.

- **Overall Split Score:** `0.3242` → `0.0783` (**75.8% reduction in distribution shift**)
- **Train vs Validation Target JS:** `0.5031` → `0.0810`
- **Train vs Test Target JS:** `0.3966` → `0.0839`
- **Validation vs Test Target JS:** `0.4137` → `0.0496`

## Objective Function & Canonical Dimension Weights

The composite objective minimizes cross-split behavioral divergence across 8 engineering dimensions:
```text
TOTAL_SPLIT_SCORE =
  0.30 * target_distribution_distance
  0.20 * regime_distribution_distance
  0.15 * reason_distribution_distance
  0.15 * control_distribution_distance
  0.08 * thermal_distribution_distance
  0.04 * compute_distribution_distance
  0.04 * occupancy_distribution_distance
  0.04 * environment_distribution_distance
```

## Detailed Dimension Comparison: Current Split vs New Split

| Dimension | Weight | Old Train/Val JS | New Train/Val JS | Old Train/Test JS | New Train/Test JS | Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Target** | `0.30` | `0.5031` | **`0.0810`** | `0.3966` | **`0.0839`** | `+81.7%` |
| **Regime** | `0.20` | `0.1822` | **`0.0339`** | `0.0880` | **`0.0223`** | `+79.2%` |
| **Reason** | `0.15` | `0.3744` | **`0.1654`** | `0.3602` | **`0.1243`** | `+60.6%` |
| **Control** | `0.15` | `0.3801` | **`0.0450`** | `0.2783` | **`0.0309`** | `+88.5%` |
| **Thermal** | `0.08` | `0.3969` | **`0.0966`** | `0.3208` | **`0.1539`** | `+65.1%` |
| **Compute** | `0.04` | `0.0905` | **`0.0964`** | `0.1918` | **`0.0324`** | `+54.4%` |
| **Occupancy** | `0.04` | `0.1474` | **`0.1303`** | `0.5257` | **`0.1349`** | `+60.6%` |
| **Environment** | `0.04` | `0.2710` | **`0.0205`** | `0.2182` | **`0.0306`** | `+89.6%` |

## Optimization Diagnostics

- **Optimizer Algorithm:** Constrained Simulated Annealing with $O(1)$ Incremental Histogram Updates
- **Optimizer Seed:** `42`
- **Candidate Allocations Evaluated:** `20,000`
- **Optimization Wall Time:** `10.21 seconds`
- **Family Balance Constraint:** Strictly enforced 14 / 3 / 3 per family across all 5 families
- **Scenario Overlap:** Strictly 0 (empty intersection between Train, Validation, and Test)
