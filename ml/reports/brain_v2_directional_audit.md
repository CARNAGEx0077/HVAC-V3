# HVEAC Brain V2 — Directional Control & Counterfactual Audit Report

## Executive Summary

**Overall Directional Status:** `PASS`  
**Counterfactual Perturbation Status:** `PASS`

## Directional Case Evaluations (Section 14)

| Case | Condition / Regimes | Model Output | Expected Contract | Result |
| :--- | :--- | :---: | :---: | :---: |
| **CASE_A_HOT_RISING_HIGH_LOAD** | 25.5°C rising with 3500W load -> cooling target <= 25.0°C | `24.0°C` | Contract bounds | **`PASS`** |
| **CASE_B_HOT_RISING_MODERATE_LOAD** | 25.0°C rising with 1800W load -> cooling target <= 24.5°C | `24.0°C` | Contract bounds | **`PASS`** |
| **CASE_C_STABLE_ACCEPTABLE** | 23.5°C stable with low load -> maintain/relax target >= 23.5°C | `24.5°C` | Contract bounds | **`PASS`** |
| **CASE_D_COOL_ROOM** | 22.0°C cool room -> reduce cooling target >= 23.0°C | `24.5°C` | Contract bounds | **`PASS`** |
| **CASE_E_LOAD_MONOTONICITY** | Increasing thermal load (800W-4000W) must not increase setpoint | `[24.5, 24.5, 24.5, 24.5, 24.5]` | Monotonic (delta <= 0) | **`PASS`** |

## Counterfactual Physical Perturbation Experiments (Section 15)

| Experiment | Physical Perturbation | Mean Setpoint Delta | Expected Direction | Violations | Result |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **EXP_A_INCREASE_TEMP** | +2.0°C Room & Zone Temperature | `-0.5267°C` | COOLING_ORIENTATION (delta <= 0) | `0` | **`PASS`** |
| **EXP_B_INCREASE_TOTAL_LOAD** | +2000W Total Heat Load | `-0.0033°C` | COOLING_ORIENTATION (delta <= 0) | `0` | **`PASS`** |
| **EXP_C_INCREASE_COMPUTER_HEAT** | +1500W Total Computer Heat | `+0.0000°C` | COOLING_ORIENTATION (delta <= 0) | `0` | **`PASS`** |
| **EXP_D_INCREASE_OCCUPANCY_HEAT** | +500W Occupancy Heat (+5 occupants) | `+0.0000°C` | COOLING_ORIENTATION (delta <= 0) | `0` | **`PASS`** |
| **EXP_E_DECREASE_TEMP** | -2.0°C Room & Zone Temperature | `+0.2233°C` | ENERGY_SAVING / RELAX_COOLING (delta >= 0) | `0` | **`PASS`** |