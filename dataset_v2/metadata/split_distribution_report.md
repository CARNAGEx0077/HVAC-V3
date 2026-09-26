# HVEAC Dataset V2 — Comprehensive Split Distribution Report

## Overview

| Property | TRAIN | VALIDATION | TEST | TOTAL |
| :--- | :---: | :---: | :---: | :---: |
| **Scenarios** | `70` | `15` | `15` | `100` |
| **Rows** | `50,400` | `10,800` | `10,800` | `72,000` |
| **Per-Family Scenarios** | `14 / 14 / 14 / 14 / 14` | `3 / 3 / 3 / 3 / 3` | `3 / 3 / 3 / 3 / 3` | `20 / 20 / 20 / 20 / 20` |

## Target Setpoint Distribution (`optimal_room_setpoint_c`)

| Setpoint | Train Proportion | Validation Proportion | Test Proportion |
| :---: | :---: | :---: | :---: |
| **22.0°C** | `0.0043` | `0.0000` | `0.0000` |
| **22.5°C** | `0.0070` | `0.0006` | `0.0000` |
| **23.0°C** | `0.0447` | `0.0697` | `0.0544` |
| **23.5°C** | `0.1742` | `0.1751` | `0.1468` |
| **24.0°C** | `0.3833` | `0.3857` | `0.4008` |
| **24.5°C** | `0.3275` | `0.3094` | `0.3376` |
| **25.0°C** | `0.0590` | `0.0594` | `0.0605` |

## Control & Actuator Metrics

| Metric | TRAIN | VALIDATION | TEST |
| :--- | :---: | :---: | :---: |
| `mean_ac1` | `0.3368` | `0.2947` | `0.3617` |
| `mean_ac2` | `0.2694` | `0.3423` | `0.2913` |
| `mean_ac3` | `0.2911` | `0.3568` | `0.2378` |
| `mean_ac4` | `0.3767` | `0.3037` | `0.3071` |
| `max_cooling` | `1.0` | `1.0` | `1.0` |
| `aggressive_cooling_prop` | `0.2832` | `0.2643` | `0.2991` |

## Directional Control Regime Distribution (Regimes A through I)

| Regime | Train Proportion | Validation Proportion | Test Proportion |
| :--- | :---: | :---: | :---: |
| **REGIME_A_HOT_RISING** | `0.0018` | `0.0008` | `0.0010` |
| **REGIME_B_HOT_STABLE** | `0.2256` | `0.2463` | `0.2430` |
| **REGIME_C_COMFORTABLE_STABLE** | `0.6753` | `0.6673` | `0.6604` |
| **REGIME_D_COOLING_FALLING** | `0.0959` | `0.0849` | `0.0953` |
| **REGIME_E_LOW_TEMP_LOW_COOLING** | `0.0000` | `0.0000` | `0.0000` |
| **REGIME_F_HIGH_COMPUTE** | `0.2167` | `0.2000` | `0.2181` |
| **REGIME_G_HIGH_OCCUPANCY** | `0.2429` | `0.2000` | `0.2667` |
| **REGIME_H_LOCALIZED_HOTSPOT** | `0.7366` | `0.7724` | `0.8046` |
| **REGIME_I_OPPOSING_ZONES** | `0.2000` | `0.2000` | `0.2000` |

## Label Reason Distribution

| Label Reason | Train Proportion | Validation Proportion | Test Proportion |
| :--- | :---: | :---: | :---: |
| **STATE_A_OVERHEATING_COOLING** | `0.2673` | `0.2812` | `0.2880` |
| **STATE_B_ENERGY_OPTIMIZED** | `0.2435` | `0.2727` | `0.2671` |
| **HIGH_OCCUPANCY** | `0.2631` | `0.2372` | `0.1943` |
| **HIGH_COMPUTE_LOAD** | `0.1855` | `0.1516` | `0.1893` |
| **STATE_B_COMFORT_ENERGY_BALANCE** | `0.0268` | `0.0000` | `0.0605` |
| **LOCALIZED_ZONE_COOLING** | `0.0111` | `0.0552` | `0.0000` |
| **STATE_D_PREVENTIVE_COOLING** | `0.0015` | `0.0012` | `0.0009` |
| **STATE_E_PREVENTIVE_WARMING** | `0.0012` | `0.0009` | `0.0000` |