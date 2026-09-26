# HVEAC Dataset V2 — Rare Regime Coverage & Allocation Report

## Overview

This report audits the cross-split representation of rare setpoint targets and operational stress regimes.
Conforms strictly to Section 10 and Section 18 of the HVEAC V2 Specification.
Hard constraints forbid duplicating scenarios, splitting rows within a scenario, or fabricating synthetic data.

## Target Setpoint Regime Breakdown

| Target Setpoint | Total Scens | Train Scens | Val Scens | Test Scens | Total Rows | Train Rows | Val Rows | Test Rows |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **22.0°C** | `1` | `1` | `0` | `0` | `217` | `217` | `0` | `0` |
| **22.5°C** | `2` | `1` | `1` | `0` | `359` | `352` | `7` | `0` |
| **23.0°C** | `14` | `10` | `2` | `2` | `3,591` | `2,251` | `753` | `587` |
| **23.5°C** | `47` | `36` | `4` | `7` | `12,258` | `8,782` | `1,891` | `1,585` |
| **24.0°C** | `82` | `57` | `12` | `13` | `27,815` | `19,320` | `4,166` | `4,329` |
| **24.5°C** | `51` | `37` | `6` | `8` | `23,494` | `16,506` | `3,342` | `3,646` |
| **25.0°C** | `7` | `5` | `1` | `1` | `4,266` | `2,972` | `641` | `653` |

### Target Regime Allocation Analysis

- **Rare 22.0°C Regime (1 scenario):** Appears exclusively in `SCEN_0048_FAMILY_3` (217 timesteps). Because this is an indivisible physical scenario, it is legitimately assigned to **TRAIN** (70% capacity) with zero duplication or fabrication, fully complying with Section 10.
- **Rare 22.5°C Regime (2 scenarios):** Present in `SCEN_0013_FAMILY_3` (7 rows) and `SCEN_0048_FAMILY_3` (352 rows). Successfully distributed across **TRAIN** (1 scenario) and **VALIDATION** (1 scenario).
- **Operative Regimes (23.0°C to 25.0°C):** Ubiquitously represented across **ALL THREE SPLITS** (Train, Validation, and Test).

## Behavioral Stress Regimes Representation

| Behavioral Regime | Definition / Physical Condition | Total Scens | Train Scens | Val Scens | Test Scens | Coverage Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **HOT + RISING** | Physical stress condition | `26` | `16` | `5` | `5` | **`COMPLETE (Train/Val/Test)`** |
| **HIGH COMPUTE** | Physical stress condition | `26` | `19` | `3` | `4` | **`COMPLETE (Train/Val/Test)`** |
| **HIGH OCCUPANCY** | Physical stress condition | `24` | `17` | `3` | `4` | **`COMPLETE (Train/Val/Test)`** |
| **LOCALIZED HOTSPOT** | Physical stress condition | `82` | `56` | `13` | `13` | **`COMPLETE (Train/Val/Test)`** |
| **OPPOSING ZONES** | Physical stress condition | `20` | `14` | `3` | `3` | **`COMPLETE (Train/Val/Test)`** |
| **LOW TEMPERATURE** | Physical stress condition | `1` | `1` | `0` | `0` | **`PARTIAL`** |
| **HIGH COOLING** | Physical stress condition | `67` | `50` | `8` | `9` | **`COMPLETE (Train/Val/Test)`** |

## Audit Conclusion

- **Physical Disjointness:** Strictly enforced (0 scenario overlap between splits).
- **Rare Regime Coverage:** Maximum possible physical coverage achieved without artificial label patching.
- **Status:** **PASS**
