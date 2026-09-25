# HVEAC Dataset Label Audit

## Dataset
- **Total Scenarios:** 100
- **Total Simulation Rows:** 72,000
- **Master Seed:** 42
- **Audit Status:** **WARNING**

## Split
| Split | Scenarios | Rows | Percentage |
| :--- | :--- | :--- | :--- |
| **Train** | 70 | 50,400 | 70.0% |
| **Validation** | 15 | 10,800 | 15.0% |
| **Test** | 15 | 10,800 | 15.0% |

## Family Distribution
| Scenario Family | Scenarios | Rows | Share |
| :--- | :--- | :--- | :--- |
| `FAMILY_1` | 20 | 14,400 | 20.0% |
| `FAMILY_2` | 20 | 14,400 | 20.0% |
| `FAMILY_3` | 20 | 14,400 | 20.0% |
| `FAMILY_4` | 20 | 14,400 | 20.0% |
| `FAMILY_5` | 20 | 14,400 | 20.0% |

## Overall Target Distribution
| Target Variable | Min | Max | Mean | Std | Dominant Value | Dominant Share | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `optimal_ac1_cooling_level` | 0.0 | 1.0 | 0.24 | 0.32 | 0.05 | 59.1% | `PASS` |
| `optimal_ac1_setpoint_c` | 24.0 | 27.0 | 25.77 | 0.81 | 25.5 | 31.9% | `PASS` |
| `optimal_ac2_cooling_level` | 0.0 | 1.0 | 0.20 | 0.28 | 0.05 | 58.9% | `PASS` |
| `optimal_ac2_setpoint_c` | 24.0 | 27.0 | 25.83 | 0.80 | 25.5 | 31.2% | `PASS` |
| `optimal_ac3_cooling_level` | 0.0 | 1.0 | 0.20 | 0.28 | 0.05 | 59.9% | `PASS` |
| `optimal_ac3_setpoint_c` | 24.0 | 27.0 | 25.84 | 0.80 | 25.5 | 30.4% | `PASS` |
| `optimal_ac4_cooling_level` | 0.05 | 1.0 | 0.26 | 0.33 | 0.05 | 59.2% | `PASS` |
| `optimal_ac4_setpoint_c` | 24.0 | 27.0 | 25.78 | 0.83 | 25.5 | 30.0% | `PASS` |
| `optimal_room_setpoint_c` | 24.5 | 26.5 | 25.63 | 0.52 | 25.5 | 42.8% | `PASS` |

### Optimal Room Setpoint Breakdown
| Candidate Setpoint | Timestep Count | Percentage |
| :--- | :--- | :--- |
| 24.5°C | 6,528 | 9.07% |
| 25.0°C | 4,695 | 6.52% |
| 25.5°C | 30,832 | 42.82% |
| 26.0°C | 22,864 | 31.76% |
| 26.5°C | 7,081 | 9.83% |

## Per-Family Target Distribution
### FAMILY_1
- **Mean Optimal Setpoint:** 25.65°C (Std: 0.49°C)
- **Dominant Setpoint:** 25.5°C (44.2%)

| Setpoint | Count | Share |
| :--- | :--- | :--- |
| 24.5°C | 776 | 5.4% |
| 25.0°C | 1,445 | 10.0% |
| 25.5°C | 6,367 | 44.2% |
| 26.0°C | 4,303 | 29.9% |
| 26.5°C | 1,509 | 10.5% |

*Primary AC Cooling Level Means:*
- `AC-1`: 0.225 | `AC-2`: 0.176 | `AC-3`: 0.187 | `AC-4`: 0.245

### FAMILY_2
- **Mean Optimal Setpoint:** 25.80°C (Std: 0.51°C)
- **Dominant Setpoint:** 26.0°C (39.4%)

| Setpoint | Count | Share |
| :--- | :--- | :--- |
| 24.5°C | 779 | 5.4% |
| 25.0°C | 754 | 5.2% |
| 25.5°C | 4,545 | 31.6% |
| 26.0°C | 5,678 | 39.4% |
| 26.5°C | 2,644 | 18.4% |

*Primary AC Cooling Level Means:*
- `AC-1`: 0.172 | `AC-2`: 0.180 | `AC-3`: 0.217 | `AC-4`: 0.236

### FAMILY_3
- **Mean Optimal Setpoint:** 25.46°C (Std: 0.55°C)
- **Dominant Setpoint:** 25.5°C (35.8%)

| Setpoint | Count | Share |
| :--- | :--- | :--- |
| 24.5°C | 2,093 | 14.5% |
| 25.0°C | 2,309 | 16.0% |
| 25.5°C | 5,159 | 35.8% |
| 26.0°C | 4,184 | 29.1% |
| 26.5°C | 655 | 4.5% |

*Primary AC Cooling Level Means:*
- `AC-1`: 0.384 | `AC-2`: 0.201 | `AC-3`: 0.200 | `AC-4`: 0.357

### FAMILY_4
- **Mean Optimal Setpoint:** 25.64°C (Std: 0.53°C)
- **Dominant Setpoint:** 25.5°C (55.4%)

| Setpoint | Count | Share |
| :--- | :--- | :--- |
| 24.5°C | 1,440 | 10.0% |
| 25.0°C | 125 | 0.9% |
| 25.5°C | 7,970 | 55.4% |
| 26.0°C | 2,711 | 18.8% |
| 26.5°C | 2,154 | 15.0% |

*Primary AC Cooling Level Means:*
- `AC-1`: 0.218 | `AC-2`: 0.212 | `AC-3`: 0.214 | `AC-4`: 0.222

### FAMILY_5
- **Mean Optimal Setpoint:** 25.61°C (Std: 0.45°C)
- **Dominant Setpoint:** 25.5°C (47.2%)

| Setpoint | Count | Share |
| :--- | :--- | :--- |
| 24.5°C | 1,440 | 10.0% |
| 25.0°C | 62 | 0.4% |
| 25.5°C | 6,791 | 47.2% |
| 26.0°C | 5,988 | 41.6% |
| 26.5°C | 119 | 0.8% |

*Primary AC Cooling Level Means:*
- `AC-1`: 0.222 | `AC-2`: 0.231 | `AC-3`: 0.205 | `AC-4`: 0.224

## Per-Scenario Target Statistics
| Scenario ID | Family | Dominant SP | Dom % | Mean SP | Std | Min | Max | Unique |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `SCEN_0001_FAMILY_1` | FAMILY_1 | 25.5°C | 100.0% | 25.50 | 0.00 | 25.5 | 25.5 | 1 |
| `SCEN_0002_FAMILY_2` | FAMILY_2 | 26.0°C | 93.5% | 25.97 | 0.12 | 25.5 | 26.0 | 2 |
| `SCEN_0003_FAMILY_3` | FAMILY_3 | 25.5°C | 91.9% | 25.46 | 0.14 | 25.0 | 25.5 | 2 |
| `SCEN_0004_FAMILY_4` | FAMILY_4 | 25.5°C | 100.0% | 25.50 | 0.00 | 25.5 | 25.5 | 1 |
| `SCEN_0005_FAMILY_5` | FAMILY_5 | 25.5°C | 100.0% | 25.50 | 0.00 | 25.5 | 25.5 | 1 |
| `SCEN_0006_FAMILY_1` | FAMILY_1 | 25.5°C | 91.5% | 25.46 | 0.14 | 25.0 | 25.5 | 2 |
| `SCEN_0007_FAMILY_2` | FAMILY_2 | 26.0°C | 94.4% | 25.97 | 0.12 | 25.5 | 26.0 | 2 |
| `SCEN_0008_FAMILY_3` | FAMILY_3 | 26.5°C | 91.0% | 26.45 | 0.14 | 26.0 | 26.5 | 2 |
| `SCEN_0009_FAMILY_4` | FAMILY_4 | 26.0°C | 95.1% | 25.98 | 0.11 | 25.5 | 26.0 | 2 |
| `SCEN_0010_FAMILY_5` | FAMILY_5 | 25.5°C | 100.0% | 25.50 | 0.00 | 25.5 | 25.5 | 1 |
| `SCEN_0011_FAMILY_1` | FAMILY_1 | 24.5°C | 100.0% | 24.50 | 0.00 | 24.5 | 24.5 | 1 |
| `SCEN_0012_FAMILY_2` | FAMILY_2 | 25.0°C | 91.8% | 24.96 | 0.14 | 24.5 | 25.0 | 2 |
| `SCEN_0013_FAMILY_3` | FAMILY_3 | 24.5°C | 100.0% | 24.50 | 0.00 | 24.5 | 24.5 | 1 |
| `SCEN_0014_FAMILY_4` | FAMILY_4 | 24.5°C | 100.0% | 24.50 | 0.00 | 24.5 | 24.5 | 1 |
| `SCEN_0015_FAMILY_5` | FAMILY_5 | 24.5°C | 100.0% | 24.50 | 0.00 | 24.5 | 24.5 | 1 |
| `SCEN_0016_FAMILY_1` | FAMILY_1 | 26.0°C | 94.0% | 26.03 | 0.12 | 26.0 | 26.5 | 2 |
| `SCEN_0017_FAMILY_2` | FAMILY_2 | 26.5°C | 100.0% | 26.50 | 0.00 | 26.5 | 26.5 | 1 |
| `SCEN_0018_FAMILY_3` | FAMILY_3 | 26.0°C | 100.0% | 26.00 | 0.00 | 26.0 | 26.0 | 1 |
| `SCEN_0019_FAMILY_4` | FAMILY_4 | 26.0°C | 94.7% | 26.03 | 0.11 | 26.0 | 26.5 | 2 |
| `SCEN_0020_FAMILY_5` | FAMILY_5 | 26.0°C | 95.0% | 26.02 | 0.11 | 26.0 | 26.5 | 2 |
| `SCEN_0021_FAMILY_1` | FAMILY_1 | 26.0°C | 100.0% | 26.00 | 0.00 | 26.0 | 26.0 | 1 |
| `SCEN_0022_FAMILY_2` | FAMILY_2 | 25.5°C | 97.2% | 25.49 | 0.08 | 25.0 | 25.5 | 2 |
| `SCEN_0023_FAMILY_3` | FAMILY_3 | 26.0°C | 91.0% | 25.94 | 0.19 | 25.0 | 26.0 | 3 |
| `SCEN_0024_FAMILY_4` | FAMILY_4 | 26.0°C | 94.0% | 25.96 | 0.17 | 25.0 | 26.0 | 3 |
| `SCEN_0025_FAMILY_5` | FAMILY_5 | 25.5°C | 100.0% | 25.50 | 0.00 | 25.5 | 25.5 | 1 |

*(Showing 25 of 100 scenarios. Complete list in label_audit_report.json)*

## Train/Validation/Test Distribution
| Split | Mean SP | Std | Min | Max | Dominant Value | Dominant Share |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 25.65°C | 0.47 | 24.5°C | 26.5°C | 25.5°C | 47.9% |
| **Validation** | 25.52°C | 0.64 | 24.5°C | 26.5°C | 25.5°C | 34.1% |
| **Test** | 25.69°C | 0.55 | 24.5°C | 26.5°C | 26.0°C | 37.6% |

## Distribution Shift
| Comparison | Jensen-Shannon Distance | Wasserstein Distance | Status |
| :--- | :--- | :--- | :--- |
| **train_vs_val** | 0.1996 | 0.1745 | `WARNING` |
| **train_vs_test** | 0.2014 | 0.1312 | `WARNING` |

## Temporal Label Behavior
- **Mean Label Adjustments per Scenario:** 0.56
- **Mean Label Adjustments per Hour:** 0.28 changes/hr
- **Mean Longest Unchanged Period:** 6909.7s (115.2 minutes)

| Scenario ID | Family | Label Changes | Changes/hr | Longest Run (s) | First Label | Final Label |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `SCEN_0001_FAMILY_1` | FAMILY_1 | 0 | 0.0 | 7200s | 25.5°C | 25.5°C |
| `SCEN_0002_FAMILY_2` | FAMILY_2 | 1 | 0.5 | 6730s | 25.5°C | 26.0°C |
| `SCEN_0003_FAMILY_3` | FAMILY_3 | 1 | 0.5 | 6620s | 25.0°C | 25.5°C |
| `SCEN_0004_FAMILY_4` | FAMILY_4 | 0 | 0.0 | 7200s | 25.5°C | 25.5°C |
| `SCEN_0005_FAMILY_5` | FAMILY_5 | 0 | 0.0 | 7200s | 25.5°C | 25.5°C |
| `SCEN_0006_FAMILY_1` | FAMILY_1 | 1 | 0.5 | 6590s | 25.0°C | 25.5°C |
| `SCEN_0007_FAMILY_2` | FAMILY_2 | 1 | 0.5 | 6800s | 25.5°C | 26.0°C |
| `SCEN_0008_FAMILY_3` | FAMILY_3 | 1 | 0.5 | 6550s | 26.0°C | 26.5°C |
| `SCEN_0009_FAMILY_4` | FAMILY_4 | 1 | 0.5 | 6850s | 25.5°C | 26.0°C |
| `SCEN_0010_FAMILY_5` | FAMILY_5 | 0 | 0.0 | 7200s | 25.5°C | 25.5°C |
| `SCEN_0011_FAMILY_1` | FAMILY_1 | 0 | 0.0 | 7200s | 24.5°C | 24.5°C |
| `SCEN_0012_FAMILY_2` | FAMILY_2 | 1 | 0.5 | 6610s | 24.5°C | 25.0°C |
| `SCEN_0013_FAMILY_3` | FAMILY_3 | 0 | 0.0 | 7200s | 24.5°C | 24.5°C |
| `SCEN_0014_FAMILY_4` | FAMILY_4 | 0 | 0.0 | 7200s | 24.5°C | 24.5°C |
| `SCEN_0015_FAMILY_5` | FAMILY_5 | 0 | 0.0 | 7200s | 24.5°C | 24.5°C |

## Label Reasons
| Family | Dominant Reason | Primary Reasons Breakdown |
| :--- | :--- | :--- |
| `FAMILY_1` | **REDUCE_ENERGY** | REDUCE_ENERGY: 59.4%, COMFORT_ENERGY_BALANCE: 15.2%, HIGH_OCCUPANCY: 10.0% |
| `FAMILY_2` | **REDUCE_ENERGY** | REDUCE_ENERGY: 50.5%, HIGH_OCCUPANCY: 30.0%, COMFORT_ENERGY_BALANCE: 15.0% |
| `FAMILY_3` | **HIGH_COMPUTE_LOAD** | HIGH_COMPUTE_LOAD: 90.0%, REDUCE_ENERGY: 10.0% |
| `FAMILY_4` | **HIGH_OCCUPANCY** | HIGH_OCCUPANCY: 95.0%, REDUCE_ENERGY: 5.0% |
| `FAMILY_5` | **HIGH_COMPUTE_LOAD** | HIGH_COMPUTE_LOAD: 75.0%, HIGH_OCCUPANCY: 25.0% |

## Family Separation
Pairwise Jensen-Shannon Distance Matrix across 5 Scenario Families:

| Family | F1 | F2 | F3 | F4 | F5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **F1 (FAMILY_1)** | 0.0000 | 0.1606 | 0.1806 | 0.2327 | 0.2947 |
| **F2 (FAMILY_2)** | 0.1606 | 0.0000 | 0.2741 | 0.2634 | 0.3242 |
| **F3 (FAMILY_3)** | 0.1806 | 0.2741 | 0.0000 | 0.3272 | 0.3079 |
| **F4 (FAMILY_4)** | 0.2327 | 0.2634 | 0.3272 | 0.0000 | 0.3001 |
| **F5 (FAMILY_5)** | 0.2947 | 0.3242 | 0.3079 | 0.3001 | 0.0000 |

## Relationship Audit
| Current State Feature | Pearson Correlation (r) | p-value | Responsiveness Note |
| :--- | :--- | :--- | :--- |
| `room_average_temperature_c` | -0.7892 | 0.0000e+00 | Strong response |
| `maximum_temperature_c` | -0.7035 | 0.0000e+00 | Strong response |
| `temperature_difference_c` | -0.1360 | 0.0000e+00 | Moderate response |
| `total_computer_heat_watts` | -0.1862 | 0.0000e+00 | Moderate response |
| `total_occupancy_heat_watts` | -0.0413 | 0.0000e+00 | Mild/orthogonal response |
| `total_heat_load_watts` | -0.4061 | 0.0000e+00 | Strong response |
| `outdoor_temperature_c` | -0.5724 | 0.0000e+00 | Strong response |
| `humidity_percent` | -0.7388 | 0.0000e+00 | Strong response |

*Note: Correlation measures systematic response to physical drivers, not causality.*

## Warnings
- [WARNING] Train vs Validation shift: JS Distance=0.1996 (WARNING)
- [WARNING] Train vs Test shift: JS Distance=0.2014 (WARNING)

## Final Audit Status
### **WARNING**
- Failures: 0
- Warnings: 2
