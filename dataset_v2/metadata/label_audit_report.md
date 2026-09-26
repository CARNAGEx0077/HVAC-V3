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
| `optimal_ac1_cooling_level` | 0.0 | 1.0 | 0.33 | 0.35 | 0.05 | 40.3% | `PASS` |
| `optimal_ac1_setpoint_c` | 21.5 | 25.5 | 24.10 | 0.78 | 24.0 | 35.1% | `PASS` |
| `optimal_ac2_cooling_level` | 0.0 | 1.0 | 0.28 | 0.31 | 0.05 | 45.6% | `PASS` |
| `optimal_ac2_setpoint_c` | 22.0 | 25.5 | 24.16 | 0.76 | 24.0 | 28.9% | `PASS` |
| `optimal_ac3_cooling_level` | 0.0 | 1.0 | 0.29 | 0.32 | 0.05 | 44.9% | `PASS` |
| `optimal_ac3_setpoint_c` | 22.0 | 25.5 | 24.16 | 0.77 | 24.0 | 29.4% | `PASS` |
| `optimal_ac4_cooling_level` | 0.05 | 1.0 | 0.35 | 0.37 | 0.05 | 37.9% | `PASS` |
| `optimal_ac4_setpoint_c` | 21.5 | 25.5 | 24.08 | 0.81 | 24.0 | 34.9% | `PASS` |
| `optimal_room_setpoint_c` | 22.0 | 25.0 | 24.07 | 0.50 | 24.0 | 38.6% | `PASS` |

### Optimal Room Setpoint Breakdown
| Candidate Setpoint | Timestep Count | Percentage |
| :--- | :--- | :--- |
| 22.0°C | 217 | 0.30% |
| 22.5°C | 359 | 0.50% |
| 23.0°C | 3,591 | 4.99% |
| 23.5°C | 12,258 | 17.03% |
| 24.0°C | 27,815 | 38.63% |
| 24.5°C | 23,494 | 32.63% |
| 25.0°C | 4,266 | 5.92% |

## Per-Family Target Distribution
### FAMILY_1
- **Mean Optimal Setpoint:** 24.04°C (Std: 0.44°C)
- **Dominant Setpoint:** 24.0°C (38.9%)

| Setpoint | Count | Share |
| :--- | :--- | :--- |
| 23.0°C | 739 | 5.1% |
| 23.5°C | 2,742 | 19.0% |
| 24.0°C | 5,604 | 38.9% |
| 24.5°C | 5,315 | 36.9% |

*Primary AC Cooling Level Means:*
- `AC-1`: 0.357 | `AC-2`: 0.273 | `AC-3`: 0.277 | `AC-4`: 0.367

### FAMILY_2
- **Mean Optimal Setpoint:** 24.23°C (Std: 0.49°C)
- **Dominant Setpoint:** 24.5°C (33.4%)

| Setpoint | Count | Share |
| :--- | :--- | :--- |
| 23.0°C | 49 | 0.3% |
| 23.5°C | 2,643 | 18.4% |
| 24.0°C | 4,614 | 32.0% |
| 24.5°C | 4,815 | 33.4% |
| 25.0°C | 2,279 | 15.8% |

*Primary AC Cooling Level Means:*
- `AC-1`: 0.239 | `AC-2`: 0.239 | `AC-3`: 0.280 | `AC-4`: 0.297

### FAMILY_3
- **Mean Optimal Setpoint:** 23.92°C (Std: 0.58°C)
- **Dominant Setpoint:** 24.0°C (35.3%)

| Setpoint | Count | Share |
| :--- | :--- | :--- |
| 22.0°C | 217 | 1.5% |
| 22.5°C | 359 | 2.5% |
| 23.0°C | 760 | 5.3% |
| 23.5°C | 3,700 | 25.7% |
| 24.0°C | 5,084 | 35.3% |
| 24.5°C | 3,631 | 25.2% |
| 25.0°C | 649 | 4.5% |

*Primary AC Cooling Level Means:*
- `AC-1`: 0.449 | `AC-2`: 0.275 | `AC-3`: 0.290 | `AC-4`: 0.439

### FAMILY_4
- **Mean Optimal Setpoint:** 24.27°C (Std: 0.39°C)
- **Dominant Setpoint:** 24.5°C (43.3%)

| Setpoint | Count | Share |
| :--- | :--- | :--- |
| 23.0°C | 28 | 0.2% |
| 23.5°C | 1,088 | 7.6% |
| 24.0°C | 5,712 | 39.7% |
| 24.5°C | 6,234 | 43.3% |
| 25.0°C | 1,338 | 9.3% |

*Primary AC Cooling Level Means:*
- `AC-1`: 0.263 | `AC-2`: 0.270 | `AC-3`: 0.281 | `AC-4`: 0.279

### FAMILY_5
- **Mean Optimal Setpoint:** 23.91°C (Std: 0.48°C)
- **Dominant Setpoint:** 24.0°C (47.2%)

| Setpoint | Count | Share |
| :--- | :--- | :--- |
| 23.0°C | 2,015 | 14.0% |
| 23.5°C | 2,085 | 14.5% |
| 24.0°C | 6,801 | 47.2% |
| 24.5°C | 3,499 | 24.3% |

*Primary AC Cooling Level Means:*
- `AC-1`: 0.363 | `AC-2`: 0.362 | `AC-3`: 0.337 | `AC-4`: 0.394

## Per-Scenario Target Statistics
| Scenario ID | Family | Dominant SP | Dom % | Mean SP | Std | Min | Max | Unique |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `SCEN_0001_FAMILY_1` | FAMILY_1 | 24.0°C | 56.2% | 24.21 | 0.26 | 23.5 | 24.5 | 3 |
| `SCEN_0002_FAMILY_2` | FAMILY_2 | 24.0°C | 99.0% | 24.00 | 0.05 | 23.5 | 24.0 | 2 |
| `SCEN_0003_FAMILY_3` | FAMILY_3 | 24.0°C | 97.2% | 23.99 | 0.08 | 23.5 | 24.0 | 2 |
| `SCEN_0004_FAMILY_4` | FAMILY_4 | 24.0°C | 88.5% | 24.05 | 0.16 | 23.5 | 24.5 | 3 |
| `SCEN_0005_FAMILY_5` | FAMILY_5 | 24.0°C | 100.0% | 24.00 | 0.00 | 24.0 | 24.0 | 1 |
| `SCEN_0006_FAMILY_1` | FAMILY_1 | 24.0°C | 98.9% | 23.99 | 0.05 | 23.5 | 24.0 | 2 |
| `SCEN_0007_FAMILY_2` | FAMILY_2 | 24.5°C | 97.2% | 24.49 | 0.08 | 24.0 | 24.5 | 2 |
| `SCEN_0008_FAMILY_3` | FAMILY_3 | 25.0°C | 90.1% | 24.94 | 0.19 | 24.0 | 25.0 | 3 |
| `SCEN_0009_FAMILY_4` | FAMILY_4 | 24.5°C | 95.8% | 24.47 | 0.13 | 23.5 | 24.5 | 3 |
| `SCEN_0010_FAMILY_5` | FAMILY_5 | 24.0°C | 100.0% | 24.00 | 0.00 | 24.0 | 24.0 | 1 |
| `SCEN_0011_FAMILY_1` | FAMILY_1 | 23.0°C | 72.9% | 23.14 | 0.22 | 23.0 | 23.5 | 2 |
| `SCEN_0012_FAMILY_2` | FAMILY_2 | 23.5°C | 98.1% | 23.49 | 0.07 | 23.0 | 23.5 | 2 |
| `SCEN_0013_FAMILY_3` | FAMILY_3 | 23.5°C | 94.4% | 23.47 | 0.14 | 22.5 | 23.5 | 3 |
| `SCEN_0014_FAMILY_4` | FAMILY_4 | 23.5°C | 97.8% | 23.49 | 0.07 | 23.0 | 23.5 | 2 |
| `SCEN_0015_FAMILY_5` | FAMILY_5 | 23.0°C | 100.0% | 23.00 | 0.00 | 23.0 | 23.0 | 1 |
| `SCEN_0016_FAMILY_1` | FAMILY_1 | 24.5°C | 100.0% | 24.50 | 0.00 | 24.5 | 24.5 | 1 |
| `SCEN_0017_FAMILY_2` | FAMILY_2 | 24.5°C | 100.0% | 24.50 | 0.00 | 24.5 | 24.5 | 1 |
| `SCEN_0018_FAMILY_3` | FAMILY_3 | 24.5°C | 92.6% | 24.46 | 0.13 | 24.0 | 24.5 | 2 |
| `SCEN_0019_FAMILY_4` | FAMILY_4 | 24.5°C | 61.5% | 24.31 | 0.24 | 24.0 | 24.5 | 2 |
| `SCEN_0020_FAMILY_5` | FAMILY_5 | 24.0°C | 98.2% | 24.01 | 0.07 | 24.0 | 24.5 | 2 |
| `SCEN_0021_FAMILY_1` | FAMILY_1 | 24.0°C | 100.0% | 24.00 | 0.00 | 24.0 | 24.0 | 1 |
| `SCEN_0022_FAMILY_2` | FAMILY_2 | 23.5°C | 81.5% | 23.59 | 0.19 | 23.5 | 24.0 | 2 |
| `SCEN_0023_FAMILY_3` | FAMILY_3 | 24.5°C | 96.1% | 24.48 | 0.13 | 23.5 | 24.5 | 3 |
| `SCEN_0024_FAMILY_4` | FAMILY_4 | 25.0°C | 93.2% | 24.95 | 0.19 | 24.0 | 25.0 | 3 |
| `SCEN_0025_FAMILY_5` | FAMILY_5 | 24.5°C | 96.8% | 24.48 | 0.09 | 24.0 | 24.5 | 2 |

*(Showing 25 of 100 scenarios. Complete list in label_audit_report.json)*

## Train/Validation/Test Distribution
| Split | Mean SP | Std | Min | Max | Dominant Value | Dominant Share |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 24.09°C | 0.44 | 22.0°C | 25.0°C | 24.0°C | 48.6% |
| **Validation** | 23.86°C | 0.65 | 22.5°C | 25.0°C | 24.5°C | 34.2% |
| **Test** | 24.22°C | 0.55 | 23.0°C | 25.0°C | 24.5°C | 51.4% |

## Distribution Shift
| Comparison | Jensen-Shannon Distance | Wasserstein Distance | Status |
| :--- | :--- | :--- | :--- |
| **train_vs_val** | 0.5031 | 0.3202 | `FAIL` |
| **train_vs_test** | 0.3967 | 0.2400 | `FAIL` |

## Temporal Label Behavior
- **Mean Label Adjustments per Scenario:** 1.61
- **Mean Label Adjustments per Hour:** 0.81 changes/hr
- **Mean Longest Unchanged Period:** 6599.9s (110.0 minutes)

| Scenario ID | Family | Label Changes | Changes/hr | Longest Run (s) | First Label | Final Label |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `SCEN_0001_FAMILY_1` | FAMILY_1 | 2 | 1.0 | 4050s | 23.5°C | 24.5°C |
| `SCEN_0002_FAMILY_2` | FAMILY_2 | 1 | 0.5 | 7130s | 23.5°C | 24.0°C |
| `SCEN_0003_FAMILY_3` | FAMILY_3 | 1 | 0.5 | 7000s | 23.5°C | 24.0°C |
| `SCEN_0004_FAMILY_4` | FAMILY_4 | 3 | 1.5 | 5950s | 23.5°C | 24.0°C |
| `SCEN_0005_FAMILY_5` | FAMILY_5 | 0 | 0.0 | 7200s | 24.0°C | 24.0°C |
| `SCEN_0006_FAMILY_1` | FAMILY_1 | 1 | 0.5 | 7120s | 23.5°C | 24.0°C |
| `SCEN_0007_FAMILY_2` | FAMILY_2 | 3 | 1.5 | 6960s | 24.0°C | 24.5°C |
| `SCEN_0008_FAMILY_3` | FAMILY_3 | 2 | 1.0 | 6490s | 24.0°C | 25.0°C |
| `SCEN_0009_FAMILY_4` | FAMILY_4 | 4 | 2.0 | 6850s | 23.5°C | 24.5°C |
| `SCEN_0010_FAMILY_5` | FAMILY_5 | 0 | 0.0 | 7200s | 24.0°C | 24.0°C |
| `SCEN_0011_FAMILY_1` | FAMILY_1 | 2 | 1.0 | 4840s | 23.0°C | 23.0°C |
| `SCEN_0012_FAMILY_2` | FAMILY_2 | 1 | 0.5 | 7060s | 23.0°C | 23.5°C |
| `SCEN_0013_FAMILY_3` | FAMILY_3 | 2 | 1.0 | 6800s | 22.5°C | 23.5°C |
| `SCEN_0014_FAMILY_4` | FAMILY_4 | 1 | 0.5 | 7040s | 23.0°C | 23.5°C |
| `SCEN_0015_FAMILY_5` | FAMILY_5 | 0 | 0.0 | 7200s | 23.0°C | 23.0°C |

## Label Reasons
| Family | Dominant Reason | Primary Reasons Breakdown |
| :--- | :--- | :--- |
| `FAMILY_1` | **STATE_B_ENERGY_OPTIMIZED** | STATE_B_ENERGY_OPTIMIZED: 51.4%, STATE_A_OVERHEATING_COOLING: 22.8%, HIGH_OCCUPANCY: 9.5% |
| `FAMILY_2` | **STATE_B_ENERGY_OPTIMIZED** | STATE_B_ENERGY_OPTIMIZED: 44.7%, HIGH_OCCUPANCY: 24.2%, STATE_A_OVERHEATING_COOLING: 22.3% |
| `FAMILY_3` | **HIGH_COMPUTE_LOAD** | HIGH_COMPUTE_LOAD: 53.2%, STATE_A_OVERHEATING_COOLING: 36.9%, STATE_B_ENERGY_OPTIMIZED: 9.7% |
| `FAMILY_4` | **HIGH_OCCUPANCY** | HIGH_OCCUPANCY: 68.4%, STATE_A_OVERHEATING_COOLING: 26.5%, STATE_B_ENERGY_OPTIMIZED: 4.8% |
| `FAMILY_5` | **HIGH_COMPUTE_LOAD** | HIGH_COMPUTE_LOAD: 30.3%, STATE_A_OVERHEATING_COOLING: 27.7%, HIGH_OCCUPANCY: 22.3% |

## Family Separation
Pairwise Jensen-Shannon Distance Matrix across 5 Scenario Families:

| Family | F1 | F2 | F3 | F4 | F5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **F1 (FAMILY_1)** | 0.0000 | 0.3168 | 0.2340 | 0.2957 | 0.1731 |
| **F2 (FAMILY_2)** | 0.3168 | 0.0000 | 0.2685 | 0.1749 | 0.3958 |
| **F3 (FAMILY_3)** | 0.2340 | 0.2685 | 0.0000 | 0.3217 | 0.2739 |
| **F4 (FAMILY_4)** | 0.2957 | 0.1749 | 0.3217 | 0.0000 | 0.3741 |
| **F5 (FAMILY_5)** | 0.1731 | 0.3958 | 0.2739 | 0.3741 | 0.0000 |

## Relationship Audit
| Current State Feature | Pearson Correlation (r) | p-value | Responsiveness Note |
| :--- | :--- | :--- | :--- |
| `room_average_temperature_c` | -0.7997 | 0.0000e+00 | Strong response |
| `maximum_temperature_c` | -0.9189 | 0.0000e+00 | Strong response |
| `temperature_difference_c` | -0.3852 | 0.0000e+00 | Strong response |
| `total_computer_heat_watts` | -0.2998 | 0.0000e+00 | Moderate response |
| `total_occupancy_heat_watts` | +0.0145 | 9.5000e-05 | Mild/orthogonal response |
| `total_heat_load_watts` | -0.3879 | 0.0000e+00 | Strong response |
| `outdoor_temperature_c` | -0.3400 | 0.0000e+00 | Strong response |
| `humidity_percent` | -0.3306 | 0.0000e+00 | Strong response |

*Note: Correlation measures systematic response to physical drivers, not causality.*

## Warnings
- [WARNING] Train vs Validation shift: JS Distance=0.5031 (FAIL)
- [WARNING] Train vs Test shift: JS Distance=0.3967 (FAIL)

## Final Audit Status
### **WARNING**
- Failures: 0
- Warnings: 2
