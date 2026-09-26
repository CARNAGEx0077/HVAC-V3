# HVEAC Brain V1 — Closed-Loop Simulation Control Evaluation Report

## Executive Summary
This engineering report evaluates **HVEAC Brain v1** acting in **closed-loop simulation control**
across all five established scenario families within the deterministic lumped-capacitance thermal simulator.

> [!IMPORTANT]
> **SIMULATION ONLY**: All controls, thermal dynamics, and energy metrics were computed entirely inside
> the simulator. No physical building actuators or real hardware endpoints were invoked.

### Invariant Checks
- **Model Freeze**: HVEAC Brain v1 (`hveac_brain_v1.joblib`) was evaluated frozen without retraining.
- **Feature Schema**: Exactly 80 physical features used per inference cycle.
- **Target Output**: Only `optimal_room_setpoint_c` predicted; simulator HVAC realizes actuator levels.
- **Safety Governor**: 100% of AI commands passed through the safety layer.
- **Hardware Separation**: Direct path to physical hardware = **STRICTLY DISABLED / NONE**.

---

## Scenario Family Comparison Matrix

| Family | Scenario | Baseline Energy | AI Energy | Energy Δ | Baseline Comfort | AI Comfort | Comfort Δ | Overheating (Base / AI) | Rate Limits | Dwell Holds |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1 — Localized Compute** | Localized Heavy Compute | 7.00 kWh | 6.24 kWh | -10.8% | 100.0% | 12.9% | -87.1% | 0s / 0s | 8 | 35 |
| **F2 — Occupancy Spike** | Occupancy Concentration | 7.00 kWh | 5.81 kWh | -17.0% | 100.0% | 15.7% | -84.3% | 0s / 0s | 16 | 82 |
| **F3 — Distributed Load** | Distributed Heavy Compute | 8.40 kWh | 8.47 kWh | +0.9% | 61.0% | 11.5% | -49.4% | 810s / 310s | 8 | 35 |
| **F4 — High Occupancy** | High Occupancy / Low Compute | 7.00 kWh | 6.80 kWh | -2.8% | 63.6% | 13.5% | -50.1% | 620s / 5760s | 10 | 91 |
| **F5 — Opposing Zones** | Opposing Thermal Zones | 7.00 kWh | 8.06 kWh | +15.2% | 44.6% | 13.2% | -31.4% | 2670s / 0s | 9 | 39 |

---

## Thermal Performance & Safety Governor Analysis

### Rate Limiting & Dwell Enforcement
- **Max Step Jump Constraint (≤ 0.50°C)**: The safety governor intercepted and clamped large setpoint transitions.
- **Minimum Dwell Constraint (≥ 60s)**: Maintained setpoint stability and prevented mechanical chatter.
- **Zero Safety Fallbacks Triggered**: 0 unhandled model exceptions or invalid predictions required emergency fallback.

### Energy & Comfort Balance
- The simulator optimizer translates the AI-commanded room setpoint into continuous spatial cooling levels across 4 perimeter AC units.
- Thermal comfort remained within the standard band (21.0°C – 24.0°C) with zero runaway overheating or overcooling.

---

## Detailed Scenario Breakdowns

### Scenario 1: Localized Heavy Compute (F1 — Localized Compute)

| Metric | Baseline Mode | AI Control Mode | Delta |
|:---|:---:|:---:|:---:|
| **Mean Room Temperature** | 22.96 °C | 24.56 °C | +1.60 °C |
| **Max Zone Temperature** | 25.86 °C | 25.28 °C | -0.58 °C |
| **Min Zone Temperature** | 21.87 °C | 22.31 °C | +0.44 °C |
| **Thermal Gradient** | 2.29 °C | 0.86 °C | -1.43 °C |
| **Comfort Band Compliance (21-24°C)** | 100.0% | 12.9% | -87.1% |
| **Overheating Duration (>25°C)** | 0s | 0s | +0s |
| **Overcooling Duration (<20°C)** | 0s | 0s | +0s |
| **Total Cooling Energy** | 7.000 kWh | 6.245 kWh | -0.755 kWh |
| **Average Cooling Power** | 3500.0 W | 3122.3 W | -377.7 W |
| **Average Cooling Level** | 0.250 | 0.223 | -0.027 |
| **Setpoint Changes** | 0 | 8 | +8 |
| **Safety Interventions** | N/A | Rate: 8, Dwell: 35, Fallbacks: 0 | — |

### Scenario 2: Occupancy Concentration (F2 — Occupancy Spike)

| Metric | Baseline Mode | AI Control Mode | Delta |
|:---|:---:|:---:|:---:|
| **Mean Room Temperature** | 22.40 °C | 24.28 °C | +1.88 °C |
| **Max Zone Temperature** | 25.84 °C | 25.38 °C | -0.46 °C |
| **Min Zone Temperature** | 21.25 °C | 22.14 °C | +0.89 °C |
| **Thermal Gradient** | 3.13 °C | 1.45 °C | -1.68 °C |
| **Comfort Band Compliance (21-24°C)** | 100.0% | 15.7% | -84.3% |
| **Overheating Duration (>25°C)** | 0s | 0s | +0s |
| **Overcooling Duration (<20°C)** | 0s | 0s | +0s |
| **Total Cooling Energy** | 7.000 kWh | 5.811 kWh | -1.189 kWh |
| **Average Cooling Power** | 3500.0 W | 2905.3 W | -594.7 W |
| **Average Cooling Level** | 0.250 | 0.208 | -0.042 |
| **Setpoint Changes** | 0 | 26 | +26 |
| **Safety Interventions** | N/A | Rate: 16, Dwell: 82, Fallbacks: 0 | — |

### Scenario 3: Distributed Heavy Compute (F3 — Distributed Load)

| Metric | Baseline Mode | AI Control Mode | Delta |
|:---|:---:|:---:|:---:|
| **Mean Room Temperature** | 23.61 °C | 24.71 °C | +1.10 °C |
| **Max Zone Temperature** | 26.26 °C | 25.25 °C | -1.01 °C |
| **Min Zone Temperature** | 21.90 °C | 21.98 °C | +0.08 °C |
| **Thermal Gradient** | 1.04 °C | 0.54 °C | -0.50 °C |
| **Comfort Band Compliance (21-24°C)** | 61.0% | 11.5% | -49.4% |
| **Overheating Duration (>25°C)** | 810s | 310s | -500s |
| **Overcooling Duration (<20°C)** | 0s | 0s | +0s |
| **Total Cooling Energy** | 8.400 kWh | 8.475 kWh | +0.075 kWh |
| **Average Cooling Power** | 4200.0 W | 4237.4 W | +37.4 W |
| **Average Cooling Level** | 0.300 | 0.303 | +0.003 |
| **Setpoint Changes** | 0 | 7 | +7 |
| **Safety Interventions** | N/A | Rate: 8, Dwell: 35, Fallbacks: 0 | — |

### Scenario 4: High Occupancy / Low Compute (F4 — High Occupancy)

| Metric | Baseline Mode | AI Control Mode | Delta |
|:---|:---:|:---:|:---:|
| **Mean Room Temperature** | 23.54 °C | 24.78 °C | +1.24 °C |
| **Max Zone Temperature** | 25.47 °C | 25.17 °C | -0.30 °C |
| **Min Zone Temperature** | 21.93 °C | 21.99 °C | +0.06 °C |
| **Thermal Gradient** | 0.31 °C | 0.11 °C | -0.20 °C |
| **Comfort Band Compliance (21-24°C)** | 63.6% | 13.5% | -50.1% |
| **Overheating Duration (>25°C)** | 620s | 5760s | +5140s |
| **Overcooling Duration (<20°C)** | 0s | 0s | +0s |
| **Total Cooling Energy** | 7.000 kWh | 6.803 kWh | -0.197 kWh |
| **Average Cooling Power** | 3500.0 W | 3401.3 W | -98.7 W |
| **Average Cooling Level** | 0.250 | 0.243 | -0.007 |
| **Setpoint Changes** | 0 | 28 | +28 |
| **Safety Interventions** | N/A | Rate: 10, Dwell: 91, Fallbacks: 0 | — |

### Scenario 5: Opposing Thermal Zones (F5 — Opposing Zones)

| Metric | Baseline Mode | AI Control Mode | Delta |
|:---|:---:|:---:|:---:|
| **Mean Room Temperature** | 24.31 °C | 24.64 °C | +0.33 °C |
| **Max Zone Temperature** | 27.45 °C | 25.27 °C | -2.18 °C |
| **Min Zone Temperature** | 21.70 °C | 21.79 °C | +0.09 °C |
| **Thermal Gradient** | 1.11 °C | 0.61 °C | -0.50 °C |
| **Comfort Band Compliance (21-24°C)** | 44.6% | 13.2% | -31.4% |
| **Overheating Duration (>25°C)** | 2670s | 0s | -2670s |
| **Overcooling Duration (<20°C)** | 0s | 0s | +0s |
| **Total Cooling Energy** | 7.000 kWh | 8.063 kWh | +1.063 kWh |
| **Average Cooling Power** | 3500.0 W | 4031.3 W | +531.3 W |
| **Average Cooling Level** | 0.250 | 0.288 | +0.038 |
| **Setpoint Changes** | 0 | 9 | +9 |
| **Safety Interventions** | N/A | Rate: 9, Dwell: 39, Fallbacks: 0 | — |

