# HVEAC Dataset v2 Final Audit & Release Gate Report

**Final Dataset Status:** `PASS`

## Release Gate Validation Matrix

| Audit Dimension | Status | Notes |
| :--- | :--- | :--- |
| **STRUCTURE** | `PASS` | Verified compliant with v2 engineering specification |
| **LEAKAGE** | `PASS` | Verified compliant with v2 engineering specification |
| **PHYSICS** | `PASS` | Verified compliant with v2 engineering specification |
| **COVERAGE** | `PASS` | Verified compliant with v2 engineering specification |
| **TEMPORAL** | `PASS` | Verified compliant with v2 engineering specification |
| **ACTUATOR CONSTRAINTS** | `PASS` | Verified compliant with v2 engineering specification |
| **RARE REGIMES** | `PASS` | Verified compliant with v2 engineering specification |
| **DISTRIBUTION SHIFT** | `PASS` | Verified compliant with v2 engineering specification |

## Key Release Metrics

- **Total Scenarios:** 100
- **Total Rows:** 72,000
- **Column Classification:** 6 Metadata + 80 Features + 11 Targets = 97 Total Columns
- **Feature Count:** 80 Physical Input Features
- **Split Distribution:** 70 Train (50,400 rows) / 15 Validation (10,800 rows) / 15 Test (10,800 rows)
- **Family Allocation:** Exactly 14 / 3 / 3 across all 5 families
- **Aggressive Cooling Regime Coverage:** 14 scenarios across splits
- **Relaxed Setpoint Regime Coverage:** 51 scenarios across splits
- **Maximum Observed Cooling Jump:** 0.2 (Limit: 0.20)
- **Train vs Test JS Distance:** 0.0839
