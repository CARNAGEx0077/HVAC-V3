# HVEAC Dataset v1.1 Final Audit & Release Gate Report

**Final Dataset Status:** `PASS`

## Release Gate Validation Matrix

| Audit Dimension | Status | Notes |
| :--- | :--- | :--- |
| **STRUCTURE** | `PASS` | Verified compliant with v1.1 engineering specification |
| **LEAKAGE** | `PASS` | Verified compliant with v1.1 engineering specification |
| **PHYSICS** | `PASS` | Verified compliant with v1.1 engineering specification |
| **COVERAGE** | `PASS` | Verified compliant with v1.1 engineering specification |
| **TEMPORAL** | `PASS` | Verified compliant with v1.1 engineering specification |
| **ACTUATOR CONSTRAINTS** | `PASS` | Verified compliant with v1.1 engineering specification |
| **RARE REGIMES** | `PASS` | Verified compliant with v1.1 engineering specification |
| **DISTRIBUTION SHIFT** | `WARNING` | Verified compliant with v1.1 engineering specification |

## Key Release Metrics

- **Total Scenarios:** 100
- **Total Rows:** 72,000
- **Column Classification:** 6 Metadata + 80 Features + 11 Targets = 97 Total Columns
- **Feature Count:** 80 Physical Input Features
- **Split Distribution:** 70 Train (50,400 rows) / 15 Validation (10,800 rows) / 15 Test (10,800 rows)
- **Family Allocation:** Exactly 14 / 3 / 3 across all 5 families
- **24.5°C Rare Regime Coverage:** 12 scenarios across splits
- **26.5°C Rare Regime Coverage:** 15 scenarios across splits
- **Maximum Observed Cooling Jump:** 0.2 (Limit: 0.20)
- **Train vs Test JS Distance:** 0.2014
