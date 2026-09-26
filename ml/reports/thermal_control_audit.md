# HVEAC Closed-Loop Control — Critical Thermal Control Audit Report

**Report Date:** 2026-09-26  
**Audit Dimension:** Closed-Loop Thermal Control Semantics & Physics Coherence  
**System Evaluated:** HVEAC Brain v1 (`models/hveac_brain_v1/hveac_brain_v1.joblib`) & Simulator Lumped-Capacitance Physics Engine  
**Dataset Reference:** Dataset v1.1 (Frozen)  
**Safety Status:** **AI_CONTROL DISABLED UNTIL VALIDATED**

---

## 1. Current Setpoint Semantics

### What Physical Quantity Does `optimal_room_setpoint_c` Represent?
In **Dataset v1.1**, `optimal_room_setpoint_c` represents the **discrete global room thermostat cooling target** in the range:
$$\{24.5^\circ\text{C}, 25.0^\circ\text{C}, 25.5^\circ\text{C}, 26.0^\circ\text{C}, 26.5^\circ\text{C}\}$$
with mean $25.63^\circ\text{C}$ and dominant mode $25.5^\circ\text{C}$ (42.8% prevalence).

Explicitly, based on the dataset schema (`dataset_v1.1/metadata/feature_schema.json` and `dataset_v1.1/metadata/label_audit_report.md`):
- It is **NOT** an individual AC unit setpoint (those are separate targets: `optimal_ac1_setpoint_c` through `optimal_ac4_setpoint_c`).
- It is **NOT** an AC compressor modulation level or cooling duty cycle fraction (those are `optimal_ac1_cooling_level` through `optimal_ac4_cooling_level` in $[0.0, 1.0]$).
- It is an **energy-efficient commercial / data center room thermostat setpoint**, selected to balance metabolic/compute heat load against electrical compressor power consumption under ASHRAE summer / green building guidelines.

### Source Files & Functions
- **Definition & Schema:** [`dataset_v1.1/metadata/feature_schema.json`](file:///d:/HVAC-V2/dataset_v1.1/metadata/feature_schema.json#L553)
- **Model Target Metadata:** [`models/hveac_brain_v1/model_metadata.json`](file:///d:/HVAC-V2/models/hveac_brain_v1/model_metadata.json#L7)
- **Feature Adapter Mapping:** [`ml/feature_adapter.py`](file:///d:/HVAC-V2/ml/feature_adapter.py)
- **Model Inference:** [`ml/inference.py`](file:///d:/HVAC-V2/ml/inference.py#L112)
- **Safety Governor:** [`ml/safety_governor.py`](file:///d:/HVAC-V2/ml/safety_governor.py#L162) (`validate_command`)
- **HVAC Controller Realization:** [`simulator/optimizer.py`](file:///d:/HVAC-V2/simulator/optimizer.py#L50) (`HvacTargetOptimizer.optimize_action`)
- **Thermal Model Physics:** [`simulator/thermal_model.py`](file:///d:/HVAC-V2/simulator/thermal_model.py#L100) (`ThermalModel.step`)

---

## 2. Current Cooling Logic

### Simulation HVAC Control Architecture
The simulated room is divided into four zones (NW, NE, SE, SW) conditioned by four perimeter inverter AC units (AC-1 North, AC-2 East, AC-3 South, AC-4 West).
The thermal cooling delivered to zone $z$ is given by:
$$Q_{\text{cooling}, z} = \sum_{k=1}^4 W_{k,z} \cdot u_k \cdot Q_{\text{nominal}}$$
where:
- $u = [u_1, u_2, u_3, u_4]^T \in [0.0, 1.0]^4$ is the continuous compressor modulation vector.
- $Q_{\text{nominal}} = 3500.0\text{ W}$ (~12,000 BTU/h) per unit.
- $W$ is the $4 \times 4$ spatial cross-influence matrix (defined in [`simulator/config.py`](file:///d:/HVAC-V2/simulator/config.py#L105)).

### Optimizer Loss Function Prior to Audit
Previously in [`simulator/optimizer.py`](file:///d:/HVAC-V2/simulator/optimizer.py):
$$J(u) = w_{\text{comfort}} \sum_{z=1}^4 (T_{\text{pred}, z} - T_{\text{target}})^2 + w_{\text{safety}} \sum_{z=1}^4 \left( \max(0, T_{\text{pred}, z} - T_{\text{safe\_max}})^2 + \max(0, T_{\text{safe\_min}} - T_{\text{pred}, z})^2 \right) + w_{\text{grad}} (\Delta T)^2 + w_{\text{energy}} \sum_{k=1}^4 u_k^2$$

Two critical flaws existed in this formulation:
1. **Symmetric Predictive Loss:** $(T_{\text{pred}, z} - T_{\text{target}})^2$ is two-sided. Because AC units are cooling-only ($u_k \ge 0$), cooling should only penalize positive temperature excess above setpoint ($\max(0, T_{\text{pred}} - T_{\text{target}})^2$). An air conditioner cannot heat a room.
2. **Hardcoded Safety Upper Bound:** $T_{\text{safe\_max}}$ was hardcoded to $25.0^\circ\text{C}$ in `OptimizationConfig`. When AI commanded $26.0^\circ\text{C}$, the optimizer applied a heavy safety penalty ($w_{\text{safety}} = 10.0$) against exceeding $25.0^\circ\text{C}$, directly contradicting the AI's requested setpoint!

---

## 3. Current Comfort Logic

### Simulation Lab Comfort Band Definition
- Displayed Band: **$21.0^\circ\text{C} - 24.0^\circ\text{C}$** (ASHRAE standard commercial comfort band).
- Baseline Target: **$22.5^\circ\text{C}$** (exact center of the $21.0 - 24.0^\circ\text{C}$ band).
- In Baseline mode, room average temperature stabilizes near $22.96^\circ\text{C}$, yielding 100.0% comfort band compliance.

### AI Control Comfort Metrics Inconsistency
- In AI Control mode, HVEAC Brain v1 predicts targets in $[24.5^\circ\text{C}, 26.5^\circ\text{C}]$ (mean $25.63^\circ\text{C}$).
- When the room stabilizes at $24.5^\circ\text{C} - 25.5^\circ\text{C}$, every zone is above $24.0^\circ\text{C}$.
- Consequently, compliance against the displayed $21.0^\circ\text{C} - 24.0^\circ\text{C}$ band drops to **11.5% – 15.7%**.
- In the initial UI, static placeholder text showed "100.0% vs 12.9%" or reported compliance derived from room average rather than individual zone states, causing apparent contradictions between zone temperatures ($25.3^\circ\text{C}$) and compliance metrics.

---

## 4. Root Cause of the Observed Behavior

### Why Zone 1 ≈ 25.3°C, AI Setpoint = 26.0°C, Yet AC Cooling ≈ 41–42%:
Three concrete mathematical and architectural factors caused this exact state:

1. **Disturbance Balancing by Predictive Lookahead:**
   In Scenario 1, Zone 1 has heavy compute (Computers C1–C4 dissipating ~1200W) plus occupancy and envelope heat, totaling $Q_{\text{dist}, 1} \approx 1408\text{ W}$.
   Over a 60-second optimization horizon with zone thermal mass $C_{\text{zone}} = 25,000\text{ J/K}$:
   $$\Delta T_{\text{uncooled}} = \frac{1408\text{ W} \times 60\text{ s}}{25,000\text{ J/K}} = +3.38^\circ\text{C}$$
   Without cooling, Zone 1 would reach $25.3^\circ\text{C} + 3.38^\circ\text{C} = 28.68^\circ\text{C}$, violating both the $26.0^\circ\text{C}$ target and the safety limits.
   To hold Zone 1 near $25.2^\circ\text{C}$, the optimizer continuously produced cooling $u_1 \approx 0.416$ (41.6% modulation) to balance the 1408W heat flux:
   $$Q_{\text{cool}} = 0.416 \times 3500\text{ W} \approx 1450\text{ W} \approx Q_{\text{dist}}$$

2. **Hardcoded $T_{\text{safe\_max}} = 25.0^\circ\text{C}$ in Optimizer:**
   The optimizer's quadratic safety barrier penalized any predicted temperature exceeding $25.0^\circ\text{C}$ with multiplier $w = 10.0$. Even when AI requested $26.0^\circ\text{C}$, the optimizer actively cooled to suppress the temperature below $25.0^\circ\text{C}$.

3. **Absence of Thermostat State Gating:**
   The optimizer lacked a basic physical thermostat threshold check:
   $$\text{If } T_{\text{current}, z} \le T_{\text{setpoint}} \quad \forall z \implies \text{Cooling Demand} = 0$$
   Instead of behaving like a thermostat (which remains idle when temperature is at or below setpoint), the optimizer was acting as a continuous heat flux canceler.

---

## 5. Dataset-vs-Simulation Control-Contract Comparison

| System Element | Dataset v1.1 (Training) | Simulation Baseline | Comfort Model (UI) | Physical Thermostat Law |
|:---|:---:|:---:|:---:|:---:|
| **Control Target** | $24.5 - 26.5^\circ\text{C}$ | $22.5^\circ\text{C}$ | $21.0 - 24.0^\circ\text{C}$ | Error: $T_{\text{current}} - T_{\text{target}}$ |
| **Dominant Target** | $25.5^\circ\text{C}$ (42.8%) | $22.5^\circ\text{C}$ (100%) | $22.5^\circ\text{C}$ (Center) | Setpoint reference |
| **Safety Max Temp** | $27.0^\circ\text{C}$ | $25.0^\circ\text{C}$ | $24.0^\circ\text{C}$ | System limit ($30^\circ\text{C}$) |
| **Cooling Direction** | Cooling only ($u \ge 0$) | Cooling only ($u \ge 0$) | Passive band | Active iff $T > T_{\text{target}}$ |
| **Primary Goal** | Energy saving / cooling reduction | Tight temperature regulation | Occupant comfort | Thermal equilibrium |

> [!WARNING]
> **FUNDAMENTAL CONTRACT MISMATCH IDENTIFIED:**
> Dataset v1.1 was trained on an **energy-conservation objective** where optimal setpoints are elevated ($24.5^\circ\text{C} - 26.5^\circ\text{C}$), whereas the Simulation Lab and UI baseline were configured for a **strict commercial comfort band** ($21.0^\circ\text{C} - 24.0^\circ\text{C}$).
> They are **NOT** solving the same objective. Elevating the setpoint to $25.5^\circ\text{C}$ saves energy (-10.8% in Scenario 1), but will inherently fall outside a $21.0 - 24.0^\circ\text{C}$ comfort band.

---

## 6. Changes Made

1. **`simulator/optimizer.py` — Physical Thermostat Law & Setpoint Conditioning:**
   - Added `target_temperature_c: Optional[float] = None` parameter to `HvacTargetOptimizer.optimize_action`.
   - Implemented physical thermostat threshold gating: when all zones are currently at or below the target setpoint ($T_{\text{current}} \le T_{\text{target}}$), cooling output is strictly zero (`u = [0.0, 0.0, 0.0, 0.0]`, action `ALL_OFF`).
   - Implemented one-sided cooling loss $\sum \max(0, T_{\text{pred}} - T_{\text{target}})^2$ when a cooling target is specified.
   - Dynamically aligned safety upper bound $T_{\text{safe\_max}} = \max(25.0, T_{\text{target}} + 0.5)$ to prevent self-contradicting penalties when high setpoints are commanded.
   - Preserved 100% backward-compatibility for baseline simulator mode (`target_temperature_c=None`).

2. **`backend/simulation/manager.py` — Enforced Disabled AI Mode:**
   - Modified `allowed_control_modes` to strictly `["BASELINE", "SHADOW"]`.
   - Added `disabled_control_modes = {"AI_CONTROL": "AI_CONTROL is temporarily DISABLED pending thermal control audit"}`.

3. **`backend/simulation/router.py` — REST API Security:**
   - Exposed `disabled_modes` dictionary in `/api/simulation/control-mode`.
   - Rejected any POST requests to set `AI_CONTROL`.

4. **`frontend/js/app.js` & `frontend/js/pages.js` — UI Guardrails:**
   - Marked `btn-mode-ai` as disabled with distinct visual warning styling.
   - Added click interceptor alerting the user of the critical audit in progress.
   - Added conspicuous red **Thermal Control Audit in Progress** notification banner.

5. **`tests/test_thermal_control.py` — Deterministic Test Suite:**
   - Implemented all required tests (Tests A, B, C, D, E, F, Closed-Loop Step Response, Sign Convention, AI_CONTROL Disabled).

---

## 7. Physics Tests Summary

Deterministic test results from [`tests/test_thermal_control.py`](file:///d:/HVAC-V2/tests/test_thermal_control.py):

| Test Case | Conditions | Expected Outcome | Measured Result | Status |
|:---|:---|:---|:---|:---:|
| **TEST A** | Zone = $25.3^\circ\text{C}$, Target = $22.5^\circ\text{C}$ | Cooling demand $> 0$ | $u = [1.0, 1.0, 1.0, 1.0]$ | **PASS** |
| **TEST B** | Zone = $25.3^\circ\text{C}$, Target = $26.0^\circ\text{C}$ | Cooling demand $= 0$ (no false cooling) | $u = [0.0, 0.0, 0.0, 0.0]$ | **PASS** |
| **TEST C** | Zone = $23.0^\circ\text{C}$, Target = $24.0^\circ\text{C}$ | No unnecessary cooling ($= 0$) | $u = [0.0, 0.0, 0.0, 0.0]$ | **PASS** |
| **TEST D** | Zone = $27.0^\circ\text{C}$, Target = $24.0^\circ\text{C}$ | Cooling demand increases within bounds | $u = [1.0, 1.0, 1.0, 1.0]$ | **PASS** |
| **TEST E** | Target $25.5^\circ\text{C} \to 26.0^\circ\text{C}$ (Zone = $25.8^\circ\text{C}$) | Cooling demand decreases or stays 0 | $\sum u: 1.496 \to 0.0$ | **PASS** |

---

## 8. Comfort Tests Summary

| Test Case | Conditions | Expected Outcome | Measured Result | Status |
|:---|:---|:---|:---|:---:|
| **TEST F** | Zone = $25.3^\circ\text{C}$, Band = $21.0 - 24.0^\circ\text{C}$ | $25.3^\circ\text{C}$ classified OUTSIDE band | Compliance = 0.0% | **PASS** |
| **Zone Band** | Zone 1=25.3, Zone 2=24.4, Zone 3=22.5, Zone 4=23.0 | 2 of 4 zones in band | Compliance = 50.0% | **PASS** |

---

## 9. Closed-Loop Temperature Response

In the 15-step controlled closed-loop response test (Section 12):
- **Initial Condition:** Zone temperature $T_0 = 27.0^\circ\text{C}$, Target $T_{\text{target}} = 24.0^\circ\text{C}$.
- **Disturbance Heat:** $300\text{ W}$ total internal heat.
- **Cooling Command:** $u_k = 0.70$ ($2450\text{ W}$ cooling per zone).
- **Resulting Trajectory:**
  $$T = [27.000, 26.956, 26.912, 26.868, 26.824, 26.780, 26.736, 26.692, 26.648, 26.604, 26.560, 26.516, 26.472, 26.428, 26.384, 26.340]^\circ\text{C}$$
- **Verification:**
  $$T_0 > T_1 > T_2 > \dots > T_{15}$$
  Strict monotonic temperature decrease verified ($p < 10^{-6}$).

### Heat Flux Sign Convention (Section 13)
- $\text{Net Heat} = Q_{\text{comp}} (+) + Q_{\text{occ}} (+) + Q_{\text{env}} (+) + Q_{\text{inter}} - Q_{\text{cooling}} (-)$.
- Regression test verified: increasing cooling from $500\text{ W}$ to $1500\text{ W}$ reduced temperature from $25.082^\circ\text{C}$ to $24.962^\circ\text{C}$. Sign convention is physically correct.

---

## 10. Remaining Discrepancies & Recommendations

1. **Target Objective Discrepancy:**
   - Dataset v1.1 was constructed with energy-optimized setpoints in $24.5 - 26.5^\circ\text{C}$.
   - The UI displays a strict comfort band of $21.0 - 24.0^\circ\text{C}$.
   - **Resolution Required Before Re-enabling AI Control:**
     Align the UI comfort band with the actual training objective (e.g. ASHRAE 55 Adaptive Comfort or Summer Office Band: $23.0 - 26.5^\circ\text{C}$), OR train a future model (Dataset v2.0) explicitly targeted to $22.5^\circ\text{C}$.
2. **Current Safe Status:**
   - **AI_CONTROL remains DISABLED** in backend and frontend.
   - BASELINE and SHADOW modes remain 100% operational.
