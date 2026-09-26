# HVEAC Control Objective Audit — Analysis of V1 Optimization Flaws

**Document Date:** 2026-09-26  
**Subject:** Mathematical and Architectural Audit of Dataset v1 / Brain v1 Control Objective  
**Target File:** `dataset_v2/metadata/v1_control_objective_audit.md`

---

## 1. Executive Summary

In HVEAC Dataset v1.1, the machine learning model was trained to predict `optimal_room_setpoint_c` from 80 physical features. However, closed-loop simulation revealed a critical control failure:
$$\text{Current Zone Temperature} = 25.0^\circ\text{C}, \quad \text{Thermal Trend} = \text{Rising} \implies \text{Optimizer Target} = 26.0^\circ\text{C}$$

For an air-conditioning cooling controller, commanding a setpoint **warmer** than the currently overheating room allows the space to warm further without engaging active cooling, contradicting basic control principles.

This audit documents the exact mathematical formulation, decision process, and architectural root causes that generated this behavior in the V1 optimizer.

---

## 2. V1 Optimization Formulation

In `generator/optimization/objective.py` and `generator/optimization/optimizer.py`, candidate room setpoints $s \in \{22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0\}^\circ\text{C}$ were evaluated over a forward lookahead horizon $\Delta t_{\text{horizon}} = 120\text{ s}$ ($8 \times 15\text{s}$ sub-steps) using `_forward_rollout`.

### Cost Function
$$J(s) = w_{\text{comfort}} \cdot \text{Cost}_{\text{comfort}} + w_{\text{overheat}} \cdot \text{Cost}_{\text{overheat}} + w_{\text{overcool}} \cdot \text{Cost}_{\text{overcool}} + w_{\text{energy}} \cdot \text{Cost}_{\text{energy}} + w_{\text{switch}} \cdot \text{Cost}_{\text{switch}}$$

Where:
- $w_{\text{comfort}} = 2.5$
- $w_{\text{overheat}} = 5.0$
- $w_{\text{overcool}} = 3.5$
- $w_{\text{energy}} = 0.9$
- $w_{\text{switch}} = 0.4$

### Component Formulations:
1. **Comfort Penalty:**
   $$\text{Cost}_{\text{comfort}} = (\text{PMV} - \text{target\_pmv})^2 = (\text{PMV} - 0.0)^2$$
   Calculated via Fanger's ISO 7730 / ASHRAE 55 PMV equation with:
   - Metabolic rate: $1.1\text{ met}$ (seated office work)
   - Clothing insulation: $0.5\text{ clo}$ (light summer indoor clothing)
   - Air speed: $0.15\text{ m/s}$
   - Relative humidity: $\approx 50 - 65\%$

2. **Overheating Penalty:**
   $$\text{Cost}_{\text{overheat}} = \max\left(0.0, T_{\text{pred}} - T_{\text{upper\_bound}}\right)^2$$
   where $T_{\text{upper\_bound}} = 26.5^\circ\text{C}$ (`upper_comfort_threshold_c` in `generator/config.py`).

3. **Overcooling Penalty:**
   $$\text{Cost}_{\text{overcool}} = \max\left(0.0, T_{\text{lower\_bound}} - T_{\text{pred}}\right)^2$$
   where $T_{\text{lower\_bound}} = 22.5^\circ\text{C}$.

4. **Energy Penalty:**
   $$\text{Cost}_{\text{energy}} = \left(\frac{Q_{\text{cooling}}}{Q_{\text{max}}}\right)^{1.2} = \left(\frac{\sum u_k \cdot 4500\text{ W}}{18,000\text{ W}}\right)^{1.2}$$

5. **Switching Penalty:**
   $$\text{Cost}_{\text{switch}} = \frac{|s - s_{\text{current}}|}{5.0^\circ\text{C}}$$

---

## 3. Mathematical Root Cause of the $25.0^\circ\text{C} \to 26.0^\circ\text{C}$ Decision

When Zone 1 is at $25.0^\circ\text{C}$ with $600\text{ W}$ computer heat and $200\text{ W}$ occupancy heat (total disturbance load $\approx 900\text{ W}$):

### Candidate Comparison:
| Metric | Candidate $s = 24.5^\circ\text{C}$ | Candidate $s = 25.0^\circ\text{C}$ | Candidate $s = 25.5^\circ\text{C}$ | Candidate $s = 26.0^\circ\text{C}$ |
|:---|:---:|:---:|:---:|:---:|
| **Predicted End Temp ($T_{\text{pred}}$)** | $24.74^\circ\text{C}$ | $25.09^\circ\text{C}$ | $25.44^\circ\text{C}$ | $25.79^\circ\text{C}$ |
| **PMV Value** | $-0.31$ | $-0.19$ | $-0.07$ | $+0.06$ |
| **Comfort Penalty ($2.5 \cdot \text{PMV}^2$)** | $0.2465$ | $0.0902$ | $0.0110$ | $0.0085$ |
| **Overheat Penalty ($T > 26.5$)** | **$0.0000$** | **$0.0000$** | **$0.0000$** | **$0.0000$** |
| **Overcool Penalty ($T < 22.5$)** | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ |
| **Cooling Power Needed ($u_{\text{avg}}$)** | $0.29$ | $0.27$ | $0.25$ | $0.20$ |
| **Energy Cost ($0.9 \cdot (Q/Q_{\text{max}})^{1.2}$)** | $0.1937$ | $0.1845$ | $0.1787$ | $0.1763$ |
| **Switching Cost** | $0.1200$ | $0.0800$ | $0.0400$ | $0.0000$ (if $s_{\text{curr}}=26$) |
| **Total Cost $J(s)$** | **$0.5602$** | **$0.3548$** | **$0.2297$** | **$0.1848$ (WINNER)** |

### Five Failure Mechanics in V1:
1. **Fanger PMV Neutral Point at $25.5^\circ\text{C}$:**
   For light summer office clothing ($\text{clo} = 0.5$) and seated office work ($\text{met} = 1.1$), thermal neutrality ($\text{PMV} = 0.0$) occurs at $\approx 25.5^\circ\text{C} - 25.8^\circ\text{C}$. Thus, $25.79^\circ\text{C}$ yielded a *lower* comfort penalty ($0.0085$) than $24.74^\circ\text{C}$ ($0.2465$).
2. **Inert Overheating Penalty:**
   The upper boundary was fixed at $26.5^\circ\text{C}$. Since $25.79^\circ\text{C} < 26.5^\circ\text{C}$, the overheating penalty was **identically zero**.
3. **Monotonic Energy Bias Toward Warmer Targets:**
   Because compressor power increases with cooling, warmer setpoints always minimize the energy term.
4. **Lack of Directional Thermal Error:**
   The optimizer evaluated setpoint quality solely on the predicted state $T_{\text{pred}}$, without considering the thermal error $T_{\text{current}} - T_{\text{setpoint}}$ or the rate of change $\frac{dT}{dt}$.
5. **No Directional Invariant:**
   There was no constraint enforcing that a zone above comfort limits or warming rapidly must receive a cooling recommendation ($s \le T_{\text{current}}$).

---

## 4. Architectural Resolution for V2

To eliminate this flaw, HVEAC Brain V2 requires:
1. An explicit **Directional Thermal Control Law**:
   $$\text{If } T_{\text{zone}} \ge T_{\text{comfort\_max}} \text{ or } \frac{dT}{dt} > 0 \text{ toward overheat} \implies s^* \le T_{\text{current}}$$
2. Short-horizon predictive overheating detection that accounts for disturbance heat flux $Q_{\text{dist}}$.
3. Multi-state thermal partitioning (States A through E) separating overheating defense from energy minimization.
4. Retention of energy optimization strictly in State B (comfortable and thermally stable).
