# HVEAC V3 — Thermal Simulation Data Engine

The **HVEAC Thermal Simulation Data Engine** is a deterministic, physics-grounded synthetic data generation engine. It simulates the thermodynamic interactions between building envelopes, 4 thermal zones, 10 fixed computers, human occupants, and 4 perimeter air conditioning units to produce machine-readable datasets for training the **HVEAC Thermal Intelligence** machine learning model.

> [!IMPORTANT]
> **SYNTHETIC DATA DISCLAIMER**:
> All computational workloads, occupancy headcounts, ambient meteorological profiles, thermal heat fluxes, and HVAC control recommendations produced by this engine are **SYNTHETICALLY GENERATED**. They are designed for machine learning model development and offline benchmark calibration. No physical sensors or live hardware devices are queried during simulation.

---

## 1. Physical Room Model & 2D Spatial Layout

The simulation models an enclosed rectangular commercial space ($12.0\text{ m} \times 10.0\text{ m} \times 3.0\text{ m}$ ceiling height) divided into four open-plan thermal zones:

```text
       (0, 10)                        (6, 10)                       (12, 10)
          +------------------------------[AC-1]----------------------------+
          |                               |                                |
          |   [C1] (2,8)      [C2] (4,8)  |             [C5] (8,8)         |
          |          ZONE_1 (NW)          |            ZONE_2 (NE)         |
          |   [C3] (2,6.5)    [C4] (4,6.5)|             [C6] (10,6.5)      |
          |                               |                                |
   (0, 5) [AC-4]                          |                              [AC-2] (12, 5)
          |                               |                                |
          |   [C10] (2,3.5)               |             [C7] (10,3.5)      |
          |          ZONE_4 (SW)          |            ZONE_3 (SE)         |
          |   [C9] (4,2)                  |   [C8] (8,2)                   |
          |                               |                                |
          +------------------------------[AC-3]----------------------------+
       (0, 0)                         (6, 0)                        (12, 0)
```

### Fixed Spatial Inventory

1. **Four Thermal Zones ($6.0\text{ m} \times 5.0\text{ m} \times 3.0\text{ m}$ each, Area: $30\text{ m}^2$, Volume: $90\text{ m}^3$)**:
   - `ZONE_1` (North-West): $X \in [0, 6)$, $Y \in [5, 10]$
   - `ZONE_2` (North-East): $X \in [6, 12]$, $Y \in [5, 10]$
   - `ZONE_3` (South-East): $X \in [6, 12]$, $Y \in [0, 5)$
   - `ZONE_4` (South-West): $X \in [0, 6)$, $Y \in [0, 5)$

2. **Ten Fixed Computers**:
   - **Computers 1–4**: Fixed physical cluster in `ZONE_1` (NW):
     - `Computer 1`: $(x=2.0, y=8.0)$
     - `Computer 2`: $(x=4.0, y=8.0)$
     - `Computer 3`: $(x=2.0, y=6.5)$
     - `Computer 4`: $(x=4.0, y=6.5)$
   - **Computers 5–6**: Fixed positions in `ZONE_2` (NE):
     - `Computer 5`: $(x=8.0, y=8.0)$
     - `Computer 6`: $(x=10.0, y=6.5)$
   - **Computers 7–8**: Fixed positions in `ZONE_3` (SE):
     - `Computer 7`: $(x=10.0, y=3.5)$
     - `Computer 8`: $(x=8.0, y=2.0)$
   - **Computers 9–10**: Fixed positions in `ZONE_4` (SW):
     - `Computer 9`: $(x=4.0, y=2.0)$
     - `Computer 10`: $(x=2.0, y=3.5)$
   *Hardware locations remain strictly identical across all scenarios.*

3. **Four Perimeter Air Conditioners**:
   - `AC-1`: North Wall mounted at $(x=6.0, y=10.0)$
   - `AC-2`: East Wall mounted at $(x=12.0, y=5.0)$
   - `AC-3`: South Wall mounted at $(x=6.0, y=0.0)$
   - `AC-4`: West Wall mounted at $(x=0.0, y=5.0)$

---

## 2. Thermal Dynamic Equations & Physical Assumptions

The simulation uses a lumped-capacitance thermodynamic model with explicit inter-zone energy conservation.

### Zone Energy Balance

For each zone $z \in \{1, 2, 3, 4\}$ at timestep $t$:

$$Q_{net, z} = Q_{comp, z} + Q_{occ, z} + Q_{env, z} + Q_{interzone, z} - Q_{hvac, z}$$

$$\frac{dT_z}{dt} = \frac{Q_{net, z}}{C_{zone}}$$

$$T_z(t + \Delta t) = T_z(t) + \frac{Q_{net, z} \cdot \Delta t}{C_{zone}}$$

Where:
- $C_{zone} = V_{zone} \cdot \rho_{air} \cdot c_{p, air} \cdot \gamma_{thermal\_mass} \approx 90\text{ m}^3 \times 1.204\text{ kg/m}^3 \times 1005\text{ J/(kg}\cdot\text{K)} \times 4.5 \approx 490,000\text{ J/K}$.
- $\Delta t = 10.0\text{ seconds}$ (default timestep).

### Computational Heat Model
$$Q_{comp}(c) = P_{idle} + \alpha_{cpu} \left(\frac{\text{CPU}\%}{100}\right) + \alpha_{gpu} \left(\frac{\text{GPU}\%}{100}\right)$$
- $P_{idle} = 45.0\text{ W}$ (motherboard, fans, SSD idle draw)
- $\alpha_{cpu} = 125.0\text{ W}$ (rated maximum processor dissipation)
- $\alpha_{gpu} = 150.0\text{ W}$ (rated maximum graphics dissipation)
- Maximum dissipation per computer at 100% load: $320.0\text{ W}$.

### Occupancy Sensible Heat Model
$$Q_{occ, z} = N_{occ, z} \cdot q_{sensible}$$
- $q_{sensible} = 85.0\text{ W}$ (ASHRAE standard metabolic sensible heat release for office sedentary activity).

### Building Envelope Heat Transfer
$$Q_{env, z} = (U \cdot A)_{ext} \cdot (T_{outdoor} - T_z) + Q_{solar, z}$$
- Exterior wall transmission coefficient: $(U \cdot A)_{ext} = 21.5\text{ W/K}$ per zone.
- Solar irradiance gain distributed across exterior facade windows.

### Inter-Zone Heat Transfer (Energy Conserving)
$$Q_{interzone, z} = \sum_{j \neq z} K_{z, j} (T_j - T_z)$$
- Adjacent zone convective coupling ($K_{adj} = 110.0\text{ W/K}$) between:
  - $(1 \leftrightarrow 2)$, $(2 \leftrightarrow 3)$, $(3 \leftrightarrow 4)$, $(4 \leftrightarrow 1)$
- Cross-room diagonal diffusion ($K_{diag} = 15.0\text{ W/K}$) between:
  - $(1 \leftrightarrow 3)$, $(2 \leftrightarrow 4)$
- **Conservation Guarantee**: $\sum_{z=1}^4 Q_{interzone, z} \equiv 0.0\text{ W}$ at all timesteps.

### HVAC Cooling Spatial Distribution
Each AC unit delivers cooling modulated by cooling level $u_k \in [0.0, 1.0]$:
$$Q_{ac, k} = u_k \cdot Q_{nominal} \quad (Q_{nominal} = 3500.0\text{ W})$$
Cooling is spatially distributed via the influence matrix $\mathbf{W} \in \mathbb{R}^{4 \times 4}$:
$$Q_{hvac, z} = \sum_{k=1}^4 W_{k, z} \cdot Q_{ac, k}$$

| AC Unit | Mounted Wall | Zone 1 Weight | Zone 2 Weight | Zone 3 Weight | Zone 4 Weight | Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AC-1** | North | 0.45 | 0.45 | 0.05 | 0.05 | 1.00 |
| **AC-2** | East | 0.05 | 0.45 | 0.45 | 0.05 | 1.00 |
| **AC-3** | South | 0.05 | 0.05 | 0.45 | 0.45 | 1.00 |
| **AC-4** | West | 0.45 | 0.05 | 0.05 | 0.45 | 1.00 |

---

## 3. Five Simulation Scenarios

### Scenario 1: Localized Heavy Compute
- **Dynamics**: Computers 1–4 (cluster in Zone 1) ramp to sustained 80–95% CPU/GPU workload. Computers 5–10 remain in background/idle state (10–25% CPU). Occupancy is evenly distributed (3–4 per zone).
- **Result**: Forms a sharp computational thermal hotspot centered in `ZONE_1`.

### Scenario 2: Occupancy Concentration
- **Dynamics**: All 10 computers maintain uniform light workloads (18–28% CPU). Occupancy surges to 16 people in `ZONE_3` (team presentation/meeting) while other zones have 1–2 people.
- **Result**: Forms an occupancy-driven thermal hotspot in `ZONE_3`.

### Scenario 3: Distributed Heavy Compute
- **Dynamics**: All 10 computers across all 4 zones perform intense batch compute (80–95% CPU/GPU). Occupancy is evenly distributed.
- **Result**: Thermal load rises broadly across the entire room, requiring uniform multi-unit cooling.

### Scenario 4: High Occupancy / Low Computer Load
- **Dynamics**: All 10 computers operate in idle/low-power states (8–14% CPU). Room occupancy fills to capacity (30–36 people distributed across all zones).
- **Result**: Metabolic sensible heat (~2900 W) completely dominates over computational heat (~550 W) by > 5:1.

### Scenario 5: Opposing Thermal Zones
- **Dynamics**: West side (Computers 1, 2, 3, 4 in Zone 1 + Computers 9, 10 in Zone 4) runs heavy compute (80–95%). East side (Zone 2 & Zone 3) hosts high occupancy (20+ occupants) with low compute.
- **Result**: Creates two distinct opposing thermal zones powered by fundamentally different heat sources (compute in West vs people in East).

---

## 4. Multi-Objective Target / Label Optimization

For every timestep, the simulator solves a constrained optimization problem to compute the ground-truth optimal HVAC action $\mathbf{u}^* = [u_1^*, u_2^*, u_3^*, u_4^*] \in [0, 1]^4$:

$$\min_{\mathbf{u} \in [0, 1]^4} J(\mathbf{u}) = w_{comfort} \sum_{z=1}^4 (T_{z}'(\mathbf{u}) - 22.5)^2 + w_{penalty} \sum_{z=1}^4 \left( \max(0, T_z' - 25.0)^2 + \max(0, 20.0 - T_z')^2 \right) + w_{grad} (\max_z T_z' - \min_z T_z')^2 + w_{energy} \sum_{k=1}^4 u_k^2$$

- **Target Outputs**:
  - `optimal_cooling_ac1..4`: Continuous cooling levels $[0.0, 1.0]$.
  - `optimal_temperature_c`: Optimal target setpoint ($22.5^\circ\text{C}$).
  - `optimal_hvac_action`: Categorical engineering action (`ECO_MAINTAIN`, `MODERATE_UNIFORM_COOL`, `MAX_UNIFORM_COOL`, `TARGETED_WEST_COMPUTE_COOL`, `TARGETED_EAST_OCCUPANCY_COOL`, `OPPOSING_ZONE_COMPENSATE`).

---

## 5. CLI Usage & Reproducibility

### Generate Complete Dataset (All 5 Scenarios)
```powershell
python -m simulator.cli --all
```

### Run a Single Scenario
```powershell
python -m simulator.cli --scenario 1
```

### Generate Multiple Runs with Controlled Stochastic Variations
```powershell
python -m simulator.cli --all --runs 3 --duration 7200 --seed 42
```

### Supported CLI Flags
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--scenario N` | N/A | Execute scenario 1, 2, 3, 4, or 5 |
| `--all` | N/A | Execute all 5 scenarios and build combined dataset |
| `--runs N` | `1` | Number of parameter variation runs per scenario |
| `--duration SEC` | `7200.0` | Duration per run in seconds (7200s = 2 hours) |
| `--timestep SEC` | `10.0` | Simulation timestep in seconds (720 steps per 2h run) |
| `--seed INT` | `42` | Master random seed for exact reproducibility |
| `--output-dir PATH`| `simulation_dataset` | Output directory for CSV, JSONL, and metadata |
| `--format` | `csv,jsonl` | Formats to export (`csv`, `jsonl`, or `csv,jsonl`) |

---

## 6. Output Dataset Schema

Files generated in `simulation_dataset/`:
- `scenario_1.csv` & `scenario_1.jsonl`
- `scenario_2.csv` & `scenario_2.jsonl`
- `scenario_3.csv` & `scenario_3.jsonl`
- `scenario_4.csv` & `scenario_4.jsonl`
- `scenario_5.csv` & `scenario_5.jsonl`
- `combined_dataset.csv` & `combined_dataset.jsonl`
- `dataset_metadata.json`

### Field Definitions (Flat CSV)
- **Metadata**: `scenario_id`, `scenario_name`, `run_id`, `timestep_index`, `timestamp_seconds`.
- **Occupancy Features**: `total_occupancy`, `occupancy_zone_1..4`.
- **Computer Features** (for $i=1..10$): `cpu_util_comp_i`, `gpu_util_comp_i`, `workload_cat_comp_i`, `heat_w_comp_i`.
- **Environment Features**: `outdoor_temperature_c`, `relative_humidity_percent`, `solar_irradiance_w_m2`.
- **Operational HVAC**: `cooling_level_ac1..4`, `setpoint_ac1..4`, `state_ac1..4`.
- **Zone Thermal State**: `zone_1..4_temperature_c`, `room_average_temperature_c`, `max_zone_temperature_c`, `min_zone_temperature_c`, `temperature_gradient_c`.
- **Zone Heat Fluxes (W)**: `comp_heat_zone_1..4_w`, `occ_heat_zone_1..4_w`, `env_heat_zone_1..4_w`, `interzone_heat_zone_1..4_w`, `hvac_cooling_zone_1..4_w`, `net_heat_zone_1..4_w`.
- **Cluster Heat Totals (W)**: `total_computational_heat_w`, `total_occupancy_heat_w`, `total_envelope_heat_w`, `total_hvac_cooling_w`, `total_net_heat_w`.
- **ML Target Labels**: `optimal_cooling_ac1..4`, `optimal_temperature_c`, `optimal_hvac_action`, `target_objective_cost`.

---

## 7. Known Modeling Boundaries & Limitations

1. **Lumped-Capacitance vs CFD**: The engine does not solve 3D Navier-Stokes computational fluid dynamics. It uses 4 well-mixed control volume thermal zones with lumped thermal capacitances.
2. **Simplified Air Infiltration**: Air exchange is modeled via interzone convective coupling and envelope conductance, omitting active pressure differentials or open doorway wind buffeting.
3. **Synthetic Variables**: All telemetry generated by this module is synthetic and intended exclusively for model training and simulation benchmarks.

---

## 8. HVEAC Control Center UI Integration (Simulation Lab)

The simulator outputs and real-time engine are directly integrated into the HVEAC Control Center via:
- **Route**: `#/simulation` in the single-page application dashboard.
- **REST Endpoints**: `/api/simulation/scenarios`, `/api/simulation/state`, `/api/simulation/select`, `/api/simulation/start`, `/api/simulation/pause`, `/api/simulation/reset`, `/api/simulation/speed`, `/api/simulation/comparison`.
- **WebSocket**: `/ws/simulation` streaming full synthetic state updates at ~10 Hz.
- **Interactive Visuals**: Bilinear gradient Canvas heatmap with localized compute and occupancy auras, perimeter AC indicators with dynamic airflow vectors, 10 workstation hover cards, 5-minute multi-line history chart, and pitch presentation mode.
- **Individual Simulated Node Telemetry**: Dedicated engineering table displaying live telemetry for Computers 1–10 (CPU%, GPU%, Workload Category, Heat Dissipation Watts, Thermal Contribution, Status, and Trends) synchronized with room heatmap markers.
- **Gradual Workload Trajectories**: Deterministic cubic hermite ramps and continuous thermal oscillations starting at $t=0$, guaranteeing smooth realistic telemetry progression without random jitter.


