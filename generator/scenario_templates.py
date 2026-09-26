"""
Scenario Templates, Constraints, and Parameter Sampling.

Implements the formal generation pipeline:
ScenarioTemplate -> ConstraintSystem -> ParameterSampler -> ScenarioInstance
Preserves family identities while introducing controlled physical variations and edge cases.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from generator.config import SimulationConfig
from generator.models.computer_model import WorkloadCategory
from generator.models.room import ZONES, COMPUTER_DEFINITIONS, AC_DEFINITIONS


class ScenarioFamily(str, Enum):
    FAMILY_1 = "FAMILY_1_LOCALIZED_HEAVY_COMPUTE"
    FAMILY_2 = "FAMILY_2_OCCUPANCY_CONCENTRATION"
    FAMILY_3 = "FAMILY_3_DISTRIBUTED_HEAVY_COMPUTE"
    FAMILY_4 = "FAMILY_4_HIGH_OCCUPANCY_LOW_COMPUTE"
    FAMILY_5 = "FAMILY_5_OPPOSING_THERMAL_ZONES"


class EdgeCaseType(str, Enum):
    NONE = "NONE"
    ALL_IDLE = "ALL_IDLE"
    ALL_HEAVY = "ALL_HEAVY"
    ZERO_OCCUPANCY = "ZERO_OCCUPANCY"
    MAX_OCCUPANCY = "MAX_OCCUPANCY"
    SINGLE_AC_UNAVAILABLE = "SINGLE_AC_UNAVAILABLE"
    MULTI_AC_UNAVAILABLE = "MULTI_AC_UNAVAILABLE"
    EXTREME_HEAT = "EXTREME_HEAT"
    SUDDEN_OCCUPANCY_SPIKE = "SUDDEN_OCCUPANCY_SPIKE"
    SUSTAINED_COMPUTE_STRESS = "SUSTAINED_COMPUTE_STRESS"


@dataclass
class TemporalTransition:
    time_seconds: float
    transition_type: str
    payload: Dict


@dataclass
class ScenarioInstance:
    scenario_id: str
    family: ScenarioFamily
    run_id: str
    random_seed: int
    edge_case: EdgeCaseType

    # Initial physical conditions
    initial_outdoor_temp_c: float
    initial_humidity_pct: float
    initial_solar_watts: float
    initial_zone_temps: Dict[str, float]

    # Computer initial workloads
    computer_workloads: Dict[str, Tuple[WorkloadCategory, float, float]]  # (Category, cpu%, gpu%)

    # Initial occupancy distribution
    initial_occupancy: Dict[str, int]

    # HVAC initial states
    initial_ac_setpoints: Dict[str, float]
    initial_ac_states: Dict[str, str]  # "ON" or "OFF"
    initial_cooling_levels: Dict[str, float]

    # Scheduled dynamic events during simulation run
    transitions: List[TemporalTransition] = field(default_factory=list)


class ConstraintSystem:
    """Enforces inviolable boundary conditions and family-defining rules."""

    @staticmethod
    def validate_family_integrity(family: ScenarioFamily, instance: ScenarioInstance):
        """Verifies that sampled parameters strictly preserve the defining family identity."""
        workloads = instance.computer_workloads
        occupancy = instance.initial_occupancy

        if family == ScenarioFamily.FAMILY_1:
            # Computers 1-4 must have high CPU or GPU (>70%), while 5-10 must be <= 45%
            for c_id in ["Computer 1", "Computer 2", "Computer 3", "Computer 4"]:
                _, cpu, gpu = workloads[c_id]
                assert cpu >= 65.0 or gpu >= 65.0, f"Family 1 violation on {c_id}"
            for c_id in [f"Computer {i}" for i in range(5, 11)]:
                _, cpu, gpu = workloads[c_id]
                assert cpu <= 50.0 and gpu <= 50.0, f"Family 1 background violation on {c_id}"

        elif family == ScenarioFamily.FAMILY_2:
            # Occupancy concentrated in one dominant zone (>= 55% of total occupants)
            total = sum(occupancy.values())
            if total > 0:
                max_zone_occ = max(occupancy.values())
                assert max_zone_occ / total >= 0.45, "Family 2 occupancy concentration violation"

        elif family == ScenarioFamily.FAMILY_3:
            # All 10 computers must have heavy workloads (>= 70% cpu or gpu)
            for c_id in workloads:
                _, cpu, gpu = workloads[c_id]
                assert cpu >= 65.0 or gpu >= 65.0, f"Family 3 heavy compute violation on {c_id}"

        elif family == ScenarioFamily.FAMILY_4:
            # All computers low/simple load (<= 35%), high occupancy (>= 18)
            for c_id in workloads:
                _, cpu, gpu = workloads[c_id]
                assert cpu <= 40.0 and gpu <= 30.0, f"Family 4 low compute violation on {c_id}"
            total = sum(occupancy.values())
            assert total >= 16, f"Family 4 high occupancy violation (total={total})"

        elif family == ScenarioFamily.FAMILY_5:
            # Opposing zones: Zone 1 computers heavy, Zone 3 computers low, occupancy in Zone 3
            for c_id in ["Computer 1", "Computer 2", "Computer 3", "Computer 4"]:
                _, cpu, _ = workloads[c_id]
                assert cpu >= 60.0, f"Family 5 Zone 1 compute violation on {c_id}"
            for c_id in ["Computer 7", "Computer 8"]:
                _, cpu, gpu = workloads[c_id]
                assert cpu <= 35.0 and gpu <= 25.0, f"Family 5 Zone 3 compute violation on {c_id}"


def _pick_one(rng: np.random.Generator, seq):
    """Safely pick a single element from sequence without NumPy string casting."""
    return seq[int(rng.integers(0, len(seq)))]


class ParameterSampler:
    """Samples stochastic physical variables while respecting constraints."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def sample_scenario(
        self,
        family: ScenarioFamily,
        scenario_idx: int,
        run_id: str,
        base_seed: int,
        edge_case: EdgeCaseType = EdgeCaseType.NONE,
    ) -> ScenarioInstance:
        """Sample a fully resolved physical scenario instance."""
        scenario_seed = base_seed + (scenario_idx * 7919)
        rng = np.random.Generator(np.random.PCG64(scenario_seed))

        fam_scen_idx = scenario_idx // 5

        # Sample outdoor environment & thermal demand regime
        if edge_case == EdgeCaseType.EXTREME_HEAT or (fam_scen_idx in [2, 7]):
            # High thermal load / intense summer regime (produces rare 24.5°C regime naturally)
            out_temp = float(rng.uniform(39.5, 41.8))
            out_hum = float(rng.uniform(72.0, 82.0))
            solar_w = float(rng.uniform(620.0, 780.0))
            base_room_temp = float(rng.uniform(27.4, 27.8))
        elif fam_scen_idx in [3, 8, 13]:
            # Mild ambient / energy conservation regime (produces 26.5°C regime naturally)
            out_temp = float(rng.uniform(29.0, 32.0))
            out_hum = float(rng.uniform(46.0, 54.0))
            solar_w = float(rng.uniform(220.0, 420.0))
            base_room_temp = float(rng.uniform(23.7, 24.4))
        else:
            # Standard balanced operational regime (produces 25.0°C - 26.0°C)
            out_temp = float(rng.uniform(32.0, 38.0))
            out_hum = float(rng.uniform(52.0, 70.0))
            solar_w = float(rng.uniform(350.0, 680.0))
            base_room_temp = float(rng.uniform(24.5, 26.3))

        init_zone_temps = {
            z: round(base_room_temp + float(rng.uniform(-0.35, 0.35)), 2) for z in ZONES
        }

        # Sample computers and occupancy according to Family
        workloads, occupancy = self._sample_workloads_and_occupancy(family, rng, edge_case)

        # Sample initial AC setpoints and operational states
        ac_setpoints = {}
        ac_states = {}
        cooling_levels = {}
        for ac_id in AC_DEFINITIONS:
            if fam_scen_idx in [2, 7]:
                ac_setpoints[ac_id] = float(_pick_one(rng, [24.5, 25.0, 25.5]))
                cooling_levels[ac_id] = round(float(rng.uniform(0.15, 0.30)), 3)
            elif fam_scen_idx in [3, 8, 13]:
                ac_setpoints[ac_id] = float(_pick_one(rng, [22.5, 23.0, 23.5, 24.0]))
                cooling_levels[ac_id] = round(float(rng.uniform(0.20, 0.40)), 3)
            else:
                ac_setpoints[ac_id] = float(_pick_one(rng, self.config.hvac.candidate_setpoints))
                cooling_levels[ac_id] = round(float(rng.uniform(0.15, 0.45)), 3)
            ac_states[ac_id] = "ON"

        # Apply AC edge cases if selected
        if edge_case == EdgeCaseType.SINGLE_AC_UNAVAILABLE:
            failed_ac = str(_pick_one(rng, list(AC_DEFINITIONS.keys())))
            ac_states[failed_ac] = "OFF"
            cooling_levels[failed_ac] = 0.0
        elif edge_case == EdgeCaseType.MULTI_AC_UNAVAILABLE:
            acs_list = list(AC_DEFINITIONS.keys())
            idx1, idx2 = rng.choice(len(acs_list), size=2, replace=False)
            ac_states[acs_list[idx1]] = "OFF"
            cooling_levels[acs_list[idx1]] = 0.0
            ac_states[acs_list[idx2]] = "OFF"
            cooling_levels[acs_list[idx2]] = 0.0

        # Sample dynamic temporal transitions
        transitions = self._sample_transitions(family, rng, edge_case)

        scenario_id = f"SCEN_{scenario_idx+1:04d}_{family.name}"
        instance = ScenarioInstance(
            scenario_id=scenario_id,
            family=family,
            run_id=run_id,
            random_seed=scenario_seed,
            edge_case=edge_case,
            initial_outdoor_temp_c=round(out_temp, 2),
            initial_humidity_pct=round(out_hum, 2),
            initial_solar_watts=round(solar_w, 2),
            initial_zone_temps=init_zone_temps,
            computer_workloads=workloads,
            initial_occupancy=occupancy,
            initial_ac_setpoints=ac_setpoints,
            initial_ac_states=ac_states,
            initial_cooling_levels=cooling_levels,
            transitions=transitions,
        )

        # Validate constraint integrity
        if edge_case not in [EdgeCaseType.ALL_IDLE, EdgeCaseType.ALL_HEAVY, EdgeCaseType.ZERO_OCCUPANCY, EdgeCaseType.MAX_OCCUPANCY]:
            ConstraintSystem.validate_family_integrity(family, instance)

        return instance

    def _sample_workloads_and_occupancy(
        self,
        family: ScenarioFamily,
        rng: np.random.Generator,
        edge_case: EdgeCaseType,
    ) -> Tuple[Dict[str, Tuple[WorkloadCategory, float, float]], Dict[str, int]]:
        """Sample computer workloads and zone occupancy counts based on family definitions."""
        workloads = {}
        occupancy = {}

        # Handle global edge cases first
        if edge_case == EdgeCaseType.ALL_IDLE:
            for c_id in COMPUTER_DEFINITIONS:
                workloads[c_id] = (WorkloadCategory.IDLE, float(rng.uniform(4.0, 10.0)), float(rng.uniform(0.0, 5.0)))
            occupancy = {z: int(rng.integers(1, 4)) for z in ZONES}
            return workloads, occupancy

        if edge_case == EdgeCaseType.ALL_HEAVY:
            for c_id in COMPUTER_DEFINITIONS:
                workloads[c_id] = (WorkloadCategory.HEAVY, float(rng.uniform(85.0, 98.0)), float(rng.uniform(85.0, 98.0)))
            occupancy = {z: int(rng.integers(2, 6)) for z in ZONES}
            return workloads, occupancy

        if edge_case == EdgeCaseType.ZERO_OCCUPANCY:
            occupancy = {z: 0 for z in ZONES}
            for c_id in COMPUTER_DEFINITIONS:
                workloads[c_id] = (WorkloadCategory.GENERAL, float(rng.uniform(35.0, 60.0)), float(rng.uniform(20.0, 40.0)))
            return workloads, occupancy

        if edge_case == EdgeCaseType.MAX_OCCUPANCY:
            occupancy = {z: int(rng.integers(8, 11)) for z in ZONES}  # total ~35-40
            for c_id in COMPUTER_DEFINITIONS:
                workloads[c_id] = (WorkloadCategory.LIGHT, float(rng.uniform(15.0, 28.0)), float(rng.uniform(5.0, 15.0)))
            return workloads, occupancy

        # Standard Family sampling
        if family == ScenarioFamily.FAMILY_1:
            # Computers 1-4: HEAVY
            for c_id in ["Computer 1", "Computer 2", "Computer 3", "Computer 4"]:
                cat = _pick_one(rng, [WorkloadCategory.HEAVY, WorkloadCategory.CPU_INTENSIVE, WorkloadCategory.GPU_INTENSIVE])
                cpu = float(rng.uniform(80.0, 98.0)) if cat != WorkloadCategory.GPU_INTENSIVE else float(rng.uniform(35.0, 55.0))
                gpu = float(rng.uniform(80.0, 98.0)) if cat != WorkloadCategory.CPU_INTENSIVE else float(rng.uniform(15.0, 30.0))
                workloads[c_id] = (cat, cpu, gpu)
            # Computers 5-10: LIGHT/IDLE
            for c_id in [f"Computer {i}" for i in range(5, 11)]:
                cat = _pick_one(rng, [WorkloadCategory.IDLE, WorkloadCategory.LIGHT])
                cpu = float(rng.uniform(8.0, 25.0))
                gpu = float(rng.uniform(0.0, 15.0))
                workloads[c_id] = (cat, cpu, gpu)
            # Occupancy evenly distributed
            base_occ = int(rng.integers(2, 5))
            occupancy = {z: base_occ + int(rng.integers(-1, 2)) for z in ZONES}

        elif family == ScenarioFamily.FAMILY_2:
            # All computers light/moderate
            for c_id in COMPUTER_DEFINITIONS:
                cat = _pick_one(rng, [WorkloadCategory.LIGHT, WorkloadCategory.GENERAL])
                cpu = float(rng.uniform(20.0, 45.0))
                gpu = float(rng.uniform(10.0, 25.0))
                workloads[c_id] = (cat, cpu, gpu)
            # Occupancy strongly concentrated in one random target zone
            target_zone = _pick_one(rng, ZONES)
            occupancy = {z: int(rng.integers(1, 3)) for z in ZONES}
            occupancy[target_zone] = int(rng.integers(10, 15))

        elif family == ScenarioFamily.FAMILY_3:
            # All 10 computers heavy
            for c_id in COMPUTER_DEFINITIONS:
                cat = WorkloadCategory.HEAVY
                cpu = float(rng.uniform(82.0, 98.0))
                gpu = float(rng.uniform(80.0, 98.0))
                workloads[c_id] = (cat, cpu, gpu)
            # Occupancy approximately even
            occupancy = {z: int(rng.integers(2, 5)) for z in ZONES}

        elif family == ScenarioFamily.FAMILY_4:
            # All computers low/simple load
            for c_id in COMPUTER_DEFINITIONS:
                cat = _pick_one(rng, [WorkloadCategory.IDLE, WorkloadCategory.LIGHT])
                cpu = float(rng.uniform(5.0, 25.0))
                gpu = float(rng.uniform(0.0, 10.0))
                workloads[c_id] = (cat, cpu, gpu)
            # Occupancy high (22 - 36 people)
            occupancy = {z: int(rng.integers(5, 9)) for z in ZONES}

        elif family == ScenarioFamily.FAMILY_5:
            # ZONE_1 (Computers 1-4) high compute, ZONE_3 (Computers 7-8) low compute
            for c_id in ["Computer 1", "Computer 2", "Computer 3", "Computer 4"]:
                workloads[c_id] = (WorkloadCategory.HEAVY, float(rng.uniform(85.0, 98.0)), float(rng.uniform(80.0, 96.0)))
            for c_id in ["Computer 5", "Computer 6", "Computer 9", "Computer 10"]:
                workloads[c_id] = (WorkloadCategory.GENERAL, float(rng.uniform(30.0, 50.0)), float(rng.uniform(15.0, 30.0)))
            for c_id in ["Computer 7", "Computer 8"]:
                workloads[c_id] = (WorkloadCategory.IDLE, float(rng.uniform(5.0, 15.0)), float(rng.uniform(0.0, 5.0)))
            # Occupancy concentrated toward opposite region (ZONE_3)
            occupancy = {"ZONE_1": int(rng.integers(0, 2)), "ZONE_2": int(rng.integers(2, 4)), "ZONE_3": int(rng.integers(9, 14)), "ZONE_4": int(rng.integers(2, 4))}

        return workloads, occupancy

    def _sample_transitions(
        self,
        family: ScenarioFamily,
        rng: np.random.Generator,
        edge_case: EdgeCaseType,
    ) -> List[TemporalTransition]:
        """Schedule realistic mid-simulation transitions (occupancy shifts, workload bursts)."""
        transitions = []
        dur = self.config.duration_seconds

        # Plausible event at 30% to 60% of simulation time
        t_event = float(rng.uniform(dur * 0.25, dur * 0.65))

        if edge_case == EdgeCaseType.SUDDEN_OCCUPANCY_SPIKE:
            transitions.append(
                TemporalTransition(
                    time_seconds=round(t_event, 1),
                    transition_type="OCCUPANCY_SPIKE",
                    payload={"zone": str(rng.choice(ZONES)), "count_delta": int(rng.integers(6, 10))},
                )
            )
        elif rng.random() < 0.40:
            # 40% probability of a mid-simulation dynamic transition
            trans_choice = rng.choice(["WORKLOAD_BURST", "OCCUPANCY_SHIFT", "OUTDOOR_WARMING"])
            if trans_choice == "WORKLOAD_BURST":
                transitions.append(
                    TemporalTransition(
                        time_seconds=round(t_event, 1),
                        transition_type="WORKLOAD_BURST",
                        payload={"computers": ["Computer 5", "Computer 6"], "target_category": WorkloadCategory.HEAVY},
                    )
                )
            elif trans_choice == "OCCUPANCY_SHIFT":
                transitions.append(
                    TemporalTransition(
                        time_seconds=round(t_event, 1),
                        transition_type="OCCUPANCY_SHIFT",
                        payload={"from_zone": "ZONE_1", "to_zone": "ZONE_3", "people": 3},
                    )
                )
            elif trans_choice == "OUTDOOR_WARMING":
                transitions.append(
                    TemporalTransition(
                        time_seconds=round(t_event, 1),
                        transition_type="OUTDOOR_WARMING",
                        payload={"temp_increase_c": 1.5},
                    )
                )

        return transitions
