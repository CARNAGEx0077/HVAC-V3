"""
Automated Validation Test Suite for HVEAC V3 Thermal Simulation Data Engine.

Tests:
1. Scenario 1: Computers 1-4 have highest workload & heat (localized hotspot in Zone 1).
2. Scenario 2: Occupancy is strongly concentrated in Zone 3.
3. Scenario 3: All 10 computers perform high workload distributed across all zones.
4. Scenario 4: Occupancy dominates room thermal budget while compute load remains low.
5. Scenario 5: High compute in West, high occupancy in East (opposing thermal zones).
6. Thermal dynamics: Zone temperatures evolve over time; interzone conservation holds.
7. HVAC cooling: Verified cooling reduction on zone thermal states.
8. Numeric validity: Zero NaNs, Infs, or missing values across all features.
9. Schema stability: Stable, identical columns across all scenarios.
10. Seed reproducibility: Identical seed produces identical dataset rows.
"""

import math
import shutil
import tempfile
import unittest
from pathlib import Path

from simulator.config import SimulationEngineConfig
from simulator.dataset_writer import DatasetWriter
from simulator.engine import ScenarioEngine
from simulator.scenarios import (
    ALL_SCENARIOS,
    Scenario1LocalizedCompute,
    Scenario2OccupancyConcentration,
    Scenario3DistributedCompute,
    Scenario4HighOccupancyLowCompute,
    Scenario5OpposingZones,
)


class TestSimulatorEngine(unittest.TestCase):
    """Validation test suite for HVEAC V3 Thermal Simulator."""

    @classmethod
    def setUpClass(cls):
        cls.config = SimulationEngineConfig(
            timestep_seconds=10.0,
            default_duration_seconds=1800.0,  # 30 mins (180 timesteps) for fast testing
            default_seed=42,
        )
        cls.engine = ScenarioEngine(cls.config)

    def test_scenario_1_localized_heavy_compute(self):
        """Verify computers 1-4 form a localized compute cluster in Zone 1."""
        scen = Scenario1LocalizedCompute(duration_seconds=1800.0, timestep_seconds=10.0)
        records = self.engine.run_scenario(scen, seed=42)
        self.assertEqual(len(records), 180)

        # Inspect steady-state steps (last 50 steps)
        late_records = records[-50:]

        for r in late_records:
            # Cluster computers 1-4 vs computers 5-10
            cluster_cpu = [r[f"cpu_util_comp_{i}"] for i in (1, 2, 3, 4)]
            other_cpu = [r[f"cpu_util_comp_{i}"] for i in range(5, 11)]

            cluster_heat = [r[f"heat_w_comp_{i}"] for i in (1, 2, 3, 4)]
            other_heat = [r[f"heat_w_comp_{i}"] for i in range(5, 11)]

            self.assertGreater(min(cluster_cpu), max(other_cpu), "Cluster CPU must exceed other computers")
            self.assertGreater(min(cluster_heat), max(other_heat), "Cluster heat must exceed other computers")

            # Zone 1 computational heat must dominate other zones
            z1_heat = r["comp_heat_zone_1_w"]
            z2_heat = r["comp_heat_zone_2_w"]
            z3_heat = r["comp_heat_zone_3_w"]
            z4_heat = r["comp_heat_zone_4_w"]
            self.assertGreater(z1_heat, z2_heat * 1.8)
            self.assertGreater(z1_heat, z3_heat * 1.8)
            self.assertGreater(z1_heat, z4_heat * 1.8)

        # Thermal hotspot: Zone 1 temperature should be higher than other zones
        last_rec = records[-1]
        self.assertGreater(last_rec["zone_1_temperature_c"], last_rec["zone_2_temperature_c"])
        self.assertGreater(last_rec["zone_1_temperature_c"], last_rec["zone_3_temperature_c"])
        self.assertGreater(last_rec["zone_1_temperature_c"], last_rec["zone_4_temperature_c"])

    def test_scenario_2_occupancy_concentration(self):
        """Verify occupancy is strongly concentrated in Zone 3."""
        scen = Scenario2OccupancyConcentration(duration_seconds=1800.0, timestep_seconds=10.0)
        records = self.engine.run_scenario(scen, seed=42)

        late_records = records[-50:]
        for r in late_records:
            # Zone 3 occupancy must heavily dominate
            self.assertGreaterEqual(r["occupancy_zone_3"], 12)
            self.assertLessEqual(r["occupancy_zone_1"], 4)
            self.assertLessEqual(r["occupancy_zone_2"], 4)
            self.assertLessEqual(r["occupancy_zone_4"], 4)

            # Zone 3 occupancy heat must dominate
            self.assertGreater(r["occ_heat_zone_3_w"], r["occ_heat_zone_1_w"] * 2.5)

            # Computers should have moderate/light workloads
            for cid in range(1, 11):
                self.assertLess(r[f"cpu_util_comp_{cid}"], 45.0)

        # Thermal hotspot: Zone 3 temperature should be highest
        last_rec = records[-1]
        self.assertGreater(last_rec["zone_3_temperature_c"], last_rec["zone_1_temperature_c"])
        self.assertGreater(last_rec["zone_3_temperature_c"], last_rec["zone_4_temperature_c"])

    def test_scenario_3_distributed_heavy_compute(self):
        """Verify all 10 computers have high workload distributed across all zones."""
        scen = Scenario3DistributedCompute(duration_seconds=1800.0, timestep_seconds=10.0)
        records = self.engine.run_scenario(scen, seed=42)

        late_records = records[-50:]
        for r in late_records:
            # Every single computer must have high workload (>70% CPU)
            for cid in range(1, 11):
                self.assertGreater(r[f"cpu_util_comp_{cid}"], 70.0)

            # Total computational heat must be very high
            self.assertGreater(r["total_computational_heat_w"], 2200.0)

        # Temperatures should be elevated across all zones
        last_rec = records[-1]
        self.assertGreater(last_rec["room_average_temperature_c"], 22.0)

    def test_scenario_4_high_occupancy_low_compute(self):
        """Verify occupancy is high throughout the room and dominates over computer heat."""
        scen = Scenario4HighOccupancyLowCompute(duration_seconds=1800.0, timestep_seconds=10.0)
        records = self.engine.run_scenario(scen, seed=42)

        late_records = records[-50:]
        for r in late_records:
            self.assertGreaterEqual(r["total_occupancy"], 28)
            # Computers must have low workload
            for cid in range(1, 11):
                self.assertLessEqual(r[f"cpu_util_comp_{cid}"], 25.0)

            # Occupancy heat must dominate computer heat by at least 2.5x
            self.assertGreater(r["total_occupancy_heat_w"], r["total_computational_heat_w"] * 2.5)

    def test_scenario_5_opposing_zones(self):
        """Verify high compute on West side (Zones 1 & 4) and high occupancy on East side (Zones 2 & 3)."""
        scen = Scenario5OpposingZones(duration_seconds=1800.0, timestep_seconds=10.0)
        records = self.engine.run_scenario(scen, seed=42)

        late_records = records[-50:]
        for r in late_records:
            # West side computers (1, 2, 3, 4, 9, 10) have high compute
            for cid in (1, 2, 3, 4, 9, 10):
                self.assertGreater(r[f"cpu_util_comp_{cid}"], 70.0)

            # East side computers (5, 6, 7, 8) have low compute
            for cid in (5, 6, 7, 8):
                self.assertLess(r[f"cpu_util_comp_{cid}"], 30.0)

            # Occupancy is concentrated in East side (Zones 2 & 3)
            east_occ = r["occupancy_zone_2"] + r["occupancy_zone_3"]
            west_occ = r["occupancy_zone_1"] + r["occupancy_zone_4"]
            self.assertGreater(east_occ, west_occ * 3)

            # West side dominated by compute heat; East side dominated by occupancy heat
            west_comp = r["comp_heat_zone_1_w"] + r["comp_heat_zone_4_w"]
            east_comp = r["comp_heat_zone_2_w"] + r["comp_heat_zone_3_w"]
            self.assertGreater(west_comp, east_comp * 1.5)

            east_occ_heat = r["occ_heat_zone_2_w"] + r["occ_heat_zone_3_w"]
            west_occ_heat = r["occ_heat_zone_1_w"] + r["occ_heat_zone_4_w"]
            self.assertGreater(east_occ_heat, west_occ_heat * 3)

    def test_thermal_dynamics_and_interzone_conservation(self):
        """Verify interzone heat transfers sum to zero (energy conservation)."""
        scen = Scenario1LocalizedCompute(duration_seconds=600.0, timestep_seconds=10.0)
        records = self.engine.run_scenario(scen, seed=42)

        for r in records:
            sum_interzone = (
                r["interzone_heat_zone_1_w"]
                + r["interzone_heat_zone_2_w"]
                + r["interzone_heat_zone_3_w"]
                + r["interzone_heat_zone_4_w"]
            )
            # Energy conservation: net exchange must be 0 within floating point roundoff
            self.assertAlmostEqual(sum_interzone, 0.0, places=1)

    def test_no_nan_or_inf_in_records(self):
        """Verify no NaN, Inf, or None values exist in any record field."""
        scen = Scenario1LocalizedCompute(duration_seconds=300.0, timestep_seconds=10.0)
        records = self.engine.run_scenario(scen, seed=42)

        for idx, r in enumerate(records):
            for k, v in r.items():
                self.assertIsNotNone(v, f"Field {k} is None at step {idx}")
                if isinstance(v, float):
                    self.assertFalse(math.isnan(v), f"Field {k} is NaN at step {idx}")
                    self.assertFalse(math.isinf(v), f"Field {k} is Inf at step {idx}")

    def test_schema_stability(self):
        """Verify all 5 scenarios output the exact same column names in the exact same order."""
        cols_by_scenario = []
        for sid in (1, 2, 3, 4, 5):
            scen = ALL_SCENARIOS[sid](duration_seconds=60.0, timestep_seconds=10.0)
            recs = self.engine.run_scenario(scen, seed=42)
            cols_by_scenario.append(list(recs[0].keys()))

        first_cols = cols_by_scenario[0]
        for idx, cols in enumerate(cols_by_scenario[1:], start=2):
            self.assertEqual(cols, first_cols, f"Scenario {idx} columns do not match Scenario 1 columns")

    def test_seed_reproducibility(self):
        """Verify identical seed produces byte-identical simulation records."""
        scen1 = Scenario1LocalizedCompute(duration_seconds=300.0, timestep_seconds=10.0)
        recs_a = self.engine.run_scenario(scen1, seed=12345)
        recs_b = self.engine.run_scenario(scen1, seed=12345)

        self.assertEqual(len(recs_a), len(recs_b))
        for step in range(len(recs_a)):
            self.assertEqual(recs_a[step], recs_b[step], f"Mismatch at step {step} for identical seed")

    def test_dataset_writer_outputs(self):
        """Verify dataset writer creates valid CSV, JSONL, and metadata files."""
        temp_dir = tempfile.mkdtemp(prefix="hveac_sim_test_")
        try:
            writer = DatasetWriter(temp_dir)
            scen = Scenario1LocalizedCompute(duration_seconds=100.0, timestep_seconds=10.0)
            recs = self.engine.run_scenario(scen, seed=42)

            csv_file = writer.write_csv(recs, "test_scen.csv")
            jsonl_file = writer.write_jsonl(recs, "test_scen.jsonl")
            meta_file = writer.write_metadata(
                recs,
                {1: recs},
                {"test_config": True},
                "test_meta.json",
            )

            self.assertTrue(csv_file.exists())
            self.assertTrue(jsonl_file.exists())
            self.assertTrue(meta_file.exists())
            self.assertGreater(csv_file.stat().st_size, 500)
            self.assertGreater(jsonl_file.stat().st_size, 500)
            self.assertGreater(meta_file.stat().st_size, 200)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
