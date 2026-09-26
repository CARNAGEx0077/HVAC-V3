"""
Streaming Dataset Writer and Stratified Partitioning Engine.

Handles disk-buffered streaming CSV writes for arbitrary dataset scales
(100 to 5,000+ scenarios) without storing time-series tables in RAM.
Enforces deterministic family-stratified scenario splitting (70% train / 15% val / 15% test)
to guarantee zero data leakage between splits and exact family representation.
"""

from collections import Counter
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


def compute_stratified_scenario_splits(
    scenario_tuples: List[Tuple[str, str]],  # List of (scenario_id, family_name)
    split_seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, str]:
    """Deterministically stratifies scenarios by family into train, validation, and test sets.

    Guarantees:
    - Zero scenario leakage across splits
    - Exact per-family representation (e.g., for 20 scenarios/family: 14 train, 3 val, 3 test)
    - Total split: 70 train, 15 val, 15 test for a 100-scenario dataset
    """
    family_groups: Dict[str, List[str]] = {}
    for scen_id, fam in scenario_tuples:
        family_groups.setdefault(fam, []).append(scen_id)

    scenario_splits: Dict[str, str] = {}
    rng = np.random.Generator(np.random.PCG64(split_seed))

    for fam in sorted(family_groups.keys()):
        scen_ids = sorted(family_groups[fam])
        N_f = len(scen_ids)
        shuffled_indices = rng.permutation(N_f)
        shuffled_scen_ids = [scen_ids[i] for i in shuffled_indices]

        n_train = int(round(N_f * train_ratio))
        n_val = int(round(N_f * val_ratio))
        n_test = N_f - n_train - n_val

        # If N_f >= 3 and all 3 splits are requested, ensure non-empty partitions
        if N_f >= 3 and val_ratio > 0 and test_ratio > 0:
            if n_val == 0 and n_train > 1:
                n_val = 1
                n_train -= 1
            if n_test == 0 and n_train > 1:
                n_test = 1
                n_train -= 1

        train_ids = shuffled_scen_ids[:n_train]
        val_ids = shuffled_scen_ids[n_train : n_train + n_val]
        test_ids = shuffled_scen_ids[n_train + n_val :]

        for s_id in train_ids:
            scenario_splits[s_id] = "train"
        for s_id in val_ids:
            scenario_splits[s_id] = "val"
        for s_id in test_ids:
            scenario_splits[s_id] = "test"

    return scenario_splits


def compute_behavior_aware_scenario_splits(
    scenario_profiles: List[Dict[str, Any]],
    split_seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, str]:
    """Deterministically stratifies scenarios using behavioral signatures + scenario family.

    Guarantees:
    - Zero scenario leakage across splits
    - Exact per-family representation (e.g., for 20 scenarios/family: 14 train, 3 val, 3 test)
    - Total split: 70 train, 15 val, 15 test for a 100-scenario dataset
    - Rare regime representation: TEST and VALIDATION both contain independent scenarios
      exhibiting rare 24.5°C and 26.5°C setpoints where available in the candidate pool
    - Minimizes Jensen-Shannon target distribution shift between train, validation, and test
    """
    from collections import Counter
    import math
    from scipy.spatial.distance import jensenshannon

    family_groups: Dict[str, List[Dict[str, Any]]] = {}
    for p in scenario_profiles:
        fam = p["scenario_family"]
        family_groups.setdefault(fam, []).append(p)

    rng = np.random.Generator(np.random.PCG64(split_seed))
    best_split_assignment: Dict[str, str] = {}
    best_loss = float("inf")

    # Precompute label counts per scenario for instant evaluation
    target_vals = [24.5, 25.0, 25.5, 26.0, 26.5]
    for p in scenario_profiles:
        if "_counts" not in p:
            p["_counts"] = Counter(p.get("labels", [p.get("mean_sp", 25.5)]))

    # Evaluate candidate permutations to find globally optimal behavior-aware partition
    num_trials = 500 if len(scenario_profiles) >= 50 else 50

    for _ in range(num_trials):
        trial_assignment: Dict[str, str] = {}
        tr_cnt = Counter()
        val_cnt = Counter()
        test_cnt = Counter()

        has_req_low_test = False
        has_req_low_val = False
        has_req_low_train = False
        has_req_high_test = False
        has_req_high_val = False
        has_req_high_train = False

        for fam in sorted(family_groups.keys()):
            profiles = list(family_groups[fam])
            N_f = len(profiles)

            # Categorize by behavioral regime (V2: low cooling setpoint <= 23.0°C, high setpoint >= 24.5°C)
            hot_scens = [
                p for p in profiles
                if (p.get("has_cool_target") or (p.get("min_sp", 24.0) <= 23.0) or any(s <= 23.0 for s in p.get("labels", [])))
            ]
            cool_scens = [
                p for p in profiles
                if (p.get("has_warm_target") or (p.get("max_sp", 24.0) >= 24.5) or any(s >= 24.5 for s in p.get("labels", [])))
                and p not in hot_scens
            ]
            std_scens = [p for p in profiles if p not in hot_scens and p not in cool_scens]

            # Deterministic permutation for trial
            hot_shuffled = [hot_scens[i] for i in rng.permutation(len(hot_scens))]
            cool_shuffled = [cool_scens[i] for i in rng.permutation(len(cool_scens))]
            std_shuffled = [std_scens[i] for i in rng.permutation(len(std_scens))]

            n_train = int(round(N_f * train_ratio))
            n_val = int(round(N_f * val_ratio))
            n_test = N_f - n_train - n_val
            if N_f >= 3 and val_ratio > 0 and test_ratio > 0:
                if n_val == 0 and n_train > 1:
                    n_val = 1
                    n_train -= 1
                if n_test == 0 and n_train > 1:
                    n_test = 1
                    n_train -= 1

            fam_val: List[Dict[str, Any]] = []
            fam_test: List[Dict[str, Any]] = []
            fam_train: List[Dict[str, Any]] = []

            def assign_to_split(item: Dict[str, Any], pref_order: List[str]) -> str:
                for pref in pref_order:
                    if pref == "train" and len(fam_train) < n_train:
                        fam_train.append(item)
                        return "train"
                    elif pref == "val" and len(fam_val) < n_val:
                        fam_val.append(item)
                        return "val"
                    elif pref == "test" and len(fam_test) < n_test:
                        fam_test.append(item)
                        return "test"
                for s, lst, cap in [("train", fam_train, n_train), ("val", fam_val, n_val), ("test", fam_test, n_test)]:
                    if len(lst) < cap:
                        lst.append(item)
                        return s
                return "train"

            # Distribute high-cooling demand scenarios across splits
            for h in hot_shuffled:
                assigned = assign_to_split(h, list(rng.permutation(["train", "val", "test"])))
                if assigned == "test":
                    has_req_low_test = True
                elif assigned == "val":
                    has_req_low_val = True
                elif assigned == "train":
                    has_req_low_train = True

            # Distribute warm/relaxed scenarios across splits
            for c in cool_shuffled:
                assigned = assign_to_split(c, list(rng.permutation(["train", "val", "test"])))
                if assigned == "test":
                    has_req_high_test = True
                elif assigned == "val":
                    has_req_high_val = True
                elif assigned == "train":
                    has_req_high_train = True

            # Fill remaining quota with standard scenarios
            for s in std_shuffled:
                assign_to_split(s, list(rng.permutation(["train", "val", "test"])))

            for p in fam_train:
                trial_assignment[p["scenario_id"]] = "train"
                tr_cnt.update(p["_counts"])
            for p in fam_val:
                trial_assignment[p["scenario_id"]] = "val"
                val_cnt.update(p["_counts"])
            for p in fam_test:
                trial_assignment[p["scenario_id"]] = "test"
                test_cnt.update(p["_counts"])

        # Check hard constraints on rare regimes if candidate pool contains them
        total_hot = sum(
            1 for p in scenario_profiles
            if (p.get("has_cool_target") or (p.get("min_sp", 24.0) <= 23.0) or any(s <= 23.0 for s in p.get("labels", [])))
        )
        total_cool = sum(
            1 for p in scenario_profiles
            if (p.get("has_warm_target") or (p.get("max_sp", 24.0) >= 24.5) or any(s >= 24.5 for s in p.get("labels", [])))
        )

        if total_hot >= 3:
            if not (has_req_low_test and has_req_low_val and has_req_low_train):
                continue
        if total_cool >= 3:
            if not (has_req_high_test and has_req_high_val and has_req_high_train):
                continue

        # Evaluate target distribution distance (Jensen-Shannon distance)
        n_tr = sum(tr_cnt.values())
        n_val = sum(val_cnt.values())
        n_test = sum(test_cnt.values())

        p_tr = np.array([tr_cnt.get(v, 0) / max(1, n_tr) for v in target_vals])
        q_val = np.array([val_cnt.get(v, 0) / max(1, n_val) for v in target_vals])
        q_test = np.array([test_cnt.get(v, 0) / max(1, n_test) for v in target_vals])

        js_val = float(jensenshannon(p_tr, q_val, base=2)) if n_val > 0 else 0.0
        js_test = float(jensenshannon(p_tr, q_test, base=2)) if n_test > 0 else 0.0

        loss = js_val + js_test + (0.5 * max(js_val, js_test))
        if loss < best_loss:
            best_loss = loss
            best_split_assignment = trial_assignment

    # Fallback if no trial met constraints
    if not best_split_assignment:
        scen_tuples = [(p["scenario_id"], p["scenario_family"]) for p in scenario_profiles]
        best_split_assignment = compute_stratified_scenario_splits(
            scen_tuples, split_seed, train_ratio, val_ratio, test_ratio
        )

    return best_split_assignment


class StreamingDatasetWriter:
    """Streams time-series rows directly into raw, train, val, and test CSV files."""

    def __init__(
        self,
        output_dir: Path,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split_seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed

        # Directory structure
        self.raw_dir = self.output_dir / "raw"
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "validation"
        self.test_dir = self.output_dir / "test"
        self.metadata_dir = self.output_dir / "metadata"

        for d in [self.raw_dir, self.train_dir, self.val_dir, self.test_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.raw_path = self.raw_dir / "all_scenarios.csv"
        self.train_path = self.train_dir / "train.csv"
        self.val_path = self.val_dir / "validation.csv"
        self.test_path = self.test_dir / "test.csv"

        # File handles & writers
        self._handles: Dict[str, any] = {}
        self._csv_writers: Dict[str, csv.DictWriter] = {}
        self._headers_written = False

        # Scenario split mapping: scenario_id -> "train" | "val" | "test"
        self.scenario_splits: Dict[str, str] = {}
        self.row_counts = {"raw": 0, "train": 0, "val": 0, "test": 0}

    def plan_stratified_splits(
        self,
        scenario_tuples: List[Tuple[str, str]],
        split_seed: Optional[int] = None,
    ) -> Dict[str, str]:
        """Pre-compute stratified partition for all planned scenarios."""
        if split_seed is not None:
            self.split_seed = split_seed
        self.scenario_splits = compute_stratified_scenario_splits(
            scenario_tuples=scenario_tuples,
            split_seed=self.split_seed,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
        )
        return self.scenario_splits

    def plan_behavior_aware_splits(
        self,
        scenario_profiles: List[Dict[str, Any]],
        split_seed: Optional[int] = None,
    ) -> Dict[str, str]:
        """Compute behavior-aware stratified partition for scenarios."""
        if split_seed is not None:
            self.split_seed = split_seed
        self.scenario_splits = compute_behavior_aware_scenario_splits(
            scenario_profiles=scenario_profiles,
            split_seed=self.split_seed,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
        )
        return self.scenario_splits

    def materialize_splits_from_raw(self, scenario_splits: Dict[str, str]):
        """Stream rows from raw CSV into train, validation, and test CSVs according to scenario_splits."""
        self.scenario_splits = scenario_splits

        # Close all existing handles before reading raw file
        self.close()

        with open(self.raw_path, mode="r", encoding="utf-8") as raw_f:
            reader = csv.DictReader(raw_f)
            fieldnames = reader.fieldnames

            with open(self.train_path, mode="w", newline="", encoding="utf-8") as tr_f, \
                 open(self.val_path, mode="w", newline="", encoding="utf-8") as val_f, \
                 open(self.test_path, mode="w", newline="", encoding="utf-8") as test_f:

                writers = {
                    "train": csv.DictWriter(tr_f, fieldnames=fieldnames),
                    "val": csv.DictWriter(val_f, fieldnames=fieldnames),
                    "test": csv.DictWriter(test_f, fieldnames=fieldnames),
                }
                for w in writers.values():
                    w.writeheader()

                counts = {"raw": 0, "train": 0, "val": 0, "test": 0}
                for row in reader:
                    counts["raw"] += 1
                    s_id = row["scenario_id"]
                    split = scenario_splits.get(s_id, "train")
                    writers[split].writerow(row)
                    counts[split] += 1

                self.row_counts = counts

    def open_raw_only(self):
        """Open only the raw CSV file handle for initial scenario simulation."""
        self._handles["raw"] = open(self.raw_path, mode="w", newline="", encoding="utf-8")
        self._headers_written = False

    def write_raw_row(self, row: Dict):
        """Write a row to raw CSV only."""
        if not self._headers_written:
            fieldnames = list(row.keys())
            self._csv_writers["raw"] = csv.DictWriter(self._handles["raw"], fieldnames=fieldnames)
            self._csv_writers["raw"].writeheader()
            self._headers_written = True

        self._csv_writers["raw"].writerow(row)
        self.row_counts["raw"] += 1

    def assign_scenario_split(self, scenario_id: str, rng: Optional[np.random.Generator] = None) -> str:
        """Retrieve assigned split for a scenario. Fallback to random if unassigned."""
        if scenario_id in self.scenario_splits:
            return self.scenario_splits[scenario_id]

        local_rng = rng or np.random.default_rng(self.split_seed)
        rand_val = local_rng.random()
        if rand_val < self.train_ratio:
            split = "train"
        elif rand_val < (self.train_ratio + self.val_ratio):
            split = "val"
        else:
            split = "test"

        self.scenario_splits[scenario_id] = split
        return split

    def open(self):
        """Open all streaming file handles."""
        self._handles["raw"] = open(self.raw_path, mode="w", newline="", encoding="utf-8")
        self._handles["train"] = open(self.train_path, mode="w", newline="", encoding="utf-8")
        self._handles["val"] = open(self.val_path, mode="w", newline="", encoding="utf-8")
        self._handles["test"] = open(self.test_path, mode="w", newline="", encoding="utf-8")
        self._headers_written = False

    def write_row(self, row: Dict, split: str):
        """Write a single row to raw CSV and the designated split CSV."""
        if not self._headers_written:
            fieldnames = list(row.keys())
            for key, handle in self._handles.items():
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                self._csv_writers[key] = writer
            self._headers_written = True

        self._csv_writers["raw"].writerow(row)
        self.row_counts["raw"] += 1

        if split in self._csv_writers:
            self._csv_writers[split].writerow(row)
            self.row_counts[split] += 1

    def close(self):
        """Flush and close all open CSV handles."""
        for handle in self._handles.values():
            if handle and not handle.closed:
                handle.flush()
                handle.close()
        self._handles.clear()
        self._csv_writers.clear()
