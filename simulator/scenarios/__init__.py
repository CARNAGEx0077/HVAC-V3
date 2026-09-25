"""
Scenarios Package for HVEAC V3 Thermal Simulation.
"""

from simulator.scenarios.base_scenario import BaseScenario
from simulator.scenarios.scenario_1 import Scenario1LocalizedCompute
from simulator.scenarios.scenario_2 import Scenario2OccupancyConcentration
from simulator.scenarios.scenario_3 import Scenario3DistributedCompute
from simulator.scenarios.scenario_4 import Scenario4HighOccupancyLowCompute
from simulator.scenarios.scenario_5 import Scenario5OpposingZones

ALL_SCENARIOS = {
    1: Scenario1LocalizedCompute,
    2: Scenario2OccupancyConcentration,
    3: Scenario3DistributedCompute,
    4: Scenario4HighOccupancyLowCompute,
    5: Scenario5OpposingZones,
}

__all__ = [
    "BaseScenario",
    "Scenario1LocalizedCompute",
    "Scenario2OccupancyConcentration",
    "Scenario3DistributedCompute",
    "Scenario4HighOccupancyLowCompute",
    "Scenario5OpposingZones",
    "ALL_SCENARIOS",
]
