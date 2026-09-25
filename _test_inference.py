"""Quick E2E inference test - run from project root."""
import sys
sys.path.insert(0, ".")
from ml.inference import predict_room_setpoint
from ml.feature_adapter import FeatureAdapter

adapter = FeatureAdapter()
state = {
    "occupancy": {"total": 12, "zones": {"zone_1": 3, "zone_2": 4, "zone_3": 3, "zone_4": 2}},
    "environment": {"outdoor_temperature_c": 32.0, "humidity_percent": 65.0, "solar_irradiance_w_m2": 300.0},
    "hvac": {"acs": [
        {"id": "AC-1", "wall": "north", "state": "COOLING", "setpoint_c": 23.0, "cooling_level": 0.6},
        {"id": "AC-2", "wall": "east", "state": "COOLING", "setpoint_c": 23.5, "cooling_level": 0.5},
        {"id": "AC-3", "wall": "south", "state": "COOLING", "setpoint_c": 23.0, "cooling_level": 0.4},
        {"id": "AC-4", "wall": "west", "state": "OFF", "setpoint_c": 24.0, "cooling_level": 0.0},
    ]},
    "thermal": {
        "zone_temperatures_c": {"zone_1": 24.5, "zone_2": 23.8, "zone_3": 24.1, "zone_4": 23.5},
        "room_average_temperature_c": 23.97,
        "minimum_temperature_c": 23.5,
        "maximum_temperature_c": 24.5,
        "temperature_difference_c": 1.0,
        "computer_heat_watts": 1200.0,
        "occupancy_heat_watts": 840.0,
        "hvac_cooling_watts": 5250.0,
    },
    "computers": [{"id": i, "cpu_util_percent": 25.0+i, "gpu_util_percent": 10.0+i, "workload_category": "LIGHT", "heat_watts": 85.0+i*5} for i in range(1, 11)],
}

features, warnings = adapter.adapt(state)
print(f"Features mapped: {len(features)}")
print(f"Warnings: {len(warnings)}")
for w in warnings:
    print(f"  - {w}")

result = predict_room_setpoint(features)
print(f"\nStatus: {result['status']}")
print(f"Predicted setpoint: {result['predicted_class']} C")
print(f"Confidence: {result['confidence']}")
print(f"Mode: {result['mode']}")
print(f"Class probabilities: {result['class_probabilities']}")
print(f"Feature warnings from inference: {len(result.get('feature_warnings', []))}")
