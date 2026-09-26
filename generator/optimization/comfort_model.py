"""
Thermal Comfort Model Implementation (ISO 7730 / ASHRAE 55 PMV-PPD).

Provides an extensible ComfortModel abstraction and a full Fanger PMV/PPD
implementation parameterized with environmental physics, metabolic rates,
clothing insulation, and Indian comfort baseline references.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple
import numpy as np

from generator.config import ComfortConfig
from generator.models.room import ZONES


@dataclass
class ZoneComfortMetrics:
    zone: str
    pmv: float
    ppd: float
    comfort_score: float
    comfort_penalty: float
    overheating_penalty: float
    overcooling_penalty: float


@dataclass
class RoomComfortMetrics:
    room_comfort_score: float
    room_comfort_penalty: float
    room_overheating_penalty: float
    room_overcooling_penalty: float
    mean_pmv: float
    mean_ppd: float
    zone_metrics: Dict[str, ZoneComfortMetrics]


class ComfortModel(ABC):
    """Abstract base class for thermal comfort evaluation."""

    @abstractmethod
    def evaluate_zone(
        self,
        air_temp_c: float,
        humidity_percent: float,
        air_speed_m_s: float,
        mean_radiant_temp_c: float = None,
    ) -> ZoneComfortMetrics:
        """Evaluate comfort metrics for a single zone."""
        pass

    @abstractmethod
    def evaluate_room(
        self,
        zone_temps: Dict[str, float],
        humidity_percent: float,
    ) -> RoomComfortMetrics:
        """Evaluate overall room comfort metrics across all zones."""
        pass


class FangerPMVComfortModel(ComfortModel):
    """Standard ISO 7730 / ASHRAE 55 Fanger PMV and PPD thermal comfort model."""

    def __init__(self, config: ComfortConfig = None):
        self.config = config or ComfortConfig()

    def _calc_saturated_vapor_pressure_kpa(self, temp_c: float) -> float:
        """Antoine / Magnus formula for saturated water vapor pressure."""
        return 0.1 * np.exp(18.956 - 4030.18 / (temp_c + 235.0))

    @staticmethod
    @lru_cache(maxsize=16384)
    def _compute_pmv_ppd_cached(
        air_temp_c: float,
        mean_radiant_temp_c: float,
        rel_humidity_pct: float,
        air_velocity_m_s: float,
        met_rate: float,
        clo_insulation: float,
    ) -> Tuple[float, float]:
        # Convert units
        m = met_rate * 58.15  # W/m2 metabolic rate
        w = 0.0               # External mechanical work is 0 for office tasks
        icl = clo_insulation * 0.155  # m2 K / W thermal resistance of clothing

        # Saturated vapor pressure and partial water vapor pressure in Pa
        p_sat = 0.1 * np.exp(18.956 - 4030.18 / (air_temp_c + 235.0)) * 1000.0
        pa = (rel_humidity_pct / 100.0) * p_sat

        fcl = 1.0 + 0.2 * clo_insulation if clo_insulation < 0.5 else 1.05 + 0.1 * clo_insulation

        # Heat loss through respiration and diffusion
        hl_diff = 3.05e-3 * (5733.0 - 6.99 * (m - w) - pa)
        hl_sweat = max(0.0, 0.42 * ((m - w) - 58.15))
        hl_lat_resp = 1.7e-5 * m * (5867.0 - pa)
        hl_dry_resp = 0.0014 * m * (34.0 - air_temp_c)

        # Iterative calculation of clothing surface temperature Tcl
        t_air_k = air_temp_c + 273.15
        t_rad_k = mean_radiant_temp_c + 273.15
        t_skin_c = 35.7 - 0.028 * (m - w)
        t_skin_k = t_skin_c + 273.15

        t_cl_k = t_skin_k - (t_skin_c - air_temp_c) * 0.5  # initial guess
        for _ in range(35):
            hc_nat = 2.38 * (abs(t_cl_k - t_air_k) ** 0.25)
            hc_forced = 12.1 * np.sqrt(max(0.01, air_velocity_m_s))
            hc = max(hc_nat, hc_forced)

            hr = 3.96e-8 * fcl * (t_cl_k ** 4 - t_rad_k ** 4)
            hc_loss = fcl * hc * (t_cl_k - t_air_k)

            t_cl_new_k = t_skin_k - icl * (hr + hc_loss)
            if abs(t_cl_new_k - t_cl_k) < 0.001:
                t_cl_k = t_cl_new_k
                break
            t_cl_k = 0.5 * (t_cl_k + t_cl_new_k)

        hc_nat = 2.38 * (abs(t_cl_k - t_air_k) ** 0.25)
        hc_forced = 12.1 * np.sqrt(max(0.01, air_velocity_m_s))
        hc = max(hc_nat, hc_forced)

        r_loss = 3.96e-8 * fcl * (t_cl_k ** 4 - t_rad_k ** 4)
        c_loss = fcl * hc * (t_cl_k - t_air_k)

        # Internal heat load L
        heat_load = (m - w) - hl_diff - hl_sweat - hl_lat_resp - hl_dry_resp - r_loss - c_loss

        # Thermal sensation coefficient
        ts = 0.303 * np.exp(-0.036 * m) + 0.028
        pmv = ts * heat_load
        pmv = float(np.clip(pmv, -3.5, 3.5))

        # Predicted Percentage Dissatisfied
        ppd = 100.0 - 95.0 * np.exp(-0.03353 * (pmv ** 4) - 0.2179 * (pmv ** 2))
        ppd = float(np.clip(ppd, 5.0, 100.0))

        return round(pmv, 3), round(ppd, 2)

    def calculate_pmv_ppd(
        self,
        air_temp_c: float,
        mean_radiant_temp_c: float,
        rel_humidity_pct: float,
        air_velocity_m_s: float,
        met_rate: float,
        clo_insulation: float,
    ) -> Tuple[float, float]:
        """Calculates Fanger PMV and PPD with LRU memoization."""
        return self._compute_pmv_ppd_cached(
            round(air_temp_c, 2),
            round(mean_radiant_temp_c, 2),
            round(rel_humidity_pct, 1),
            air_velocity_m_s,
            met_rate,
            clo_insulation,
        )

    def evaluate_zone(
        self,
        air_temp_c: float,
        humidity_percent: float,
        air_speed_m_s: float = None,
        mean_radiant_temp_c: float = None,
        zone_name: str = "ZONE",
    ) -> ZoneComfortMetrics:
        """Evaluate standard comfort indices and penalties for a zone."""
        v_air = air_speed_m_s or self.config.air_speed_m_s
        t_rad = mean_radiant_temp_c if mean_radiant_temp_c is not None else air_temp_c

        pmv, ppd = self.calculate_pmv_ppd(
            air_temp_c=air_temp_c,
            mean_radiant_temp_c=t_rad,
            rel_humidity_pct=humidity_percent,
            air_velocity_m_s=v_air,
            met_rate=self.config.metabolic_rate_met,
            clo_insulation=self.config.clothing_insulation_clo,
        )

        # Comfort score: 100 at 5% PPD (minimum possible dissatisfaction), 0 at 100% PPD
        comfort_score = max(0.0, min(100.0, (100.0 - ppd) * (100.0 / 95.0)))

        # Comfort penalty: quadratic penalty for deviation from neutral PMV
        comfort_penalty = (pmv - self.config.target_pmv) ** 2

        # Overheating and overcooling penalties based on comfort envelope
        upper_bound = self.config.upper_comfort_threshold_c
        lower_bound = self.config.lower_comfort_threshold_c

        overheat_penalty = max(0.0, air_temp_c - upper_bound) ** 2
        overcool_penalty = max(0.0, lower_bound - air_temp_c) ** 2

        return ZoneComfortMetrics(
            zone=zone_name,
            pmv=pmv,
            ppd=ppd,
            comfort_score=round(comfort_score, 2),
            comfort_penalty=round(comfort_penalty, 4),
            overheating_penalty=round(overheat_penalty, 4),
            overcooling_penalty=round(overcool_penalty, 4),
        )

    def evaluate_room(
        self,
        zone_temps: Dict[str, float],
        humidity_percent: float,
    ) -> RoomComfortMetrics:
        """Evaluate comfort across all 4 zones and aggregate to room level."""
        zone_metrics = {}
        for zone in ZONES:
            zone_metrics[zone] = self.evaluate_zone(
                air_temp_c=zone_temps[zone],
                humidity_percent=humidity_percent,
                zone_name=zone,
            )

        avg_score = float(np.mean([m.comfort_score for m in zone_metrics.values()]))
        avg_penalty = float(np.mean([m.comfort_penalty for m in zone_metrics.values()]))
        avg_overheat = float(np.mean([m.overheating_penalty for m in zone_metrics.values()]))
        avg_overcool = float(np.mean([m.overcooling_penalty for m in zone_metrics.values()]))
        avg_pmv = float(np.mean([m.pmv for m in zone_metrics.values()]))
        avg_ppd = float(np.mean([m.ppd for m in zone_metrics.values()]))

        return RoomComfortMetrics(
            room_comfort_score=round(avg_score, 2),
            room_comfort_penalty=round(avg_penalty, 4),
            room_overheating_penalty=round(avg_overheat, 4),
            room_overcooling_penalty=round(avg_overcool, 4),
            mean_pmv=round(avg_pmv, 3),
            mean_ppd=round(avg_ppd, 2),
            zone_metrics=zone_metrics,
        )
