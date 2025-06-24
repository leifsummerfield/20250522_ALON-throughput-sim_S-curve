"""Motion profile definitions for stage simulation."""
from dataclasses import dataclass


@dataclass
class MotionProfile:
    name: str
    max_velocity: float  # mm/s
    max_acceleration: float  # mm/s^2
    jerk_max: float = 10000  # mm/s^3
    settle_time: float = 0.02  # s
    color: str = "blue"


# Example profile used in notebooks or quick tests
stage_profile = MotionProfile(
    name="Stage",
    max_velocity=300,
    max_acceleration=2000,
    color="blue",
)
