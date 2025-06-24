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


# Example profiles used in notebooks or quick tests
linear_motor = MotionProfile(
    name="Linear Motor",
    max_velocity=300,
    max_acceleration=2000,
    color="blue",
)

stepper_motor = MotionProfile(
    name="Stepper Motor",
    max_velocity=50,
    max_acceleration=8000,
    color="orange",
)
