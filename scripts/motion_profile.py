# scripts/motion_profile.py

from dataclasses import dataclass

@dataclass
class MotionProfile:
    name: str
    max_velocity: float        # mm/s
    max_acceleration: float    # mm/s^2
    jerk_max: float = 10000    # mm/s^3, default value
    settle_time: float = 0.02  # s (set per-stage)
    color: str = "blue"

# Example profiles
linear_motor = MotionProfile(name="Linear Motor", max_velocity=300, max_acceleration=2000, color="blue")
stepper_motor = MotionProfile(name="Stepper Motor", max_velocity=50, max_acceleration=8000, color="orange")


from dataclasses import dataclass


