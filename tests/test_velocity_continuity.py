import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'scripts'))

import numpy as np
from motion_profile import MotionProfile
from simulator import ExperimentSimulator


def test_velocity_continuity_after_accel():
    profile = MotionProfile(name="Stage", max_velocity=5, max_acceleration=25, jerk_max=40000)
    sim = ExperimentSimulator(
        profile=profile,
        scan_length=10,
        step_size=0.005,
        num_lines=1,
        index_time=0.06,
        overhead_per_line=0.156,
    )
    t, x, v, a, phase = sim.simulate_scurve_scan_with_backoff()
    phase = np.array(phase)
    accel_end = np.where(phase == "accel")[0][-1]
    const_start = accel_end + 1
    diff = abs(v[const_start] - v[accel_end])
    assert diff < 1e-6
