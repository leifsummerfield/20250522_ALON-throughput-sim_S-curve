import numpy as np
import math
from motion_profile import MotionProfile


def _time_array(duration: float, dt: float) -> np.ndarray:
    """Return a time array from 0 to duration inclusive using step dt."""
    arr = np.arange(0, duration, dt)
    if not np.isclose(arr[-1], duration):
        arr = np.append(arr, duration)
    return arr


def s_curve_to_velocity(v_end: float, amax: float, jmax: float, dt: float = 0.0005):
    """
    Generate a jerk-limited S-curve to accelerate from 0 to v_end.
    Returns arrays: time, position, velocity, acceleration
    """
    # Threshold velocity below which we never saturate amax
    v_thresh = amax**2 / jmax
    if v_end <= v_thresh:
        # Triangular jerk-limited profile (no constant-accel segment)
        t1 = np.sqrt(v_end / jmax)
        t2 = 0.0
    else:
        # Trapezoidal S-curve with constant-accel segment
        t1 = amax / jmax
        t2 = (v_end - v_thresh) / amax
    t3 = t1

    # Time grids for each phase
    t1_phase = _time_array(t1, dt)
    t2_phase = _time_array(t2, dt)[1:] if t2 > 0 else np.array([])
    t3_phase = _time_array(t3, dt)

    # Phase 1: jerk up
    a1 = jmax * t1_phase
    v1 = 0.5 * jmax * t1_phase**2
    x1 = (jmax / 6) * t1_phase**3

    # Phase 2: constant accel
    if t2 > 0:
        a2 = np.full_like(t2_phase, amax)
        v2 = v1[-1] + amax * t2_phase
        x2 = x1[-1] + v1[-1] * t2_phase + 0.5 * amax * t2_phase**2
    else:
        a2 = np.array([])
        v2 = np.array([])
        x2 = np.array([])

    # Phase 3: jerk down
    base_v = v2[-1] if t2 > 0 else v1[-1]
    base_x = x2[-1] if t2 > 0 else x1[-1]
    a3 = amax - jmax * t3_phase
    v3 = base_v + amax * t3_phase - 0.5 * jmax * t3_phase**2
    x3 = base_x + base_v * t3_phase + 0.5 * amax * t3_phase**2 - (jmax / 6) * t3_phase**3

    # Concatenate across phases
    t = np.concatenate([
        t1_phase,
        (t1_phase[-1] + t2_phase) if t2 > 0 else np.array([]),
        (t1_phase[-1] + (t2_phase[-1] if t2 > 0 else 0) + t3_phase)
    ])
    x = np.concatenate([x1, x2, x3])
    v = np.concatenate([v1, v2, v3])
    a = np.concatenate([a1, a2, a3])

    return t, x, v, a


def s_curve_phase_durations(vmax: float, amax: float, jmax: float, distance: float):
    """Return (t_accel, t_const, t_decel) durations for a jerk limited S-curve move."""
    # Time to ramp acceleration with max jerk
    v_thresh = amax ** 2 / jmax

    if vmax <= v_thresh:
        # We never reach amax - triangular jerk limited profile
        t1 = math.sqrt(vmax / jmax)
        t2 = 0.0
    else:
        # Trapezoidal profile with constant acceleration segment
        t1 = amax / jmax
        t2 = (vmax - v_thresh) / amax

    t3 = t1
    t_accel = t1 + t2 + t3

    # Distance travelled during acceleration phase
    d1 = jmax * t1 ** 3 / 6
    v1 = 0.5 * jmax * t1 ** 2
    d2 = v1 * t2 + 0.5 * amax * t2 ** 2
    v2 = v1 + amax * t2
    d3 = v2 * t3 + 0.5 * amax * t3 ** 2 - jmax * t3 ** 3 / 6
    d_accel = d1 + d2 + d3

    if distance <= 2 * d_accel:
        # Not enough distance to reach vmax - triangular profile overall
        t_total = (6 * distance / jmax) ** (1 / 3)
        return t_total, 0.0, t_total

    t_const = (distance - 2 * d_accel) / vmax
    t_decel = t_accel
    return t_accel, t_const, t_decel


class ExperimentSimulator:
    def __init__(
        self,
        profile: MotionProfile,
        scan_length: float,
        step_size: float,
        num_lines: int,
        index_time: float,
        overhead_per_line: float
    ):
        self.profile = profile
        self.scan_length = scan_length
        self.step_size = step_size
        self.num_lines = num_lines
        self.index_time = index_time
        self.overhead_per_line = overhead_per_line

    def simulate_scurve_scan_with_backoff(self, dt: float = 0.0005):
        """
        Simulate back-to-back S-curve accel to vmax,
        constant-velocity scan, then mirrored S-curve decel.
        Returns: time, position, velocity, acceleration, phase.
        """
        amax = self.profile.max_acceleration
        vmax = self.profile.max_velocity
        jmax = getattr(self.profile, "jerk_max", 10000)
        L = self.scan_length

        # ------ Acceleration ------
        t_accel, x_accel, v_accel, a_accel = s_curve_to_velocity(
            vmax, amax, jmax, dt
        )

        # ------ Constant Velocity Scan ------
        t_scan = np.arange(0, L / vmax, dt)
        x_scan = x_accel[-1] + vmax * t_scan
        v_scan = np.full_like(t_scan, vmax)
        a_scan = np.zeros_like(t_scan)

        # ------ Deceleration (mirror of accel) ------
        t_decel = t_accel
        a_decel = -a_accel[::-1]
        v_decel = v_accel[::-1]
        # Position steps back from the end of scan
        x_decel = x_scan[-1] + (x_accel[-1] - x_accel[::-1])

        # ------ Concatenate full profile ------
        t = np.concatenate([
            t_accel,
            t_scan + t_accel[-1],
            t_decel + t_accel[-1] + t_scan[-1]
        ])
        x = np.concatenate([x_accel, x_scan, x_decel])
        v = np.concatenate([v_accel, v_scan, v_decel])
        a = np.concatenate([a_accel, a_scan, a_decel])
        phase = np.array(
            ["accel"] * len(t_accel)
            + ["const"] * len(t_scan)
            + ["decel"] * len(t_decel)
        )

        return t, x, v, a, phase

    def compute_time_breakdown(self):
        """
        Compute total times for each phase across the full raster.
        """
        vmax = self.profile.max_velocity
        amax = self.profile.max_acceleration
        jmax = getattr(self.profile, "jerk_max", 10000)
        L = self.scan_length

        t_accel, t_const, t_decel = s_curve_phase_durations(
            vmax, amax, jmax, L
        )
        t_index = self.index_time
        t_settle = self.profile.settle_time
        t_overhead = self.overhead_per_line

        breakdown = {
            "Acceleration": t_accel * self.num_lines,
            "Constant":     t_const * self.num_lines,
            "Deceleration": t_decel * self.num_lines,
            "Index":        t_index * self.num_lines,
            "Settle":       t_settle * self.num_lines,
            "Overhead":     t_overhead * self.num_lines,
        }
        breakdown["Total"] = sum(breakdown.values())
        return breakdown
