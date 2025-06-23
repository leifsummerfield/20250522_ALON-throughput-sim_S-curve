import numpy as np
import pandas as pd
from motion_profile import MotionProfile
from simulator import ExperimentSimulator

def sweep_velocity(stage_name, accel, jerk, settle_time, color, velocity_range, experiment_config):
    """
    Sweep max_velocity, return velocity_list and total_time_list for this stage type.
    """
    results = []
    for vmax in velocity_range:
        profile = MotionProfile(
            name=stage_name,
            max_velocity=vmax,
            max_acceleration=accel,
            jerk_max=jerk,
            settle_time=settle_time,
            color=color
        )
        sim = ExperimentSimulator(
            profile=profile,
            scan_length=experiment_config['scan_length'],
            step_size=experiment_config['step_size'],
            num_lines=experiment_config['num_lines'],
            index_time=experiment_config['index_time'],
            overhead_per_line=experiment_config['overhead_per_line']
        )
        breakdown = sim.compute_time_breakdown()
        results.append((vmax, breakdown['Total']))
    velocities, total_times = zip(*results)
    return np.array(velocities), np.array(total_times)

def sweep_accel(stage_name, vmax, jerk, settle_time, color, accel_range, experiment_config):
    """
    Sweep max_acceleration, return accel_list and total_time_list for this stage type.
    """
    results = []
    for accel in accel_range:
        profile = MotionProfile(
            name=stage_name,
            max_velocity=vmax,
            max_acceleration=accel,
            jerk_max=jerk,
            settle_time=settle_time,
            color=color
        )
        sim = ExperimentSimulator(
            profile=profile,
            scan_length=experiment_config['scan_length'],
            step_size=experiment_config['step_size'],
            num_lines=experiment_config['num_lines'],
            index_time=experiment_config['index_time'],
            overhead_per_line=experiment_config['overhead_per_line']
        )
        breakdown = sim.compute_time_breakdown()
        results.append((accel, breakdown['Total']))
    accels, total_times = zip(*results)
    return np.array(accels), np.array(total_times)

def sweep_velocity_accel(stage_name, jerk, settle_time, color, velocity_range, accel_range, experiment_config):
    """
    2D sweep: For each (v, a) pair, compute total experiment time. Returns meshgrid and Z.
    """
    Z = np.zeros((len(accel_range), len(velocity_range)))
    for i, accel in enumerate(accel_range):
        for j, vmax in enumerate(velocity_range):
            profile = MotionProfile(
                name=stage_name,
                max_velocity=vmax,
                max_acceleration=accel,
                jerk_max=jerk,
                settle_time=settle_time,
                color=color
            )
            sim = ExperimentSimulator(
                profile=profile,
                scan_length=experiment_config['scan_length'],
                step_size=experiment_config['step_size'],
                num_lines=experiment_config['num_lines'],
                index_time=experiment_config['index_time'],
                overhead_per_line=experiment_config['overhead_per_line']
            )
            breakdown = sim.compute_time_breakdown()
            Z[i, j] = breakdown['Total']
    V, A = np.meshgrid(velocity_range, accel_range)
    return V, A, Z

def analyze_stages_from_csv(
    csv_path,
    experiment_config,
    index_time_lookup=None,
    settle_time_lookup=None,
    overhead_per_line=0.001,
    jerk_lookup=None,
    default_jerk=10000
):
    """
    Reads CSV with columns:
        Stage Type, Supplier and PN, Speed (mm/s), Acceleration (m/s^2)
    For each, simulates and returns:
        - list of supplier/pn (labels)
        - list of stage types
        - list of breakdown dicts
    Optionally accepts index_time_lookup, settle_time_lookup, and jerk_lookup dicts keyed by Stage Type.
    """
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp1252")

    labels = []
    stage_types = []
    breakdowns = []
    for _, row in df.iterrows():
        stage_type = row['Stage Type']
        label = f"{row['Supplier and PN']}"  # can abbreviate later if desired
        max_velocity = float(row['Speed (mm/s)'])
        max_acceleration = float(row['Acceleration (m/s^2)'])
        # Optionally, provide per-stage-type settle/index time here:
        settle_time = settle_time_lookup[stage_type] if settle_time_lookup else 0.02
        index_time = index_time_lookup[stage_type] if index_time_lookup else experiment_config['index_time']
        jerk = None
        if jerk_lookup and stage_type in jerk_lookup:
            jerk = jerk_lookup[stage_type]
        else:
            jerk = default_jerk

        profile = MotionProfile(
            name=label,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            jerk_max=jerk,
            settle_time=settle_time,
            color="blue" if stage_type.lower().startswith("lin") else "orange"
        )
        sim = ExperimentSimulator(
            profile=profile,
            scan_length=experiment_config['scan_length'],
            step_size=experiment_config['step_size'],
            num_lines=experiment_config['num_lines'],
            index_time=index_time,
            overhead_per_line=overhead_per_line
        )
        breakdown = sim.compute_time_breakdown()
        labels.append(label)
        stage_types.append(stage_type)
        breakdowns.append(breakdown)
    return labels, stage_types, breakdowns
