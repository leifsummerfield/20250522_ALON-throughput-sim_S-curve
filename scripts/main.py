"""
ALON Throughput Simulation Main Script
======================================

Modular, CLI-driven, and extensible main script for LA-ICP-MS imaging stage throughput analysis.

Major use-cases (selectable via CLI or run all):
    1. Single line scan debugging
    2. Parametric sweeps (velocity, acceleration, 2D)
    3. Vendor/part number table analysis

To add new use-cases, define a new function and add to the USE_CASES dict.
CLI Usage Examples:

# Or simply run without arguments to execute the default single_cycle case
# using settings defined at the top of this file:
# python scripts/main.py

# Run ALL use-cases using the default baseline config
python scripts/main.py --base-dir "C:/Users/lsummerfield/Documents/ESL Data Analytics/20250522_ALON-throughput-sim_S-curve"

# Run only the single-cycle debugging use-case (no sweeps, no vendor table)
python scripts/main.py --base-dir "..." --usecase single_cycle

# Run parametric sweeps with a specific config file
python scripts/main.py --base-dir "..." --usecase parametric --config tv3_highres.json

# Run all use-cases, generate PDF report
python scripts/main.py --base-dir "..." --pdf-report

# Generate PDF report only (assuming previous results/charts already exist)
python scripts/main.py --base-dir "..." --pdf-report --usecase none

# Change logging verbosity
python scripts/main.py --base-dir "..." --loglevel DEBUG

# List available use-cases
python scripts/main.py --help

NOTE:
- All config files should be placed in the config/ directory at the project base.
- All vendor/part tables should be in data/stage_data.csv.
- Results and PDF reports are auto-named and stored in the results/ directory.
"""

import argparse
import logging
import json
from pathlib import Path
import datetime
import sys

import numpy as np
from fpdf import FPDF

from motion_profile import MotionProfile
from simulator import ExperimentSimulator
from analysis import (
    sweep_velocity, sweep_accel, sweep_velocity_accel,
    analyze_stages_from_csv
)
from plotter import (
    plot_velocity_vs_total_time,
    plot_accel_vs_total_time,
    plot_velocity_accel_surface,
    plot_single_cycle_breakdown_bar,
    plot_scurve_velocity_and_accel_vs_time,
    plot_vendor_stages_stacked_bar,
    plot_velocity_vs_position_scurve,
)
# =============================
# DEFAULT RUN SETTINGS
# =============================
DEFAULT_SETTINGS = {
    "base_dir": Path(__file__).resolve().parents[1],
    "config": "tv3_baseline.json",
    "usecase": ["single_cycle"],
    "pdf_report": False,
    "loglevel": "INFO",
}


# =======================
# 1. CONFIGURATION UTILS
# =======================

def load_config(config_path: Path) -> dict:
    """Load and validate a JSON configuration file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"Config file is not valid JSON: {config_path}")
        sys.exit(1)

def setup_dirs(base_dir: Path) -> dict:
    """Create and return all working subdirectories."""
    dirs = {
        "config": base_dir / "config",
        "results": base_dir / "results",
        "parametric": base_dir / "results" / "parametric",
        "single_cycle": base_dir / "results" / "single_cycle",
        "data": base_dir / "data",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

# ==============================
# 2. EXPERIMENT UTILITY FUNCTIONS
# ==============================

def build_stage_profile(config: dict, settle_time, jerk):
    """Create and return a motion profile object based on config."""
    return MotionProfile(
        name="Stage",
        max_velocity=config["stage"]["max_velocity"],
        max_acceleration=config["stage"]["max_acceleration"],
        jerk_max=jerk,
        settle_time=settle_time,
        color="blue"
    )

def print_geometry(config, num_lines):
    print(
        f"Raster geometry: {config['scan_length_mm']} mm (X) x {config['array_height_mm']} mm (Y), "
        f"{num_lines} lines @ {config['step_size_mm']*1000:.1f} um step"
    )

# ==========================
# 3. USE-CASE IMPLEMENTATION
# ==========================

def usecase_single_cycle_debug(config, dirs, params):
    """Run single line scan debugging for the stage (S-curve)."""
    logging.info("Running use-case: Single Line Scan Debugging")
    stage = build_stage_profile(
        config, params['settle_time'], params['jerk']
    )
    scan_length = params['scan_length']
    step_size = params['step_size']
    index_time = params['index_time']
    comms_overhead = params['comms_overhead']

    profiles = [stage]
    simulators = [
        ExperimentSimulator(
            profile=stage,
            scan_length=scan_length,
            step_size=step_size,
            num_lines=1,
            index_time=index_time,
            overhead_per_line=comms_overhead
        )
    ]

    # Print used motion parameters FIRST
    for p in profiles:
        print(f"\n[{p.name} MOTION PARAMETERS]")
        print(f"  Max Velocity     : {p.max_velocity:.4f} mm/s")
        print(f"  Max Acceleration : {p.max_acceleration:.4f} mm/s²")
        print(f"  Max Jerk         : {getattr(p, 'jerk_max', 10000):.4f} mm/s³")
        print(f"  Settle Time      : {p.settle_time:.4f} s")
        print(f"  Color            : {p.color}")

    breakdowns = [sim.compute_time_breakdown() for sim in simulators]
    for p, breakdown in zip(profiles, breakdowns):
        print(f"\n[{p.name} S-curve Phase Timing Breakdown]")
        for k, v in breakdown.items():
            print(f"  {k:12s}: {v:8.4f} sec")
        print(f"  [Total Time]: {breakdown['Total']:.4f} sec\n")


    plot_scurve_velocity_and_accel_vs_time(profiles, simulators, dirs["single_cycle"])
    plot_single_cycle_breakdown_bar(breakdowns, profiles, dirs["single_cycle"])
    plot_velocity_vs_position_scurve(profiles, simulators, dirs["single_cycle"])

def usecase_parametric_sweep(config, dirs, params):
    """
    Run parametric sweeps for velocity, acceleration, and velocity-acceleration surfaces (all S-curve).
    """
    logging.info("Running use-case: Parametric Sweep Analysis")
    settle_time = params['settle_time']
    jerk = params['jerk']
    experiment_config = params['experiment_config']

    vel_range = np.linspace(*config["velocity_sweep"])
    accel_range = np.linspace(*config["acceleration_sweep"])
    v_vals, t_vals = sweep_velocity(
        stage_name="Stage",
        accel=config["stage"]["max_acceleration"],
        jerk=jerk,
        settle_time=settle_time,
        color="blue",
        velocity_range=vel_range,
        experiment_config=experiment_config
    )
    plot_velocity_vs_total_time(
        [v_vals], [t_vals],
        ["Stage"],
        ["blue"],
        dirs["parametric"]
    )

    accel_range = np.linspace(*config["acceleration_sweep"])
    a_vals, t_acc_vals = sweep_accel(
        stage_name="Stage",
        vmax=config["stage"]["max_velocity"],
        jerk=jerk,
        settle_time=settle_time,
        color="blue",
        accel_range=accel_range,
        experiment_config=experiment_config
    )
    plot_accel_vs_total_time(
        [a_vals], [t_acc_vals],
        ["Stage"],
        ["blue"],
        dirs["parametric"]
    )

    # 2D surface
    v_2d = np.linspace(*config["velocity_2d_sweep"])
    a_2d = np.linspace(*config["acceleration_2d_sweep"])
    V, A, Z = sweep_velocity_accel(
        stage_name="Stage",
        jerk=jerk,
        settle_time=settle_time,
        color="blue",
        velocity_range=v_2d,
        accel_range=a_2d,
        experiment_config=experiment_config
    )
    # Optionally, uncomment for 2D surface plots
    # plot_velocity_accel_surface(V, A, Z, "Stage", dirs["parametric"], "velocity_accel_surface.png")

def usecase_vendor_table(config, dirs, params):
    """
    Analyze and plot stacked bar chart for vendor stages (all S-curve).
    """
    logging.info("Running use-case: Vendor Table/Part Number Analysis")
    csv_path = dirs["data"] / "stage_data.csv"
    settle_time_lookup = None
    index_time_lookup = None
    jerk_lookup = None
    labels, stage_types, breakdowns = analyze_stages_from_csv(
        csv_path,
        params['experiment_config'],
        index_time_lookup=index_time_lookup,
        settle_time_lookup=settle_time_lookup,
        overhead_per_line=params['comms_overhead'],
        jerk_lookup=jerk_lookup,
        default_jerk=params['jerk']
    )
    plot_vendor_stages_stacked_bar(labels, stage_types, breakdowns, dirs["parametric"])

# Register use-cases
USE_CASES = {
    "single_cycle": usecase_single_cycle_debug,
    "parametric": usecase_parametric_sweep,
    "vendor_table": usecase_vendor_table,
}

# ========================
# 4. REPORT GENERATION
# ========================

def generate_pdf_report(
    parameters_dict,
    charts_dirs,
    output_pdf_path,
    notes=None,
):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "LA-ICP-MS Throughput Simulation Report", ln=True, align="C")
    pdf.ln(2)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 10, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Experiment Parameters:", ln=True)
    pdf.set_font("Arial", size=11)
    for k, v in parameters_dict.items():
        pdf.cell(0, 7, f"{k}: {v}", ln=True)
    if notes:
        pdf.ln(4)
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 7, "Experiment Notes:", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 7, notes)
    chart_files = []
    for charts_dir in charts_dirs:
        for img in sorted(Path(charts_dir).glob("*.png")):
            chart_files.append(img)
    for img_path in chart_files:
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        caption = img_path.stem.replace("_", " ").capitalize()
        pdf.cell(0, 10, caption, ln=True, align="C")
        pdf.image(str(img_path), x=15, w=180)
        pdf.ln(3)
    pdf.output(str(output_pdf_path))
    print(f"PDF report saved to: {output_pdf_path}")

# ================
# 5. MAIN ENTRYPOINT
# ================

def main():
    parser = argparse.ArgumentParser(
        description="ALON Throughput Simulation — modular, multi-use-case main script"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(DEFAULT_SETTINGS['base_dir']),
        help="Project base directory (e.g., C:/Users/lsummerfield/Documents/ESL Data Analytics/20250522_ALON-throughput-sim_S-curve)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_SETTINGS["config"],
        help="Name of config file to load from the config/ directory"
    )
    parser.add_argument(
        "--usecase",
        type=str,
        nargs="*",
        choices=list(USE_CASES.keys()) + ["all"],
        default=DEFAULT_SETTINGS["usecase"],
        help="Which use-case(s) to run (default: all)"
    )
    parser.add_argument(
        "--pdf-report",
        action="store_true", default=DEFAULT_SETTINGS["pdf_report"],
        help="Generate PDF report from latest results"
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default=DEFAULT_SETTINGS["loglevel"],
        help="Set logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel.upper()), format="%(levelname)s: %(message)s")
    base_dir = Path(args.base_dir)
    dirs = setup_dirs(base_dir)
    config_path = dirs["config"] / args.config
    config = load_config(config_path)

    # --- Derived parameters ---
    scan_length = config["scan_length_mm"]
    step_size = config["step_size_mm"]
    array_height = config["array_height_mm"]
    index_time = config["index_time_s"]
    comms_overhead = config["comms_overhead_s"]
    settle_time = config["settle_time_s"]
    jerk = config["stage"].get("jerk_max", 10000)
    num_lines = int(array_height / step_size) + 1

    print_geometry(config, num_lines)

    experiment_config = dict(
        scan_length=scan_length,
        step_size=step_size,
        num_lines=num_lines,
        index_time=index_time,
        overhead_per_line=comms_overhead
    )
    params = dict(
        scan_length=scan_length,
        step_size=step_size,
        array_height=array_height,
        index_time=index_time,
        comms_overhead=comms_overhead,
        settle_time=settle_time,
        jerk=jerk,
        num_lines=num_lines,
        experiment_config=experiment_config
    )

    if "all" in args.usecase:
        run_cases = USE_CASES.values()
    else:
        run_cases = [USE_CASES[uc] for uc in args.usecase]
    for fn in run_cases:
        fn(config, dirs, params)

    if args.pdf_report:
        parameters_dict = {
            "Config File": str(config_path),
            "Scan Length (mm)": scan_length,
            "Step Size (mm)": step_size,
            "Array Height (mm)": array_height,
            "Number of Lines": num_lines,
            "Index Time (s)": index_time,
            "Settle Time (s)": settle_time,
            "Jerk (mm/s³)": jerk,
            "Comms Overhead (s)": comms_overhead,
        }
        parameters_dict.update({
            "Velocity Sweep Range (mm/s)": f"{config['velocity_sweep'][0]} to {config['velocity_sweep'][1]}",
            "Acceleration Sweep Range (mm/s^2)": f"{config['acceleration_sweep'][0]} to {config['acceleration_sweep'][1]}",
            "Vendor Table Source": str(dirs["data"] / "stage_data.csv"),
        })
        config_name = config.get("name", "experiment")
        generate_pdf_report(
            parameters_dict=parameters_dict,
            charts_dirs=[dirs["single_cycle"], dirs["parametric"]],
            output_pdf_path=dirs["results"] / f"{config_name}_experiment_report.pdf",
            notes=f"Config: {config_name} | Automated batch run."
        )

if __name__ == "__main__":
    main()
