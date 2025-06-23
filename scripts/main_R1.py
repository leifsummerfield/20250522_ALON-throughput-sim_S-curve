# scripts/main.py

"""
ALON Throughput Simulation Main Script
======================================

This script drives all major analysis and plotting for LA-ICP-MS imaging stage throughput.
It is divided into three main use-cases:
    1. Debugging a single line scan (velocity/phase time series and breakdown)
    2. Parametric sweeps for design decisions (velocity, acceleration, and 2D)
    3. Vendor/part number table analysis (stacked bar chart for performance trade-off study)

You can use each section independently, but all share a common set of base experiment parameters.
"""

from pathlib import Path
import numpy as np
from fpdf import FPDF
from pathlib import Path
import datetime
import json

# --- Core project imports (these files must exist in scripts/) ---
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
    plot_single_cycle,
    plot_single_cycle_breakdown_bar,
    plot_vendor_stages_stacked_bar,
    plot_velocity_vs_position,
    plot_velocity_vs_time
)

# ========== GLOBAL EXPERIMENT PARAMETERS ==========
# Set these to match your imaging geometry and controller timing

# Base folder for all results/plots
base_dir = Path(r"C:\Users\lsummerfield\Documents\ESL Data Analytics\20250522_ALON throughput sim")
config_dir = base_dir / "config"
results_dir = base_dir / 'results'
param_dir = results_dir / 'parametric'
single_dir = results_dir / 'single_cycle'
results_dir.mkdir(parents=True, exist_ok=True)
param_dir.mkdir(parents=True, exist_ok=True)
single_dir.mkdir(parents=True, exist_ok=True)


# --- Choose config file to use ---
config_path = config_dir / "tv3_baseline.json"  # Change this to run a different config
with open(config_path, "r") as f:
    config = json.load(f)

# --- Core geometry and timing ---
# (Set these before any use case!)
scan_length = 10.0    # mm, length of scan line (X axis)
step_size = 0.005     # mm, step size between lines (Y axis) -- e.g., 5um
array_height = 5.0    # mm, total raster height (Y)
num_lines = int(array_height / step_size) + 1  # integer number of scan lines to cover height
index_time = 0.02     # s, time to move between lines (should match controller and mechanics)
comms_overhead = 0.050 # s, per line, communication/controller overhead

# --- Stage-specific timing (edit as needed per your mechanics) ---
settle_linear = 0.100   # s, stage settle time for linear motor
settle_stepper = 0.050  # s, stage settle time for stepper


scan_length = config["scan_length_mm"]
step_size = config["step_size_mm"]
array_height = config["array_height_mm"]
index_time = config["index_time_s"]
settle_linear = config["settle_linear_s"]
settle_stepper = config["settle_stepper_s"]
comms_overhead = config["comms_overhead_s"]

num_lines = int(array_height / step_size) + 1


linear_motor = MotionProfile(
    name="Linear Motor",
    max_velocity=config["linear_motor"]["max_velocity"],
    max_acceleration=config["linear_motor"]["max_acceleration"],
    settle_time=settle_linear,
    color="blue"
)
stepper_motor = MotionProfile(
    name="Stepper Motor",
    max_velocity=config["stepper_motor"]["max_velocity"],
    max_acceleration=config["stepper_motor"]["max_acceleration"],
    settle_time=settle_stepper,
    color="orange"
)


vel_range = np.linspace(*config["velocity_sweep"])  # [start, stop, num]
accel_range = np.linspace(*config["acceleration_sweep"])



# --- Pack experiment config for easy passing ---
experiment_config = dict(
    scan_length=scan_length,
    step_size=step_size,
    num_lines=num_lines,
    index_time=index_time,
    overhead_per_line=comms_overhead
)

print(f"Raster geometry: {scan_length} mm (X) x {array_height} mm (Y), "
      f"{num_lines} lines @ {step_size*1000:.1f} um step")

# ======================================
# == 1. SINGLE LINE SCAN DEBUGGING    ==
# ======================================
# Focus: One "cycle" (scan + index + settle + comms) for both architectures.
#        - Use for hardware validation, motion model debugging, phase breakdown understanding.

# --- Canonical stage profiles for single-cycle debugging ---
linear_motor = MotionProfile(
    name="Linear Motor",
    max_velocity=100,       # mm/s (set to typical system value)
    max_acceleration=2000,  # mm/s^2 (set to typical for linear)
    settle_time=settle_linear,
    color="blue"
)
stepper_motor = MotionProfile(
    name="Stepper Motor",
    max_velocity=50,        # mm/s (typical for stepper)
    max_acceleration=8000,  # mm/s^2 (stepper, often much higher)
    settle_time=settle_stepper,
    color="orange"
)

# --- Instantiate simulators for just 1 line scan (makes cycle plots clear) ---
profiles = [linear_motor, stepper_motor]
simulators = [
    ExperimentSimulator(
        profile=linear_motor,
        scan_length=scan_length,
        step_size=step_size,
        num_lines=1,              # Only a single scan for this mode
        index_time=index_time,
        overhead_per_line=comms_overhead
    ),
    ExperimentSimulator(
        profile=stepper_motor,
        scan_length=scan_length,
        step_size=step_size,
        num_lines=1,
        index_time=index_time,
        overhead_per_line=comms_overhead
    )
]

# --- Phase breakdown for each single scan cycle ---
breakdowns = [sim.compute_time_breakdown() for sim in simulators]

# --- Plot velocity/time for one cycle ---
plot_velocity_vs_time(profiles, simulators, single_dir)
plot_velocity_vs_position(profiles, simulators, single_dir)

# --- Plot stacked bar chart of phase times for single scan ---
plot_single_cycle_breakdown_bar(breakdowns, profiles, single_dir)

# =======================================================================
# == 2. PARAMETRIC SWEEP ANALYSIS (velocity, acceleration, and surfaces) ==
# =======================================================================
# Focus: How does total imaging time depend on max velocity and acceleration?
#        - Use to explore design space, find "diminishing returns," and compare architectures.

# --- Define sweep ranges (EDIT to match hardware capabilities) ---
vel_range = np.linspace(10, 400, 25)         # mm/s, min to max, typical ranges
accel_linear = 2000                          # mm/s^2 (fix for linear sweep)
accel_stepper = 8000                         # mm/s^2 (fix for stepper sweep)

# --- Sweep max velocity for both architectures ---
v_lin, t_lin = sweep_velocity(
    stage_name="Linear Motor",
    accel=accel_linear,
    settle_time=settle_linear,
    color='blue',
    velocity_range=vel_range,
    experiment_config=experiment_config
)
v_step, t_step = sweep_velocity(
    stage_name="Stepper Motor",
    accel=accel_stepper,
    settle_time=settle_stepper,
    color='orange',
    velocity_range=vel_range,
    experiment_config=experiment_config
)

# --- Plot velocity vs. total experiment time for both stage types ---
plot_velocity_vs_total_time(
    [v_lin, v_step], [t_lin, t_step], ["Linear Motor", "Stepper Motor"], ["blue", "orange"], param_dir
)

# --- Sweep acceleration for both (velocity fixed) ---
accel_range = np.linspace(500, 12000, 30)       # mm/s^2, reasonable bracket for typical hardware
fixed_velocity_linear = 300                     # mm/s, fixed for linear
fixed_velocity_stepper = 50                     # mm/s, fixed for stepper

a_lin, t_acc_lin = sweep_accel(
    stage_name="Linear Motor",
    vmax=fixed_velocity_linear,
    settle_time=settle_linear,
    color='blue',
    accel_range=accel_range,
    experiment_config=experiment_config
)
a_step, t_acc_step = sweep_accel(
    stage_name="Stepper Motor",
    vmax=fixed_velocity_stepper,
    settle_time=settle_stepper,
    color='orange',
    accel_range=accel_range,
    experiment_config=experiment_config
)

plot_accel_vs_total_time(
    [a_lin, a_step], [t_acc_lin, t_acc_step], ["Linear Motor", "Stepper Motor"], ["blue", "orange"], param_dir
)

# --- 2D surface sweep: velocity vs acceleration for each architecture ---
v_2d = np.linspace(10, 400, 20)
a_2d = np.linspace(500, 12000, 20)

V_lin, A_lin, Z_lin = sweep_velocity_accel(
    stage_name="Linear Motor",
    settle_time=settle_linear,
    color='blue',
    velocity_range=v_2d,
    accel_range=a_2d,
    experiment_config=experiment_config
)
V_step, A_step, Z_step = sweep_velocity_accel(
    stage_name="Stepper Motor",
    settle_time=settle_stepper,
    color='orange',
    velocity_range=v_2d,
    accel_range=a_2d,
    experiment_config=experiment_config
)

#plot_velocity_accel_surface(V_lin, A_lin, Z_lin, "Linear Motor", param_dir, "velocity_accel_surface_linear.png")
#plot_velocity_accel_surface(V_step, A_step, Z_step, "Stepper Motor", param_dir, "velocity_accel_surface_stepper.png")

# ===========================================================
# == 3. VENDOR TABLE/PART NUMBER ANALYSIS (Trade Study)    ==
# ===========================================================
# Focus: Side-by-side stacked bar chart comparing actual stages from multiple vendors/part numbers.
#        - Useful for real-world trade studies, purchasing, and reporting.

# --- CSV file of vendor stage parameters ---
# Make sure your CSV contains these columns (case-sensitive):
#   "Stage Type", "Supplier and PN", "Speed (mm/s)", "Acceleration (m/s^2)"
csv_path = base_dir / "data" / "stage_data.csv"

# --- Provide group timing if you want to override for a given Stage Type ---
settle_time_lookup = {
    "Linear": settle_linear,    # Reuse from global above
    "Stepper": settle_stepper
}
index_time_lookup = {
    "Linear": index_time,       # Use the same index_time for both, or split if needed
    "Stepper": index_time
}

# --- Run simulation for every row in the CSV ---
labels, stage_types, breakdowns = analyze_stages_from_csv(
    csv_path,
    experiment_config,
    index_time_lookup=index_time_lookup,
    settle_time_lookup=settle_time_lookup,
    overhead_per_line=comms_overhead
)

# --- Plot: Each bar is a vendor/part number, sorted fastest to slowest, phase colored ---
plot_vendor_stages_stacked_bar(labels, stage_types, breakdowns, param_dir)

# ===================
# === END OF SCRIPT ===
# ===================
"""
CRITICAL VARIABLES TO EDIT/SET:
    - scan_length, step_size, array_height: define your raster geometry
    - index_time, comms_overhead: define your system's timing, from software/hardware
    - settle_linear, settle_stepper: edit for your hardware (may depend on load, controller, etc.)
    - Canonical linear and stepper velocities/accels for parametric sweeps (see above)
    - CSV file path and columns for vendor/part number study

To add another use case or plot, just build on the existing code sections!
"""

def generate_pdf_report(
    parameters_dict,
    charts_dirs,
    output_pdf_path,
    settings_path=None,
    notes=None,
):
    """
    Generate a PDF report:
        - Page 1: experiment parameters (from dict or settings.txt)
        - Following pages: all .png files found in the given directories, each with a caption
    
    Args:
        parameters_dict: dict of experiment parameters (can be empty if settings_path is given)
        charts_dirs: list of Path objects to directories where charts are stored
        output_pdf_path: Path to PDF output file
        settings_path: (optional) Path to settings.txt if you prefer reading parameters from file
        notes: (optional) String with any comments or extra description for the run
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ===== PAGE 1: PARAMETERS =====
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "LA-ICP-MS Throughput Simulation Report", ln=True, align="C")
    pdf.ln(2)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 10, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    
    # If settings.txt file provided, use it for this section
    if settings_path and Path(settings_path).exists():
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Experiment Parameters (from settings.txt):", ln=True)
        pdf.set_font("Arial", size=11)
        with open(settings_path, 'r') as f:
            for line in f:
                pdf.cell(0, 7, line.strip(), ln=True)
    elif parameters_dict:
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Experiment Parameters:", ln=True)
        pdf.set_font("Arial", size=11)
        for k, v in parameters_dict.items():
            pdf.cell(0, 7, f"{k}: {v}", ln=True)
    else:
        pdf.set_font("Arial", "I", 11)
        pdf.cell(0, 8, "No experiment parameters provided.", ln=True)
    if notes:
        pdf.ln(4)
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 7, "Experiment Notes:", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 7, notes)
    
    # ===== PAGE 2+: CHARTS =====
    # Gather all .png files from the specified chart directories
    chart_files = []
    for charts_dir in charts_dirs:
        for img in sorted(Path(charts_dir).glob("*.png")):
            chart_files.append(img)
    
    for img_path in chart_files:
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        # Use stem as caption, improve readability
        caption = img_path.stem.replace("_", " ").capitalize()
        pdf.cell(0, 10, caption, ln=True, align="C")
        # Add image (resize to fit nicely on page)
        pdf.image(str(img_path), x=15, w=180)  # Adjust x/w as needed
        pdf.ln(3)
    
    pdf.output(str(output_pdf_path))
    print(f"PDF report saved to: {output_pdf_path}")

parameters_dict = {
    "Config File " : str(config_path), 
    "Scan Length (mm)": scan_length,
    "Step Size (mm)": step_size,
    "Array Height (mm)": array_height,
    "Number of Lines": num_lines,
    "Index Time (s)": index_time,
    "Settle Time (Linear, s)": settle_linear,
    "Settle Time (Stepper, s)": settle_stepper,
    "Comms Overhead (s)": comms_overhead,
    "Canonical Max Velocity (Linear, mm/s)": linear_motor.max_velocity if 'linear_motor' in locals() else None,
    "Canonical Max Velocity (Stepper, mm/s)": stepper_motor.max_velocity if 'stepper_motor' in locals() else None,
    "Canonical Acceleration (Linear, mm/s^2)": linear_motor.max_acceleration if 'linear_motor' in locals() else None,
    "Canonical Acceleration (Stepper, mm/s^2)": stepper_motor.max_acceleration if 'stepper_motor' in locals() else None,
    # Add other key parameters if you introduce more variables!
}

parameters_dict.update({
    "Velocity Sweep Range (mm/s)": f"{vel_range[0]} to {vel_range[-1]}",
    "Acceleration Sweep Range (mm/s^2)": f"{accel_range[0]} to {accel_range[-1]}",
    "Vendor Table Source": str(csv_path) if 'csv_path' in locals() else "N/A"
})


if __name__ == "__main__":
    config_name = config.get("name", "experiment")
    generate_pdf_report(
        parameters_dict=parameters_dict,
        charts_dirs=[single_dir, param_dir],
        output_pdf_path=results_dir / f"{config_name}_experiment_report.pdf",
        notes=f"Config: {config_name} | Automated batch run."
    )
