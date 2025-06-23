import matplotlib.pyplot as plt
import numpy as np

def plot_velocity_vs_total_time(velocities_list, times_list, labels, colors, out_dir):
    plt.figure(figsize=(10, 6))
    for v, t, label, color in zip(velocities_list, times_list, labels, colors):
        plt.plot(v, t, label=label, color=color, marker='o')
    plt.xlabel("Max Velocity (mm/s)")
    plt.ylabel("Total Experiment Time (s)")
    plt.title("Velocity vs. Total Experiment Time")
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(out_dir / "velocity_vs_time.png")
    plt.show()

def plot_accel_vs_total_time(accels_list, times_list, labels, colors, out_dir):
    plt.figure(figsize=(10, 6))
    for a, t, label, color in zip(accels_list, times_list, labels, colors):
        plt.plot(a, t, label=label, color=color, marker='o')
    plt.xlabel("Max Acceleration (mm/s²)")
    plt.ylabel("Total Experiment Time (s)")
    plt.title("Acceleration vs. Total Experiment Time")
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(out_dir / "acceleration_vs_time.png")
    plt.show()

def plot_velocity_accel_surface(V, A, Z, label, out_dir, fname):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(V, A, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_xlabel('Max Velocity (mm/s)')
    ax.set_ylabel('Max Acceleration (mm/s²)')
    ax.set_zlabel('Total Time (s)')
    ax.set_title(f'Total Time vs Velocity and Acceleration ({label})')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(out_dir / fname)
    plt.show()

def plot_single_cycle_breakdown_bar(breakdowns, profiles, out_dir):
    phase_labels = ["Acceleration", "Constant", "Deceleration", "Index", "Settle", "Overhead"]
    phase_colors = [
        "#1976D2", "#388E3C", "#FBC02D", "#E64A19", "#7B1FA2", "#616161"
    ]
    profile_labels = [p.name for p in profiles]
    phase_values = []
    for breakdown in breakdowns:
        phase_values.append([breakdown[ph] for ph in phase_labels])
    phase_values = np.array(phase_values)
    fig, ax = plt.subplots(figsize=(9, 6))
    ind = np.arange(len(profiles))
    bottom = np.zeros(len(profiles))
    for i, (phase, color) in enumerate(zip(phase_labels, phase_colors)):
        vals = phase_values[:, i]
        ax.bar(ind, vals, bottom=bottom, label=phase, color=color)
        bottom += vals
    # Annotate total time
    for idx, breakdown in enumerate(breakdowns):
        total = breakdown['Total']
        ax.text(idx, total + total*0.01, f"{total:.2f}s", ha='center', va='bottom', fontweight='bold')
    ax.set_xticks(ind)
    ax.set_xticklabels(profile_labels)
    ax.set_ylabel("Time (s) for One Line Scan Cycle")
    ax.set_title("Single Line Scan Cycle Time Breakdown (Stacked Bar)")
    ax.legend(title="Phase", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / "single_cycle_time_breakdown_stacked_bar.png")
    plt.show()

def plot_vendor_stages_stacked_bar(labels, stage_types, breakdowns, out_dir):
    phase_labels = ["Acceleration", "Constant", "Deceleration", "Index", "Settle", "Overhead"]
    phase_colors = [
        "#1976D2", "#388E3C", "#FBC02D", "#E64A19", "#7B1FA2", "#616161"
    ]
    bar_edge_colors = ["blue" if t.lower().startswith("lin") else "orange" for t in stage_types]
    phase_values = []
    for breakdown in breakdowns:
        phase_values.append([breakdown[ph] for ph in phase_labels])
    phase_values = np.array(phase_values)
    totals = [b['Total'] for b in breakdowns]
    sort_idx = np.argsort(totals)
    labels = [labels[i] for i in sort_idx]
    bar_edge_colors = [bar_edge_colors[i] for i in sort_idx]
    phase_values = phase_values[sort_idx, :]
    totals = [totals[i] for i in sort_idx]
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.7), 7))
    ind = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for i, (phase, color) in enumerate(zip(phase_labels, phase_colors)):
        vals = phase_values[:, i]
        bars = ax.bar(ind, vals, bottom=bottom, label=phase, color=color, edgecolor=bar_edge_colors, linewidth=2)
        bottom += vals
    for idx, total in enumerate(totals):
        ax.text(idx, total + total*0.01, f"{total:.1f}s", ha='center', va='bottom', fontweight='bold', rotation=90)
    ax.set_xticks(ind)
    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=8)
    ax.set_ylabel("Total Time (s) for Full Raster")
    ax.set_title("Vendor Stage Performance Comparison: Time Breakdown by Phase")
    ax.legend(title="Phase", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / "vendor_stages_time_breakdown.png")
    plt.show()

def plot_velocity_vs_position_scurve(profiles, simulators, out_dir):
    plt.figure(figsize=(10, 6))
    for p, sim in zip(profiles, simulators):
        t, x, v, a, phase = sim.simulate_scurve_scan_with_backoff()
        plt.plot(x, v, label=p.name, color=p.color, lw=2)
        scan_L = sim.scan_length
        plt.axvspan(x[0], x[0]+scan_L, color=p.color, alpha=0.1, label=f"{p.name} scan region")
        idx_accel_end = np.where(x >= x[0])[0][0]
        idx_const_end = np.where(x >= x[0]+scan_L)[0][0]
        plt.scatter([x[idx_accel_end], x[idx_const_end]],
                    [v[idx_accel_end], v[idx_const_end]],
                    color='black', zorder=5)
        plt.text(x[0], max(v)*1.03, "Scan Start", ha='left', va='bottom', color=p.color)
        plt.text(x[0]+scan_L, max(v)*1.03, "Scan End", ha='right', va='bottom', color=p.color)
    plt.xlabel("Stage Position (mm)")
    plt.ylabel("Velocity (mm/s)")
    plt.title("Velocity vs. Position (S-curve, Accel/Const/Decel) for Scan with Backoff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "velocity_vs_position_with_backoff.png")
    plt.show()

def plot_scurve_velocity_and_accel_vs_time(profiles, simulators, out_dir):
    plt.figure(figsize=(10, 6))
    for p, sim in zip(profiles, simulators):
        t, x, v, a, phase = sim.simulate_scurve_scan_with_backoff()
        plt.plot(t, v, label=f"{p.name} velocity", lw=2)
        plt.plot(t, a, label=f"{p.name} accel", lw=1, ls="--")
        # Annotate phase boundaries if desired:
        accel_end = np.where(np.array(phase) == "const")[0][0]
        decel_start = np.where(np.array(phase) == "decel")[0][0]
        plt.axvline(t[accel_end], color=p.color, ls=':', lw=1, alpha=0.7)
        plt.axvline(t[decel_start], color=p.color, ls=':', lw=1, alpha=0.7)
        # Annotate text
        plt.text(t[accel_end]/2, np.max(v)*0.8, "accel", color=p.color, ha="center", va="bottom")
        plt.text((t[accel_end]+t[decel_start])/2, np.max(v)*0.9, "const", color=p.color, ha="center", va="bottom")
        plt.text((t[decel_start]+t[-1])/2, np.max(v)*0.8, "decel", color=p.color, ha="center", va="bottom")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity / Acceleration (mm/s, mm/s²)")
    plt.title("S-curve Velocity and Acceleration vs. Time (Full Scan with Backoff)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "scurve_velocity_accel_vs_time.png")
    plt.show()
