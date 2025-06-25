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
    """Plot timeline-style breakdown for a single line scan cycle."""
    phase_labels = ["Acceleration", "Constant", "Deceleration", "Index", "Settle", "Overhead"]
    phase_colors = [
        "#1976D2", "#388E3C", "#FBC02D", "#E64A19", "#7B1FA2", "#616161"
    ]

    profile_labels = [p.name for p in profiles]
    fig, ax = plt.subplots(figsize=(10, 2 + len(profiles)))

    max_total = max(bd["Total"] for bd in breakdowns)

    for row, (label, bd) in enumerate(zip(profile_labels, breakdowns)):
        start = 0.0
        annot_positions = []
        for phase, color in zip(phase_labels, phase_colors):
            duration = bd[phase]
            ax.barh(row, duration, left=start, height=0.5, color=color)

            x_center = start + duration / 2
            base_y = row + 0.6
            offset_count = sum(
                1 for pos in annot_positions if abs(x_center - pos) < 0.05 * max_total
            )
            y_text = base_y + 0.3 * offset_count
            ax.annotate(
                f"{phase}\n{duration:.2f}s",
                xy=(x_center, row),
                xytext=(x_center, y_text),
                textcoords="data",
                ha="center",
                va="bottom",
                fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.5, color="black"),
            )
            annot_positions.append(x_center)
            start += duration

        ax.text(start + max_total * 0.02, row, f"{bd['Total']:.2f}s", va="center", ha="left", fontweight="bold")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in phase_colors]
    ax.legend(handles, phase_labels, title="Phase", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xlabel("Time (s) for One Line Scan Cycle")
    ax.set_xlim(0, max_total * 1.2)
    ax.set_yticks(np.arange(len(profile_labels)))
    ax.set_yticklabels(profile_labels)
    ax.set_title("Single Line Scan Cycle Time Breakdown (Timeline)")
    plt.tight_layout()
    plt.savefig(out_dir / "single_cycle_time_breakdown_timeline.png")
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

        phase_arr = np.array(phase)
        accel_end_idx = np.where(phase_arr == "const")[0][0]
        decel_start_idx = np.where(phase_arr == "decel")[0][0]

        span_start = x[accel_end_idx]
        span_end = x[decel_start_idx]

        plt.axvspan(span_start, span_end, color=p.color, alpha=0.1,
                    label=f"{p.name} scan region")
        plt.scatter([span_start, span_end],
                    [v[accel_end_idx], v[decel_start_idx]],
                    color='black', zorder=5)
        plt.text(span_start, max(v) * 1.03, "Scan Start",
                 ha='left', va='bottom', color=p.color)
        plt.text(span_end, max(v) * 1.03, "Scan End",
                 ha='right', va='bottom', color=p.color)
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
