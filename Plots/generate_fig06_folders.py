"""Generate Fig. 6: Monochrome repository folder tree and per-set HDF5 schema.

Outputs (saved alongside this script in Plots/):
  - fig06_folders.pdf  (vector PDF for Overleaf / Nature Scientific Data)
  - fig06_folders.png  (300 DPI preview PNG)

Usage:
  python3 Plots/generate_fig06_folders.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Strict monochrome (black, white, grayscale) journal palette
C_WHITE = "#FFFFFF"
C_BORDER = "#333333"
C_LINE = "#555555"
C_TEXT = "#111111"
C_SUBTEXT = "#4A4A4A"

# (facecolor, edgecolor, textcolor, label)
BADGE_STYLES = {
    "dir": ("#222222", "#222222", "#FFFFFF", "DIR"),
    "grp": ("#222222", "#222222", "#FFFFFF", "GRP"),
    "h5": ("#595959", "#222222", "#FFFFFF", "HDF5"),
    "attr": ("#BFBFBF", "#333333", "#111111", "ATTR"),
    "csv": ("#E5E5E5", "#333333", "#111111", "CSV"),
    "ds": ("#F2F2F2", "#444444", "#111111", "DATA"),
    "mp4": ("#D4D4D4", "#333333", "#111111", "MP4"),
    "doc": ("#FFFFFF", "#555555", "#222222", "TXT"),
}


def draw_badge(ax, x: float, y: float, kind: str, w: float = 0.086, h: float = 0.030):
    bg, ec, fg, label = BADGE_STYLES[kind]
    box = Rectangle(
        (x, y - h / 2),
        w,
        h,
        facecolor=bg,
        edgecolor=ec,
        linewidth=0.7,
        zorder=4,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y,
        label,
        ha="center",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color=fg,
        family="serif",
        zorder=5,
    )


def draw_tree(ax, nodes: list[dict], x_root: float = 0.04, y_top: float = 0.895, dy: float = 0.066):
    """Draw an orthogonal file/HDF5 tree inside `ax` (coordinates in [0, 1])."""
    indent = 0.062
    badge_w = 0.086
    positions: dict[int, tuple[float, float]] = {}

    y_cur = y_top
    for i, node in enumerate(nodes):
        lvl = node["level"]
        x = x_root + lvl * indent
        if i > 0:
            y_cur -= dy + node.get("extra_dy_before", 0.0)
        y = y_cur
        positions[i] = (x, y)

        # Orthogonal connector from nearest preceding parent (level - 1)
        if lvl > 0:
            parent_idx = max(j for j in range(i) if nodes[j]["level"] == lvl - 1)
            px, py = positions[parent_idx]
            branch_x = px + 0.022
            ax.plot([branch_x, branch_x], [py - 0.016, y], color=C_LINE, lw=0.8, zorder=2)
            ax.plot([branch_x, x], [y, y], color=C_LINE, lw=0.8, zorder=2)

        # Subtle monochrome highlight box for the expanded HDF5 file
        if node.get("highlight"):
            hl = Rectangle(
                (x - 0.008, y - 0.022),
                0.970 - x,
                0.044,
                facecolor="#F3F3F3",
                edgecolor="#555555",
                linewidth=0.8,
                linestyle="--",
                zorder=3,
            )
            ax.add_patch(hl)

        # Type badge
        if node.get("kind"):
            draw_badge(ax, x, y, node["kind"], w=badge_w, h=0.030)
            text_x = x + badge_w + 0.016
        else:
            text_x = x + 0.005

        # Main label
        weight = "bold" if node.get("bold") or node.get("kind") in {"dir", "grp"} else "normal"
        ax.text(
            text_x,
            y + (0.005 if node.get("sub") else 0.0),
            node["name"],
            ha="left",
            va="center",
            fontsize=7.8,
            fontweight=weight,
            color=C_TEXT,
            family="monospace",
            zorder=5,
        )

        # Right-aligned description
        if node.get("note"):
            ax.text(
                0.958,
                y,
                node["note"],
                ha="right",
                va="center",
                fontsize=7.6,
                color=C_SUBTEXT,
                family="serif",
                zorder=5,
            )

        # Multi-line field list below group name
        if node.get("sub"):
            ax.text(
                text_x,
                y - 0.016,
                node["sub"],
                ha="left",
                va="top",
                fontsize=7.0,
                color=C_SUBTEXT,
                family="serif",
                linespacing=1.25,
                zorder=5,
            )

    return positions


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
            "font.monospace": ["Liberation Mono", "DejaVu Sans Mono", "Courier New"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(11.4, 6.2),
        gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.06},
    )
    fig.patch.set_facecolor(C_WHITE)

    for ax in (ax_a, ax_b):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        panel_box = Rectangle(
            (0.01, 0.02),
            0.98,
            0.915,
            facecolor=C_WHITE,
            edgecolor=C_BORDER,
            linewidth=0.9,
            zorder=1,
        )
        ax.add_patch(panel_box)

    # Centered serif panel titles (Nature Scientific Data style)
    ax_a.text(
        0.50,
        0.964,
        "Dataset Directory Structure",
        fontsize=11.0,
        fontweight="bold",
        color=C_TEXT,
        family="serif",
        va="center",
        ha="center",
    )
    ax_b.text(
        0.50,
        0.964,
        "Per-Set HDF5 Schema (sets/set_XXXX.h5)",
        fontsize=11.0,
        fontweight="bold",
        color=C_TEXT,
        family="serif",
        va="center",
        ha="center",
    )

    # Panel (a): Repository folder tree (plain descriptions, no specs/counts)
    repo_nodes = [
        {"level": 0, "kind": "dir", "name": "Clean&JerkDataset/", "note": "Repository root", "bold": True},
        {"level": 1, "kind": "doc", "name": "README.md", "note": "Quick Guide"},
        {"level": 1, "kind": "doc", "name": "checksums.sha256", "note": "File integrity manifest"},
        {"level": 1, "kind": "csv", "name": "data_dictionary.csv", "note": "Variable definitions"},
        {"level": 1, "kind": "csv", "name": "participant_summary.csv", "note": "Participant demographics"},
        {"level": 1, "kind": "csv", "name": "sets_summary.csv", "note": "Recording set metadata"},
        {"level": 1, "kind": "csv", "name": "device_summary.csv", "note": "Sensor placement map"},
        {"level": 1, "kind": "csv", "name": "phase_markers_summary.csv", "note": "Consensus phase markers"},
        {"level": 1, "kind": "dir", "name": "sets/", "note": "Per-set data archives"},
        {
            "level": 2,
            "kind": "h5",
            "name": "set_0001.h5 ... set_0054.h5",
            "note": "Expanded on right",
            "highlight": True,
            "bold": True,
        },
        {"level": 1, "kind": "dir", "name": "videos/", "note": "Synchronized videos"},
        {"level": 2, "kind": "mp4", "name": "set_XXXX_front.mp4", "note": "Frontal view"},
        {"level": 2, "kind": "mp4", "name": "set_XXXX_side.mp4", "note": "Sagittal view"},
    ]
    draw_tree(ax_a, repo_nodes, x_root=0.04, y_top=0.892, dy=0.068)

    # Panel (b): HDF5 schema of one set file (uniform 2-3 word descriptions)
    h5_nodes = [
        {"level": 0, "kind": "h5", "name": "set_XXXX.h5", "note": "Per-set HDF5 archive", "bold": True},
        {"level": 1, "kind": "attr", "name": "/metadata", "note": "Set recording metadata"},
        {"level": 1, "kind": "attr", "name": "/participant_metadata", "note": "Participant profile metadata"},
        {"level": 1, "kind": "grp", "name": "/imu_data/", "note": "Synchronized IMU streams"},
        {"level": 2, "kind": "ds", "name": "accel", "note": "Tri-axial linear acceleration"},
        {"level": 2, "kind": "ds", "name": "gyro", "note": "Tri-axial angular velocity"},
        {"level": 2, "kind": "ds", "name": "time_s", "note": "Elapsed recording time"},
        {"level": 2, "kind": "ds", "name": "sample_idx", "note": "Sample index"},
        {"level": 2, "kind": "ds", "name": "sensor_names", "note": "IMU short codes"},
        {"level": 2, "kind": "ds", "name": "axis_names", "note": "Local axis labels"},
        {"level": 1, "kind": "grp", "name": "/phase_markers/", "note": "Phase annotations"},
        {"level": 2, "kind": "grp", "name": "consensus/", "note": "Consensus phase markers"},
        {"level": 2, "kind": "grp", "name": "annotators/", "note": "Individual annotator markers"},
    ]
    draw_tree(ax_b, h5_nodes, x_root=0.04, y_top=0.892, dy=0.068)

    out_dir = Path(__file__).resolve().parent
    pdf_path = out_dir / "fig06_folders.pdf"
    png_path = out_dir / "fig06_folders.png"

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
