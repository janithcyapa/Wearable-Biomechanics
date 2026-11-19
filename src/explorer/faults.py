"""Automated fault detection, 4-panel joint/IMU inspection plots, and fault_log.csv logger."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PUB_FONT = dict(family="Arial, Helvetica, sans-serif", size=11, color="#111827")
_PUB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

JOINT_SENSOR_PAIRS: dict[str, tuple[str, str]] = {
    "right_knee": ("RKP", "RKD"),
    "left_knee": ("LKP", "LKD"),
    "right_hip": ("RHP", "RHD"),
    "left_hip": ("LHP", "LHD"),
    "right_ankle": ("RAP", "RAD"),
    "left_ankle": ("LAP", "LAD"),
    "right_shoulder": ("RSP", "RSD"),
    "left_shoulder": ("LSP", "LSD"),
    "right_elbow": ("REP", "RED"),
    "left_elbow": ("LEP", "LED"),
    "right_wrist": ("RWP", "RWD"),
    "left_wrist": ("LWP", "LWD"),
    "spine": ("US", "LS"),
}

FAULT_LOG_COLUMNS = [
    "fault_id",
    "set_id",
    "sensor_code",
    "fault_type",
    "severity",
    "start_imu_idx",
    "end_imu_idx",
    "start_s",
    "end_s",
    "description",
]


def _find_freeze_runs(vals: np.ndarray, min_run: int) -> list[tuple[int, int]]:
    """Return [(start_idx, end_idx), ...] for consecutive identical finite runs >= min_run."""
    if len(vals) < min_run:
        return []
    valid = ~np.isnan(vals)
    eq = (vals[1:] == vals[:-1]) & valid[1:] & valid[:-1]
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for i, flag in enumerate(eq):
        if flag:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start + 1) >= min_run:
                runs.append((run_start, i))
            run_start = None
    if run_start is not None and (len(vals) - run_start) >= min_run:
        runs.append((run_start, len(vals) - 1))
    return runs


def detect_set_faults(set_data: dict, freeze_sec: float = 0.5) -> pd.DataFrame:
    """Scan all 26 IMUs in `set_data` and return a DataFrame of auto-detected faults."""
    sensors = set_data["sensors"]
    acc = set_data["accel"]
    gyr = set_data["gyro"]
    time_s = set_data["time_s"]
    fs = float(set_data["metadata"].get("imu_fs_hz", 100.0))
    n = len(time_s)
    min_run = max(5, int(round(freeze_sec * fs)))

    rows: list[dict] = []
    for s_idx, code in enumerate(sensors):
        a_sl = acc[:, s_idx, :]
        g_sl = gyr[:, s_idx, :]

        if np.all(np.isnan(a_sl)) and np.all(np.isnan(g_sl)):
            rows.append(
                {
                    "sensor_code": code,
                    "fault_type": "missing_or_all_nan",
                    "start_imu_idx": 0,
                    "end_imu_idx": n - 1,
                    "start_s": float(time_s[0]),
                    "end_s": float(time_s[-1]),
                    "details": "All 6 channels are NaN",
                }
            )
            continue

        for mod_name, arr, axes, clip_bound in [
            ("Accel", a_sl, ("AX", "AY", "AZ"), 80.0),
            ("Gyro", g_sl, ("GX", "GY", "GZ"), 2000.0),
        ]:
            for ax_i, ax_lbl in enumerate(axes):
                v = arr[:, ax_i]
                nan_mask = np.isnan(v)
                nan_cnt = int(nan_mask.sum())
                if nan_cnt == n:
                    rows.append(
                        {
                            "sensor_code": code,
                            "fault_type": "all_nan",
                            "start_imu_idx": 0,
                            "end_imu_idx": n - 1,
                            "start_s": float(time_s[0]),
                            "end_s": float(time_s[-1]),
                            "details": f"{ax_lbl} all NaN",
                        }
                    )
                    continue
                if nan_cnt > 0:
                    idx = np.where(nan_mask)[0]
                    rows.append(
                        {
                            "sensor_code": code,
                            "fault_type": "partial_nan",
                            "start_imu_idx": int(idx[0]),
                            "end_imu_idx": int(idx[-1]),
                            "start_s": float(time_s[idx[0]]),
                            "end_s": float(time_s[idx[-1]]),
                            "details": f"{ax_lbl} has {nan_cnt} NaN samples",
                        }
                    )

                if np.nanstd(v) == 0:
                    rows.append(
                        {
                            "sensor_code": code,
                            "fault_type": "flatline",
                            "start_imu_idx": 0,
                            "end_imu_idx": n - 1,
                            "start_s": float(time_s[0]),
                            "end_s": float(time_s[-1]),
                            "details": f"{ax_lbl} constant={np.nanmean(v):.3f}",
                        }
                    )
                else:
                    for rs, re_idx in _find_freeze_runs(v, min_run=min_run):
                        rows.append(
                            {
                                "sensor_code": code,
                                "fault_type": "freeze",
                                "start_imu_idx": rs,
                                "end_imu_idx": re_idx,
                                "start_s": float(time_s[rs]),
                                "end_s": float(time_s[re_idx]),
                                "details": f"{ax_lbl} frozen for {(re_idx - rs + 1) / fs:.2f}s",
                            }
                        )

                clip_mask = np.abs(v) > clip_bound
                if np.any(clip_mask):
                    c_idx = np.where(clip_mask)[0]
                    rows.append(
                        {
                            "sensor_code": code,
                            "fault_type": "clipping",
                            "start_imu_idx": int(c_idx[0]),
                            "end_imu_idx": int(c_idx[-1]),
                            "start_s": float(time_s[c_idx[0]]),
                            "end_s": float(time_s[c_idx[-1]]),
                            "details": f"{ax_lbl} peak={np.nanmax(np.abs(v)):.1f} (>{clip_bound})",
                        }
                    )

    if not rows:
        return pd.DataFrame(
            columns=["sensor_code", "fault_type", "start_imu_idx", "end_imu_idx", "start_s", "end_s", "details"]
        )
    return pd.DataFrame(rows)


def plot_joint_4panel(
    set_data: dict,
    joint_name: str = "right_knee",
    lift_id: int | None = None,
) -> go.Figure:
    """Plot 4 subplots (Accel & Gyro for both sensors of a joint) with vertical phase marker labels."""
    if joint_name not in JOINT_SENSOR_PAIRS:
        raise ValueError(f"Unknown joint '{joint_name}'. Choose from {list(JOINT_SENSOR_PAIRS)}")

    s1, s2 = JOINT_SENSOR_PAIRS[joint_name]
    sensors = set_data["sensors"]
    idx1, idx2 = sensors.index(s1), sensors.index(s2)

    t = set_data["time_s"]
    acc1, gyr1 = set_data["accel"][:, idx1, :], set_data["gyro"][:, idx1, :]
    acc2, gyr2 = set_data["accel"][:, idx2, :], set_data["gyro"][:, idx2, :]
    cons_df = set_data["consensus_df"]

    if lift_id is not None and not cons_df.empty:
        cons_plot = cons_df[cons_df["lift_id"] == lift_id]
        if not cons_plot.empty:
            t_min = float(cons_plot["time_s"].min()) - 0.5
            t_max = float(cons_plot["time_s"].max()) + 0.5
            mask = (t >= t_min) & (t <= t_max)
            t = t[mask]
            acc1, gyr1 = acc1[mask], gyr1[mask]
            acc2, gyr2 = acc2[mask], gyr2[mask]
    else:
        cons_plot = cons_df

    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.10,
        horizontal_spacing=0.07,
        subplot_titles=(
            f"<b>{s1} — Accelerometer (m/s²)</b>",
            f"<b>{s2} — Accelerometer (m/s²)</b>",
            f"<b>{s1} — Gyroscope (deg/s)</b>",
            f"<b>{s2} — Gyroscope (deg/s)</b>",
        ),
    )

    for ax_i, ax_name in enumerate(("X", "Y", "Z")):
        c_acc = _PUB_PALETTE[ax_i]
        c_gyr = _PUB_PALETTE[ax_i + 3]
        fig.add_trace(
            go.Scatter(x=t, y=acc1[:, ax_i], mode="lines", name=f"A{ax_name}",
                       legendgroup=f"A{ax_name}", line=dict(width=1.2, color=c_acc)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=t, y=acc2[:, ax_i], mode="lines", name=f"A{ax_name}",
                       legendgroup=f"A{ax_name}", showlegend=False, line=dict(width=1.2, color=c_acc)),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(x=t, y=gyr1[:, ax_i], mode="lines", name=f"G{ax_name}",
                       legendgroup=f"G{ax_name}", line=dict(width=1.2, color=c_gyr)),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(x=t, y=gyr2[:, ax_i], mode="lines", name=f"G{ax_name}",
                       legendgroup=f"G{ax_name}", showlegend=False, line=dict(width=1.2, color=c_gyr)),
            row=2, col=2,
        )

    # Vertical dashed lines + vertical text labels along the top of the plot
    if not cons_plot.empty:
        for _, row in cons_plot.iterrows():
            ts = float(row["time_s"])
            label = f"L{int(row['lift_id'])}:{row['marker_name']}" if lift_id is None else str(row["marker_name"])
            for col_i in (1, 2):
                fig.add_vline(x=ts, line_width=1, line_dash="dot", line_color="#6b7280", row=1, col=col_i)
                fig.add_vline(x=ts, line_width=1, line_dash="dot", line_color="#6b7280", row=2, col=col_i)
            fig.add_annotation(
                x=ts, y=1.01, yref="paper", xref="x",
                text=label, showarrow=False, textangle=-90,
                xanchor="center", yanchor="bottom",
                font=dict(size=9, color="#374151"),
            )
            fig.add_annotation(
                x=ts, y=1.01, yref="paper", xref="x2",
                text=label, showarrow=False, textangle=-90,
                xanchor="center", yanchor="bottom",
                font=dict(size=9, color="#374151"),
            )

    lift_label = f"Lift {lift_id}" if lift_id is not None else "Full Set"
    fig.update_layout(
        title=dict(
            text=f"<b>{set_data['set_id']} — {joint_name.replace('_', ' ').title()} ({s1} & {s2}) | {lift_label}</b>",
            x=0.5, y=0.98, font=dict(size=14),
        ),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_PUB_FONT,
        height=680,
        hovermode="x unified",
        margin=dict(l=60, r=25, t=125, b=65),
        legend=dict(orientation="h", y=-0.10, x=0.5, xanchor="center"),
    )
    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(
                showline=True, linewidth=1.2, linecolor="#1f2937", mirror=True,
                ticks="outside", showgrid=True, gridcolor="#e5e7eb", gridwidth=0.7,
                zeroline=False, row=r, col=c,
            )
            fig.update_yaxes(
                showline=True, linewidth=1.2, linecolor="#1f2937", mirror=True,
                ticks="outside", showgrid=True, gridcolor="#e5e7eb", gridwidth=0.7,
                zeroline=True, zerolinecolor="#d1d5db", zerolinewidth=0.8, row=r, col=c,
            )
    fig.update_yaxes(title_text="Accel (m/s²)", row=1, col=1)
    fig.update_yaxes(title_text="Gyro (deg/s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    return fig


def launch_fault_logger(dataset_dir: str | Path, set_data: dict) -> None:
    """Display an interactive ipywidgets form to append/delete fault rows in `fault_log.csv`."""
    import ipywidgets as widgets
    from IPython.display import display

    fault_csv = Path(dataset_dir) / "fault_log.csv"
    set_id = set_data["set_id"]
    fs = float(set_data["metadata"].get("imu_fs_hz", 100.0))
    n_samples = len(set_data["time_s"])
    max_s = float(set_data["time_s"][-1])

    sensor_w = widgets.Dropdown(
        options=list(set_data["sensors"]) + ["ALL", "VIDEO"],
        value=set_data["sensors"][0],
        description="Sensor:",
        layout=widgets.Layout(width="200px"),
    )
    type_w = widgets.Dropdown(
        options=["flatline", "freeze", "dropout", "clipping", "partial_nan", "missing", "missing_camera", "sync_issue", "other"],
        value="freeze",
        description="Fault Type:",
        layout=widgets.Layout(width="230px"),
    )
    sev_w = widgets.Dropdown(
        options=["minor", "major", "critical"],
        value="minor",
        description="Severity:",
        layout=widgets.Layout(width="190px"),
    )
    start_s_w = widgets.FloatText(value=0.0, description="Start (s):", layout=widgets.Layout(width="170px"))
    end_s_w = widgets.FloatText(value=round(max_s, 3), description="End (s):", layout=widgets.Layout(width="170px"))
    desc_w = widgets.Text(value="", placeholder="Describe the fault...", description="Description:", layout=widgets.Layout(width="520px"))

    btn_add = widgets.Button(description="➕ Add Fault to fault_log.csv", button_style="success", layout=widgets.Layout(width="220px"))
    del_id_w = widgets.Text(value="", placeholder="fault_0001", description="Delete ID:", layout=widgets.Layout(width="220px"))
    btn_del = widgets.Button(description="🗑️ Delete Fault", button_style="danger", layout=widgets.Layout(width="140px"))
    out = widgets.Output()

    def _load_df() -> pd.DataFrame:
        if fault_csv.exists():
            return pd.read_csv(fault_csv)
        return pd.DataFrame(columns=FAULT_LOG_COLUMNS)

    def _render_table():
        df = _load_df()
        sub = df[df["set_id"] == set_id]
        print(f"📋 Faults logged for {set_id}: {len(sub)} (Total in fault_log.csv: {len(df)})")
        display(sub)

    def _on_add(_b):
        with out:
            out.clear_output()
            df = _load_df()
            if df.empty:
                next_num = 1
            else:
                nums = [int(str(x).split("_")[1]) for x in df["fault_id"] if "_" in str(x)]
                next_num = (max(nums) + 1) if nums else (len(df) + 1)
            fid = f"fault_{next_num:04d}"

            s_idx = int(np.clip(round(float(start_s_w.value) * fs), 0, n_samples - 1))
            e_idx = int(np.clip(round(float(end_s_w.value) * fs), 0, n_samples - 1))
            new_row = {
                "fault_id": fid,
                "set_id": set_id,
                "sensor_code": sensor_w.value,
                "fault_type": type_w.value,
                "severity": sev_w.value,
                "start_imu_idx": s_idx,
                "end_imu_idx": e_idx,
                "start_s": round(float(start_s_w.value), 4),
                "end_s": round(float(end_s_w.value), 4),
                "description": desc_w.value.strip(),
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(fault_csv, index=False)
            print(f"✅ Saved {fid} to {fault_csv}")
            _render_table()

    def _on_del(_b):
        with out:
            out.clear_output()
            df = _load_df()
            target = del_id_w.value.strip()
            if not target or target not in set(df["fault_id"].astype(str)):
                print(f"⚠️ Fault ID '{target}' not found.")
            else:
                df = df[df["fault_id"].astype(str) != target].reset_index(drop=True)
                df.to_csv(fault_csv, index=False)
                print(f"🗑️ Deleted {target} from {fault_csv}")
            _render_table()

    btn_add.on_click(_on_add)
    btn_del.on_click(_on_del)

    ui = widgets.VBox(
        [
            widgets.HBox([sensor_w, type_w, sev_w]),
            widgets.HBox([start_s_w, end_s_w, desc_w]),
            widgets.HBox([btn_add, del_id_w, btn_del]),
            out,
        ]
    )
    display(ui)
    with out:
        _render_table()
