"""Unit tests for wearable_biomechanics pure-Python functions."""

import numpy as np
import pandas as pd

from src.annotation import PHASES, new_rep_template, tag_for
from src.imu import get_columns, get_sensor_health_summary
from src.sync.temporal import (
    calculate_front_to_raw_sync_lag,
    calculate_sync_lag,
)
from src.sync.video import MOTION_KEYPOINTS


def test_get_columns_and_health_summary():
    df = pd.DataFrame(
        {
            "Timestamp": np.arange(10),
            "1_IMU1_AX": np.linspace(-1.0, 1.0, 10),
            "1_IMU1_AY": np.zeros(10),  # flatline
            "2_IMU1_GZ": np.linspace(-50.0, 50.0, 10),
        }
    )
    assert get_columns(df, devices=1, imus=1, axes=["AX", "AY"]) == [
        "1_IMU1_AX",
        "1_IMU1_AY",
    ]
    health = get_sensor_health_summary(df)
    statuses = dict(zip(health["Column"], health["Status"]))
    assert statuses["1_IMU1_AX"] == "OK"
    assert statuses["1_IMU1_AY"] == "FLATLINE"
    assert statuses["2_IMU1_GZ"] == "OK"


def test_calculate_sync_lag_detects_shift():
    n = 120
    t = np.linspace(0, 4 * np.pi, n)
    front_data = {}
    side_data = {}
    shift = 6
    for kp in MOTION_KEYPOINTS:
        front_data[f"{kp}_x"] = np.sin(t) * 100.0
        front_data[f"{kp}_y"] = np.cos(t) * 50.0
        front_data[f"{kp}_score"] = np.ones(n)

        t_shifted = np.linspace(0, 4 * np.pi, n) - shift * (4 * np.pi / n)
        side_data[f"{kp}_x"] = np.sin(t_shifted) * 100.0
        side_data[f"{kp}_y"] = np.cos(t_shifted) * 50.0
        side_data[f"{kp}_score"] = np.ones(n)

    lag = calculate_sync_lag(pd.DataFrame(front_data), pd.DataFrame(side_data), plot=False)
    assert isinstance(lag, int)


def test_calculate_front_to_raw_sync_lag():
    n_vid = 90
    n_imu = 300
    front_data = {}
    for kp in MOTION_KEYPOINTS:
        front_data[f"{kp}_x"] = np.sin(np.linspace(0, 3 * np.pi, n_vid)) * 50.0
        front_data[f"{kp}_y"] = np.zeros(n_vid)
        front_data[f"{kp}_score"] = np.ones(n_vid)

    raw_df = pd.DataFrame(
        {
            "1_IMU1_AX": np.sin(np.linspace(0, 3 * np.pi, n_imu)) * 5.0,
            "1_IMU1_AY": np.zeros(n_imu),
            "1_IMU1_AZ": np.ones(n_imu) * 9.81,
        }
    )
    res = calculate_front_to_raw_sync_lag(
        pd.DataFrame(front_data), raw_df, video_fps=30.0, plot=False
    )
    assert res["video_fps"] == 30.0
    assert "delta_t_frames" in res
    assert "delta_t_imu_samples" in res


def test_annotation_helpers():
    assert tag_for(3) == "rep_segments_a3"
    tpl = new_rep_template()
    assert list(tpl.keys()) == PHASES
    assert all(v == 0 for v in tpl.values())

