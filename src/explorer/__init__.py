"""Clean&JerkDataset loader and publication-grade Plotly visualization tools."""

from .faults import (
    JOINT_SENSOR_PAIRS,
    detect_set_faults,
    launch_fault_logger,
    plot_joint_4panel,
)
from .loader import (
    DEFAULT_CLEAN_DATASET_DIR,
    load_set_h5,
    load_summary_tables,
)
from .plots import (
    plot_cohort_overview,
    plot_marker_agreement_summary,
    plot_sensor_health_overview,
    plot_set_imu_with_markers,
    plot_set_sensor_status,
)

__all__ = [
    "DEFAULT_CLEAN_DATASET_DIR",
    "JOINT_SENSOR_PAIRS",
    "detect_set_faults",
    "launch_fault_logger",
    "load_set_h5",
    "load_summary_tables",
    "plot_cohort_overview",
    "plot_joint_4panel",
    "plot_marker_agreement_summary",
    "plot_sensor_health_overview",
    "plot_set_imu_with_markers",
    "plot_set_sensor_status",
]

