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
from .paper_figures import (
    build_annotation_agreement_table,
    build_imu_quality_table,
    build_paper_tables,
    plot_all_sets_annotation_sd_heatmap,
    plot_annotation_agreement_figure,
    plot_fig4_annotation_scheme,
    plot_imu_validation_figure,
    plot_set_annotator_grid,
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
    "build_annotation_agreement_table",
    "build_imu_quality_table",
    "build_paper_tables",
    "detect_set_faults",
    "launch_fault_logger",
    "load_set_h5",
    "load_summary_tables",
    "plot_all_sets_annotation_sd_heatmap",
    "plot_annotation_agreement_figure",
    "plot_cohort_overview",
    "plot_fig4_annotation_scheme",
    "plot_imu_validation_figure",
    "plot_joint_4panel",
    "plot_marker_agreement_summary",
    "plot_sensor_health_overview",
    "plot_set_annotator_grid",
    "plot_set_imu_with_markers",
    "plot_set_sensor_status",
]

