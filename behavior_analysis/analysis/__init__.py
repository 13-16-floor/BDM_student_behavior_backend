"""
Analysis module for statistical computations on PISA data.
"""

from .attitude_clustering import (
    add_attitude_labels,
    create_attitude_features,
    get_attitude_statistics,
    perform_attitude_clustering,
    print_attitude_report,
    validate_attitude_columns,
)
from .barrier_analysis import (
    BarrierClusterResults,
    BarrierIndexConfig,
    FeatureImportanceResults,
    RegressionResults,
    analyze_barriers_by_cluster,
    calculate_feature_importance_regression,
    calculate_feature_importance_rf,
    characterize_barrier_clusters,
    construct_barrier_index,
    perform_barrier_clustering,
    perform_comprehensive_barrier_analysis,
    print_barrier_analysis_report,
    run_regression_baseline,
    run_regression_with_country_fe,
    run_regression_with_ses,
    validate_required_columns,
)
from .basic_stats import calculate_column_mean, describe_dataset, get_column_statistics
from .score_clustering import (
    add_cluster_labels,
    get_cluster_statistics,
    print_clustering_report,
)

__all__ = [
    # Basic stats
    "calculate_column_mean",
    "get_column_statistics",
    "describe_dataset",
    "validate_attitude_columns",
    "create_attitude_features",
    "perform_attitude_clustering",
    "add_attitude_labels",
    "get_attitude_statistics",
    "print_attitude_report",
    # Score clustering
    "add_cluster_labels",
    "get_cluster_statistics",
    "print_clustering_report",
    # Barrier analysis - data classes
    "BarrierIndexConfig",
    "RegressionResults",
    "FeatureImportanceResults",
    "BarrierClusterResults",
    # Barrier analysis - core functions
    "validate_required_columns",
    "construct_barrier_index",
    "run_regression_baseline",
    "run_regression_with_ses",
    "run_regression_with_country_fe",
    "analyze_barriers_by_cluster",
    "calculate_feature_importance_rf",
    "calculate_feature_importance_regression",
    "perform_barrier_clustering",
    "characterize_barrier_clusters",
    "perform_comprehensive_barrier_analysis",
    "print_barrier_analysis_report",
]
