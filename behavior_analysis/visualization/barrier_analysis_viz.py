"""
Visualization module for barrier analysis.

Generates comprehensive visualizations for:
- Barrier distribution plots
- Regression model comparisons
- Feature importance charts
- Cluster profiles
- Heatmaps and correlation matrices
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..analysis.barrier_analysis import FeatureImportanceResults, RegressionResults
from ..utils.file_utils import ensure_directory_exists
from ..utils.logger import get_logger

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def plot_barrier_distribution(barrier_data: dict[str, Any], output_path: str) -> str:
    """
    Create histogram and KDE plot of barrier index distribution.

    Args:
        barrier_data: Dictionary with barrier index statistics
        output_path: Output file path

    Returns:
        Path to saved visualization
    """
    logger = get_logger()
    logger.info("Creating barrier distribution plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract barrier values (mock data for visualization structure)
    # In real usage, this would come from the DataFrame
    # For now, create a placeholder
    ax.set_xlabel("Barrier Index (0-100)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Barrier Index")
    ax.text(
        0.5,
        0.5,
        "Barrier distribution visualization\n(requires barrier index data)",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )

    ensure_directory_exists(str(Path(output_path).parent))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Barrier distribution plot saved to: %s", output_path)
    return output_path


def plot_regression_comparison(
    regression_results: list[RegressionResults], output_path: str
) -> str:
    """
    Create bar chart comparing R² and coefficients across models.

    Shows:
    - R² values for each model
    - Barrier index coefficients
    - RMSE values

    Args:
        regression_results: List of RegressionResults
        output_path: Output file path

    Returns:
        Path to saved visualization
    """
    logger = get_logger()
    logger.info("Creating regression comparison plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Extract data
    model_names = [r.model_name for r in regression_results]
    r_squared_values = [r.r_squared for r in regression_results]
    barrier_coeffs = [r.coefficients.get("barrier_index", 0) for r in regression_results]

    # Plot R² values
    axes[0].bar(model_names, r_squared_values, color="steelblue", alpha=0.7)
    axes[0].set_ylabel("R² Value")
    axes[0].set_title("Model Performance (R²)")
    axes[0].set_ylim(0, max(r_squared_values) * 1.2 if r_squared_values else 1)
    axes[0].tick_params(axis="x", rotation=15)

    # Add value labels on bars
    for i, v in enumerate(r_squared_values):
        axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom")

    # Plot barrier coefficients
    colors = ["red" if c < 0 else "green" for c in barrier_coeffs]
    axes[1].bar(model_names, barrier_coeffs, color=colors, alpha=0.7)
    axes[1].set_ylabel("Coefficient Value")
    axes[1].set_title("Barrier Index Coefficient")
    axes[1].axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    axes[1].tick_params(axis="x", rotation=15)

    # Add value labels
    for i, v in enumerate(barrier_coeffs):
        axes[1].text(
            i,
            v + (0.5 if v > 0 else -0.5),
            f"{v:.2f}",
            ha="center",
            va="bottom" if v > 0 else "top",
        )

    ensure_directory_exists(str(Path(output_path).parent))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Regression comparison plot saved to: %s", output_path)
    return output_path


def plot_barriers_by_cluster(cluster_stats: dict[str, Any], output_path: str) -> str:
    """
    Create grouped bar chart showing barrier levels across score clusters.

    Includes:
    - Overall barrier index by cluster
    - Error bars showing standard deviations

    Args:
        cluster_stats: Statistics from analyze_barriers_by_cluster()
        output_path: Output file path

    Returns:
        Path to saved visualization
    """
    logger = get_logger()
    logger.info("Creating barriers by cluster plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data
    clusters = list(cluster_stats.keys())
    mean_barriers = [
        cluster_stats[c]["mean_barrier"]
        for c in clusters
        if cluster_stats[c]["mean_barrier"] is not None
    ]
    std_barriers = [
        cluster_stats[c]["std_barrier"]
        for c in clusters
        if cluster_stats[c]["std_barrier"] is not None
    ]

    # Create bar plot
    x_pos = np.arange(len(clusters))
    bars = ax.bar(x_pos, mean_barriers, yerr=std_barriers, capsize=5, alpha=0.7, color="coral")

    ax.set_xlabel("Score Cluster")
    ax.set_ylabel("Mean Barrier Index")
    ax.set_title("Barrier Index Distribution Across Score Clusters")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(clusters)

    # Add value labels
    for _i, (bar, val) in enumerate(zip(bars, mean_barriers, strict=True)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.1f}",
            ha="center",
            va="bottom",
        )

    ensure_directory_exists(str(Path(output_path).parent))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Barriers by cluster plot saved to: %s", output_path)
    return output_path


def plot_dimension_breakdown_by_cluster(cluster_stats: dict[str, Any], output_path: str) -> str:
    """
    Create stacked bar chart showing dimension composition by cluster.

    Args:
        cluster_stats: Statistics with dimension breakdown
        output_path: Output file path

    Returns:
        Path to saved visualization
    """
    logger = get_logger()
    logger.info("Creating dimension breakdown plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract dimension data
    clusters = list(cluster_stats.keys())
    dimensions = [
        "dim_access_resources",
        "dim_internet_access",
        "dim_learning_disabilities",
        "dim_geographic_isolation",
    ]
    dimension_labels = [
        "Access to Resources",
        "Internet Access",
        "Learning Disabilities",
        "Geographic Isolation",
    ]

    # Prepare data matrix
    data = []
    for dim in dimensions:
        dim_values = []
        for cluster in clusters:
            breakdown = cluster_stats[cluster].get("dimension_breakdown", {})
            val = breakdown.get(dim, 0)
            dim_values.append(val if val is not None else 0)
        data.append(dim_values)

    # Create stacked bar chart
    x_pos = np.arange(len(clusters))
    bottom = np.zeros(len(clusters))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]

    for _i, (dim_data, label, color) in enumerate(zip(data, dimension_labels, colors, strict=True)):
        ax.bar(x_pos, dim_data, bottom=bottom, label=label, color=color, alpha=0.8)
        bottom += np.array(dim_data)

    ax.set_xlabel("Score Cluster")
    ax.set_ylabel("Dimension Score")
    ax.set_title("Barrier Dimension Breakdown by Score Cluster")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(clusters)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    ensure_directory_exists(str(Path(output_path).parent))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Dimension breakdown plot saved to: %s", output_path)
    return output_path


def plot_feature_importance(
    importance_results: FeatureImportanceResults,
    output_path: str,
    top_n: int = 15,
) -> str:
    """
    Create horizontal bar chart of feature importances.

    Args:
        importance_results: FeatureImportanceResults object
        output_path: Output file path
        top_n: Number of top features to display

    Returns:
        Path to saved visualization
    """
    logger = get_logger()
    logger.info("Creating feature importance plot...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Get top N features
    top_features = importance_results.top_n_features[:top_n]
    features = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]

    # Create horizontal bar chart
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importances, color="steelblue", alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel("Importance Score")
    ax.set_title(
        f"Top {top_n} Feature Importances ({importance_results.method.replace('_', ' ').title()})"
    )
    ax.invert_yaxis()  # Highest importance at the top

    # Add value labels
    for _i, (bar, val) in enumerate(zip(bars, importances, strict=True)):
        ax.text(
            val,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
        )

    ensure_directory_exists(str(Path(output_path).parent))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Feature importance plot saved to: %s", output_path)
    return output_path


def plot_barrier_clusters_radar(
    cluster_stats: dict[str, Any],
    dimension_names: list[str],
    output_path: str,
) -> str:
    """
    Create radar chart showing barrier profiles for each cluster.

    Each cluster gets a line showing its mean values across
    the 4 barrier dimensions.

    Args:
        cluster_stats: Cluster characteristics dictionary
        dimension_names: Names of barrier dimensions
        output_path: Output file path

    Returns:
        Path to saved visualization
    """
    logger = get_logger()
    logger.info("Creating barrier clusters radar plot...")

    # Number of dimensions
    num_dims = len(dimension_names)
    angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

    # Plot each cluster
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
    for i, (cluster_id, stats) in enumerate(cluster_stats.items()):
        values = []
        dim_means = stats.get("dimension_means", {})

        for dim in dimension_names:
            val = dim_means.get(dim, 0)
            values.append(val if val is not None else 0)

        values += values[:1]  # Complete the circle

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=cluster_id,
            color=colors[i % len(colors)],
        )
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])

    # Set labels
    ax.set_xticks(angles[:-1])
    short_labels = [
        "Resources",
        "Internet",
        "Learning",
        "Geographic",
    ]
    ax.set_xticklabels(short_labels)
    ax.set_title("Barrier Cluster Profiles (Radar Chart)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    ensure_directory_exists(str(Path(output_path).parent))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Barrier clusters radar plot saved to: %s", output_path)
    return output_path


def plot_barrier_correlation_heatmap(correlation_matrix: pd.DataFrame, output_path: str) -> str:
    """
    Create heatmap showing correlations between barrier dimensions.

    Args:
        correlation_matrix: Correlation matrix DataFrame
        output_path: Output file path

    Returns:
        Path to saved visualization
    """
    logger = get_logger()
    logger.info("Creating barrier correlation heatmap...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title("Barrier Dimension Correlation Matrix")

    ensure_directory_exists(str(Path(output_path).parent))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Correlation heatmap saved to: %s", output_path)
    return output_path


def plot_cluster_academic_outcomes(cluster_performance: dict[str, Any], output_path: str) -> str:
    """
    Create box plots showing academic performance by barrier cluster.

    Args:
        cluster_performance: Performance statistics by cluster
        output_path: Output file path

    Returns:
        Path to saved visualization
    """
    logger = get_logger()
    logger.info("Creating cluster academic outcomes plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data
    clusters = list(cluster_performance.keys())
    mean_scores = [
        cluster_performance[c]["mean_score"]
        for c in clusters
        if cluster_performance[c]["mean_score"] is not None
    ]
    std_scores = [
        cluster_performance[c]["std_score"]
        for c in clusters
        if cluster_performance[c]["std_score"] is not None
    ]

    # Create bar plot with error bars
    x_pos = np.arange(len(clusters))
    bars = ax.bar(
        x_pos,
        mean_scores,
        yerr=std_scores,
        capsize=5,
        alpha=0.7,
        color="mediumseagreen",
    )

    ax.set_xlabel("Barrier Cluster")
    ax.set_ylabel("Mean Math Score (PV1MATH)")
    ax.set_title("Academic Performance by Barrier Cluster")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(clusters, rotation=15)

    # Add value labels
    for _i, (bar, val) in enumerate(zip(bars, mean_scores, strict=True)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.1f}",
            ha="center",
            va="bottom",
        )

    ensure_directory_exists(str(Path(output_path).parent))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Cluster academic outcomes plot saved to: %s", output_path)
    return output_path


def plot_regression_residuals(regression_result: RegressionResults, output_path: str) -> str:
    """
    Create diagnostic plots for regression model validation.

    Args:
        regression_result: RegressionResults object
        output_path: Output file path

    Returns:
        Path to saved visualization
    """
    logger = get_logger()
    logger.info("Creating regression residuals plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Check if residuals sample is available
    if regression_result.residuals_sample and len(regression_result.residuals_sample) > 0:
        # Extract fitted values and residuals
        fitted_values = [point[0] for point in regression_result.residuals_sample]
        residuals = [point[1] for point in regression_result.residuals_sample]

        # Create scatter plot
        ax.scatter(
            fitted_values,
            residuals,
            alpha=0.5,
            s=20,
            color="steelblue",
            edgecolors="none",
        )

        # Add reference line at y=0
        ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="Zero residual")

        # Add LOESS smoothing line (using numpy polynomial fit as approximation)
        if len(fitted_values) > 10:
            sorted_indices = np.argsort(fitted_values)
            x_sorted = np.array(fitted_values)[sorted_indices]
            y_sorted = np.array(residuals)[sorted_indices]

            # Use polynomial fit to approximate trend
            try:
                z = np.polyfit(x_sorted, y_sorted, 3)
                p = np.poly1d(z)
                x_smooth = np.linspace(min(x_sorted), max(x_sorted), 100)
                y_smooth = p(x_smooth)
                ax.plot(x_smooth, y_smooth, color="orange", linewidth=2, label="Trend line")
            except Exception:
                pass

        # Labels and title
        ax.set_xlabel("Fitted Values (Predicted Scores)", fontsize=11)
        ax.set_ylabel("Residuals (Actual - Predicted)", fontsize=11)
        ax.set_title(
            f"Residual Plot - {regression_result.model_name}\n(Sample: {len(fitted_values):,} points)",
            fontsize=12,
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        stats_text = f"Mean: {mean_residual:.2f}\nStd: {std_residual:.2f}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )
    else:
        # No residuals available - show placeholder
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.set_title(f"Residual Plot - {regression_result.model_name}")
        ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
        ax.text(
            0.5,
            0.5,
            "Residual data not available\n(enable residuals_sample in regression)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

    ensure_directory_exists(str(Path(output_path).parent))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Regression residuals plot saved to: %s", output_path)
    return output_path


def create_all_barrier_visualizations(
    results: dict[str, Any], output_directory: str
) -> dict[str, str]:
    """
    Generate all barrier analysis visualizations.

    Args:
        results: Complete results from perform_comprehensive_barrier_analysis()
        output_directory: Directory to save all visualizations

    Returns:
        Dictionary mapping visualization names to file paths
    """
    logger = get_logger()
    logger.info("Creating all barrier visualizations in: %s", output_directory)

    ensure_directory_exists(output_directory)
    viz_paths = {}

    # 1. Barrier distribution (if available)
    if "barrier_statistics" in results:
        path = str(Path(output_directory) / "barrier_distribution.png")
        viz_paths["barrier_distribution"] = plot_barrier_distribution(
            results["barrier_statistics"], path
        )

    # 2. Regression comparison
    if "regression_comparison" in results:
        regression_list = []
        for key in [
            "regression_baseline",
            "regression_with_ses",
            "regression_with_country_fe",
        ]:
            if key in results:
                regression_list.append(results[key])

        if regression_list:
            path = str(Path(output_directory) / "regression_comparison.png")
            viz_paths["regression_comparison"] = plot_regression_comparison(regression_list, path)

    # 3. Barriers by score cluster
    if "barriers_by_cluster" in results:
        path = str(Path(output_directory) / "barriers_by_score_cluster.png")
        viz_paths["barriers_by_cluster"] = plot_barriers_by_cluster(
            results["barriers_by_cluster"], path
        )

        # 4. Dimension breakdown
        path = str(Path(output_directory) / "dimension_breakdown_by_cluster.png")
        viz_paths["dimension_breakdown"] = plot_dimension_breakdown_by_cluster(
            results["barriers_by_cluster"], path
        )

    # 5. Feature importance (RF)
    if "feature_importance_rf" in results:
        path = str(Path(output_directory) / "feature_importance_rf.png")
        viz_paths["feature_importance_rf"] = plot_feature_importance(
            results["feature_importance_rf"], path
        )

    # 6. Feature importance (Regression)
    if "feature_importance_regression" in results:
        path = str(Path(output_directory) / "feature_importance_regression.png")
        viz_paths["feature_importance_regression"] = plot_feature_importance(
            results["feature_importance_regression"], path
        )

    # 7. Barrier clusters radar
    if "cluster_characteristics" in results:
        dimension_names = [
            "dim_access_resources",
            "dim_internet_access",
            "dim_learning_disabilities",
            "dim_geographic_isolation",
        ]
        path = str(Path(output_directory) / "barrier_clusters_radar.png")
        viz_paths["barrier_clusters_radar"] = plot_barrier_clusters_radar(
            results["cluster_characteristics"], dimension_names, path
        )

    # 8. Correlation heatmap (if dimension correlation data available)
    if "dimension_correlations" in results:
        path = str(Path(output_directory) / "barrier_correlation_heatmap.png")
        viz_paths["correlation_heatmap"] = plot_barrier_correlation_heatmap(
            results["dimension_correlations"], path
        )

    # 9. Cluster academic outcomes
    if "cluster_academic_outcomes" in results:
        path = str(Path(output_directory) / "cluster_academic_outcomes.png")
        viz_paths["cluster_academic_outcomes"] = plot_cluster_academic_outcomes(
            results["cluster_academic_outcomes"], path
        )

    # 10. Regression residuals (first available model)
    if "regression_baseline" in results:
        path = str(Path(output_directory) / "regression_residuals.png")
        viz_paths["regression_residuals"] = plot_regression_residuals(
            results["regression_baseline"], path
        )

    logger.info("Created %d visualizations", len(viz_paths))
    return viz_paths


# ============================================================================
# CSV EXPORT FUNCTIONS
# ============================================================================


def export_results_to_csv(results: dict[str, Any], output_directory: str) -> dict[str, str]:
    """
    Export all analysis results to CSV files.

    Creates separate CSV files for:
    - Regression model comparison
    - Barrier statistics by cluster
    - Feature importance rankings
    - Cluster characteristics

    Args:
        results: Complete analysis results
        output_directory: Output directory

    Returns:
        Dictionary mapping result types to CSV paths
    """
    logger = get_logger()
    logger.info("Exporting results to CSV in: %s", output_directory)

    ensure_directory_exists(output_directory)
    csv_paths = {}

    # 1. Regression model comparison
    if "regression_comparison" in results:
        comparison = results["regression_comparison"]
        rows = []
        for model_name, stats in comparison["models"].items():
            rows.append(
                {
                    "Model": model_name,
                    "R_Squared": stats["r_squared"],
                    "RMSE": stats["rmse"],
                    "Barrier_Coefficient": stats.get("barrier_coefficient", None),
                    "Sample_Size": stats["sample_size"],
                }
            )

        df = pd.DataFrame(rows)
        path = str(Path(output_directory) / "regression_model_comparison.csv")
        df.to_csv(path, index=False)
        csv_paths["regression_comparison"] = path
        logger.info("Exported regression comparison to: %s", path)

    # 2. Barrier statistics by cluster
    if "barriers_by_cluster" in results:
        rows = []
        for cluster, stats in results["barriers_by_cluster"].items():
            row = {
                "Cluster": cluster,
                "Sample_Count": stats["sample_count"],
                "Weighted_Count": stats["weighted_count"],
                "Mean_Barrier": stats.get("mean_barrier"),
                "Weighted_Mean_Barrier": stats.get("weighted_mean_barrier"),
                "Std_Barrier": stats.get("std_barrier"),
                "Q25": stats.get("q25"),
                "Median": stats.get("median"),
                "Q75": stats.get("q75"),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        path = str(Path(output_directory) / "barrier_statistics_by_cluster.csv")
        df.to_csv(path, index=False)
        csv_paths["barriers_by_cluster"] = path
        logger.info("Exported barrier statistics to: %s", path)

    # 3. Feature importance rankings (RF)
    if "feature_importance_rf" in results:
        importance_data = results["feature_importance_rf"]
        rows = [
            {"Feature": feature, "Importance": importance, "Rank": i + 1}
            for i, (feature, importance) in enumerate(importance_data.top_n_features)
        ]

        df = pd.DataFrame(rows)
        path = str(Path(output_directory) / "feature_importance_rankings.csv")
        df.to_csv(path, index=False)
        csv_paths["feature_importance"] = path
        logger.info("Exported feature importance to: %s", path)

    # 4. Barrier cluster characteristics
    if "cluster_characteristics" in results:
        rows = []
        for cluster_id, stats in results["cluster_characteristics"].items():
            row = {
                "Cluster": cluster_id,
                "Sample_Count": stats["sample_count"],
                "Weighted_Count": stats["weighted_count"],
            }
            # Add dimension means
            for dim_name, dim_value in stats.get("dimension_means", {}).items():
                row[dim_name] = dim_value
            rows.append(row)

        df = pd.DataFrame(rows)
        path = str(Path(output_directory) / "barrier_cluster_characteristics.csv")
        df.to_csv(path, index=False)
        csv_paths["cluster_characteristics"] = path
        logger.info("Exported cluster characteristics to: %s", path)

    # 5. Dimension correlations (if available)
    if "dimension_correlations" in results:
        corr_df = results["dimension_correlations"]
        path = str(Path(output_directory) / "dimension_correlations.csv")
        corr_df.to_csv(path)
        csv_paths["dimension_correlations"] = path
        logger.info("Exported dimension correlations to: %s", path)

    logger.info("Exported %d CSV files", len(csv_paths))
    return csv_paths
