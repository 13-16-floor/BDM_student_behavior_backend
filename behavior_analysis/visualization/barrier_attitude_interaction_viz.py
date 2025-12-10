"""
Visualization module for Barrier × Attitude Interaction Analysis.

Generates visualizations including:
- Interaction effect plot (slopes at different barrier levels)
- Heatmap of mean scores by barrier × attitude groups
- Marginal effects plot
- Model comparison charts
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..analysis.barrier_attitude_interaction import InteractionResults
from ..utils.file_utils import ensure_directory_exists
from ..utils.logger import get_logger

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


def plot_interaction_effect(
    results: InteractionResults,
    output_path: str,
) -> str:
    """
    Create interaction effect visualization showing how attitude effect
    varies across barrier levels.

    Args:
        results: InteractionResults from main model
        output_path: Output file path

    Returns:
        Path to saved visualization
    """
    logger = get_logger()
    logger.info("Creating interaction effect plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract marginal effects
    barrier_levels = []
    attitude_effects = []
    level_names = []

    for level_name, effect_data in results.marginal_effects.items():
        barrier_levels.append(effect_data["barrier_level"])
        attitude_effects.append(effect_data["attitude_effect"])
        level_names.append(level_name)

    # Create bar plot
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]  # Green, Orange, Red
    bars = ax.bar(level_names, attitude_effects, color=colors, alpha=0.7, edgecolor="black")

    # Add zero line
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)

    # Add value labels
    for bar, val in zip(bars, attitude_effects, strict=True):
        ypos = val + 0.5 if val > 0 else val - 1.5
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            f"{val:+.2f}",
            ha="center",
            va="bottom" if val > 0 else "top",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Barrier Level", fontsize=12)
    ax.set_ylabel("Effect of +1 SD Attitude on Score", fontsize=12)
    ax.set_title(
        "Marginal Effect of Attitude at Different Barrier Levels\n"
        f"(Interaction β = {results.interaction_effect:.4f})",
        fontsize=13,
    )

    # Add interpretation text
    interpretation_short = (
        "Compensatory Effect"
        if results.interaction_effect > 0.1
        else ("Cumulative Disadvantage" if results.interaction_effect < -0.1 else "Additive Effect")
    )
    ax.text(
        0.02,
        0.98,
        f"Pattern: {interpretation_short}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    ensure_directory_exists(str(Path(output_path).parent))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Interaction effect plot saved to: %s", output_path)
    return output_path


def plot_interaction_heatmap(
    plot_data: dict[str, Any],
    output_path: str,
) -> str:
    """
    Create heatmap showing mean scores by barrier × attitude groups.

    Args:
        plot_data: Data from prepare_interaction_plot_data()
        output_path: Output file path

    Returns:
        Path to saved visualization
    """
    logger = get_logger()
    logger.info("Creating interaction heatmap...")

    heatmap_data = plot_data["heatmap_data"]
    barrier_labels = plot_data["barrier_labels"]
    attitude_labels = plot_data["attitude_labels"]

    # Create matrix
    matrix = np.zeros((len(barrier_labels), len(attitude_labels)))
    for i, b_label in enumerate(barrier_labels):
        for j, a_label in enumerate(attitude_labels):
            key = (b_label, a_label)
            if key in heatmap_data:
                matrix[i, j] = heatmap_data[key]["mean_score"]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean Score (Within-Country)", fontsize=11)

    # Set labels
    ax.set_xticks(np.arange(len(attitude_labels)))
    ax.set_yticks(np.arange(len(barrier_labels)))
    ax.set_xticklabels(attitude_labels)
    ax.set_yticklabels(barrier_labels)
    ax.set_xlabel("Attitude Level", fontsize=12)
    ax.set_ylabel("Barrier Level", fontsize=12)
    ax.set_title(
        "Mean Academic Score by Barrier × Attitude\n(Within-Country Deviation)",
        fontsize=13,
    )

    # Add text annotations
    for i in range(len(barrier_labels)):
        for j in range(len(attitude_labels)):
            key = (barrier_labels[i], attitude_labels[j])
            if key in heatmap_data:
                n = heatmap_data[key]["n"]
                score = heatmap_data[key]["mean_score"]
                text_color = "white" if abs(score) > matrix.max() * 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{score:+.1f}\n(n={n:,})",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

    ensure_directory_exists(str(Path(output_path).parent))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Interaction heatmap saved to: %s", output_path)
    return output_path


def plot_slopes_comparison(
    plot_data: dict[str, Any],
    output_path: str,
) -> str:
    """
    Create plot comparing attitude-score slopes across barrier groups.

    This visualizes the interaction effect: if slopes differ,
    there's an interaction.

    Args:
        plot_data: Data from prepare_interaction_plot_data()
        output_path: Output file path

    Returns:
        Path to saved visualization
    """
    logger = get_logger()
    logger.info("Creating slopes comparison plot...")

    slopes_data = plot_data["slopes_by_barrier"]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "Low Barrier": "#2ecc71",
        "Medium Barrier": "#f39c12",
        "High Barrier": "#e74c3c",
    }

    # Plot regression lines for each barrier group
    x_range = np.linspace(-2, 2, 100)  # Standardized attitude range

    for barrier_level, data in slopes_data.items():
        slope = data["slope"]
        intercept = data["intercept"]
        y_pred = intercept + slope * x_range

        color = colors.get(barrier_level, "gray")
        ax.plot(
            x_range,
            y_pred,
            label=f"{barrier_level} (β={slope:.2f}, n={data['n']:,})",
            color=color,
            linewidth=2.5,
        )

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Attitude Score (Standardized)", fontsize=12)
    ax.set_ylabel("Score Deviation from Country Mean", fontsize=12)
    ax.set_title(
        "Attitude-Score Relationship by Barrier Level\n"
        "(Parallel lines = No interaction, Diverging lines = Interaction)",
        fontsize=12,
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    ensure_directory_exists(str(Path(output_path).parent))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Slopes comparison plot saved to: %s", output_path)
    return output_path


def plot_model_comparison(
    main_results: InteractionResults,
    pooled_results: InteractionResults,
    output_path: str,
) -> str:
    """
    Create comparison between within-country and pooled models.

    Highlights the importance of controlling for country effects.

    Args:
        main_results: Results from within-country model
        pooled_results: Results from pooled model
        output_path: Output file path

    Returns:
        Path to saved visualization
    """
    logger = get_logger()
    logger.info("Creating model comparison plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = ["Pooled\n(No Country Control)", "Within-Country\n(Country FE)"]
    r2_values = [pooled_results.r_squared, main_results.r_squared]
    interaction_values = [
        pooled_results.interaction_effect,
        main_results.interaction_effect,
    ]

    # R² comparison
    ax1 = axes[0]
    bars1 = ax1.bar(models, r2_values, color=["#e74c3c", "#2ecc71"], alpha=0.7, edgecolor="black")
    ax1.set_ylabel("R²", fontsize=12)
    ax1.set_title("Model Explanatory Power", fontsize=13)
    for bar, val in zip(bars1, r2_values, strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.4f}",
            ha="center",
            fontsize=11,
        )

    # Interaction coefficient comparison
    ax2 = axes[1]
    colors2 = ["red" if v < 0 else "green" for v in interaction_values]
    bars2 = ax2.bar(models, interaction_values, color=colors2, alpha=0.7, edgecolor="black")
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax2.set_ylabel("Interaction Coefficient (β₃)", fontsize=12)
    ax2.set_title("Barrier × Attitude Interaction Effect", fontsize=13)
    for bar, val in zip(bars2, interaction_values, strict=True):
        ypos = val + 0.02 if val > 0 else val - 0.05
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            f"{val:+.4f}",
            ha="center",
            fontsize=11,
        )

    fig.suptitle(
        "Impact of Country Control on Interaction Analysis",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    ensure_directory_exists(str(Path(output_path).parent))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Model comparison plot saved to: %s", output_path)
    return output_path


def plot_stratified_coefficients(
    stratified_results: dict[str, InteractionResults],
    output_path: str,
    title_suffix: str = "",
) -> str:
    """
    Create coefficient comparison across strata.

    Args:
        stratified_results: Results from stratified analysis
        output_path: Output file path
        title_suffix: Additional text for title

    Returns:
        Path to saved visualization
    """
    logger = get_logger()
    logger.info("Creating stratified coefficients plot...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    strata = list(stratified_results.keys())
    barrier_coefs = [stratified_results[s].coefficients.get("barrier", 0) for s in strata]
    attitude_coefs = [stratified_results[s].coefficients.get("attitude", 0) for s in strata]
    r2_values = [stratified_results[s].r_squared for s in strata]

    # Barrier coefficient
    ax1 = axes[0]
    colors1 = ["red" if v < 0 else "green" for v in barrier_coefs]
    ax1.bar(strata, barrier_coefs, color=colors1, alpha=0.7, edgecolor="black")
    ax1.axhline(y=0, color="black", linestyle="--")
    ax1.set_ylabel("Coefficient", fontsize=11)
    ax1.set_title("Barrier Effect by Stratum", fontsize=12)
    ax1.tick_params(axis="x", rotation=15)

    # Attitude coefficient
    ax2 = axes[1]
    colors2 = ["green" if v > 0 else "red" for v in attitude_coefs]
    ax2.bar(strata, attitude_coefs, color=colors2, alpha=0.7, edgecolor="black")
    ax2.axhline(y=0, color="black", linestyle="--")
    ax2.set_ylabel("Coefficient", fontsize=11)
    ax2.set_title("Attitude Effect by Stratum", fontsize=12)
    ax2.tick_params(axis="x", rotation=15)

    # R²
    ax3 = axes[2]
    ax3.bar(strata, r2_values, color="steelblue", alpha=0.7, edgecolor="black")
    ax3.set_ylabel("R²", fontsize=11)
    ax3.set_title("Model Fit by Stratum", fontsize=12)
    ax3.tick_params(axis="x", rotation=15)

    fig.suptitle(f"Stratified Analysis Results{title_suffix}", fontsize=13, fontweight="bold")

    ensure_directory_exists(str(Path(output_path).parent))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Stratified coefficients plot saved to: %s", output_path)
    return output_path


def create_all_interaction_visualizations(
    results: dict[str, Any],
    output_directory: str,
) -> dict[str, str]:
    """
    Generate all interaction analysis visualizations.

    Args:
        results: Complete results from perform_comprehensive_interaction_analysis()
        output_directory: Directory to save visualizations

    Returns:
        Dictionary mapping visualization names to file paths
    """
    logger = get_logger()
    logger.info("Creating all interaction visualizations in: %s", output_directory)

    ensure_directory_exists(output_directory)
    viz_paths = {}

    # 1. Interaction effect plot
    if "main_model" in results:
        path = str(Path(output_directory) / "interaction_effect.png")
        viz_paths["interaction_effect"] = plot_interaction_effect(results["main_model"], path)

    # 2. Interaction heatmap
    if "plot_data" in results:
        path = str(Path(output_directory) / "interaction_heatmap.png")
        viz_paths["interaction_heatmap"] = plot_interaction_heatmap(results["plot_data"], path)

        # 3. Slopes comparison
        path = str(Path(output_directory) / "slopes_comparison.png")
        viz_paths["slopes_comparison"] = plot_slopes_comparison(results["plot_data"], path)

    # 4. Model comparison
    if "main_model" in results and "pooled_model" in results:
        path = str(Path(output_directory) / "model_comparison.png")
        viz_paths["model_comparison"] = plot_model_comparison(
            results["main_model"], results["pooled_model"], path
        )

    # 5. Stratified analysis - by barrier
    if "stratified_by_barrier" in results:
        path = str(Path(output_directory) / "stratified_by_barrier.png")
        viz_paths["stratified_by_barrier"] = plot_stratified_coefficients(
            results["stratified_by_barrier"], path, " (by Barrier Level)"
        )

    # 6. Stratified analysis - by attitude
    if "stratified_by_attitude" in results:
        path = str(Path(output_directory) / "stratified_by_attitude.png")
        viz_paths["stratified_by_attitude"] = plot_stratified_coefficients(
            results["stratified_by_attitude"], path, " (by Attitude Level)"
        )

    logger.info("Created %d visualizations", len(viz_paths))
    return viz_paths


def export_interaction_results_to_csv(
    results: dict[str, Any],
    output_directory: str,
) -> dict[str, str]:
    """
    Export interaction analysis results to CSV files.

    Args:
        results: Complete analysis results
        output_directory: Output directory

    Returns:
        Dictionary mapping result types to CSV paths
    """
    logger = get_logger()
    logger.info("Exporting interaction results to CSV...")

    ensure_directory_exists(output_directory)
    csv_paths = {}

    # Main model coefficients
    if "main_model" in results:
        main_model = results["main_model"]
        rows = [
            {"Coefficient": name, "Value": value} for name, value in main_model.coefficients.items()
        ]
        rows.append({"Coefficient": "R_squared", "Value": main_model.r_squared})
        rows.append({"Coefficient": "Adj_R_squared", "Value": main_model.adj_r_squared})
        rows.append({"Coefficient": "RMSE", "Value": main_model.rmse})
        rows.append({"Coefficient": "Sample_Size", "Value": main_model.sample_size})

        df = pd.DataFrame(rows)
        path = str(Path(output_directory) / "main_model_coefficients.csv")
        df.to_csv(path, index=False)
        csv_paths["main_model"] = path

    # Marginal effects
    if "main_model" in results:
        main_model = results["main_model"]
        rows = [
            {
                "Barrier_Level": level,
                "Barrier_Value": data["barrier_level"],
                "Attitude_Effect": data["attitude_effect"],
            }
            for level, data in main_model.marginal_effects.items()
        ]

        df = pd.DataFrame(rows)
        path = str(Path(output_directory) / "marginal_effects.csv")
        df.to_csv(path, index=False)
        csv_paths["marginal_effects"] = path

    # Stratified results
    for key in ["stratified_by_barrier", "stratified_by_attitude"]:
        if key in results:
            rows = []
            for stratum, stratum_results in results[key].items():
                row = {
                    "Stratum": stratum,
                    "N": stratum_results.sample_size,
                    "R_squared": stratum_results.r_squared,
                    "Barrier_Coef": stratum_results.coefficients.get("barrier", None),
                    "Attitude_Coef": stratum_results.coefficients.get("attitude", None),
                    "Interaction_Coef": stratum_results.interaction_effect,
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            path = str(Path(output_directory) / f"{key}.csv")
            df.to_csv(path, index=False)
            csv_paths[key] = path

    logger.info("Exported %d CSV files", len(csv_paths))
    return csv_paths
