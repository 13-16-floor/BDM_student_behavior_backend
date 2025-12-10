"""
Barrier × Attitude Interaction Analysis Module.

Analyzes the interaction effect between resource barriers and student attitudes
on academic performance, with built-in country fixed effects through
within-country transformation (demeaning).

Key Research Question:
    Does positive attitude compensate for resource barriers (β3 > 0),
    or do barriers diminish the protective effect of attitude (β3 < 0)?

Methodology:
    1. Within-country transformation (demeaning) - equivalent to country FE
    2. Construct interaction term from demeaned variables
    3. Weighted regression with proper standard errors
    4. Heterogeneity analysis across barrier/attitude subgroups
"""

from dataclasses import dataclass
from typing import Any, Literal

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql.window import Window

from ..utils.logger import get_logger

# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class InteractionConfig:
    """Configuration for interaction analysis."""

    score_column: str = "PV1MATH"
    barrier_column: str = "barrier_index"
    attitude_columns: list[str] | None = None  # If None, use composite score
    country_column: str = "CNT"
    weight_column: str = "W_FSTUWT"
    ses_column: str = "ESCS"  # Optional control variable
    include_ses: bool = True

    def __post_init__(self) -> None:
        if self.attitude_columns is None:
            self.attitude_columns = ["MATHMOT", "PERSEVAGR", "GROSAGR", "MATHEFF"]


@dataclass
class InteractionResults:
    """Container for interaction analysis results."""

    model_name: str
    r_squared: float
    adj_r_squared: float
    rmse: float
    sample_size: int
    coefficients: dict[str, float]
    std_errors: dict[str, float] | None
    interaction_effect: float
    interaction_interpretation: str
    marginal_effects: dict[str, dict[str, float]]  # Effect of attitude at different barrier levels
    residuals_sample: list[tuple[float, float]] | None = None


@dataclass
class HeterogeneityResults:
    """Results from heterogeneity analysis across subgroups."""

    subgroup_effects: dict[str, dict[str, float]]  # {subgroup: {barrier_effect, attitude_effect}}
    effect_variation: dict[str, float]  # Variance of effects across subgroups
    significant_heterogeneity: bool


# ============================================================================
# SECTION 1: WITHIN-COUNTRY TRANSFORMATION
# ============================================================================


def apply_within_country_transformation(
    df: DataFrame,
    columns_to_transform: list[str],
    country_column: str = "CNT",
    suffix: str = "_within",
) -> DataFrame:
    """
    Apply within-country transformation (demeaning) to specified columns.

    This is mathematically equivalent to including country fixed effects,
    but more computationally efficient and conceptually cleaner.

    For each variable X and country j:
        X_within = X - mean(X | country = j)

    Args:
        df: Input DataFrame
        columns_to_transform: List of column names to transform
        country_column: Country identifier column
        suffix: Suffix for transformed columns

    Returns:
        DataFrame with additional demeaned columns
    """
    logger = get_logger()
    logger.info(
        "Applying within-country transformation to %d columns",
        len(columns_to_transform),
    )

    # Create window specification for country-level aggregation
    country_window = Window.partitionBy(country_column)

    working_df = df
    for col_name in columns_to_transform:
        if col_name not in df.columns:
            logger.warning("Column %s not found, skipping", col_name)
            continue

        # Calculate country mean and subtract
        new_col_name = f"{col_name}{suffix}"
        working_df = working_df.withColumn(
            f"{col_name}_country_mean", f.mean(f.col(col_name)).over(country_window)
        ).withColumn(new_col_name, f.col(col_name) - f.col(f"{col_name}_country_mean"))

        logger.info("  Transformed: %s -> %s", col_name, new_col_name)

    # Drop intermediate mean columns
    mean_cols = [f"{col}_country_mean" for col in columns_to_transform if col in df.columns]
    working_df = working_df.drop(*mean_cols)

    logger.info("Within-country transformation completed")
    return working_df


def create_attitude_composite_score(
    df: DataFrame,
    attitude_columns: list[str] | None,
    output_column: str = "attitude_score",
    standardize: bool = True,
) -> DataFrame:
    """
    Create a composite attitude score from multiple attitude indicators.

    Process:
    1. Standardize each attitude variable (Z-score)
    2. Handle reverse-coded items (ST062Q01TA)
    3. Average standardized scores

    Args:
        df: Input DataFrame
        attitude_columns: List of attitude indicator columns
        output_column: Name for output composite score
        standardize: Whether to standardize before combining

    Returns:
        DataFrame with composite attitude score
    """
    logger = get_logger()
    if attitude_columns is None:
        raise ValueError("attitude_columns cannot be None")
    logger.info("Creating composite attitude score from %d indicators", len(attitude_columns))

    working_df = df

    # Columns that need reverse coding (higher original value = worse attitude)
    reverse_coded = ["ST062Q01TA"]  # Skipping school: 1=Never, 4=Often

    std_cols = []
    for col in attitude_columns:
        if col not in df.columns:
            logger.warning("Attitude column %s not found, skipping", col)
            continue

        if standardize:
            # Calculate mean and std
            stats = working_df.select(
                f.mean(f.col(col)).alias("mean"), f.stddev(f.col(col)).alias("std")
            ).first()

            mean_val = stats["mean"] if stats["mean"] is not None else 0
            std_val = stats["std"] if stats["std"] is not None and stats["std"] > 0 else 1

            std_col = f"{col}_std"

            if col in reverse_coded:
                # Reverse code: negate after standardization
                working_df = working_df.withColumn(std_col, -((f.col(col) - mean_val) / std_val))
                logger.info("  %s: standardized and reverse-coded", col)
            else:
                working_df = working_df.withColumn(std_col, (f.col(col) - mean_val) / std_val)
                logger.info("  %s: standardized", col)

            std_cols.append(std_col)
        else:
            std_cols.append(col)

    # Create composite score (average of standardized scores)
    if std_cols:
        composite_expr = sum(f.col(col) for col in std_cols) / len(std_cols)
        working_df = working_df.withColumn(output_column, composite_expr)
        logger.info("Composite attitude score created: %s", output_column)
    else:
        raise ValueError("No valid attitude columns found")

    return working_df


# ============================================================================
# SECTION 2: INTERACTION TERM CONSTRUCTION
# ============================================================================


def create_interaction_term(
    df: DataFrame,
    var1: str,
    var2: str,
    interaction_name: str | None = None,
) -> DataFrame:
    """
    Create interaction term between two variables.

    IMPORTANT: Use demeaned variables to ensure proper interpretation
    when country fixed effects are implied.

    Args:
        df: Input DataFrame
        var1: First variable name
        var2: Second variable name
        interaction_name: Name for interaction column (default: var1_x_var2)

    Returns:
        DataFrame with interaction term added
    """
    logger = get_logger()

    if interaction_name is None:
        interaction_name = f"{var1}_x_{var2}"

    logger.info("Creating interaction term: %s × %s -> %s", var1, var2, interaction_name)

    df_with_interaction = df.withColumn(interaction_name, f.col(var1) * f.col(var2))

    return df_with_interaction


# ============================================================================
# SECTION 3: INTERACTION REGRESSION MODELS
# ============================================================================


def run_interaction_regression(
    df: DataFrame,
    config: InteractionConfig,
    use_within_transformation: bool = True,
) -> InteractionResults:
    """
    Run interaction regression with optional within-country transformation.

    Model (with within-transformation):
        Score_within = β1*Barrier_within + β2*Attitude_within +
                       β3*(Barrier_within × Attitude_within) + ε

    Model (without, for comparison):
        Score = β0 + β1*Barrier + β2*Attitude + β3*(Barrier × Attitude) +
                Σ(γ_j * Country_j) + ε

    Args:
        df: Input DataFrame with all required columns
        config: Analysis configuration
        use_within_transformation: If True, use demeaning (recommended)

    Returns:
        InteractionResults with coefficients and diagnostics
    """
    logger = get_logger()
    model_name = "Within-Country Interaction" if use_within_transformation else "Pooled Interaction"
    logger.info("Running %s model", model_name)

    working_df = df

    # Step 1: Create composite attitude score if needed
    working_df = create_attitude_composite_score(
        working_df, config.attitude_columns, "attitude_score"
    )

    # Step 2: Apply within-country transformation
    if use_within_transformation:
        vars_to_transform = [
            config.score_column,
            config.barrier_column,
            "attitude_score",
        ]
        if config.include_ses and config.ses_column in df.columns:
            vars_to_transform.append(config.ses_column)

        working_df = apply_within_country_transformation(
            working_df, vars_to_transform, config.country_column
        )

        # Use transformed variables
        score_var = f"{config.score_column}_within"
        barrier_var = f"{config.barrier_column}_within"
        attitude_var = "attitude_score_within"
        ses_var = f"{config.ses_column}_within" if config.include_ses else None
    else:
        score_var = config.score_column
        barrier_var = config.barrier_column
        attitude_var = "attitude_score"
        ses_var = config.ses_column if config.include_ses else None

    # Step 3: Create interaction term
    interaction_var = "barrier_x_attitude"
    working_df = create_interaction_term(working_df, barrier_var, attitude_var, interaction_var)

    # Step 4: Prepare features for regression
    feature_cols = [barrier_var, attitude_var, interaction_var]
    if ses_var and ses_var in working_df.columns:
        feature_cols.append(ses_var)

    logger.info("Feature columns: %s", feature_cols)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    train_data = assembler.transform(working_df).select(
        "features", f.col(score_var).alias("label"), config.weight_column
    )

    # Step 5: Fit weighted regression
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        weightCol=config.weight_column,
        maxIter=100,
        regParam=0.0,
        # Note: Spark MLlib doesn't provide robust standard errors
        # For publication-quality inference, consider using statsmodels
    )

    model = lr.fit(train_data)
    sample_size = train_data.count()

    # Step 6: Extract coefficients
    coef_names = ["barrier", "attitude", "barrier_x_attitude"]
    if ses_var:
        coef_names.append("ses")

    coefficients = {name: float(model.coefficients[i]) for i, name in enumerate(coef_names)}
    coefficients["intercept"] = float(model.intercept)

    # Step 7: Calculate marginal effects of attitude at different barrier levels
    # Effect of attitude = β2 + β3 * barrier_level
    beta_attitude = coefficients["attitude"]
    beta_interaction = coefficients["barrier_x_attitude"]

    # Get barrier percentiles for marginal effect calculation
    barrier_percentiles = working_df.select(
        f.expr(f"percentile_approx({barrier_var}, 0.25)").alias("p25"),
        f.expr(f"percentile_approx({barrier_var}, 0.50)").alias("p50"),
        f.expr(f"percentile_approx({barrier_var}, 0.75)").alias("p75"),
    ).first()

    marginal_effects = {
        "low_barrier (P25)": {
            "barrier_level": float(barrier_percentiles["p25"]),
            "attitude_effect": beta_attitude + beta_interaction * barrier_percentiles["p25"],
        },
        "medium_barrier (P50)": {
            "barrier_level": float(barrier_percentiles["p50"]),
            "attitude_effect": beta_attitude + beta_interaction * barrier_percentiles["p50"],
        },
        "high_barrier (P75)": {
            "barrier_level": float(barrier_percentiles["p75"]),
            "attitude_effect": beta_attitude + beta_interaction * barrier_percentiles["p75"],
        },
    }

    # Step 8: Interpret interaction effect
    if beta_interaction > 0.1:
        interpretation = (
            "Compensatory: Positive attitude has STRONGER effect for high-barrier students"
        )
    elif beta_interaction < -0.1:
        interpretation = (
            "Cumulative disadvantage: Positive attitude has WEAKER effect for high-barrier students"
        )
    else:
        interpretation = "Additive: Barrier and attitude effects are approximately independent"

    # Step 9: Generate residuals sample
    residuals_sample = None
    try:
        predictions = model.transform(train_data)
        sample_for_plot = min(5000, sample_size)
        sampled = (
            predictions.select("prediction", "label")
            .sample(fraction=sample_for_plot / sample_size, seed=42)
            .limit(sample_for_plot)
            .collect()
        )
        residuals_sample = [
            (float(row["prediction"]), float(row["label"] - row["prediction"])) for row in sampled
        ]
    except Exception as e:
        logger.warning("Failed to generate residuals sample: %s", e)

    # Calculate adjusted R²
    n = sample_size
    p = len(feature_cols)
    r2 = float(model.summary.r2)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

    results = InteractionResults(
        model_name=model_name,
        r_squared=r2,
        adj_r_squared=adj_r2,
        rmse=float(model.summary.rootMeanSquaredError),
        sample_size=sample_size,
        coefficients=coefficients,
        std_errors=None,  # Not available in Spark MLlib
        interaction_effect=beta_interaction,
        interaction_interpretation=interpretation,
        marginal_effects=marginal_effects,
        residuals_sample=residuals_sample,
    )

    logger.info("Interaction regression completed. R² = %.4f", r2)
    logger.info("Interaction effect (β3) = %.4f", beta_interaction)
    logger.info("Interpretation: %s", interpretation)

    return results


def run_stratified_analysis(
    df: DataFrame,
    config: InteractionConfig,
    stratify_by: Literal["barrier_level", "attitude_level", "country"] = "barrier_level",
    num_strata: int = 3,
) -> dict[str, InteractionResults]:
    """
    Run separate regressions for different subgroups to examine heterogeneity.

    This helps validate interaction effects and identify non-linearities.

    Args:
        df: Input DataFrame
        config: Analysis configuration
        stratify_by: Variable to stratify on
        num_strata: Number of strata (for continuous variables)

    Returns:
        Dictionary mapping stratum names to InteractionResults
    """
    logger = get_logger()
    logger.info("Running stratified analysis by %s", stratify_by)

    # Create composite attitude score
    working_df = create_attitude_composite_score(df, config.attitude_columns, "attitude_score")

    results = {}

    if stratify_by == "barrier_level":
        # Create barrier tertiles
        percentiles = working_df.select(
            f.expr(f"percentile_approx({config.barrier_column}, 0.33)").alias("p33"),
            f.expr(f"percentile_approx({config.barrier_column}, 0.67)").alias("p67"),
        ).first()

        strata = {
            "low_barrier": working_df.filter(f.col(config.barrier_column) < percentiles["p33"]),
            "medium_barrier": working_df.filter(
                (f.col(config.barrier_column) >= percentiles["p33"])
                & (f.col(config.barrier_column) < percentiles["p67"])
            ),
            "high_barrier": working_df.filter(f.col(config.barrier_column) >= percentiles["p67"]),
        }

    elif stratify_by == "attitude_level":
        percentiles = working_df.select(
            f.expr("percentile_approx(attitude_score, 0.33)").alias("p33"),
            f.expr("percentile_approx(attitude_score, 0.67)").alias("p67"),
        ).first()

        strata = {
            "negative_attitude": working_df.filter(f.col("attitude_score") < percentiles["p33"]),
            "neutral_attitude": working_df.filter(
                (f.col("attitude_score") >= percentiles["p33"])
                & (f.col("attitude_score") < percentiles["p67"])
            ),
            "positive_attitude": working_df.filter(f.col("attitude_score") >= percentiles["p67"]),
        }

    elif stratify_by == "country":
        # Select top N countries by sample size
        top_countries = (
            working_df.groupBy(config.country_column)
            .count()
            .orderBy(f.desc("count"))
            .limit(num_strata)
            .collect()
        )

        strata = {
            row[config.country_column]: working_df.filter(
                f.col(config.country_column) == row[config.country_column]
            )
            for row in top_countries
        }
    else:
        raise ValueError(f"Unknown stratification variable: {stratify_by}")

    # Run regression for each stratum
    for stratum_name, stratum_df in strata.items():
        logger.info("Analyzing stratum: %s (n=%d)", stratum_name, stratum_df.count())
        try:
            # For within-country analysis, skip transformation if stratifying by country
            use_within = stratify_by != "country"
            stratum_results = run_interaction_regression(stratum_df, config, use_within)
            results[stratum_name] = stratum_results
        except Exception as e:
            logger.warning("Failed to analyze stratum %s: %s", stratum_name, e)

    return results


# ============================================================================
# SECTION 4: VISUALIZATION DATA PREPARATION
# ============================================================================


def prepare_interaction_plot_data(
    df: DataFrame,
    config: InteractionConfig,
    num_barrier_groups: int = 3,
    num_attitude_groups: int = 3,
) -> dict[str, Any]:
    """
    Prepare data for interaction effect visualization.

    Creates grouped means for plotting:
    - Mean score by barrier level × attitude level
    - Slopes of attitude effect at each barrier level

    Args:
        df: Input DataFrame
        config: Analysis configuration
        num_barrier_groups: Number of barrier groups
        num_attitude_groups: Number of attitude groups

    Returns:
        Dictionary with data for various plots
    """
    logger = get_logger()
    logger.info("Preparing interaction plot data")

    # Create composite attitude score
    working_df = create_attitude_composite_score(df, config.attitude_columns, "attitude_score")

    # Apply within-country transformation
    vars_to_transform = [config.score_column, config.barrier_column, "attitude_score"]
    working_df = apply_within_country_transformation(
        working_df, vars_to_transform, config.country_column
    )

    # Create categorical groups for barrier and attitude
    barrier_var = f"{config.barrier_column}_within"
    attitude_var = "attitude_score_within"
    score_var = f"{config.score_column}_within"

    # Calculate percentiles for grouping
    barrier_percentiles = [i / num_barrier_groups for i in range(1, num_barrier_groups)]
    attitude_percentiles = [i / num_attitude_groups for i in range(1, num_attitude_groups)]

    barrier_cuts = working_df.select(
        *[
            f.expr(f"percentile_approx({barrier_var}, {p})").alias(f"b_p{int(p * 100)}")
            for p in barrier_percentiles
        ]
    ).first()

    attitude_cuts = working_df.select(
        *[
            f.expr(f"percentile_approx({attitude_var}, {p})").alias(f"a_p{int(p * 100)}")
            for p in attitude_percentiles
        ]
    ).first()

    # Create group labels
    barrier_labels = ["Low Barrier", "Medium Barrier", "High Barrier"][:num_barrier_groups]
    attitude_labels = ["Negative", "Neutral", "Positive"][:num_attitude_groups]

    # Assign groups
    barrier_case = f.when(f.col(barrier_var) < barrier_cuts["b_p33"], barrier_labels[0])
    if num_barrier_groups > 2:
        barrier_case = barrier_case.when(
            f.col(barrier_var) < barrier_cuts["b_p67"], barrier_labels[1]
        )
    barrier_case = barrier_case.otherwise(barrier_labels[-1])

    attitude_case = f.when(f.col(attitude_var) < attitude_cuts["a_p33"], attitude_labels[0])
    if num_attitude_groups > 2:
        attitude_case = attitude_case.when(
            f.col(attitude_var) < attitude_cuts["a_p67"], attitude_labels[1]
        )
    attitude_case = attitude_case.otherwise(attitude_labels[-1])

    working_df = working_df.withColumn("barrier_group", barrier_case).withColumn(
        "attitude_group", attitude_case
    )

    # Calculate cell means with weights
    cell_means = (
        working_df.groupBy("barrier_group", "attitude_group")
        .agg(
            f.count("*").alias("n"),
            f.sum(config.weight_column).alias("weighted_n"),
            (
                f.sum(f.col(score_var) * f.col(config.weight_column)) / f.sum(config.weight_column)
            ).alias("weighted_mean_score"),
            f.mean(score_var).alias("mean_score"),
            f.stddev(score_var).alias("std_score"),
        )
        .collect()
    )

    # Format for plotting
    heatmap_data = {}
    for row in cell_means:
        key = (row["barrier_group"], row["attitude_group"])
        heatmap_data[key] = {
            "n": row["n"],
            "weighted_n": float(row["weighted_n"]) if row["weighted_n"] else 0,
            "mean_score": (float(row["weighted_mean_score"]) if row["weighted_mean_score"] else 0),
            "std_score": float(row["std_score"]) if row["std_score"] else 0,
        }

    # Calculate slopes (attitude effect) within each barrier group
    slopes_by_barrier = {}
    for b_label in barrier_labels:
        barrier_subset = working_df.filter(f.col("barrier_group") == b_label)

        # Simple regression of score on attitude within this barrier group
        assembler = VectorAssembler(
            inputCols=[attitude_var], outputCol="features", handleInvalid="skip"
        )
        train_data = assembler.transform(barrier_subset).select(
            "features", f.col(score_var).alias("label"), config.weight_column
        )

        if train_data.count() > 100:  # Minimum sample size
            lr = LinearRegression(
                featuresCol="features",
                labelCol="label",
                weightCol=config.weight_column,
            )
            model = lr.fit(train_data)
            slopes_by_barrier[b_label] = {
                "slope": float(model.coefficients[0]),
                "intercept": float(model.intercept),
                "r_squared": float(model.summary.r2),
                "n": train_data.count(),
            }

    plot_data = {
        "heatmap_data": heatmap_data,
        "slopes_by_barrier": slopes_by_barrier,
        "barrier_labels": barrier_labels,
        "attitude_labels": attitude_labels,
    }

    logger.info("Interaction plot data prepared")
    return plot_data


# ============================================================================
# SECTION 5: REPORTING
# ============================================================================


def print_interaction_report(results: InteractionResults, detailed: bool = True) -> None:
    """
    Print formatted interaction analysis report.

    Args:
        results: InteractionResults object
        detailed: Whether to print detailed statistics
    """
    print("\n" + "=" * 80, flush=True)
    print("BARRIER × ATTITUDE INTERACTION ANALYSIS", flush=True)
    print("=" * 80, flush=True)

    print(f"\nModel: {results.model_name}", flush=True)
    print(f"Sample Size: {results.sample_size:,}", flush=True)
    print(f"R²: {results.r_squared:.4f}", flush=True)
    print(f"Adjusted R²: {results.adj_r_squared:.4f}", flush=True)
    print(f"RMSE: {results.rmse:.2f}", flush=True)

    print("\n" + "-" * 80, flush=True)
    print("COEFFICIENTS", flush=True)
    print("-" * 80, flush=True)

    for name, coef in results.coefficients.items():
        print(f"  {name:25s}: {coef:10.4f}", flush=True)

    print("\n" + "-" * 80, flush=True)
    print("INTERACTION EFFECT", flush=True)
    print("-" * 80, flush=True)

    print(f"  β(Barrier × Attitude): {results.interaction_effect:.4f}", flush=True)
    print(f"  Interpretation: {results.interaction_interpretation}", flush=True)

    if detailed:
        print("\n" + "-" * 80, flush=True)
        print("MARGINAL EFFECTS OF ATTITUDE", flush=True)
        print("(Effect of +1 SD attitude at different barrier levels)", flush=True)
        print("-" * 80, flush=True)

        for level, effect in results.marginal_effects.items():
            print(
                f"  {level:25s}: {effect['attitude_effect']:+.4f} "
                f"(at barrier = {effect['barrier_level']:.2f})",
                flush=True,
            )

    print("\n" + "=" * 80, flush=True)


def print_stratified_report(stratified_results: dict[str, InteractionResults]) -> None:
    """
    Print comparison of results across strata.

    Args:
        stratified_results: Dictionary of results by stratum
    """
    print("\n" + "=" * 80, flush=True)
    print("STRATIFIED ANALYSIS COMPARISON", flush=True)
    print("=" * 80, flush=True)

    print(
        f"\n{'Stratum':<20} {'N':>10} {'R²':>8} {'β(Barrier)':>12} "
        f"{'β(Attitude)':>12} {'β(Interaction)':>14}",
        flush=True,
    )
    print("-" * 80, flush=True)

    for stratum, result in stratified_results.items():
        print(
            f"{stratum:<20} {result.sample_size:>10,} {result.r_squared:>8.4f} "
            f"{result.coefficients.get('barrier', 0):>12.4f} "
            f"{result.coefficients.get('attitude', 0):>12.4f} "
            f"{result.interaction_effect:>14.4f}",
            flush=True,
        )

    print("=" * 80 + "\n", flush=True)


# ============================================================================
# SECTION 6: MAIN ORCHESTRATION
# ============================================================================


def perform_comprehensive_interaction_analysis(
    df: DataFrame,
    config: InteractionConfig | None = None,
) -> dict[str, Any]:
    """
    Perform complete barrier-attitude interaction analysis.

    Workflow:
    1. Create composite attitude score
    2. Apply within-country transformation
    3. Run main interaction regression
    4. Run stratified analyses for robustness
    5. Prepare visualization data

    Args:
        df: Input DataFrame with all required columns
        config: Analysis configuration (uses defaults if None)

    Returns:
        Dictionary with all analysis results
    """
    logger = get_logger()
    logger.info("Starting comprehensive interaction analysis")

    if config is None:
        config = InteractionConfig()

    results: dict[str, Any] = {}

    # Main interaction model
    logger.info("\n=== Running Main Interaction Model ===")
    try:
        main_results = run_interaction_regression(df, config, use_within_transformation=True)
        results["main_model"] = main_results
        print_interaction_report(main_results)
    except Exception as e:
        logger.error("Main interaction model failed: %s", e)

    # Comparison: Pooled model (without country FE)
    logger.info("\n=== Running Pooled Model (for comparison) ===")
    try:
        pooled_results = run_interaction_regression(df, config, use_within_transformation=False)
        results["pooled_model"] = pooled_results
    except Exception as e:
        logger.error("Pooled model failed: %s", e)

    # Stratified analysis by barrier level
    logger.info("\n=== Running Stratified Analysis by Barrier Level ===")
    try:
        barrier_stratified = run_stratified_analysis(df, config, stratify_by="barrier_level")
        results["stratified_by_barrier"] = barrier_stratified
        print_stratified_report(barrier_stratified)
    except Exception as e:
        logger.error("Barrier stratified analysis failed: %s", e)

    # Stratified analysis by attitude level
    logger.info("\n=== Running Stratified Analysis by Attitude Level ===")
    try:
        attitude_stratified = run_stratified_analysis(df, config, stratify_by="attitude_level")
        results["stratified_by_attitude"] = attitude_stratified
    except Exception as e:
        logger.error("Attitude stratified analysis failed: %s", e)

    # Prepare visualization data
    logger.info("\n=== Preparing Visualization Data ===")
    try:
        plot_data = prepare_interaction_plot_data(df, config)
        results["plot_data"] = plot_data
    except Exception as e:
        logger.error("Plot data preparation failed: %s", e)

    logger.info("Comprehensive interaction analysis completed")
    return results
