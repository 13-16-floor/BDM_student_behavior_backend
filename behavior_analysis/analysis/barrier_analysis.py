"""
Barrier analysis module for PISA student data.

Implements comprehensive barrier analysis including:
1. Barrier index construction from 4 dimensions
2. Regression analysis of barriers on academic performance
3. Distribution analysis across score clusters
4. Feature importance using Random Forest
5. K-means clustering by barrier profile
"""

from dataclasses import dataclass
from typing import Any, Literal

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from ..utils.logger import get_logger

# Barrier dimensions configuration
BARRIER_DIMENSIONS: dict[str, dict[str, Any]] = {
    "access_to_resources": {
        "name": "Access to Resources",
        "columns": ["HEDRES", "ST011Q01TA", "ST011Q02TA", "ST011Q03TA", "ST013Q01TA"],
        "weight": 0.25,
        "reverse_code": True,  # Higher values = more resources = lower barriers
    },
    "internet_access": {
        "name": "Internet Access",
        "columns": ["ICTRES", "ST011Q06TA", "IC001Q01TA"],
        "weight": 0.25,
        "reverse_code": True,  # Higher values = better access = lower barriers
    },
    "learning_disabilities": {
        "name": "Learning Disabilities (Proxy)",
        "columns": ["ST127Q01TA", "ST127Q02TA", "ST118Q01NA", "ST119Q01NA"],
        "weight": 0.25,
        "reverse_code": False,  # Higher values = more disabilities = higher barriers
    },
    "geographic_isolation": {
        "name": "Geographic Isolation (Proxy)",
        "columns": ["SC001Q01TA"],
        "weight": 0.25,
        "reverse_code": False,  # Lower values (village) = more isolation = higher barriers
    },
}


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class BarrierIndexConfig:
    """Configuration for barrier index calculation."""

    access_to_resources_cols: list[str]
    internet_access_cols: list[str]
    learning_disabilities_cols: list[str]
    geographic_isolation_cols: list[str]
    weights: dict[str, float]
    handle_missing: Literal["drop", "impute_mean", "impute_zero"]
    standardize: bool = True


@dataclass
class RegressionResults:
    """Container for regression analysis results."""

    model_name: str
    r_squared: float
    rmse: float
    coefficients: dict[str, float]
    p_values: dict[str, float] | None
    sample_size: int
    residuals_sample: list[tuple[float, float]] | None = None  # (fitted, residual) pairs


@dataclass
class FeatureImportanceResults:
    """Container for feature importance analysis results."""

    method: Literal["random_forest", "standardized_regression"]
    feature_importances: dict[str, float]
    model_accuracy: float | None
    top_n_features: list[tuple[str, float]]


@dataclass
class BarrierClusterResults:
    """Results from k-means clustering on barrier profiles."""

    num_clusters: int
    cluster_centers: dict[str, dict[str, float]]
    cluster_sizes: dict[str, int]
    cluster_characteristics: dict[str, str]
    silhouette_score: float | None
    inertia: float


# ============================================================================
# SECTION 1: VALIDATION AND UTILITY FUNCTIONS
# ============================================================================


def validate_required_columns(df: DataFrame, required_cols: list[str]) -> tuple[bool, list[str]]:
    """
    Validate that all required columns exist in DataFrame.

    Args:
        df: Input Spark DataFrame
        required_cols: List of required column names

    Returns:
        Tuple of (all_valid, missing_columns)
    """
    logger = get_logger()
    existing_cols = set(df.columns)
    missing_cols = [col for col in required_cols if col not in existing_cols]

    if missing_cols:
        logger.warning("Missing required columns: %s", missing_cols)
        return (False, missing_cols)

    logger.info("All required columns validated successfully")
    return (True, [])


def standardize_column(df: DataFrame, column_name: str, output_col: str | None = None) -> DataFrame:
    """
    Standardize a column using z-score normalization.

    Args:
        df: Input DataFrame
        column_name: Column to standardize
        output_col: Output column name (default: {column_name}_std)

    Returns:
        DataFrame with standardized column added
    """
    if output_col is None:
        output_col = f"{column_name}_std"

    # Calculate mean and stddev
    stats = df.select(
        f.mean(f.col(column_name)).alias("mean"),
        f.stddev(f.col(column_name)).alias("std"),
    ).first()

    mean_val = stats["mean"]
    std_val = stats["std"]

    # Avoid division by zero
    if std_val is None or std_val == 0:
        return df.withColumn(output_col, f.lit(0.0))

    # Standardize: (x - mean) / std
    return df.withColumn(output_col, (f.col(column_name) - mean_val) / std_val)


def get_all_barrier_columns(config: BarrierIndexConfig) -> list[str]:
    """
    Get all barrier columns from configuration.

    Args:
        config: Barrier index configuration

    Returns:
        List of all barrier column names
    """
    return (
        config.access_to_resources_cols
        + config.internet_access_cols
        + config.learning_disabilities_cols
        + config.geographic_isolation_cols
    )


# ============================================================================
# SECTION 2: BARRIER INDEX CONSTRUCTION
# ============================================================================


def calculate_dimension_score(
    df: DataFrame,
    columns: list[str],
    dimension_name: str,
    standardize: bool = True,
    reverse_coding: bool = False,
    handle_missing: Literal["drop", "impute_mean", "impute_zero"] = "impute_mean",
) -> DataFrame:
    """
    Calculate score for a single barrier dimension.

    Args:
        df: Input DataFrame
        columns: List of column names in this dimension
        dimension_name: Name for the output dimension column
        standardize: Whether to standardize before averaging
        reverse_coding: If True, reverse the scale (for positive indicators)
        handle_missing: Strategy for handling missing values

    Returns:
        DataFrame with dimension score column added
    """
    logger = get_logger()
    logger.info("Calculating dimension score for: %s", dimension_name)

    # Filter columns that exist
    existing_cols = [col for col in columns if col in df.columns]
    if not existing_cols:
        raise ValueError(f"No valid columns found for dimension {dimension_name}")

    # Standardize each column if requested
    working_df = df
    std_cols = []

    for col in existing_cols:
        if standardize:
            std_col_name = f"{col}_std_{dimension_name}"
            working_df = standardize_column(working_df, col, std_col_name)
            std_cols.append(std_col_name)
        else:
            std_cols.append(col)

    # Handle missing values based on strategy
    if handle_missing == "impute_mean":
        # Calculate mean for imputation
        for col in std_cols:
            mean_val = working_df.select(f.mean(f.col(col))).first()[0]
            if mean_val is not None:
                working_df = working_df.withColumn(
                    col, f.when(f.col(col).isNull(), mean_val).otherwise(f.col(col))
                )
    elif handle_missing == "impute_zero":
        for col in std_cols:
            working_df = working_df.withColumn(
                col, f.when(f.col(col).isNull(), 0.0).otherwise(f.col(col))
            )

    # Calculate average across columns
    dimension_expr = sum(f.col(col) for col in std_cols) / len(std_cols)

    # Apply reverse coding if needed
    if reverse_coding:
        dimension_expr = -dimension_expr

    working_df = working_df.withColumn(dimension_name, dimension_expr)

    logger.info("Dimension score calculated: %s", dimension_name)
    return working_df


def construct_barrier_index(
    df: DataFrame,
    config: BarrierIndexConfig,
    index_column: str = "barrier_index",
) -> DataFrame:
    """
    Construct comprehensive barrier index from all dimensions.

    Process:
    1. Validate all required columns exist
    2. Standardize each variable within dimensions
    3. Calculate dimension-level scores
    4. Apply dimension weights
    5. Create composite index
    6. Scale to 0-100 range

    Args:
        df: Input Spark DataFrame
        config: Barrier index configuration
        index_column: Name for output barrier index column (default: barrier_index)

    Returns:
        DataFrame with barrier index and dimension scores added

    Raises:
        ValueError: If required columns are missing
    """
    logger = get_logger()
    logger.info("Constructing barrier index...")

    # Validate all barrier columns exist
    all_cols = get_all_barrier_columns(config)
    all_valid, missing_cols = validate_required_columns(df, all_cols)

    if not all_valid:
        raise ValueError(f"Cannot construct barrier index. Missing columns: {missing_cols}")

    # Calculate dimension scores
    working_df = df

    # Access to Resources
    working_df = calculate_dimension_score(
        working_df,
        config.access_to_resources_cols,
        "dim_access_resources",
        standardize=config.standardize,
        reverse_coding=True,  # Higher resources = lower barriers
        handle_missing=config.handle_missing,
    )

    # Internet Access
    working_df = calculate_dimension_score(
        working_df,
        config.internet_access_cols,
        "dim_internet_access",
        standardize=config.standardize,
        reverse_coding=True,  # Better access = lower barriers
        handle_missing=config.handle_missing,
    )

    # Learning Disabilities
    working_df = calculate_dimension_score(
        working_df,
        config.learning_disabilities_cols,
        "dim_learning_disabilities",
        standardize=config.standardize,
        reverse_coding=False,  # Higher disabilities = higher barriers
        handle_missing=config.handle_missing,
    )

    # Geographic Isolation
    working_df = calculate_dimension_score(
        working_df,
        config.geographic_isolation_cols,
        "dim_geographic_isolation",
        standardize=config.standardize,
        reverse_coding=True,  # Reverse: 1(village)=high barrier, 5(city)=low barrier
        handle_missing=config.handle_missing,
    )

    # Combine dimensions with weights
    barrier_expr = (
        config.weights.get("access_to_resources", 0.25) * f.col("dim_access_resources")
        + config.weights.get("internet_access", 0.25) * f.col("dim_internet_access")
        + config.weights.get("learning_disabilities", 0.25) * f.col("dim_learning_disabilities")
        + config.weights.get("geographic_isolation", 0.25) * f.col("dim_geographic_isolation")
    )

    working_df = working_df.withColumn("barrier_raw", barrier_expr)

    # Scale to 0-100 range
    # First, normalize to 0-1 using min-max scaling
    min_max = working_df.select(
        f.min("barrier_raw").alias("min_val"), f.max("barrier_raw").alias("max_val")
    ).first()

    min_val = min_max["min_val"]
    max_val = min_max["max_val"]

    if max_val != min_val:
        working_df = working_df.withColumn(
            index_column,
            ((f.col("barrier_raw") - min_val) / (max_val - min_val)) * 100,
        )
    else:
        working_df = working_df.withColumn(index_column, f.lit(50.0))

    # Drop intermediate column
    working_df = working_df.drop("barrier_raw")

    logger.info("Barrier index constructed successfully")
    return working_df


def categorize_barrier_level(
    df: DataFrame,
    index_column: str = "barrier_index",
    thresholds: dict[str, tuple[float, float]] | None = None,
) -> DataFrame:
    """
    Categorize barrier index into low/medium/high levels.

    Args:
        df: DataFrame with barrier index
        index_column: Name of barrier index column
        thresholds: Custom thresholds (default: tertiles at 33.33, 66.67)

    Returns:
        DataFrame with barrier_level column added
    """
    logger = get_logger()
    logger.info("Categorizing barrier levels...")

    if thresholds is None:
        thresholds = {
            "low": (0, 33.33),
            "medium": (33.33, 66.67),
            "high": (66.67, 100),
        }

    barrier_level_expr = (
        f.when(f.col(index_column) < thresholds["low"][1], "low")
        .when(
            (f.col(index_column) >= thresholds["medium"][0])
            & (f.col(index_column) < thresholds["medium"][1]),
            "medium",
        )
        .otherwise("high")
    )

    result_df = df.withColumn("barrier_level", barrier_level_expr)

    logger.info("Barrier levels categorized")
    return result_df


# ============================================================================
# SECTION 3: ANALYSIS 1 - REGRESSION ANALYSIS
# ============================================================================


def run_regression_baseline(
    df: DataFrame,
    score_column: str = "PV1MATH",
    barrier_column: str = "barrier_index",
    weight_column: str = "W_FSTUWT",
) -> RegressionResults:
    """
    Model 1: Baseline regression of barrier index on test scores.

    Model: score = β0 + β1*barrier_index + ε

    Args:
        df: Input DataFrame
        score_column: Academic performance column
        barrier_column: Barrier index column
        weight_column: Sampling weight column

    Returns:
        RegressionResults with model statistics
    """
    logger = get_logger()
    logger.info("Running baseline regression model...")

    # Prepare data (skip rows with null values)
    assembler = VectorAssembler(
        inputCols=[barrier_column], outputCol="features", handleInvalid="skip"
    )
    train_data = assembler.transform(df).select(
        "features", f.col(score_column).alias("label"), weight_column
    )

    # Fit regression model with weights
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        weightCol=weight_column,
        maxIter=100,
        regParam=0.0,
    )

    model = lr.fit(train_data)

    # Extract results
    coefficients = {"barrier_index": float(model.coefficients[0])}
    sample_size = train_data.count()

    # Generate residuals sample for diagnostic plot (sample 5000 points)
    residuals_sample = None
    try:
        predictions = model.transform(train_data)
        sample_size_for_plot = min(5000, sample_size)
        sampled = (
            predictions.select("prediction", "label")
            .sample(fraction=sample_size_for_plot / sample_size, seed=42)
            .limit(sample_size_for_plot)
            .collect()
        )

        residuals_sample = [
            (float(row["prediction"]), float(row["label"] - row["prediction"])) for row in sampled
        ]
    except Exception as e:
        logger.warning("Failed to generate residuals sample: %s", e)

    results = RegressionResults(
        model_name="Baseline",
        r_squared=float(model.summary.r2),
        rmse=float(model.summary.rootMeanSquaredError),
        coefficients=coefficients,
        p_values=None,  # Not directly available in Spark MLlib
        sample_size=sample_size,
        residuals_sample=residuals_sample,
    )

    logger.info("Baseline regression completed. R² = %.4f", results.r_squared)
    return results


def run_regression_with_ses(
    df: DataFrame,
    score_column: str = "PV1MATH",
    barrier_column: str = "barrier_index",
    ses_column: str = "ESCS",
    weight_column: str = "W_FSTUWT",
) -> RegressionResults:
    """
    Model 2: Regression controlling for socioeconomic status.

    Model: score = β0 + β1*barrier_index + β2*SES + ε

    Args:
        df: Input DataFrame
        score_column: Academic performance column
        barrier_column: Barrier index column
        ses_column: SES control variable
        weight_column: Sampling weight column

    Returns:
        RegressionResults with model statistics
    """
    logger = get_logger()
    logger.info("Running regression with SES control...")

    # Validate SES column exists
    if ses_column not in df.columns:
        raise ValueError(f"SES column '{ses_column}' not found in DataFrame")

    # Prepare data (skip rows with null values)
    assembler = VectorAssembler(
        inputCols=[barrier_column, ses_column],
        outputCol="features",
        handleInvalid="skip",
    )
    train_data = assembler.transform(df).select(
        "features", f.col(score_column).alias("label"), weight_column
    )

    # Fit regression model
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        weightCol=weight_column,
        maxIter=100,
        regParam=0.0,
    )

    model = lr.fit(train_data)

    # Extract results
    coefficients = {
        "barrier_index": float(model.coefficients[0]),
        "ESCS": float(model.coefficients[1]),
    }
    sample_size = train_data.count()

    # Generate residuals sample for diagnostic plot (sample 5000 points)
    residuals_sample = None
    try:
        predictions = model.transform(train_data)
        sample_size_for_plot = min(5000, sample_size)
        sampled = (
            predictions.select("prediction", "label")
            .sample(fraction=sample_size_for_plot / sample_size, seed=42)
            .limit(sample_size_for_plot)
            .collect()
        )

        residuals_sample = [
            (float(row["prediction"]), float(row["label"] - row["prediction"])) for row in sampled
        ]
    except Exception as e:
        logger.warning("Failed to generate residuals sample: %s", e)

    results = RegressionResults(
        model_name="With SES Control",
        r_squared=float(model.summary.r2),
        rmse=float(model.summary.rootMeanSquaredError),
        coefficients=coefficients,
        p_values=None,
        sample_size=sample_size,
        residuals_sample=residuals_sample,
    )

    logger.info("SES regression completed. R² = %.4f", results.r_squared)
    return results


def run_regression_with_country_fe(
    df: DataFrame,
    score_column: str = "PV1MATH",
    barrier_column: str = "barrier_index",
    ses_column: str = "ESCS",
    country_column: str = "CNT",
    weight_column: str = "W_FSTUWT",
) -> RegressionResults:
    """
    Model 3: Regression with SES control + country fixed effects.

    Model: score = β0 + β1*barrier_index + β2*SES + Σ(γi*Country_i) + ε

    Uses one-hot encoding for country fixed effects.

    Args:
        df: Input DataFrame
        score_column: Academic performance column
        barrier_column: Barrier index column
        ses_column: SES control variable
        country_column: Country identifier column
        weight_column: Sampling weight column

    Returns:
        RegressionResults with model statistics
    """
    logger = get_logger()
    logger.info("Running regression with country fixed effects...")

    # Validate columns exist
    if country_column not in df.columns:
        raise ValueError(f"Country column '{country_column}' not found in DataFrame")

    # Get unique countries
    countries = [row[country_column] for row in df.select(country_column).distinct().collect()]
    logger.info("Found %d unique countries", len(countries))

    # Create dummy variables for countries (drop first to avoid multicollinearity)
    working_df = df
    for i, country in enumerate(countries[1:], start=1):
        dummy_col = f"country_dummy_{i}"
        working_df = working_df.withColumn(
            dummy_col, f.when(f.col(country_column) == country, 1.0).otherwise(0.0)
        )

    # Prepare feature columns
    feature_cols = [barrier_column, ses_column] + [
        f"country_dummy_{i}" for i in range(1, len(countries))
    ]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    train_data = assembler.transform(working_df).select(
        "features", f.col(score_column).alias("label"), weight_column
    )

    # Fit regression model
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        weightCol=weight_column,
        maxIter=100,
        regParam=0.0,
    )

    model = lr.fit(train_data)

    # Extract results
    coefficients = {
        "barrier_index": float(model.coefficients[0]),
        "ESCS": float(model.coefficients[1]),
    }
    # Add country coefficients
    for i in range(1, len(countries)):
        coefficients[f"country_{countries[i]}"] = float(model.coefficients[i + 1])

    sample_size = train_data.count()

    # Generate residuals sample for diagnostic plot (sample 5000 points)
    residuals_sample = None
    try:
        predictions = model.transform(train_data)
        sample_size_for_plot = min(5000, sample_size)
        sampled = (
            predictions.select("prediction", "label")
            .sample(fraction=sample_size_for_plot / sample_size, seed=42)
            .limit(sample_size_for_plot)
            .collect()
        )

        residuals_sample = [
            (float(row["prediction"]), float(row["label"] - row["prediction"])) for row in sampled
        ]
    except Exception as e:
        logger.warning("Failed to generate residuals sample: %s", e)

    results = RegressionResults(
        model_name="With Country Fixed Effects",
        r_squared=float(model.summary.r2),
        rmse=float(model.summary.rootMeanSquaredError),
        coefficients=coefficients,
        p_values=None,
        sample_size=sample_size,
        residuals_sample=residuals_sample,
    )

    logger.info("Country FE regression completed. R² = %.4f", results.r_squared)
    return results


def compare_regression_models(
    results: list[RegressionResults],
) -> dict[str, Any]:
    """
    Compare multiple regression models and return summary statistics.

    Args:
        results: List of RegressionResults from different models

    Returns:
        Dictionary with comparative statistics
    """
    logger = get_logger()
    logger.info("Comparing %d regression models...", len(results))

    comparison: dict[str, Any] = {
        "num_models": len(results),
        "models": {},
    }

    for result in results:
        comparison["models"][result.model_name] = {
            "r_squared": result.r_squared,
            "rmse": result.rmse,
            "barrier_coefficient": result.coefficients.get("barrier_index", None),
            "sample_size": result.sample_size,
        }

    # Find best model by R²
    best_model = max(results, key=lambda r: r.r_squared)
    comparison["best_model"] = best_model.model_name
    comparison["best_r_squared"] = best_model.r_squared

    logger.info("Model comparison completed. Best model: %s", best_model.model_name)
    return comparison


# ============================================================================
# SECTION 4: ANALYSIS 2 - DISTRIBUTION ACROSS CLUSTERS
# ============================================================================


def analyze_barriers_by_cluster(
    df: DataFrame,
    cluster_column: str = "score_cluster",
    barrier_column: str = "barrier_index",
    dimension_columns: list[str] | None = None,
    weight_column: str = "W_FSTUWT",
) -> dict[str, Any]:
    """
    Analyze barrier distribution across score clusters from Part A.

    Statistics calculated for each cluster:
    - Mean barrier index
    - Median barrier index
    - Standard deviation
    - Quartiles (25th, 75th percentile)
    - Percentage with high barriers (>75th percentile)
    - Dimension-level breakdown

    Args:
        df: DataFrame with both cluster labels and barrier index
        cluster_column: Score cluster column from Part A
        barrier_column: Barrier index column
        dimension_columns: Individual dimension columns to analyze
        weight_column: Sampling weight

    Returns:
        Dictionary with statistics by cluster level
    """
    logger = get_logger()
    logger.info("Analyzing barriers by cluster...")

    if dimension_columns is None:
        dimension_columns = [
            "dim_access_resources",
            "dim_internet_access",
            "dim_learning_disabilities",
            "dim_geographic_isolation",
        ]

    # Calculate statistics by cluster
    cluster_stats = (
        df.groupBy(cluster_column)
        .agg(
            f.count(barrier_column).alias("count"),
            f.sum(weight_column).alias("weighted_count"),
            f.mean(barrier_column).alias("mean_barrier"),
            (f.sum(f.col(barrier_column) * f.col(weight_column)) / f.sum(weight_column)).alias(
                "weighted_mean_barrier"
            ),
            f.stddev(barrier_column).alias("std_barrier"),
            f.expr(f"percentile_approx({barrier_column}, 0.25)").alias("q25"),
            f.expr(f"percentile_approx({barrier_column}, 0.50)").alias("median"),
            f.expr(f"percentile_approx({barrier_column}, 0.75)").alias("q75"),
        )
        .collect()
    )

    # Format results
    result = {}
    for row in cluster_stats:
        cluster_level = row[cluster_column]

        # Dimension breakdown
        dimension_stats = {}
        for dim_col in dimension_columns:
            if dim_col in df.columns:
                dim_stats = (
                    df.filter(f.col(cluster_column) == cluster_level)
                    .select(f.mean(dim_col).alias("mean_dim"))
                    .first()
                )
                dimension_stats[dim_col] = (
                    float(dim_stats["mean_dim"])
                    if dim_stats and dim_stats["mean_dim"] is not None
                    else None
                )

        result[cluster_level] = {
            "sample_count": row["count"],
            "weighted_count": (float(row["weighted_count"]) if row["weighted_count"] else 0),
            "mean_barrier": (
                float(row["mean_barrier"]) if row["mean_barrier"] is not None else None
            ),
            "weighted_mean_barrier": (
                float(row["weighted_mean_barrier"])
                if row["weighted_mean_barrier"] is not None
                else None
            ),
            "std_barrier": (float(row["std_barrier"]) if row["std_barrier"] is not None else None),
            "q25": float(row["q25"]) if row["q25"] is not None else None,
            "median": float(row["median"]) if row["median"] is not None else None,
            "q75": float(row["q75"]) if row["q75"] is not None else None,
            "dimension_breakdown": dimension_stats,
        }

    logger.info("Barrier analysis by cluster completed")
    return result


# ============================================================================
# SECTION 5: ANALYSIS 3 - FEATURE IMPORTANCE
# ============================================================================


def calculate_feature_importance_rf(
    df: DataFrame,
    barrier_columns: list[str],
    target_column: str = "PV1MATH",
    num_trees: int = 100,
    max_depth: int = 10,
    top_n: int = 10,
) -> FeatureImportanceResults:
    """
    Calculate feature importance using Random Forest.

    Trains a Random Forest model to predict academic performance
    from individual barrier indicators, then extracts feature importances.

    Args:
        df: Input DataFrame
        barrier_columns: List of all individual barrier indicator columns
        target_column: Academic performance column
        num_trees: Number of trees in random forest
        max_depth: Maximum tree depth
        top_n: Number of top features to highlight

    Returns:
        FeatureImportanceResults with importance scores
    """
    logger = get_logger()
    logger.info("Calculating feature importance using Random Forest...")

    # Filter columns that exist
    existing_cols = [col for col in barrier_columns if col in df.columns]
    if not existing_cols:
        raise ValueError("No valid barrier columns found")

    logger.info("Using %d features for importance calculation", len(existing_cols))

    # Prepare data (skip rows with null values)
    assembler = VectorAssembler(inputCols=existing_cols, outputCol="features", handleInvalid="skip")
    train_data = assembler.transform(df).select("features", f.col(target_column).alias("label"))

    # Train Random Forest
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="label",
        numTrees=num_trees,
        maxDepth=max_depth,
        seed=42,
    )

    model = rf.fit(train_data)

    # Extract feature importances
    importances = model.featureImportances.toArray()
    feature_importance_dict = {
        existing_cols[i]: float(importances[i]) for i in range(len(existing_cols))
    }

    # Sort and get top N
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]

    # Calculate R² as model accuracy metric
    predictions = model.transform(train_data)
    r2 = predictions.stat.corr("prediction", "label") ** 2 if predictions.count() > 0 else None

    results = FeatureImportanceResults(
        method="random_forest",
        feature_importances=feature_importance_dict,
        model_accuracy=r2,
        top_n_features=top_features,
    )

    logger.info("RF feature importance calculated. Top feature: %s", top_features[0][0])
    return results


def calculate_feature_importance_regression(
    df: DataFrame,
    barrier_columns: list[str],
    target_column: str = "PV1MATH",
    weight_column: str = "W_FSTUWT",
    top_n: int = 10,
) -> FeatureImportanceResults:
    """
    Calculate feature importance using standardized regression coefficients.

    Standardizes all variables, then uses absolute values of coefficients
    as importance measures.

    Args:
        df: Input DataFrame
        barrier_columns: List of barrier indicator columns
        target_column: Academic performance column
        weight_column: Sampling weight
        top_n: Number of top features to return

    Returns:
        FeatureImportanceResults with standardized coefficients
    """
    logger = get_logger()
    logger.info("Calculating feature importance using standardized regression...")

    # Filter columns that exist
    existing_cols = [col for col in barrier_columns if col in df.columns]
    if not existing_cols:
        raise ValueError("No valid barrier columns found")

    # Standardize all features
    working_df = df
    std_cols = []
    for col in existing_cols:
        std_col = f"{col}_std_reg"
        working_df = standardize_column(working_df, col, std_col)
        std_cols.append(std_col)

    # Prepare data (skip rows with null values)
    assembler = VectorAssembler(inputCols=std_cols, outputCol="features", handleInvalid="skip")
    train_data = assembler.transform(working_df).select(
        "features", f.col(target_column).alias("label"), weight_column
    )

    # Fit regression
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        weightCol=weight_column,
        maxIter=100,
        regParam=0.0,
    )

    model = lr.fit(train_data)

    # Extract absolute coefficients as importance
    coefficients = model.coefficients.toArray()
    feature_importance_dict = {
        existing_cols[i]: abs(float(coefficients[i])) for i in range(len(existing_cols))
    }

    # Normalize to sum to 1
    total = sum(feature_importance_dict.values())
    if total > 0:
        feature_importance_dict = {k: v / total for k, v in feature_importance_dict.items()}

    # Sort and get top N
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]

    results = FeatureImportanceResults(
        method="standardized_regression",
        feature_importances=feature_importance_dict,
        model_accuracy=float(model.summary.r2),
        top_n_features=top_features,
    )

    logger.info("Regression feature importance calculated. Top feature: %s", top_features[0][0])
    return results


# ============================================================================
# SECTION 6: ANALYSIS 4 - K-MEANS BARRIER CLUSTERING
# ============================================================================


def perform_barrier_clustering(
    df: DataFrame,
    dimension_columns: list[str],
    num_clusters: int = 4,
    max_iterations: int = 100,
    seed: int = 42,
) -> tuple[DataFrame, BarrierClusterResults]:
    """
    Perform k-means clustering based on barrier dimension profiles.

    Groups students into clusters based on their barrier profiles
    across the 4 dimensions. Default 4 clusters to identify distinct
    barrier patterns (e.g., high-internet-low-resources, etc.)

    Args:
        df: DataFrame with barrier dimension scores
        dimension_columns: List of barrier dimension column names
        num_clusters: Number of clusters (default: 4)
        max_iterations: Maximum k-means iterations
        seed: Random seed for reproducibility

    Returns:
        Tuple of (clustered_df, BarrierClusterResults)
    """
    logger = get_logger()
    logger.info("Performing k-means clustering with k=%d...", num_clusters)

    # Validate columns exist
    existing_cols = [col for col in dimension_columns if col in df.columns]
    if len(existing_cols) < len(dimension_columns):
        missing = set(dimension_columns) - set(existing_cols)
        logger.warning("Missing dimension columns: %s", missing)

    # Prepare features (skip rows with null values)
    assembler = VectorAssembler(
        inputCols=existing_cols, outputCol="barrier_features", handleInvalid="skip"
    )
    data_with_features = assembler.transform(df)

    # Scale features
    scaler = StandardScaler(
        inputCol="barrier_features",
        outputCol="scaled_features",
        withMean=True,
        withStd=True,
    )
    scaler_model = scaler.fit(data_with_features)
    scaled_data = scaler_model.transform(data_with_features)

    # Perform k-means
    kmeans = KMeans(
        k=num_clusters,
        featuresCol="scaled_features",
        predictionCol="barrier_cluster",
        maxIter=max_iterations,
        seed=seed,
    )

    model = kmeans.fit(scaled_data)
    clustered_df = model.transform(scaled_data)

    # Extract cluster centers
    centers = model.clusterCenters()
    cluster_centers_dict = {}
    for i, center in enumerate(centers):
        cluster_centers_dict[f"cluster_{i}"] = {
            existing_cols[j]: float(center[j]) for j in range(len(existing_cols))
        }

    # Calculate cluster sizes
    cluster_sizes_rows = clustered_df.groupBy("barrier_cluster").count().collect()
    cluster_sizes = {
        f"cluster_{row['barrier_cluster']}": row["count"] for row in cluster_sizes_rows
    }

    # Create cluster characteristics (simplified descriptions)
    cluster_characteristics = {}
    for cluster_id, center_values in cluster_centers_dict.items():
        # Find dominant dimension
        max_dim = max(center_values.items(), key=lambda x: abs(x[1]))
        cluster_characteristics[cluster_id] = (
            f"High in {max_dim[0]}" if max_dim[1] > 0 else f"Low in {max_dim[0]}"
        )

    results = BarrierClusterResults(
        num_clusters=num_clusters,
        cluster_centers=cluster_centers_dict,
        cluster_sizes=cluster_sizes,
        cluster_characteristics=cluster_characteristics,
        silhouette_score=None,  # Would require additional computation
        inertia=float(model.summary.trainingCost),
    )

    logger.info("K-means clustering completed. Inertia: %.2f", results.inertia)
    return (clustered_df, results)


def characterize_barrier_clusters(
    df: DataFrame,
    cluster_column: str = "barrier_cluster",
    dimension_columns: list[str] | None = None,
    weight_column: str = "W_FSTUWT",
) -> dict[str, Any]:
    """
    Generate descriptive statistics for each barrier cluster.

    For each cluster, calculate:
    - Mean values for each barrier dimension
    - Dominant barrier type(s)
    - Sample size and weighted population
    - Average academic performance

    Args:
        df: DataFrame with barrier clusters
        cluster_column: Cluster assignment column
        dimension_columns: Barrier dimension columns
        weight_column: Sampling weight

    Returns:
        Dictionary with cluster characteristics
    """
    logger = get_logger()
    logger.info("Characterizing barrier clusters...")

    if dimension_columns is None:
        dimension_columns = [
            "dim_access_resources",
            "dim_internet_access",
            "dim_learning_disabilities",
            "dim_geographic_isolation",
        ]

    # Get unique clusters
    clusters = [row[cluster_column] for row in df.select(cluster_column).distinct().collect()]

    result = {}
    for cluster_id in clusters:
        cluster_data = df.filter(f.col(cluster_column) == cluster_id)

        # Calculate statistics
        stats = cluster_data.agg(
            f.count("*").alias("count"),
            f.sum(weight_column).alias("weighted_count"),
        ).first()

        # Dimension means
        dimension_means = {}
        for dim_col in dimension_columns:
            if dim_col in df.columns:
                mean_val = cluster_data.select(f.mean(dim_col)).first()[0]
                dimension_means[dim_col] = float(mean_val) if mean_val is not None else None

        result[f"cluster_{cluster_id}"] = {
            "sample_count": stats["count"],
            "weighted_count": (float(stats["weighted_count"]) if stats["weighted_count"] else 0),
            "dimension_means": dimension_means,
        }

    logger.info("Cluster characterization completed")
    return result


def analyze_cluster_academic_outcomes(
    df: DataFrame,
    cluster_column: str = "barrier_cluster",
    score_column: str = "PV1MATH",
    weight_column: str = "W_FSTUWT",
) -> dict[str, Any]:
    """
    Analyze academic outcomes by barrier cluster.

    Args:
        df: DataFrame with barrier clusters and scores
        cluster_column: Barrier cluster column
        score_column: Academic performance column
        weight_column: Sampling weight

    Returns:
        Dictionary with performance statistics by cluster
    """
    logger = get_logger()
    logger.info("Analyzing academic outcomes by barrier cluster...")

    cluster_performance = (
        df.groupBy(cluster_column)
        .agg(
            f.mean(score_column).alias("mean_score"),
            (f.sum(f.col(score_column) * f.col(weight_column)) / f.sum(weight_column)).alias(
                "weighted_mean_score"
            ),
            f.stddev(score_column).alias("std_score"),
            f.min(score_column).alias("min_score"),
            f.max(score_column).alias("max_score"),
        )
        .collect()
    )

    result = {}
    for row in cluster_performance:
        cluster_id = row[cluster_column]
        result[f"cluster_{cluster_id}"] = {
            "mean_score": (float(row["mean_score"]) if row["mean_score"] is not None else None),
            "weighted_mean_score": (
                float(row["weighted_mean_score"])
                if row["weighted_mean_score"] is not None
                else None
            ),
            "std_score": (float(row["std_score"]) if row["std_score"] is not None else None),
            "min_score": (float(row["min_score"]) if row["min_score"] is not None else None),
            "max_score": (float(row["max_score"]) if row["max_score"] is not None else None),
        }

    logger.info("Academic outcomes analysis completed")
    return result


# ============================================================================
# SECTION 7: ORCHESTRATION AND REPORTING
# ============================================================================


def perform_comprehensive_barrier_analysis(
    df: DataFrame,
    config: BarrierIndexConfig,
    include_clustering: bool = True,
) -> dict[str, Any]:
    """
    Perform all barrier analyses and return comprehensive results.

    Workflow:
    1. Construct barrier index and dimensions
    2. Run all three regression models
    3. Analyze barriers by score clusters
    4. Calculate feature importance
    5. Perform barrier clustering (optional)
    6. Generate summary statistics

    Args:
        df: Input DataFrame with all required columns
        config: Barrier index configuration
        include_clustering: Whether to perform k-means clustering

    Returns:
        Dictionary containing all analysis results
    """
    logger = get_logger()
    logger.info("Starting comprehensive barrier analysis...")

    results: dict[str, Any] = {}

    # Step 1: Construct barrier index (already done in caller)
    # Assuming df already has barrier_index column

    # Step 2: Regression analyses
    logger.info("\n=== Running Regression Analyses ===")
    regression_results = []

    try:
        baseline = run_regression_baseline(df)
        regression_results.append(baseline)
        results["regression_baseline"] = baseline
    except Exception as e:  # noqa: BLE001
        logger.error("Baseline regression failed: %s", e)

    try:
        with_ses = run_regression_with_ses(df)
        regression_results.append(with_ses)
        results["regression_with_ses"] = with_ses
    except Exception as e:  # noqa: BLE001
        logger.error("SES regression failed: %s", e)

    try:
        with_country_fe = run_regression_with_country_fe(df)
        regression_results.append(with_country_fe)
        results["regression_with_country_fe"] = with_country_fe
    except Exception as e:  # noqa: BLE001
        logger.error("Country FE regression failed: %s", e)

    if regression_results:
        results["regression_comparison"] = compare_regression_models(regression_results)

    # Step 3: Distribution analysis by clusters
    logger.info("\n=== Analyzing Barriers by Clusters ===")
    try:
        cluster_analysis = analyze_barriers_by_cluster(df)
        results["barriers_by_cluster"] = cluster_analysis
    except Exception as e:  # noqa: BLE001
        logger.error("Cluster analysis failed: %s", e)

    # Step 4: Feature importance
    logger.info("\n=== Calculating Feature Importance ===")
    all_barrier_cols = get_all_barrier_columns(config)

    try:
        rf_importance = calculate_feature_importance_rf(df, all_barrier_cols)
        results["feature_importance_rf"] = rf_importance
    except Exception as e:  # noqa: BLE001
        logger.error("RF feature importance failed: %s", e)

    try:
        reg_importance = calculate_feature_importance_regression(df, all_barrier_cols)
        results["feature_importance_regression"] = reg_importance
    except Exception as e:  # noqa: BLE001
        logger.error("Regression feature importance failed: %s", e)

    # Step 5: K-means clustering (optional)
    if include_clustering:
        logger.info("\n=== Performing Barrier Clustering ===")
        try:
            dimension_cols = [
                "dim_access_resources",
                "dim_internet_access",
                "dim_learning_disabilities",
                "dim_geographic_isolation",
            ]
            clustered_df, cluster_results = perform_barrier_clustering(df, dimension_cols)
            results["barrier_clustering"] = cluster_results

            # Characterize clusters
            cluster_chars = characterize_barrier_clusters(clustered_df)
            results["cluster_characteristics"] = cluster_chars

            # Academic outcomes
            cluster_outcomes = analyze_cluster_academic_outcomes(clustered_df)
            results["cluster_academic_outcomes"] = cluster_outcomes
        except Exception as e:  # noqa: BLE001
            logger.error("Barrier clustering failed: %s", e)

    logger.info("\nComprehensive barrier analysis completed")
    return results


def print_barrier_analysis_report(results: dict[str, Any], verbose: bool = True) -> None:
    """
    Print formatted barrier analysis report to console.

    Args:
        results: Results from perform_comprehensive_barrier_analysis()
        verbose: Whether to print detailed statistics
    """
    print("\n" + "=" * 80, flush=True)
    print("BARRIER ANALYSIS REPORT", flush=True)
    print("=" * 80, flush=True)

    # Regression Results
    if "regression_comparison" in results:
        print("\n" + "-" * 80, flush=True)
        print("REGRESSION ANALYSIS", flush=True)
        print("-" * 80, flush=True)

        comparison = results["regression_comparison"]
        for model_name, stats in comparison["models"].items():
            print(f"\n{model_name}:", flush=True)
            print(f"  R²: {stats['r_squared']:.4f}", flush=True)
            print(f"  RMSE: {stats['rmse']:.2f}", flush=True)
            if stats["barrier_coefficient"] is not None:
                print(
                    f"  Barrier Coefficient: {stats['barrier_coefficient']:.4f}",
                    flush=True,
                )
            print(f"  Sample Size: {stats['sample_size']:,}", flush=True)

    # Barriers by Cluster
    if "barriers_by_cluster" in results:
        print("\n" + "-" * 80, flush=True)
        print("BARRIERS BY SCORE CLUSTER", flush=True)
        print("-" * 80, flush=True)

        for cluster, stats in results["barriers_by_cluster"].items():
            print(f"\n{cluster.upper()}:", flush=True)
            print(f"  Sample Size: {stats['sample_count']:,}", flush=True)
            if stats["mean_barrier"] is not None:
                print(f"  Mean Barrier: {stats['mean_barrier']:.2f}", flush=True)
            if stats["weighted_mean_barrier"] is not None:
                print(
                    f"  Weighted Mean Barrier: {stats['weighted_mean_barrier']:.2f}",
                    flush=True,
                )

    # Feature Importance
    if "feature_importance_rf" in results:
        print("\n" + "-" * 80, flush=True)
        print("TOP 10 IMPORTANT FEATURES (Random Forest)", flush=True)
        print("-" * 80, flush=True)

        for i, (feature, importance) in enumerate(
            results["feature_importance_rf"].top_n_features, 1
        ):
            print(f"  {i}. {feature}: {importance:.4f}", flush=True)

    # Cluster Summary
    if "barrier_clustering" in results:
        print("\n" + "-" * 80, flush=True)
        print("BARRIER CLUSTERING SUMMARY", flush=True)
        print("-" * 80, flush=True)

        clustering = results["barrier_clustering"]
        print(f"\nNumber of Clusters: {clustering.num_clusters}", flush=True)
        print(f"Inertia: {clustering.inertia:.2f}", flush=True)

        for cluster_id, size in clustering.cluster_sizes.items():
            print(f"\n{cluster_id}: {size:,} students", flush=True)
            if cluster_id in clustering.cluster_characteristics:
                print(
                    f"  Characteristic: {clustering.cluster_characteristics[cluster_id]}",
                    flush=True,
                )

    print("\n" + "=" * 80, flush=True)
