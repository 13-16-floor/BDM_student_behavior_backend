"""
Unit tests for barrier × attitude interaction analysis module.

Tests:
- Within-country transformation (demeaning)
- Attitude composite score creation
- Interaction term construction
- Interaction regression model
- Stratified analysis
- Comprehensive workflow
"""

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from behavior_analysis.analysis.barrier_attitude_interaction import (
    InteractionConfig,
    apply_within_country_transformation,
    create_attitude_composite_score,
    create_interaction_term,
    perform_comprehensive_interaction_analysis,
    run_interaction_regression,
)


@pytest.fixture(scope="session")  # type: ignore[misc]
def spark() -> SparkSession:
    """Create Spark session for testing."""
    spark_session = (
        SparkSession.builder.master("local[1]")
        .appName("InteractionAnalysisTests")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield spark_session
    spark_session.stop()


@pytest.fixture  # type: ignore[misc]
def sample_interaction_data(spark: SparkSession) -> DataFrame:
    """Create sample data with barrier index, attitudes, and control variables."""
    schema = StructType(
        [
            StructField("student_id", IntegerType(), False),
            # Core variables
            StructField("PV1MATH", DoubleType(), True),
            StructField("barrier_index", DoubleType(), True),
            # Attitude variables
            StructField("MATHMOT", DoubleType(), True),  # Math motivation
            StructField("PERSEVAGR", DoubleType(), True),  # Perseverance
            StructField("GROSAGR", DoubleType(), True),  # Growth mindset
            StructField("MATHEFF", DoubleType(), True),  # Math self-efficacy
            # Control variables
            StructField("ESCS", DoubleType(), True),
            StructField("CNT", StringType(), True),
            StructField("W_FSTUWT", DoubleType(), True),
        ]
    )

    # Create sample data with 3 countries, varying barriers and attitudes
    data = [
        # Country A - High SES
        (
            1,
            550.0,
            -0.5,
            0.8,
            0.7,
            0.9,
            0.8,
            1.2,
            "A",
            100.0,
        ),  # Low barrier, high attitude
        (
            2,
            520.0,
            0.0,
            0.5,
            0.4,
            0.6,
            0.5,
            1.0,
            "A",
            100.0,
        ),  # Medium barrier, medium attitude
        (
            3,
            480.0,
            0.8,
            0.2,
            0.1,
            0.3,
            0.2,
            0.8,
            "A",
            100.0,
        ),  # High barrier, low attitude
        (
            4,
            530.0,
            -0.3,
            0.7,
            0.6,
            0.8,
            0.7,
            1.1,
            "A",
            100.0,
        ),  # Low barrier, high attitude
        (
            5,
            500.0,
            0.3,
            0.4,
            0.3,
            0.5,
            0.4,
            0.9,
            "A",
            100.0,
        ),  # Medium barrier, medium attitude
        # Country B - Medium SES
        (6, 490.0, -0.4, 0.6, 0.5, 0.7, 0.6, 0.5, "B", 100.0),
        (7, 470.0, 0.1, 0.3, 0.2, 0.4, 0.3, 0.3, "B", 100.0),
        (8, 440.0, 0.9, 0.1, 0.0, 0.2, 0.1, 0.1, "B", 100.0),
        (9, 480.0, -0.2, 0.5, 0.4, 0.6, 0.5, 0.4, "B", 100.0),
        (10, 450.0, 0.5, 0.2, 0.1, 0.3, 0.2, 0.2, "B", 100.0),
        # Country C - Low SES
        (11, 430.0, -0.3, 0.4, 0.3, 0.5, 0.4, -0.5, "C", 100.0),
        (12, 410.0, 0.2, 0.2, 0.1, 0.3, 0.2, -0.7, "C", 100.0),
        (13, 380.0, 1.0, 0.0, -0.1, 0.1, 0.0, -0.9, "C", 100.0),
        (14, 420.0, -0.1, 0.3, 0.2, 0.4, 0.3, -0.6, "C", 100.0),
        (15, 390.0, 0.6, 0.1, 0.0, 0.2, 0.1, -0.8, "C", 100.0),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture  # type: ignore[misc]
def interaction_config() -> InteractionConfig:
    """Create default interaction configuration."""
    return InteractionConfig(
        score_column="PV1MATH",
        barrier_column="barrier_index",
        attitude_columns=["MATHMOT", "PERSEVAGR", "GROSAGR", "MATHEFF"],
        country_column="CNT",
        weight_column="W_FSTUWT",
        ses_column="ESCS",
        include_ses=True,
    )


class TestWithinCountryTransformation:
    """Test within-country transformation (demeaning)."""

    def test_single_column_transformation(self, spark: SparkSession) -> None:
        """Test demeaning a single column within country groups."""
        # Create simple test data
        data = [
            ("A", 100.0),
            ("A", 200.0),
            ("A", 300.0),
            ("B", 400.0),
            ("B", 600.0),
        ]
        df = spark.createDataFrame(data, ["CNT", "value"])

        # Apply transformation
        result = apply_within_country_transformation(
            df, columns_to_transform=["value"], country_column="CNT", suffix="_within"
        )

        # Collect results
        result_data = result.select("CNT", "value", "value_within").collect()

        # Country A mean = 200, so demeaned values: -100, 0, 100
        country_a = [row for row in result_data if row["CNT"] == "A"]
        assert len(country_a) == 3
        assert abs(country_a[0]["value_within"] - (-100.0)) < 1e-6
        assert abs(country_a[1]["value_within"] - 0.0) < 1e-6
        assert abs(country_a[2]["value_within"] - 100.0) < 1e-6

        # Country B mean = 500, so demeaned values: -100, 100
        country_b = [row for row in result_data if row["CNT"] == "B"]
        assert len(country_b) == 2
        assert abs(country_b[0]["value_within"] - (-100.0)) < 1e-6
        assert abs(country_b[1]["value_within"] - 100.0) < 1e-6

    def test_multiple_columns_transformation(self, sample_interaction_data: DataFrame) -> None:
        """Test demeaning multiple columns simultaneously."""
        result = apply_within_country_transformation(
            sample_interaction_data,
            columns_to_transform=["PV1MATH", "barrier_index", "ESCS"],
            country_column="CNT",
        )

        # Verify new columns created
        assert "PV1MATH_within" in result.columns
        assert "barrier_index_within" in result.columns
        assert "ESCS_within" in result.columns

        # Verify within-country means are approximately 0
        country_stats = result.groupBy("CNT").agg(
            f.mean("PV1MATH_within").alias("mean_score_within"),
            f.mean("barrier_index_within").alias("mean_barrier_within"),
        )

        for row in country_stats.collect():
            assert abs(row["mean_score_within"]) < 1e-6
            assert abs(row["mean_barrier_within"]) < 1e-6


class TestAttitudeCompositeScore:
    """Test attitude composite score creation."""

    def test_composite_score_creation(self, sample_interaction_data: DataFrame) -> None:
        """Test creating composite attitude score from multiple variables."""
        attitude_cols = ["MATHMOT", "PERSEVAGR", "GROSAGR", "MATHEFF"]

        result = create_attitude_composite_score(
            sample_interaction_data,
            attitude_columns=attitude_cols,
            output_column="attitude_score",
        )

        # Verify column created
        assert "attitude_score" in result.columns

        # Verify no nulls (assuming input has no nulls)
        null_count = result.filter(f.col("attitude_score").isNull()).count()
        assert null_count == 0

        # Verify composite is within reasonable range
        stats = result.select(
            f.min("attitude_score").alias("min"),
            f.max("attitude_score").alias("max"),
            f.mean("attitude_score").alias("mean"),
        ).collect()[0]

        assert stats["min"] is not None
        assert stats["max"] is not None
        assert stats["mean"] is not None


class TestInteractionTerm:
    """Test interaction term construction."""

    def test_interaction_term_creation(self, spark: SparkSession) -> None:
        """Test creating interaction term from two variables."""
        data = [
            (1, 2.0, 3.0),
            (2, -1.0, 4.0),
            (3, 0.0, 5.0),
        ]
        df = spark.createDataFrame(data, ["id", "x", "y"])

        result = create_interaction_term(df, var1="x", var2="y", interaction_name="x_times_y")

        # Verify interaction term
        result_data = result.select("id", "x", "y", "x_times_y").collect()

        assert abs(result_data[0]["x_times_y"] - 6.0) < 1e-6  # 2 * 3
        assert abs(result_data[1]["x_times_y"] - (-4.0)) < 1e-6  # -1 * 4
        assert abs(result_data[2]["x_times_y"] - 0.0) < 1e-6  # 0 * 5


class TestInteractionRegression:
    """Test interaction regression model."""

    def test_basic_regression(
        self, sample_interaction_data: DataFrame, interaction_config: InteractionConfig
    ) -> None:
        """Test running basic interaction regression."""
        result = run_interaction_regression(
            sample_interaction_data, interaction_config, use_within_transformation=True
        )

        # Verify results structure
        assert result.model_name is not None
        assert result.r_squared >= 0.0
        assert result.r_squared <= 1.0
        assert result.sample_size > 0
        assert "barrier_within" in result.coefficients
        assert "attitude_within" in result.coefficients
        assert "interaction" in result.coefficients

        # Verify interaction effect is captured
        assert result.interaction_effect is not None
        assert result.interaction_interpretation is not None

        # Verify marginal effects
        assert "low_barrier" in result.marginal_effects
        assert "medium_barrier" in result.marginal_effects
        assert "high_barrier" in result.marginal_effects

    def test_regression_without_transformation(
        self, sample_interaction_data: DataFrame, interaction_config: InteractionConfig
    ) -> None:
        """Test pooled regression without within-country transformation."""
        result = run_interaction_regression(
            sample_interaction_data, interaction_config, use_within_transformation=False
        )

        # Should still produce valid results
        assert result.r_squared >= 0.0
        assert result.sample_size > 0


class TestComprehensiveAnalysis:
    """Test comprehensive interaction analysis workflow."""

    def test_comprehensive_workflow(
        self, sample_interaction_data: DataFrame, interaction_config: InteractionConfig
    ) -> None:
        """Test complete analysis workflow."""
        results = perform_comprehensive_interaction_analysis(
            sample_interaction_data, interaction_config
        )

        # Verify all components present
        assert "main_model" in results
        assert "pooled_model" in results
        assert "stratified_by_barrier" in results
        assert "stratified_by_attitude" in results
        assert "plot_data" in results

        # Verify main model results
        main_model = results["main_model"]
        assert main_model.sample_size > 0
        assert main_model.r_squared >= 0.0

        # Verify stratified results
        barrier_stratified = results["stratified_by_barrier"]
        assert "low" in barrier_stratified
        assert "medium" in barrier_stratified
        assert "high" in barrier_stratified

    def test_missing_columns_handling(
        self, spark: SparkSession, interaction_config: InteractionConfig
    ) -> None:
        """Test handling of missing required columns."""
        # Create data missing attitude column
        data = [
            (1, 500.0, 0.5, 0.5, "A", 100.0),
        ]
        df = spark.createDataFrame(
            data, ["id", "PV1MATH", "barrier_index", "ESCS", "CNT", "W_FSTUWT"]
        )

        # Should handle missing columns gracefully
        with pytest.raises((ValueError, KeyError)):
            run_interaction_regression(df, interaction_config, use_within_transformation=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
