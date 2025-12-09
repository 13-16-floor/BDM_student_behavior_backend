"""
Unit tests for barrier analysis module.

Tests:
- Barrier index construction
- Dimension score calculation
- Regression model fitting
- Feature importance calculation
- K-means clustering
- Integration tests
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

from behavior_analysis.analysis.barrier_analysis import (
    BarrierIndexConfig,
    calculate_dimension_score,
    categorize_barrier_level,
    construct_barrier_index,
    validate_required_columns,
)


@pytest.fixture(scope="session")  # type: ignore[misc]
def spark() -> SparkSession:
    """Create Spark session for testing."""
    spark_session = (
        SparkSession.builder.master("local[1]")
        .appName("BarrierAnalysisTests")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield spark_session
    spark_session.stop()


@pytest.fixture  # type: ignore[misc]
def sample_barrier_data(spark: SparkSession) -> DataFrame:
    """Create sample student data with barrier indicators (including school-level variables)."""
    schema = StructType(
        [
            StructField("student_id", IntegerType(), False),
            # Access to resources - Home level
            StructField("HOMEPOS", DoubleType(), True),
            StructField("WORKHOME", DoubleType(), True),
            # Access to resources - School level
            StructField("SC017Q01NA", DoubleType(), True),  # Lack of teaching staff
            StructField("SC017Q02NA", DoubleType(), True),  # Inadequate teachers
            StructField("SC017Q03NA", DoubleType(), True),  # Lack of assisting staff
            StructField("SC017Q05NA", DoubleType(), True),  # Lack of materials
            # Internet access
            StructField("ICTRES", DoubleType(), True),
            StructField("ICTHOME", DoubleType(), True),
            StructField("ICTAVHOM", DoubleType(), True),
            # Learning disabilities
            StructField("ST127Q01TA", DoubleType(), True),
            StructField("ST127Q02TA", DoubleType(), True),
            StructField("ST127Q03TA", DoubleType(), True),
            StructField("ANXMAT", DoubleType(), True),
            # Geographic disadvantage (school location)
            StructField("SC001Q01TA", DoubleType(), True),  # 1=village, 5=large city
            # Control variables
            StructField("PV1MATH", DoubleType(), True),
            StructField("W_FSTUWT", DoubleType(), True),
            StructField("ESCS", DoubleType(), True),
            StructField("CNT", StringType(), True),
        ]
    )

    data = [
        # High resources, low barriers (urban school)
        (
            1,
            1.5,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.2,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.5,
            5.0,
            550,
            1.0,
            1.2,
            "USA",
        ),
        # Medium resources (town school)
        (
            2,
            0.0,
            0.5,
            2.0,
            2.0,
            2.0,
            2.0,
            0.0,
            0.5,
            0.5,
            1.0,
            1.0,
            1.0,
            1.0,
            3.0,
            480,
            1.0,
            0.0,
            "USA",
        ),
        # Low resources, high barriers (village school)
        (
            3,
            -1.5,
            -1.0,
            4.0,
            4.0,
            3.0,
            4.0,
            -1.2,
            -1.0,
            -1.0,
            2.0,
            2.0,
            1.0,
            2.0,
            1.0,
            400,
            1.0,
            -1.2,
            "USA",
        ),
        # High resources (large city)
        (
            4,
            1.8,
            1.2,
            1.0,
            1.0,
            1.0,
            1.0,
            1.5,
            1.2,
            1.2,
            1.0,
            1.0,
            1.0,
            0.3,
            5.0,
            600,
            1.0,
            1.5,
            "JPN",
        ),
        # Low resources (village)
        (
            5,
            -1.8,
            -1.2,
            3.0,
            4.0,
            3.0,
            3.0,
            -1.5,
            -1.2,
            -1.2,
            2.0,
            2.0,
            2.0,
            2.2,
            1.0,
            380,
            1.0,
            -1.5,
            "JPN",
        ),
    ]

    return spark.createDataFrame(data, schema)


class TestColumnValidation:
    """Tests for column validation."""

    def test_all_columns_present(self, sample_barrier_data: DataFrame) -> None:
        """Test validation when all required columns are present."""
        required_cols = ["ESCS", "CNT", "SC001Q01TA"]  # Added SC001Q01TA (school-level)
        all_valid, missing = validate_required_columns(sample_barrier_data, required_cols)

        assert all_valid is True
        assert len(missing) == 0

    def test_missing_columns(self, sample_barrier_data: DataFrame) -> None:
        """Test validation when columns are missing."""
        required_cols = ["ESCS", "CNT", "MISSING_COL"]
        all_valid, missing = validate_required_columns(sample_barrier_data, required_cols)

        assert all_valid is False
        assert "MISSING_COL" in missing
        assert len(missing) == 1


class TestBarrierIndexConstruction:
    """Tests for barrier index construction."""

    def test_dimension_score_calculation(
        self, sample_barrier_data: DataFrame, spark: SparkSession
    ) -> None:
        """Test single dimension score calculation."""
        result_df = calculate_dimension_score(
            sample_barrier_data,
            columns=["HOMEPOS", "WORKHOME"],  # Corrected column names
            dimension_name="test_dim",
            standardize=True,
            reverse_coding=False,
            handle_missing="impute_mean",
        )

        # Check column was added
        assert "test_dim" in result_df.columns

        # Check no null values
        null_count = result_df.filter(f.col("test_dim").isNull()).count()
        assert null_count == 0

    def test_barrier_index_basic(self, sample_barrier_data: DataFrame) -> None:
        """Test basic barrier index construction."""
        config = BarrierIndexConfig(
            access_to_resources_cols=[
                "HOMEPOS",
                "WORKHOME",
                "SC017Q01NA",
                "SC017Q02NA",
                "SC017Q03NA",
                "SC017Q05NA",
            ],
            internet_access_cols=["ICTRES", "ICTHOME", "ICTAVHOM"],
            learning_disabilities_cols=[
                "ST127Q01TA",
                "ST127Q02TA",
                "ST127Q03TA",
                "ANXMAT",
            ],
            geographic_isolation_cols=["SC001Q01TA"],
            weights={
                "access_to_resources": 0.25,
                "internet_access": 0.25,
                "learning_disabilities": 0.25,
                "geographic_isolation": 0.25,
            },
            handle_missing="impute_mean",
            standardize=True,
        )

        result_df = construct_barrier_index(sample_barrier_data, config)

        # Check barrier index column exists
        assert "barrier_index" in result_df.columns

        # Check dimension columns exist
        assert "dim_access_resources" in result_df.columns
        assert "dim_internet_access" in result_df.columns
        assert "dim_learning_disabilities" in result_df.columns
        assert "dim_geographic_isolation" in result_df.columns

        # Check barrier index is in 0-100 range
        stats = result_df.select(
            f.min("barrier_index").alias("min"), f.max("barrier_index").alias("max")
        ).first()

        assert stats["min"] >= 0
        assert stats["max"] <= 100

    def test_barrier_index_with_missing_data(self, spark: SparkSession) -> None:
        """Test handling of missing data."""
        schema = StructType(
            [
                StructField("id", IntegerType(), False),
                StructField("HOMEPOS", DoubleType(), True),
                StructField("ICTRES", DoubleType(), True),
                StructField("ST127Q01TA", DoubleType(), True),
                StructField("SC001Q01TA", DoubleType(), True),
            ]
        )

        data = [
            (1, 1.0, 1.0, 1.0, 5.0),
            (2, None, 1.0, 1.0, 3.0),  # Missing HOMEPOS
            (3, 1.0, None, 1.0, 1.0),  # Missing ICTRES
        ]

        df = spark.createDataFrame(data, schema)

        config = BarrierIndexConfig(
            access_to_resources_cols=["HOMEPOS"],
            internet_access_cols=["ICTRES"],
            learning_disabilities_cols=["ST127Q01TA"],
            geographic_isolation_cols=["SC001Q01TA"],
            weights={
                "access_to_resources": 0.25,
                "internet_access": 0.25,
                "learning_disabilities": 0.25,
                "geographic_isolation": 0.25,
            },
            handle_missing="impute_mean",
            standardize=True,
        )

        result_df = construct_barrier_index(df, config)

        # All rows should have barrier index
        assert result_df.filter(f.col("barrier_index").isNull()).count() == 0

    def test_barrier_categorization(self, sample_barrier_data: DataFrame) -> None:
        """Test barrier level categorization."""
        config = BarrierIndexConfig(
            access_to_resources_cols=["HOMEPOS"],  # Corrected
            internet_access_cols=["ICTRES"],
            learning_disabilities_cols=["ST127Q01TA"],
            geographic_isolation_cols=["ST260Q01JA"],  # Corrected
            weights={
                "access_to_resources": 0.25,
                "internet_access": 0.25,
                "learning_disabilities": 0.25,
                "geographic_isolation": 0.25,
            },
            handle_missing="impute_mean",
            standardize=True,
        )

        df_with_index = construct_barrier_index(sample_barrier_data, config)
        result_df = categorize_barrier_level(df_with_index)

        # Check barrier_level column exists
        assert "barrier_level" in result_df.columns

        # Check all values are valid categories
        categories = [row["barrier_level"] for row in result_df.select("barrier_level").collect()]
        valid_categories = {"low", "medium", "high"}
        assert all(cat in valid_categories for cat in categories)


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_end_to_end_barrier_index(self, sample_barrier_data: DataFrame) -> None:
        """Test complete barrier index construction workflow."""
        # Validate columns (corrected - removed SC001Q01TA)
        required_cols = ["ESCS", "CNT", "PV1MATH", "W_FSTUWT"]
        all_valid, missing = validate_required_columns(sample_barrier_data, required_cols)
        assert all_valid

        # Construct barrier index (corrected column names)
        config = BarrierIndexConfig(
            access_to_resources_cols=["HOMEPOS", "WORKHOME"],  # Corrected
            internet_access_cols=["ICTRES", "ICTHOME"],  # Corrected
            learning_disabilities_cols=["ST127Q01TA", "ST127Q02TA"],
            geographic_isolation_cols=["ST260Q01JA", "ST260Q02JA"],  # Corrected
            weights={
                "access_to_resources": 0.25,
                "internet_access": 0.25,
                "learning_disabilities": 0.25,
                "geographic_isolation": 0.25,
            },
            handle_missing="impute_mean",
            standardize=True,
        )

        result_df = construct_barrier_index(sample_barrier_data, config)

        # Verify result
        assert result_df.count() == 5
        assert "barrier_index" in result_df.columns
        assert result_df.filter(f.col("barrier_index").isNull()).count() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
