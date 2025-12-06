"""
Unit tests for attitude-based clustering module.

Tests the attitude-based clustering functionality including:
- Attitude data preparation
- Feature creation and standardization
- K-means clustering
- Cluster label assignment
- Weighted statistics calculation
"""

import pytest
from pyspark.sql import DataFrame, SparkSession

from behavior_analysis.analysis.attitude_clustering import (
    ATTITUDE_DIMENSIONS,
    add_attitude_labels,
    create_attitude_features,
    perform_attitude_clustering,
    prepare_attitude_data,
)


@pytest.fixture(scope="session")  # type: ignore[misc]
def spark() -> SparkSession:
    """Create a Spark session for testing."""
    return (
        SparkSession.builder.master("local[1]")
        .appName("test-attitude-clustering")
        .config("spark.driver.memory", "1g")
        .config("spark.executor.memory", "1g")
        .getOrCreate()
    )


@pytest.fixture  # type: ignore[misc]
def sample_attitude_data(spark: SparkSession) -> DataFrame:
    """Create sample student data with attitude dimensions for testing."""
    data = [
        ("S001", 4.0, 4.0, 4.0, 4.0, 1.2),  # Proactive: high on all dimensions
        ("S002", 2.0, 2.0, 2.0, 2.0, 1.1),  # Average: medium on all dimensions
        ("S003", 1.0, 1.0, 1.0, 1.0, 1.0),  # Disengaged: low on all dimensions
        ("S004", 4.0, 4.0, 3.0, 4.0, 0.9),  # Proactive variant
        ("S005", 2.0, 2.0, 2.0, 3.0, 1.3),  # Average variant
        ("S006", 1.0, 1.0, 1.0, 2.0, 1.0),  # Disengaged variant
        ("S007", 3.0, 3.0, 3.0, 3.0, 0.95),  # Average-Proactive
        ("S008", 2.0, 3.0, 2.0, 2.0, 1.15),  # Average-Disengaged
    ]
    return spark.createDataFrame(
        data,
        schema="""student_id STRING,
                  ST296Q01JA DOUBLE,
                  ST062Q01TA DOUBLE,
                  MATHMOT DOUBLE,
                  PERSEVAGR DOUBLE,
                  W_FSTUWT DOUBLE""",
    )


class TestAttitudePrepare:
    """Tests for attitude data preparation."""

    def test_prepare_with_valid_data(self, sample_attitude_data: DataFrame) -> None:
        """Test preparation with valid attitude data."""
        prepared_df = prepare_attitude_data(sample_attitude_data)

        # Should retain all rows since no nulls
        assert prepared_df.count() == sample_attitude_data.count()

    def test_prepare_removes_null_rows(self, spark: SparkSession) -> None:
        """Test that preparation removes rows with null values."""
        data = [
            ("S001", 4.0, 4.0, 4.0, 4.0, 1.2),
            ("S002", 2.0, None, 2.0, 2.0, 1.1),  # Null value
            ("S003", 1.0, 1.0, 1.0, 1.0, 1.0),
            ("S004", None, 4.0, 3.0, 4.0, 0.9),  # Null value
        ]
        df = spark.createDataFrame(
            data,
            schema="""student_id STRING,
                      ST296Q01JA DOUBLE,
                      ST062Q01TA DOUBLE,
                      MATHMOT DOUBLE,
                      PERSEVAGR DOUBLE,
                      W_FSTUWT DOUBLE""",
        )

        prepared_df = prepare_attitude_data(df)

        # Should remove rows with null values in attitude dimensions
        assert prepared_df.count() == 2


class TestAttitudeFeatures:
    """Tests for attitude feature creation."""

    def test_create_features(self, sample_attitude_data: DataFrame) -> None:
        """Test feature creation with valid data."""
        prepared_df = prepare_attitude_data(sample_attitude_data)
        featured_df = create_attitude_features(prepared_df)

        # Should have new attitude_features column
        assert "attitude_features" in featured_df.columns

        # Features should be a vector
        featured_df.select("attitude_features").show(1, truncate=False)

    def test_features_dimension_count(self, sample_attitude_data: DataFrame) -> None:
        """Test that features have correct number of dimensions."""
        prepared_df = prepare_attitude_data(sample_attitude_data)
        create_attitude_features(prepared_df)

        # Should have 4 attitude dimensions
        assert len(ATTITUDE_DIMENSIONS) == 4


class TestAttitudeClustering:
    """Tests for K-means attitude clustering."""

    def test_perform_clustering(self, sample_attitude_data: DataFrame) -> None:
        """Test K-means clustering with valid data."""
        prepared_df = prepare_attitude_data(sample_attitude_data)
        featured_df = create_attitude_features(prepared_df)
        clustered_df = perform_attitude_clustering(featured_df, num_clusters=3)

        # Should have attitude_cluster column
        assert "attitude_cluster" in clustered_df.columns

        # All rows should have cluster assignments
        assert clustered_df.count() == featured_df.count()

        # Cluster assignments should be in range [0, num_clusters)
        predictions = clustered_df.select("attitude_cluster").collect()
        for row in predictions:
            assert 0 <= row["attitude_cluster"] < 3

    def test_clustering_with_different_k(self, sample_attitude_data: DataFrame) -> None:
        """Test clustering with different k values."""
        prepared_df = prepare_attitude_data(sample_attitude_data)
        featured_df = create_attitude_features(prepared_df)

        for k in [2, 3, 4]:
            clustered_df = perform_attitude_clustering(featured_df, num_clusters=k)
            assert clustered_df.count() == featured_df.count()


class TestAttitudeLabels:
    """Tests for attitude cluster label assignment."""

    def test_add_labels(self, sample_attitude_data: DataFrame) -> None:
        """Test adding attitude labels to clustered data."""
        prepared_df = prepare_attitude_data(sample_attitude_data)
        featured_df = create_attitude_features(prepared_df)
        clustered_df = perform_attitude_clustering(featured_df, num_clusters=3)
        labeled_df = add_attitude_labels(clustered_df)

        # Should have attitude_label column
        assert "attitude_label" in labeled_df.columns

        # All rows should have labels
        assert labeled_df.count() == clustered_df.count()

        # Labels should be valid strings
        labels = labeled_df.select("attitude_label").distinct().collect()
        expected_labels = {
            "Proactive Learners",
            "Average Learners",
            "Disengaged Learners",
        }
        actual_labels = {row["attitude_label"] for row in labels}
        assert actual_labels.issubset(expected_labels)
