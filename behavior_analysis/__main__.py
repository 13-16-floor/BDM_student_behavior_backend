"""
Main application entry point for Behavior Analysis.

This module orchestrates the complete workflow:
1. Convert SPSS files to Parquet format
2. Load data using Spark
3. Perform statistical analysis
"""

import logging
import sys
import warnings

from pyspark.sql import DataFrame

from .analysis.barrier_analysis import (
    BarrierIndexConfig,
    construct_barrier_index,
    perform_comprehensive_barrier_analysis,
    print_barrier_analysis_report,
    validate_required_columns,
)
from .analysis.score_clustering import (
    add_cluster_labels,
    get_cluster_statistics,
    print_clustering_report,
)
from .config import AppConfig, load_config
from .data.converter import SPSSToParquetConverter
from .data.spark_manager import SparkSessionManager
from .utils.file_utils import ensure_directory_exists
from .utils.logger import setup_logger
from .visualization.barrier_analysis_viz import (
    create_all_barrier_visualizations,
    export_results_to_csv,
)
from .visualization.score_clustering_viz import create_all_visualizations


def perform_score_clustering_analysis(
    student_df: DataFrame, config: AppConfig, logger: logging.Logger
) -> None:
    """
    Perform score-based clustering analysis on student data using math scores.

    This function performs a complete score-based clustering workflow:
    1. Adds cluster labels to students based on their PV1MATH scores
       (dividing them into low/medium/high performance groups)
    2. Computes weighted cluster statistics using PISA sample weights (W_FSTUWT)
    3. Prints a detailed clustering report to console
    4. Logs comprehensive statistics for each cluster
    5. Generates and saves visualizations to the artifacts directory

    Args:
        student_df: Spark DataFrame containing student data with PV1MATH
                   and W_FSTUWT columns
        config: Application configuration containing paths and settings
        logger: Logger instance for recording analysis steps and results
    """
    # Add cluster labels based on PV1MATH scores
    logger.info("\nPerforming score-based clustering...")
    clustered_df = add_cluster_labels(
        student_df, score_column="PV1MATH", cluster_column="score_cluster"
    )

    # Get cluster statistics
    logger.info("Computing cluster statistics with weights...")
    cluster_stats = get_cluster_statistics(
        clustered_df,
        score_column="PV1MATH",
        weight_column="W_FSTUWT",
        cluster_column="score_cluster",
    )

    # Print clustering report
    print_clustering_report(cluster_stats, verbose=True)

    # Log statistics
    logger.info("\nCluster Statistics Summary:")
    for level, stats in sorted(cluster_stats.items()):
        logger.info(f"\n{level.upper()}:")
        logger.info(f"  Sample Size: {stats['sample_count']:,}")
        logger.info(f"  Weighted Population: {stats['weighted_count']:,.0f}")
        logger.info(f"  Population %: {stats['population_percentage']:.2f}%")
        if stats["mean_score"] is not None:
            logger.info(f"  Mean Score: {stats['mean_score']:.2f}")
        if stats["weighted_mean_score"] is not None:
            logger.info(f"  Weighted Mean Score: {stats['weighted_mean_score']:.2f}")

    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    create_all_visualizations(
        cluster_stats, output_directory=config.get_artifact_path("visualizations")
    )


def perform_barrier_analysis(
    student_df: DataFrame,
    school_df: DataFrame,
    config: AppConfig,
    logger: logging.Logger,
    include_score_clusters: bool = True,
) -> None:
    """
    Perform comprehensive barrier analysis on student data.

    This function performs the complete barrier analysis workflow:
    1. Selects only required columns to avoid OOM
    2. Joins student and school data
    3. Validates required columns
    4. Constructs barrier index from 4 dimensions
    5. Adds score cluster labels from Part A
    6. Runs 4 core analyses (regression, distribution, importance, clustering)
    7. Generates and saves visualizations

    Args:
        student_df: Spark DataFrame containing student data (full dataset)
        school_df: Spark DataFrame containing school data (full dataset)
        config: Application configuration
        logger: Logger instance
        include_score_clusters: Whether to include cluster analysis from Part A
    """
    logger.info("\n" + "=" * 70)
    logger.info("BARRIER ANALYSIS")
    logger.info("=" * 70)

    # Step 1: Select only required columns to avoid OOM
    logger.info("\nSelecting required columns to optimize memory usage...")

    # Define required columns
    student_required_cols = [
        # Join key
        "CNTSCHID",
        # Score and weights
        "PV1MATH",
        "W_FSTUWT",
        # Control variables
        "CNT",
        "ESCS",
        # Dimension 1: Access to Resources
        "HOMEPOS",
        "WORKHOME",
        # Dimension 2: Internet Access
        "ICTRES",
        "ICTHOME",
        "ICTAVHOM",
        # Dimension 3: Learning Disabilities
        "ST127Q01TA",
        "ST127Q02TA",
        "ST127Q03TA",
        "ANXMAT",
    ]

    school_required_cols = [
        # Join key
        "CNTSCHID",
        # Dimension 1: Access to Resources (school level)
        "SC017Q01NA",
        "SC017Q02NA",
        "SC017Q03NA",
        "SC017Q05NA",
        # Dimension 4: Geographic Disadvantage
        "SC001Q01TA",
    ]

    # Select columns from student data
    logger.info(f"Selecting {len(student_required_cols)} columns from student data...")
    student_subset = student_df.select(*student_required_cols)

    # Select columns from school data
    logger.info(f"Selecting {len(school_required_cols)} columns from school data...")
    school_subset = school_df.select(*school_required_cols)

    # Step 2: Join student and school data
    logger.info("\nJoining student and school data on CNTSCHID...")
    df_joined = student_subset.join(school_subset, on="CNTSCHID", how="left")

    # Count joined records
    joined_count = df_joined.count()
    logger.info(f"Joined dataset: {joined_count:,} records")

    # Step 3: Validate required columns (control variables only)
    logger.info("\nValidating required columns...")
    required_critical_cols = ["ESCS", "CNT", "SC001Q01TA"]
    all_valid, missing_cols = validate_required_columns(df_joined, required_critical_cols)

    if not all_valid:
        error_msg = (
            f"Cannot proceed with barrier analysis. Missing required columns: {missing_cols}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("All required control variables validated successfully")

    # Step 4: Configure barrier index
    logger.info("\nConfiguring barrier dimensions...")
    barrier_config = BarrierIndexConfig(
        # Dimension 1: Access to Resources (home + school level)
        access_to_resources_cols=[
            "HOMEPOS",
            "WORKHOME",
            "SC017Q01NA",
            "SC017Q02NA",
            "SC017Q03NA",
            "SC017Q05NA",
        ],
        # Dimension 2: Internet Access
        internet_access_cols=["ICTRES", "ICTHOME", "ICTAVHOM"],
        # Dimension 3: Learning Disabilities
        learning_disabilities_cols=["ST127Q01TA", "ST127Q02TA", "ST127Q03TA", "ANXMAT"],
        # Dimension 4: Geographic Disadvantage (school location)
        geographic_isolation_cols=["SC001Q01TA"],
        weights={
            "access_to_resources": config.barrier.ACCESS_RESOURCES_WEIGHT,
            "internet_access": config.barrier.INTERNET_ACCESS_WEIGHT,
            "learning_disabilities": config.barrier.LEARNING_DISABILITIES_WEIGHT,
            "geographic_isolation": config.barrier.GEOGRAPHIC_ISOLATION_WEIGHT,
        },
        handle_missing=config.barrier.MISSING_DATA_STRATEGY,
        standardize=True,
    )

    # Step 5: Construct barrier index
    logger.info("\nConstructing barrier index...")
    df_with_barriers = construct_barrier_index(df_joined, barrier_config)
    logger.info("Barrier index constructed successfully")

    # Cache the DataFrame for reuse
    df_with_barriers.cache()
    logger.info("Cached DataFrame with barriers for analysis")

    # Step 6: Add score clusters if requested
    if include_score_clusters:
        logger.info("\nAdding score cluster labels for cross-analysis...")
        df_with_barriers = add_cluster_labels(
            df_with_barriers, score_column="PV1MATH", cluster_column="score_cluster"
        )

    # Step 7: Perform comprehensive analysis
    logger.info("\nPerforming comprehensive barrier analysis...")
    results = perform_comprehensive_barrier_analysis(
        df_with_barriers, barrier_config, include_clustering=True
    )

    # Step 8: Print report
    logger.info("\nGenerating analysis report...")
    print_barrier_analysis_report(results, verbose=True)

    # Step 9: Generate visualizations
    logger.info("\nGenerating visualizations...")
    viz_dir = config.get_artifact_path("visualizations/barriers")
    viz_paths = create_all_barrier_visualizations(results, viz_dir)

    # Step 10: Export to CSV
    logger.info("\nExporting results to CSV...")
    csv_paths = export_results_to_csv(results, viz_dir)

    # Unpersist cached DataFrame
    df_with_barriers.unpersist()

    logger.info("\n" + "=" * 70)
    logger.info("BARRIER ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Visualizations saved to: {viz_dir}")
    logger.info(f"Total visualizations: {len(viz_paths)}")
    logger.info(f"Total CSV exports: {len(csv_paths)}")


def main(target_column: str | None = None, run_barrier_analysis: bool = True) -> None:
    """
    Main application entrypoint.

    Workflow:
    1. Initialize configuration and logging
    2. Convert SPSS files to Parquet (if not already converted)
    3. Start Spark session
    4. Load student data
    5. Phase 2: Perform score-based clustering analysis (Part A)
    6. Phase 3: Perform barrier analysis (Part B)

    Args:
        target_column: Column name to analyze. Defaults to ST059Q01TA.
        run_barrier_analysis: Whether to run Part B barrier analysis
    """
    # Default target column
    if target_column is None:
        target_column = "ST059Q01TA"

    # Suppress Java/Spark warnings for cleaner output
    warnings.filterwarnings("ignore")

    print("\n" + "=" * 70)
    print("PISA Behavior Analysis Application")
    print("=" * 70 + "\n")

    # 1. Initialize configuration and logging
    config = load_config()
    logger = setup_logger(log_dir=config.data.LOG_DIR)

    logger.info("=" * 70)
    logger.info("Behavior Analysis Application Started")
    logger.info("=" * 70)

    try:
        # 2. Ensure output directory exists
        ensure_directory_exists(config.data.PARQUET_DIR)
        logger.info(f"Parquet output directory: {config.data.PARQUET_DIR}")

        # 3. Convert SPSS files to Parquet
        logger.info("\n" + "-" * 70)
        logger.info("PHASE 1: SPSS to Parquet Conversion")
        logger.info("-" * 70)

        converter = SPSSToParquetConverter(config.conversion)

        for dataset_name, _spss_filename in config.data.SPSS_FILES.items():
            spss_path = config.get_spss_path(dataset_name)
            parquet_path = config.get_parquet_path(dataset_name)

            logger.info(f"\nProcessing dataset: {dataset_name}")
            logger.info(f"  SPSS file: {spss_path}")
            logger.info(f"  Parquet file: {parquet_path}")

            # Check if already converted
            if converter.is_converted(parquet_path, spss_path):
                logger.info("  Status: Already converted, skipping...")
                print(f"✓ {dataset_name}: Already converted")
            else:
                logger.info("  Status: Converting...")
                print(f"⟳ {dataset_name}: Converting...")

                success = converter.convert_file(spss_path, parquet_path)

                if success:
                    logger.info("  Result: Conversion successful")
                    print(f"✓ {dataset_name}: Conversion completed")
                else:
                    logger.error("  Result: Conversion failed")
                    print(f"✗ {dataset_name}: Conversion failed")

        # 4. Spark Analysis Phase
        logger.info("\n" + "-" * 70)
        logger.info("PHASE 2: Score-based Clustering Analysis")
        logger.info("-" * 70)

        # Initialize Spark session with context manager for automatic cleanup
        with SparkSessionManager(config.spark) as spark:
            logger.info("Spark session initialized")

            # Load student data
            student_parquet_path = config.get_parquet_path("student")
            logger.info(f"\nLoading student data from: {student_parquet_path}")

            student_df = spark.read.parquet(student_parquet_path)
            student_count = student_df.count()
            logger.info(f"Student data loaded: {student_count:,} records")

            # Verify required columns exist
            required_columns = ["PV1MATH", "W_FSTUWT"]
            missing_columns = [col for col in required_columns if col not in student_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            logger.info(f"Required columns verified: {required_columns}")

            # Perform score-based clustering analysis
            perform_score_clustering_analysis(student_df, config, logger)

            # Phase 3: Barrier Analysis
            if run_barrier_analysis:
                logger.info("\n" + "-" * 70)
                logger.info("PHASE 3: Barrier Analysis")
                logger.info("-" * 70)

                # Load school data for barrier analysis
                school_parquet_path = config.get_parquet_path("school")
                logger.info(f"\nLoading school data from: {school_parquet_path}")

                school_df = spark.read.parquet(school_parquet_path)
                school_count = school_df.count()
                logger.info(f"School data loaded: {school_count:,} records")

                try:
                    perform_barrier_analysis(
                        student_df, school_df, config, logger, include_score_clusters=True
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error("Barrier analysis failed: %s", e, exc_info=True)
                    print(f"\n✗ Barrier analysis failed: {e}", flush=True)
                    # Continue execution even if barrier analysis fails

    except KeyboardInterrupt:
        logger.info("\nApplication interrupted by user")
        print("\n\n⚠ Application interrupted by user")
        sys.exit(1)

    except Exception as e:  # noqa: BLE001
        logger.error("Application failed with error: %s", e, exc_info=True)
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Support command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "attitude":
            # Run attitude clustering analysis
            from .analysis.attitude_analysis import run_attitude_clustering

            run_attitude_clustering()
        elif sys.argv[1] == "score_attitude_cross_analysis":
            # Run score-attitude cross-dimensional analysis
            from .analysis.score_attitude_cross_demo import (
                run_score_attitude_cross_analysis,
            )

            run_score_attitude_cross_analysis()
        else:
            # Run default score clustering with optional column
            main(target_column=sys.argv[1])
    else:
        # Run default score clustering
        main()
