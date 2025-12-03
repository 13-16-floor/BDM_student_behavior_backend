"""
Analysis module for statistical computations on PISA data.
"""

from .basic_stats import calculate_column_mean, describe_dataset, get_column_statistics

__all__ = [
    "calculate_column_mean",
    "get_column_statistics",
    "describe_dataset",
]
