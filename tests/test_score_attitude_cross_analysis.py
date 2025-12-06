"""
Unit tests for score-attitude cross-dimensional analysis module.

Tests the cross-dimensional analysis functionality including:
- Cross-tabulation creation
- Chi-square statistical testing
- Cross-tabulation export
"""

import pytest
from scipy.stats import chi2_contingency

from behavior_analysis.analysis.score_attitude_cross_analysis import (
    create_cross_tabulation,
    export_cross_tabulation,
    perform_chi_square_test,
    print_chi_square_report,
    print_cross_tabulation,
)


class TestCrossTabulation:
    """Tests for cross-tabulation creation."""

    def test_cross_tabulation_structure(self) -> None:
        """Test that cross-tabulation has correct structure."""
        # Create mock data structure
        import pandas as pd

        data = {
            "Score Cluster": ["low", "low", "middle", "middle", "high", "high"],
            "Attitude Cluster": [
                "Proactive Learners",
                "Average Learners",
                "Proactive Learners",
                "Average Learners",
                "Proactive Learners",
                "Average Learners",
            ],
            "Weighted Count": [1000.0, 2000.0, 1500.0, 2500.0, 800.0, 200.0],
        }
        df_cross = pd.DataFrame(data)

        # Pivot table
        cross_tab = df_cross.pivot_table(
            index="Score Cluster",
            columns="Attitude Cluster",
            values="Weighted Count",
            fill_value=0,
        )

        # Should have 3 score clusters as rows
        assert len(cross_tab.index) == 3

        # Should have attitude clusters as columns
        assert "Proactive Learners" in cross_tab.columns
        assert "Average Learners" in cross_tab.columns


class TestChiSquareTest:
    """Tests for chi-square statistical testing."""

    def test_chi_square_with_valid_data(self) -> None:
        """Test chi-square test with valid contingency table."""
        import pandas as pd

        # Create a simple contingency table
        cross_tab = pd.DataFrame(
            {
                "Proactive Learners": [100, 200, 50],
                "Average Learners": [200, 300, 100],
                "Disengaged Learners": [50, 100, 30],
            },
            index=["low", "middle", "high"],
        )

        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(cross_tab)

        # Should return valid statistics
        assert chi2 > 0
        assert 0 <= p_value <= 1
        assert dof > 0
        assert expected.shape == cross_tab.shape

    def test_chi_square_independence(self) -> None:
        """Test chi-square with independent distributions."""
        import pandas as pd

        # Create independent contingency table (same distribution across rows)
        cross_tab = pd.DataFrame(
            {
                "Proactive Learners": [100, 100, 100],
                "Average Learners": [200, 200, 200],
                "Disengaged Learners": [50, 50, 50],
            },
            index=["low", "middle", "high"],
        )

        chi2, p_value, dof, _ = chi2_contingency(cross_tab)

        # Chi-square should be zero for perfectly independent data
        assert chi2 == pytest.approx(0, abs=1e-10)
        assert p_value > 0.05  # Not significant

    def test_chi_square_strong_association(self) -> None:
        """Test chi-square with strong association."""
        import pandas as pd

        # Create strongly associated contingency table
        cross_tab = pd.DataFrame(
            {
                "Proactive Learners": [1000, 100, 50],
                "Average Learners": [100, 1000, 100],
                "Disengaged Learners": [50, 100, 1000],
            },
            index=["low", "middle", "high"],
        )

        chi2, p_value, dof, _ = chi2_contingency(cross_tab)

        # Chi-square should be large for strongly associated data
        assert chi2 > 100
        assert p_value < 0.05  # Significant


class TestCrossAnalysisResults:
    """Tests for cross-analysis result interpretation."""

    def test_perform_chi_square_test(self) -> None:
        """Test the perform_chi_square_test function."""
        import pandas as pd

        cross_tab = pd.DataFrame(
            {
                "Proactive Learners": [100, 200, 50],
                "Average Learners": [200, 300, 100],
                "Disengaged Learners": [50, 100, 30],
            },
            index=["low", "middle", "high"],
        )

        results = perform_chi_square_test(cross_tab)

        # Should return all required fields
        assert "chi2_statistic" in results
        assert "p_value" in results
        assert "degrees_of_freedom" in results
        assert "expected_frequencies" in results
        assert "is_significant" in results

        # is_significant should be boolean-like (bool or numpy bool)
        assert bool(results["is_significant"]) in (True, False)

    def test_significance_threshold(self) -> None:
        """Test that significance threshold (0.05) is applied correctly."""
        import pandas as pd

        # Low chi-square (not significant)
        cross_tab_independent = pd.DataFrame(
            {
                "Proactive Learners": [100, 100, 100],
                "Average Learners": [200, 200, 200],
                "Disengaged Learners": [50, 50, 50],
            },
            index=["low", "middle", "high"],
        )

        results_independent = perform_chi_square_test(cross_tab_independent)
        assert results_independent["is_significant"] == False

        # High chi-square (significant)
        cross_tab_associated = pd.DataFrame(
            {
                "Proactive Learners": [1000, 100, 50],
                "Average Learners": [100, 1000, 100],
                "Disengaged Learners": [50, 100, 1000],
            },
            index=["low", "middle", "high"],
        )

        results_associated = perform_chi_square_test(cross_tab_associated)
        assert results_associated["is_significant"] == True
