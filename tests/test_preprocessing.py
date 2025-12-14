# tests/test_preprocessing.py
"""
Unit tests for the preprocessing module.

Tests cover:
- Missing value handling (imputation)
- Train/test splitting with stratification
- Feature scaling with StandardScaler
- The high-level prepare_train_test_data function
"""

import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from nfl_run_pass import preprocessing
from nfl_run_pass.config import CONFIG


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_feature_matrix():
    """Create a simple feature matrix for testing."""
    return pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature_b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "feature_c": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


def make_feature_matrix_with_nans():
    """Create a feature matrix containing NaN values."""
    return pd.DataFrame({
        "feature_a": [1.0, np.nan, 3.0, 4.0, np.nan],
        "feature_b": [10.0, 20.0, np.nan, 40.0, 50.0],
        "feature_c": [0, 1, 0, np.nan, 1],
    })


def make_target_vector():
    """Create a target vector with balanced classes."""
    return pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], name="is_pass")


def make_target_vector_imbalanced():
    """Create a target vector with imbalanced classes (70% class 0)."""
    return pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], name="is_pass")


def make_sample_df_for_pipeline():
    """
    Create a complete dataframe suitable for prepare_train_test_data.
    Must include all required raw columns for feature engineering.
    """
    n = 100  # Need enough samples for splitting
    np.random.seed(42)
    
    return pd.DataFrame({
        # Required raw columns
        "yardline_100": np.random.randint(1, 100, n),
        "goal_to_go": np.random.choice([0, 1], n, p=[0.9, 0.1]),
        "ydstogo": np.random.randint(1, 20, n),
        "shotgun": np.random.choice([0, 1], n),
        "no_huddle": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        "score_differential": np.random.randint(-21, 21, n),
        "qtr": np.random.choice([1, 2, 3, 4], n),
        "half_seconds_remaining": np.random.randint(0, 1800, n),
        "posteam": np.random.choice(["KC", "BUF"], n),
        "home_team": np.random.choice(["KC", "BUF"], n),
        "down": np.random.choice([1, 2, 3, 4], n),
        "game_seconds_remaining": np.random.randint(0, 3600, n),
        # Target column
        "is_pass": np.random.choice([0, 1], n),
    })


# ---------------------------------------------------------------------
# Tests for handle_missing_values
# ---------------------------------------------------------------------


class TestHandleMissingValues:
    """Tests for the handle_missing_values function."""

    def test_returns_dataframe_without_nans(self):
        """Should return a DataFrame with no NaN values."""
        X = make_feature_matrix_with_nans()
        assert X.isna().any().any(), "Test setup: X should have NaNs"
        
        X_imputed = preprocessing.handle_missing_values(X)
        
        assert not X_imputed.isna().any().any(), "Result should have no NaNs"

    def test_does_not_modify_original(self):
        """Should return a copy, not modify the original DataFrame."""
        X = make_feature_matrix_with_nans()
        nan_count_before = X.isna().sum().sum()
        
        X_imputed = preprocessing.handle_missing_values(X)
        nan_count_after = X.isna().sum().sum()
        
        # Original should be unchanged
        assert nan_count_before == nan_count_after
        # Result should be different object
        assert X_imputed is not X

    def test_numeric_columns_filled_with_median(self):
        """Numeric columns should have NaN filled with column median."""
        X = pd.DataFrame({
            "col": [1.0, 2.0, 3.0, np.nan, 5.0]
        })
        # Median of [1, 2, 3, 5] = 2.5
        
        X_imputed = preprocessing.handle_missing_values(X)
        
        assert X_imputed["col"].iloc[3] == 2.5

    def test_preserves_non_nan_values(self):
        """Non-NaN values should remain unchanged."""
        X = make_feature_matrix_with_nans()
        X_imputed = preprocessing.handle_missing_values(X)
        
        # Check specific non-NaN values
        assert X_imputed["feature_a"].iloc[0] == 1.0
        assert X_imputed["feature_b"].iloc[0] == 10.0

    def test_handles_all_nan_column(self):
        """Should handle columns that are entirely NaN."""
        X = pd.DataFrame({
            "normal": [1.0, 2.0, 3.0],
            "all_nan": [np.nan, np.nan, np.nan],
        })
        
        X_imputed = preprocessing.handle_missing_values(X)
        
        # All-NaN column should be filled with 0 (fallback)
        assert not X_imputed["all_nan"].isna().any()

    def test_handles_empty_dataframe(self):
        """Should handle empty DataFrame gracefully."""
        X = pd.DataFrame()
        
        X_imputed = preprocessing.handle_missing_values(X)
        
        assert len(X_imputed) == 0

    def test_preserves_column_order(self):
        """Should preserve the original column order."""
        X = make_feature_matrix_with_nans()
        original_cols = list(X.columns)
        
        X_imputed = preprocessing.handle_missing_values(X)
        
        assert list(X_imputed.columns) == original_cols

    def test_preserves_index(self):
        """Should preserve the original index."""
        X = make_feature_matrix_with_nans()
        X.index = ["a", "b", "c", "d", "e"]
        
        X_imputed = preprocessing.handle_missing_values(X)
        
        assert list(X_imputed.index) == ["a", "b", "c", "d", "e"]


# ---------------------------------------------------------------------
# Tests for split_train_test
# ---------------------------------------------------------------------


class TestSplitTrainTest:
    """Tests for the split_train_test function."""

    def test_returns_four_elements(self):
        """Should return X_train, X_test, y_train, y_test."""
        X = make_feature_matrix()
        y = make_target_vector()
        
        result = preprocessing.split_train_test(X, y)
        
        assert len(result) == 4

    def test_split_preserves_total_samples(self):
        """Train + test should equal original size."""
        X = make_feature_matrix()
        y = make_target_vector()
        
        X_train, X_test, y_train, y_test = preprocessing.split_train_test(X, y)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)

    def test_split_uses_config_test_size(self, monkeypatch):
        """Should use CONFIG.train_test.test_size by default."""
        X = make_feature_matrix()
        y = make_target_vector()
        
        monkeypatch.setattr(CONFIG.train_test, "test_size", 0.3, raising=False)
        
        X_train, X_test, y_train, y_test = preprocessing.split_train_test(X, y)
        
        expected_test_size = int(len(X) * 0.3)
        assert len(X_test) == expected_test_size

    def test_split_uses_explicit_test_size(self):
        """Should use explicit test_size when provided."""
        X = make_feature_matrix()
        y = make_target_vector()
        
        X_train, X_test, y_train, y_test = preprocessing.split_train_test(
            X, y, test_size=0.4
        )
        
        expected_test_size = int(len(X) * 0.4)
        assert len(X_test) == expected_test_size

    def test_split_is_reproducible_with_random_state(self):
        """Same random_state should give identical splits."""
        X = make_feature_matrix()
        y = make_target_vector()
        
        result1 = preprocessing.split_train_test(X, y, random_state=123)
        result2 = preprocessing.split_train_test(X, y, random_state=123)
        
        pd.testing.assert_frame_equal(result1[0], result2[0])  # X_train
        pd.testing.assert_frame_equal(result1[1], result2[1])  # X_test

    def test_split_differs_with_different_random_state(self):
        """Different random_state should give different splits."""
        X = make_feature_matrix()
        y = make_target_vector()
        
        result1 = preprocessing.split_train_test(X, y, random_state=123)
        result2 = preprocessing.split_train_test(X, y, random_state=456)
        
        # At least one should differ (very unlikely to be identical)
        assert not result1[0].equals(result2[0]) or not result1[1].equals(result2[1])

    def test_stratification_preserves_class_proportions(self):
        """With stratify=True, train and test should have similar class ratios."""
        # Create larger dataset for better stratification test
        np.random.seed(42)
        X = pd.DataFrame({"f": np.random.randn(1000)})
        y = pd.Series([0] * 700 + [1] * 300)  # 70% class 0, 30% class 1
        
        X_train, X_test, y_train, y_test = preprocessing.split_train_test(
            X, y, stratify=True, test_size=0.2, random_state=42
        )
        
        # Calculate proportions
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        original_ratio = y.mean()
        
        # Proportions should be very close (within 5%)
        assert abs(train_ratio - original_ratio) < 0.05
        assert abs(test_ratio - original_ratio) < 0.05

    def test_no_stratification_when_disabled(self):
        """With stratify=False, should not stratify."""
        X = make_feature_matrix()
        y = make_target_vector()
        
        # This should run without error even with stratify=False
        X_train, X_test, y_train, y_test = preprocessing.split_train_test(
            X, y, stratify=False
        )
        
        assert len(X_train) + len(X_test) == len(X)

    def test_X_train_and_X_test_are_dataframes(self):
        """X_train and X_test should be DataFrames."""
        X = make_feature_matrix()
        y = make_target_vector()
        
        X_train, X_test, y_train, y_test = preprocessing.split_train_test(X, y)
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)

    def test_y_train_and_y_test_are_series(self):
        """y_train and y_test should be Series."""
        X = make_feature_matrix()
        y = make_target_vector()
        
        X_train, X_test, y_train, y_test = preprocessing.split_train_test(X, y)
        
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

    def test_column_names_preserved(self):
        """Column names should be preserved in splits."""
        X = make_feature_matrix()
        y = make_target_vector()
        
        X_train, X_test, y_train, y_test = preprocessing.split_train_test(X, y)
        
        assert list(X_train.columns) == list(X.columns)
        assert list(X_test.columns) == list(X.columns)


# ---------------------------------------------------------------------
# Tests for scale_features
# ---------------------------------------------------------------------


class TestScaleFeatures:
    """Tests for the scale_features function."""

    def test_returns_three_elements(self):
        """Should return X_train_scaled, X_test_scaled, scaler."""
        X_train = make_feature_matrix()[:7]
        X_test = make_feature_matrix()[7:]
        
        result = preprocessing.scale_features(X_train, X_test)
        
        assert len(result) == 3

    def test_returns_standard_scaler(self):
        """Third element should be a fitted StandardScaler."""
        X_train = make_feature_matrix()[:7]
        X_test = make_feature_matrix()[7:]
        
        X_train_scaled, X_test_scaled, scaler = preprocessing.scale_features(
            X_train, X_test
        )
        
        assert isinstance(scaler, StandardScaler)

    def test_train_set_has_mean_near_zero(self):
        """Scaled training set should have mean ≈ 0 for each column."""
        X_train = make_feature_matrix()[:7]
        X_test = make_feature_matrix()[7:]
        
        X_train_scaled, X_test_scaled, scaler = preprocessing.scale_features(
            X_train, X_test
        )
        
        means = X_train_scaled.mean()
        for col in means.index:
            assert abs(means[col]) < 1e-10, f"Column {col} mean is not near 0"

    def test_train_set_has_std_near_one(self):
        """Scaled training set should have std ≈ 1 for each column."""
        X_train = make_feature_matrix()[:7]
        X_test = make_feature_matrix()[7:]
        
        X_train_scaled, X_test_scaled, scaler = preprocessing.scale_features(
            X_train, X_test
        )
        
        stds = X_train_scaled.std(ddof=0)  # ddof=0 to match sklearn
        for col in stds.index:
            assert abs(stds[col] - 1.0) < 1e-10, f"Column {col} std is not near 1"

    def test_scaler_fit_only_on_train(self):
        """Scaler should be fit on training data only."""
        # Create train and test with very different distributions
        X_train = pd.DataFrame({"f": [1, 2, 3, 4, 5]})
        X_test = pd.DataFrame({"f": [100, 200, 300]})  # Very different scale
        
        X_train_scaled, X_test_scaled, scaler = preprocessing.scale_features(
            X_train, X_test
        )
        
        # Scaler parameters should reflect training data
        assert scaler.mean_[0] == pytest.approx(3.0)  # mean of [1,2,3,4,5]
        
        # Test set should be transformed using train statistics
        # (100 - 3) / std should give a large positive number
        assert X_test_scaled["f"].iloc[0] > 10  # Much larger than training range

    def test_preserves_column_names(self):
        """Scaled DataFrames should have same column names."""
        X_train = make_feature_matrix()[:7]
        X_test = make_feature_matrix()[7:]
        
        X_train_scaled, X_test_scaled, scaler = preprocessing.scale_features(
            X_train, X_test
        )
        
        assert list(X_train_scaled.columns) == list(X_train.columns)
        assert list(X_test_scaled.columns) == list(X_test.columns)

    def test_preserves_index(self):
        """Scaled DataFrames should have same index."""
        X_train = make_feature_matrix()[:7]
        X_test = make_feature_matrix()[7:]
        X_train.index = list("abcdefg")
        X_test.index = list("hij")
        
        X_train_scaled, X_test_scaled, scaler = preprocessing.scale_features(
            X_train, X_test
        )
        
        assert list(X_train_scaled.index) == list("abcdefg")
        assert list(X_test_scaled.index) == list("hij")

    def test_returns_dataframes_not_arrays(self):
        """Should return DataFrames, not numpy arrays."""
        X_train = make_feature_matrix()[:7]
        X_test = make_feature_matrix()[7:]
        
        X_train_scaled, X_test_scaled, scaler = preprocessing.scale_features(
            X_train, X_test
        )
        
        assert isinstance(X_train_scaled, pd.DataFrame)
        assert isinstance(X_test_scaled, pd.DataFrame)

    def test_scaler_can_transform_new_data(self):
        """Returned scaler should work on new data."""
        X_train = make_feature_matrix()[:7]
        X_test = make_feature_matrix()[7:]
        
        X_train_scaled, X_test_scaled, scaler = preprocessing.scale_features(
            X_train, X_test
        )
        
        # Transform new data
        X_new = pd.DataFrame({"feature_a": [5.0], "feature_b": [50.0], "feature_c": [1]})
        X_new_scaled = scaler.transform(X_new)
        
        assert X_new_scaled.shape == (1, 3)


# ---------------------------------------------------------------------
# Tests for prepare_train_test_data
# ---------------------------------------------------------------------


class TestPrepareTrainTestData:
    """Tests for the high-level prepare_train_test_data function."""

    def test_returns_six_elements(self):
        """Should return X_train, X_test, y_train, y_test, scaler, feature_cols."""
        df = make_sample_df_for_pipeline()
        
        result = preprocessing.prepare_train_test_data(df)
        
        assert len(result) == 6

    def test_all_return_types_correct(self):
        """Should return correct types for all elements."""
        df = make_sample_df_for_pipeline()
        
        X_train, X_test, y_train, y_test, scaler, feature_cols = \
            preprocessing.prepare_train_test_data(df)
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert isinstance(scaler, StandardScaler)
        assert isinstance(feature_cols, list)

    def test_no_nans_in_output(self):
        """X_train and X_test should have no NaN values."""
        df = make_sample_df_for_pipeline()
        # Introduce some NaNs
        df.loc[0, "yardline_100"] = np.nan
        df.loc[5, "ydstogo"] = np.nan
        
        X_train, X_test, y_train, y_test, scaler, feature_cols = \
            preprocessing.prepare_train_test_data(df)
        
        assert not X_train.isna().any().any()
        assert not X_test.isna().any().any()

    def test_feature_cols_matches_X_columns(self):
        """feature_cols should match X_train and X_test columns."""
        df = make_sample_df_for_pipeline()
        
        X_train, X_test, y_train, y_test, scaler, feature_cols = \
            preprocessing.prepare_train_test_data(df)
        
        assert list(X_train.columns) == feature_cols
        assert list(X_test.columns) == feature_cols

    def test_with_scaling_enabled(self):
        """With scale=True, features should be scaled."""
        df = make_sample_df_for_pipeline()
        
        X_train, X_test, y_train, y_test, scaler, feature_cols = \
            preprocessing.prepare_train_test_data(df, scale=True)
        
        # Scaler should be returned
        assert scaler is not None
        
        # X_train should have mean ≈ 0
        means = X_train.mean()
        for col in means.index:
            assert abs(means[col]) < 1e-10

    def test_with_scaling_disabled(self):
        """With scale=False, features should not be scaled."""
        df = make_sample_df_for_pipeline()
        
        X_train, X_test, y_train, y_test, scaler, feature_cols = \
            preprocessing.prepare_train_test_data(df, scale=False)
        
        # Scaler should be None
        assert scaler is None
        
        # Values should not be standardized (check a feature with known range)
        assert X_train["down"].max() <= 4  # down is 1-4, not scaled

    def test_uses_default_target_col(self):
        """Should use CONFIG.data.target_col by default."""
        df = make_sample_df_for_pipeline()
        
        X_train, X_test, y_train, y_test, scaler, feature_cols = \
            preprocessing.prepare_train_test_data(df)
        
        # y should contain values from is_pass column
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})

    def test_uses_custom_target_col(self):
        """Should use custom target_col when specified."""
        df = make_sample_df_for_pipeline()
        df["custom_target"] = np.random.choice([0, 1], len(df))
        
        X_train, X_test, y_train, y_test, scaler, feature_cols = \
            preprocessing.prepare_train_test_data(df, target_col="custom_target")
        
        # y should match custom_target column values
        all_y = pd.concat([y_train, y_test]).sort_index()
        expected = df["custom_target"].loc[all_y.index]
        pd.testing.assert_series_equal(all_y, expected, check_names=False)

    def test_split_proportions_correct(self):
        """Train/test split should match CONFIG.train_test.test_size."""
        df = make_sample_df_for_pipeline()
        n = len(df)
        
        X_train, X_test, y_train, y_test, scaler, feature_cols = \
            preprocessing.prepare_train_test_data(df)
        
        expected_test_size = int(n * CONFIG.train_test.test_size)
        assert len(X_test) == expected_test_size
        assert len(X_train) == n - expected_test_size


# ---------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------


class TestPreprocessingIntegration:
    """Integration tests for the full preprocessing workflow."""

    def test_full_pipeline_is_reproducible(self):
        """Running preprocessing twice should give identical results."""
        df = make_sample_df_for_pipeline()
        
        result1 = preprocessing.prepare_train_test_data(df)
        result2 = preprocessing.prepare_train_test_data(df)
        
        pd.testing.assert_frame_equal(result1[0], result2[0])  # X_train
        pd.testing.assert_frame_equal(result1[1], result2[1])  # X_test
        pd.testing.assert_series_equal(result1[2], result2[2])  # y_train
        pd.testing.assert_series_equal(result1[3], result2[3])  # y_test
        assert result1[5] == result2[5]  # feature_cols

    def test_scaler_can_transform_single_sample(self):
        """Scaler should work for single-sample inference."""
        df = make_sample_df_for_pipeline()
        
        X_train, X_test, y_train, y_test, scaler, feature_cols = \
            preprocessing.prepare_train_test_data(df)
        
        # Simulate inference: single sample
        single_sample = X_test.iloc[[0]]
        scaled_sample = scaler.transform(single_sample)
        
        assert scaled_sample.shape == (1, len(feature_cols))

    def test_handles_edge_case_small_dataset(self):
        """Should handle small datasets (edge case)."""
        # Create minimal dataset (10 samples)
        df = make_sample_df_for_pipeline().head(10)
        
        X_train, X_test, y_train, y_test, scaler, feature_cols = \
            preprocessing.prepare_train_test_data(df)
        
        assert len(X_train) + len(X_test) == 10
