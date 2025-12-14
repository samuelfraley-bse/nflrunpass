# tests/test_features.py
"""
Unit tests for the feature engineering module.

Tests cover:
- Individual engineered feature calculations
- Feature column selection
- Feature matrix construction
- Edge cases and error handling
"""

import pandas as pd
import numpy as np
import pytest

from nfl_run_pass import features
from nfl_run_pass.config import CONFIG


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_sample_df_with_required_cols():
    """
    Create a sample dataframe with all columns required for feature engineering.
    Includes the target column 'is_pass' as would be created by data_loading.
    """
    return pd.DataFrame({
        # Required raw columns from CONFIG.features.required_raw_cols
        "yardline_100": [75, 15, 5, 50],          # distance to opponent end zone
        "goal_to_go": [0, 0, 1, 0],               # 1 if goal-to-go situation
        "ydstogo": [10, 3, 2, 15],                # yards to first down
        "shotgun": [0, 1, 1, 0],                  # formation
        "no_huddle": [0, 0, 1, 0],                # tempo
        "score_differential": [-7, 0, 14, 3],    # offense score - defense score
        "qtr": [1, 2, 4, 3],                      # quarter
        "half_seconds_remaining": [900, 100, 60, 500],
        "posteam": ["KC", "KC", "BUF", "BUF"],   # team with possession
        "home_team": ["KC", "KC", "KC", "BUF"],  # home team
        "down": [1, 3, 2, 1],
        "game_seconds_remaining": [3600, 1900, 300, 2100],
        # Target column (created by data_loading)
        "is_pass": [1, 0, 1, 0],
        # Extra columns that might exist in real data
        "play_type": ["pass", "run", "pass", "run"],
        "play_id": [1, 2, 3, 4],
    })


def make_minimal_df_missing_cols():
    """Create a dataframe missing required columns for error testing."""
    return pd.DataFrame({
        "down": [1, 2],
        "ydstogo": [10, 5],
        # Missing most required columns
    })


def make_base_df(n_rows: int) -> pd.DataFrame:
    """
    Create a base dataframe with all required columns, filled with default values.
    This ensures all columns have the same length.
    """
    return pd.DataFrame({
        "yardline_100": [50] * n_rows,
        "goal_to_go": [0] * n_rows,
        "ydstogo": [10] * n_rows,
        "shotgun": [0] * n_rows,
        "no_huddle": [0] * n_rows,
        "score_differential": [0] * n_rows,
        "qtr": [1] * n_rows,
        "half_seconds_remaining": [900] * n_rows,
        "posteam": ["KC"] * n_rows,
        "home_team": ["KC"] * n_rows,
        "down": [1] * n_rows,
        "game_seconds_remaining": [3600] * n_rows,
    })


# ---------------------------------------------------------------------
# Tests for _ensure_required_columns (internal helper)
# ---------------------------------------------------------------------


def test_ensure_required_columns_passes_with_all_cols():
    """Should not raise when all required columns are present."""
    df = make_sample_df_with_required_cols()
    # Should not raise
    features._ensure_required_columns(df, CONFIG.features.required_raw_cols)


def test_ensure_required_columns_raises_with_missing_cols():
    """Should raise KeyError listing missing columns."""
    df = make_minimal_df_missing_cols()
    
    with pytest.raises(KeyError) as exc_info:
        features._ensure_required_columns(df, CONFIG.features.required_raw_cols)
    
    # Check that error message mentions missing columns
    assert "missing" in str(exc_info.value).lower()


# ---------------------------------------------------------------------
# Tests for add_engineered_features
# ---------------------------------------------------------------------


class TestAddEngineeredFeatures:
    """Tests for the add_engineered_features function."""

    def test_returns_dataframe_with_new_columns(self):
        """Should add all engineered feature columns."""
        df = make_sample_df_with_required_cols()
        df_feat = features.add_engineered_features(df)
        
        expected_new_cols = [
            "is_red_zone",
            "is_goal_to_go",
            "short_ydstogo",
            "medium_ydstogo",
            "long_ydstogo",
            "is_trailing",
            "is_tied",
            "is_leading",
            "is_fourth_qtr",
            "late_half",
            "is_home_offense",
        ]
        
        for col in expected_new_cols:
            assert col in df_feat.columns, f"Missing column: {col}"

    def test_does_not_modify_original_dataframe(self):
        """Should return a copy, not modify in place."""
        df = make_sample_df_with_required_cols()
        original_cols = set(df.columns)
        
        df_feat = features.add_engineered_features(df)
        
        # Original should be unchanged
        assert set(df.columns) == original_cols
        # Result should have more columns
        assert len(df_feat.columns) > len(df.columns)

    def test_is_red_zone_calculation(self):
        """is_red_zone should be 1 when yardline_100 <= 20."""
        df = make_base_df(4)
        df["yardline_100"] = [21, 20, 15, 1]
        
        df_feat = features.add_engineered_features(df)
        
        expected = [0, 1, 1, 1]  # 21 is not red zone, 20/15/1 are
        assert df_feat["is_red_zone"].tolist() == expected

    def test_is_goal_to_go_uses_existing_column(self):
        """is_goal_to_go should use the goal_to_go column from data."""
        df = make_base_df(4)
        df["goal_to_go"] = [0, 1, 1, 0]
        
        df_feat = features.add_engineered_features(df)
        
        expected = [0, 1, 1, 0]
        assert df_feat["is_goal_to_go"].tolist() == expected

    def test_ydstogo_buckets_are_mutually_exclusive(self):
        """Each play should be in exactly one ydstogo bucket."""
        df = make_base_df(6)
        df["ydstogo"] = [1, 3, 4, 7, 8, 15]
        
        df_feat = features.add_engineered_features(df)
        
        # Sum of buckets should be 1 for each row
        bucket_sum = (
            df_feat["short_ydstogo"] + 
            df_feat["medium_ydstogo"] + 
            df_feat["long_ydstogo"]
        )
        assert all(bucket_sum == 1), "Each row should be in exactly one bucket"

    def test_short_ydstogo_threshold(self):
        """short_ydstogo should be 1 when ydstogo <= 3."""
        df = make_base_df(4)
        df["ydstogo"] = [1, 2, 3, 4]
        
        df_feat = features.add_engineered_features(df)
        
        expected = [1, 1, 1, 0]
        assert df_feat["short_ydstogo"].tolist() == expected

    def test_medium_ydstogo_threshold(self):
        """medium_ydstogo should be 1 when 4 <= ydstogo <= 7."""
        df = make_base_df(5)
        df["ydstogo"] = [3, 4, 5, 7, 8]
        
        df_feat = features.add_engineered_features(df)
        
        expected = [0, 1, 1, 1, 0]
        assert df_feat["medium_ydstogo"].tolist() == expected

    def test_long_ydstogo_threshold(self):
        """long_ydstogo should be 1 when ydstogo > 7."""
        df = make_base_df(4)
        df["ydstogo"] = [7, 8, 10, 20]
        
        df_feat = features.add_engineered_features(df)
        
        expected = [0, 1, 1, 1]
        assert df_feat["long_ydstogo"].tolist() == expected

    def test_score_state_mutually_exclusive(self):
        """Each play should have exactly one score state."""
        df = make_base_df(5)
        df["score_differential"] = [-10, -1, 0, 1, 10]
        
        df_feat = features.add_engineered_features(df)
        
        state_sum = (
            df_feat["is_trailing"] + 
            df_feat["is_tied"] + 
            df_feat["is_leading"]
        )
        assert all(state_sum == 1), "Each row should have exactly one score state"

    def test_is_trailing_calculation(self):
        """is_trailing should be 1 when score_differential < 0."""
        df = make_base_df(3)
        df["score_differential"] = [-7, 0, 7]
        
        df_feat = features.add_engineered_features(df)
        
        expected = [1, 0, 0]
        assert df_feat["is_trailing"].tolist() == expected

    def test_is_tied_calculation(self):
        """is_tied should be 1 when score_differential == 0."""
        df = make_base_df(3)
        df["score_differential"] = [-7, 0, 7]
        
        df_feat = features.add_engineered_features(df)
        
        expected = [0, 1, 0]
        assert df_feat["is_tied"].tolist() == expected

    def test_is_leading_calculation(self):
        """is_leading should be 1 when score_differential > 0."""
        df = make_base_df(3)
        df["score_differential"] = [-7, 0, 7]
        
        df_feat = features.add_engineered_features(df)
        
        expected = [0, 0, 1]
        assert df_feat["is_leading"].tolist() == expected

    def test_is_fourth_qtr_calculation(self):
        """is_fourth_qtr should be 1 only in quarter 4."""
        df = make_base_df(4)
        df["qtr"] = [1, 2, 3, 4]
        
        df_feat = features.add_engineered_features(df)
        
        expected = [0, 0, 0, 1]
        assert df_feat["is_fourth_qtr"].tolist() == expected

    def test_late_half_calculation(self):
        """late_half should be 1 when half_seconds_remaining <= 120."""
        df = make_base_df(4)
        df["half_seconds_remaining"] = [121, 120, 60, 0]
        
        df_feat = features.add_engineered_features(df)
        
        expected = [0, 1, 1, 1]
        assert df_feat["late_half"].tolist() == expected

    def test_is_home_offense_calculation(self):
        """is_home_offense should be 1 when posteam == home_team."""
        df = make_base_df(3)
        df["posteam"] = ["KC", "BUF", "KC"]
        df["home_team"] = ["KC", "KC", "BUF"]
        
        df_feat = features.add_engineered_features(df)
        
        expected = [1, 0, 0]  # Only first row has posteam == home_team
        assert df_feat["is_home_offense"].tolist() == expected

    def test_handles_nan_in_shotgun(self):
        """Should fill NaN in shotgun with 0."""
        df = make_base_df(3)
        df["shotgun"] = [1, np.nan, 0]
        
        df_feat = features.add_engineered_features(df)
        
        expected = [1, 0, 0]
        assert df_feat["shotgun"].tolist() == expected

    def test_handles_nan_in_no_huddle(self):
        """Should fill NaN in no_huddle with 0."""
        df = make_base_df(3)
        df["no_huddle"] = [np.nan, 1, 0]
        
        df_feat = features.add_engineered_features(df)
        
        expected = [0, 1, 0]
        assert df_feat["no_huddle"].tolist() == expected

    def test_handles_nan_in_score_differential(self):
        """Should fill NaN in score_differential with 0."""
        df = make_base_df(3)
        df["score_differential"] = [np.nan, 7, -7]
        
        df_feat = features.add_engineered_features(df)
        
        # NaN becomes 0, which means is_tied should be 1
        assert df_feat["score_differential"].iloc[0] == 0
        assert df_feat["is_tied"].iloc[0] == 1

    def test_raises_with_missing_required_columns(self):
        """Should raise KeyError when required columns are missing."""
        df = make_minimal_df_missing_cols()
        
        with pytest.raises(KeyError):
            features.add_engineered_features(df)


# ---------------------------------------------------------------------
# Tests for get_feature_columns
# ---------------------------------------------------------------------


class TestGetFeatureColumns:
    """Tests for the get_feature_columns function."""

    def test_returns_list_of_strings(self):
        """Should return a list of column name strings."""
        cols = features.get_feature_columns()
        
        assert isinstance(cols, list)
        assert all(isinstance(c, str) for c in cols)

    def test_includes_base_numeric_columns(self):
        """Should include all base numeric candidates from CONFIG."""
        cols = features.get_feature_columns()
        
        for base_col in CONFIG.features.base_numeric_candidates:
            assert base_col in cols, f"Missing base column: {base_col}"

    def test_includes_engineered_columns(self):
        """Should include all engineered feature candidates from CONFIG."""
        cols = features.get_feature_columns()
        
        for eng_col in CONFIG.features.engineered_feature_candidates:
            assert eng_col in cols, f"Missing engineered column: {eng_col}"

    def test_no_duplicate_columns(self):
        """Should not have duplicate column names."""
        cols = features.get_feature_columns()
        
        assert len(cols) == len(set(cols)), "Duplicate columns found"

    def test_base_columns_come_first(self):
        """Base numeric columns should appear before engineered columns."""
        cols = features.get_feature_columns()
        base = list(CONFIG.features.base_numeric_candidates)
        
        # Check that base columns appear in order at the start
        for i, base_col in enumerate(base):
            if base_col in cols:
                assert cols.index(base_col) == i, \
                    f"Base column {base_col} not in expected position"


# ---------------------------------------------------------------------
# Tests for build_feature_matrix
# ---------------------------------------------------------------------


class TestBuildFeatureMatrix:
    """Tests for the build_feature_matrix function."""

    def test_returns_tuple_of_three(self):
        """Should return (X, y, feature_cols) tuple."""
        df = make_sample_df_with_required_cols()
        result = features.build_feature_matrix(df)
        
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_X_is_dataframe_with_correct_columns(self):
        """X should be a DataFrame with feature columns."""
        df = make_sample_df_with_required_cols()
        X, y, feature_cols = features.build_feature_matrix(df)
        
        assert isinstance(X, pd.DataFrame)
        assert list(X.columns) == feature_cols

    def test_y_is_series_with_target_values(self):
        """y should be a Series with 0/1 target values."""
        df = make_sample_df_with_required_cols()
        X, y, feature_cols = features.build_feature_matrix(df)
        
        assert isinstance(y, pd.Series)
        assert set(y.unique()).issubset({0, 1})

    def test_X_and_y_have_same_length(self):
        """X and y should have the same number of rows."""
        df = make_sample_df_with_required_cols()
        X, y, feature_cols = features.build_feature_matrix(df)
        
        assert len(X) == len(y) == len(df)

    def test_feature_cols_matches_X_columns(self):
        """feature_cols should exactly match X.columns."""
        df = make_sample_df_with_required_cols()
        X, y, feature_cols = features.build_feature_matrix(df)
        
        assert list(X.columns) == feature_cols

    def test_uses_default_target_col_from_config(self):
        """Should use CONFIG.data.target_col by default."""
        df = make_sample_df_with_required_cols()
        X, y, feature_cols = features.build_feature_matrix(df)
        
        expected_y = df[CONFIG.data.target_col].astype(int)
        pd.testing.assert_series_equal(y, expected_y, check_names=False)

    def test_uses_custom_target_col_when_specified(self):
        """Should use custom target_col when provided."""
        df = make_sample_df_with_required_cols()
        df["custom_target"] = [0, 1, 0, 1]
        
        X, y, feature_cols = features.build_feature_matrix(df, target_col="custom_target")
        
        expected_y = pd.Series([0, 1, 0, 1])
        pd.testing.assert_series_equal(y, expected_y, check_names=False)

    def test_raises_when_target_col_missing(self):
        """Should raise KeyError when target column doesn't exist."""
        df = make_sample_df_with_required_cols()
        df = df.drop(columns=["is_pass"])
        
        with pytest.raises(KeyError) as exc_info:
            features.build_feature_matrix(df)
        
        assert "target" in str(exc_info.value).lower() or "is_pass" in str(exc_info.value)

    def test_raises_when_feature_cols_missing_after_engineering(self, monkeypatch):
        """Should raise if CONFIG specifies features that aren't created."""
        df = make_sample_df_with_required_cols()
        
        # Add a fake feature to CONFIG that won't be created
        fake_features = CONFIG.features.engineered_feature_candidates + ("fake_feature",)
        monkeypatch.setattr(
            CONFIG.features, 
            "engineered_feature_candidates", 
            fake_features, 
            raising=False
        )
        
        with pytest.raises(KeyError) as exc_info:
            features.build_feature_matrix(df)
        
        assert "fake_feature" in str(exc_info.value)

    def test_all_feature_values_are_numeric(self):
        """All feature values should be numeric (int or float)."""
        df = make_sample_df_with_required_cols()
        X, y, feature_cols = features.build_feature_matrix(df)
        
        for col in X.columns:
            assert pd.api.types.is_numeric_dtype(X[col]), \
                f"Column {col} is not numeric"


# ---------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------


class TestFeatureEngineeringIntegration:
    """Integration tests for the full feature engineering workflow."""

    def test_full_workflow_produces_valid_output(self):
        """End-to-end test of feature engineering."""
        df = make_sample_df_with_required_cols()
        
        # Run full workflow
        X, y, feature_cols = features.build_feature_matrix(df)
        
        # Verify output
        assert len(X) == 4
        assert len(y) == 4
        assert len(feature_cols) == len(X.columns)
        assert not X.isna().any().any(), "X should have no NaN values"

    def test_feature_engineering_is_deterministic(self):
        """Running feature engineering twice should give identical results."""
        df = make_sample_df_with_required_cols()
        
        X1, y1, cols1 = features.build_feature_matrix(df)
        X2, y2, cols2 = features.build_feature_matrix(df)
        
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)
        assert cols1 == cols2
