# tests/test_data_loading.py

import pandas as pd
import pytest

from nfl_run_pass import data_loading
from nfl_run_pass.config import CONFIG


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_sample_df():
    """Small helper dataframe used in multiple tests."""
    return pd.DataFrame(
        {
            "season": [2022, 2022, 2023, 2023],
            "play_type": ["pass", "run", "pass", "punt"],
            "play_id": [1, 2, 3, 4],
        }
    )


# ---------------------------------------------------------------------
# Tests for load_raw_data
# ---------------------------------------------------------------------


def test_load_raw_data_with_valid_path(tmp_path):
    # Create a temporary CSV
    df_in = make_sample_df()
    csv_path = tmp_path / "plays.csv"
    df_in.to_csv(csv_path, index=False)

    # Call loader with explicit path
    df_out = data_loading.load_raw_data(csv_path)

    pd.testing.assert_frame_equal(df_in, df_out)


def test_load_raw_data_with_invalid_path_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        data_loading.load_raw_data(missing_path)


def test_load_raw_data_uses_config_paths_raw_data(tmp_path, monkeypatch):
    # Create a temporary CSV
    df_in = make_sample_df()
    csv_path = tmp_path / "plays_config.csv"
    df_in.to_csv(csv_path, index=False)

    # Point CONFIG.paths.raw_data at this temp file
    monkeypatch.setattr(CONFIG.paths, "raw_data", csv_path, raising=False)

    df_out = data_loading.load_raw_data()

    pd.testing.assert_frame_equal(df_in, df_out)


# ---------------------------------------------------------------------
# Tests for filter_season
# ---------------------------------------------------------------------


def test_filter_season_with_explicit_season():
    df = make_sample_df()

    filtered = data_loading.filter_season(df, season=2023)

    # Only 2023 rows remain
    assert set(filtered["season"].unique()) == {2023}
    assert len(filtered) == 2


def test_filter_season_uses_config_default(monkeypatch):
    df = make_sample_df()

    # Set default season in CONFIG
    monkeypatch.setattr(CONFIG.data, "season", 2022, raising=False)

    filtered = data_loading.filter_season(df)

    assert set(filtered["season"].unique()) == {2022}
    assert len(filtered) == 2


def test_filter_season_missing_column_raises():
    df = pd.DataFrame({"year": [2022, 2023]})

    with pytest.raises(KeyError):
        data_loading.filter_season(df, season=2023, season_col="season")


# ---------------------------------------------------------------------
# Tests for filter_run_pass_plays
# ---------------------------------------------------------------------


def test_filter_run_pass_plays_explicit_values():
    df = make_sample_df()

    filtered = data_loading.filter_run_pass_plays(
        df,
        play_type_col="play_type",
        pass_values=["pass"],
        run_values=["run"],
    )

    # Should remove the "punt" row
    assert set(filtered["play_type"].unique()) == {"pass", "run"}
    assert len(filtered) == 3


def test_filter_run_pass_plays_uses_config_defaults(monkeypatch):
    df = make_sample_df()

    monkeypatch.setattr(CONFIG.data, "play_type_col", "play_type", raising=False)
    monkeypatch.setattr(CONFIG.data, "pass_values", ["pass"], raising=False)
    monkeypatch.setattr(CONFIG.data, "run_values", ["run"], raising=False)

    filtered = data_loading.filter_run_pass_plays(df)

    assert set(filtered["play_type"].unique()) == {"pass", "run"}
    assert len(filtered) == 3


def test_filter_run_pass_plays_missing_column_raises():
    df = pd.DataFrame({"some_other_col": ["pass", "run", "punt"]})

    with pytest.raises(KeyError):
        data_loading.filter_run_pass_plays(
            df,
            play_type_col="play_type",
            pass_values=["pass"],
            run_values=["run"],
        )


# ---------------------------------------------------------------------
# Tests for add_is_pass_target
# ---------------------------------------------------------------------


def test_add_is_pass_target_explicit_args():
    df = make_sample_df()

    df_with_target = data_loading.add_is_pass_target(
        df,
        play_type_col="play_type",
        pass_values=["pass"],
        target_col="is_pass",
    )

    assert "is_pass" in df_with_target.columns

    # Check that pass plays are 1, others 0
    expected = (df["play_type"] == "pass").astype(int)
    pd.testing.assert_series_equal(
        df_with_target["is_pass"], expected, check_names=False
    )


def test_add_is_pass_target_uses_config_defaults(monkeypatch):
    df = make_sample_df()

    monkeypatch.setattr(CONFIG.data, "play_type_col", "play_type", raising=False)
    monkeypatch.setattr(CONFIG.data, "pass_values", ["pass"], raising=False)
    monkeypatch.setattr(CONFIG.data, "target_col", "is_pass", raising=False)

    df_with_target = data_loading.add_is_pass_target(df)

    assert "is_pass" in df_with_target.columns
    expected = (df["play_type"] == "pass").astype(int)
    pd.testing.assert_series_equal(
        df_with_target["is_pass"], expected, check_names=False
    )


def test_add_is_pass_target_missing_play_type_col_raises():
    df = pd.DataFrame({"other_col": ["pass", "run"]})

    with pytest.raises(KeyError):
        data_loading.add_is_pass_target(
            df,
            play_type_col="play_type",
            pass_values=["pass"],
            target_col="is_pass",
        )


# ---------------------------------------------------------------------
# Integration test for load_and_prepare_run_pass_data
# ---------------------------------------------------------------------


def test_load_and_prepare_run_pass_data_pipeline(tmp_path, monkeypatch):
    # Build raw dataframe with multiple seasons and play types
    df_raw = pd.DataFrame(
        {
            "season": [2022, 2023, 2023, 2023],
            "play_type": ["punt", "run", "pass", "field_goal"],
            "play_id": [1, 2, 3, 4],
        }
    )

    csv_path = tmp_path / "raw_data.csv"
    df_raw.to_csv(csv_path, index=False)

    # Configure CONFIG to point at this CSV and define data settings
    monkeypatch.setattr(CONFIG.paths, "raw_data", csv_path, raising=False)
    monkeypatch.setattr(CONFIG.data, "season", 2023, raising=False)
    monkeypatch.setattr(CONFIG.data, "play_type_col", "play_type", raising=False)
    monkeypatch.setattr(CONFIG.data, "pass_values", ["pass"], raising=False)
    monkeypatch.setattr(CONFIG.data, "run_values", ["run"], raising=False)
    monkeypatch.setattr(CONFIG.data, "target_col", "is_pass", raising=False)

    df_model = data_loading.load_and_prepare_run_pass_data()

    # Only 2023 season, only run/pass plays
    assert set(df_model["season"].unique()) == {2023}
    assert set(df_model["play_type"].unique()) == {"run", "pass"}

    # Target column exists and is correct
    assert "is_pass" in df_model.columns
    expected = (df_model["play_type"] == "pass").astype(int)
    pd.testing.assert_series_equal(
        df_model["is_pass"], expected, check_names=False
    )
