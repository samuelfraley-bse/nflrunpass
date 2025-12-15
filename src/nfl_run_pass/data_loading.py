"""
Data loading and basic preparation utilities for the NFL run/pass model.

This module:
- Reads the raw play-by-play CSV
- Filters to a specific season
- Filters to run and pass plays only
- Creates the binary target column `is_pass`

It uses CONFIG from config.py so that paths, season, and column names
are controlled in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Iterable

import pandas as pd

from .config import CONFIG

PathLike = Union[str, Path]


# ---------------------------------------------------------------------
# Core loading functions
# ---------------------------------------------------------------------


def load_raw_data(csv_path: Optional[PathLike] = None) -> pd.DataFrame:
    """
    Load the raw NFL play-by-play data from a CSV file.

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to the CSV file with play-by-play data. If None, the
        default from CONFIG.paths.raw_data is used.

    Returns
    -------
    df : pd.DataFrame
        Raw dataframe as read from disk.
    """
    if csv_path is None:
        csv_path = CONFIG.paths.raw_data

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


# ---------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------


def filter_season(
    df: pd.DataFrame,
    season: Optional[int] = None,
    season_col: str = "season",
) -> pd.DataFrame:
    """
    Filter the dataframe to a single season, if specified.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.
    season : int, optional
        Season to keep (e.g., 2023). If None, the dataframe is returned
        unchanged. If not provided, defaults to CONFIG.data.season.
    season_col : str, default="season"
        Name of the column indicating the season.

    Returns
    -------
    filtered : pd.DataFrame
        Dataframe filtered to the requested season (if provided).
    """
    if season is None:
        season = CONFIG.data.season

    if season is None:
        return df

    if season_col not in df.columns:
        raise KeyError(
            f"Column '{season_col}' not found in dataframe. "
            "Cannot filter by season."
        )

    filtered = df[df[season_col] == season].copy()
    return filtered


def filter_run_pass_plays(
    df: pd.DataFrame,
    play_type_col: Optional[str] = None,
    pass_values: Optional[Iterable[str]] = None,
    run_values: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Filter the dataframe to only keep run and pass plays.

    Parameters
    ----------
    df : pd.DataFrame
        Play-by-play dataframe.
    play_type_col : str, optional
        Name of the column that indicates the play type in the dataset.
        Defaults to CONFIG.data.play_type_col.
    pass_values : iterable of str, optional
        Values in `play_type_col` that correspond to passing plays.
        Defaults to CONFIG.data.pass_values.
    run_values : iterable of str, optional
        Values in `play_type_col` that correspond to rushing plays.
        Defaults to CONFIG.data.run_values.

    Returns
    -------
    filtered : pd.DataFrame
        Dataframe containing only run and pass plays.
    """
    if play_type_col is None:
        play_type_col = CONFIG.data.play_type_col
    if pass_values is None:
        pass_values = CONFIG.data.pass_values
    if run_values is None:
        run_values = CONFIG.data.run_values

    if play_type_col not in df.columns:
        raise KeyError(
            f"Column '{play_type_col}' not found in dataframe. "
            "Update CONFIG.data.play_type_col or adapt this function."
        )

    mask_pass = df[play_type_col].isin(pass_values)
    mask_run = df[play_type_col].isin(run_values)
    filtered = df[mask_pass | mask_run].copy()

    return filtered


def add_is_pass_target(
    df: pd.DataFrame,
    play_type_col: Optional[str] = None,
    pass_values: Optional[Iterable[str]] = None,
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add a binary target column indicating whether the play is a pass.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe (typically already filtered to run+pass plays).
    play_type_col : str, optional
        Column with play type labels. Defaults to CONFIG.data.play_type_col.
    pass_values : iterable of str, optional
        Values in `play_type_col` that are considered passes.
        Defaults to CONFIG.data.pass_values.
    target_col : str, optional
        Name of the target column to create. Defaults to
        CONFIG.data.target_col.

    Returns
    -------
    df_with_target : pd.DataFrame
        Dataframe with a new 0/1 target column.
    """
    if play_type_col is None:
        play_type_col = CONFIG.data.play_type_col
    if pass_values is None:
        pass_values = CONFIG.data.pass_values
    if target_col is None:
        target_col = CONFIG.data.target_col

    if play_type_col not in df.columns:
        raise KeyError(
            f"Column '{play_type_col}' not found in dataframe. "
            f"Cannot create target column '{target_col}'."
        )

    df = df.copy()
    df[target_col] = df[play_type_col].isin(pass_values).astype(int)
    return df


# ---------------------------------------------------------------------
# High-level convenience pipeline
# ---------------------------------------------------------------------


def load_and_prepare_run_pass_data(
    csv_path: Optional[PathLike] = None,
) -> pd.DataFrame:
    """
    End-to-end loader that mirrors the first part of the Kaggle notebook:

    1. Load raw CSV
    2. Filter to the configured season (e.g., 2023)
    3. Filter to run/pass plays
    4. Create the binary `is_pass` target

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to the raw CSV file. If None, CONFIG.paths.raw_data is used.

    Returns
    -------
    df_model : pd.DataFrame
        Prepared dataframe restricted to one season, run/pass plays only,
        with an `is_pass` target column ready for feature engineering.
    """
    df_raw = load_raw_data(csv_path)
    df_season = filter_season(df_raw)
    df_run_pass = filter_run_pass_plays(df_season)
    df_model = add_is_pass_target(df_run_pass)
    return df_model
