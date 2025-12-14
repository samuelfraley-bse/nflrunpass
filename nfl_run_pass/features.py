"""
Feature engineering utilities for the NFL run/pass model.

This module:
- Creates pre-snap engineered features (red zone, goal-to-go, etc.)
- Selects the final set of feature columns based on CONFIG
- Returns a clean feature matrix X and target vector y
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import CONFIG


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _ensure_required_columns(df: pd.DataFrame, required_cols: Tuple[str, ...]) -> None:
    """Raise a helpful error if any required columns are missing."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            "The following required columns are missing from the dataframe: "
            f"{missing}. Make sure your data_loading step produced them "
            "and that the column names in CONFIG.features.required_raw_cols "
            "match your CSV."
        )


# ---------------------------------------------------------------------
# Core feature engineering
# ---------------------------------------------------------------------


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pre-snap engineered features to the dataframe.

    This mirrors (in library form) the transformations from your
    "New Cell 3 (no leakage)" Kaggle cell. All features are derived
    from information available *before* the snap.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least the columns specified in
        CONFIG.features.required_raw_cols.

    Returns
    -------
    df_feat : pd.DataFrame
        Dataframe with engineered feature columns added.
    """
    _ensure_required_columns(df, CONFIG.features.required_raw_cols)

    df = df.copy()

    # --- Field position / scoring context --------------------------------

    # Red zone: inside opponent 20-yard line
    df["is_red_zone"] = (df["yardline_100"] <= 20).astype(int)

    # Goal-to-go situations (already a column in nflfastR-style data)
    # goal_to_go is usually 1/0 or True/False
    df["is_goal_to_go"] = df["goal_to_go"].astype(int)

    # Yards-to-go buckets: short / medium / long
    df["short_ydstogo"] = (df["ydstogo"] <= 3).astype(int)
    df["medium_ydstogo"] = ((df["ydstogo"] > 3) & (df["ydstogo"] <= 7)).astype(int)
    df["long_ydstogo"] = (df["ydstogo"] > 7).astype(int)

    # --- Formation / tempo ------------------------------------------------

    # shotgun and no_huddle are typically already 0/1 or True/False
    # We keep them as 0/1 integers for the model
    df["shotgun"] = df["shotgun"].fillna(0).astype(int)
    df["no_huddle"] = df["no_huddle"].fillna(0).astype(int)

    # --- Score state ------------------------------------------------------

    # score_differential is usually offense_score - defense_score
    # (already present in nflfastR). We keep it numeric and also
    # derive categorical score state indicators.
    df["score_differential"] = df["score_differential"].fillna(0)

    df["is_trailing"] = (df["score_differential"] < 0).astype(int)
    df["is_tied"] = (df["score_differential"] == 0).astype(int)
    df["is_leading"] = (df["score_differential"] > 0).astype(int)

    # --- Time context -----------------------------------------------------

    # Fourth quarter flag
    df["is_fourth_qtr"] = (df["qtr"] == 4).astype(int)

    # "Late half" indicator: under 2 minutes left in the half
    df["late_half"] = (df["half_seconds_remaining"] <= 120).astype(int)

    # --- Home/away offense ------------------------------------------------

    # Whether the offensive team is the home team
    df["is_home_offense"] = (df["posteam"] == df["home_team"]).astype(int)

    return df


# ---------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------


def get_feature_columns() -> List[str]:
    """
    Get the final list of feature columns to use for modeling,
    based on CONFIG.

    Returns
    -------
    feature_cols : list of str
        Column names that should be used as features.
    """
    base = list(CONFIG.features.base_numeric_candidates)
    engineered = list(CONFIG.features.engineered_feature_candidates)

    # Ensure uniqueness and preserve order (base first, then engineered)
    feature_cols: List[str] = []
    for col in base + engineered:
        if col not in feature_cols:
            feature_cols.append(col)

    return feature_cols


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Given a dataframe with raw columns, add engineered features and
    construct the (X, y) pair for modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe produced by data_loading.load_and_prepare_run_pass_data().
        Must contain the target column and all required raw columns.
    target_col : str, optional
        Name of the target column. If None, defaults to CONFIG.data.target_col.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with columns as returned by get_feature_columns().
    y : pd.Series
        Target vector (0/1 indicating run/pass).
    feature_cols : list of str
        The list of feature column names used in X (for saving later).
    """
    if target_col is None:
        target_col = CONFIG.data.target_col

    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in dataframe. "
            "Make sure you ran add_is_pass_target / load_and_prepare_run_pass_data "
            "before building features."
        )

    # 1) Add engineered features
    df_feat = add_engineered_features(df)

    # 2) Select feature columns
    feature_cols = get_feature_columns()

    missing_feats = [c for c in feature_cols if c not in df_feat.columns]
    if missing_feats:
        raise KeyError(
            "The following configured feature columns are missing after "
            "feature engineering: "
            f"{missing_feats}. Either adjust CONFIG.features.* or make sure "
            "add_engineered_features creates them."
        )

    X = df_feat[feature_cols].copy()
    y = df_feat[target_col].astype(int).copy()

    return X, y, feature_cols
