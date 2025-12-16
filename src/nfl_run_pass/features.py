"""
FIXED features.py - 2-minute drill is CONTEXT AWARE

KEY FIXES:
1. two_minute_drill_ONLY_IF_TRAILING (not when leading by 21!)
2. final_minute_ONLY_IF_CLOSE (not when game is decided!)
3. Stronger leading_late_game signal

The model was treating "under 2 min" as urgency regardless of score.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import CONFIG


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


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pre-snap engineered features with CONTEXT-AWARE time features.
    """
    _ensure_required_columns(df, CONFIG.features.required_raw_cols)

    df = df.copy()

    # =======================================================================
    # BASIC FEATURES
    # =======================================================================

    df["is_red_zone"] = (df["yardline_100"] <= 20).astype(int)
    df["is_goal_to_go"] = df["goal_to_go"].astype(int)
    
    df["short_ydstogo"] = (df["ydstogo"] <= 3).astype(int)
    df["medium_ydstogo"] = ((df["ydstogo"] > 3) & (df["ydstogo"] <= 7)).astype(int)
    df["long_ydstogo"] = (df["ydstogo"] > 7).astype(int)

    df["shotgun"] = df["shotgun"].fillna(0).astype(int)
    df["no_huddle"] = df["no_huddle"].fillna(0).astype(int)

    df["is_trailing"] = (df["score_differential"] < 0).astype(int)
    df["is_tied"] = (df["score_differential"] == 0).astype(int)
    df["is_leading"] = (df["score_differential"] > 0).astype(int)

    df["is_fourth_qtr"] = (df["qtr"] == 4).astype(int)
    df["late_half"] = (df["half_seconds_remaining"] <= 120).astype(int)
    df["is_home_offense"] = (df["posteam"] == df["home_team"]).astype(int)

    # =======================================================================
    # SHORT YARDAGE (highest priority)
    # =======================================================================

    df["fourth_and_one"] = ((df["down"] == 4) & (df["ydstogo"] == 1)).astype(int)
    df["fourth_and_two"] = ((df["down"] == 4) & (df["ydstogo"] == 2)).astype(int)
    df["fourth_and_three"] = ((df["down"] == 4) & (df["ydstogo"] == 3)).astype(int)
    df["third_and_one"] = ((df["down"] == 3) & (df["ydstogo"] == 1)).astype(int)
    df["third_and_two"] = ((df["down"] == 3) & (df["ydstogo"] == 2)).astype(int)
    df["third_and_three"] = ((df["down"] == 3) & (df["ydstogo"] == 3)).astype(int)

    # =======================================================================
    # GOAL LINE
    # =======================================================================
    
    df["goal_line_short"] = (
        (df["yardline_100"] <= 3) & (df["ydstogo"] <= 3)
    ).astype(int)
    
    df["goal_line_one_yard"] = (
        (df["yardline_100"] == 1) & (df["ydstogo"] == 1)
    ).astype(int)

    # =======================================================================
    # 2-MINUTE DRILL - NOW CONTEXT AWARE!
    # =======================================================================
    
    # Basic 2-minute indicator (for all teams)
    under_two_minutes = (df["half_seconds_remaining"] <= 120) & (df["half_seconds_remaining"] > 0)
    under_one_minute = (df["half_seconds_remaining"] <= 60) & (df["half_seconds_remaining"] > 0)
    
    # ✅ FIX: Only count as "2-minute drill" if you're losing or tied!
    # If you're winning big, it's clock management, not a drill
    df["two_minute_drill"] = (
        under_two_minutes & 
        (df["score_differential"] <= 3)  # Only if losing or close
    ).astype(int)
    
    # ✅ FIX: Trailing-specific 2-minute drill
    df["two_minute_drill_trailing"] = (
        under_two_minutes & 
        (df["score_differential"] < 0)  # Actually losing
    ).astype(int)
    
    # ✅ FIX: Final minute only matters in close games
    df["final_minute"] = (
        under_one_minute & 
        (abs(df["score_differential"]) <= 8)  # Within one score
    ).astype(int)
    
    # ✅ NEW: 2-minute drill + long yardage (pass situation)
    df["two_minute_and_long"] = (
        under_two_minutes & 
        (df["ydstogo"] >= 7) &
        (df["score_differential"] <= 3)  # Only if close
    ).astype(int)

    # =======================================================================
    # SCORE CONTEXT
    # =======================================================================
    
    # Trailing by multiple scores
    df["trailing_multi_score"] = (df["score_differential"] <= -9).astype(int)
    
    # ✅ STRENGTHEN: Leading late = kill clock (STRONG RUN signal)
    df["leading_late_game"] = (
        (df["score_differential"] >= 3) &  # Lowered threshold from 7
        (df["game_seconds_remaining"] <= 300) &
        (df["qtr"] == 4)
    ).astype(int)
    
    # ✅ NEW: Massive lead late = DEFINITELY run
    df["blowout_lead_late"] = (
        (df["score_differential"] >= 14) &
        (df["game_seconds_remaining"] <= 300) &
        (df["qtr"] == 4)
    ).astype(int)
    
    # Close game in 4th
    df["close_game_fourth_qtr"] = (
        (abs(df["score_differential"]) <= 3) &
        (df["qtr"] == 4)
    ).astype(int)

    # =======================================================================
    # DOWN × DISTANCE
    # =======================================================================
    
    df["down_x_ydstogo"] = df["down"] * df["ydstogo"]
    
    df["third_or_fourth_and_long"] = (
        (df["down"].isin([3, 4])) & (df["ydstogo"] >= 10)
    ).astype(int)
    
    df["first_and_ten"] = (
        (df["down"] == 1) & (df["ydstogo"] == 10)
    ).astype(int)

    # =======================================================================
    # SCORE × TIME INTERACTION
    # =======================================================================
    
    # This captures urgency properly:
    # - Leading late (positive × low time) = NEGATIVE contribution = RUN
    # - Trailing late (negative × low time) = POSITIVE contribution = PASS
    time_remaining_pct = np.clip(df["game_seconds_remaining"] / 3600, 0, 1)
    df["score_time_pressure"] = df["score_differential"] * (1 - time_remaining_pct)

    return df


def get_feature_columns() -> List[str]:
    """Get the final list of feature columns to use for modeling."""
    base = list(CONFIG.features.base_numeric_candidates)
    engineered = list(CONFIG.features.engineered_feature_candidates)

    feature_cols: List[str] = []
    for col in base + engineered:
        if col not in feature_cols:
            feature_cols.append(col)

    return feature_cols


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Build feature matrix for modeling."""
    if target_col is None:
        target_col = CONFIG.data.target_col

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    # Add engineered features
    df_feat = add_engineered_features(df)

    # Select feature columns
    feature_cols = get_feature_columns()

    missing_feats = [c for c in feature_cols if c not in df_feat.columns]
    if missing_feats:
        raise KeyError(
            f"The following configured feature columns are missing: {missing_feats}"
        )

    X = df_feat[feature_cols].copy()
    y = df_feat[target_col].astype(int).copy()

    return X, y, feature_cols