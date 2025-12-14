"""
Central configuration for the nfl_run_pass library.

This file is the single source of truth for:
- data paths and basic filtering
- target / label configuration
- feature engineering choices
- model training and hyperparameter settings
- where to save trained artifacts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any


# --------------------------------------------------------------------
# Paths & data configuration
# --------------------------------------------------------------------

from pathlib import Path

# repo root = two levels up from this file: src/nfl_run_pass/config.py
REPO_ROOT = Path(__file__).resolve().parents[1]

@dataclass
class PathsConfig:
    raw_data: Path = REPO_ROOT / "data" / "raw" / "pbp_2021_2023.csv"
    artifacts_dir: Path = REPO_ROOT / "artifacts"



@dataclass
class DataConfig:
    """Configuration related to raw data and basic run/pass filtering."""

    # Only use this season in the model (as in the notebook: season == 2023)
    season: int = 2023

    # Column with play type labels ("run", "pass", etc.)
    play_type_col: str = "play_type"

    # Values that should be interpreted as pass/run for df_model
    pass_values: Tuple[str, ...] = ("pass",)
    run_values: Tuple[str, ...] = ("run",)

    # Target column we create in df_model
    target_col: str = "is_pass"

    # Optional: columns you often look at when debugging
    preview_cols: Tuple[str, ...] = (
        "season",
        "week",
        "game_id",
        "play_id",
        "play_type",
        "down",
        "ydstogo",
    )


# --------------------------------------------------------------------
# Feature configuration
# --------------------------------------------------------------------


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # Base numeric columns used directly (pre-snap).
    base_numeric_candidates: Tuple[str, ...] = (
        "down",
        "ydstogo",
        "yardline_100",
        "game_seconds_remaining",
    )

    # Engineered pre-snap features you create in features.py
    engineered_feature_candidates: Tuple[str, ...] = (
        "is_red_zone",
        "is_goal_to_go",
        "short_ydstogo",
        "medium_ydstogo",
        "long_ydstogo",
        "shotgun",
        "no_huddle",
        "score_differential",
        "is_trailing",
        "is_tied",
        "is_leading",
        "is_fourth_qtr",
        "late_half",
        "is_home_offense",
    )

    # Raw columns needed to compute those engineered features
    required_raw_cols: Tuple[str, ...] = (
        "yardline_100",
        "goal_to_go",
        "ydstogo",
        "shotgun",
        "no_huddle",
        "score_differential",
        "qtr",
        "half_seconds_remaining",
        "posteam",
        "home_team",
        "down",
        "game_seconds_remaining",
    )


# --------------------------------------------------------------------
# Model & training configuration
# --------------------------------------------------------------------


@dataclass
class TrainTestConfig:
    """Train/test split and related settings."""

    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True  # stratify by y (run vs pass)


@dataclass
class LogisticRegressionConfig:
    """Hyperparameters for logistic regression and its tuning."""

    # Base model args
    max_iter: int = 1000

    # GridSearchCV configuration
    use_grid_search: bool = True
    param_grid: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            "C": [0.01, 0.1, 1.0, 10.0],
            "class_weight": [None, "balanced"],
        }
    )
    cv_folds: int = 3
    scoring: str = "f1"
    n_jobs: int = -1
    verbose: int = 1


# --------------------------------------------------------------------
# Artifact saving configuration
# --------------------------------------------------------------------


@dataclass
class ArtifactConfig:
    """Filenames for saved artifacts."""

    model_filename: str = "log_reg_model.pkl"
    scaler_filename: str = "scaler.pkl"
    feature_cols_filename: str = "feature_cols.json"


# --------------------------------------------------------------------
# Top-level config object
# --------------------------------------------------------------------


@dataclass
class Config:
    """Bundle all configuration sections together."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train_test: TrainTestConfig = field(default_factory=TrainTestConfig)
    log_reg: LogisticRegressionConfig = field(default_factory=LogisticRegressionConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)


# ✅ Global, importable config instance
CONFIG = Config()
