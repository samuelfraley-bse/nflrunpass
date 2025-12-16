"""
Updated config with context-aware time features and blowout logic
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any


REPO_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class PathsConfig:
    raw_data: Path = REPO_ROOT / "data" / "raw" / "pbp_2021_2023.csv"
    artifacts_dir: Path = REPO_ROOT / "artifacts"


@dataclass
class DataConfig:
    """Configuration related to raw data and basic run/pass filtering."""
    season: int = 2023
    play_type_col: str = "play_type"
    pass_values: Tuple[str, ...] = ("pass",)
    run_values: Tuple[str, ...] = ("run",)
    target_col: str = "is_pass"
    preview_cols: Tuple[str, ...] = (
        "season", "week", "game_id", "play_id", "play_type",
        "down", "ydstogo",
    )


@dataclass
class FeatureConfig:
    """Feature configuration with context-aware time features."""

    base_numeric_candidates: Tuple[str, ...] = (
        "down",
        "ydstogo",
        "yardline_100",
    )

    engineered_feature_candidates: Tuple[str, ...] = (
        # --- Field position ---
        "is_red_zone",
        "is_goal_to_go",
        "short_ydstogo",
        "medium_ydstogo",
        "long_ydstogo",
        
        # --- Formation ---
        "shotgun",
        "no_huddle",
        
        # --- Score state ---
        "is_trailing",
        "is_tied",
        "is_leading",
        
        # --- Time ---
        "is_fourth_qtr",
        "late_half",
        
        # --- Other ---
        "is_home_offense",
        
        # --- SHORT YARDAGE ---
        "fourth_and_one",
        "fourth_and_two",
        "fourth_and_three",
        "third_and_one",
        "third_and_two",
        "third_and_three",
        
        # --- Goal line ---
        "goal_line_short",
        "goal_line_one_yard",
        
        # --- 2-MINUTE DRILL (now context-aware!) ---
        "two_minute_drill",              # Only if losing/tied/close
        "two_minute_drill_trailing",     # Only if actually losing
        "final_minute",                  # Only if within 8 points
        "two_minute_and_long",          # Long yardage in 2-min drill
        
        # --- Score CONTEXT ---
        "trailing_multi_score",          # Down 9+
        "leading_late_game",            # Up 3+ with <5 min
        "blowout_lead_late",            # ✅ NEW: Up 14+ with <5 min = HEAVY RUN
        "close_game_fourth_qtr",        # Within 3 in Q4
        "score_time_pressure",          # Continuous interaction
        
        # --- Down × distance ---
        "down_x_ydstogo",
        "third_or_fourth_and_long",
        "first_and_ten",
    )

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


@dataclass
class TrainTestConfig:
    """Train/test split and related settings."""
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


@dataclass
class LogisticRegressionConfig:
    """Hyperparameters for logistic regression and its tuning."""
    max_iter: int = 1000
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


@dataclass
class ArtifactConfig:
    """Filenames for saved artifacts."""
    model_filename: str = "log_reg_model.pkl"
    scaler_filename: str = "scaler.pkl"
    feature_cols_filename: str = "feature_cols.json"


@dataclass
class Config:
    """Bundle all configuration sections together."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train_test: TrainTestConfig = field(default_factory=TrainTestConfig)
    log_reg: LogisticRegressionConfig = field(default_factory=LogisticRegressionConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)


CONFIG = Config()