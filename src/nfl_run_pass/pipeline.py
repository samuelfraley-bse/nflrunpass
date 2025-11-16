"""
End-to-end training pipeline for the NFL run/pass prediction model.

This module ties together:
- data loading
- feature engineering + preprocessing
- model training (logistic regression + optional tuning)
- evaluation
- artifact saving (model, scaler, feature columns)

Typical usage (in a notebook or script):

    from nfl_run_pass.pipeline import run_training_pipeline

    results = run_training_pipeline()
    model = results["model"]
    metrics = results["metrics"]
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import json

import joblib
import numpy as np
import pandas as pd

from .config import CONFIG
from .data_loading import load_and_prepare_run_pass_data
from .preprocessing import prepare_train_test_data
from .models import train_log_reg_model  # to be implemented next
from .evaluation import evaluate_run_pass_model  # to be implemented next

PathLike = Union[str, Path]


# ---------------------------------------------------------------------
# Artifact saving / loading
# ---------------------------------------------------------------------


def _ensure_artifacts_dir(artifacts_dir: Optional[PathLike] = None) -> Path:
    """
    Ensure the artifacts directory exists and return it.
    """
    if artifacts_dir is None:
        artifacts_dir = CONFIG.paths.artifacts_dir

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def save_artifacts(
    model: Any,
    scaler: Any,
    feature_cols: List[str],
    artifacts_dir: Optional[PathLike] = None,
) -> Dict[str, Path]:
    """
    Save model, scaler, and feature column names to disk.

    Paths and filenames are controlled by CONFIG.paths.artifacts_dir and
    CONFIG.artifacts.*.

    Parameters
    ----------
    model : Any
        Trained scikit-learn model.
    scaler : Any
        Fitted scaler (e.g., StandardScaler).
    feature_cols : list of str
        Ordered list of feature column names used to train the model.
    artifacts_dir : str or Path, optional
        Directory in which to save artifacts. If None, uses
        CONFIG.paths.artifacts_dir.

    Returns
    -------
    artifact_paths : dict
        Dictionary with keys "model", "scaler", "feature_cols" and
        corresponding Path objects.
    """
    artifacts_dir = _ensure_artifacts_dir(artifacts_dir)

    model_path = artifacts_dir / CONFIG.artifacts.model_filename
    scaler_path = artifacts_dir / CONFIG.artifacts.scaler_filename
    feature_cols_path = artifacts_dir / CONFIG.artifacts.feature_cols_filename

    # Save using joblib for sklearn objects
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # Save feature columns as JSON
    with feature_cols_path.open("w") as f:
        json.dump(feature_cols, f, indent=2)

    return {
        "model": model_path,
        "scaler": scaler_path,
        "feature_cols": feature_cols_path,
    }


def load_artifacts(
    artifacts_dir: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """
    Load model, scaler, and feature column names from disk.

    Parameters
    ----------
    artifacts_dir : str or Path, optional
        Directory from which to load artifacts. If None, uses
        CONFIG.paths.artifacts_dir.

    Returns
    -------
    artifacts : dict
        Dictionary with keys:
            - "model": trained model
            - "scaler": fitted scaler
            - "feature_cols": list of feature names
    """
    if artifacts_dir is None:
        artifacts_dir = CONFIG.paths.artifacts_dir

    artifacts_dir = Path(artifacts_dir)

    model_path = artifacts_dir / CONFIG.artifacts.model_filename
    scaler_path = artifacts_dir / CONFIG.artifacts.scaler_filename
    feature_cols_path = artifacts_dir / CONFIG.artifacts.feature_cols_filename

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")
    if not feature_cols_path.exists():
        raise FileNotFoundError(f"Feature cols file not found at: {feature_cols_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with feature_cols_path.open("r") as f:
        feature_cols = json.load(f)

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------


def run_training_pipeline(
    csv_path: Optional[PathLike] = None,
    save_artifacts_flag: bool = True,
    artifacts_dir: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """
    Run the full training pipeline:

    1. Load and prepare data (filter season, run/pass, create is_pass).
    2. Build features, split into train/test, and scale.
    3. Train a logistic regression model (with optional GridSearchCV).
    4. Evaluate on train and test sets.
    5. Optionally save model, scaler, and feature columns as artifacts.

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to the raw CSV file. If None, uses CONFIG.paths.raw_data.
    save_artifacts_flag : bool, default=True
        Whether to save model, scaler, and feature columns to disk.
    artifacts_dir : str or Path, optional
        Directory in which to save artifacts. If None, uses
        CONFIG.paths.artifacts_dir.

    Returns
    -------
    results : dict
        Dictionary containing:
            - "model": trained model
            - "scaler": fitted scaler
            - "feature_cols": list of feature names
            - "metrics": dict with train/test metrics
            - "artifact_paths": dict of saved artifact paths (if saved)
            - "config": snapshot of training-related config used
    """
    # -------------------------
    # 1) Load & basic prep
    # -------------------------
    df_model = load_and_prepare_run_pass_data(csv_path=csv_path)

    # -------------------------
    # 2) Features + preprocessing
    # -------------------------
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        feature_cols,
    ) = prepare_train_test_data(df_model)

    # -------------------------
    # 3) Model training
    # -------------------------
    # This will be implemented in models.py
    model, model_info = train_log_reg_model(
        X_train,
        y_train,
    )

    # -------------------------
    # 4) Evaluation
    # -------------------------
    # This will be implemented in evaluation.py
    metrics = evaluate_run_pass_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    # -------------------------
    # 5) Save artifacts (optional)
    # -------------------------
    artifact_paths: Optional[Dict[str, Path]] = None
    if save_artifacts_flag:
        artifact_paths = save_artifacts(
            model=model,
            scaler=scaler,
            feature_cols=feature_cols,
            artifacts_dir=artifacts_dir,
        )

    # Snapshot of relevant config pieces used for this run
    config_snapshot = {
        "paths": {
            "raw_data": str(CONFIG.paths.raw_data),
            "artifacts_dir": str(
                artifacts_dir if artifacts_dir is not None else CONFIG.paths.artifacts_dir
            ),
        },
        "data": asdict(CONFIG.data),
        "features": {
            "base_numeric_candidates": list(CONFIG.features.base_numeric_candidates),
            "engineered_feature_candidates": list(
                CONFIG.features.engineered_feature_candidates
            ),
        },
        "train_test": asdict(CONFIG.train_test),
        "log_reg": {
            "max_iter": CONFIG.log_reg.max_iter,
            "use_grid_search": CONFIG.log_reg.use_grid_search,
            "param_grid": CONFIG.log_reg.param_grid,
            "cv_folds": CONFIG.log_reg.cv_folds,
            "scoring": CONFIG.log_reg.scoring,
        },
    }

    results: Dict[str, Any] = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "model_info": model_info,
        "artifact_paths": artifact_paths,
        "config": config_snapshot,
    }

    return results
