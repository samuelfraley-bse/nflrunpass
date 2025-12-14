"""
Model training utilities for the NFL run/pass prediction project.

This module focuses on logistic regression, mirroring the Kaggle notebook:
- Base LogisticRegression(max_iter=1000)
- Optional GridSearchCV over C and class_weight

The main entry point is `train_log_reg_model`, which is called by
pipeline.run_training_pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from .config import CONFIG


# ---------------------------------------------------------------------
# Helpers to construct models and parameter grids
# ---------------------------------------------------------------------


def create_base_log_reg_model(
    max_iter: Optional[int] = None,
) -> LogisticRegression:
    """
    Create the base LogisticRegression model.

    Parameters
    ----------
    max_iter : int, optional
        Maximum number of iterations for the solver. If None, uses
        CONFIG.log_reg.max_iter.

    Returns
    -------
    model : LogisticRegression
        Unfitted logistic regression model.
    """
    if max_iter is None:
        max_iter = CONFIG.log_reg.max_iter

    # You can explicitly set solver/penalty if desired; for now we
    # let scikit-learn choose a suitable default (lbfgs, L2 penalty).
    model = LogisticRegression(max_iter=max_iter)
    return model


def get_param_grid() -> Dict[str, Any]:
    """
    Get the hyperparameter grid for GridSearchCV from CONFIG.

    Returns
    -------
    param_grid : dict
        Dictionary mapping parameter names to lists of values.
    """
    return CONFIG.log_reg.param_grid
    


# ---------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------


def train_log_reg_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_grid_search: Optional[bool] = None,
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """
    Train a logistic regression model on the given training data.

    Depending on CONFIG.log_reg.use_grid_search (or the `use_grid_search`
    argument), this will either:
    - Fit a single LogisticRegression model, or
    - Run GridSearchCV over a parameter grid and return the best model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (ideally already scaled).
    y_train : pd.Series
        Training labels (0/1 for run/pass).
    use_grid_search : bool, optional
        Whether to run GridSearchCV. If None, uses
        CONFIG.log_reg.use_grid_search.

    Returns
    -------
    model : LogisticRegression
        Trained logistic regression model (best estimator if grid search).
    info : dict
        Dictionary with training metadata, e.g.:
            - "used_grid_search": bool
            - "best_params": dict or None
            - "best_score": float or None
            - "cv_results": dict or None (if grid search)
    """
    if use_grid_search is None:
        use_grid_search = CONFIG.log_reg.use_grid_search

    base_model = create_base_log_reg_model()

    if not use_grid_search:
        # Simple fit without hyperparameter tuning
        model = base_model.fit(X_train, y_train)

        info: Dict[str, Any] = {
            "used_grid_search": False,
            "best_params": None,
            "best_score": None,
            "cv_results": None,
        }
        return model, info

    # With GridSearchCV
    param_grid = get_param_grid()
    gs_cfg = CONFIG.log_reg

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=gs_cfg.cv_folds,
        scoring=gs_cfg.scoring,
        n_jobs=gs_cfg.n_jobs,
        verbose=gs_cfg.verbose,
        refit=True,  # refit on the full training set using best params
    )

    grid_search.fit(X_train, y_train)

    best_model: LogisticRegression = grid_search.best_estimator_

    info = {
        "used_grid_search": True,
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "cv_results": grid_search.cv_results_,
    }

    return best_model, info
