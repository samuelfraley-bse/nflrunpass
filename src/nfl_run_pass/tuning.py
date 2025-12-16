"""
tuning.py

Standalone hyperparameter tuning utilities for the NFL run/pass model.

This module allows running GridSearchCV independently of the main
pipeline, useful for experimentation and comparing configurations.

It uses:
- CONFIG.log_reg.param_grid
- CONFIG.log_reg.cv_folds
- CONFIG.log_reg.scoring
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from .config import CONFIG
from .preprocessing import prepare_train_test_data
from .features import build_feature_matrix
from .models import create_base_log_reg_model


# ------------------------------------------------------------
# 1. Standalone hyperparameter tuning
# ------------------------------------------------------------

def run_log_reg_tuning(df_model: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
    """
    Run GridSearchCV for logistic regression on df_model.

    Returns
    -------
    best_model : fitted LogisticRegression
    tuning_results : dict
        - best_params
        - best_score
        - cv_results
    """

    # Build feature matrix (not scaled yet)
    X, y, feature_cols = build_feature_matrix(df_model)

    # Split and scale the same way as pipeline
    X_train, X_test, y_train, y_test, scaler, _ = prepare_train_test_data(df_model)

    # Construct model and param grid
    base_model = create_base_log_reg_model()
    param_grid = CONFIG.log_reg.param_grid
    cfg = CONFIG.log_reg

    gs = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cfg.cv_folds,
        scoring=cfg.scoring,
        n_jobs=cfg.n_jobs,
        verbose=cfg.verbose,
        refit=True
    )

    gs.fit(X_train, y_train)

    tuning_results = {
        "best_params": gs.best_params_,
        "best_score": gs.best_score_,
        "cv_results": gs.cv_results_,
    }

    return gs.best_estimator_, tuning_results


# ------------------------------------------------------------
# 2. Compare baseline model vs tuned model
# ------------------------------------------------------------

def compare_models(df_model: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare performance between:
    - baseline logistic regression (no grid search)
    - tuned logistic regression (GridSearchCV)

    Returns
    -------
    comparison : dict
        {
            "baseline": {"accuracy": ..., "f1": ...},
            "tuned":    {"accuracy": ..., "f1": ...},
            "best_params": {...}
        }
    """

    # Standard preprocessing
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_train_test_data(df_model)

    # ---- Baseline model ----
    baseline_model = create_base_log_reg_model()
    baseline_model.fit(X_train, y_train)

    baseline_preds = baseline_model.predict(X_test)
    baseline_metrics = {
        "accuracy": accuracy_score(y_test, baseline_preds),
        "f1": f1_score(y_test, baseline_preds)
    }

    # ---- Tuned model ----
    tuned_model, tuning_results = run_log_reg_tuning(df_model)

    tuned_preds = tuned_model.predict(X_test)
    tuned_metrics = {
        "accuracy": accuracy_score(y_test, tuned_preds),
        "f1": f1_score(y_test, tuned_preds)
    }

    return {
        "baseline": baseline_metrics,
        "tuned": tuned_metrics,
        "best_params": tuning_results["best_params"],
        "cv_best_score": tuning_results["best_score"],
    }
