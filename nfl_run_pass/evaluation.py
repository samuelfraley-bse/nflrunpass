"""
Evaluation utilities for the NFL run/pass prediction model.

This module computes common classification metrics for both the
training and test sets, mirroring what you inspected in the Kaggle
notebook:

- accuracy
- precision
- recall
- F1 score
- ROC-AUC (if probabilities / scores are available)
- confusion matrix on the test set
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def _get_scores_for_roc_auc(model: Any, X: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Try to extract continuous scores for ROC-AUC.

    Priority:
    - If model has predict_proba, use probability of class 1.
    - Else if model has decision_function, use that.
    - Else return None.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        # assume binary classification; take probability of positive class
        return probs[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # decision_function may return shape (n_samples,) or (n_samples, 2)
        if scores.ndim == 2 and scores.shape[1] == 2:
            return scores[:, 1]
        return scores
    return None


def _compute_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute standard binary classification metrics.
    """
    metrics: Dict[str, float] = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # ROC-AUC only if scores are provided and both classes are present
    if y_scores is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
        except ValueError:
            # e.g., if only one class present in a particular fold/subset
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def evaluate_run_pass_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Evaluate a trained model on both training and test sets.

    Parameters
    ----------
    model : fitted classifier
        Typically a LogisticRegression instance (possibly from GridSearchCV).
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target values (0/1 for run/pass).
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Test target values.

    Returns
    -------
    metrics_dict : dict
        Dictionary containing:
            - 'train': dict of metrics (accuracy, precision, recall, f1, roc_auc)
            - 'test': dict of metrics (same keys)
            - 'confusion_matrix': 2x2 list (test set, labels in order [0, 1])
    """
    # -------------------------
    # Train metrics
    # -------------------------
    y_train_pred = model.predict(X_train)
    y_train_scores = _get_scores_for_roc_auc(model, X_train)

    train_metrics = _compute_classification_metrics(
        y_true=y_train,
        y_pred=y_train_pred,
        y_scores=y_train_scores,
    )

    # -------------------------
    # Test metrics
    # -------------------------
    y_test_pred = model.predict(X_test)
    y_test_scores = _get_scores_for_roc_auc(model, X_test)

    test_metrics = _compute_classification_metrics(
        y_true=y_test,
        y_pred=y_test_pred,
        y_scores=y_test_scores,
    )

    # Confusion matrix on test set
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    cm_list = cm.tolist()

    metrics_dict: Dict[str, Any] = {
        "train": train_metrics,
        "test": test_metrics,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm_list,
        },
    }

    return metrics_dict
