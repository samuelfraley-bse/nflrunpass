"""
Preprocessing utilities for the NFL run/pass model.

This module:
- Builds the feature matrix (X, y) from a prepared dataframe
- Splits into train/test sets
- Applies standardization (StandardScaler) to numeric features

It mirrors the preprocessing steps in the Kaggle notebook:
- feature engineering
- train/test split (with stratification)
- scaling with StandardScaler
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import CONFIG
from .features import build_feature_matrix


# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------
def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """
    Simple imputation for missing values in features.

    - For numeric columns: fill NaN with the column median.
    - For any remaining columns: fill remaining NaN with 0.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with possible NaNs.

    Returns
    -------
    X_imputed : pd.DataFrame
        Feature matrix with no NaNs.
    """
    X = X.copy()

    # Fill numeric columns with their median
    numeric_cols = X.select_dtypes(include=["number"]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    # For any remaining NaNs (e.g. in non-numeric cols, if any), fill with 0
    X = X.fillna(0)

    return X


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    stratify: Optional[bool] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split X and y into train and test sets using scikit-learn's
    train_test_split, with defaults coming from CONFIG.train_test.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float, optional
        Proportion of the dataset to include in the test split.
        If None, uses CONFIG.train_test.test_size.
    random_state : int, optional
        Controls the shuffling applied before the split.
        If None, uses CONFIG.train_test.random_state.
    stratify : bool, optional
        If True, stratify by y. If None, uses CONFIG.train_test.stratify.

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
    """
    tt_cfg = CONFIG.train_test

    if test_size is None:
        test_size = tt_cfg.test_size
    if random_state is None:
        random_state = tt_cfg.random_state
    if stratify is None:
        stratify = tt_cfg.stratify

    stratify_arg = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit a StandardScaler on the training data and apply it to both
    train and test sets.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Test feature matrix.

    Returns
    -------
    X_train_scaled : pd.DataFrame
        Scaled training features (same columns and index as X_train).
    X_test_scaled : pd.DataFrame
        Scaled test features (same columns and index as X_test).
    scaler : StandardScaler
        Fitted scaler instance (for saving and later inference).
    """
    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(X_train)
    X_test_arr = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_arr, index=X_train.index, columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        X_test_arr, index=X_test.index, columns=X_test.columns
    )

    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------
# High-level convenience pipeline
# ---------------------------------------------------------------------

def prepare_train_test_data(
    df_model: pd.DataFrame,
    target_col: Optional[str] = None,
    scale: bool = True,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    Optional[StandardScaler],
    List[str],
]:
    """
    High-level preprocessing function that takes the prepared dataframe
    (from data_loading.load_and_prepare_run_pass_data), and returns
    train/test splits, optionally scaled.
    """
    # 1) Build feature matrix
    X, y, feature_cols = build_feature_matrix(df_model, target_col=target_col)

    # ✅ 1.5) Handle missing values before splitting/scaling
    X = handle_missing_values(X)

    # 2) Train/test split
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # 3) Optional scaling
    scaler: Optional[StandardScaler] = None
    if scale:
        X_train, X_test, scaler = scale_features(X_train, X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_cols
