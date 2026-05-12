"""
This module contains functions to compute conformal prediction sets.
"""

import numpy as np
import pandas as pd


def compute_nc_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    For each calibration example, return 1 - p_true.

    probs: array of shape (n_samples, n_classes) with predicted probabilities
        between 0 and 1.
    labels: array of shape (n_samples,) with values in {0, 1, ..., n_classes-1}
    """
    p_true = probs[np.arange(len(labels)), labels]
    return 1.0 - p_true


def find_threshold(nonconformity: np.ndarray, alpha: float) -> float:
    """
    Return the (1-alpha)-quantile (higher interpolation) of the nc scores.
    """
    return np.quantile(nonconformity, 1 - alpha, method="higher")


def predict_conformal_sets(model, X: pd.DataFrame, q_hat: float) -> list[set[int]]:
    """
    For each row in X, compute the conformal prediction set.
    Returns a list of sets.
    """
    probs = model.predict_proba(X)  # shape (n, 2)
    nonconf_matrix = 1.0 - probs  # shape (n, 2): nc score for label = 0, 1
    # include c whenver nonconf_matrix[i,c] <= q_hat
    return [set(np.where(nc_row <= q_hat)[0]) for nc_row in nonconf_matrix]


def evaluate_sets(pred_sets: list, y_true: pd.Series) -> dict:
    """
    Compute empirical coverage and average set size.
    """
    hits = [y_true.iloc[i] in pred_sets[i] for i in range(len(y_true))]
    coverage = np.mean(hits)
    avg_size = np.mean([len(s) for s in pred_sets])
    return {"coverage": coverage, "avg_size": avg_size}


def build_cp_groups(pred_sets, y_test, test_idx, X_test_full):
    """
    Build a DataFrame of conformal prediction results with subgroup info.
    - pred_sets: list of prediction sets for each test sample (in order of test_idx).
    - y_test: Series or array of true labels for test samples.
    - test_idx: index of the test samples corresponding to pred_sets.
    - X_test_full: DataFrame of test features **including protected attributes**.
    """
    cp_df = pd.DataFrame(index=test_idx.copy())
    # Store prediction sets (convert each to a set of ints for consistency)
    cp_df['pred_set']   = pd.Series(pred_sets, index=test_idx).apply(lambda s: {int(x) for x in s})
    cp_df['true_label'] = y_test.reindex(test_idx)
    # Bring in protected attributes from the full test set
    cp_df['frau1']    = X_test_full.loc[test_idx, 'frau1']          # 1 = female, 0 = male
    # Derive 'nongerman' flag ('maxdeutsch1' indicates German (1) vs non-German (0))
    cp_df['nongerman'] = np.where(X_test_full.loc[test_idx, 'maxdeutsch1'] == 0, 1, 0)
    # Handle missing nationality information if applicable
    if 'maxdeutsch.Missing.' in X_test_full.columns:
        missing_mask = (X_test_full.loc[test_idx, 'maxdeutsch.Missing.'] == 1)
        cp_df.loc[missing_mask, 'nongerman'] = np.nan
    # Derive intersectional subgroup flags
    cp_df['nongerman_male']   = np.where((cp_df['nongerman'] == 1) & (cp_df['frau1'] == 0), 1, 0)
    cp_df['nongerman_female'] = np.where((cp_df['nongerman'] == 1) & (cp_df['frau1'] == 1), 1, 0)
    # (Optional) drop samples with missing subgroup info
    cp_df = cp_df.dropna(subset=['nongerman'])
    return cp_df