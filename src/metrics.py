"""AUROC helpers for multi-label targets, tolerant of classes with one class present.

Every dataset here is multi-label and long-tailed, so a subset -- or a rare disease in an
otherwise large split -- routinely contains only negatives for some class, where
`roc_auc_score` raises. The `nan_*` variants skip those classes instead of failing.

The `*_many` variants take a stack of predictions from several models
(n_models, n_samples, n_classes) and return one score per model, which is the shape the
leave-many-out comparisons need.
"""

import numpy as np
import sklearn.metrics as skm # type: ignore
from typing import Dict, Optional, List, Union


def nan_macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the macro AUROC, ignoring labels with only one class present.

    Args:
        y_true: Ground truth binary labels, shape (n_samples, n_classes).
        y_score: Predicted scores, shape (n_samples, n_classes).

    Returns:
        Macro AUROC score, ignoring labels with only one class present.
    """
    n_classes = y_true.shape[1]
    score = np.nanmean(
        [
            skm.roc_auc_score(y_true[:, i], y_score[:, i])
            for i in range(n_classes)
            if np.unique(y_true[:, i]).size
            > 1  # make sure at least one positive and one negative sample exists
        ]
    )
    return score


def auroc_per_class(
    y_true: np.ndarray, y_score: np.ndarray, class_names: Optional[List[str]] = None
) -> Union[np.ndarray, Dict[str, float]]:
    """Compute the AUROC for each class.

    Args:
        y_true: Ground truth binary labels, shape (n_samples, n_classes).
        y_score: Predicted scores, shape (n_samples, n_classes).

    Returns:
        Array of AUROC scores for each class.
    """
    n_classes = y_true.shape[1]
    scores = np.array(
        [skm.roc_auc_score(y_true[:, i], y_score[:, i]) for i in range(n_classes)]
    )
    if class_names is not None:
        return {class_names[i]: scores[i] for i in range(n_classes)}
    else:
        return scores


def nan_macro_auroc_many(y_true: np.ndarray, y_scores: np.ndarray) -> np.ndarray:
    """
    Helper function to compute macro AUROC for multiple sets of predictions.
    Args:
        y_true: Ground truth binary labels, shape (n_samples, n_classes).
        y_scores: Predicted scores, shape (n_models, n_samples, n_classes).
    """
    n_models = y_scores.shape[0]
    scores = np.array([nan_macro_auroc(y_true, y_scores[i]) for i in range(n_models)])
    assert scores.shape == (n_models,)
    return scores


def auroc_per_class_many(
    y_true: np.ndarray, y_scores: np.ndarray, class_names: Optional[List[str]] = None
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Helper function to compute per-class AUROC for multiple sets of predictions.
    Args:
        y_true: Ground truth binary labels, shape (n_samples, n_classes).
        y_scores: Predicted scores, shape (n_models, n_samples, n_classes).
    """
    n_models = y_scores.shape[0]
    n_classes = y_true.shape[1]
    scores = np.array(
        [
            [skm.roc_auc_score(y_true[:, i], y_scores[m, :, i]) for i in range(n_classes)]
            for m in range(n_models)
        ]
    )
    assert scores.shape == (n_models, n_classes)
    if class_names is not None:
        return {class_names[i]: scores[:, i] for i in range(n_classes)}
    else:
        return scores
