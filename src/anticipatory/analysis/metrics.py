import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all evaluation metrics for a classification probe."""
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mutual_info": mutual_info_score(y_true, y_pred),
        "adjusted_mutual_info": adjusted_mutual_info_score(y_true, y_pred),
        "n_samples": len(y_true),
        "n_classes_true": len(np.unique(y_true)),
        "n_classes_pred": len(np.unique(y_pred)),
    }


def partial_r_squared(
    y_true: np.ndarray,
    y_pred_full: np.ndarray,
    y_pred_reduced: np.ndarray,
) -> float:
    """Partial R^2: improvement in prediction from the full model over the reduced model.

    Computes the fraction of residual variance from the reduced model that the
    full model explains:
        partial_R2 = 1 - SS_res_full / SS_res_reduced

    For classification: uses indicator-based sum of squares (one-hot encoding).
    """
    n_classes = max(y_true.max(), y_pred_full.max(), y_pred_reduced.max()) + 1

    # One-hot encode
    y_oh = np.eye(n_classes)[y_true]
    pred_full_oh = np.eye(n_classes)[y_pred_full]
    pred_reduced_oh = np.eye(n_classes)[y_pred_reduced]

    ss_res_full = np.sum((y_oh - pred_full_oh) ** 2)
    ss_res_reduced = np.sum((y_oh - pred_reduced_oh) ** 2)

    if ss_res_reduced == 0:
        return 0.0
    return 1.0 - ss_res_full / ss_res_reduced
