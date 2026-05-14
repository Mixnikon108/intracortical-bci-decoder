import warnings
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from typing import Optional

from anticipatory.analysis.metrics import compute_all_metrics

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="The least populated class in y")


def residualize(
    X: np.ndarray,
    clf: LogisticRegression,
) -> np.ndarray:
    """Project out the subspace spanned by the classifier weight vectors.

    Removes the linear component of X that encodes the conditioning variable,
    leaving the residual neural activity for probing other variables.
    """
    W = clf.coef_  # [n_classes, n_features]
    # SVD of the weight matrix to get orthonormal basis of its row space
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    # Keep components with non-negligible singular values
    rank = np.sum(S > 1e-10 * S[0])
    V = Vt[:rank, :].T  # [n_features, rank] orthonormal basis
    # Project out: X_residual = X - X @ V @ V^T
    projection = X @ V @ V.T
    return X - projection


def run_residualized_probe(
    X: np.ndarray,
    y_current: np.ndarray,
    y_target: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 10,
    C_values: list = None,
    seed: int = 42,
) -> dict:
    """Two-stage residualized probe with sentence-level grouped cross-validation.

    Stage A: Decode current character from neural features, project out that subspace.
    Stage B: Decode target character (N+1 or N-1) from residualized features.

    Args:
        X: [N, 192] neural feature vectors
        y_current: [N] current character labels
        y_target: [N] target character labels (next or previous)
        groups: [N] sentence IDs for grouped CV
        n_splits: number of CV folds
        C_values: regularization values for inner CV
        seed: random seed

    Returns:
        dict with metrics, per-fold results, and predictions
    """
    if C_values is None:
        C_values = [0.01, 0.1, 1.0, 10.0]

    # Adjust n_splits if not enough groups
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    all_y_true = []
    all_y_pred = []
    all_y_pred_raw = []  # predictions WITHOUT residualization (for comparison)
    fold_results = []

    t0 = time.time()
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y_target, groups)):
        fold_start = time.time()
        X_train, X_test = X[train_idx], X[test_idx]
        y_cur_train, y_cur_test = y_current[train_idx], y_current[test_idx]
        y_tgt_train, y_tgt_test = y_target[train_idx], y_target[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Stage A: Decode current character
        clf_current = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            C=1.0,
        )
        clf_current.fit(X_train_s, y_cur_train)
        current_acc = clf_current.score(X_test_s, y_cur_test)

        # Residualize
        X_train_res = residualize(X_train_s, clf_current)
        X_test_res = residualize(X_test_s, clf_current)

        # Stage B: Decode target from residuals — tune C via inner CV
        best_C = _tune_regularization(
            X_train_res, y_tgt_train, groups[train_idx], C_values, seed
        )

        clf_target = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            C=best_C,
        )
        clf_target.fit(X_train_res, y_tgt_train)
        y_pred = clf_target.predict(X_test_res)

        # Also predict WITHOUT residualization (raw probe)
        clf_raw = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            C=best_C,
        )
        clf_raw.fit(X_train_s, y_tgt_train)
        y_pred_raw = clf_raw.predict(X_test_s)

        all_y_true.extend(y_tgt_test)
        all_y_pred.extend(y_pred)
        all_y_pred_raw.extend(y_pred_raw)

        fold_metrics = compute_all_metrics(y_tgt_test, y_pred)
        fold_metrics["current_char_accuracy"] = current_acc
        fold_metrics["best_C"] = best_C
        fold_metrics["fold"] = fold_idx
        fold_results.append(fold_metrics)

        elapsed = time.time() - fold_start
        total = time.time() - t0
        eta = (total / (fold_idx + 1)) * (n_splits - fold_idx - 1)
        print(f"    Fold {fold_idx+1}/{n_splits}: "
              f"acc={fold_metrics['balanced_accuracy']:.4f}  "
              f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    # Aggregate metrics
    y_true_all = np.array(all_y_true)
    y_pred_all = np.array(all_y_pred)
    y_pred_raw_all = np.array(all_y_pred_raw)

    results = compute_all_metrics(y_true_all, y_pred_all)
    raw_metrics = compute_all_metrics(y_true_all, y_pred_raw_all)

    results["raw_balanced_accuracy"] = raw_metrics["balanced_accuracy"]
    results["raw_cohen_kappa"] = raw_metrics["cohen_kappa"]
    results["per_fold"] = fold_results
    results["y_true"] = y_true_all
    results["y_pred"] = y_pred_all
    results["y_pred_raw"] = y_pred_raw_all
    results["mean_current_accuracy"] = np.mean(
        [f["current_char_accuracy"] for f in fold_results]
    )

    return results


def run_simple_probe(
    X: np.ndarray,
    y_target: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 10,
    C_values: list = None,
    seed: int = 42,
) -> dict:
    """Simple (non-residualized) probe for baselines and controls."""
    if C_values is None:
        C_values = [0.01, 0.1, 1.0, 10.0]

    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    all_y_true = []
    all_y_pred = []

    for train_idx, test_idx in cv.split(X, y_target, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_target[train_idx], y_target[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        best_C = _tune_regularization(
            X_train_s, y_train, groups[train_idx], C_values, seed
        )

        clf = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            C=best_C,
        )
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    y_true_all = np.array(all_y_true)
    y_pred_all = np.array(all_y_pred)
    results = compute_all_metrics(y_true_all, y_pred_all)
    results["y_true"] = y_true_all
    results["y_pred"] = y_pred_all
    return results


def _tune_regularization(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C_values: list,
    seed: int,
) -> float:
    """Select best regularization C via inner 3-fold CV."""
    n_groups = len(np.unique(groups))
    inner_splits = min(3, n_groups)

    if inner_splits < 2 or len(np.unique(y)) < 2:
        return 1.0

    inner_cv = StratifiedGroupKFold(
        n_splits=inner_splits, shuffle=True, random_state=seed
    )

    best_C = C_values[0]
    best_score = -1.0

    for C in C_values:
        scores = []
        try:
            for tr, te in inner_cv.split(X, y, groups):
                clf = LogisticRegression(
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                    C=C,
                )
                clf.fit(X[tr], y[tr])
                scores.append(clf.score(X[te], y[te]))
        except ValueError:
            continue

        mean_score = np.mean(scores) if scores else -1.0
        if mean_score > best_score:
            best_score = mean_score
            best_C = C

    return best_C
