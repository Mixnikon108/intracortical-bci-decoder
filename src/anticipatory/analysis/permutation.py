import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from typing import Callable

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="The least populated class in y")


def shuffle_within_classes(
    y_target: np.ndarray,
    y_condition: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle target labels within each condition class.

    Preserves the marginal distribution of the conditioning variable while
    destroying the neural-to-target association.
    """
    y_perm = y_target.copy()
    for c in np.unique(y_condition):
        mask = y_condition == c
        y_perm[mask] = rng.permutation(y_target[mask])
    return y_perm


def permutation_test(
    score_fn: Callable,
    y_target: np.ndarray,
    y_condition: np.ndarray,
    n_permutations: int = 1000,
    n_jobs: int = -1,
    seed: int = 42,
) -> dict:
    """Run permutation test by shuffling target labels within condition classes.

    Args:
        score_fn: callable(y_shuffled) -> float. Should run the full
                  CV pipeline with the shuffled labels and return the metric.
        y_target: [N] target labels to shuffle
        y_condition: [N] conditioning labels (shuffle within these classes)
        n_permutations: number of permutation iterations
        n_jobs: parallel jobs (-1 = all cores)
        seed: random seed

    Returns:
        dict with 'observed', 'null_distribution', 'p_value', 'z_score'
    """
    # Observed score (with real labels)
    observed = score_fn(y_target)

    # Generate permuted label sets
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=n_permutations)

    def _single_permutation(perm_seed):
        perm_rng = np.random.default_rng(perm_seed)
        y_perm = shuffle_within_classes(y_target, y_condition, perm_rng)
        return score_fn(y_perm)

    print(f"    Permutation test: {n_permutations} iterations, n_jobs={n_jobs}", flush=True)
    t0 = time.time()

    null_dist = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_single_permutation)(s) for s in seeds
    )
    null_dist = np.array(null_dist)

    elapsed = time.time() - t0
    print(f"    Permutations completed in {elapsed/60:.1f} min", flush=True)

    # p-value: fraction of permutations >= observed
    p_value = (np.sum(null_dist >= observed) + 1) / (n_permutations + 1)

    # z-score
    null_mean = null_dist.mean()
    null_std = null_dist.std()
    z_score = (observed - null_mean) / null_std if null_std > 0 else 0.0

    return {
        "observed": observed,
        "null_distribution": null_dist,
        "p_value": p_value,
        "z_score": z_score,
        "null_mean": null_mean,
        "null_std": null_std,
    }
