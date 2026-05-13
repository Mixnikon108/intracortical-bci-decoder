"""Experiments 1 & 2: Residualized next-character (N+1) and reverse-time (N-1) probes."""

import numpy as np
from typing import List

from anticipatory.data.features import CharacterFeatureSet
from anticipatory.analysis.linear_probe import run_residualized_probe
from anticipatory.analysis.permutation import permutation_test
from anticipatory.analysis.metrics import compute_all_metrics


def run_anticipatory_probe(
    features: CharacterFeatureSet,
    n_splits: int = 10,
    C_values: list = None,
    seed: int = 42,
) -> dict:
    """Experiment 1: residualized probe predicting next character (N+1)."""
    return run_residualized_probe(
        X=features.X,
        y_current=features.y_current,
        y_target=features.y_next,
        groups=features.sentence_ids,
        n_splits=n_splits,
        C_values=C_values,
        seed=seed,
    )


def run_reverse_time_probe(
    features: CharacterFeatureSet,
    n_splits: int = 10,
    C_values: list = None,
    seed: int = 42,
) -> dict:
    """Experiment 2: residualized probe predicting previous character (N-1)."""
    return run_residualized_probe(
        X=features.X,
        y_current=features.y_current,
        y_target=features.y_prev,
        groups=features.sentence_ids,
        n_splits=n_splits,
        C_values=C_values,
        seed=seed,
    )


def run_core_probes_with_permutation(
    features: CharacterFeatureSet,
    n_splits: int = 10,
    n_permutations: int = 1000,
    C_values: list = None,
    n_jobs: int = -1,
    seed: int = 42,
) -> dict:
    """Run both N+1 and N-1 probes with permutation testing.

    Returns:
        dict with keys 'next', 'prev', 'next_permutation', 'prev_permutation'
    """
    print("Running N+1 (anticipatory) probe...")
    next_results = run_anticipatory_probe(features, n_splits, C_values, seed)
    print(f"  N+1 balanced accuracy: {next_results['balanced_accuracy']:.4f}")
    print(f"  N+1 Cohen's kappa:     {next_results['cohen_kappa']:.4f}")

    print("Running N-1 (reverse-time) probe...")
    prev_results = run_reverse_time_probe(features, n_splits, C_values, seed)
    print(f"  N-1 balanced accuracy: {prev_results['balanced_accuracy']:.4f}")
    print(f"  N-1 Cohen's kappa:     {prev_results['cohen_kappa']:.4f}")

    results = {
        "next": next_results,
        "prev": prev_results,
        "asymmetry": next_results["balanced_accuracy"] - prev_results["balanced_accuracy"],
    }

    # Permutation test for N+1
    if n_permutations > 0:
        print(f"Running permutation test for N+1 ({n_permutations} shuffles)...")

        def score_fn_next(y_shuffled):
            r = run_residualized_probe(
                features.X, features.y_current, y_shuffled,
                features.sentence_ids, n_splits, C_values, seed,
            )
            return r["balanced_accuracy"]

        perm_next = permutation_test(
            score_fn_next, features.y_next, features.y_current,
            n_permutations, n_jobs, seed,
        )
        results["next_permutation"] = perm_next
        print(f"  N+1 p-value: {perm_next['p_value']:.4f}")

        print(f"Running permutation test for N-1 ({n_permutations} shuffles)...")

        def score_fn_prev(y_shuffled):
            r = run_residualized_probe(
                features.X, features.y_current, y_shuffled,
                features.sentence_ids, n_splits, C_values, seed,
            )
            return r["balanced_accuracy"]

        perm_prev = permutation_test(
            score_fn_prev, features.y_prev, features.y_current,
            n_permutations, n_jobs, seed,
        )
        results["prev_permutation"] = perm_prev
        print(f"  N-1 p-value: {perm_prev['p_value']:.4f}")

    return results
