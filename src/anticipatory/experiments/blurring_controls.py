"""Experiment 3: Temporal blurring controls — sigma sweep and causal filter."""

import warnings
from typing import List

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="The least populated class in y")

from anticipatory.data.loader import SessionData
from anticipatory.data.features import extract_character_features
from anticipatory.experiments.core_probes import run_anticipatory_probe, run_reverse_time_probe


def run_sigma_sweep(
    sessions: List[SessionData],
    sigma_values: list = None,
    window: str = "q1",
    min_duration: int = 20,
    n_splits: int = 10,
    C_values: list = None,
    seed: int = 42,
) -> dict:
    """Experiment 3a: Run probes at multiple smoothing levels.

    Tests whether the anticipatory signal survives at sigma=0 (no smoothing).
    """
    if sigma_values is None:
        sigma_values = [0, 1, 2, 4, 8]

    results = {}
    for sigma in sigma_values:
        print(f"\n--- Sigma = {sigma} ---")
        features = extract_character_features(
            sessions, sigma=sigma, window=window,
            min_duration=min_duration, smoothing_type="gaussian",
        )
        ll = features.filter_transition_type(["ll"])
        print(f"  {len(ll)} letter-letter transitions")

        next_r = run_anticipatory_probe(ll, n_splits, C_values, seed)
        prev_r = run_reverse_time_probe(ll, n_splits, C_values, seed)

        results[sigma] = {
            "next_balanced_accuracy": next_r["balanced_accuracy"],
            "next_kappa": next_r["cohen_kappa"],
            "prev_balanced_accuracy": prev_r["balanced_accuracy"],
            "prev_kappa": prev_r["cohen_kappa"],
            "asymmetry": next_r["balanced_accuracy"] - prev_r["balanced_accuracy"],
            "n_samples": len(ll),
            "full_next": next_r,
            "full_prev": prev_r,
        }

        print(f"  N+1 acc={next_r['balanced_accuracy']:.4f}  "
              f"N-1 acc={prev_r['balanced_accuracy']:.4f}  "
              f"asymmetry={next_r['balanced_accuracy'] - prev_r['balanced_accuracy']:.4f}")

    return results


def run_causal_filter_comparison(
    sessions: List[SessionData],
    sigma_values: list = None,
    window: str = "q1",
    min_duration: int = 20,
    n_splits: int = 10,
    C_values: list = None,
    seed: int = 42,
) -> dict:
    """Experiment 3b: Compare Gaussian vs causal smoothing.

    If N+1 signal disappears under causal filter but persists under Gaussian,
    it was entirely due to the symmetric forward tail leaking future activity.
    """
    if sigma_values is None:
        sigma_values = [2, 4, 8]

    results = {}
    for sigma in sigma_values:
        print(f"\n--- Sigma = {sigma} ---")
        row = {}

        for stype in ["gaussian", "causal"]:
            features = extract_character_features(
                sessions, sigma=sigma, window=window,
                min_duration=min_duration, smoothing_type=stype,
            )
            ll = features.filter_transition_type(["ll"])

            next_r = run_anticipatory_probe(ll, n_splits, C_values, seed)

            row[stype] = {
                "next_balanced_accuracy": next_r["balanced_accuracy"],
                "next_kappa": next_r["cohen_kappa"],
                "n_samples": len(ll),
            }
            print(f"  {stype:8s}: N+1 acc={next_r['balanced_accuracy']:.4f}")

        row["causal_drop"] = (
            row["gaussian"]["next_balanced_accuracy"]
            - row["causal"]["next_balanced_accuracy"]
        )
        results[sigma] = row

    return results
