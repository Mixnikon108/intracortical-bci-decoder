"""Experiment 6: Isolated single-letter control.

If the probe finds above-chance accuracy on isolated letters (no sequential context),
there is a methodological bug — single letters have no real "next character."
"""

from typing import List

import numpy as np

from anticipatory.data.loader import SingleLetterData
from anticipatory.data.preprocessing import gaussian_smooth
from anticipatory.data.vocabulary import Vocabulary, CHARS
from anticipatory.analysis.linear_probe import run_simple_probe
from anticipatory.analysis.metrics import compute_all_metrics


def run_isolated_letter_control(
    single_letter_data: List[SingleLetterData],
    sigma: float = 4.0,
    n_splits: int = 10,
    seed: int = 42,
) -> dict:
    """Run probe on isolated single-letter trials with randomized "next" labels.

    Expected result: balanced accuracy ~ 1/31 ≈ 3.2% (chance).
    Significant above-chance → methodological bug.
    """
    vocab = Vocabulary()
    rng = np.random.default_rng(seed)

    all_X = []
    all_y_current = []
    all_groups = []  # session as grouping unit

    for sess_idx, sl_data in enumerate(single_letter_data):
        # Gather all trials and compute session normalization stats
        session_trials = []
        session_labels = []

        for char in CHARS:
            if char not in sl_data.cubes:
                continue
            cube = sl_data.cubes[char]  # [n_trials, n_timesteps, 192]
            char_idx = vocab.char_to_idx[char]

            for trial_i in range(cube.shape[0]):
                trial = cube[trial_i, :, :].astype(np.float32)
                trial = gaussian_smooth(trial, sigma)
                session_trials.append(trial)
                session_labels.append(char_idx)

        if not session_trials:
            continue

        # Compute session-level normalization
        concat = np.concatenate(session_trials, axis=0)
        mean, std = concat.mean(axis=0), concat.std(axis=0)
        std[std < 1e-6] = 1.0

        # Extract features: mean activity over central portion (bins 50-150 of ~201)
        for trial, label in zip(session_trials, session_labels):
            trial_norm = (trial - mean) / std
            n_bins = trial_norm.shape[0]
            center_start = max(0, n_bins // 4)
            center_end = min(n_bins, 3 * n_bins // 4)
            feature = trial_norm[center_start:center_end, :].mean(axis=0)

            all_X.append(feature)
            all_y_current.append(label)
            all_groups.append(sess_idx)

    X = np.array(all_X, dtype=np.float32)
    y_current = np.array(all_y_current)
    groups = np.array(all_groups)

    # Assign random "next character" labels (uniform)
    y_fake_next = rng.integers(0, vocab.n_classes, size=len(y_current))

    print(f"Isolated letter control: {len(X)} trials, {len(np.unique(y_current))} classes")
    print(f"Chance level: {1.0 / vocab.n_classes:.4f}")

    # Run probe
    results = run_simple_probe(X, y_fake_next, groups, n_splits=n_splits, seed=seed)
    results["chance_level"] = 1.0 / vocab.n_classes
    results["n_trials"] = len(X)
    results["n_sessions"] = len(single_letter_data)

    # Also run current-character probe (should be high — sanity check)
    current_results = run_simple_probe(X, y_current, groups, n_splits=n_splits, seed=seed)
    results["current_char_accuracy"] = current_results["balanced_accuracy"]

    print(f"  Random-next balanced accuracy: {results['balanced_accuracy']:.4f}")
    print(f"  Current-char balanced accuracy: {current_results['balanced_accuracy']:.4f}")

    return results
