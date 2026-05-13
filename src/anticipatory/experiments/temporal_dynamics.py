"""Experiment 5: Temporal dynamics — when does anticipatory information emerge?"""

from typing import List

import numpy as np

from anticipatory.data.loader import SessionData
from anticipatory.data.features import extract_character_features
from anticipatory.experiments.core_probes import run_anticipatory_probe, run_reverse_time_probe


def run_quartile_profile(
    sessions: List[SessionData],
    sigma: float = 4.0,
    min_duration: int = 20,
    n_splits: int = 10,
    C_values: list = None,
    seed: int = 42,
) -> dict:
    """Run probes on each temporal quartile within characters.

    Tests whether anticipatory information grows toward character offset (Q4)
    or is present throughout execution (flat profile).
    """
    quartiles = ["q1", "q2", "q3", "q4"]
    results = {}

    for q in quartiles:
        print(f"\n--- Quartile {q.upper()} ---")
        features = extract_character_features(
            sessions, sigma=sigma, window=q,
            min_duration=min_duration, smoothing_type="gaussian",
        )
        ll = features.filter_transition_type(["ll"])
        print(f"  {len(ll)} letter-letter transitions")

        next_r = run_anticipatory_probe(ll, n_splits, C_values, seed)
        prev_r = run_reverse_time_probe(ll, n_splits, C_values, seed)

        results[q] = {
            "next_balanced_accuracy": next_r["balanced_accuracy"],
            "next_kappa": next_r["cohen_kappa"],
            "prev_balanced_accuracy": prev_r["balanced_accuracy"],
            "prev_kappa": prev_r["cohen_kappa"],
            "asymmetry": next_r["balanced_accuracy"] - prev_r["balanced_accuracy"],
            "n_samples": len(ll),
        }

        print(f"  N+1 acc={next_r['balanced_accuracy']:.4f}  "
              f"N-1 acc={prev_r['balanced_accuracy']:.4f}")

    return results


def run_sliding_window(
    sessions: List[SessionData],
    sigma: float = 4.0,
    window_frac: float = 0.25,
    step_frac: float = 0.10,
    min_duration: int = 20,
    n_splits: int = 10,
    C_values: list = None,
    seed: int = 42,
) -> dict:
    """Sliding window analysis for smooth temporal profile.

    Extracts features from overlapping windows of size `window_frac` of character
    duration, stepped by `step_frac`. Provides finer resolution than quartiles.
    """
    from anticipatory.data.features import CharacterFeatureSet, _extract_window, _classify_transition
    from anticipatory.data.preprocessing import gaussian_smooth
    from anticipatory.data.vocabulary import Vocabulary
    from anticipatory.analysis.linear_probe import run_residualized_probe

    vocab = Vocabulary()
    positions = np.arange(0.0, 1.0 - window_frac + step_frac / 2, step_frac)
    results = {}

    for center_pos in positions:
        start_frac = center_pos
        end_frac = center_pos + window_frac
        label = f"{start_frac:.2f}-{end_frac:.2f}"
        print(f"\n--- Window {label} ---")

        # Custom extraction for this window
        all_X, all_curr, all_next, all_sent = [], [], [], []
        global_sent_id = 0

        for sess_idx, session in enumerate(sessions):
            valid_idx = np.where(session.valid_mask)[0]
            if len(valid_idx) == 0:
                continue

            smoothed_trials = []
            for i in valid_idx:
                n_bins = session.time_bins[i]
                trial = session.neural[i, :n_bins, :].astype(np.float32)
                trial = gaussian_smooth(trial, sigma)
                smoothed_trials.append(trial)

            concat = np.concatenate(smoothed_trials, axis=0)
            mean, std = concat.mean(axis=0), concat.std(axis=0)
            std[std < 1e-6] = 1.0

            for trial_i, i in enumerate(valid_idx):
                trial = (smoothed_trials[trial_i] - mean) / std
                prompt = session.prompts[i]
                encoded = vocab.encode(prompt)
                n_chars = len(encoded)
                if n_chars < 3:
                    global_sent_id += 1
                    continue

                for j in range(1, n_chars - 1):
                    start = int(session.letter_starts[i, j])
                    dur = int(np.floor(session.letter_durations[i, j]))
                    if dur < min_duration or start + dur > trial.shape[0]:
                        continue
                    if not (vocab.is_letter(encoded[j]) and vocab.is_letter(encoded[j + 1])):
                        continue

                    w_start = start + int(start_frac * dur)
                    w_end = start + int(end_frac * dur)
                    w_end = max(w_end, w_start + 1)
                    w_end = min(w_end, trial.shape[0])

                    feat = trial[w_start:w_end, :].mean(axis=0)
                    all_X.append(feat)
                    all_curr.append(encoded[j])
                    all_next.append(encoded[j + 1])
                    all_sent.append(global_sent_id)

                global_sent_id += 1

        X = np.array(all_X, dtype=np.float32)
        y_curr = np.array(all_curr)
        y_next = np.array(all_next)
        groups = np.array(all_sent)

        if len(X) > 0:
            r = run_residualized_probe(X, y_curr, y_next, groups, n_splits, C_values, seed)
            results[label] = {
                "center": center_pos + window_frac / 2,
                "next_balanced_accuracy": r["balanced_accuracy"],
                "next_kappa": r["cohen_kappa"],
                "n_samples": len(X),
            }
            print(f"  N+1 acc={r['balanced_accuracy']:.4f} (n={len(X)})")

    return results
