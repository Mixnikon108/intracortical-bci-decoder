"""Experiment 8: Bigram frequency modulation.

Tests whether the brain anticipates more for frequent character transitions
(suggesting practice-dependent motor automatization).
"""

from collections import Counter

import numpy as np
from scipy.stats import spearmanr

from anticipatory.data.features import CharacterFeatureSet
from anticipatory.analysis.linear_probe import run_residualized_probe
from anticipatory.data.vocabulary import Vocabulary


def run_bigram_frequency_analysis(
    features: CharacterFeatureSet,
    min_bigram_count: int = 30,
    n_splits: int = 10,
    C_values: list = None,
    seed: int = 42,
) -> dict:
    """Correlate bigram frequency with per-bigram anticipatory probe accuracy.

    For each bigram type (N, N+1), compute:
    - frequency in the corpus
    - balanced accuracy of the residualized probe on that bigram's instances

    Then compute Spearman rank correlation between frequency and accuracy.
    """
    # Count bigram frequencies
    bigram_counts = Counter()
    for curr, nxt in zip(features.y_current, features.y_next):
        bigram_counts[(curr, nxt)] += 1

    # Filter to bigrams with sufficient instances
    valid_bigrams = {bg: count for bg, count in bigram_counts.items() if count >= min_bigram_count}
    print(f"Bigrams with >= {min_bigram_count} instances: {len(valid_bigrams)} / {len(bigram_counts)}")

    if len(valid_bigrams) < 5:
        print("  Too few valid bigrams for correlation analysis")
        return {"error": "insufficient_bigrams", "n_valid": len(valid_bigrams)}

    # Run residualized probe on the full dataset to get per-sample predictions
    full_results = run_residualized_probe(
        features.X, features.y_current, features.y_next,
        features.sentence_ids, n_splits, C_values, seed,
    )

    y_true = full_results["y_true"]
    y_pred = full_results["y_pred"]

    # Compute per-bigram accuracy
    bigram_accuracies = {}
    bigram_frequencies = {}

    for (curr, nxt), count in valid_bigrams.items():
        # Find indices where this bigram occurs
        mask = (features.y_current == curr) & (features.y_next == nxt)
        # Match against the cross-validated predictions
        # (predictions are in the same order as the full dataset)
        bigram_true = y_true[mask]
        bigram_pred = y_pred[mask]

        if len(bigram_true) == 0:
            continue

        accuracy = np.mean(bigram_pred == bigram_true)
        bigram_accuracies[(curr, nxt)] = accuracy
        bigram_frequencies[(curr, nxt)] = count

    # Compute correlation
    bigrams = sorted(bigram_accuracies.keys())
    freqs = np.array([bigram_frequencies[bg] for bg in bigrams])
    accs = np.array([bigram_accuracies[bg] for bg in bigrams])

    rho, p_value = spearmanr(freqs, accs)

    vocab = Vocabulary()
    bigram_details = []
    for bg in bigrams:
        bigram_details.append({
            "current": vocab.idx_to_char[bg[0]],
            "next": vocab.idx_to_char[bg[1]],
            "frequency": int(bigram_frequencies[bg]),
            "accuracy": float(bigram_accuracies[bg]),
        })
    bigram_details.sort(key=lambda x: x["frequency"], reverse=True)

    print(f"  Spearman rho: {rho:.4f}, p-value: {p_value:.4f}")
    print(f"  Top-5 most frequent bigrams:")
    for d in bigram_details[:5]:
        print(f"    '{d['current']}' -> '{d['next']}': "
              f"freq={d['frequency']}, acc={d['accuracy']:.3f}")

    return {
        "spearman_rho": rho,
        "spearman_p_value": p_value,
        "n_valid_bigrams": len(bigrams),
        "frequencies": freqs,
        "accuracies": accs,
        "bigram_details": bigram_details,
    }
