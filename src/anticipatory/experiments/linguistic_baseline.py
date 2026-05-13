"""Experiment 4: Linguistic statistics baselines (bigram + n-gram)."""

import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

from anticipatory.data.features import CharacterFeatureSet
from anticipatory.data.vocabulary import Vocabulary
from anticipatory.analysis.metrics import compute_all_metrics


def run_bigram_baseline(
    features: CharacterFeatureSet,
    n_splits: int = 10,
    seed: int = 42,
) -> dict:
    """Experiment 4a: Predict N+1 from one-hot encoding of N (bigram statistics).

    This is the pure linguistic baseline — no neural data involved.
    """
    vocab = Vocabulary()
    n_classes = vocab.n_classes

    # One-hot encode current character
    X_onehot = np.eye(n_classes, dtype=np.float32)[features.y_current]

    from anticipatory.analysis.linear_probe import run_simple_probe

    results = run_simple_probe(
        X=X_onehot,
        y_target=features.y_next,
        groups=features.sentence_ids,
        n_splits=n_splits,
        seed=seed,
    )
    results["method"] = "bigram_onehot"
    return results


def run_context_baseline(
    features: CharacterFeatureSet,
    context_size: int = 5,
    n_splits: int = 10,
    seed: int = 42,
) -> dict:
    """Predict N+1 from context window of previous character labels (n-gram proxy).

    For positions with fewer than context_size predecessors, zero-pad.
    """
    vocab = Vocabulary()
    n_classes = vocab.n_classes

    # Build context features: [y_{N}, y_{N-1}, ..., y_{N-context_size+1}] one-hot concatenated
    N = len(features)
    X_ctx = np.zeros((N, n_classes * context_size), dtype=np.float32)

    # Group by sentence for context building
    for sent_id in np.unique(features.sentence_ids):
        mask = features.sentence_ids == sent_id
        indices = np.where(mask)[0]
        chars = features.y_current[indices]

        for local_i, global_i in enumerate(indices):
            for offset in range(context_size):
                pos = local_i - offset
                if pos >= 0:
                    char_idx = chars[pos]
                    X_ctx[global_i, offset * n_classes + char_idx] = 1.0

    from anticipatory.analysis.linear_probe import run_simple_probe

    results = run_simple_probe(
        X=X_ctx,
        y_target=features.y_next,
        groups=features.sentence_ids,
        n_splits=n_splits,
        seed=seed,
    )
    results["method"] = f"context_{context_size}gram"
    return results


def run_majority_vote_baseline(
    features: CharacterFeatureSet,
) -> dict:
    """Predict N+1 as the most frequent successor for each N (empirical bigram)."""
    # Build bigram frequency table
    bigram_counts = Counter()
    for curr, nxt in zip(features.y_current, features.y_next):
        bigram_counts[(curr, nxt)] += 1

    # For each current char, find the most frequent next char
    vocab = Vocabulary()
    most_frequent_next = {}
    for curr in range(vocab.n_classes):
        successors = {nxt: bigram_counts[(curr, nxt)] for nxt in range(vocab.n_classes)}
        if sum(successors.values()) > 0:
            most_frequent_next[curr] = max(successors, key=successors.get)
        else:
            most_frequent_next[curr] = 0

    # Predict
    y_pred = np.array([most_frequent_next.get(c, 0) for c in features.y_current])
    results = compute_all_metrics(features.y_next, y_pred)
    results["method"] = "majority_vote_bigram"
    return results


def run_all_linguistic_baselines(
    features: CharacterFeatureSet,
    n_splits: int = 10,
    seed: int = 42,
) -> dict:
    """Run all linguistic baselines and return comparison."""
    print("Running linguistic baselines...")

    print("  Bigram (one-hot) baseline...")
    bigram = run_bigram_baseline(features, n_splits, seed)
    print(f"    balanced accuracy: {bigram['balanced_accuracy']:.4f}")

    print("  5-gram context baseline...")
    context5 = run_context_baseline(features, context_size=5, n_splits=n_splits, seed=seed)
    print(f"    balanced accuracy: {context5['balanced_accuracy']:.4f}")

    print("  Majority vote baseline...")
    majority = run_majority_vote_baseline(features)
    print(f"    balanced accuracy: {majority['balanced_accuracy']:.4f}")

    return {
        "bigram": bigram,
        "context_5gram": context5,
        "majority_vote": majority,
    }
