"""Visualize the temporal footprint of a character's neural representation.

Trains a current-character classifier once, then sweeps predict_proba across
a sliding window aligned to each character's onset. Shows the full lifecycle:
anticipatory buildup before onset, peak during execution, perseverative decay.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from anticipatory.data.loader import load_all_sessions
from anticipatory.data.preprocessing import gaussian_smooth
from anticipatory.data.vocabulary import Vocabulary, IDX_TO_CHAR

import yaml


def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    data_root = Path(__file__).parent.parent / config["data"]["root"]
    sigma = 0  # No smoothing — raw signal
    min_duration = config["preprocessing"]["min_char_duration"]
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    vocab = Vocabulary()

    # --- Parameters ---
    window_ms = 50       # sliding window size in ms
    step_ms = 10         # step size in ms
    window_bins = window_ms // 10
    step_bins = step_ms // 10
    margin_bins = 80     # how far before/after target onset to look (800ms)

    print("Loading sessions...")
    sessions = load_all_sessions(data_root, config["data"]["hmm_source"])

    # --- Extract per-character time courses + metadata ---
    print("Extracting character data...")
    chars = []  # list of dicts with neural trace, labels, etc.

    for sess_idx, session in enumerate(sessions):
        valid_idx = np.where(session.valid_mask)[0]
        if len(valid_idx) == 0:
            continue

        # Smooth all trials
        smoothed = []
        for i in valid_idx:
            n_bins = session.time_bins[i]
            trial = session.neural[i, :n_bins, :].astype(np.float32)
            trial = gaussian_smooth(trial, sigma)
            smoothed.append(trial)

        # Session normalization stats
        concat = np.concatenate(smoothed, axis=0)
        sess_mean, sess_std = concat.mean(axis=0), concat.std(axis=0)
        sess_std[sess_std < 1e-6] = 1.0

        for trial_i, i in enumerate(valid_idx):
            trial = (smoothed[trial_i] - sess_mean) / sess_std
            prompt = session.prompts[i]
            encoded = vocab.encode(prompt)
            n_chars = len(encoded)
            if n_chars < 3:
                continue

            for j in range(1, n_chars - 1):
                start = int(session.letter_starts[i, j])
                dur = int(np.floor(session.letter_durations[i, j]))
                if dur < min_duration or start + dur > trial.shape[0]:
                    continue
                if not (vocab.is_letter(encoded[j]) and vocab.is_letter(encoded[j + 1])):
                    continue

                # Q1 feature for training
                q1_end = start + dur // 4
                q1_feat = trial[start:max(q1_end, start + 1), :].mean(axis=0)

                chars.append({
                    "trial": trial,
                    "start": start,
                    "dur": dur,
                    "y_current": encoded[j],
                    "y_next": encoded[j + 1],
                    "q1_feat": q1_feat,
                    "sess": sess_idx,
                })

    print(f"  {len(chars)} letter-letter characters extracted")

    # --- Train/test split ---
    X_q1 = np.array([c["q1_feat"] for c in chars])
    y_current = np.array([c["y_current"] for c in chars])
    y_next = np.array([c["y_next"] for c in chars])

    indices = np.arange(len(chars))
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)

    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    # --- Train classifier on CURRENT character identity (once) ---
    print("Training current-character classifier...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_q1[train_idx])

    clf = LogisticRegression(
        solver="lbfgs", class_weight="balanced",
        max_iter=2000, C=1.0, random_state=42,
    )
    clf.fit(X_train, y_current[train_idx])
    print(f"  Classes: {len(clf.classes_)}")
    train_acc = clf.score(X_train, y_current[train_idx])
    test_acc = clf.score(scaler.transform(X_q1[test_idx]), y_current[test_idx])
    print(f"  Train acc: {train_acc:.3f}, Test acc: {test_acc:.3f}")

    # --- Sweep predict_proba across time for test characters ---
    print("Computing temporal signal trace...")
    time_positions = np.arange(-margin_bins, margin_bins + 1, step_bins)
    n_positions = len(time_positions)

    # Store P(correct current char) at each time position
    all_probs = []          # [n_test, n_positions]
    all_current_labels = []
    valid_count = 0

    for idx in test_idx:
        c = chars[idx]
        trial = c["trial"]
        onset = c["start"]
        correct_current = c["y_current"]

        probs_row = np.full(n_positions, np.nan)

        for t_i, t_offset in enumerate(time_positions):
            w_start = onset + t_offset
            w_end = w_start + window_bins

            if w_start < 0 or w_end > trial.shape[0]:
                continue

            feat = trial[w_start:w_end, :].mean(axis=0).reshape(1, -1)
            feat_scaled = scaler.transform(feat)

            prob_dist = clf.predict_proba(feat_scaled)[0]

            # Find P(correct current char)
            class_idx = np.where(clf.classes_ == correct_current)[0]
            if len(class_idx) > 0:
                probs_row[t_i] = prob_dist[class_idx[0]]

        all_probs.append(probs_row)
        all_current_labels.append(correct_current)
        valid_count += 1

    all_probs = np.array(all_probs)
    all_current_labels = np.array(all_current_labels)
    time_ms = time_positions * 10  # convert bins to ms

    print(f"  Processed {valid_count} test characters")

    # --- Figure A: Average signal trace ---
    mean_prob = np.nanmean(all_probs, axis=0)
    sem_prob = np.nanstd(all_probs, axis=0) / np.sqrt(np.sum(~np.isnan(all_probs), axis=0))
    chance = 1.0 / len(clf.classes_)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_ms, mean_prob, color="#2196F3", linewidth=2)
    ax.fill_between(time_ms, mean_prob - sem_prob, mean_prob + sem_prob,
                     color="#2196F3", alpha=0.2)
    ax.axhline(chance, color="gray", linestyle="--", linewidth=1, label=f"Chance ({chance:.3f})")
    ax.axvline(0, color="red", linestyle="-", linewidth=1.5, alpha=0.7, label="Character onset")

    ax.set_xlabel("Time relative to character onset (ms)")
    ax.set_ylabel("P(correct character)")
    ax.set_title("Temporal Footprint of Character Neural Representation")
    ax.legend()
    ax.set_xlim(time_ms[0], time_ms[-1])

    plt.tight_layout()
    plt.savefig(figures_dir / "fig_temporal_signal_trace.png", dpi=300)
    plt.savefig(figures_dir / "fig_temporal_signal_trace.pdf")
    plt.close()
    print("  Saved: fig_temporal_signal_trace")

    # --- Figure B: Per-letter breakdown (top 10 most frequent) ---
    letter_counts = {}
    for lbl in all_current_labels:
        letter_counts[lbl] = letter_counts.get(lbl, 0) + 1
    top_letters = sorted(letter_counts.keys(), key=lambda x: -letter_counts[x])[:10]

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.tab10
    for i, letter_idx in enumerate(top_letters):
        mask = all_current_labels == letter_idx
        letter_mean = np.nanmean(all_probs[mask], axis=0)
        char_label = IDX_TO_CHAR[letter_idx]
        if char_label == ">":
            char_label = "space"
        ax.plot(time_ms, letter_mean, linewidth=1.5, color=cmap(i),
                label=f"'{char_label}' (n={mask.sum()})", alpha=0.8)

    ax.axhline(chance, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="red", linestyle="-", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Time relative to character onset (ms)")
    ax.set_ylabel("P(correct character)")
    ax.set_title("Character Signal Trace by Letter Identity")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_xlim(time_ms[0], time_ms[-1])

    plt.tight_layout()
    plt.savefig(figures_dir / "fig_temporal_per_letter.png", dpi=300)
    plt.savefig(figures_dir / "fig_temporal_per_letter.pdf")
    plt.close()
    print("  Saved: fig_temporal_per_letter")

    # --- Figure C: Heatmap (letters × time) ---
    top15 = sorted(letter_counts.keys(), key=lambda x: -letter_counts[x])[:15]
    heatmap_data = []
    letter_labels_hm = []
    for letter_idx in top15:
        mask = all_current_labels == letter_idx
        letter_mean = np.nanmean(all_probs[mask], axis=0)
        heatmap_data.append(letter_mean)
        ch = IDX_TO_CHAR[letter_idx]
        letter_labels_hm.append(f"'{ch}' (n={mask.sum()})")

    heatmap_data = np.array(heatmap_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd",
                    extent=[time_ms[0], time_ms[-1], len(top15) - 0.5, -0.5],
                    interpolation="bilinear")
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(letter_labels_hm)
    ax.set_xlabel("Time relative to character onset (ms)")
    ax.set_title("Character Signal Heatmap")
    ax.axvline(0, color="white", linestyle="-", linewidth=2, alpha=0.8)

    plt.colorbar(im, ax=ax, label="P(correct character)")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig_temporal_heatmap.png", dpi=300)
    plt.savefig(figures_dir / "fig_temporal_heatmap.pdf")
    plt.close()
    print("  Saved: fig_temporal_heatmap")

    print("\nDone!")


if __name__ == "__main__":
    main()
