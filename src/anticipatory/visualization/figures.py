"""Publication-quality figures for all experiments."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
from typing import Optional

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_alignment_validation(
    neural: np.ndarray,
    letter_starts: np.ndarray,
    letter_durations: np.ndarray,
    prompt: str,
    n_bins: int,
    save_path: Optional[Path] = None,
):
    """Plot neural raster with HMM character boundaries overlaid."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={"height_ratios": [3, 1]})

    trial = neural[:n_bins, :]

    # Neural raster
    ax = axes[0]
    ax.imshow(trial.T, aspect="auto", cmap="viridis", interpolation="none",
              vmin=np.percentile(trial, 5), vmax=np.percentile(trial, 95))
    ax.set_ylabel("Channel")
    ax.set_title(f'Neural activity — "{prompt}"')

    # Character boundaries
    n_chars = len(prompt)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_chars, 20)))
    for j in range(n_chars):
        start = int(letter_starts[j])
        dur = int(np.floor(letter_durations[j]))
        if dur <= 0:
            break
        ax.axvline(start, color="white", linewidth=0.5, alpha=0.7)
        ax.text(start + dur / 2, -2, prompt[j], ha="center", va="bottom",
                fontsize=7, color="white", fontweight="bold")

    # Population firing rate
    ax = axes[1]
    pop_rate = trial.sum(axis=1)
    ax.plot(pop_rate, color="steelblue", linewidth=0.8)
    ax.set_xlabel("Time bin (10ms)")
    ax.set_ylabel("Population rate")

    for j in range(n_chars):
        start = int(letter_starts[j])
        dur = int(np.floor(letter_durations[j]))
        if dur <= 0:
            break
        ax.axvline(start, color="red", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_core_probes_comparison(results: dict, save_path: Optional[Path] = None):
    """Bar chart comparing N+1 vs N-1 balanced accuracy with chance line."""
    fig, ax = plt.subplots(figsize=(6, 4))

    labels = ["N+1 (anticipatory)", "N-1 (reverse)"]
    accs = [
        results["next"]["balanced_accuracy"],
        results["prev"]["balanced_accuracy"],
    ]
    colors = ["#2196F3", "#FF9800"]

    bars = ax.bar(labels, accs, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0 / 31, color="gray", linestyle="--", label="Chance (1/31)", linewidth=1)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Residualized Probe: N+1 vs N-1")
    ax.legend()

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, max(accs) * 1.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_sigma_sweep(results: dict, save_path: Optional[Path] = None):
    """Line plot of N+1 and N-1 accuracy across sigma values."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    sigmas = sorted(results.keys())
    next_accs = [results[s]["next_balanced_accuracy"] for s in sigmas]
    prev_accs = [results[s]["prev_balanced_accuracy"] for s in sigmas]

    ax.plot(sigmas, next_accs, "o-", color="#2196F3", label="N+1 (anticipatory)", linewidth=2)
    ax.plot(sigmas, prev_accs, "s--", color="#FF9800", label="N-1 (reverse)", linewidth=2)
    ax.axhline(1.0 / 31, color="gray", linestyle=":", label="Chance", linewidth=1)

    ax.set_xlabel("Gaussian smoothing sigma (bins)")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Experiment 3a: Effect of Temporal Smoothing")
    ax.legend()
    ax.set_xticks(sigmas)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_causal_comparison(results: dict, save_path: Optional[Path] = None):
    """Grouped bar chart: Gaussian vs Causal at each sigma."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    sigmas = sorted(results.keys())
    x = np.arange(len(sigmas))
    width = 0.35

    gaussian_accs = [results[s]["gaussian"]["next_balanced_accuracy"] for s in sigmas]
    causal_accs = [results[s]["causal"]["next_balanced_accuracy"] for s in sigmas]

    ax.bar(x - width / 2, gaussian_accs, width, label="Gaussian (symmetric)", color="#2196F3")
    ax.bar(x + width / 2, causal_accs, width, label="Causal (one-sided)", color="#4CAF50")
    ax.axhline(1.0 / 31, color="gray", linestyle=":", label="Chance")

    ax.set_xlabel("Smoothing parameter (bins)")
    ax.set_ylabel("N+1 Balanced Accuracy")
    ax.set_title("Experiment 3b: Gaussian vs Causal Smoothing")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sigmas])
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_linguistic_comparison(neural_results: dict, ling_results: dict,
                                save_path: Optional[Path] = None):
    """Bar chart comparing neural probe against linguistic baselines."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    methods = ["Neural\n(residualized)", "Bigram\n(one-hot)", "5-gram\ncontext", "Majority\nvote"]
    accs = [
        neural_results["balanced_accuracy"],
        ling_results["bigram"]["balanced_accuracy"],
        ling_results["context_5gram"]["balanced_accuracy"],
        ling_results["majority_vote"]["balanced_accuracy"],
    ]
    colors = ["#2196F3", "#9E9E9E", "#9E9E9E", "#9E9E9E"]

    bars = ax.bar(methods, accs, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0 / 31, color="gray", linestyle="--", label="Chance (1/31)")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Experiment 4: Neural vs Linguistic Baselines")
    ax.legend()

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_temporal_profile(quartile_results: dict, save_path: Optional[Path] = None):
    """Line plot of N+1 and N-1 accuracy across quartiles."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    quartiles = ["q1", "q2", "q3", "q4"]
    labels = ["Q1\n(0-25%)", "Q2\n(25-50%)", "Q3\n(50-75%)", "Q4\n(75-100%)"]
    next_accs = [quartile_results[q]["next_balanced_accuracy"] for q in quartiles]
    prev_accs = [quartile_results[q]["prev_balanced_accuracy"] for q in quartiles]

    ax.plot(range(4), next_accs, "o-", color="#2196F3", label="N+1", linewidth=2, markersize=8)
    ax.plot(range(4), prev_accs, "s--", color="#FF9800", label="N-1", linewidth=2, markersize=8)
    ax.axhline(1.0 / 31, color="gray", linestyle=":", label="Chance")

    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Temporal position within character")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Experiment 5: Temporal Dynamics of Anticipatory Signal")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_permutation_null(perm_results: dict, label: str = "N+1",
                          save_path: Optional[Path] = None):
    """Histogram of null distribution with observed value marked."""
    fig, ax = plt.subplots(figsize=(6, 4))

    null = perm_results["null_distribution"]
    observed = perm_results["observed"]

    ax.hist(null, bins=50, color="#BDBDBD", edgecolor="white", density=True, alpha=0.8)
    ax.axvline(observed, color="#F44336", linewidth=2, label=f"Observed ({observed:.4f})")
    ax.set_xlabel("Balanced Accuracy")
    ax.set_ylabel("Density")
    ax.set_title(f"Permutation Test — {label} (p={perm_results['p_value']:.4f})")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_bigram_frequency(bg_results: dict, save_path: Optional[Path] = None):
    """Scatter plot of bigram frequency vs anticipatory accuracy."""
    fig, ax = plt.subplots(figsize=(6, 5))

    freqs = bg_results["frequencies"]
    accs = bg_results["accuracies"]
    rho = bg_results["spearman_rho"]
    p = bg_results["spearman_p_value"]

    ax.scatter(freqs, accs, alpha=0.6, s=40, color="#2196F3", edgecolor="white", linewidth=0.5)

    # Trend line
    z = np.polyfit(np.log(freqs), accs, 1)
    x_smooth = np.linspace(freqs.min(), freqs.max(), 100)
    ax.plot(x_smooth, np.poly1d(z)(np.log(x_smooth)), "--", color="#F44336", linewidth=1.5)

    ax.set_xlabel("Bigram frequency (count)")
    ax.set_ylabel("Per-bigram accuracy")
    ax.set_title(f"Experiment 8: Bigram Frequency Modulation\n"
                 f"Spearman rho={rho:.3f}, p={p:.4f}")
    ax.set_xscale("log")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_subspace_angles(dpca_results: dict, save_path: Optional[Path] = None):
    """Bar chart of principal angles between current and next subspaces."""
    fig, ax = plt.subplots(figsize=(7, 4))

    angles = dpca_results["principal_angles_deg"]
    n = len(angles)

    ax.bar(range(n), angles, color="#2196F3", edgecolor="white")
    ax.axhline(90, color="red", linestyle="--", label="Orthogonal (90 deg)", linewidth=1)
    ax.set_xlabel("Component pair")
    ax.set_ylabel("Principal angle (degrees)")
    ax.set_title("Experiment 7: Subspace Angles (Current vs Next)")
    ax.legend()
    ax.set_ylim(0, 95)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig
