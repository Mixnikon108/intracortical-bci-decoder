"""Generate all publication-quality figures from saved results."""

import sys
import pickle
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anticipatory.visualization.figures import (
    plot_sigma_sweep,
    plot_causal_comparison,
    plot_temporal_profile,
    plot_bigram_frequency,
    plot_subspace_angles,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ── Figure 1: Core Probes (N+1 vs N-1 bar chart) ────────────────────────
# Core probes results were not saved to disk; reconstruct from sigma_sweep σ=4
# which used the same pipeline (Q1, σ=4, 3-fold CV)

def fig1_core_probes():
    """Bar chart: N+1 vs N-1 balanced accuracy with chance line."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Values from sigma_sweep σ=4 (equivalent to core probes settings)
    sweep = load_json(RESULTS_DIR / "blurring_controls" / "sigma_sweep.json")
    n1_acc = sweep["4"]["next_balanced_accuracy"]
    p1_acc = sweep["4"]["prev_balanced_accuracy"]

    labels = ["N+1\n(anticipatory)", "N-1\n(perseverative)"]
    accs = [n1_acc, p1_acc]
    colors = ["#2196F3", "#FF9800"]

    bars = ax.bar(labels, accs, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0 / 31, color="gray", linestyle="--", label="Chance (1/31)", linewidth=1)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Residualized Probe: Next vs Previous Character")
    ax.legend()

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, max(accs) * 1.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_core_probes.png")
    plt.savefig(FIGURES_DIR / "fig1_core_probes.pdf")
    plt.close()
    print("  Fig 1: Core probes comparison")


# ── Figure 2: Sigma Sweep ────────────────────────────────────────────────

def fig2_sigma_sweep():
    sweep_json = load_json(RESULTS_DIR / "blurring_controls" / "sigma_sweep.json")
    # Convert string keys to int
    sweep = {int(k): v for k, v in sweep_json.items()}
    plot_sigma_sweep(sweep, save_path=FIGURES_DIR / "fig2_sigma_sweep.png")
    plt.savefig(FIGURES_DIR / "fig2_sigma_sweep.pdf")
    plt.close()
    print("  Fig 2: Sigma sweep")


# ── Figure 3: Causal vs Gaussian ─────────────────────────────────────────

def fig3_causal_comparison():
    causal_json = load_json(RESULTS_DIR / "blurring_controls" / "causal_comparison.json")
    causal = {int(k): v for k, v in causal_json.items()}
    plot_causal_comparison(causal, save_path=FIGURES_DIR / "fig3_causal_comparison.png")
    plt.savefig(FIGURES_DIR / "fig3_causal_comparison.pdf")
    plt.close()
    print("  Fig 3: Causal vs Gaussian comparison")


# ── Figure 4: Temporal Profile (the star figure) ─────────────────────────

def fig4_temporal_profile():
    quartile = load_json(RESULTS_DIR / "temporal_dynamics" / "quartile_profile.json")
    plot_temporal_profile(quartile, save_path=FIGURES_DIR / "fig4_temporal_profile.png")
    plt.savefig(FIGURES_DIR / "fig4_temporal_profile.pdf")
    plt.close()
    print("  Fig 4: Temporal profile (quartiles)")


# ── Figure 5: Linguistic Baselines ───────────────────────────────────────

def fig5_linguistic_baselines():
    """Neural vs linguistic baselines — adapted since neural and linguistic
    measure fundamentally different things (residualized neural vs char identity)."""
    ling = load_json(RESULTS_DIR / "linguistic_baseline" / "linguistic_baselines.json")

    # Neural accuracy from sigma_sweep σ=4
    sweep = load_json(RESULTS_DIR / "blurring_controls" / "sigma_sweep.json")
    neural_acc = sweep["4"]["next_balanced_accuracy"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    methods = [
        "Neural\n(residualized\nprobe)",
        "Majority\nvote",
        "Bigram\n(one-hot N)",
        "5-gram\ncontext",
    ]
    accs = [
        neural_acc,
        ling["majority_vote"]["balanced_accuracy"],
        ling["bigram"]["balanced_accuracy"],
        ling["context_5gram"]["balanced_accuracy"],
    ]
    colors = ["#2196F3", "#BDBDBD", "#9E9E9E", "#757575"]

    bars = ax.bar(methods, accs, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0 / 31, color="gray", linestyle="--", label="Chance (1/31)")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Neural Probe vs Linguistic Baselines for N+1 Prediction")
    ax.legend()

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=9)

    # Annotation explaining the difference
    ax.annotate(
        "Neural probe uses residualized\nneural activity (identity of N removed).\n"
        "Linguistic baselines use character\nidentity directly.",
        xy=(0.98, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_linguistic_baselines.png")
    plt.savefig(FIGURES_DIR / "fig5_linguistic_baselines.pdf")
    plt.close()
    print("  Fig 5: Linguistic baselines")


# ── Figure 6: Subspace Geometry ──────────────────────────────────────────

def fig6_subspace_geometry():
    geo = load_json(RESULTS_DIR / "subspace_geometry" / "subspace_geometry.json")

    # Panel A: principal angles
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    angles = geo["dpca"]["principal_angles_deg"]
    n = len(angles)

    ax = axes[0]
    bars = ax.bar(range(1, n + 1), angles, color="#2196F3", edgecolor="white")
    ax.axhline(90, color="#F44336", linestyle="--", label="Orthogonal (90°)", linewidth=1)
    ax.axhline(geo["dpca"]["mean_principal_angle_deg"], color="#4CAF50",
               linestyle=":", label=f"Mean ({geo['dpca']['mean_principal_angle_deg']:.1f}°)", linewidth=1.5)
    ax.set_xlabel("Component pair")
    ax.set_ylabel("Principal angle (degrees)")
    ax.set_title("A. Subspace Angles (Current vs Next)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 95)

    # Panel B: canonical correlations
    ax = axes[1]
    cc = geo["cca"]["canonical_correlations"]
    ax.bar(range(1, len(cc) + 1), cc, color="#FF9800", edgecolor="white")
    ax.axhline(geo["cca"]["mean_correlation"], color="#4CAF50",
               linestyle=":", label=f"Mean ({geo['cca']['mean_correlation']:.3f})", linewidth=1.5)
    ax.set_xlabel("Canonical component")
    ax.set_ylabel("Canonical correlation")
    ax.set_title("B. Canonical Correlation Analysis")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_subspace_geometry.png")
    plt.savefig(FIGURES_DIR / "fig6_subspace_geometry.pdf")
    plt.close()
    print("  Fig 6: Subspace geometry")


# ── Figure 7: Bigram Frequency ───────────────────────────────────────────

def fig7_bigram_frequency():
    bg = load_pkl(RESULTS_DIR / "bigram_frequency" / "bigram_frequency.pkl")
    plot_bigram_frequency(bg, save_path=FIGURES_DIR / "fig7_bigram_frequency.png")
    plt.savefig(FIGURES_DIR / "fig7_bigram_frequency.pdf")
    plt.close()
    print("  Fig 7: Bigram frequency modulation")


# ── Figure 8: Isolated Control ───────────────────────────────────────────

def fig8_isolated_control():
    iso = load_json(RESULTS_DIR / "isolated_control" / "isolated_control.json")

    fig, ax = plt.subplots(figsize=(6, 4))

    labels = ["Isolated letters\n(random N+1)", "Sentence context\n(real N+1)"]
    # Get sentence N+1 from sigma_sweep σ=4
    sweep = load_json(RESULTS_DIR / "blurring_controls" / "sigma_sweep.json")
    accs = [iso["balanced_accuracy"], sweep["4"]["next_balanced_accuracy"]]
    colors = ["#9E9E9E", "#2196F3"]

    bars = ax.bar(labels, accs, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0 / 31, color="gray", linestyle="--", label="Chance (1/31)", linewidth=1)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Negative Control: Isolated Letters vs Sentence Context")
    ax.legend()

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, max(accs) * 1.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig8_isolated_control.png")
    plt.savefig(FIGURES_DIR / "fig8_isolated_control.pdf")
    plt.close()
    print("  Fig 8: Isolated letter control")


# ── Figure 9: Summary panel ─────────────────────────────────────────────

def fig9_summary():
    """Combined summary figure with key results."""
    sweep_json = load_json(RESULTS_DIR / "blurring_controls" / "sigma_sweep.json")
    sweep = {int(k): v for k, v in sweep_json.items()}
    quartile = load_json(RESULTS_DIR / "temporal_dynamics" / "quartile_profile.json")
    iso = load_json(RESULTS_DIR / "isolated_control" / "isolated_control.json")
    geo = load_json(RESULTS_DIR / "subspace_geometry" / "subspace_geometry.json")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    chance = 1.0 / 31

    # A: Core result — N+1 vs N-1
    ax = axes[0, 0]
    labels = ["N+1", "N-1"]
    accs = [sweep[4]["next_balanced_accuracy"], sweep[4]["prev_balanced_accuracy"]]
    bars = ax.bar(labels, accs, color=["#2196F3", "#FF9800"], width=0.4,
                  edgecolor="black", linewidth=0.5)
    ax.axhline(chance, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("A. Anticipatory vs Perseverative Signal")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(accs) * 1.25)

    # B: Temporal profile
    ax = axes[0, 1]
    qs = ["q1", "q2", "q3", "q4"]
    qlabels = ["Q1", "Q2", "Q3", "Q4"]
    next_a = [quartile[q]["next_balanced_accuracy"] for q in qs]
    prev_a = [quartile[q]["prev_balanced_accuracy"] for q in qs]
    ax.plot(range(4), next_a, "o-", color="#2196F3", label="N+1", linewidth=2, markersize=8)
    ax.plot(range(4), prev_a, "s--", color="#FF9800", label="N-1", linewidth=2, markersize=8)
    ax.axhline(chance, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(qlabels)
    ax.set_xlabel("Position within character")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("B. Temporal Accumulation")
    ax.legend()

    # C: Sigma sweep
    ax = axes[1, 0]
    sigmas = sorted(sweep.keys())
    n_accs = [sweep[s]["next_balanced_accuracy"] for s in sigmas]
    p_accs = [sweep[s]["prev_balanced_accuracy"] for s in sigmas]
    ax.plot(sigmas, n_accs, "o-", color="#2196F3", label="N+1", linewidth=2)
    ax.plot(sigmas, p_accs, "s--", color="#FF9800", label="N-1", linewidth=2)
    ax.axhline(chance, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Gaussian σ (bins)")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("C. Smoothing Control")
    ax.set_xticks(sigmas)
    ax.legend()

    # D: Subspace angles
    ax = axes[1, 1]
    angles = geo["dpca"]["principal_angles_deg"]
    ax.bar(range(1, len(angles) + 1), angles, color="#2196F3", edgecolor="white")
    ax.axhline(90, color="#F44336", linestyle="--", linewidth=1)
    ax.axhline(geo["dpca"]["mean_principal_angle_deg"], color="#4CAF50",
               linestyle=":", linewidth=1.5, label=f"Mean = {geo['dpca']['mean_principal_angle_deg']:.1f}°")
    ax.set_xlabel("Component pair")
    ax.set_ylabel("Principal angle (°)")
    ax.set_title("D. Subspace Separation")
    ax.legend()
    ax.set_ylim(0, 95)

    plt.suptitle("Anticipatory Character Encoding in Motor Cortex", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig9_summary.png")
    plt.savefig(FIGURES_DIR / "fig9_summary.pdf")
    plt.close()
    print("  Fig 9: Summary panel")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("Generating publication figures...")
    fig1_core_probes()
    fig2_sigma_sweep()
    fig3_causal_comparison()
    fig4_temporal_profile()
    fig5_linguistic_baselines()
    fig6_subspace_geometry()
    fig7_bigram_frequency()
    fig8_isolated_control()
    fig9_summary()
    print(f"\nAll figures saved to {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()
