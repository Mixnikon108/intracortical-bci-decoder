"""Run a single experiment or all experiments."""

import sys
import argparse
import json
import pickle
from pathlib import Path
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_data_root(config):
    return Path(__file__).parent.parent / config["data"]["root"]


def get_output_dir(experiment_name):
    d = Path(__file__).parent.parent / "results" / experiment_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_results(results, output_dir, name):
    """Save results as pickle (full) and JSON (summary)."""
    # Pickle for full results (includes numpy arrays)
    with open(output_dir / f"{name}.pkl", "wb") as f:
        pickle.dump(results, f)

    # JSON-safe summary
    def to_json_safe(obj):
        if isinstance(obj, np.ndarray):
            if obj.size < 100:
                return obj.tolist()
            return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_json_safe(v) for v in obj]
        return obj

    with open(output_dir / f"{name}.json", "w") as f:
        json.dump(to_json_safe(results), f, indent=2)


def run_core(config):
    """Experiments 1 & 2: Core anticipatory and reverse-time probes."""
    from anticipatory.data.loader import load_all_sessions
    from anticipatory.data.features import extract_character_features
    from anticipatory.experiments.core_probes import run_core_probes_with_permutation

    data_root = get_data_root(config)
    output_dir = get_output_dir("core_probes")
    sigma = config["preprocessing"]["default_sigma"]
    ac = config["analysis"]

    print("Loading sessions...")
    sessions = load_all_sessions(data_root, config["data"]["hmm_source"])

    print(f"\nExtracting features (sigma={sigma}, window=q1)...")
    features = extract_character_features(sessions, sigma=sigma, window="q1",
                                          min_duration=config["preprocessing"]["min_char_duration"])
    ll = features.filter_transition_type(["ll"])
    print(f"Letter-letter transitions: {len(ll)}")

    results = run_core_probes_with_permutation(
        ll,
        n_splits=ac["n_cv_folds"],
        n_permutations=ac["n_permutations"],
        n_jobs=ac["n_jobs"],
        seed=ac["random_seed"],
    )

    save_results(results, output_dir, "core_probes")
    print(f"\nResults saved to {output_dir}")
    return results


def run_blurring(config):
    """Experiment 3: Sigma sweep + causal filter."""
    from anticipatory.data.loader import load_all_sessions
    from anticipatory.experiments.blurring_controls import (
        run_sigma_sweep, run_causal_filter_comparison
    )

    data_root = get_data_root(config)
    output_dir = get_output_dir("blurring_controls")
    ac = config["analysis"]

    print("Loading sessions...")
    sessions = load_all_sessions(data_root, config["data"]["hmm_source"])

    print("\n=== Sigma Sweep ===")
    sweep = run_sigma_sweep(
        sessions,
        sigma_values=config["preprocessing"]["sigma_values"],
        min_duration=config["preprocessing"]["min_char_duration"],
        n_splits=ac["n_cv_folds"],
        seed=ac["random_seed"],
    )
    save_results(sweep, output_dir, "sigma_sweep")

    print("\n=== Causal Filter Comparison ===")
    causal = run_causal_filter_comparison(
        sessions,
        sigma_values=[2, 4, 8],
        min_duration=config["preprocessing"]["min_char_duration"],
        n_splits=ac["n_cv_folds"],
        seed=ac["random_seed"],
    )
    save_results(causal, output_dir, "causal_comparison")

    print(f"\nResults saved to {output_dir}")
    return {"sweep": sweep, "causal": causal}


def run_linguistic(config):
    """Experiment 4: Linguistic baselines."""
    from anticipatory.data.loader import load_all_sessions
    from anticipatory.data.features import extract_character_features
    from anticipatory.experiments.linguistic_baseline import run_all_linguistic_baselines

    data_root = get_data_root(config)
    output_dir = get_output_dir("linguistic_baseline")
    ac = config["analysis"]
    sigma = config["preprocessing"]["default_sigma"]

    print("Loading sessions...")
    sessions = load_all_sessions(data_root, config["data"]["hmm_source"])

    print(f"\nExtracting features (sigma={sigma})...")
    features = extract_character_features(sessions, sigma=sigma, window="q1",
                                          min_duration=config["preprocessing"]["min_char_duration"])
    ll = features.filter_transition_type(["ll"])

    results = run_all_linguistic_baselines(ll, ac["n_cv_folds"], ac["random_seed"])
    save_results(results, output_dir, "linguistic_baselines")

    print(f"\nResults saved to {output_dir}")
    return results


def run_temporal(config):
    """Experiment 5: Temporal dynamics."""
    from anticipatory.data.loader import load_all_sessions
    from anticipatory.experiments.temporal_dynamics import run_quartile_profile

    data_root = get_data_root(config)
    output_dir = get_output_dir("temporal_dynamics")
    ac = config["analysis"]
    sigma = config["preprocessing"]["default_sigma"]

    print("Loading sessions...")
    sessions = load_all_sessions(data_root, config["data"]["hmm_source"])

    results = run_quartile_profile(
        sessions, sigma=sigma,
        min_duration=config["preprocessing"]["min_char_duration"],
        n_splits=ac["n_cv_folds"],
        seed=ac["random_seed"],
    )
    save_results(results, output_dir, "quartile_profile")

    print(f"\nResults saved to {output_dir}")
    return results


def run_isolated(config):
    """Experiment 6: Isolated letter control."""
    from anticipatory.data.loader import load_all_single_letters
    from anticipatory.experiments.isolated_control import run_isolated_letter_control

    data_root = get_data_root(config)
    output_dir = get_output_dir("isolated_control")
    ac = config["analysis"]
    sigma = config["preprocessing"]["default_sigma"]

    print("Loading single-letter data...")
    sl_data = load_all_single_letters(data_root)

    results = run_isolated_letter_control(sl_data, sigma=sigma,
                                          n_splits=ac["n_cv_folds"],
                                          seed=ac["random_seed"])
    save_results(results, output_dir, "isolated_control")

    print(f"\nResults saved to {output_dir}")
    return results


def run_subspace(config):
    """Experiment 7: Subspace geometry."""
    from anticipatory.data.loader import load_all_sessions
    from anticipatory.data.features import extract_character_features
    from anticipatory.experiments.subspace_geometry import run_subspace_analysis

    data_root = get_data_root(config)
    output_dir = get_output_dir("subspace_geometry")
    sigma = config["preprocessing"]["default_sigma"]

    print("Loading sessions...")
    sessions = load_all_sessions(data_root, config["data"]["hmm_source"])

    features = extract_character_features(sessions, sigma=sigma, window="q1",
                                          min_duration=config["preprocessing"]["min_char_duration"])
    ll = features.filter_transition_type(["ll"])

    results = run_subspace_analysis(ll)
    save_results(results, output_dir, "subspace_geometry")

    print(f"\nResults saved to {output_dir}")
    return results


def run_bigram(config):
    """Experiment 8: Bigram frequency modulation."""
    from anticipatory.data.loader import load_all_sessions
    from anticipatory.data.features import extract_character_features
    from anticipatory.experiments.bigram_frequency import run_bigram_frequency_analysis

    data_root = get_data_root(config)
    output_dir = get_output_dir("bigram_frequency")
    ac = config["analysis"]
    sigma = config["preprocessing"]["default_sigma"]

    print("Loading sessions...")
    sessions = load_all_sessions(data_root, config["data"]["hmm_source"])

    features = extract_character_features(sessions, sigma=sigma, window="q1",
                                          min_duration=config["preprocessing"]["min_char_duration"])
    ll = features.filter_transition_type(["ll"])

    results = run_bigram_frequency_analysis(ll, n_splits=ac["n_cv_folds"],
                                            seed=ac["random_seed"])
    save_results(results, output_dir, "bigram_frequency")

    print(f"\nResults saved to {output_dir}")
    return results


EXPERIMENTS = {
    "core": run_core,
    "blurring": run_blurring,
    "linguistic": run_linguistic,
    "temporal": run_temporal,
    "isolated": run_isolated,
    "subspace": run_subspace,
    "bigram": run_bigram,
}


def main():
    parser = argparse.ArgumentParser(description="Run anticipatory encoding experiments")
    parser.add_argument("experiment", choices=list(EXPERIMENTS.keys()) + ["all"],
                        help="Which experiment to run")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config() if args.config is None else yaml.safe_load(open(args.config))

    if args.experiment == "all":
        for name, fn in EXPERIMENTS.items():
            print(f"\n{'='*60}")
            print(f"  EXPERIMENT: {name}")
            print(f"{'='*60}\n")
            fn(config)
    else:
        EXPERIMENTS[args.experiment](config)


if __name__ == "__main__":
    main()
