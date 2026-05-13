"""Experiment 0: Validate HMM alignment by overlaying character boundaries on neural data."""

import sys
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anticipatory.data.loader import load_session, SESSIONS
from anticipatory.visualization.figures import plot_alignment_validation


def main():
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_root = Path(__file__).parent.parent / config["data"]["root"]
    output_dir = Path(__file__).parent.parent / "results" / "alignment"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate 3 sessions as specified in proposal
    sessions_to_check = SESSIONS[:3]

    for session_id in sessions_to_check:
        print(f"\n=== {session_id} ===")
        session = load_session(data_root, session_id, config["data"]["hmm_source"])

        valid = np.where(session.valid_mask)[0]
        if len(valid) == 0:
            print("  No valid sentences with HMM labels")
            continue

        # Plot first 3 valid sentences per session
        for sent_i in valid[:3]:
            prompt = session.prompts[sent_i]
            n_bins = session.time_bins[sent_i]
            neural = session.neural[sent_i]
            starts = session.letter_starts[sent_i]
            durs = session.letter_durations[sent_i]

            print(f"  Sentence {sent_i}: \"{prompt}\" ({len(prompt)} chars, {n_bins} bins)")

            # Verify alignment: max letterStart + duration should be <= n_bins
            n_chars = len(prompt)
            max_end = 0
            for j in range(n_chars):
                end = int(starts[j] + np.floor(durs[j]))
                max_end = max(max_end, end)

            if max_end > n_bins:
                print(f"    WARNING: Character end ({max_end}) > time bins ({n_bins})")
            else:
                print(f"    Alignment OK: max char end = {max_end}, total bins = {n_bins}")

            save_path = output_dir / f"{session_id}_sent{sent_i}.png"
            plot_alignment_validation(neural, starts, durs, prompt, n_bins, save_path)
            print(f"    Saved: {save_path}")

    print(f"\nAlignment figures saved to {output_dir}")


if __name__ == "__main__":
    main()
