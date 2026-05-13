# Anticipatory Character Encoding in Motor Cortex

Analysis of whether human motor cortex neural activity during handwritten character production contains decodable information about the identity of the *next* character in the sequence.

Based on the Willett et al. (2021) intracortical handwriting BCI dataset (192-channel Utah array, motor cortex, participant T5).

## Research Question

> Does motor cortex encode anticipatory information about the upcoming character during execution of the current character, and is this signal attributable to motor planning rather than temporal blurring or linguistic statistics?

## Method

Residualized linear probing: decode current character from neural activity, project out that subspace, then probe the residual for next-character information. Evaluated with sentence-level grouped cross-validation and permutation testing.

## Experiments

| # | Experiment | Question |
|---|-----------|----------|
| 0 | Alignment validation | Do HMM character boundaries match neural activity? |
| 1 | Anticipatory probe (N+1) | Can we decode the next character from residualized neural activity? |
| 2 | Reverse-time control (N-1) | Is the signal directional (forward) or symmetric (blurring)? |
| 3 | Blurring controls | Does the signal survive at sigma=0 and under causal filtering? |
| 4 | Linguistic baselines | Does the neural signal exceed what bigram/n-gram statistics predict? |
| 5 | Temporal dynamics | When during character execution does anticipatory information emerge? |
| 6 | Isolated letter control | Is the signal absent when there is no sequential context? |
| 7 | Subspace geometry | Do current and next character occupy orthogonal neural subspaces? |
| 8 | Bigram frequency | Does anticipation scale with how practiced a transition is? |

## Quick Start

```bash
pip install -e .

# Validate alignment (Experiment 0)
python scripts/run_alignment.py

# Run a specific experiment
python scripts/run_experiment.py core        # Exp 1+2
python scripts/run_experiment.py blurring    # Exp 3
python scripts/run_experiment.py linguistic  # Exp 4
python scripts/run_experiment.py temporal    # Exp 5
python scripts/run_experiment.py isolated    # Exp 6
python scripts/run_experiment.py subspace    # Exp 7
python scripts/run_experiment.py bigram      # Exp 8

# Run everything
python scripts/run_experiment.py all
```

## Dataset

Willett et al. (2021), Dryad: [doi:10.5061/dryad.wh70rxwmv](https://doi.org/10.5061/dryad.wh70rxwmv)

Configure the data path in `configs/default.yaml`.

## References

- Willett et al. (2021). *Nature*, 593, 249-254.
- Zimnik & Churchland (2021). *Nature Neuroscience*, 24, 412-424.
- Chartier et al. (2018). *Neuron*, 98(5), 1042-1054.
- Kobak et al. (2016). *eLife*, 5, e10989.
