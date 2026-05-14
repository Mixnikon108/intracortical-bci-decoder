# Neural Coarticulation in Attempted Handwriting: Proposal for Phase 2

## Status: PENDING APPROVAL

---

## 1. Motivation for Pivot

Phase 1 tested anticipatory character encoding and found:
- A modest but significant N+1 signal (6.9% vs 3.2% chance, p=0.01)
- A consistently stronger N-1 (perseverative) signal (10.4%)
- Inter-character gaps containing decodable information

The anticipatory framing is weakened by the modest effect size and by N-1 dominating N+1. However, the N-1 > N+1 finding, combined with the temporal dynamics (Q1->Q4 gradient), points to a stronger story: **neural coarticulation** — the neural representation of a character is systematically modulated by its sequential context.

This is well-documented behaviorally in handwriting (Van Galen, 1991; Orliaguet et al., 1997) and neurally in speech (Chartier et al., 2018). But it has **never been demonstrated with intracortical recordings during attempted handwriting in a paralyzed participant**.

The precise novelty is: **coarticulation persists in motor cortical representations even in the absence of peripheral execution and proprioceptive feedback.**

---

## 2. Research Question

> Does the neural representation of a handwritten character in motor cortex depend on the identity of adjacent characters, and can this context-dependence be exploited to improve BCI decoding?

---

## 3. Proposed Analyses

### Analysis A: Representational Similarity Analysis (RSA)

**Objective:** Demonstrate that the same letter has different neural representations depending on surrounding context.

**Method:**
1. For each letter class (e.g., all instances of "e"), group instances by preceding character identity.
2. Compute the Representational Dissimilarity Matrix (RDM): pairwise Euclidean distances between all instances of "e".
3. Test whether within-context distances (same preceding letter) are smaller than between-context distances (different preceding letter).
4. Statistical test: permutation-based Mantel test (5,000 permutations).
5. Multiple comparisons: max-statistic correction across letter classes (controls FWER without Bonferroni conservatism).
6. Repeat for following character context (coarticulation in the forward direction).

**Critical control (from reviewer):** Trim first and last 100-150ms of each character epoch before computing features. This eliminates boundary contamination and ensures any context-dependence reflects genuine representational modulation, not temporal smearing from adjacent characters.

**Expected output:** Per-letter coarticulation strength (effect size of context modulation). Identifies which letters are most context-dependent and in which direction (backward vs forward coarticulation).

### Analysis B: Context-Aware Decoder Comparison

**Objective:** Quantify the practical BCI value of coarticulation by showing that context features improve character decoding.

**Method:**
1. **Context-free decoder:** Logistic regression predicting character N from its own neural features [192].
2. **Context-aware decoder:** Logistic regression predicting character N from concatenated features: character N neural [192] + character N-1 neural [192] + gap activity [192] = [576 features].
3. Both decoders use identical CV procedure (StratifiedGroupKFold, sentence-level grouping, L2 regularization tuned via inner CV).
4. Compare accuracy via paired permutation test on fold-level accuracies.

**Dimensionality concern (from statistics review):** 576 features with 23,262 samples gives a ~40:1 ratio, which is acceptable for L2-regularized logistic regression. Nested CV prevents overfitting.

**Baseline note (from engineering review):** Compare against our own context-free decoder with identical architecture, NOT against Willett et al.'s full CTC+LM pipeline (unfair comparison). Report Willett's numbers for reference only.

**Target metric:** 10-15% relative accuracy improvement, or demonstrate that context-aware decoder reaches equivalent accuracy 40-60ms earlier (time-to-threshold analysis).

### Analysis C: Time-to-Threshold (Latency Analysis)

**Objective:** Quantify how much earlier the context-aware decoder can make a confident prediction.

**Method:**
1. For both decoders, measure classification accuracy as a function of how many time bins into the current character have been consumed (0ms, 20ms, 40ms, ...).
2. Define a threshold accuracy (e.g., 80% of peak accuracy).
3. Measure the time at which each decoder crosses this threshold.
4. The difference is the "latency gain" from exploiting coarticulation.

**Gap activity claim:** Only claim "decoding begins during the gap" if gap-only features (no current character bins) exceed chance with p < 0.01.

### Analysis D: State-Space Trajectory Analysis

**Objective:** Visualize how sequential context modulates the neural trajectory of character execution.

**Method (from neuroscience review):**
1. Project neural activity into a low-dimensional space (PCA, top 10 components).
2. For a given letter (e.g., "e"), plot the mean neural trajectory across time, color-coded by preceding character identity (e.g., "t->e" vs "r->e" vs "n->e").
3. If trajectories start separated (different initial conditions from different preceding characters) and converge mid-execution, this is direct evidence of coarticulation.
4. If trajectories never converge, the signal is more consistent with temporal smearing.

**Why this matters:** This analysis directly distinguishes coarticulation (context modulates the representation) from temporal bleed (adjacent character activity leaks into the epoch). It provides a visual narrative that is highly publishable.

### Analysis E: Temporal Generalization Matrix

**Objective:** Map which time periods within a character share neural representations.

**Method:**
1. Train classifier at time t1, evaluate at time t2 (all pairwise combinations within a character epoch).
2. Split by preceding character context: compute separate generalization matrices for high-coarticulation vs low-coarticulation bigrams.
3. The difference reveals when and how context modulates the temporal dynamics.

---

## 4. What We Do NOT Need to Redo

The following Phase 1 results remain valid and directly support the coarticulation framing:

| Phase 1 Result | Role in Phase 2 |
|---|---|
| N-1 > N+1 (perseverative dominance) | Central finding: perseverative coarticulation is the dominant effect |
| Sigma=0 signal survives | Coarticulation is not a smoothing artifact |
| Causal ~ Gaussian | Not caused by filter forward-leak |
| Isolated control at chance | Pipeline validation (no false positives) |
| Bigram rho ~ 0 | Coarticulation is not frequency-dependent |
| Q1->Q4 temporal gradient | Temporal dynamics of context modulation |

---

## 5. Reviewer Concerns and Mitigations

Based on the advisory council review:

| Concern | Source | Mitigation |
|---|---|---|
| RSA conflates temporal bleed with coarticulation | Neuroscience reviewer | Trim 100-150ms from character edges; state-space trajectory convergence test |
| 47-degree subspace angle is near random | Neuroscience reviewer | Do not present as evidence of separation; drop or reframe |
| Context-free vs context-aware is insufficient alone | Neuroscience reviewer | Combine with RSA, trajectory analysis, and temporal generalization |
| Concatenating to 576 features risks overfitting | Statistics reviewer | L2 regularization + nested CV; 40:1 sample-to-feature ratio is adequate |
| Need 1000+ permutations for publication | Statistics reviewer | Increase to 5,000 for RSA Mantel tests; 1,000 minimum for probe permutation tests |
| 3-fold CV is insufficient | Statistics reviewer | Move to 10-fold for all publication results |
| Compare against own baseline, not Willett CTC | Engineering reviewer | Context-free decoder with identical architecture as baseline |
| N=1 participant | All reviewers | Acceptable for J. Neural Engineering; Willett's own paper is N=1. State clearly as limitation. |
| Effect size may be too small for decoder improvement | Engineering reviewer | Target 10-15% relative improvement; focus on latency gain if accuracy gain is modest |

---

## 6. Publication Strategy

### Target venue: Journal of Neural Engineering

**Why:** Routinely publishes single-participant BCI analyses. Values both neuroscience insight and engineering contribution. The combination of coarticulation characterization + decoder improvement fits their scope perfectly.

### Alternative venues:
- **Journal of Neurophysiology** — if the neuroscience findings are strong but decoder improvement is modest
- **eLife** — if both RSA/trajectory results AND decoder improvement are clean and compelling
- **NeurIPS/COSYNE workshop** — achievable with current results for a shorter submission

### Paper structure:
1. **Introduction:** Coarticulation in handwriting (behavioral literature) + neural coarticulation in speech (Chartier) + gap in intracortical handwriting data
2. **Results:**
   - RSA demonstrates context-dependent representations (Analysis A)
   - State-space trajectories show context-modulated dynamics (Analysis D)
   - Temporal generalization reveals when context matters (Analysis E)
   - Context-aware decoder outperforms context-free (Analysis B)
   - Latency analysis shows earlier decoding (Analysis C)
3. **Discussion:** Coarticulation persists without peripheral execution; implications for BCI decoder design; limitations (N=1, HMM boundaries)

---

## 7. Compute Estimate

| Analysis | Estimated Time | Notes |
|---|---|---|
| RSA with 5,000 permutations per letter | ~2-3 hours | Parallelizable |
| Context-aware decoder (10-fold CV) | ~30 min | Same pipeline as Phase 1 |
| Time-to-threshold analysis | ~2 hours | Multiple time points |
| State-space trajectories | ~10 min | PCA + visualization |
| Temporal generalization matrix | ~3-4 hours | Many train/test combinations |
| Publication figures | ~30 min | |

**Total: ~1 day of compute.** No GPU required.

---

## 8. Decision Required

This proposal reorients the project from "anticipatory character encoding" to "neural coarticulation in attempted handwriting." The data, pipeline, and controls from Phase 1 carry over. The new analyses (RSA, decoder comparison, trajectory analysis) address the weaknesses identified in Phase 1 and provide a stronger, more publishable story.

**Proceed?**
