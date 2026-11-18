# Monotone Rationale-to-Behavior Certification for LLM-Designed Architectures

## Motivation

Across both the prior-knowledge NAS branch (e.g., Prior Knowledge Guided Neural Architecture Generation, venue:icml:2025:d5970b220266) and the weight-space programming branch (Weight-Space Programming Doesn't Need Transformers, arxiv:2611.22055), a generated architecture is accepted because it attains high in-distribution accuracy; neither branch certifies that the stated design rationale (component-contribution scores, or LLM-authored fact rows with stated meanings) monotonically explains behavioral accuracy under distribution shift. The structural root cause is that both branches treat rationale and behavior as two independently-produced outputs of the same generative process, so no mechanism exists that could detect a rationale whose causal story inverts under shift — a non-monotone rationale is silently inherited by the deployed architecture.

## Key Insight

Monotone consistency between a stated rationale and measured accuracy under controlled shift is a testable invariant because rationale edits (row deletion, component contribution ablation) are interventions on the same sparse-table/component surface that behavior is measured on, so a monotone critic trained on these intervention pairs can reject or repair architectures whose rationale inverts before deployment — a property no generation-only pipeline can establish.

## Method

```pseudocode
INPUT: architecture A with rationale R = {(row_i, meaning_i)} (WSP fact rows) or {(component_j, contribution_j)} (prior-knowledge branch); shift suite S = {s_1..s_K} (controlled: fact-paraphrase, row-permutation, held-out fact-subset, component-ablation); monotone critic C_theta (2-layer MLP, hidden=256, GeLU, input dim = |phi(R)|, output scalar)

PHASE 0 — Probe calibration and redundancy/joint-cause screening
  # Explicit assumption A: deleting a rationale row/component is a valid intervention on the same causal surface the rationale describes, and phi(R) is a sufficient statistic for rationale content. A fails when rows are redundant (deletion reflects redundancy, not causal necessity) or jointly cause non-monotonicity.
  probe_size <- 512 examples (stratified by fact-subset; paired bootstrap CIs, 1000 resamples)
  min_detectable_delta <- power analysis at alpha=0.05, power=0.8 on paired bootstrap -> report value (e.g., ~4.4% at p=0.5 for 128; recompute for 512)
  # Redundancy detection: pairwise row-ablation correlation
  for each pair (row_i, row_j) in R:
    delta_i <- acc(A, R \ {row_i}) - acc(A, R)
    delta_j <- acc(A, R \ {row_j}) - acc(A, R)
    delta_ij <- acc(A, R \ {row_i, row_j}) - acc(A, R)
    if |delta_ij - (delta_i + delta_j)| < min_detectable_delta and both |delta_i|,|delta_j| < min_detectable_delta:
      flag (row_i, row_j) as REDUNDANT; exclude both from monotone-ordering subset
    if |delta_ij| > min_detectable_delta and |delta_i|,|delta_j| < min_detectable_delta:
      flag (row_i, row_j) as JOINT-CAUSE; record for repair (neither alone is load-bearing)
  # Null calibration: delete a semantically inert/duplicate row (matched null) to estimate redundancy baseline
  null_delta <- acc(A, R \ {inert_row}) - acc(A, R)
  load_bearing_rows <- {row_i : |delta_i - null_delta| > min_detectable_delta}

PHASE 1 — Rationale-to-behavior intervention pairs (restricted to load_bearing_rows)
  for each shift s_k in S:
    R_k <- apply_intervention(R, s_k)          # delete row / ablate component / paraphrase meaning
    acc_k <- evaluate(A, R_k, heldout_probe of 512 examples)    # behavioral accuracy on probe
    record pair (phi(R_k), acc_k)              # phi = rationale feature vector (row count, meaning-embedding centroid, contribution vector)
  # Only pairs where the intervention targets a load_bearing_row (or a JOINT-CAUSE pair) enter the monotone-ordering training set; redundant-row pairs are held out as a diagnostic subset.

PHASE 2 — Train monotone consistency critic
  fit C_theta with monotonicity constraint: C_theta(phi(R_a)) <= C_theta(phi(R_b)) whenever R_a is a rationale-subset of R_b
  loss = ranking loss over pairs + monotonicity penalty (projection onto monotone cone, isotonic regression over phi ordering)
  hyperparameters: 200 epochs, lr 1e-3, monotonicity weight 0.5, batch size 32, Adam optimizer

PHASE 3 — Certification and repair
  for each A:
    if C_theta(phi(R)) violates monotone ordering against measured acc under any s_k (with paired bootstrap CI excluding zero):
      mark NON-MONOTONE; repair by removing the offending row/component (smallest deletion that restores monotone ordering); if offending rows are a JOINT-CAUSE pair, remove both jointly
    else: CERTIFY
OUTPUT: certified architectures + repaired rationales
```

**Why this design.** We chose a monotone-constrained critic over an unconstrained accuracy predictor because the invariant we need is ordering, not point accuracy — an unconstrained regressor (the obvious alternative, and what PO-NAS-style surrogate models already provide, venue:neurips:2025:b7fb7d1e808d) can fit the pairs while still inverting under shift, accepting the cost that a monotone critic is less expressive and may underfit non-monotone-but-valid rationales. We chose controlled rationale interventions (row deletion, component ablation, meaning paraphrase) over naturally-occurring distribution shifts because interventions give paired counterfactuals on the same surface, accepting the cost that the shift suite must be hand-specified per architecture family. We chose deletion-based repair over full regeneration because it preserves the LLM's authored facts, accepting the cost that some non-monotone architectures cannot be repaired by deletion alone. A domain expert would not describe this as a variant of PO-NAS or of the WSP paper: PO-NAS optimizes metric combinations to predict accuracy; WSP generates architectures; neither constructs the rationale-to-behavior monotone ordering as a certification object, and the critic here is trained on interventions on the rationale itself, not on architecture descriptors.

**Why it measures what we claim.** The critic's monotone ordering C_theta(phi(R_a)) <= C_theta(phi(R_b)) for rationale-subsets measures causal monotone consistency between stated rationale and behavioral accuracy because we assume (A) that deleting a rationale row/component is a valid intervention on the same causal surface the rationale claims to describe (the sparse table row or the component contribution) and that phi(R) is a sufficient statistic for rationale content; this assumption fails when a row's meaning is redundant with another row (deletion reflects redundancy rather than causal necessity) or when two rows jointly cause non-monotonicity (neither alone is load-bearing), in which case the critic's ordering reflects row-count or redundancy rather than rationale faithfulness — which is why Phase 0 screens redundant and joint-cause rows before training. The paired accuracy acc_k under shift s_k measures behavioral sensitivity to rationale edits because we assume the held-out probe is exchangeable with the deployment distribution up to the controlled shift; this assumption fails when the probe is too small and accuracy differences fall within noise, in which case the critic fits noise rather than monotone structure — which is why Phase 0 calibrates the probe to 512 examples with paired bootstrap CIs and reports the minimum detectable delta. The repair step's smallest-deletion criterion measures rationale minimality because we assume the monotone ordering is restored by removing the causally offending row; this assumption fails when two rows jointly cause non-monotonicity, in which case repair removes one and leaves the other, certifying a partially-inverted rationale — which is why Phase 0 flags JOINT-CAUSE pairs and Phase 3 removes both jointly.

## Contribution

['A monotone consistency critic (MCC) that takes a design rationale — LLM-authored fact rows with stated meanings or component-contribution scores — and predicts behavioral accuracy under a controlled shift suite, with a monotonicity constraint enforced by isotonic projection.', 'A certification-and-repair procedure that rejects architectures whose rationale-to-behavior mapping is non-monotone and repairs them by minimal rationale deletion, bridging prior-knowledge component contributions and weight-space interpretable fact rows on a shared intervention surface.', 'A controlled shift suite (fact-paraphrase, row-permutation, held-out fact-subset, component-ablation) that operationalizes rationale interventions for both branches as paired counterfactuals.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | ARC multiple-choice probe set (held-out 512-example probe, stratified by fact-subset; paired bootstrap CIs, 1000 resamples) | Matches WSP's fact-table surface and probe regime; 512 examples gives minimum detectable delta reported by Phase 0 power analysis |
| Dataset (secondary) | Prior-knowledge NAS component-contribution rationale (venue:icml:2025:d5970b220266) on CIFAR-10 search space | Tests the certification object on a non-WSP architecture family |
| Primary metric | Monotone-ordering violation rate (fraction of rationale-subset pairs where critic inverts measured accuracy, with paired bootstrap CI excluding zero) | Directly measures the invariant claimed, not point accuracy |
| Secondary metric | Fraction of NON-MONOTONE cases converted to CERTIFY by repair | Measures repair sub-claim |
| Diagnostic metric | Violation rate on redundant-row subset vs. single-unique-row subset | Tests whether the critic tracks causal necessity rather than row-count artifacts |
| Baseline | PO-NAS-style unconstrained surrogate (venue:neurips:2025:b7fb7d1e808d) | Tests whether monotone constraint, not surrogate capacity, drives certification |
| Baseline | Unconstrained accuracy regressor on phi(R_k) pairs (2-layer MLP, hidden=256, GeLU, same input dim as C_theta) | Tests the "ordering vs point accuracy" sub-claim directly |
| Baseline | No-repair certification (certify all, no deletion step) | Isolates contribution of deletion-based repair to certified-rationale quality |
| Ablation-of-ours | Monotone critic without monotonicity penalty (weight=0) | Isolates the monotone cone projection from the ranking loss |
| Ablation-of-ours | Monotone critic without Phase 0 redundancy/joint-cause screening | Tests whether screening removes row-count artifacts |

### Why this setup validates the claim

The central claim is that a monotone-constrained critic, trained on controlled interventions on the rationale itself, can certify whether an architecture's stated rationale is causally consistent with its behavior under shift — and that this ordering invariant is what distinguishes it from an accuracy predictor. The ARC fact-table surface is the right testbed because WSP architectures expose explicit row-to-meaning rationales (E6, E12), so row deletion and meaning paraphrase are genuine interventions on the same causal surface the rationale describes. The primary metric is ordering-violation rate rather than accuracy because a point-accuracy predictor can fit the intervention pairs while still inverting under shift; only an ordering metric detects that failure. The PO-NAS-style surrogate tests whether the gain comes from monotonicity rather than surrogate capacity, the unconstrained regressor tests the ordering-vs-accuracy sub-claim, and the no-repair baseline plus the zero-monotonicity ablation isolate the two mechanism components (cone projection, deletion repair). The Phase 0 screening ablation tests whether the redundancy/joint-cause detection is necessary to avoid row-count artifacts. The secondary prior-knowledge NAS dataset tests whether the certification object generalizes beyond the WSP surface. Together these form a falsifiable test: if monotonicity is doing the work, violation rate should drop sharply for our critic but not for either unconstrained baseline, and repair should convert a measurable fraction of NON-MONOTONE cases into CERTIFY.

### Expected outcome and causal chain

**vs. PO-NAS-style unconstrained surrogate** — On a case where a rationale row is redundant with another row (so deleting either leaves accuracy flat), the PO-NAS-style surrogate fits both deletion pairs to near-identical predicted scores because it optimizes global metric combinations for point accuracy and has no ordering constraint, so it cannot flag the row as non-essential and will certify a rationale whose stated meaning over-attributes causal weight. Our method instead screens redundant rows in Phase 0 and trains on the same pairs but projects onto the monotone cone, so the redundant-row pair is forced to a near-tie in ordering and the critic flags the row as non-monotone-relevant. We expect a noticeable gap in violation rate on the redundant-row subset but parity on the single-unique-row subset.

**vs. Unconstrained accuracy regressor on phi(R_k)** — On a case where a meaning paraphrase leaves accuracy unchanged but changes the meaning-embedding centroid (row count fixed), the unconstrained regressor produces a point prediction that can drift with the centroid even though measured accuracy is flat, inverting the ordering between the paraphrased and original rationale. Our method's monotonicity penalty ties same-row-count rationales to a consistent ordering, so paraphrase-induced centroid drift cannot invert the pair. We expect a concentrated gain on the fact-paraphrase shift subset and near-parity on the row-permutation subset where centroids barely move.

**vs. No-repair certification** — On a case where two rows jointly cause non-monotonicity (neither alone suffices), no-repair certification marks the architecture NON-MONOTONE but leaves both offending rows in place, so the final certified set either excludes a repairable architecture or includes a partially-inverted rationale. Our deletion-based repair removes the smallest offending row (or both jointly if flagged as JOINT-CAUSE in Phase 0) and restores ordering, converting a fraction of NON-MONOTONE cases to CERTIFY. We expect repair to recover a clear fraction of otherwise-rejected architectures, with the recovery concentrated on cases where one row carries most of the contribution vector mass.

**vs. Ablation: no Phase 0 screening** — Without redundancy/joint-cause screening, the critic trains on redundant-row pairs and its ordering reflects row-count rather than causal necessity, so violation rate on the redundant-row subset rises to match the unconstrained baselines. We expect the screening ablation to show a clear gap on the redundant-row subset, confirming that Phase 0 is necessary to close assumption A.

### What would falsify this idea

If the monotone critic's violation-rate advantage over the unconstrained regressor is uniform across all shift subsets rather than concentrated on the redundant-row and paraphrase subsets where the ordering mechanism is predicted to matter, then monotonicity is not the causal driver and the central claim is wrong. Likewise, if deletion-based repair converts essentially no NON-MONOTONE cases to CERTIFY, the repair sub-claim fails.

## References

1. Weight-Space Programming Doesn't Need Transformers
