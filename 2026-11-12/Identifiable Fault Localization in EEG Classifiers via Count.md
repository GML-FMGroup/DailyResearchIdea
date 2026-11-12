# Identifiable Fault Localization in EEG Classifiers via Counterfactual Minimal-Edit Certification

## Motivation

Weight-space debugging of biosignal classifiers, as introduced in Mixed Signals, asks a language model to locate and repair a planted fault by inspecting weights, but it never establishes that the located sub-block is the responsible one — any edit that restores accuracy is accepted, so a large edit masking the true fault is indistinguishable from the minimal cause. This is a structural failure, not a capability failure: the field inherited weight-editing validation from factual-recall work where a counterfactual probe (swap the subject entity, check the fact flips) is well-defined, whereas EEG classification has no agreed probe protocol, so 'the minimal responsible sub-block' is never defined as a testable object. The same gap recurs across the locating-and-editing branch: minimality is asserted, never certified against alternatives, and probes are never required to hold under subject shift.

## Key Insight

A fault sub-block is only identifiable if it is the unique sparsest weight edit that restores correct classification under a channel-and-time counterfactual probe battery that is itself invariant to subject-level distribution shift, so minimality becomes a checkable uniqueness property of an edit-support rather than a byproduct of whatever repair the language model happens to write.

## Method

**What it is.** MINER (Minimal Identifiable Neural-fault Edit Resolution) takes a trained EEG classifier, a set of failing trials, and a language-model proposer, and returns a certified fault sub-block: an edit support S over weight coordinates together with a certificate that S is the unique sparsest support restoring correct classification under the probe battery. **Load-bearing assumption (stated explicitly):** the planted fault is confined to a sparse set of weight coordinates such that the true fault support is the unique sparsest sufficient edit restoring correct classification under the probe battery — i.e., the sparsest sufficient edit coincides with the causal fault rather than being an arbitrary member of a large equivalence class of sufficient edits. This assumption is not asserted; Phase 0 measures how often it holds on the benchmark, and Phase 4 reports the full equivalence class of sufficient supports rather than claiming a unique one when the assumption fails.

**How it works.**

```pseudocode
INPUT: trained net f_theta (EEG classifier), failing set D_fail, held-out subject pool P
HYPER: sparsity budget k (swept 1..K, K=8), probe battery size m=64, uniqueness margin delta=0.05,
       Phase-0 exhaustive-search net size n_small <= 2000 weight coordinates

PHASE 0 - ASSUMPTION CALIBRATION (sparsity-to-causality check)
  # On small nets (n_small <= 2000 coords) where exhaustive search is tractable:
  for each planted fault F (known support S_F, |S_F| in 1..8):
    enumerate all supports S with |S| <= |S_F| by exhaustive search over the lattice
    solve Phase 3 constraint for each S; record feasible supports with e(S)=0 on B
    record: (a) is S_F feasible? (b) is S_F the unique sparsest feasible support?
            (c) if not, what is the smallest feasible support S_min and |S_min| vs |S_F|?
  -> report fraction of planted faults where S_F is the unique sparsest sufficient edit
     (assumption holds), where S_F is feasible but not sparsest (assumption fails by
     smaller non-causal edit), and where S_F is not feasible at all (fault not sparse)
  -> this fraction is the calibration of the load-bearing assumption and is reported
     alongside all downstream F1 numbers; downstream trials are stratified by it

PHASE 1 - PROBE BATTERY CONSTRUCTION (subject-invariant)
  for each trial x in D_fail:
    for r in 1..m:
      sample channel subset C_r ~ contiguous scalp neighbourhood
      sample time window W_r ~ random contiguous 200ms span
      probe T_r(x) = x with channels C_r zeroed AND window W_r mean-substituted
  keep only probes where f_theta(T_r(x)) != f_theta(x)   # probe is discriminative
  -> battery B = {(x, T_r(x), y)} with |B| = m_valid per trial

PHASE 2 - LLM PROPOSES CANDIDATE SUPPORTS
  for k in 1..K:
    prompt LM with (architecture, layer names, per-layer weight statistics,
                   failing-trial logits, probe battery summary)
    -> candidate supports {S_1..S_n}, each |S_i| <= k, as coordinate lists

PHASE 3 - CONSTRAINT SOLVE (minimal edit per support)
  for each candidate S:
    solve min_{dS} ||dS||_0 s.t. S = supp(dS) and
      for all (x, T_r(x), y) in B:  argmax f_{theta+dS}(x) = y
                                   AND argmax f_{theta+dS}(T_r(x)) = y
    via projected gradient on the masked coordinates only, hard-threshold to k
    -> repaired net f_S, residual error e(S) on D_fail and on held-out subjects

PHASE 4 - UNIQUENESS CERTIFICATION (equivalence-class reporting)
  for each feasible S with e(S)=0:
    for each coordinate c in S:
      test S' = S \ {c}  (drop one)
      if S' also feasible -> S is NOT minimal, discard
    for each coordinate c not in S with |S|<k:
      test S'' = S union {c}; if S'' feasible with lower loss -> S not sparsest
  rank feasible supports by |S|; certify S* iff it is strictly sparsest
  and no other support of size |S*| achieves e=0 within margin delta
  # Redundancy-resolution rule: if two coordinates are redundant (either alone suffices),
  # the drop-one test reports both as non-necessary and discards S. To avoid discarding
  # the true cause, compute the minimal hitting set over drop-one failures: for each
  # redundant pair {c1,c2}, require the certified support to contain at least one of them;
  # the minimal hitting set is the smallest support consistent with all drop-one tests.
  # Report the full equivalence class E = {S : |S|=|S*|, e(S)=0} rather than claiming
  # S* is unique when |E|>1; the certificate states 'S* is in the minimal sufficient set
  # and no smaller support is sufficient' without asserting S* equals the planted cause.

PHASE 5 - SUBJECT-SHIFT VALIDITY GATE
  evaluate f_{S*} on held-out subjects P; require e(S*)=0 on P
  if fails -> return UNCERTIFIED with the shift-violating subject ids
OUTPUT: (S*, certificate, equivalence class E, or UNCERTIFIED)
```

**Why this design.** We chose an explicit combinatorial uniqueness check (Phase 4) over the Mixed Signals protocol of accepting any repair that restores accuracy, because restoring accuracy is satisfiable by many non-causal edits and only a uniqueness test over the support lattice distinguishes the true cause; the cost is that Phase 4 is exponential in k, so we bound k at K=8 and rely on the LM to keep the candidate list small. We chose a channel-and-window probe battery over input-level paraphrasing (the factual-recall analogue) because EEG has no paraphrase operator, and zeroing a scalp neighbourhood plus mean-substituting a time window is the only intervention whose semantics are defined on the signal; the cost is that these probes are coarser than entity swaps and can be non-discriminative, which is why Phase 1 filters to probes that flip the prediction. We chose to gate on held-out subjects (Phase 5) rather than certify on the failing set alone, because a support that is minimal only for the observed subjects may be fitting subject-specific noise; the cost is that some true faults will be returned UNCERTIFIED when the subject pool is small. We added Phase 0 because the load-bearing assumption — that the planted fault is the unique sparsest sufficient edit — is not guaranteed by overparameterized networks (permutation and scaling symmetries, distributed representations, and lottery-ticket-style many-sparse-subnetwork results all predict many sufficient supports), so we measure on small nets how often the sparsest sufficient edit coincides with the planted fault and stratify downstream results by that fraction. We added the minimal-hitting-set redundancy rule in Phase 4 because the drop-one test alone discards the true cause when two coordinates are redundant (either alone suffices); the hitting set is the smallest support consistent with all drop-one failures. We report the full equivalence class E rather than claiming a unique support when |E|>1, because the certificate should state 'S* is in the minimal sufficient set and no smaller support is sufficient' without asserting S* equals the planted cause. This is not a variant of Mixed Signals: that method's output is a repaired network and its evidence is restored accuracy, whereas MINER's output is a support with a uniqueness certificate (or an equivalence class) and its evidence is that no smaller support and no equal-size alternative support restores accuracy — a property Mixed Signals never computes.

**Why it measures what we claim.** The support size |S| measures fault minimality because the assumption that the planted fault is confined to a sparse weight sub-block makes the true cause the sparsest sufficient edit; this assumption fails when the fault is distributed (e.g., a shifted bias across many units), in which case |S| reflects the smallest sufficient approximation rather than the true cause. The drop-one test in Phase 4 measures causal necessity of each coordinate because removing a coordinate and re-solving tests whether the edit still restores classification without it; this fails when two coordinates are redundant (either alone suffices), in which case the test reports both as non-necessary and the support is discarded even though one of them is the cause — which is why we add the minimal-hitting-set rule. The equal-size alternative search measures uniqueness because if a second support of the same size also restores accuracy, the identification is ambiguous; this fails when the classifier has symmetry (permuted hidden units), in which case the metric reflects representational symmetry rather than genuine ambiguity, and we report the equivalence class E instead of a unique S*. The subject-shift gate measures out-of-distribution validity of the localization because a cause that is real should persist across subjects; this fails when the held-out subjects are too few to distinguish shift from noise, in which case the gate reflects sampling variance rather than causal validity. Phase 0 measures the fraction of planted faults for which the sparsest sufficient edit coincides with the planted cause because the identification claim rests on that coincidence; this fails when the fault is distributed or when a smaller non-causal edit restores accuracy on the probe battery, in which case downstream F1 reflects the size of the equivalence class rather than identification of the cause.

## Contribution

(1) A fault-localization framework that formulates weight-sub-block identification as a constrained sparsest-edit problem with an explicit uniqueness certificate, replacing the accept-any-repair protocol with a checkable minimality criterion. (2) A subject-invariant counterfactual probe protocol for EEG — channel-neighbourhood zeroing plus temporal-window mean substitution, filtered to prediction-flipping probes — that gives biosignal weight editing a well-defined intervention semantics analogous to entity-swap probes in factual recall. (3) An empirical design principle: localization should be certified on held-out subjects, and supports that pass the failing set but fail the subject-shift gate are systematically non-causal.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Synthetic-fault EEG benchmark with planted sparse weight faults (support size 1..8), subject-held-out split; plus a small-net subset (n_small <= 2000 weight coordinates) for Phase 0 exhaustive calibration | Ground-truth support enables minimality scoring; small-net subset makes the load-bearing assumption checkable by exhaustive search |
| Primary metric | Support recovery F1 against planted fault coordinates, stratified by Phase-0 assumption-holds fraction | Directly measures identification, not accuracy; stratification separates trials where the sparsest sufficient edit equals the planted cause from those where it does not |
| Secondary metric | Equivalence-class size |E| and fraction of trials where |E|>1 | Reports ambiguity honestly rather than hiding it behind a unique-support claim |
| Baseline | Mixed Signals accuracy-only repair | Tests whether uniqueness adds identification power |
| Baseline | Gradient-descent fine-tuning on failing set | Tests whether minimal sparse edit beats dense retraining |
| Baseline | Hand-crafted CSP+LDA pipeline | Tests whether the method localizes rather than re-engineers |
| Ablation-of-ours | MINER without Phase 4 uniqueness certification | Isolates the contribution of the certificate |
| Ablation-of-ours | MINER without Phase 0 calibration (assumption asserted, not measured) | Isolates the contribution of the assumption check |
| Ablation-of-ours | MINER without minimal-hitting-set redundancy rule | Isolates the contribution of the redundancy-resolution rule |

### Why this setup validates the claim

The central claim is that a certified unique sparsest support identifies the true fault cause better than any accuracy-restoring repair. This requires ground-truth fault coordinates, which only the synthetic-fault benchmark provides, so support-recovery F1 against planted coordinates is the right primary metric: it detects identification, whereas accuracy cannot distinguish a causal repair from a coincidental one. The Phase-0 small-net calibration is what makes the load-bearing assumption testable rather than asserted: on nets small enough for exhaustive search we measure the fraction of planted faults for which the sparsest sufficient edit coincides with the planted cause, and we stratify all downstream F1 by that fraction so a high F1 cannot be an artifact of trials where the assumption happens to hold. The equivalence-class size |E| is reported alongside F1 because the certificate now states 'S* is in the minimal sufficient set and no smaller support is sufficient' rather than asserting S* equals the planted cause; |E|>1 is the direct signature of the symmetry/redundancy failure mode. Each baseline tests a distinct sub-claim. Mixed Signals tests whether accepting any accuracy-restoring edit suffices — if it matches MINER on F1, uniqueness certification buys nothing. Gradient-descent fine-tuning tests whether minimality matters — if dense retraining recovers the fault coordinates equally well, sparsity is not doing identification work. CSP+LDA tests whether a non-neural pipeline accidentally localizes. The ablations are decisive: removing only the uniqueness certificate holds the LM proposer and constraint solve fixed, so any F1 gap is attributable to certification; removing only the Phase-0 calibration tests whether measuring the assumption changes reported F1; removing only the hitting-set rule tests whether redundancy resolution recovers the true cause on redundant-coordinate trials.

### Expected outcome and causal chain

**vs. Mixed Signals accuracy-only repair** — On a case where the planted fault is a sparse set of coordinates in one layer and several non-causal edits also restore accuracy on the failing trials, Mixed Signals returns the first accuracy-restoring repair it finds because its acceptance criterion is restored accuracy alone, with no lattice search over supports. Our method instead enumerates feasible supports and discards any that a drop-one or equal-size alternative test shows to be non-minimal or ambiguous, because Phase 4 certifies only strictly sparsest feasible supports and reports the equivalence class when |E|>1. So we expect a large F1 gap on trials with many accuracy-restoring decoys and near-parity on trials where the fault is the only sufficient edit — the gap concentrated exactly where decoy supports exist, and the gap reported separately for trials where Phase 0 says the assumption holds versus fails.

**vs. Gradient-descent fine-tuning on failing set** — On a case where the failing set is small and subject-correlated, dense fine-tuning spreads the correction across many weights because gradient descent has no sparsity pressure, so the resulting delta is dense and its largest coordinates need not coincide with the planted fault. Our method instead solves a masked projected-gradient problem restricted to the proposed support and hard-thresholds to k, so the edit is confined by construction. We expect fine-tuning to reach comparable accuracy but low support-recovery F1, while MINER shows high F1 with a modest accuracy cost on subjects where the fault is genuinely sparse (Phase-0 assumption-holds stratum).

**vs. Hand-crafted CSP+LDA pipeline** — On a case where the planted fault lives inside the neural classifier's weights, the CSP+LDA pipeline cannot localize it at all because it has no weight coordinates and re-derives features from scratch, so its output is a new classifier with no support to compare. Our method instead returns a support over the trained net's weight coordinates, which is directly comparable to the planted ground truth. We expect CSP+LDA to be unscoreable on support F1 and to show accuracy parity at best, confirming that localization is a distinct capability from classification performance.

**vs. MINER without Phase 4 uniqueness certification (ablation)** — This ablation holds the LM proposer and constraint solve fixed and removes only the drop-one, equal-size, and hitting-set tests, so it returns the first feasible support the LM proposes. We expect full MINER to beat it on F1 specifically on decoy-rich trials and on redundant-coordinate trials (where the hitting-set rule recovers the true cause that the ablation's first-feasible support misses), with parity on trials where the fault is the only sufficient edit.

**vs. MINER without Phase 0 calibration (ablation)** — This ablation asserts the load-bearing assumption instead of measuring it, so it reports a single F1 number without stratification. We expect the ablation's headline F1 to be inflated relative to the assumption-holds stratum of full MINER, because the ablation pools trials where the sparsest sufficient edit is a non-causal smaller edit with trials where it is the planted cause; the gap between the ablation's pooled F1 and full MINER's assumption-holds-stratum F1 is the direct measure of how much the assumption matters.

**vs. MINER without minimal-hitting-set redundancy rule (ablation)** — This ablation uses the plain drop-one test, which discards supports containing redundant coordinate pairs. We expect full MINER to recover the true cause on redundant-coordinate trials where the ablation returns UNCERTIFIED, with parity on trials without redundancy.

### What would falsify this idea

If MINER's support-recovery F1 advantage over the accuracy-only Mixed Signals baseline and over the no-certification ablation is uniform across all trials rather than concentrated on trials containing decoy supports, then uniqueness certification is not what drives identification and the central claim is wrong. Equivalently, if the ablation matches full MINER on F1 wherever decoys exist, the certificate adds no identification power; and if Phase 0 shows the sparsest sufficient edit coincides with the planted cause on only a small fraction of small-net faults, the load-bearing assumption is false on this benchmark and the identification claim must be restricted to the assumption-holds stratum.

## References

1. Mixed Signals: Weight-Space Programming for EEG and ECG
