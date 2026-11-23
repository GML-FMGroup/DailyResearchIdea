# Identifiability-Gated Recursive Repair for Vision-Language Navigation

## Motivation

Recursive self-improvement frameworks such as AeroRSI assume that the failure-repair mapping estimated in each round remains valid across rounds, but scene adaptation (as studied in GSA-VLN/GR-DUET) shifts the failure distribution so that repairs distilled in round t may target failures that no longer exist or may conflate distinct latent causes; the structural root cause is that the repair sampler is never checked for whether the observed failure-repair pairs actually identify a unique latent cause under the current policy, so recursive distillation can amplify non-identifiable repairs and degrade behavior elsewhere.

## Key Insight

A failure-repair mapping is only safe to distill when the observed continuation-verified repairs are conditionally sufficient to identify the latent failure cause, and this identifiability can be tested by comparing the predictive distribution of repairs under the current policy against a scene-adapted reference via a two-sample stationarity test on a held-out probe set.

## Method

**What it is**: IG-RSI (Identifiability-Gated Recursive Self-Improvement) wraps AeroRSI's recursive repair loop with a per-round failure-manifold identifiability monitor and an identifiability-constrained repair sampler. Inputs: current policy π_t, a probe set of scene-adapted trajectories, and candidate repairs from the rule-based sampler. Outputs: a gated set of repairs that are distilled into π_{t+1}, plus a per-repair identifiability score and a round-level stationarity flag.

**Load-bearing assumption (stated explicitly)**: The gate treats a failure cluster as non-identifiable when the conditional repair distribution p_t(r|c) differs from a frozen scene-adapted reference p_ref(r|c) by more than τ=0.05 in MMD^2. This is only valid under assumption A: the probe set is exchangeable with the current round's failures, and the repair-outcome embedding is a sufficient statistic for the latent failure cause Z. When A holds, MMD^2(p_t, p_ref) measures policy-induced failure-manifold drift; when A fails (stale probe set, embedding collapse), MMD^2 reflects probe-set staleness or noise instead. We make A explicit and add a diagnostic (step 4b) that flags when A is violated, so the gate's decision is reported together with its calibration status rather than asserted.

**How it works**:
```pseudocode
# Round t of IG-RSI
1. Roll out π_t on TravelUAV; collect failures F_t (anchor states + continuation outcomes)
2. For each failure f in F_t:
     sample K=8 candidate repairs r ~ RuleSampler(f)   # AeroRSI's rule-based proposals
     verify by continuation -> (f, r, success) triple
3. Build repair-outcome matrix M_t over F_t x {r}
4. Identifiability test per failure cluster c (clusters = k-means, k=8, on anchor-state embeddings):
     compute conditional repair distribution p_t(r | c) from M_t
     compare to reference p_ref(r | c) from scene-adapted probe set
     test statistic: two-sample MMD^2 on repair-outcome embeddings,
       RBF kernel, bandwidth = median pairwise distance on the probe set (fixed per round),
       threshold τ=0.05
     if MMD^2(p_t, p_ref) > τ: mark cluster NON-IDENTIFIABLE (drift detected)
4b. Assumption-A diagnostic (runs every round, reported alongside the gate):
     (i) probe-exchangeability check: train a logistic classifier (2-layer MLP, hidden=128)
         to separate probe-set failures from current-round failures on anchor-state embeddings;
         if held-out AUC > 0.7, flag PROBE-STALE (A violated)
     (ii) embedding-collapse check: if the effective rank of the repair-outcome embedding
         (participation ratio of singular values) < 2, flag EMBEDDING-COLLAPSE (A violated)
     if either flag is raised, the round's MMD decisions are logged as UNRELIABLE and
     the gate defaults to rejecting only clusters that also fail the bootstrap test in step 5
5. Identifiability-constrained sampler:
     for each cluster c, retain repairs whose outcome is stable across
     the 3 most recent rounds (bootstrap agreement ≥ 0.8, 200 bootstrap resamples of F_t)
     reject repairs that only appear in non-identifiable clusters
6. Distill retained repairs into π_{t+1} with AeroRSI's continuation loss,
   plus a behavior-preservation KL on non-failure states (weight 0.1)
7. Update reference p_ref <- p_t if and only if stationarity test passes
   (MMD^2(p_t, p_ref) ≤ τ on all clusters AND no Assumption-A flag raised in step 4b)
```

**Why this design**: We chose a two-sample MMD stationarity test on repair-outcome embeddings over a learned drift classifier because MMD gives a nonparametric, sample-efficient signal that does not require labels for 'drifted' failures, accepting the cost that it needs a held-out probe set and a kernel bandwidth choice (we fix the bandwidth to the median pairwise distance on the probe set, recomputed each round, so the threshold τ=0.05 is applied to a scale-normalized statistic). We chose to gate distillation at the cluster level rather than per-repair because individual repairs are too sparse to test identifiability reliably, accepting coarser rejection that may discard some good repairs alongside bad ones. We chose to update the reference distribution only when the stationarity test passes, rather than continuously, because continuous updating would let the reference track the drift and mask non-identifiability, accepting the cost that genuine scene adaptation is incorporated slowly. We added the Assumption-A diagnostic (step 4b) because the MMD test is only a drift measure under probe-exchangeability and non-collapsed embeddings; the diagnostic makes the gate's reliability explicit and prevents a stale probe set from being silently read as non-identifiability. Unlike AeroRSI, which reuses the same rule-based repair sampler every round without any stationarity check, and unlike GR-DUET, which assumes scene-adapted graphs stabilize, IG-RSI makes the identifiability of the failure-repair mapping an explicit, tested precondition for distillation.

**Why it measures what we claim**: The MMD^2 statistic between p_t(r|c) and p_ref(r|c) measures failure-manifold drift because it is a consistent estimator of the squared distance between the conditional repair distributions under assumption A (probe set exchangeable with current failures, repair-outcome embedding sufficient for the latent cause); this assumption fails when the probe set is stale (e.g., scene layout changed) or the embedding collapses, in which case MMD^2 reflects probe-set staleness or noise rather than policy-induced drift — step 4b's probe-exchangeability AUC and effective-rank checks are the diagnostics that flag exactly these failure modes. The bootstrap agreement ≥0.8 across three rounds measures repair identifiability because it operationalizes the assumption that a repair targeting a stable latent cause will produce consistent continuation outcomes across resampled failure instances; this assumption fails when a repair succeeds for spurious reasons (e.g., lucky initial state), in which case agreement reflects outcome variance rather than causal identifiability, and the 200-resample bootstrap gives an explicit variance estimate so the 0.8 threshold is applied to a distribution rather than a point estimate.

## Contribution

["IG-RSI: an identifiability-gated recursive self-improvement framework that adds a per-round failure-manifold stationarity test and an identifiability-constrained repair sampler to AeroRSI's continuation-verified repair loop.", 'A design principle: repair distillation in recursive VLN self-improvement should be gated on the identifiability of the failure-repair mapping under scene adaptation, not merely on continuation success.', 'An empirical protocol on TravelUAV measuring success rate, collisions, and a new non-identifiable-repair rate across recursive rounds.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | TravelUAV, 3 recursive rounds on a fixed 2,000-trajectory subset | Scoped round budget matches AeroRSI's reported 3-round setting |
| Primary metric | Success rate by round (per-round SR trajectory) | Directly detects gated distillation gains vs. drift |
| Secondary metric | Collision rate by round; per-cluster SR on drift-affected vs. stable clusters | Tests whether gate effect is concentrated where predicted |
| Calibration metric | Gate false-positive rate on stable clusters (Assumption-A diagnostic AUC, effective rank) | Required calibration check for the MMD gate |
| Baseline | AeroRSI (ungated RSI) | Tests whether identifiability gating beats unfiltered repair |
| Baseline | GR-DUET | Tests stationary scene-graph assumption against drift |
| Baseline | Initial supervised 2B policy | Anchors no-self-improvement floor |
| Ablation-of-ours | IG-RSI without MMD gate (constrained sampler only) | Isolates gate contribution from sampler constraint |
| Ablation-of-ours | IG-RSI without Assumption-A diagnostic (gate always trusted) | Isolates the diagnostic's contribution to gate reliability |

### Why this setup validates the claim

The central claim is that gating distillation on failure-manifold identifiability prevents recursive self-improvement from distilling repairs that are artifacts of drift, yielding more reliable round-over-round gains than ungated RSI. AeroRSI is the right primary baseline because it is the exact mechanism we wrap: if our gate adds value, the gap must appear as a divergence in per-round success trajectory, not merely a higher final number, since AeroRSI's own reported climb from 46.4% to 67.5% already establishes that ungated RSI improves. GR-DUET tests the orthogonal assumption that scene-adapted structure stabilizes without testing it, so it probes whether explicit stationarity checking matters beyond memory graphs. The no-gate ablation is the load-bearing control: removing only the MMD gate while keeping the constrained sampler tells us whether the gain comes from the identifiability test or from sampler filtering alone. The no-diagnostic ablation tests whether the Assumption-A check (probe-exchangeability AUC > 0.7, effective rank < 2) actually changes gate decisions, which is the concrete calibration story for the load-bearing assumption. Success rate by round is the correct metric because our predicted effect is a trajectory-shape effect (monotone, non-collapsing gains) rather than a single-endpoint effect; a uniform endpoint lift would not distinguish our causal story from generic regularization. The 3-round budget on a 2,000-trajectory subset is pre-registered to match AeroRSI's reported setting and to keep the gate-vs-no-gate comparison runnable within the timeline before scaling.

### Expected outcome and causal chain

**vs. AeroRSI (ungated RSI)** — On a case where a mid-training round encounters a cluster of failures whose apparent repair successes stem from a shifted scene layout, the baseline distills those repairs because its rule sampler treats every continuation success as valid supervision, so its success rate climbs then plateaus or dips as spurious repairs contaminate the policy. Our method instead flags that cluster as non-identifiable via the MMD test against the probe reference and rejects its repairs, so we expect a smaller round-one gain but a monotone, non-collapsing trajectory with the gap widening in later rounds, concentrated on the drift-affected failure clusters. The Assumption-A diagnostic is expected to raise no flag on this case (probe set exchangeable, effective rank ≥ 2), so the gate's decision is reported as reliable.

**vs. GR-DUET** — On a case where the environment's instruction style or layout drifts between rounds, the baseline's retained navigation graph encodes stale scene structure and its environment-specific training reinforces it, producing confident but wrong re-visited waypoints. Our method's stationarity test blocks reference updates until drift is confirmed benign, so the graph-derived prior is not silently corrupted; we expect GR-DUET to degrade on the drift subset while IG-RSI holds parity with its own earlier rounds, yielding a noticeable gap on drift-heavy trajectories but parity on stable ones.

**vs. Initial supervised 2B policy** — On a case where the base policy fails on a long-horizon subtask boundary, the baseline has no repair mechanism and simply repeats the same error, so its success rate is flat at 46.4% regardless of round. Our method converts those anchored failures into verified repairs and distills them, so we expect a clear round-over-round lift above the flat baseline, with the lift largest on subtask-boundary failures and near-zero on failures that the identifiability test rejects as non-stationary.

**vs. IG-RSI without MMD gate (constrained sampler only)** — On a case where the constrained sampler alone retains repairs that are stable across rounds but target a drifted latent cause, the ablation distills them because it has no distributional check against the reference, so its trajectory should match IG-RSI on stable clusters but diverge on drift-affected clusters. We expect the gap to be concentrated on drift-affected clusters, which is the falsification-relevant signature.

**vs. IG-RSI without Assumption-A diagnostic (gate always trusted)** — On a case where the probe set is stale (probe-exchangeability AUC > 0.7), the no-diagnostic variant reads probe staleness as non-identifiability and rejects good repairs, so its trajectory should underperform IG-RSI on stable clusters in exactly the rounds where the diagnostic would have flagged the probe set. We expect the diagnostic to reduce gate false positives on stable clusters, which is the calibration check for the load-bearing assumption.

### What would falsify this idea

If the MMD gate's benefit is uniform across all failure subsets rather than concentrated on the drift-affected clusters where our causal story predicts it, then the gate is acting as generic regularization rather than an identifiability mechanism and the central claim is wrong. Equally falsifying: if the no-gate ablation matches IG-RSI's round-over-round trajectory, the gate contributes nothing beyond the constrained sampler; and if the Assumption-A diagnostic never raises a flag on the stale-probe-set condition, the calibration story for the load-bearing assumption is unsupported.

## References

1. AeroRSI: Recursive Self-Improvement for Aerial Vision-Language Navigation
