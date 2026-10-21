# Routing-Sketch Verification for Speculative MoE Offloading

## Motivation

Specter's quantized draft predicts expert activations to drive prefetching, but it provides no runtime check that these predictions match the full-precision target's actual routing distribution; the speculative-sampling lineage treats draft-target agreement purely as a token acceptance-rate property, not as a distributional property over expert activations. Structurally, Specter's prefetch schedule is committed before verification, so any drift in the draft's expert activation distribution silently degrades prefetch accuracy and PCIe bandwidth utilization without being detected or corrected.

## Key Insight

The full-precision target's expert routing is a low-entropy, slowly drifting distribution that can be tracked with a fixed-size count-min sketch updated only on verified tokens, making a bounded-memory divergence test between draft-predicted and target-observed activation sets a sufficient statistic for deciding when to fall back to conservative prefetching.

## Method

**What it is.** Routing-Sketch Verification (RSV) is a runtime oracle that sits between Specter's quantized draft and its prefetch scheduler. Its inputs are the draft's predicted expert-activation set for each speculative step and a count-min sketch of the full-precision target's verified expert routing over a sliding window of recent tokens; its output is a binary decision per speculative step: keep the speculative prefetch schedule, or trigger fallback prefetch of the union of draft-predicted and sketch-estimated experts plus a draft-recalibration signal.

**How it works.**

```pseudocode
# State maintained across tokens
CMS = CountMinSketch(width=W, depth=D)   # W=2^14, D=4, conservative update
window = deque(maxlen=L)                # L=256 verified tokens
tau = 0.12                              # divergence threshold (calibrated on 512 held-out prompts)
eps = 0.01                              # support threshold for A_hat
collision_floor = 0.0                   # per-bucket expected collision mass, estimated from stationary null traces (see calibration below)

# Calibration of eps and tau against collision-noise null (offline, once per model)
# 1. Run Specter on 512 held-out prompts with no fallback; log draft-predicted and target-observed expert sets per step.
# 2. For each step, compute raw CMS estimate and collision-corrected estimate (raw - collision_floor) for every expert.
# 3. On stationary spans (no routing shift, detected by exact target frequency stability over 256-token windows), record Jaccard distance between A_pred and A_hat_raw and between A_pred and A_hat_corrected.
# 4. Set collision_floor per bucket to the 95th percentile of raw-minus-exact frequency over stationary spans.
# 5. Set eps to the smallest value such that the 95th percentile of Jaccard distance on stationary spans (using corrected estimates) is below tau; set tau to that 95th percentile plus 0.02 margin.

for each speculative step t:
    A_pred = draft.predict_expert_set(x_t)          # set of expert ids, |A_pred| <= k (k=2)
    A_hat  = {e : max(0, CMS.estimate(e) - collision_floor[e]) / |window| > eps} # collision-corrected sketch-estimated target support
    delta  = jaccard_distance(A_pred, A_hat)
    if delta > tau:
        A_fallback = A_pred UNION A_hat
        scheduler.prefetch(A_fallback)              # conservative prefetch
        draft.recalibrate(A_hat)                    # bias draft's routing prior toward A_hat
    else:
        scheduler.prefetch(A_pred)                  # Specter's original schedule
    tokens = target.verify(x_t)                     # full-precision forward pass
    for e in target.observed_experts(tokens):
        CMS.update(e)                               # conservative update, no decay
        window.append(e)
```

**Why this design.** We chose a count-min sketch over an exact hash map because the sketch gives O(1) update/query with fixed memory regardless of the number of experts, accepting the cost that collisions inflate estimates for rare experts — mitigated by conservative update, by subtracting a per-bucket collision floor estimated from stationary null traces, and by thresholding at eps=0.01 so only experts with non-negligible target probability enter A_hat. We chose Jaccard distance over KL divergence between routing distributions because Jaccard operates directly on the activation sets that the prefetcher consumes, accepting the cost that it ignores probability mass ordering within the set; a distributional divergence would require storing full routing probabilities, which the sketch does not provide. We chose to update the sketch only on verified (target) tokens rather than on draft tokens because draft routing is the quantity under test, accepting the cost that the sketch lags by one speculative step; this lag is bounded by the multi-step lookahead depth (typically 4–8 tokens) and is why the window L=256 spans many steps. We chose a hard threshold tau with a fallback union rather than a soft blending of schedules because the prefetcher must commit to a fixed set per step, accepting the cost that a poorly calibrated tau causes either over-prefetch (bandwidth waste) or under-prefetch (stalls); tau is calibrated once per model on held-out prompts against a null distribution of Jaccard distance on stationary traces, so the trigger fires on divergence above the collision-noise floor rather than on absolute sketch mass. A domain expert would not describe RSV as a variant of Specter because Specter's draft prediction is open-loop — it never compares its predicted routing against the target's realized routing — whereas RSV closes that loop with a bounded-memory distributional test, and it is not a variant of speculative decoding acceptance tests because those operate on token identities, not on the expert-activation sets that drive I/O.

**Why it measures what we claim.** The quantity `max(0, CMS.estimate(e) - collision_floor[e]) / |window|` measures the target's recent expert-activation probability because the count-min sketch with conservative update is an unbiased-on-average estimator of item frequency under the assumption that hash collisions are uniformly distributed across experts, and subtracting the per-bucket collision floor estimated from stationary null traces corrects for the systematic overestimate that arises when a hot expert collides with a cold one; this assumption fails when the expert id space is skewed such that a hot expert collides with a cold one, in which case the raw estimate reflects an upper bound contaminated by the hot expert's mass rather than the cold expert's true probability, and the corrected estimate reduces but does not eliminate this contamination. The quantity `jaccard_distance(A_pred, A_hat)` measures distributional consistency between draft and target expert routing because A_hat is a thresholded support estimate of the target's recent routing and A_pred is the draft's support prediction, so their set disagreement is a direct operationalization of support mismatch under the assumption that the sliding window L=256 is long enough for the target routing to be stationary; this assumption fails when the input distribution shifts abruptly (e.g., a topic change mid-generation), in which case the distance reflects staleness of the window rather than draft-target divergence. The fallback union `A_pred UNION A_hat` measures conservative prefetch coverage because it includes every expert that either the draft predicts or the sketch estimates as non-negligible, under the assumption that the true next-step expert lies in one of these two sets; this assumption fails when both draft and sketch miss a rare expert, in which case the fallback still stalls — which is why the sketch is updated on every verified token to keep A_hat current. The quantity `delta` measures prefetch-miss risk because support-set disagreement is monotone in the probability that the true next-step expert lies outside A_pred, under the assumption that any expert with non-negligible target probability is included in A_hat; this assumption fails when a high-mass expert just below eps=0.01 is dropped from A_hat yet frequently activated, making Jaccard low while miss risk is high, in which case delta reflects support-set agreement rather than actual miss probability.

## Contribution

['Routing-Sketch Verification (RSV), a bounded-memory runtime oracle that tests draft-target expert-activation distributional consistency using a count-min sketch of verified target routing and triggers fallback prefetch plus draft recalibration on divergence.', 'A design principle for speculative MoE offloading: prefetch schedules should be gated by a distributional consistency test over expert activations rather than committed open-loop from draft predictions, with the sketch serving as the sufficient statistic.', 'A calibration procedure for the divergence threshold tau that ties sketch-estimated support to prefetch bandwidth budget, enabling per-model tuning without full routing-distribution storage.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Mixed-domain generation traces (code, math, open QA) | Exposes abrupt routing shifts RSV must track |
| Primary metric | Time per output token (TPOT) | Directly reflects prefetch coverage and stalls |
| Baseline | Specter (quantized draft, open-loop prefetch) | Tests whether closing the routing loop helps |
| Baseline | SOTA expert-offloading without speculation | Tests gain vs. non-speculative offloading floor |
| Baseline | Specter + oracle-union prefetch (upper bound) | Tests how close RSV gets to perfect coverage |
| Baseline | Always-prefetch-union heuristic (draft-predicted ∪ last-step-observed experts, no sketch, no divergence test) | Tests whether the sketch/divergence machinery is justified over a trivial union heuristic |
| Ablation-of-ours | RSV with tau=1 (never falls back) | Isolates contribution of the divergence trigger |
| Ablation-of-ours | RSV with raw CMS estimates (no collision-floor correction) | Isolates contribution of collision correction to the trigger |
| Measurement | CMS estimate vs. exact empirical expert frequency over the same window | Establishes estimator validity and collision-contamination rate for cold experts |
| Measurement | Correlation between delta and realized prefetch-miss rate | Tests whether Jaccard distance is monotone in miss risk |

### Why this setup validates the claim

The central claim is that an open-loop draft will periodically mispredict expert activation, and that a bounded-memory test against realized target routing can catch those mispredictions and convert them into conservative prefetch that avoids stalls. The open-loop Specter baseline directly tests whether the loop-closing mechanism matters: if RSV's trigger never fires usefully, TPOT should match Specter. The non-speculative offloading baseline establishes the floor, so we can tell whether any speculative gain survives; the oracle-union baseline establishes the ceiling, so we can tell whether RSV's sketch-based estimate is close enough to perfect coverage to be worth its overhead. The always-prefetch-union heuristic tests whether the sketch and divergence test add value beyond a trivial union of draft-predicted and last-step-observed experts; if RSV does not beat it, the sketch/divergence machinery is unjustified novelty. The tau=1 ablation is diagnostic: it disables fallback entirely, so any TPOT improvement over it must come from the divergence trigger rather than from the sketch's mere presence. The raw-CMS ablation isolates the contribution of collision correction: if RSV with raw estimates matches RSV with corrected estimates, the collision floor is unnecessary. The two measurement experiments directly validate the load-bearing assumptions: comparing CMS estimates to exact frequencies establishes whether the sketch is a faithful proxy for target routing probability, and correlating delta with realized prefetch-miss rate tests whether Jaccard distance is monotone in miss risk. TPOT is the right metric because the failure mode RSV targets is a prefetch miss that serializes an expert load, which manifests as a per-token latency spike, not as a throughput or accuracy change; a token-identity metric like acceptance rate would be blind to it. If RSV's advantage is real, it should appear as TPOT reductions concentrated on tokens following routing shifts, with parity on stationary stretches. We also report the fraction of total TPOT that prefetch misses can possibly account for (oracle-union vs. non-speculative floor) to quantify the ceiling of the idea.

### Expected outcome and causal chain

**vs. Specter (open-loop quantized draft)** — On a case where the prompt shifts from code to prose mid-generation, the draft's routing prior, shaped by earlier code tokens, keeps predicting code-specialized experts while the target activates prose experts; Specter prefetches the wrong set and stalls on the miss. RSV instead sees the sketch's A_hat drift toward prose experts, the Jaccard distance crosses tau, and it prefetches the union plus recalibrates the draft. We expect a visible TPOT gap concentrated on post-shift tokens, near parity on stationary spans.

**vs. SOTA expert-offloading without speculation** — On a case with a long tail of moderately frequent experts, the non-speculative baseline pays a load latency on every expert switch because it has no lookahead at all. RSV's union prefetch, even when the draft is wrong, still covers the sketch-estimated support, so most switches hit prefetched experts. We expect RSV to beat this baseline broadly but with the largest margin on high-switch-rate segments, and to lose slightly on very stable segments where speculative overhead is pure cost.

**vs. Specter + oracle-union prefetch** — On a case where both draft and sketch miss a rare expert, the oracle baseline prefetches it anyway because it sees the true next-step set, while RSV stalls. This baseline is the ceiling, so we expect RSV to trail it by a small margin overall, with the gap widening on rare-expert-heavy subsets and nearly closing on common-expert subsets — confirming the sketch's collision-inflated estimates are the residual bottleneck rather than the trigger logic.

**vs. Always-prefetch-union heuristic (draft-predicted ∪ last-step-observed experts, no sketch, no divergence test)** — On stationary spans, the heuristic prefetches a slightly larger set than RSV (because it always includes last-step-observed experts even when the draft is confident), so we expect RSV to match or slightly beat it on TPOT due to lower bandwidth waste. On shift-heavy traces, the heuristic's last-step-observed experts lag the true shift by one step, so it misses the new experts for one step while RSV's sketch-estimated A_hat captures the drift over the 256-token window and triggers fallback earlier. We expect RSV to show a TPOT advantage concentrated on post-shift tokens; if RSV does not beat this heuristic, the sketch/divergence machinery is unjustified.

**vs. RSV with tau=1 (no fallback)** — On the same mid-generation shift case, the ablation never fires the fallback because the threshold is unreachable, so it behaves like Specter and stalls on the shift. This isolates the trigger: if the ablation matches Specter rather than RSV, the gain is attributable to fallback decisions, not to sketch bookkeeping. We expect the ablation to sit close to Specter's TPOT, with the RSV-minus-ablation gap largest exactly on shift-heavy traces.

**vs. RSV with raw CMS estimates (no collision-floor correction)** — On traces with a skewed expert distribution, raw CMS estimates inflate A_hat for cold experts due to collisions with hot experts, biasing Jaccard distance upward and causing the trigger to fire more often than necessary. We expect the raw-CMS ablation to show higher fallback rates and slightly worse TPOT than RSV with correction, with the gap largest on rare-expert-heavy subsets; if the gap is negligible, the collision floor is unnecessary.

**Measurement: CMS estimate vs. exact empirical expert frequency** — Over the same 256-token window, we compute per-expert relative error of the corrected CMS estimate versus exact frequency, and report the collision-contamination rate (fraction of cold experts whose raw estimate exceeds eps solely due to collisions). We expect relative error to be low for hot experts and higher for cold experts, with the correction reducing contamination; this establishes the estimator's validity rather than asserting it.

**Measurement: correlation between delta and realized prefetch-miss rate** — For each speculative step, we record delta and whether the true next-step expert set was covered by the prefetched set (miss indicator). We expect a positive correlation (higher delta associated with higher miss rate), supporting the claim that Jaccard distance is monotone in miss risk; a weak correlation would indicate that delta is not a faithful proxy for miss risk.

### What would falsify this idea

If RSV's TPOT improvement over open-loop Specter is uniform across stationary and shift-heavy traces rather than concentrated on post-shift tokens, the divergence trigger is not doing the causal work we claim and the gain is just prefetch overhead amortization. Equally fatal: if the tau=1 ablation already matches RSV, the fallback decision — the core of the method — is inert. Additionally, if the CMS estimate shows high relative error for cold experts even after correction, or if delta is uncorrelated with realized prefetch-miss rate, the load-bearing assumptions fail and the method's claimed sufficient-statistic property is undermined.

## References

1. Specter: Fast MoE Inference via Quantization-Enabled Multi-Step Speculative Offloading
