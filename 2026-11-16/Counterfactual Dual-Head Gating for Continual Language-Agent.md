# Counterfactual Dual-Head Gating for Continual Language-Agent Checkpoints

## Motivation

Continual-learning systems for language agents such as G-CAST promote checkpoints through a governed transition gated by a single cumulative-evaluation scalar, so the gate cannot distinguish a model that has genuinely retained previously relied-upon behavior from one that has merely re-aligned its outputs to the new demand distribution. The alignment/retention distinction was established only under a synthesized controlled dataset (Spurious Forgetting in Continual Learning of Language Models, E3), and the production loop of G-CAST has no controlled counterfactual to carry that distinction forward, so retention is never certified and entropy-prioritized selection can mask knowledge loss as re-alignment.

## Key Insight

Paired counterfactual probes synthesized from the same production evidence isolate the causal effect of a checkpoint update on retained behavior, so a two-head estimator trained on those pairs yields an alignment/retention split that is identifiable rather than an unidentifiable decomposition of a single scalar, and the cumulative-evaluation gate can therefore certify retention as a decomposed invariant.

## Method

## CDG: Counterfactual Dual-Head Gating

**(A) What it is.** CDG is a checkpoint-gating layer that sits inside G-CAST's governed transition. Its inputs are a candidate checkpoint $\theta_t$, the incumbent checkpoint $\theta_{t-1}$, and a batch of production evidence $\mathcal{E}_t$; its output is a promotion decision plus a certified retention vector $r_t$ that is logged alongside the checkpoint.

**(B) How it works.**

```pseudocode
# Phase 1: paired counterfactual probe synthesis with verified intervention
# Load-bearing assumption (stated explicitly): the retention probe is valid only
# if c_retain is constructed by replaying the t-1 harness state, not by editing
# the current context. We therefore hold h fixed at its t-1 snapshot and certify
# the intervention before using any pair.
for each evidence item e in E_t:
    # e = (context c, harness state h, observed agent action a_obs)
    h_prev = snapshot_harness(t-1)          # old tool/skill surface, frozen
    c_retain = replay_context(c, h_prev)    # no new-capability tokens introduced
    c_align  = c                             # keep full current demand
    a_ret_prev = theta_{t-1}(c_retain, h_prev)
    a_ret_new  = theta_t(c_retain, h_prev)
    a_align_prev = theta_{t-1}(c_align, h)
    a_align_new  = theta_t(c_align, h)
    # mask-validity check: intervention is certified clean only if the incumbent
    # and candidate agree on c_retain under a no-op update (identity check)
    if distance(theta_{t-1}(c_retain, h_prev), theta_t(c_retain, h_prev)) > tau_noop:
        continue   # drop pair; do not train retention head on confounded pairs
    pair = (c_retain, a_ret_prev, a_ret_new, c_align, a_align_prev, a_align_new)
    push pair into P_t

# Phase 2: dual-head estimator training (frozen backbone, two linear heads)
# Backbone: frozen 12-layer transformer encoder (hidden=768, 110M params).
# Heads: two linear layers (768 -> 1) with sigmoid output.
for pair in P_t:
    # alignment head: predicts whether theta_t improved task alignment on c_align
    y_align = 1[ reward(c_align, a_align_new) > reward(c_align, a_align_prev) ]
    # retention head: predicts whether theta_t preserved behavior on c_retain
    y_ret   = 1[ distance(a_ret_new, a_ret_prev) < tau_ret ]   # tau_ret = 0.05
    z = frozen_encoder(c_retain, c_align)
    L = BCE(align_head(z), y_align) + BCE(ret_head(z), y_ret)
    update align_head, ret_head only   # backbone frozen, lr = 1e-4, batch=32

# Phase 3: decomposed retention invariant gate
# Clusters K_prev are initialized by k-means (k=8) on prior-checkpoint
# retained-behavior embeddings and fixed thereafter; membership is assigned
# by nearest centroid at each checkpoint.
align_hat, ret_hat = mean over held-out P_t of (align_head(z), ret_head(z))
for each retained-behavior cluster k in K_prev:      # clusters from prior checkpoints
    r_t[k] = ret_hat restricted to cluster k
    require r_t[k] >= rho_k                          # rho_k = 0.9 * r_{t-1}[k]
if all k satisfy r_t[k] >= rho_k and align_hat >= align_{t-1} - delta_align:
    promote theta_t; log r_t
else:
    reject theta_t; return to G-CAST selector

# Phase 4: selector conditioning
# G-CAST's entropy-prioritized selector receives (align_hat, r_t) and
# down-weights high-entropy samples whose cluster has r_t[k] < rho_k
```

**(C) Why this design.** We chose paired counterfactual probes over a single-head scalar regression because the alignment/retention split is not identifiable from a single cumulative-evaluation score — the same score can arise from genuine retention or from re-alignment — and the paired probe is the minimal intervention that breaks that degeneracy, accepting the cost that probe synthesis requires a maskable harness-capability surface. We chose a frozen backbone with two linear heads over end-to-end fine-tuning of a shared encoder because end-to-end training would let the heads co-adapt and re-introduce the coupling that the counterfactual pairs are meant to break, accepting the cost that the heads are lower-capacity and may underfit rare clusters. We chose a per-cluster retention invariant over a global retention threshold because G-CAST's harness evolution creates capability clusters whose retention rates differ, and a global threshold would let strong clusters mask weak ones, accepting the cost that cluster assignment must be maintained across checkpoints. We chose to condition the entropy-prioritized selector on the estimated split rather than to replace it because the selector's entropy signal is still useful for demand drift, but an unconditional selector can up-weight samples whose retention is already failing; conditioning is the minimal change that prevents entropy from masking knowledge loss. A domain expert would not describe CDG as a variant of G-CAST's cumulative-evaluation gate, nor as a variant of Spurious Forgetting's Freezing strategy, because neither prior mechanism constructs counterfactual probe pairs or decomposes the gate into a per-cluster retention invariant.

**(D) Why it measures what we claim.** The alignment-head output `align_hat` measures task-alignment drift because the paired probe holds the context fixed and varies only the checkpoint, so any change in the head's prediction is attributable to the update rather than to demand drift; this assumption fails when the mask does not fully remove new capabilities, in which case `align_hat` reflects capability mismatch rather than alignment drift. The retention-head output `ret_hat` measures retained-knowledge loss because the masked context `c_retain` contains only behaviors that existed before $t-1$, so behavioral distance on that context is a direct probe of retention; this assumption fails when the mask leaks new-capability tokens into `c_retain`, in which case `ret_hat` reflects capability interference rather than knowledge loss. The per-cluster retention vector `r_t[k]` measures certified retention for cluster $k$ because the cluster partition is fixed from prior checkpoints and the threshold $\rho_k$ is anchored to the incumbent's own rate, so a passing $r_t[k]$ is equivalent to no regression on that cluster; this assumption fails when cluster membership drifts between checkpoints, in which case `r_t[k]` reflects cluster reassignment rather than retention. The selector conditioning on `(align_hat, r_t)` measures retention-aware prioritization because down-weighting high-entropy samples in failing clusters is equivalent to preventing entropy from masking knowledge loss only if entropy and retention are conditionally independent given the cluster; this assumption fails when high-entropy samples are concentrated in exactly the clusters that are failing, in which case the conditioning reflects a confound rather than a safeguard.

**Mask-validity check (explicit).** The load-bearing assumption is that `c_retain` is a clean intervention. We enforce it by (i) constructing `c_retain` via harness replay (`h_prev`), not semantic masking, and (ii) dropping any pair where the no-op identity check fails (`distance > tau_noop = 0.01`). The fraction of dropped pairs is logged as `mask_leakage`; if `mask_leakage > 0.2`, the retention vector is reported as a bound, not an invariant.

## Contribution

["A counterfactual dual-head gating layer (CDG) that synthesizes paired probes from production evidence and produces an identifiable alignment/retention split, replacing G-CAST's single-scalar cumulative-evaluation gate with a decomposed per-cluster retention invariant.", "A selector-conditioning mechanism that feeds the estimated alignment/retention split back into G-CAST's entropy-prioritized selector, preventing entropy from masking knowledge loss as re-alignment.", 'A certified retention vector logged per checkpoint, enabling post-hoc audit of which retained behaviors were certified at each promotion.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | G-CAST e-commerce demand-drift stream | Matches harness evolution and demand drift target setting. |
| Dataset | ToolBench tool-use stream (public fallback) | Tests generalization beyond proprietary G-CAST harness. |
| Primary metric | Per-cluster retention pass rate at fixed alignment | Directly measures the decomposed invariant CDG claims. |
| Secondary metric | Mask leakage fraction (dropped pairs / total) | Quantifies intervention validity and bounds identifiability. |
| Baseline | G-CAST cumulative-evaluation gate | Tests whether decomposition beats a single scalar score. |
| Baseline | Spurious Forgetting Freezing strategy | Tests whether freezing alone suffices without counterfactual probes. |
| Baseline | Unconditional entropy selector | Tests whether selector conditioning prevents masked knowledge loss. |
| Ablation-of-ours | CDG with global retention threshold | Tests whether per-cluster invariants are necessary. |
| Ablation-of-ours | CDG with semantic mask (no harness replay) | Tests whether verified intervention is necessary. |

### Why this setup validates the claim

The central claim is that a single cumulative-evaluation score is degenerate between alignment gains and retention loss, and that only paired counterfactual probes plus per-cluster invariants can separate them. The G-CAST gate baseline tests the degeneracy claim directly: if its scalar score can be matched while retention silently degrades, CDG's decomposition is doing real work. The Freezing baseline tests whether the alignment/retention split is even needed, or whether simply freezing bottom layers suppresses forgetting; if freezing matches CDG on retention but loses alignment on drift, the split is justified. The unconditional selector baseline tests the conditioning sub-claim: entropy prioritization without retention awareness should up-weight samples in failing clusters. The global-threshold ablation isolates the per-cluster invariant, since a global threshold lets strong clusters mask weak ones. The semantic-mask ablation isolates the verified-intervention step: if it matches harness-replay CDG, the mask-validity check is unnecessary. Per-cluster retention pass rate at fixed alignment is the right primary metric because it is exactly the quantity CDG certifies; a method could look good on aggregate retention while failing specific clusters, and only a per-cluster pass rate detects that. The mask leakage fraction is the right secondary metric because it directly measures the load-bearing assumption's validity and lets us report the decomposition as a bound when leakage is high. Together these form a falsifiable test: CDG must beat the gate and selector on weak clusters while matching freezing on strong clusters, and its advantage must shrink as mask leakage rises.

### Expected outcome and causal chain

**vs. G-CAST cumulative-evaluation gate** — On a case where a new product category arrives and the candidate checkpoint re-aligns strongly on new demand while quietly regressing on an older procurement flow, the baseline promotes because its cumulative score rises, since alignment gains numerically swamp retention loss in a single scalar. Our method instead probes the replayed older-flow context and sees the retention head flag a per-cluster drop below the anchored threshold, so promotion is refused. We expect a noticeable promotion-precision gap concentrated on drift-heavy segments, with parity on stable segments, and the gap should widen as mask leakage decreases.

**vs. Spurious Forgetting Freezing strategy** — On a case where the harness introduces a new tool and the model must learn to invoke it, the freezing baseline preserves old behavior but cannot absorb the new capability, producing stale actions on new-capability contexts because its bottom-layer freeze blocks the update needed for alignment. Our method instead allows the update and certifies retention separately via the replayed probe, so it accepts the checkpoint when alignment improves and retention holds. We expect freezing to match CDG on retained-behavior clusters but lag on new-capability alignment clusters.

**vs. Unconditional entropy selector** — On a case where high-entropy samples cluster in a capability group already showing retention regression, the baseline up-weights them because entropy alone signals useful learning signal, amplifying updates that further erode that cluster. Our method instead conditions the selector on the per-cluster retention vector and down-weights those samples, so the failing cluster is not compounded. We expect the gap to appear as a retention pass-rate advantage specifically on the failing cluster, with similar overall entropy coverage elsewhere.

**vs. CDG with global retention threshold (ablation)** — On a case where one strong cluster masks a weak cluster under a global threshold, the ablation promotes a checkpoint that regresses on the weak cluster because the global average stays above threshold. Our method instead requires each cluster to meet its own anchored threshold, so the weak-cluster regression is detected. We expect the ablation to match CDG on strong clusters but show lower weak-cluster retention pass rate.

**vs. CDG with semantic mask (ablation)** — On a case where new-capability tokens leak into the masked context, the ablation trains the retention head on confounded pairs and misattributes capability interference to knowledge loss. Our method instead uses harness replay and drops pairs failing the no-op identity check, so the retention head sees only certified-clean pairs. We expect the ablation to show higher mask leakage and degraded alignment/retention separation, especially on drift-heavy segments.

### What would falsify this idea

If CDG's gain over the G-CAST gate is uniform across clusters rather than concentrated on the weak clusters the retention vector predicts, then the decomposition is not the cause and the central claim fails. Likewise, if the global-threshold ablation matches per-cluster CDG on weak-cluster retention, the per-cluster invariant is unnecessary; and if the semantic-mask ablation matches harness-replay CDG, the mask-validity check is unnecessary.

## References

1. G-CAST: Governed Continual Self-Training for E-Commerce Agents under Evolving Demands
