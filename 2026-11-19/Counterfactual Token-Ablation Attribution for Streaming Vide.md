# Counterfactual Token-Ablation Attribution for Streaming Video Proactive Decisions

## Motivation

Across streaming video understanding, compression and proactive triggering are optimized as separate branches: InfiniPot-V and StreamChat compress memory without ever asking which retained tokens caused a later response, while StreamBridge and LiveStar decide when to respond without attributing that decision to specific compressed evidence. ALIVE's RL stage inherits this split — BACA assigns credit at the trajectory and local level but never at the level of individual retained memory tokens, so a trigger can be reinforced without any record of which compressed evidence made it necessary. The structural root cause is that no branch defines a joint objective in which the retained-token set and the trigger decision are evaluated together, so attribution is undefined rather than merely unoptimized.

## Key Insight

If the retained memory set under a hard cap is treated as an intervention surface, then requiring the trigger logit and answer correctness to respond monotonically to ablating tokens ranked by their counterfactual causal contribution makes each proactive decision identifiable with respect to a specific compressed evidence subset — an identifiability condition that neither separate compression (InfiniPot-V) nor separate activation (StreamBridge) can express.

## Method

**What it is.** CTA-ALIVE (Counterfactual Token-Attribution ALIVE) is an RL-stage objective layered onto ALIVE's two-stage training. Its inputs are a streaming episode's retained memory token set M (under InfiniPot-V-style hard cap C, with |M| up to 512 tokens), the model's trigger logit z and answer y at each decision point, and a set of counterfactual ablations; its outputs are an attribution score a_g per retained-token cluster g and an augmented RL reward that couples trigger probability and answer correctness to those scores.

**Load-bearing assumption (stated explicitly).** The method assumes that ablating retained memory tokens produces additive, locally linear, and monotone effects on the trigger logit z_t and answer correctness, so that a first-order linear surrogate fit on K=8 Bernoulli masks yields token attributions whose ranking is causally faithful and can be enforced by a nested-ablation monotonicity penalty. This assumption is known to fail when retained tokens are redundant (InfiniPot-V's Temporal-axis Redundancy premise implies the retained set is correlated and mutually substitutable). To make the assumption testable rather than hidden, we (i) cluster tokens by redundancy and attribute at the cluster level, (ii) report the surrogate's R^2 per decision point and condition the faithfulness claim on high-R^2 episodes, and (iii) include a random-token control to separate genuine causal necessity from generic ablation robustness.

**How it works.**

```pseudocode
# Given: policy pi (ALIVE), hard cap C=512, episode with retained set M (|M|<=C)
# For each decision point t with trigger logit z_t and answer y_t:
1.  Cluster retained tokens into G groups by near-duplicate KV/feature similarity
        (cosine >= 0.9 on mean-pooled KV vectors, or reuse InfiniPot-V TaR groups);
        G is typically 20-60 for |M|~512.
2.  Sample K=8 ablation masks m_k ~ Bernoulli(p=0.5) over the G clusters
        (each mask ablates whole clusters, not individual tokens).
3.  For each mask, re-run pi with the selected clusters masked -> z_t^(k), y_t^(k).
4.  Fit a linear surrogate over clusters:
        z_t^(k) ~ w0 + sum_g w_g * m_k[g]
        correct_t^(k) ~ v0 + sum_g v_g * m_k[g]
    Record fit quality R^2_z and R^2_c per decision point.
5.  Attribution a_g = alpha * w_g + (1-alpha) * v_g      # alpha=0.5 fixed
6.  Rank clusters by a_g; form ablation ladder L = {top-1, top-3, top-10, top-20} clusters.
7.  For each rung, ablate the top-r clusters and record (z_t, correct_t).
8.  Monotonicity penalty over clusters:
        L_mono = sum_r max(0, z_t(top-r) - z_t(top-(r-1)))
                 + sum_r max(0, correct_t(top-(r-1)) - correct_t(top-r))
9.  Attribution-alignment reward using rank-based agreement (Spearman) instead of Pearson:
        R_attr = -L_mono  +  beta * spearman(a, delta_z)   # beta=0.3
10. Augment BACA reward: R_total = R_BACA + lambda * R_attr   # lambda=0.1, annealed to 0 over 50% of RL steps
11. Update pi with PPO on R_total; keep compression fixed (no retraining of cap).
12. Control: for each decision point, also ablate a random cluster of equal rank and
        record (z_t, correct_t) to separate necessity from generic ablation robustness.
```

Each retained-token cluster gets a scalar a_g that is a weighted combination of its marginal effect on the trigger logit (w_g) and on answer correctness (v_g). The ablation ladder forces the model to expose whether high-a_g clusters are actually load-bearing: ablating them should drop trigger probability and correctness in a monotone way if attribution is faithful. The random-cluster control tests whether any observed drop is specific to high-a_g clusters or merely a generic response to ablation.

**Why this design.** We chose a linear surrogate over a Shapley-value estimator because Shapley requires 2^G evaluations and G can be tens under the cap, accepting the cost that the surrogate is only first-order faithful and may miss cluster interactions. We chose a fixed ablation ladder (top-1, top-3, top-10, top-20 clusters) over random ablations because the ladder directly tests the monotonicity claim that the motivation requires, accepting that ladder spacing is a hyperparameter that may need per-benchmark tuning. We chose to anneal lambda from 0.1 to 0 rather than keep it constant so that early RL emphasizes attribution shaping while late RL preserves ALIVE's original BACA behavior, accepting that the annealing schedule adds a second hyperparameter. We chose cluster-level attribution over token-level because the load-bearing assumption of additive token effects is false under redundancy; clustering makes the surrogate well-posed at the cost of coarser attribution. A domain expert would not describe this as a variant of Shapley or of VinePPO because the objective is not to estimate token values but to force the trigger decision to be monotone in the estimated values — an identifiability constraint on the policy, not a value estimator.

**Why it measures what we claim.** The attribution score a_g measures the causal contribution of retained-token cluster g to the trigger decision because the linear-surrogate coefficient w_g equals the cluster's average marginal effect on z_t under the ablation distribution only if cluster effects are additive; this assumption fails when clusters are strongly redundant, in which case a_g reflects a cluster's unique contribution rather than its total contribution. The monotonicity penalty L_mono measures the faithfulness of attribution because requiring trigger probability to fall as higher-ranked clusters are ablated is equivalent to the ranking being consistent with causal necessity only if the ablation ladder is nested and the model's response is locally linear in ablation size; this fails when ablating a top cluster triggers a compensatory re-encoding, in which case L_mono reflects ablation robustness rather than attribution faithfulness. The Spearman agreement term measures alignment between attribution and observed trigger shift because rank correlation over the ladder rungs equals monotone agreement without assuming linearity; this fails when the true relationship is non-monotone (e.g., sigmoidal with a plateau), in which case the rank correlation underweights mid-ladder rungs. The surrogate R^2_z and R^2_c measure whether the additivity assumption holds at a given decision point because R^2 near 1 indicates the linear model explains the ablation outcomes; this fails when R^2 is low, in which case the attribution claim is not supported for that decision point and we condition the faithfulness claim on high-R^2 episodes.

## Contribution

["A counterfactual token-ablation identifiability objective (CTA-ALIVE) that couples per-token attribution scores to proactive trigger probability and answer correctness during ALIVE's RL stage, producing decisions that are attributable to specific retained compressed evidence.", 'A design principle: monotonicity of trigger probability and correctness under ablation of ranked retained tokens is a necessary condition for joint attribution between compression and activation, and can be enforced as an RL reward without retraining the compression cap.', 'An attribution-ladder evaluation protocol that reports per-rung trigger-probability and correctness curves, enabling direct comparison of attribution faithfulness across streaming assistants.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | FSI-Bench + OmniStar | Tests feedback-seeking and streaming task coordination |
| Primary metric | Trigger-decision faithfulness (monotonicity rate over clusters) | Directly measures whether attribution ranking is causally load-bearing |
| Secondary metric | Surrogate R^2_z and R^2_c per decision point | Tests the additivity assumption explicitly |
| Secondary metric | Spearman(a, delta_z) over ladder rungs | Measures rank agreement without linearity assumption |
| Baseline | ALIVE (BACA only) | Isolates contribution of attribution shaping |
| Baseline | StreamBridge | Proactive activation without causal token attribution |
| Baseline | Dispider | Disentangled perception/decision without attribution alignment |
| Ablation-of-ours | CTA-ALIVE, lambda=0 (no R_attr) | Same architecture, attribution reward removed |
| Ablation-of-ours | CTA-ALIVE, random-cluster control | Separates necessity from generic ablation robustness |
| Ablation-of-ours | CTA-ALIVE, token-level (no clustering) | Tests whether clustering is necessary for the effect |

### Why this setup validates the claim

The central claim is that coupling trigger probability and answer correctness to causal token attributions produces more faithful, better-timed proactive decisions than reward shaping without attribution. ALIVE (BACA only) tests the sub-claim that attribution shaping adds value beyond ALIVE's existing behavior-aligned credit assignment; if CTA-ALIVE beats it, the gain cannot be attributed to the underlying streaming model. StreamBridge tests the sub-claim that a decoupled activation model without token-level causal attribution cannot achieve monotone trigger decisions; it probes whether attribution alignment, not mere proactivity, drives the effect. Dispider tests the sub-claim that disentangled asynchronous perception/decision lacks the identifiability constraint our monotonicity penalty imposes. The lambda=0 ablation is the sharpest control: identical architecture, identical data, only R_attr removed, so any gap isolates the attribution term. The random-cluster control tests whether monotonicity gains are specific to high-a_g clusters or reflect generic ablation robustness. The token-level ablation tests whether the cluster-level attribution is necessary to handle redundancy. The primary metric — monotonicity rate, the fraction of decision points where ablated top-ranked clusters produce monotone drops in trigger probability and correctness — is chosen because it directly operationalizes the faithfulness claim; a method that improves accuracy but not monotonicity would not validate the mechanism. We additionally report the fraction of decision points with surrogate R^2_z > 0.7 and condition the faithfulness claim on this subset, so the mechanism is grounded where its additivity assumption holds. FSI-Bench and OmniStar are appropriate because they contain multi-turn, feedback-seeking and proactive-trigger episodes where the predicted failure mode (non-monotone, attribution-inconsistent triggering) actually occurs.

### Expected outcome and causal chain

**vs. ALIVE (BACA only)** — On a case where a long multi-turn episode accumulates redundant retained tokens and the user issues an ambiguous request mid-stream, ALIVE's BACA reward shapes trigger timing from trajectory-level advantage without knowing which token clusters are causally load-bearing, so it can fire on a token cluster that merely correlates with salient events. Our method instead ranks clusters by marginal effect on z_t and y_t and penalizes non-monotone ablation ladders, forcing the trigger to depend on causally necessary clusters, so we expect a clear monotonicity-rate gap on multi-turn ambiguous-request subsets and near-parity on single-turn clear-request subsets. We expect the gap to be larger on decision points with surrogate R^2_z > 0.7, confirming the mechanism works where its additivity assumption holds.

**vs. StreamBridge** — On a case where a brief salient event is followed by a long redundant stretch, StreamBridge's lightweight activation model responds to surface-level streaming cues and its round-decayed compression can drop the cluster that actually carried the trigger evidence, producing a late or missed proactive response. Our method keeps compression fixed but rewards alignment between attribution and observed trigger shift, so clusters whose ablation moves z_t are protected in the ranking, and we expect higher trigger faithfulness and fewer missed-event responses on sparse-event subsets, with parity on dense-event subsets where redundancy is low.

**vs. Dispider** — On a case where perception and reaction overlap, Dispider's asynchronous design lets the processing module keep monitoring while the precise module answers, but its decision module has no signal about which retained clusters justify a trigger, so it can raise proactive interactions on contextually weak evidence. Our method adds an identifiability constraint via the monotonicity penalty, so trigger decisions must be consistent with causal necessity, and we expect a moderate faithfulness gain concentrated on overlap-heavy episodes, with smaller differences where perception and reaction are temporally separated.

**vs. random-cluster control** — If CTA-ALIVE's monotonicity gain over the random-cluster control is near zero, then the observed monotonicity reflects generic ablation robustness rather than attribution-specific causal necessity, and the central claim is not supported.

**vs. token-level ablation** — If token-level attribution matches or exceeds cluster-level attribution on monotonicity rate, then the redundancy-aware clustering is unnecessary and the simpler token-level surrogate is sufficient; we expect cluster-level to outperform token-level on high-redundancy episodes (as measured by InfiniPot-V TaR redundancy scores) and to match on low-redundancy episodes.

### What would falsify this idea

If CTA-ALIVE's gain over the lambda=0 ablation is uniform across ambiguous and clear-request subsets, or if monotonicity rate improves while trigger accuracy and answer correctness do not, or if the gain disappears when conditioned on high surrogate R^2_z, then the attribution-alignment mechanism is not doing the causal work claimed and the central claim is wrong.

## References

1. ALIVE: Making Streaming Video Models Interactive
