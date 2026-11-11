# Order-Preserving Evidence Pooling for Multi-Teacher On-Policy Distillation

## Motivation

Multi-teacher distillation for agentic visual reasoning fuses rewards from spatial and temporal specialists, but each specialist's evidence-aware reward lives on its own arbitrary scale: Reframe's OG-MOPD aggregates teacher signals without any closure rule, so the fused reward — and thus the induced preference order over student policies — changes when a teacher's reward is independently rescaled (e.g., reweighting grounded-evidence vs. observation-use terms). This is the meta-level structural gap: across multi-teacher reward-fusion trees, aggregation is treated as a weighted sum, which is not closed under monotone renormalization of teacher-specific reward units, so no fusion result transfers across teachers with different internal scales.

## Key Insight

A log-opinion pool over per-teacher evidence-quality scores, each mapped through a shared latent scale, is the unique aggregation operator whose induced preference order is invariant to independent positive-affine renormalization of teacher-specific rewards, because log-odds differences absorb exactly the scale and shift that renormalization applies.

## Method

**What it is.** We introduce CEP (Compositional Evidence Pooling), a reward-fusion layer for multi-teacher on-policy distillation. Inputs: for each rollout step, per-teacher evidence-quality scores s_k from the spatial and temporal specialists, plus each teacher's reliability weight. Output: a single fused reward R whose ordering over student policies is invariant to independent renormalization of any teacher's reward scale.

**Load-bearing assumption (stated explicitly).** CEP assumes that a single frozen monotone map phi_k per teacher, fit by isotonic regression on ~2k held-out human-verified evidence-quality labels, places each teacher's raw score on a shared probability scale such that log-odds differences across teachers reflect genuine evidence-quality differences rather than each teacher's idiosyncratic scoring bias. This assumption is verified, not merely assumed: before freezing, each phi_k is tested for monotonicity against a held-out on-policy buffer (see Step 1b), and a teacher whose score is non-monotone in evidence quality (e.g., a temporal teacher that rewards observation length) is gated out of the pool (w_k set to 0) rather than allowed to contribute biased log-odds. The reliability weights w_k are computed only after this gate, so a biased-but-low-variance teacher cannot receive high weight.

**How it works.**
```pseudocode
# Per teacher k in {spatial, temporal}:
#   s_k in R  = teacher's evidence-aware score for this step
# Step 1: shared latent scale (calibrated once, frozen during distillation)
#   Fit monotone map phi_k(s_k) = sigmoid(a_k * s_k + b_k)
#   with a_k>0, b_k fit by isotonic regression against held-out
#   human-verified evidence-quality labels (N=2k steps), then FROZEN.
# Step 1b: monotonicity gate (per teacher, before freezing)
#   On a held-out on-policy buffer of 512 steps (collected from the
#   current student policy), compute Spearman rho_k between s_k and
#   human-verified evidence-quality labels. If rho_k < 0.3, set
#   w_k = 0 for this teacher (gate closed); else gate open.
# Step 2: per-teacher log-odds opinion
#   o_k = log( phi_k(s_k) / (1 - phi_k(s_k)) )        # in R
# Step 3: reliability-weighted log-opinion pool
#   w_k = 1 / Var_residual(phi_k(s_k))  (held-out residual variance)
#   (w_k = 0 if Step 1b gate closed)
#   O   = sum_k w_k * o_k / sum_k w_k              # monotone, order-preserving
# Step 4: fused evidence reward
#   R   = sigmoid(O)                               # back to [0,1]
# Step 5: student policy update uses R in place of per-teacher rewards
#   Advantage = R - baseline(state); on-policy distillation loss weighted by Advantage
```

**Why this design.** We chose a log-opinion pool over a weighted arithmetic sum because arithmetic averaging is not closed under positive-affine renormalization of a teacher's scores — adding a constant or rescaling one teacher silently reorders the fused reward — whereas log-odds pooling absorbs exactly the shift and scale, accepting the cost that each teacher's score must first be mapped to a probability-like quantity (Step 1). We chose a shared latent scale phi_k fit by isotonic regression and then frozen over per-rollout adaptive reweighting because adaptive weights re-introduce scale dependence through the weight update, accepting the cost that the calibration set must cover the evidence-score range the student will encounter. We chose reliability weights from held-out residual variance over fixed equal weights because a noisy teacher's log-odds should contribute less, accepting the cost that the variance estimate is itself only as good as the held-out set. We added the Step 1b monotonicity gate because the load-bearing assumption — that phi_k places each teacher on a shared probability scale — fails when a teacher's score is non-monotone in evidence quality, and the gate verifies this assumption on the on-policy distribution before freezing, accepting the cost of a small on-policy buffer (512 steps) and a per-teacher Spearman test. A domain expert would not describe CEP as a variant of Reframe's OG-MOPD: OG-MOPD replaces deficient tool actions and resumes distillation, but its reward aggregation is an unconstrained sum with no closure property, and CEP's contribution is precisely the closure rule that OG-MOPD lacks.

**Why it measures what we claim.** The quantity `o_k = log(phi_k(s_k)/(1-phi_k(s_k)))` measures teacher k's evidence-quality opinion on a common latent scale because the assumption that phi_k is a fixed monotone map from raw evidence scores to a shared probability scale makes log-odds comparable across teachers; this assumption fails when a teacher's raw score is non-monotonically related to true evidence quality (e.g., a temporal teacher that rewards observation length), in which case o_k reflects that teacher's idiosyncratic scoring bias rather than evidence quality — which is why Step 1b gates such teachers out (w_k=0) when the Spearman rho against held-out on-policy labels is below 0.3. The pooled quantity `O` measures the fused evidence-quality ordering because the assumption that reliability weights are fixed and positive makes the weighted log-odds sum order-preserving in each o_k; this assumption fails when weights are estimated from a held-out set drawn from a different rollout distribution, in which case O reflects calibration-set ordering rather than on-policy ordering — mitigated by computing w_k on the same on-policy buffer used in Step 1b. The final reward `R = sigmoid(O)` measures the student's step-level evidence reward because sigmoid is strictly monotone, so the preference order over policies induced by R equals that induced by O; this fails only if the advantage baseline is state-dependent in a way that inverts ordering, which we prevent by using a state-conditional baseline.

## Contribution

(1) A compositional evidence-aware reward layer (CEP) whose fused reward induces a preference order over student policies that is invariant to independent positive-affine renormalization of each teacher's reward, via a shared latent scale plus reliability-weighted log-opinion pooling. (2) A design principle: multi-teacher distillation reward fusion must be closed under teacher-specific renormalization, and log-opinion pooling is the operator that provides this closure while arithmetic averaging does not. (3) An empirical demonstration that CEP-fused distillation preserves teacher-ordering under deliberately perturbed teacher reward scales where sum-fusion reorders.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Reframe's 14 image/video benchmarks | Matches the multi-teacher agentic setting under test |
| Primary metric | Evidence-grounded answer accuracy | Directly reflects step-level evidence quality |
| Baseline | Mix-RL | Tests whether naive multi-teacher mixing suffices |
| Baseline | OG-MOPD (Reframe) | Tests whether unconstrained sum aggregation is enough |
| Baseline | Unweighted log-odds pool | Tests whether reliability weighting matters |
| Ablation-of-ours | CEP with adaptive (unfrozen) phi | Tests the frozen-calibration design choice |
| Ablation-of-ours | CEP without Step 1b monotonicity gate | Tests whether the gate is necessary to prevent biased-but-low-variance teachers from receiving high weight |
| Diagnostic | Rank correlation (Kendall tau) of fused rewards before/after independent rescaling of each teacher's scores | Quantifies how often renormalization reorders the fused reward in baselines |
| Diagnostic | Per-teacher reliability diagram and ECE on held-out benchmark | Validates that phi_k(s_k) is on a common probability scale |

### Why this setup validates the claim

The central claim is that closure under per-teacher positive-affine renormalization yields a fused reward whose policy ordering tracks true evidence quality better than unconstrained aggregation. Mix-RL tests the sub-claim that naive mixing is insufficient; OG-MOPD tests whether Reframe's unconstrained sum already captures the benefit, isolating CEP's closure rule as the causal ingredient. The unweighted log-odds baseline isolates the reliability-weighting sub-claim, and the adaptive-phi ablation isolates the frozen-calibration sub-claim, so a gain can be attributed to a specific design choice rather than to log-odds pooling generically. The Step 1b gate ablation tests the load-bearing assumption directly: if removing the gate does not hurt accuracy, the monotonicity verification is not doing the work we claim. The rank-correlation diagnostic establishes that the problem is empirically significant by measuring how often independent renormalization actually reorders the fused reward in baselines. The per-teacher ECE diagnostic validates that phi_k(s_k) is on a common probability scale rather than merely monotone per teacher. Evidence-grounded answer accuracy is the right metric because the predicted effect is an ordering improvement over policies, and this metric directly reads out whether the student prefers higher-evidence-quality steps; a metric like raw reward magnitude would not distinguish ordering from scale. Together these form a falsifiable test: if CEP's advantage is real, it must appear against each baseline for the reason that baseline lacks, and disappear when the corresponding CEP component is ablated.

### Expected outcome and causal chain

**vs. Mix-RL** — On a case where the spatial teacher's scores are rescaled (e.g., a benchmark where spatial evidence scores span a wider range), Mix-RL's arithmetic mixing silently reorders the fused reward, so the student is pushed toward temporal-heavy steps that look high after rescaling but carry weak spatial evidence, producing answers that cite the wrong region. Our method maps each teacher through a frozen monotone phi_k into log-odds before pooling, so renormalization is absorbed exactly and the ordering is preserved, so we expect a noticeable accuracy gap on spatially demanding image subsets but near-parity on temporal-only video subsets. The rank-correlation diagnostic should show Mix-RL's fused-reward ordering changing (Kendall tau < 0.9) under rescaling while CEP's remains stable (tau > 0.99).

**vs. OG-MOPD (Reframe)** — On a case where one teacher emits a deficient tool action and the other emits a correct one, OG-MOPD's unconstrained sum lets the deficient teacher's large raw score dominate the fused reward, so the student imitates the deficient action and resumes distillation on a corrupted trajectory. CEP instead pools reliability-weighted log-odds, so the noisy teacher's opinion is down-weighted by its held-out residual variance and the correct teacher's log-odds dominates, so we expect a concentrated gain on the tool-action-deficiency subset with only mild gains elsewhere.

**vs. Unweighted log-odds pool** — On a case where the temporal teacher rewards observation length rather than evidence quality, the unweighted pool gives that biased teacher equal log-odds voice, so the student learns to lengthen observations and its evidence-grounded accuracy stalls on cases needing concise spatial checks. CEP's variance-derived weights shrink the biased teacher's contribution, so we expect a moderate gap favoring CEP specifically on cases where the two teachers disagree, and parity where they agree. The Step 1b gate should close for this biased teacher (Spearman rho < 0.3 on the on-policy buffer), setting w_k=0 and eliminating its contribution entirely.

**vs. CEP without Step 1b gate** — On the same biased-teacher case, removing the gate allows the length-rewarding temporal teacher to receive high weight because its residual variance is low despite its bias, inverting the intended down-weighting and contradicting the 'vs. Unweighted log-odds pool' causal story. We expect the gated CEP to outperform the ungated variant specifically on teacher-disagreement subsets, confirming that the monotonicity verification is necessary for the load-bearing assumption to hold.

### What would falsify this idea

If CEP's accuracy gain over Mix-RL and OG-MOPD is uniform across all subsets rather than concentrated on the renormalization-sensitive and teacher-disagreement subsets where the causal chain predicts it, the closure-rule explanation is wrong. Likewise, if the adaptive-phi ablation matches frozen phi, the frozen-calibration design choice is not doing the work we claim, and if the Step 1b gate ablation matches gated CEP, the monotonicity verification is not necessary.

## References

1. Reframe: Observation-Aligned Multi-Teacher On-Policy Distillation for Agentic Visual Reasoning
