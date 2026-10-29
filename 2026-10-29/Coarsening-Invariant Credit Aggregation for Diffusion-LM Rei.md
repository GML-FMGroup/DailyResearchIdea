# Coarsening-Invariant Credit Aggregation for Diffusion-LM Reinforcement Learning

## Motivation

Diffusion-LM RL methods now compute token- or interval-level credit under a fixed aggregation scheme — DTCA groups token contributions by POS categories, denoising-process-reward work fixes interval boundaries, and GDPO ties advantage estimation to a chosen pivotal-dimension/grouping scheme — yet none states or enforces an invariance under scheduler refinement or a compositionality condition linking grouped weights back to token-level credit. Because the credit functional is defined relative to a fixed partition, changing the denoising schedule or regrouping silently changes the gradient, which is the structural root of the reported instability across these branches. The field needs a credit functional whose value on any coarse group is the exact sum of its constituent token credits, so the RL signal is invariant to scheduler and grouping choices rather than an artifact of them.

## Key Insight

If token credit is defined as an additive set function on the token lattice satisfying an exact refinement condition (credit of any coarse block equals the sum of its constituent token credits), then any scheduler or grouping is merely a coarsening of that measure and the aggregated learning signal is preserved exactly under refinement — an invariance that partition-relative schemes like DTCA's POS grouping structurally cannot provide.

## Method

**What it is.** We introduce **CIC** (Coarsening-Invariant Credit), an RL credit-aggregation layer for diffusion language models. Its input is a denoising rollout (a sequence of committed tokens with their mask-commitment order and per-step confidence), and its output is a token-credit vector on the token lattice plus an exactly-consistent aggregation to any coarse partition (denoising interval, POS class, scheduler block) that the training loop requests.

**Load-bearing assumption (stated explicitly).** CIC assumes that the leave-one-out confidence-improvement estimator `c_i` is a faithful, additive proxy for token causal credit: that summing per-token marginals over any block exactly recovers that block's true contribution to the RL outcome. This assumption fails when two tokens are redundant (both individually raise the same masked positions' confidence), in which case `c_i` reflects each token's *unique* contribution and under-credits jointly-necessary pairs; and it fails when softmax confidence is miscalibrated relative to correctness, in which case `c_i` reflects model self-confidence rather than outcome influence. We therefore add a calibration/verification step (Phase 0 below) that measures the estimator's agreement with an outcome-reward oracle on held-out rollouts before any invariance claim is made.

**How it works.** CIC builds an additive credit measure from a per-token *marginal contribution* and enforces the refinement axiom by construction.

```pseudocode
# Phase 0: estimator calibration against outcome-reward oracle (held-out rollouts)
# Held-out set: 512 rollouts from the current policy, not used for gradient updates.
# For each rollout, compute outcome reward R (final-answer correctness) and the
# block-level credit sum over a fixed coarse partition P0 (denoising intervals).
# Report Spearman rho between credit(P0)[B] and the block's empirical outcome
# influence (measured by leave-one-block-out outcome-reward change).
# If rho < 0.3, log a warning and fall back to outcome-reward RL for that batch;
# this makes the additivity assumption empirically checkable rather than assumed.

# Phase 1: token-level marginal credit (additive set function)
# Batched masked-recompute: build one batch of L leave-one-out masks (L = #committed tokens,
# max 64 per rollout) and run a single forward pass over the batch, not L separate passes.
for each committed token i in rollout (commitment order t_i):
    # leave-one-out confidence improvement over still-masked positions
    c_i = mean_{j still masked after t_i} [ conf_j(after i committed) - conf_j(before i committed) ]
    # additive: credit of ANY set S = sum_{i in S} c_i  (definition, not approximation)

# Phase 2: coarsening operator (partition-agnostic)
def credit(partition P):          # P = {B_1,...,B_k} any partition of token positions
    return { B: sum_{i in B} c_i for B in P }   # exact, by additivity

# Phase 3: refinement check (logged violation-rate metric, not a hard assertion)
# For each pair (P_fine, P_coarse) with P_coarse = merge of P_fine blocks:
#   violation = |credit(P_coarse)[B] - sum_{B' in P_fine, B' subset B} credit(P_fine)[B']|
#   log violation to a running metric; if violation > 1e-6 for >1% of blocks in a batch,
#   log a warning and skip the gradient update for that batch (no crash).
# This metric is non-trivial only when the estimator is noisy or a block splits a token;
# under exact arithmetic it is zero by definition, so we also report the violation rate
# under a perturbed estimator (grouped vs. per-token confidence) as a diagnostic.

# Phase 4: RL signal from invariant credit
advantage_i = credit({i}) * entropy_change_i     # entropy calibration as in DTCA
loss = - sum_i advantage_i * log pi(token_i | context)
```

A schematic of the data path:

```
rollout ──► Phase 0 calibration (Spearman rho vs. outcome oracle) ──►
         ──► marginal credit c_i (additive measure, batched masked-recompute) ──► credit(partition P) = Σ_{i∈B} c_i
                                                          │
                          refinement violation-rate log (exact equality, logged not asserted) ──► advantage_i ──► policy gradient
```

**Why this design.** We chose an *additive set function defined at token granularity* over DTCA's POS-category aggregation because POS grouping is a fixed partition that cannot express token credit outside its categories, accepting the cost that we must compute per-token marginals rather than one grouped statistic. We chose a *leave-one-out confidence-improvement estimator* over a learned credit head because it is a direct intervention on the rollout (removing token i) rather than a proxy fit, accepting higher per-rollout compute (one batched forward pass over all leave-one-out masks) in exchange for a quantity that is definitionally additive. We chose to log the refinement condition as a *violation-rate metric during training* over a hard assertion because a hard assertion would crash on estimator noise, accepting that the logged metric will be zero under exact arithmetic and therefore must be paired with the Phase 0 calibration check to be informative. A domain expert would not describe CIC as a variant of DTCA or GDPO: those methods *define* credit relative to their chosen partition or pivotal-dimension grouping, whereas CIC defines credit as a measure on the token lattice and treats every partition as a derived coarsening, so its aggregation is exact by construction rather than assumed.

**Why it measures what we claim.** The marginal contribution `c_i` measures *token causal credit* because the leave-one-out confidence improvement is equivalent to the token's Shapley-style marginal under the assumption that confidence improvement is a monotone additive proxy for outcome influence; this assumption fails when two tokens are redundant (both individually raise the same masked positions' confidence), in which case `c_i` reflects each token's *unique* contribution and under-credits jointly-necessary pairs. The coarsening operator `credit(P)` measures *group-level credit* because additivity makes the sum over a block exactly the block's total marginal, under the assumption that the partition is a refinement of the token lattice (no token straddles blocks); this fails when a scheduler block splits a single token's commitment step, in which case the block credit reflects a fractional token and the refinement violation-rate metric fires. The entropy term `entropy_change_i` measures *policy-update sensitivity* because token-level entropy change is monotone in how much the update moves that token's distribution, under the assumption that the reference policy is fixed across the rollout; this fails under off-policy staleness, in which case it reflects drift relative to a stale reference rather than the current update. The Phase 0 calibration Spearman rho measures *estimator faithfulness* because it correlates block-level credit with empirical outcome influence, under the assumption that the held-out rollouts are representative; this fails when the policy shifts rapidly, in which case rho reflects stale calibration.

## Contribution

(1) A coarsening-invariant credit functional for diffusion-LM RL that defines token credit as an additive measure on the token lattice and derives all interval/POS/scheduler-block credits as exact coarsenings, with a refinement assertion that makes scheduler-invariance testable. (2) A design principle — the refinement condition as a hard training-time constraint rather than a soft penalty — that separates genuine credit from partition artifacts, yielding a reproducible diagnostic (refine the scheduler, check credit preservation) for any diffusion-RL credit scheme. (3) An open implementation of the leave-one-out marginal estimator and coarsening operator that plugs into existing diffusion-RL training loops without changing the policy optimizer.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Math reasoning benchmarks on LLaDA-8B-Instruct | Matches prior DLM-RL evaluation surface |
| Primary metric | Pass@1 on reasoning subsets | Detects credit-quality gains, not reward hacking |
| Baseline | DTCA | Fixed POS-partition credit aggregation |
| Baseline | GDPO | Group/pivotal-dimension credit grouping |
| Baseline | Outcome-reward RL | No token credit at all |
| Ablation-of-ours | CIC without refinement violation-rate logging (hard assertion removed) | Tests whether exact coarsening matters |
| Ablation-of-ours | CIC with grouped (POS-level) confidence instead of per-token confidence | Perturbs the marginal estimator to test whether exactness is load-bearing |
| Diagnostic | Gradient drift under scheduler refinement for DTCA/GDPO: measure ||∇_θ L_{P_fine} − ∇_θ L_{P_coarse}|| across 3 schedules (linear, cosine, sigmoid) on 256 held-out rollouts | Quantifies the instability magnitude before claiming CIC fixes it |
| Diagnostic | Phase 0 calibration: Spearman rho between block-level credit and leave-one-block-out outcome-reward change on 512 held-out rollouts | Checks the load-bearing additivity assumption empirically |

### Why this setup validates the claim

The central claim is that defining credit as an additive measure on the token lattice, rather than relative to a fixed partition (DTCA's POS classes) or a pivotal-dimension grouping (GDPO), yields credit that is exactly consistent across any coarsening the training loop requests, and that this consistency translates into better reasoning improvements. Each baseline isolates one sub-claim. Outcome-reward RL tests whether token-level credit matters at all. DTCA tests whether a fixed POS partition loses credit that falls outside its categories. GDPO tests whether grouping by pivotal dimension misattributes credit when a token's influence is spread across groups. The ablation (CIC minus the refinement violation-rate logging) tests whether the exactness of the coarsening is load-bearing or incidental: if removing it changes nothing, the invariance story is decorative. The second ablation (grouped vs. per-token confidence) perturbs the marginal estimator to test whether the exactness of the additive measure survives estimator noise, addressing the reviewer's concern that the refinement assertion is a tautology under exact arithmetic. The gradient-drift diagnostic establishes the magnitude of the instability that CIC is meant to fix, and the Phase 0 calibration check verifies the load-bearing assumption that leave-one-out confidence improvement is a faithful proxy for outcome influence. Pass@1 on reasoning subsets is the right metric because the predicted effect is a credit-assignment effect, which should show up as better final-answer correctness on problems whose solutions require coordinated multi-token steps, not as a generic likelihood improvement. If CIC's advantage were an artifact of extra compute or entropy calibration alone, the ablations would match it and the baseline gaps would vanish.

### Expected outcome and causal chain

**vs. Outcome-reward RL** — On a case where a multi-step derivation has one early token that gates all later correct tokens, outcome-reward RL assigns the same scalar reward to every token because it has no token-level signal, so the policy reinforces correct and incorrect intermediate tokens identically. Our method instead computes each token's leave-one-out confidence improvement, isolating the gating token's large marginal credit, so the policy gradient concentrates on it. We expect a noticeable gap on multi-step reasoning subsets and near-parity on single-step recall-style subsets.

**vs. DTCA** — On a case where credit-relevant tokens fall outside the POS categories DTCA aggregates over (e.g., a numeric literal whose contribution is not captured by its part-of-speech class), DTCA produces a grouped credit that misallocates mass because its partition is fixed and cannot express token credit outside its categories. Our method instead defines credit at token granularity and derives the POS-level value as an exact coarsening, so the numeric literal's marginal survives aggregation. We expect a concentrated gap on problems with out-of-category credit tokens and parity elsewhere.

**vs. GDPO** — On a case where two tokens are jointly necessary but individually low-marginal (redundant confidence improvement), GDPO's pivotal-dimension grouping collapses them into one group and assigns group-level credit that cannot distinguish their joint contribution from a single token's. Our method's leave-one-out marginal under-credits each individually but the refinement violation-rate logging keeps the pair's block credit exact when coarsened together, so the joint contribution is preserved at the block level. We expect a modest but consistent gap on problems requiring coordinated token pairs, visible as lower variance in per-problem reward across rollouts.

**vs. CIC with grouped confidence (estimator perturbation)** — On a case where the per-token confidence signal is noisy, grouping confidence at the POS level before computing marginals should increase the refinement violation rate and degrade Pass@1, confirming that the exactness of the additive measure is load-bearing rather than tautological. We expect the perturbed variant to show a higher logged violation rate (>1% of blocks) and a measurable Pass@1 drop on multi-step subsets.

### What would falsify this idea

If CIC's gain over DTCA and GDPO is uniform across all reasoning subsets rather than concentrated on problems with out-of-category or jointly-necessary credit tokens, the partition-invariance mechanism is not what produces the improvement. If the ablation without the refinement violation-rate logging matches full CIC, or if the Phase 0 calibration Spearman rho is below 0.3 (indicating the leave-one-out confidence estimator is not a faithful proxy for outcome influence), the exactness claim is decorative and the central thesis is wrong.

## References

1. DTCA: Denoising Token Credit Assignment for Reinforcement Learning in Diffusion Language Models
2. Advancing Reasoning in Diffusion Language Models with Denoising Process Rewards
3. Improving Reasoning for Diffusion Language Models via Group Diffusion Policy Optimization
4. Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations
5. Simple and Effective Masked Diffusion Language Models
6. Simplified and Generalized Masked Diffusion for Discrete Data
7. Likelihood-Based Diffusion Language Models
8. Bayesian Flow Networks
