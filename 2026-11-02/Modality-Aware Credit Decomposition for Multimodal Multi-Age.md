# Modality-Aware Credit Decomposition for Multimodal Multi-Agent Prompt Optimization

## Motivation

Existing multimodal multi-agent systems such as Mdocagent fix modality-specific prompts, and even the general prompt-optimization framework CEPO unifies prompt structure but feeds the optimizer only a single text-level scalar reward, so per-agent and per-modality prompts receive no differentiated credit signal. This structural limitation means a vision agent and a text agent sharing a coordinator prompt are updated by the same scalar even when their contributions to the final answer differ, preventing joint optimization of modality-specific roles. The root cause is that the optimizer's feedback channel is modality-agnostic: it cannot attribute outcome variance to the modality whose prompt caused it.

## Key Insight

Because each modality-specific agent's prompt is a controlled intervention on a distinct input channel, the outcome can be additively decomposed into per-modality prompt effects plus a cross-modal interaction term, and this decomposition is identifiable from paired counterfactual rollouts that swap exactly one agent's prompt while holding the others fixed.

## Method

# MACD: Modality-Aware Credit Decomposition for multi-agent prompt optimization
# Input: MAS with coordinator prompt p_c and modality-specific executor prompts {p_m} for m in {text, image, table} (M=3)
# Output: updated prompts {p_c, p_m} with per-modality credit signals

# [Load-bearing assumption, stated explicitly]
# Swapping exactly one modality agent's prompt while holding the other prompts fixed is treated as a valid intervention
# whose marginal reward delta c_m(x) isolates that modality's causal contribution, i.e., per-modality prompt effects are
# assumed additively separable up to a two-way interaction term. This assumption is NOT verified by construction: because
# executor outputs feed downstream agents' context, swapping p_m changes o_m and therefore changes what the coordinator and
# sibling agents condition on, so c_m is a bundled downstream effect rather than a channel-isolated main effect. We therefore
# (i) estimate credits by Shapley-style averaging over the modality-prompt coalition (all 2^M=8 subsets, feasible for M=3)
# so downstream mediation is averaged out rather than attributed to whichever agent was swapped, (ii) attach a bootstrap
# standard error to every credit and normalize by that error rather than by raw magnitude, and (iii) relabel c_int as an
# unidentifiable higher-order residual (not 'cross-modal synergy') and gate coordinator updates on a significance test.

1. BASELINE ROLLOUT
   For each task x, run the MAS with current prompts, record final answer y and per-agent intermediate outputs {o_m}.
   Compute task reward R(x) in [0,1] (exact-match or F1).

2. COALITION ROLLOUTS (Shapley-style, replaces fixed-others single swap)
   For each modality m and each candidate prompt p_m' in the optimizer's proposal pool (size K=4):
     For each subset S of modalities containing m (there are 2^(M-1)=4 such subsets for M=3):
       Re-run the MAS with p_m' substituted for p_m and the prompts of all modalities in S also swapped to their
       current best candidates, all other prompts fixed; record R_{m',S}(x).
     Also run the joint swap S = {text, image, table} with all modality prompts replaced simultaneously, recording R_all'(x).
   Total rollouts per task per round: K * (2^(M-1)) * M + 1 = 4 * 4 * 3 + 1 = 49 (vs. 13 in the original single-swap design);
   this is the compute cost of averaging out downstream mediation.

3. MODALITY CREDIT ESTIMATION (Shapley over modality-prompt coalition)
   For each modality m, compute the Shapley-style marginal credit averaged over coalition orderings:
     c_m(x) = (1/M) * sum_{S subset of modalities not containing m} [ R_{m',S∪{m}}(x) - R_{m',S}(x) ] / C(M-1, |S|)
   where the inner difference is the marginal effect of adding m's swap to coalition S, and the outer average is over all
   orderings (equivalent to the Shapley value of the modality-prompt intervention game).
   Compute the higher-order residual (relabelled from 'interaction credit'):
     c_int(x) = R_all'(x) - R(x) - sum_m c_m(x)   # unidentifiable higher-order residual, NOT cross-modal synergy
   Bootstrap variance: repeat steps 1-3 with B=5 independent rollout seeds per candidate (temperature 0.7, top-p 0.95)
   and compute the bootstrap standard error se(c_m) and se(c_int) as the standard deviation across the B resamples.
   Normalize credits by estimated standard error (replaces raw-magnitude normalization):
     c_m <- c_m / (se(c_m) + eps),   c_int <- c_int / (se(c_int) + eps),   eps = 1e-6.

4. ROLE-AWARE UPDATE (gated on significance)
   Coordinator prompt p_c receives the higher-order residual plus a role-consistency term, but ONLY if the residual is
   significant at the 0.05 level under the bootstrap:
     delta_c = eta_c * (c_int + lambda * mean_m c_m) * 1[ |c_int| / se(c_int) > 1.96 ],  eta_c = 0.1, lambda = 0.3
   Each executor prompt p_m receives its own Shapley-style marginal credit:
     delta_m = eta_m * c_m,  eta_m = 0.2
   Apply updates via the base optimizer (CEPO-guided textual gradient): append credit-weighted critique to the prompt and
   let the optimizer rewrite it.

5. ITERATE for T=5 rounds, keeping a replay buffer of (prompt, credit, se) pairs to stabilize variance.

# [Missing equivalence, stated explicitly]
# c_int measures coordinator-attributable cross-modal coupling under assumption A: the coordinator prompt is the sole shared
# causal parent across modalities and higher-order interactions are negligible. This assumption fails (failure mode F) when
# an executor prompt feeds another agent's input, in which case c_int misattributes that executor's effect to the coordinator.
# We therefore report c_int with its bootstrap standard error and gate coordinator updates on the significance test above;
# the misattribution is quantified in the ablation that perturbs an executor prompt known to affect another agent's input.

**Why this design.** We chose Shapley-style coalition averaging over fixed-others single swaps because the latter is confounded by downstream mediation (swapping p_m changes o_m, which changes what the coordinator conditions on), whereas averaging the marginal effect over all 2^M=8 modality-prompt coalitions makes the credit identifiable up to the higher-order residual by construction; the cost is 49 rollouts per task per round (vs. 13), which we accept for M=3. We chose additive decomposition with an explicit higher-order residual over a simple per-agent reward average because averaging ignores cross-modal coupling, accepting the cost that the residual is noisier and must be down-weighted for the coordinator (lambda=0.3) and gated on a bootstrap significance test. We chose to route the residual to the coordinator rather than to individual executors because the coordinator is the only prompt that causally spans all modalities, accepting the cost that coordinator updates are less directly attributable to a single modality and that executor-to-executor input dependencies can misattribute effects to the coordinator (failure mode F). A domain expert would not describe this as a variant of CEPO because CEPO only structures prompts and passes a scalar to an existing optimizer, whereas MACD changes the optimizer's feedback channel to a vector of modality-specific Shapley-style counterfactual credits with bootstrap standard errors; it also differs from VinePPO-style per-step Monte-Carlo value estimation because the intervention is on the prompt (a structural variable), not on a sampled action prefix, and the target is prompt credit rather than step value.

**Why it measures what we claim.** The Shapley-style credit c_m(x) measures the causal contribution of modality m's prompt to the outcome because the named assumption is that averaging the marginal effect of m's swap over all modality-prompt coalitions averages out downstream mediation (no hidden confound between prompts); this assumption fails when prompts are functionally redundant (e.g., text and image agents both encode the same instruction), in which case c_m reflects only the residual effect not already captured by the others. The higher-order residual c_int(x) measures coordinator-attributable cross-modal coupling because the assumption is that the joint effect decomposes additively up to a two-way interaction and that the coordinator is the sole shared causal parent; this fails when three-way or higher-order interactions dominate or when an executor prompt feeds another agent's input, in which case c_int absorbs those effects and misattributes them to the coordinator. The bootstrap-error normalization measures relative modality importance because the assumption is that credits are comparable in reward units and that the bootstrap standard error captures rollout noise; this fails when one modality's reward scale is compressed (e.g., near-saturated accuracy), in which case normalization amplifies noise. The role-aware routing (coordinator gets c_int, executors get c_m) measures role-specific update signal because the assumption is that the coordinator prompt is the only shared causal parent across modalities; this fails when an executor prompt also affects other agents' inputs, in which case its marginal credit is underestimated and the residual is overstated.

## Contribution

(1) A modality-aware credit decomposition framework (MACD) that extends a prompt optimizer's feedback channel from a single scalar to a vector of per-modality marginal credits plus a cross-modal interaction term, computed from paired counterfactual prompt swaps. (2) A role-aware credit routing rule that assigns interaction credit to the shared coordinator prompt and marginal credit to modality-specific executor prompts, yielding differentiated update signals without a learned credit head. (3) An empirical demonstration that counterfactual prompt-swap credit is identifiable and stable enough to drive joint multimodal prompt optimization, with a replay buffer that controls variance across rounds.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Mixed-modality MAS benchmarks (document-understanding + table/image QA tasks) | Tasks where text, image, table agents all contribute |
| Primary metric | Task reward (exact-match / F1) per round | Directly the quantity credits decompose |
| Diagnostic metric | R^2 of additive model: regress joint-swap reward on sum of Shapley credits plus residual on held-out tasks | Tests the additive-decomposition assumption directly |
| Diagnostic metric | Bootstrap standard error se(c_m), se(c_int) per round | Quantifies credit noise and gates coordinator updates |
| Baseline | CEPO + scalar-reward optimizer (compute-matched: 49 rollouts/task/round) | Tests whether vector credit beats scalar feedback at equal budget |
| Baseline | Per-agent reward average (no higher-order residual; compute-matched: 49 rollouts/task/round) | Tests whether residual routing matters at equal budget |
| Baseline | Random prompt swap, no credit attribution (compute-matched: 49 rollouts/task/round) | Tests whether causal credit beats blind search at equal budget |
| Ablation-of-ours | MACD with higher-order residual zeroed (c_int=0) | Isolates contribution of coordinator routing |
| Ablation-of-ours | MACD with executor prompt perturbed to affect another agent's input | Quantifies misattribution to coordinator (failure mode F) |
| Ablation-of-ours | MACD with lambda calibrated from residual variance (lambda in {0.1,0.3,0.5}) | Tests sensitivity to fixed lambda=0.3 |

### Why this setup validates the claim

The central claim is that decomposing the scalar task reward into modality-specific Shapley-style counterfactual credits gives the prompt optimizer a better feedback channel than a single scalar, and that routing the higher-order residual to the coordinator captures cross-modal coupling that per-agent averaging misses. The dataset choice matters because the claim only has teeth on tasks where multiple modalities genuinely co-contribute; on single-modality tasks the decomposition collapses to a trivial scalar and the method should show parity, which is itself a useful null. The CEPO-scalar baseline tests the first sub-claim (vector vs. scalar feedback) at equal rollout budget (49/task/round), the per-agent-average baseline tests the second sub-claim (residual is load-bearing) at equal budget, and the random-swap baseline tests whether the causal intervention structure — not merely extra rollouts — produces the gain. The interaction-zeroed ablation is the sharpest test: if zeroing c_int does not hurt, the role-aware routing is decorative. The R^2 diagnostic directly tests the additive-decomposition assumption: if residuals are large, c_int absorbs them and lambda should be calibrated from residual structure rather than fixed at 0.3. The executor-perturbation ablation quantifies failure mode F (executor prompt feeding another agent's input). Task reward is the right metric because credits are defined in reward units; any proxy metric would confound attribution with measurement. Together these form a falsifiable test: the method must beat scalar feedback specifically on cross-modal-synergy subsets, not uniformly, and the additive model must explain a substantial fraction of joint-swap variance.

### Expected outcome and causal chain

**vs. CEPO + scalar-reward optimizer (compute-matched: 49 rollouts/task/round)** — On a case where a table agent's prompt is the true bottleneck but the text agent's fluent output masks the error, the scalar baseline produces a diffuse critique that rewrites the text prompt because its gradient is dominated by the most verbose agent. Our method instead computes c_table as the Shapley-style marginal reward change under a table-prompt swap averaged over modality coalitions, so the update lands on the table prompt because the intervention isolates its channel up to downstream mediation. We expect a noticeable gap on tasks where one modality is silently wrong but the answer is still partially fluent, and parity on tasks where the bottleneck modality is already the loudest. Because the baseline is compute-matched, any gap cannot be attributed to extra rollouts.

**vs. Per-agent reward average (no higher-order residual; compute-matched: 49 rollouts/task/round)** — On a case requiring joint reasoning where neither the text nor the image prompt alone fixes the answer, but their combination does, the averaging baseline produces near-zero credit for both agents because each marginal swap leaves reward flat, so no prompt is updated. Our method instead captures the joint-swap gain in c_int and routes it to the coordinator only when the bootstrap significance test passes (|c_int|/se(c_int) > 1.96), so the coordinator prompt is rewritten to enforce the cross-modal coupling. We expect the gain to concentrate on multi-hop cross-modal questions, with the ablation (c_int zeroed) reverting to the averaging baseline's flat behavior on exactly those items.

**vs. Random prompt swap, no credit attribution (compute-matched: 49 rollouts/task/round)** — On a case where the proposal pool contains one good and three mediocre candidates per modality, the random-swap baseline produces noisy reward deltas that cancel across rounds because it cannot tell which modality's swap caused the change. Our method instead normalizes per-modality credits by their bootstrap standard error so the good candidate's marginal effect survives normalization and drives the update. We expect faster per-round reward climb in the first two rounds and a smaller variance across rounds, with the gap shrinking if the proposal pool is uniformly good (where random search also works).

**vs. MACD with higher-order residual zeroed (c_int=0)** — On cross-modal-synergy items, zeroing c_int removes the coordinator update, so the method reverts to per-agent Shapley credits and should match the averaging baseline on exactly those items. If performance is unchanged, the role-aware coordinator routing is not doing the causal work we claim.

**vs. MACD with executor prompt perturbed to affect another agent's input (failure mode F)** — On tasks where an executor prompt feeds another agent's input, the perturbation should inflate c_int and misattribute the executor's effect to the coordinator; we expect the coordinator update to be misdirected and performance to drop relative to unperturbed MACD, quantifying the misattribution.

**vs. MACD with lambda calibrated from residual variance (lambda in {0.1,0.3,0.5})** — If the R^2 diagnostic shows large residuals, the fixed lambda=0.3 should be suboptimal; we expect the calibrated lambda to outperform the fixed value on high-residual subsets, confirming that lambda should be set from residual structure.

### What would falsify this idea

If the gain over the scalar CEPO baseline is uniform across single-modality and cross-modal-synergy subsets rather than concentrated on the latter, the credit-decomposition story is wrong and the method is just a noisier optimizer. If zeroing the higher-order residual leaves performance unchanged, the role-aware coordinator routing is not doing the causal work we claim. If the R^2 of the additive model is low on held-out tasks, the additive-decomposition assumption fails and c_int is absorbing unidentifiable higher-order effects rather than measuring cross-modal coupling.

## References

1. CEPO: A General Framework for Improving Prompt Optimization in Multi-Agent Systems
