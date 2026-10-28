# Segmentation-Invariant Credit Assignment for Turn-Anchored Agent RL

## Motivation

Across the turn-anchored RL literature, the turn is treated as an observable, well-defined unit: MPO anchors multiscale advantage estimation at turns and defines turn-level importance ratios relative to them, T-PPO's EGAE assumes a truncation point that cleanly ends a usable segment, and GSPO pushes the same assumption to the sequence level. Yet assistant turns are routinely split by tool calls, merged by formatting, or truncated at context-window boundaries, so the same token-level trajectory yields different advantages and different policy ratios depending on where the partition happens to fall — a hidden source of variance and bias that no existing branch models, because none formalizes what makes a turn-anchored learning signal identifiable under segmentation.

## Key Insight

If the turn-anchored advantage and turn-level importance ratio are defined as expectations over an equivalence class of segmentations that share the same token-level transition structure, the resulting gradient is exactly invariant to any refinement or coarsening of the partition, so segmentation ambiguity becomes a nuisance variable that cancels rather than a bias that propagates.

## Method

**What it is.** SIRCA (Segmentation-Invariant Ratio and Credit Assignment) is a drop-in replacement for the advantage estimator and importance ratio inside a turn-anchored policy-gradient loop (e.g., the MPO objective). Its inputs are a token-level trajectory with per-token log-probs under the behavior and current policies, a token-level reward, and a critic value per token; its outputs are a segmentation-invariant advantage $\hat{A}^{\text{inv}}_t$ and a segmentation-invariant ratio $\rho^{\text{inv}}_t$ for every token, which are then plugged into the unchanged PPO/MPO clipped surrogate.

**How it works.** The key object is the set of *admissible segmentations* of the token stream: any partition into contiguous segments such that every segment boundary is a *legal* boundary (a position where the token-level transition structure is preserved — i.e., no boundary is placed inside an atomic tool-call payload or inside a special-token span). Refinement/coarsening of this partition is exactly the operation that changes turn boundaries without changing the underlying transitions. The boundary-atomicity mask $m_t$ is constructed by a deterministic recipe from the tokenizer and tool-call grammar: $m_t=1$ iff token $t$ is not inside a tool-call payload span (delimited by the tool-call open/close special tokens), not inside any special-token span, and not at a reward-reset boundary (a token position immediately following a tool observation or environment reset, identified by the environment's step boundary metadata). This mask is fixed before training and is not learned.

```pseudocode
input: tokens x_{1:T}, behavior logprobs log pi_b(x_t|h_t),
       current logprobs log pi_theta(x_t|h_t), per-token reward r_t,
       per-token critic V_t, legal-boundary mask m_t in {0,1}

# Phase 1: enumerate the segmentation equivalence class
B = {t : m_t = 1}                      # legal boundary positions
S = all partitions of [1..T] whose cuts lie in B   # the equivalence class
# (in practice S is represented implicitly; see Phase 3)

# Phase 2: segmentation-invariant per-token advantage
#   For each segmentation s in S, the turn-anchored advantage of token t is
#   the turn-level TD residual of the segment containing t, spread over tokens.
for each s in S:
    for each segment [a,b] in s:
        delta_s(a,b) = (sum_{t=a}^{b} r_t + V_{b+1}) - V_a
        A_s(t) = delta_s(a,b) / (b - a + 1)   for t in [a,b]
A_inv(t) = (1/|S|) * sum_{s in S} A_s(t)          # marginalize over S

# Phase 3: closed-form computation of A_inv without enumerating S
#   A_inv(t) = sum over legal segment [a,b] containing t of
#              P_s(a,b) * delta(a,b)/(b-a+1),
#   where P_s(a,b) is the uniform-over-partitions probability that [a,b]
#   is a segment.  P_s(a,b) factorizes over the legal-boundary lattice and
#   is computed by a forward-backward pass over B in O(T * |B|).
A_inv = forward_backward_marginalize(r, V, m)      # O(T * |B|)

# Phase 4: segmentation-invariant turn-level importance ratio
#   rho_inv(t) = exp( mean over segmentations of the segment-level log-ratio
#                    of the segment containing t )
#   computed by the same forward-backward marginalization over log-ratios.
log_ratio(t) = log pi_theta(x_t|h_t) - log pi_b(x_t|h_t)
rho_inv(t)   = exp( forward_backward_marginalize_logratio(log_ratio, m) )

# Phase 5: unchanged clipped surrogate with invariant quantities
L = -min( rho_inv(t) * A_inv(t),
          clip(rho_inv(t), 1-eps, 1+eps) * A_inv(t) )
update theta by gradient of L
```

Hyperparameters: clip range $\epsilon$ (inherited from the base optimizer, e.g., 0.2); a boundary-atomicity mask $m_t$ fixed by the tokenizer/tool-call grammar and environment step-boundary metadata (not learned); no extra temperature or mixing weight is introduced, because the marginalization is a uniform average over the equivalence class by construction.

**Why this design.** We chose to *marginalize analytically over the segmentation equivalence class* rather than to *learn a segmentation* or to *fix a canonical segmentation*, because the ambiguity is a nuisance variable created by an external process (tool-call formatting, truncation) that carries no information about the optimal policy, so estimating it would add variance without adding signal — accepting the cost that the equivalence class must be specified by a legal-boundary mask rather than discovered. We chose a *uniform* measure over admissible segmentations rather than a learned or length-biased measure, because a uniform measure is the unique choice under which refinement and coarsening of the partition provably cancel in the expectation, which is the invariance we need — accepting the cost that genuinely informative boundary cues (e.g., a tool-call boundary being more semantically meaningful) are not exploited. We chose a *forward-backward marginalization* over the legal-boundary lattice rather than Monte-Carlo enumeration of segmentations, because the lattice factorizes and the marginal can be computed in $O(T\cdot|B|)$ with no sampling noise — accepting the cost that the implementation must respect the boundary-atomicity mask, so malformed masks silently break the invariance. This is *not* a variant of MPO's multiscale advantage estimation: MPO averages credit across tokens *within a fixed turn*, whereas SIRCA averages across *turn partitions themselves*, and the object being averaged over (the segmentation equivalence class) does not exist in MPO, T-PPO, or GSPO, so a domain expert would not describe SIRCA as a reparameterization of any of them.

**Why it measures what we claim.** The quantity $\hat{A}^{\text{inv}}_t$ measures *segmentation-invariant credit* because the uniform average over the equivalence class is, by construction, the unique functional of the token-level TD residuals that is unchanged under refinement/coarsening of any partition whose cuts lie in the legal-boundary set; this equivalence holds under the assumption that all admissible segmentations are *a priori* equally plausible descriptions of the same underlying token-level transition structure, and this assumption fails when the environment emits a boundary cue that genuinely correlates with reward structure (e.g., a tool-call boundary that reliably precedes a reward reset), in which case $\hat{A}^{\text{inv}}_t$ reflects an average over segmentations that are not actually exchangeable and under-weights the informative boundary. The quantity $\rho^{\text{inv}}_t$ measures *segmentation-invariant policy ratio* because the same forward-backward marginalization applied to segment-level log-ratios yields the expectation of the segment-level ratio over the equivalence class, which equals the token-level ratio whenever the partition is trivial and stays fixed under refinement; this equivalence holds under the assumption that the segment-level log-ratio factorizes additively over tokens within a segment, and this assumption fails when the policy's log-prob is not additive across a segment (e.g., a tool-call payload whose likelihood is computed by an external parser), in which case $\rho^{\text{inv}}_t$ reflects a ratio computed on a partially observed likelihood and can drift from the true policy ratio.

**(D) Named conditions and failure modes.** The invariance claim rests on two named conditions: (C1) *Exchangeability*: all segmentations in the equivalence class are a priori equally plausible descriptions of the same token-level transition structure. (C1) fails when a boundary cue correlates with reward resets (e.g., a tool-observation boundary that reliably precedes a reward reset), in which case the uniform measure assigns mass to segmentations that cut across the reward-reset boundary and $\hat{A}^{\text{inv}}_t$ systematically mixes pre- and post-reset credit. To prevent this, the boundary-atomicity mask $m_t$ includes reward-reset boundaries as hard cuts (i.e., $m_t=0$ at any token position immediately following a tool observation or environment reset), so the equivalence class contains only segmentations that do not cut across the informative boundary; invariance is then claimed only over the residual nuisance class (formatting/truncation splits). (C2) *Additivity*: the segment-level log-ratio factorizes additively over tokens within a segment. (C2) fails when an external parser computes the likelihood of a tool-call payload, in which case $\rho^{\text{inv}}_t$ is computed on a partially observed likelihood and can drift from the true policy ratio. To prevent this, the ratio term is computed on the token-level ratio directly for any segment that contains a tool-call payload (i.e., the forward-backward marginalization is applied only to segments whose tokens are all outside tool-call payload spans; for payload tokens, $\rho^{\text{inv}}_t$ is set to the token-level ratio $\exp(\log\pi_\theta(x_t|h_t) - \log\pi_b(x_t|h_t))$).

## Contribution

(1) SIRCA, a segmentation-invariant advantage estimator and turn-level importance ratio that marginalize analytically over the equivalence class of admissible turn partitions via a forward-backward pass on the legal-boundary lattice, yielding per-token learning signals that are unchanged under any refinement or coarsening of the partition. (2) A design principle: turn-anchored credit assignment should be defined on the *equivalence class* of segmentations rather than on a single realized segmentation, which removes a variance source that is invisible to fixed-segmentation estimators such as MPO and GSPO. (3) A boundary-atomicity mask specification (tokenizer/grammar-derived) that makes the admissible segmentation class well-defined and reproducible across tool-call and truncation settings.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-turn tool-use agent tasks with variable turn segmentation (e.g., ALFWorld, WebShop, and a synthetic tool-call benchmark with controlled reward-reset boundaries) | Boundary ambiguity is the failure mode under test |
| Primary metric | Success rate under re-segmented evaluation rollouts (perturbing formatting/truncation splits only, holding reward-reset boundaries fixed) | Directly probes segmentation invariance of credit |
| Baseline | MPO (turn-anchored multiscale advantage) | Tests claim that within-turn averaging is insufficient |
| Baseline | T-PPO (truncated rollouts, EGAE) | Tests claim that truncation boundaries inject segmentation variance |
| Baseline | GSPO (token-level ratio, fixed segmentation) | Tests claim that fixed-segmentation ratios break under refinement |
| Ablation-of-ours | SIRCA with canonical fixed segmentation (no marginalization) | Isolates the marginalization as the causal ingredient |
| Ablation-of-ours | SIRCA with uniform measure over all legal partitions (no reward-reset hard cuts) | Tests the exchangeability assumption (C1) |
| Ablation-of-ours | SIRCA with token-level ratio for all tokens (no ratio marginalization) | Tests the additivity assumption (C2) |
| Diagnostic | Re-segment the same trajectories and measure advantage/ratio drift for MPO, T-PPO, GSPO | Quantifies segmentation-induced variance before the fix |

### Why this setup validates the claim

The central claim is that ambiguity over where turns begin and end is a nuisance variable, and that marginalizing analytically over the admissible segmentation class yields credit and ratios that are unchanged under refinement/coarsening, which in turn reduces gradient variance and improves policy learning on tasks where segmentation is externally imposed. To falsify this, the setup must make segmentation vary in a way that is orthogonal to the underlying transition structure but correlated with how baselines compute advantages. MPO tests the sub-claim that averaging within a fixed turn is not enough: if MPO's within-turn averaging already absorbed boundary noise, our marginalization would add nothing. T-PPO tests the sub-claim that truncation boundaries are a source of the same nuisance: if EGAE's truncated advantage were already segmentation-invariant, our gain over it should vanish. GSPO tests the sub-claim that token-level ratios anchored to a canonical segmentation drift under refinement. The ablation — SIRCA with a fixed canonical segmentation instead of marginalization — is the decisive control: it holds the surrogate, clip range, and critic fixed, so any gap between it and full SIRCA is attributable to the marginalization itself rather than to the clipped surrogate or the boundary mask. The additional ablations test the named conditions: SIRCA with uniform measure over all legal partitions (no reward-reset hard cuts) tests whether exchangeability (C1) is necessary; SIRCA with token-level ratio for all tokens tests whether additivity (C2) is necessary. The diagnostic experiment measures advantage/ratio drift under re-segmentation for each baseline, grounding the effect size. Success rate under re-segmented evaluation rollouts is the right primary metric because it is measured on rollouts whose segmentation is perturbed relative to training, so a method that has genuinely learned segmentation-invariant credit should retain performance where a method that overfit to one segmentation degrades. If the marginalization is doing what we claim, the effect should be concentrated on the re-segmented subset and near-parity on the canonical subset — a differential pattern, not a uniform lift.

### Expected outcome and causal chain

**vs. MPO** — On a case where a tool-call payload spans a token region that MPO's fixed turn boundary cuts through, MPO produces a turn-level TD residual that mixes reward from the tool result with reward from the surrounding reasoning, so its advantage for the payload tokens is diluted by unrelated credit because its averaging is confined to one fixed turn. Our method instead marginalizes the TD residual over all legal segmentations containing those tokens because the forward-backward pass weights each admissible segment by its uniform-over-partitions probability, so we expect a noticeable gap on the re-segmented subset containing tool-call-spanning boundaries but near-parity on the canonical subset where MPO's fixed turn is already legal.

**vs. T-PPO** — On a case where a truncated rollout ends mid-turn and the next rollout resumes, T-PPO produces an advantage whose truncation boundary is treated as a hard segment edge, so tokens near the boundary receive credit computed against a value estimate from a different rollout context because EGAE's advantage is anchored to the truncation point rather than marginalized over alternative legal cuts. Our method instead computes the invariant advantage by averaging over segmentations that do not privilege the truncation point because the legal-boundary mask excludes only atomic spans and reward-reset boundaries, not arbitrary truncation positions, so we expect a visible reduction in variance-driven instability on long-horizon tasks with frequent truncation, showing up as a smaller gap between early and late training success on the truncated subset.

**vs. GSPO** — On a case where the same token stream admits two legal segmentations that differ only in where a reasoning-to-tool transition is cut, GSPO produces a token-level ratio that is identical under both segmentations only if the policy log-prob factorizes the same way, but its ratio is computed against a single canonical segmentation, so the ratio drifts when the evaluation segmentation is refined because the segment-level log-ratio is not re-marginalized. Our method instead computes the invariant ratio by forward-backward marginalizing segment-level log-ratios over the equivalence class because the marginal equals the token-level ratio on trivial partitions and stays fixed under refinement, so we expect GSPO to lose success rate on the re-segmented subset while SIRCA holds, with the gap widening as the number of legal boundaries per turn grows.

**vs. SIRCA with uniform measure over all legal partitions (no reward-reset hard cuts)** — On a case where a tool-observation boundary reliably precedes a reward reset, the uniform measure over all legal partitions assigns mass to segmentations that cut across the reward-reset boundary, so the marginalized TD residual systematically mixes pre- and post-reset credit because the exchangeability assumption (C1) fails. Our method instead includes reward-reset boundaries as hard cuts in the boundary-atomicity mask, so the equivalence class contains only segmentations that do not cut across the informative boundary, and we expect the full SIRCA to outperform this ablation on tasks with reward-reset boundaries, with the gap concentrated on the re-segmented subset.

**vs. SIRCA with token-level ratio for all tokens (no ratio marginalization)** — On a case where a tool-call payload is parsed by an external parser, the segment-level log-ratio does not factorize additively over tokens, so the marginalized ratio drifts from the true policy ratio because the additivity assumption (C2) fails. Our method instead computes the token-level ratio directly for payload tokens, so the ratio is verifiably additive, and we expect the full SIRCA to outperform this ablation on tasks with externally-parsed tool-call payloads, with the gap concentrated on the re-segmented subset.

### What would falsify this idea

If the gain over MPO, T-PPO, and GSPO is uniform across the canonical and re-segmented subsets rather than concentrated on the re-segmented subset where the predicted boundary-ambiguity failure occurs, the marginalization is not the causal ingredient and the central claim is wrong. Equivalently, if the fixed-segmentation ablation matches full SIRCA on the re-segmented subset, the equivalence-class averaging adds no signal beyond the clipped surrogate.

## References

1. Multiscale Policy Optimization for Long-Horizon LLM Agents
2. Truncated Proximal Policy Optimization
3. Group Sequence Policy Optimization
4. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
5. OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text
6. Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations
7. Solving Quantitative Reasoning Problems with Language Models
