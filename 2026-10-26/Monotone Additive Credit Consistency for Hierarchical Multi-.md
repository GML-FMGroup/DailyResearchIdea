# Monotone Additive Credit Consistency for Hierarchical Multi-Agent On-Policy Distillation

## Motivation

Hierarchical multi-agent systems trained with on-policy distillation (MultiAgent-OPD) and on-policy distillation frameworks like GKD both supervise student-generated tokens with teacher scores, but they treat each sub-task's token scores as independently decomposable: the credit assigned to a Searcher's step and the credit assigned to the composed Answerer sequence are never required to agree. This shared residual gap means that when sub-tasks compose, the sum of step-level credits can drift arbitrarily from the sequence-level teacher score, so the process supervision silently misattributes credit across sub-task boundaries and the composition invariant that would make credit assignment consistent is simply absent from the training objective.

## Key Insight

Enforcing that step-level contrastive advantages sum to the sequence-level teacher score up to a bounded residual turns credit decomposition from an unconstrained per-token regression into a constrained additive decomposition, so any misattribution across sub-task boundaries becomes a directly penalized training signal rather than an invisible bias.

## Method

**What it is.** We propose MACC (Monotone Additive Credit Consistency), a training objective for a shared-parameter Leader-Searcher-Answerer team. Inputs: student-generated search and summary tokens, the frozen Teacher's sequence-level score for the full trajectory, and a counterfactual contrastive scorer that produces step-level advantages. Output: a set of step-level credits whose sum is structurally tied to the sequence-level teacher score, plus a scalar loss used to update the team.

**Load-bearing assumption (stated explicitly).** MACC assumes the frozen Teacher's sequence-level score is a valid additive potential over trajectory steps: that leave-one-step-out counterfactual deltas sum to the sequence score up to bounded noise, so the residual `target - sum_adv` measures student credit misattribution rather than teacher non-additivity. This assumption fails when the Teacher rewards a global property (e.g., answer fluency) that no single step contributes to, or when steps are strongly synergistic (removing step i invalidates later steps), in which case the residual reflects teacher non-additivity or synergy rather than student misattribution. We therefore calibrate teacher additivity per domain before training (step 0 below) and gate the consistency penalty on the measured additivity gap.

**How it works.**

```pseudocode
# Per training episode
# 0. Teacher-additivity calibration (run once per domain, before training)
#    Sample N_calib = 512 teacher-scored trajectories from the training set
#    (no student involved; teacher scores full trajectories and leave-one-step-out
#    counterfactuals). For each trajectory compute
#       gap = |tau_score - sum_i (tau_score - teacher.sequence_score(question, tau_{-i}))|
#    Report mean and 90th-percentile gap. Set lambda = 0.5 if mean gap < 0.1 * std(tau_score)
#    over the calibration set; else set lambda = 0 (consistency penalty disabled) and log
#    the measured gap as a diagnostic. This gate prevents the penalty from firing on
#    teacher non-additivity.
# 1. Student team generates trajectory tau = (s_1..s_k search steps, a answer)
tau = team.rollout(question)
# 2. Frozen Teacher scores the FULL sequence
tau_score = teacher.sequence_score(question, tau)          # scalar
# 3. Counterfactual contrastive scorer: for each step i,
#    contrast tau against a leave-step-i-out counterfactual tau_{-i}
for i in 1..k:
    tau_minus_i = team.rollout_without_step(question, i)   # counterfactual
    delta_i = tau_score - teacher.sequence_score(question, tau_minus_i)
    # step-level advantage from contrastive pair
    adv_i = delta_i - baseline_i                            # baseline = running mean, beta=0.9
# 4. Sequence-level teacher score decomposed additively
target = tau_score
sum_adv = sum(adv_i for i in 1..k)
# 5. Monotone additive consistency residual
residual = target - sum_adv
# 6. Training objective
L_consistency = (residual ** 2)                          # bounded residual penalty, lambda=0.5 (gated by step 0)
L_policy = -sum(logprob_i * adv_i)                       # on-policy distillation / RL
L_total = L_policy + lambda * L_consistency
# 7. Monotonicity: enforce adv_i is non-decreasing in delta_i via a
#    monotone projection (isotonic clip) before use
adv_i = isotonic_clip(adv_i, delta_i)                    # guarantees adv_i <= adv_j if delta_i <= delta_j
update(team, L_total)
```

**Why this design.** We chose a leave-one-step-out counterfactual contrast (rather than a learned value head) because the counterfactual gives a directly interpretable marginal contribution of each step, accepting the cost of k extra rollouts per episode; this is the same cost profile as on-policy distillation in GKD, so it does not change the asymptotic training budget. We chose a squared residual penalty (rather than a hard equality constraint) because the Teacher's sequence score is itself noisy across rollouts, so a bounded residual tolerates teacher variance while still penalizing systematic drift, accepting that a small bias can persist when teacher noise is large. We chose isotonic clipping (rather than a free learned advantage) because monotonicity of advantage in the contrastive delta is the structural property that makes the additive decomposition interpretable as credit, accepting a small expressivity loss when the true credit function is non-monotone. The step-0 calibration gate is a direct consequence of the load-bearing assumption: because the residual is only interpretable as student misattribution when the teacher is approximately additive, we measure teacher additivity on 512 teacher-only trajectories and disable the penalty (lambda=0) when the mean gap exceeds 10% of the teacher-score standard deviation, so the penalty cannot fire on teacher non-additivity. A domain expert would not describe MACC as a variant of GKD or MultiAgent-OPD: those methods supervise tokens independently with teacher scores and never impose an additive consistency constraint linking the sum of step advantages to the sequence score, so the residual term is a new structural object, not a reweighting of existing losses.

**Why it measures what we claim.** The residual `target - sum_adv` measures cross-sub-task credit consistency because it is exactly the violation of the additive decomposition invariant under the assumption that the Teacher's sequence score is a valid additive potential over steps; this assumption fails when the Teacher's score is non-additive (e.g., it rewards a global property like answer fluency that no single step contributes to), in which case the residual reflects teacher non-additivity rather than student credit misattribution. The step-0 calibration gate makes this failure mode measurable: we report the mean and 90th-percentile additivity gap on 512 teacher-only trajectories and disable the penalty when the mean gap exceeds 10% of the teacher-score standard deviation, so the residual is only used as a training signal in domains where it tracks misattribution. The counterfactual delta `tau_score - score(tau_{-i})` measures the marginal causal contribution of step i because it isolates the score change under a leave-one-out intervention on the trajectory; this assumption fails when steps are strongly synergistic (removing step i also invalidates later steps), in which case delta_i reflects a joint rather than marginal effect. The isotonic clip measures monotone credit ordering because it enforces that a larger contrastive delta yields a larger advantage under the assumption that the Teacher's score is monotone in step quality; this fails when the Teacher is locally non-monotone, in which case the clip flattens genuine orderings.

## Contribution

(1) A monotone additive consistency loss that constrains the sum of counterfactual step-level advantages to match the sequence-level teacher score up to a bounded residual, turning credit decomposition into a structurally constrained rather than unconstrained regression. (2) A design principle showing that leave-one-step-out counterfactual contrasts plus isotonic clipping yield step credits that are both causally grounded and monotone in the teacher score, which independent per-token distillation cannot produce. (3) An end-to-end training procedure for shared-parameter hierarchical multi-agent teams that integrates this consistency constraint with on-policy distillation and RL signals.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-hop QA with search (HotpotQA + MuSiQue mix) | Multi-step trajectories expose per-step credit attribution |
| Dataset (second domain) | Tool-use trajectories (ToolBench subset, 2k episodes) | Tests residual as reusable diagnostic beyond QA |
| Primary metric | Macro avg@4 exact-match accuracy | Matches hierarchical MAS eval; detects credit misallocation |
| Diagnostic metric | Residual magnitude vs. downstream error correlation (Pearson r) | Separates teacher non-additivity from student misattribution |
| Baseline | MultiAgent-OPD (C-MAS + RL) | Tests whether additive consistency beats token-level distillation |
| Baseline | GKD / on-policy distillation | Tests whether residual constraint adds over pure OPD |
| Baseline | Prompted Critic (no trained credit) | Tests whether learned contrastive credit beats prompted critique |
| Ablation-of-ours | MACC w/o isotonic clip (free advantage) | Isolates contribution of monotone credit ordering |
| Ablation-of-ours | MACC w/o calibration gate (lambda fixed at 0.5) | Isolates contribution of teacher-additivity gating |
| Ablation-of-ours | MACC with 1 extra rollout (masked-prefix counterfactual) | Tests cheaper approximation at k=1-2 extra rollouts |
| Synthetic control | Known additive teacher vs. known non-additive teacher (2k synthetic trajectories each) | Shows residual tracks misattribution only in additive case |

### Why this setup validates the claim

The central claim is that tying step-level credits to the sequence-level teacher score via an additive consistency residual produces better credit attribution than token-level supervision, and that this gap shows up specifically where multi-step credit is ambiguous. Multi-hop QA is the right testbed because trajectories contain search steps whose marginal contribution is genuinely heterogeneous — some steps retrieve the bridging entity, others are near-redundant — so credit misallocation has measurable downstream cost. MultiAgent-OPD tests the sub-claim that additive consistency beats independent token-level teacher scores; GKD tests whether the residual term is doing work beyond plain on-policy distillation; the prompted-critic baseline tests whether learned contrastive credit beats an LLM critique heuristic. The isotonic-clip ablation is the load-bearing one: if monotone credit ordering is what makes the decomposition interpretable, removing it should degrade performance concentrated on trajectories with many low-marginal steps, not uniformly. The calibration-gate ablation and the synthetic additive-vs-non-additive teacher control directly test the load-bearing assumption: if the residual tracks misattribution only when the teacher is additive, the gate is necessary; if the residual correlates with downstream error equally in both synthetic teachers, the residual is measuring teacher non-additivity rather than student credit. The second-domain tool-use experiment tests whether the residual is a reusable diagnostic rather than a single-task trick. Macro avg@4 exact match is the right metric because credit misallocation manifests as wrong final answers on compositional questions, not as loss differences; it directly reflects whether the team answered correctly across attempts.

### Expected outcome and causal chain

**vs. MultiAgent-OPD** — On a multi-hop question where the searcher retrieves a bridging entity in step 2 but a distractor in step 3, MultiAgent-OPD supervises each token with the sequence-level teacher score, so both the bridging step and the distractor step receive the same high-level signal and the distractor's harm is never localized. Our method computes a leave-step-out counterfactual: removing step 3 raises the teacher score, yielding a negative delta_3 and negative advantage that directly suppresses the distractor behavior. We expect a noticeable accuracy gap concentrated on questions with ≥3 search steps and at least one distractor, with near-parity on single-hop questions where no credit ambiguity exists.

**vs. GKD / on-policy distillation** — On a case where the answerer produces a fluent but unsupported summary, GKD assigns high probability to those tokens because the teacher score for the full sequence is high, propagating fluent-but-wrong behavior. Our residual term penalizes the mismatch between the summed step advantages and the sequence score, so the fluency contribution that no step's marginal delta explains is absorbed into the residual rather than credited to the answerer. We expect GKD to match us on answer-fluency-heavy subsets but to trail us on subsets requiring faithful grounding, with the gap widening as trajectory length grows.

**vs. Prompted Critic** — On a case where step 1 is a correct but low-value search and step 2 is the decisive retrieval, a prompted critic tends to assign similar mid-level scores to both because it judges each step in isolation without a counterfactual anchor. Our contrastive delta directly measures that removing step 1 barely changes the teacher score while removing step 2 collapses it, so the advantage ordering is grounded in an intervention rather than a judgment. We expect a moderate but consistent gap on questions where the decisive step is not the first, and parity on questions where the first step is decisive.

**vs. MACC w/o calibration gate** — On domains where the teacher is non-additive (e.g., fluency-heavy summaries), the ungated variant fires the consistency penalty on teacher non-additivity, training the team to match an artifact of the teacher's scoring function. We expect the gated MACC to match the ungated variant on additive domains (where the gate stays on) and to outperform it on non-additive domains (where the gate disables the penalty), with the gap concentrated on trajectories where the calibration gap exceeds 10% of the teacher-score standard deviation.

**vs. MACC with 1 extra rollout (masked-prefix counterfactual)** — Replacing full leave-one-step-out rollouts with a single-pass masked-prefix counterfactual reduces the rollout budget from k to 1 extra rollout per episode. We expect this cheaper approximation to retain most of the accuracy gain on trajectories with ≤4 steps, with the gap to full MACC widening on longer trajectories where masking a prefix is a coarser intervention than a full leave-one-out rollout.

**Synthetic additive vs. non-additive teacher control** — On 2k synthetic trajectories with a known additive teacher (score = sum of per-step potentials), the residual should correlate with downstream error (Pearson r > 0.5) because it measures student misattribution. On 2k synthetic trajectories with a known non-additive teacher (score = global fluency reward), the residual should correlate weakly with downstream error (r < 0.2) because it measures teacher non-additivity. If the residual correlates equally in both, the residual is not measuring what we claim.

### What would falsify this idea

If MACC's gain over GKD and MultiAgent-OPD is uniform across trajectory-length and distractor-presence subsets rather than concentrated on multi-step, distractor-heavy questions where credit ambiguity is predicted to bite, the additive-consistency mechanism is not doing what we claim. Likewise, if the isotonic-clip ablation matches full MACC on exactly those ambiguous subsets, monotone credit ordering is not the load-bearing structural property. If the synthetic additive-vs-non-additive teacher control shows the residual correlates with downstream error equally in both teachers, the residual is measuring teacher non-additivity rather than student credit misattribution, and the load-bearing assumption is false.

## References

1. MultiAgent-OPD: End-to-End Training of Hierarchical Multi-Agent Systems with On-Policy Distillation
2. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
3. Explanations from Large Language Models Make Small Reasoners Better
