# Martingale-Preserving Credit Assignment for Compacted Agent Contexts

## Motivation

Long-horizon language agents increasingly rely on context compaction (summarization, truncation, retrieval-based rewriting) to fit turn histories into a fixed window, yet every turn-level credit-assignment scheme in current RL pipelines — MPO's turn-anchored Multiscale Advantage Estimation, T-PPO's EGAE over truncated rollouts, and turn-level importance sampling — silently assumes the credit-assignment context is complete. The structural problem across these paper-trees is that no branch formalizes which statistics the advantage estimator depends on, so compaction is applied as an empirical hope and the resulting advantage estimates are biased in a way no one can measure. This paper makes 'what compaction destroys' a well-posed, testable condition by identifying the turn-local score-martingale filtration as the minimal sufficient statistic and correcting the estimator when compaction violates it.

## Key Insight

A turn-level advantage estimator is invariant to a compaction operator exactly when the operator preserves the turn-local score-martingale filtration, so the Doob-decomposition residual of the compacted filtration is a computable, closed-form bias term that can be subtracted rather than ignored.

## Method

## MARTY — Martingale-Anchored Residual-corrected Turn-level credit (Y = turn advantage)

**(A) What it is.** MARTY takes a compacted turn history `C(h_{1:t})` produced by any compaction operator (summarizer, truncator, or retrieval rewriter) and a per-turn reward stream, and returns a turn-advantage estimate `Â_t` that is invariant to `C` whenever `C` preserves the turn-local score-martingale filtration, and is bias-corrected by a Doob-decomposition residual when `C` violates it only on a measurable credit-relevant subspace. **Load-bearing assumption (stated explicitly):** the correction is valid only for compaction operators that are *non-anticipative* — the compacted context for turn `t`, `C(h_{1:T})|_t`, depends only on `h_{1:t}`. Under non-anticipativity, the predictable compensator `pred_t = E[delta_t | F_{t-1}]` captures the entire systematic (correctable) bias in the turn advantage, and the martingale remainder `mart_t = delta_t - pred_t` has zero conditional mean under the turn-local filtration. For anticipative operators (e.g., a retrieval rewriter that selects context using turns `> t`), this assumption fails: `delta_t` is measurable w.r.t. a strictly larger look-ahead filtration, `mart_t` carries nonzero mean, and the correction is incomplete. We therefore (i) restrict the invariance claim and correction to non-anticipative operators, (ii) report anticipative operators as out of scope for the closed-form correction, and (iii) provide a held-out-batch estimate of the full residual mean `E[delta_t]` as a diagnostic for how much bias remains uncorrected under anticipative operators.

**(B) How it works.**

```pseudocode
Input: turn history h_{1:T}, per-turn rewards r_{1:T}, compaction operator C,
       critic V_phi, credit-relevant subspace projector P (see below),
       residual clip kappa=2.0, subspace threshold tau=0.05,
       non-anticipativity flag na_C (true if C(h_{1:T})|_t depends only on h_{1:t})

# Phase 1: turn-local score-martingale filtration
for t in 1..T:
    S_t   = turn-score statistic = sum of token log-probs of assistant turn t under pi_old
    F_t   = sigma(S_1..S_t, r_1..r_{t-1})            # turn-local filtration
    M_t   = S_t - E[S_t | F_{t-1}]                    # score martingale increment

# Phase 2: compaction-invariance test
C_h   = C(h_{1:T})                                    # compacted context
for t in 1..T:
    Mhat_t = M_t evaluated on C_h                       # re-derive increment from compacted ctx
    delta_t = M_t - Mhat_t                              # per-turn invariance residual

# Phase 3: Doob-decomposition residual on credit-relevant subspace
A_t = sum_{s>=t} gamma^{s-t} r_s - V_phi(h_{1:t})       # MPO-style turn-anchored advantage
P_t = P( delta_t )                                     # project residual onto credit-relevant subspace
if ||P_t|| <= tau:
    Ahat_t = A_t                                        # invariant branch: no correction
else:
    # Doob decomposition: delta_t = predictable part + martingale part
    pred_t = E[delta_t | F_{t-1}]                       # predictable compensator
    mart_t = delta_t - pred_t                           # martingale remainder
    if na_C:
        Ahat_t = A_t - clip(pred_t, -kappa, kappa)      # subtract only compensator
    else:
        # anticipative operator: closed-form correction out of scope;
        # use held-out-batch estimate of full residual mean as diagnostic
        pred_t_ho = mean_{batch} delta_t                # held-out estimate of E[delta_t]
        Ahat_t = A_t - clip(pred_t_ho, -kappa, kappa)   # diagnostic correction only

# Phase 4: turn-level importance weighting (unchanged from MPO)
rho_t = pi_theta(turn_t | C_h) / pi_old(turn_t | h_{1:t})
update theta with surrogate: rho_t * Ahat_t
```

**Credit-relevant subspace `P`.** `P` is the orthogonal projector onto the span of turn-score statistics `{S_s : s <= t}` that carry nonzero reward covariance, estimated once per batch by regressing `r_{t:T}` onto `{S_s}` and keeping coordinates whose standardized coefficient exceeds `tau=0.05`. This is a fixed linear operator, not a learned head.

**(C) Why this design.** We chose a martingale-filtration invariant over a learned sufficiency predictor because the filtration is a structural object that makes the invariance claim checkable without distributional assumptions, accepting the cost that the estimator needs per-turn score statistics that some agent harnesses do not expose. We chose to correct only the predictable compensator `pred_t` rather than the full residual `delta_t` because the martingale remainder `mart_t` carries zero conditional mean under the filtration and subtracting it would inject variance without removing bias, accepting the cost that residual bias from the martingale part is left uncorrected by design. We chose a fixed linear projector `P` over a learned disentanglement head because the credit-relevant subspace is defined by a regression coefficient threshold that is inspectable and does not require the (false) assumption that turn-score statistics are conditionally independent of reward, accepting the cost that `P` may miss nonlinear credit channels. This is structurally distinct from MPO's Multiscale Advantage Estimation: MPO computes `A_t` on the full history and never asks whether the compacted context preserves the statistics `A_t` depends on, whereas MARTY makes the preservation condition the object being tested and corrected. A domain expert would not describe MARTY as a variant of MPO because MPO has no compaction operator in its estimator and no residual term; the estimator lives on a different filtration.

**(D) Why it measures what we claim.** The per-turn invariance residual `delta_t = M_t - Mhat_t` measures *compaction-induced credit distortion* because it is the difference between the score-martingale increment evaluated on the full and compacted histories, which is zero exactly when the compaction preserves the turn-local filtration; this assumption fails when the compaction operator rewrites turn content in a way that changes the conditional expectation `E[S_t | F_{t-1}]` (e.g., a summarizer that drops a tool-call observation feeding turn `t`'s score), in which case `delta_t` reflects the summarizer's rewriting error rather than a genuine change in credit. The predictable compensator `pred_t = E[delta_t | F_{t-1}]` measures *systematic (correctable) bias* because it is the `F_{t-1}`-measurable part of the Doob decomposition, which is exactly the part of the distortion that a bias correction can remove without touching the martingale noise; this assumption fails when the compaction operator is *anticipative* — i.e., its error is state-dependent on future turns not captured by `F_{t-1}` (e.g., a retrieval rewriter that selects context based on the future turn) — in which case `pred_t` under-corrects and the residual `mart_t` carries nonzero mean. The credit-relevant subspace projector `P` measures *which distortion channels matter for the advantage* because it keeps only score-statistic coordinates with nonzero reward covariance, which is exactly the subspace where a bias in `Mhat_t` propagates to `A_t`; this assumption fails when reward covariance is estimated from too few turns (small-batch regime), in which case `P` reflects sampling noise rather than credit relevance and the correction is applied to the wrong coordinates.

## Contribution

(1) A turn-advantage estimator, MARTY, that is invariant to any compaction operator preserving the turn-local score-martingale filtration and that subtracts a Doob-decomposition compensator residual when the invariant is violated on a measurable credit-relevant subspace. (2) A concrete design principle: the credit-relevant subspace is identified by a fixed linear projector from reward-covariance regression, not a learned head, making the correction inspectable and decoupled from the (false) independence assumption that dooms learned disentanglement approaches. (3) An operational definition of 'credit-relevant compaction loss' as the predictable compensator of the score-martingale residual, which turns the previously empirical question of whether a summarizer hurts RL into a per-turn diagnostic quantity that can be logged and thresholded.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Long-horizon tool-use with summarizer compaction | Compaction is the variable MARTY corrects for |
| Primary metric | Turn-level advantage bias vs. full-history oracle | Directly measures the claimed distortion |
| Baseline | MPO (Multiscale Advantage Estimation) | Tests compaction-blind estimator gap |
| Baseline | GRPO with token-level ratios | Tests turn-vs-token credit granularity gap |
| Baseline | T-PPO truncated rollouts | Tests truncation-induced filtration break |
| Baseline | MARTY with anticipative retrieval rewriter (diagnostic) | Tests the non-anticipativity assumption |
| Ablation-of-ours | MARTY minus compensator correction | Isolates the Doob-residual contribution |
| Ablation-of-ours | MARTY with held-out residual-mean correction | Tests the anticipative-operator fallback |

### Why this setup validates the claim

The central claim is that MARTY's turn advantage stays close to the full-history oracle advantage precisely when compaction preserves the turn-local score-martingale filtration, and that subtracting only the predictable compensator recovers the systematic part of the distortion. A dataset that pairs a long-horizon tool-use environment with an aggressive summarizer compaction operator is the right testbed because it creates a measurable gap between the full history and the compacted context that a compaction-blind estimator cannot see. MPO tests the sub-claim that estimating advantage on the compacted context alone is biased: it has no compaction operator and no residual term, so any oracle gap it shows is exactly the distortion MARTY predicts and corrects. GRPO with token-level ratios tests the sub-claim that turn-level credit granularity matters: its token ratios average over the same distortion but at finer granularity, so if MARTY's advantage is closer to oracle than GRPO's, the gain is attributable to turn-local filtration rather than to a generic variance reduction. T-PPO tests the sub-claim that truncation is a special case of filtration violation: its truncated rollouts drop the same turn-score statistics that a summarizer drops, so a smaller MARTY gain against T-PPO than against MPO would localize the effect to summarizer-style rewriting. The anticipative retrieval-rewriter baseline directly tests the load-bearing non-anticipativity assumption: because its compaction uses future turns, the closed-form compensator is out of scope, and we expect the held-out residual-mean diagnostic to show a larger uncorrected bias than the non-anticipative summarizer case. The ablation (MARTY minus compensator correction) is the load-bearing control: it keeps the invariance test but skips the Doob correction, so if the full method beats the ablation, the compensator is doing real bias-removal work rather than the invariance test alone. The held-out residual-mean ablation tests the fallback correction for anticipative operators, isolating how much of the bias is recoverable without the F_{t-1}-measurability guarantee. The primary metric — turn-level advantage bias against the full-history oracle — is the right detector because MARTY's claim is about estimator bias, not final return; a method can win on return through variance reduction while still being biased, and the oracle gap isolates the specific quantity the method claims to fix. Return and pass-rate are recorded but are secondary, since they confound bias correction with exploration noise.

### Expected outcome and causal chain

**vs. MPO (Multiscale Advantage Estimation)** — On a case where a summarizer drops a tool-call observation that feeds turn `t`'s score, MPO produces a turn advantage computed on the compacted context with no notion that the conditional expectation `E[S_t | F_{t-1}]` has shifted, because MPO's estimator has no compaction operator and no residual term and therefore cannot detect the filtration break. Our method instead re-derives the martingale increment on the compacted context, forms `delta_t`, projects it onto the credit-relevant subspace, and subtracts only the predictable compensator, because the Doob decomposition isolates the `F_{t-1}`-measurable bias that a correction can remove without injecting martingale noise. We expect a noticeable oracle-advantage gap on the subset of turns whose compaction drops a reward-relevant observation, and near-parity on turns where the summarizer preserves the score statistics — a gap that concentrates where the predicted failure mode lives rather than being uniform.

**vs. GRPO with token-level ratios** — On a case where a long assistant turn contains one decisive tool call and many filler tokens, GRPO produces a turn advantage that averages the distortion over token-level ratios, because its credit granularity is token-level and it has no turn-local filtration to test against, so the compaction-induced shift is smeared across tokens rather than localized to the turn whose score statistic moved. Our method instead keeps the turn as the credit unit and tests invariance at the turn-score level, because the filtration `F_t` is defined over turn-score statistics, so the distortion is attributed to the specific turn whose conditional expectation changed. We expect MARTY to show a smaller oracle-advantage bias than GRPO on multi-turn episodes with heterogeneous turn lengths, with the gap widening as the number of compaction events per episode grows and shrinking toward parity on single-turn or uniformly short-turn episodes.

**vs. T-PPO truncated rollouts** — On a case where truncation cuts off the turn immediately preceding a high-reward turn, T-PPO produces a truncated advantage that silently conditions on a shorter history, because its truncated rollouts and EGAE break the same turn-local filtration that a summarizer breaks, but it treats the truncation as a fixed rollout budget rather than as a measurable distortion. Our method instead treats truncation as one compaction operator among several, re-derives the martingale increment under it, and corrects only the predictable compensator, because the Doob decomposition applies to any operator that violates the filtration on a measurable subspace. We expect MARTY's gain over T-PPO to be smaller than its gain over MPO, since T-PPO's truncation is a milder filtration violation than summarizer rewriting, with the residual gap concentrated on episodes where the truncation boundary lands on a reward-relevant turn.

**vs. MARTY with anticipative retrieval rewriter (diagnostic)** — On a case where the retrieval rewriter selects context for turn `t` using turns `> t`, the closed-form compensator `pred_t = E[delta_t | F_{t-1}]` is out of scope because `delta_t` is measurable w.r.t. a look-ahead filtration, so the martingale remainder `mart_t` carries nonzero mean and the correction is incomplete. Our method instead flags the operator as anticipative, applies the held-out residual-mean diagnostic correction, and reports the residual bias as a measure of the non-anticipativity assumption's cost. We expect the anticipative case to show a larger uncorrected oracle-advantage bias than the non-anticipative summarizer case, with the held-out diagnostic recovering only part of the gap, confirming that the closed-form correction is valid only under non-anticipativity.

### What would falsify this idea

If MARTY's oracle-advantage bias reduction is uniform across compaction-preserving and compaction-violating subsets rather than concentrated on the violating subset, the claim that the invariance residual measures compaction-induced credit distortion is wrong. If the full method and the MARTY-minus-compensator ablation are indistinguishable on the violating subset, the claim that the predictable compensator removes systematic bias is wrong and the gain is attributable to the invariance test alone. If the anticipative retrieval-rewriter diagnostic shows no larger residual bias than the non-anticipative summarizer, the non-anticipativity assumption is not load-bearing and the stated scope restriction is unnecessary.

## References

1. Multiscale Policy Optimization for Long-Horizon LLM Agents
2. Truncated Proximal Policy Optimization
3. Group Sequence Policy Optimization
4. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
5. OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text
6. Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations
7. Solving Quantitative Reasoning Problems with Language Models
