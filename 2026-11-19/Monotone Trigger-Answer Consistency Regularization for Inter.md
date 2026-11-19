# Monotone Trigger-Answer Consistency Regularization for Interactive Streaming Video Models

## Motivation

Across the proactive streaming branch (StreamBridge's decoupled activation model, LiveStar's response-silence decoding, Dispider's disentangled perception/decision/reaction) and ALIVE's BACA RL stage, the trigger decision and the response generation are optimized as separable quantities, so nothing forces a high-confidence trigger to co-occur with a high-quality answer. ALIVE's BACA combines trajectory-level and local advantages to refine decisions, responses, and silence, but its credit assignment does not couple trigger confidence to downstream answer correctness, leaving a model free to fire confidently and then answer poorly across multi-turn interaction. The root cause is structural: trigger confidence and answer quality are scored by separate heads/objectives with no shared ordering constraint, so rank inversions between them are never penalized.

## Key Insight

Within a fixed multi-turn interaction policy, trigger confidence and downstream answer correctness share the same trajectory-level credit, so enforcing rank monotonicity between them is a well-posed ordering constraint on a single policy surface rather than an attempt to manufacture independence between coupled variables.

## Method

**What it is.** We introduce MTAC (Monotone Trigger-Answer Consistency), an auxiliary ranking regularizer added to ALIVE's BACA reinforcement-learning stage. Its input is a batch of multi-turn streaming trajectories with per-turn trigger confidence scores and per-turn answer correctness scores; its output is an added loss term that penalizes rank inversions between those two scores within each trajectory, plus a calibrated trigger threshold used at inference.

**Load-bearing assumption (stated explicitly).** MTAC assumes that within a single multi-turn trajectory, the per-turn trigger confidence `c_t` and the judge-scored answer correctness `q_t` are two noisy measurements of one shared latent 'turn quality' that the policy can be pushed to order consistently, so that penalizing rank inversions between them yields a gradient that improves the policy rather than fitting judge and decoding noise. This assumption is load-bearing: if it fails, the pairwise hinge has no well-posed target. To make it testable and to guard against the two known failure modes (judge leniency toward confident-sounding wrong answers, and trigger logits being decoding-state artifacts), we add three concrete specifications inside the existing structure: (i) `q_t` is defined only on turns where a response was actually emitted, and for silent turns we estimate a counterfactual answer quality via a held-out answer head (a frozen copy of the answer head from the ALIVE-SFT checkpoint, evaluated on the silent turn's state with a forced response), so the ranking constraint is defined over a complete, comparable set of turns rather than a judge-biased subset; (ii) `c_t` is calibrated against empirical trigger precision using a reliability diagram on a held-out ALIVE-SFT split (512 turns), and the calibrated confidence `c_t_cal` is used in the hinge instead of the raw sigmoid logit; (iii) we audit judge bias by comparing `q_t` against a human-annotated subset of 200 turns (reporting Cohen's kappa) and by checking whether `q_t` is systematically higher on high-`c_t` turns after controlling for human quality. If neither a counterfactual quality estimate nor a calibrated confidence can be obtained, the pairwise ranking term is dropped in favor of a joint trigger-answer head trained on a single shared reward.

**How it works.**

```pseudocode
Input: ALIVE-RL-3K trajectories tau = {(s_t, a_t, trigger_conf c_t, answer_quality q_t)}_{t=1..T}
       BACA advantage A_t (trajectory-level + local, as in ALIVE)
       margin m = 0.1, temperature beta = 1.0, lambda = 0.5
       calibration set: 512 held-out ALIVE-SFT turns
       human-verified subset: 200 turns for judge-bias audit

Phase 0 - Calibration and target completion:
  # Calibrate c_t: fit a 1D logistic regression (Platt scaling) on held-out turns
  # mapping raw trigger logit to empirical trigger precision (reliability diagram).
  c_t_cal = Platt(trigger_head_logit(s_t))          # calibrated confidence
  # Define q_t only on emitted-response turns; for silent turns, estimate
  # counterfactual quality via a frozen answer head (ALIVE-SFT copy):
  if response emitted: q_t = normalized_answer_correctness(a_t)
  else: q_t = frozen_answer_head(s_t, forced_response)  # counterfactual estimate
  # Judge-bias audit: on 200 human-annotated turns, compute Cohen's kappa
  # between q_t and human quality; if kappa < 0.6, flag judge bias.

Phase 1 - Score collection per turn:
  for each turn t in tau:
    c_t = c_t_cal                                    # calibrated trigger confidence
    q_t = as defined in Phase 0                      # complete, comparable set

Phase 2 - Within-trajectory rank-inversion penalty:
  for each ordered pair (i, j) with c_i > c_j:
    delta_ij = max(0, m - beta*(q_i - q_j))          # hinge on quality gap
    L_rank += delta_ij * w_ij,  w_ij = softplus(c_i - c_j)

Phase 3 - Combine with BACA:
  L_total = L_BACA(A_t) + lambda * L_rank / |pairs|
  update policy with L_total

Phase 4 - Inference threshold calibration:
  fit trigger threshold theta* on held-out ALIVE-SFT split by maximizing
  monotone agreement (Spearman) between c_t_cal and q_t; deploy theta*.
```

**Why this design.** We chose a *within-trajectory pairwise hinge* over a global calibration loss (e.g., temperature scaling on trigger logits) because the motivation is about ordering *across turns of one interaction*, not marginal calibration; a global loss would let high-confidence-low-quality turns be offset by low-confidence-high-quality turns elsewhere, accepting the cost that pairwise terms scale quadratically in turns and we must subsample pairs per trajectory. We chose to *weight the hinge by softplus(c_i - c_j)* over uniform weighting because large confidence gaps are where inversions are most damaging for proactive interaction, accepting the cost that near-tied confidence pairs contribute little gradient. We chose to *add the term to BACA's existing advantage-based loss* rather than replace BACA with a pure ranking objective, because BACA already supplies the trajectory-level credit that makes q_t meaningful across turns and removing it would reintroduce the very decoupling we target, accepting the cost that lambda must be tuned against the BACA scale. A domain expert would not describe this as a variant of Process Reward Models (Lightman 2023) or VinePPO-style per-step Monte-Carlo value estimation, because those estimate *step value* from rollouts; MTAC instead imposes an *ordering constraint between two existing heads' outputs* and leaves value estimation to BACA unchanged. The Phase 0 calibration and counterfactual completion are not new mechanisms but concrete specifications of how `c_t` and `q_t` are obtained, making the load-bearing assumption testable and guarding against the two known failure modes.

**Why it measures what we claim.** The computational quantity `L_rank` measures *monotone consistency between trigger confidence and answer quality* because it is a weighted count of rank inversions between `c_t_cal` and `q_t` within a trajectory; this holds under the assumption that `q_t` (judge-scored answer correctness, with counterfactual estimates on silent turns) is a faithful proxy for downstream answer quality, and this assumption fails when the judge is systematically lenient toward confident-sounding but wrong answers, in which case L_rank reflects judge bias rather than true monotonicity. The quantity `c_t_cal = Platt(trigger_head_logit)` measures *trigger confidence* because it is the calibrated version of the same head ALIVE uses to decide proactive response; this assumption fails when the trigger head's logit is dominated by silence-decoding artifacts rather than interaction readiness, in which case `c_t_cal` reflects decoding state rather than proactive intent. The calibrated threshold `theta*` measures *deployable trigger decision* because it is fit to maximize Spearman agreement on held-out turns; this fails when the held-out distribution drifts from the streaming deployment distribution, in which case theta* reflects the training split rather than the target stream.

## Contribution

["MTAC, a within-trajectory pairwise rank-inversion regularizer that couples ALIVE's trigger-confidence head to downstream answer-correctness scores during BACA RL, turning the previously implicit separability assumption into an explicit ordering constraint.", 'A trigger-threshold calibration procedure that fits the proactive firing threshold to maximize monotone agreement between confidence and answer quality on held-out multi-turn interactions, providing a deployable decision rule rather than a fixed threshold.', "An empirical design principle: monotone consistency between trigger confidence and answer quality is a trainable property of interactive streaming policies, not an emergent one, and can be imposed without modifying BACA's advantage estimator."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | FSI-Bench + ALIVE-RL-3K held-out turns | Tests feedback-seeking and multi-turn trigger ordering |
| Primary metric | Spearman agreement between calibrated trigger confidence `c_t_cal` and answer quality `q_t` | Directly measures the monotonicity claim |
| Secondary metric | Cohen's kappa between `q_t` and human-annotated quality on 200-turn subset | Validates that `q_t` tracks true quality (judge-bias audit) |
| Baseline | ALIVE (BACA, unmodified) | Isolates whether ranking term adds anything |
| Baseline | StreamBridge | Tests proactive-trigger baseline without monotone coupling |
| Baseline | LiveStar | Tests response-silence decoding baseline for timing |
| Baseline | Temperature-scaling calibration on trigger logits (no ranking term) | Tests whether global calibration alone recovers the ordering gain |
| Baseline | VinePPO-style per-step Monte-Carlo value estimator | Tests whether step-value estimation alone recovers the ordering gain |
| Ablation-of-ours | MTAC with uniform pair weights | Tests whether softplus confidence-gap weighting matters |
| Ablation-of-ours | MTAC without Phase 0 calibration/counterfactual completion | Tests whether the load-bearing assumption repair is necessary |

### Why this setup validates the claim

The central claim is that within-trajectory monotone consistency between trigger confidence and answer quality is both measurable and improvable without sacrificing BACA's trajectory-level credit. FSI-Bench is the right dataset because it is the only one in this evidence set that scores feedback seeking and subsequent task completion per turn, which is exactly where an ordering violation between "should I speak" and "was what I said good" becomes observable. The primary metric, Spearman agreement between `c_t_cal` and `q_t`, is the direct empirical counterpart of L_rank; if MTAC's mechanism works, this statistic must rise relative to ALIVE, and if it does not rise, the mechanism is inert regardless of aggregate benchmark scores. The secondary metric, Cohen's kappa on a 200-turn human-annotated subset, directly tests the load-bearing assumption that `q_t` is a faithful proxy for true answer quality; a kappa below 0.6 would indicate judge bias and invalidate the ranking target. Unmodified ALIVE tests the sub-claim that the ranking term is not redundant with BACA's advantage signal. StreamBridge tests whether a memory-buffer plus decoupled activation approach already captures the ordering implicitly; if MTAC matches it without the buffer, the ordering constraint is doing the work. LiveStar tests the timing sub-claim, since its response-silence decoding is the strongest existing alternative for deciding when to speak. The temperature-scaling baseline tests whether a global calibration loss alone recovers the ordering gain, and the VinePPO-style baseline tests whether per-step value estimation alone does so; if neither matches MTAC, the novelty claim rests on a demonstrated gap. The uniform-weight ablation is the load-bearing control: it shares MTAC's pairwise hinge but drops the softplus(c_i - c_j) weighting, so any gap between it and full MTAC is attributable specifically to the confidence-gap weighting rationale, not to adding a ranking loss at all. The no-Phase-0 ablation tests whether the calibration and counterfactual completion are necessary for the ranking term to have a well-posed target.

### Expected outcome and causal chain

**vs. ALIVE (BACA, unmodified)** — On a case where a trajectory contains an early low-stakes turn answered correctly and a later high-stakes turn answered wrongly, BACA's trajectory-level advantage assigns similar credit to both turns because it aggregates over the whole interaction, so the trigger head can become confident on the wrong turn without penalty. MTAC instead penalizes the rank inversion between the high `c_t_cal` on the wrong turn and the low `c_t_cal` on the correct turn via the weighted hinge, pushing the trigger head to align with per-turn quality. We expect a visible Spearman gain concentrated on trajectories with high within-trajectory variance in answer quality, with near-parity on trajectories where quality is uniformly high.

**vs. StreamBridge** — On a case where the memory buffer compresses a long multi-turn history and a decoupled activation model fires on a salient but ultimately unanswerable moment, StreamBridge produces a proactive response whose quality is low while its activation score stays high, because activation and answer quality are trained by separate objectives with no ordering constraint between them. MTAC adds exactly that ordering constraint on top of BACA's shared credit, so the trigger confidence is pulled toward turns where the answer head actually succeeds. We expect MTAC to show a moderate Spearman advantage on long-history FSI-Bench subsets and rough parity on short single-turn subsets.

**vs. LiveStar** — On a case where response-silence decoding fires at a moment chosen by single-forward-pass verification but the subsequent answer is only partially correct, LiveStar's timing decision and answer quality are decoupled by construction, so the silence decoder can be confident on a turn the answer head handles poorly. MTAC's pairwise term directly couples those two heads' outputs within the trajectory, so confident-but-poor turns are penalized. We expect the gap to appear as fewer high-confidence low-quality turns in the MTAC trigger distribution, i.e. a thinner upper-left tail, rather than as a uniform shift in mean confidence.

**vs. Temperature-scaling calibration on trigger logits** — This baseline applies Platt scaling to the trigger logits but adds no ranking term. It can improve marginal calibration but cannot enforce within-trajectory ordering, because a global monotone transform preserves the rank order of `c_t` within each trajectory. We expect it to show no Spearman gain over ALIVE, confirming that the ordering constraint (not calibration alone) is doing the work.

**vs. VinePPO-style per-step Monte-Carlo value estimator** — This baseline estimates per-step value from rollouts but does not impose an ordering constraint between trigger confidence and answer quality. We expect it to improve per-step credit assignment but to leave rank inversions between `c_t` and `q_t` unpenalized, resulting in a smaller Spearman gain than MTAC. If it matches MTAC, the novelty claim is weakened.

**vs. MTAC without Phase 0 calibration/counterfactual completion** — This ablation uses raw sigmoid logits for `c_t` and judge scores only on emitted turns, leaving silent turns out of the ranking. We expect it to show a smaller or noisier Spearman gain, confirming that the load-bearing assumption repair (calibration and counterfactual completion) is necessary for the ranking term to have a well-posed target.

### What would falsify this idea

If MTAC's Spearman gain over ALIVE is uniform across trajectories regardless of within-trajectory quality variance, rather than concentrated on high-variance trajectories where rank inversions actually occur, then the ranking term is acting as a generic regularizer and the monotone-consistency claim is wrong. Equally fatal: if the uniform-weight ablation matches full MTAC, the confidence-gap weighting rationale is unsupported and the mechanism reduces to a plain pairwise loss; and if the human-verified kappa is below 0.6, the load-bearing assumption that `q_t` tracks true quality fails and the ranking term should be dropped.

## References

1. ALIVE: Making Streaming Video Models Interactive
