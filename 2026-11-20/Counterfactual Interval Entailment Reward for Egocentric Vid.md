# Counterfactual Interval Entailment Reward for Egocentric Video Captioning

## Motivation

Existing egocentric captioning methods, such as EgoCaptioner (arxiv:2611.26380), refine captions with GRPO using step-level temporal alignment rewards that only score overlap between predicted intervals and reference timestamps. This structural proxy allows brief actions and order confusions to pass because a caption can overlap the correct reference interval without being causally entailed by the visual evidence within that interval; the reward never tests whether the narrated step is predictable from its assigned interval or unpredictable from the rest of the video. The field therefore lacks a per-interval counterfactual test that instantiates grounded understanding rather than timestamp proximity.

## Key Insight

A caption step is temporally grounded only if it is both necessary (unpredictable when its assigned interval is masked) and sufficient (entailed by that interval alone), and this counterfactual entailment signal can be computed as a dense per-step reward that GRPO optimizes directly without changing EgoCaptioner's caption format.

## Method

**What it is** — We introduce CIER (Counterfactual Interval Entailment Reward), a per-step reward module that plugs into EgoCaptioner's GRPO refinement. Given a video V, a policy-generated caption C = (overview O, timed steps {(s_i, [t_i^start, t_i^end], d_i)}), and a learned entailment head E, CIER produces for each step i a scalar reward r_i^CIER that combines a prior-corrected necessity score and a sufficiency score; the GRPO objective is augmented with the mean of r_i^CIER alongside EgoCaptioner's existing factual, coverage, alignment, and redundancy rewards.

**How it works** —
```pseudocode
Input: video V, caption C = (O, {(s_i, I_i, d_i)}_{i=1..N}), entailment head E_phi,
       masker M, policy pi_theta, reference captions R, alignment reward r_align,
       text-only prior baseline p_prior(d_i) = pi_theta(d_i | O)  # no visual frames
Hyperparams: lambda_nec = 1.0, lambda_suf = 1.0, tau = 0.5 (calibrated per step-type on 512 held-out examples),
             K = 4 (mask granularity), G = 8 (GRPO group size)

1. For each step i, build three visual contexts:
     V_i^in  = frames of V restricted to interval I_i
     V_i^out = frames of V outside I_i, plus K randomly sampled frames inside I_i
                (so V_i^out retains scene context but removes the step's distinctive motion)
     V_i^prior = no frames (text-only context: O only)

2. Prior-corrected necessity score (counterfactual unpredictability):
     p_in  = log pi_theta(d_i | O, V_i^in)         # log-prob of step text given its own interval
     p_out = log pi_theta(d_i | O, V_i^out)        # log-prob given the masked remainder
     p_prior = log pi_theta(d_i | O)               # log-prob under text-only prior
     nec_i = sigmoid( ((p_out - p_prior) - (p_in - p_prior)) / tau )
           = sigmoid( (p_out - p_in) / tau )       # prior cancels; explicit for clarity
     # tau is calibrated per step-type (generic verb vs. specific action) on 512 held-out
     # examples so the sigmoid is not saturated by prior-driven steps.

3. Sufficiency score (entailment):
     suf_i = E_phi( V_i^in , d_i ) in [0,1]    # entailment head trained on (interval, step) pairs

4. CIER reward per step:  r_i^CIER = lambda_nec * nec_i + lambda_suf * suf_i

5. GRPO update: replace EgoCaptioner's step-level reward with
     r_i = r_align_i + r_i^CIER    (other EgoCaptioner rewards unchanged)
   and optimize the group-relative advantage over G=8 sampled captions as in EgoCaptioner.

6. Entailment head training (offline, before GRPO):
     positive pairs: (I_i, d_i) from EgoCaptioner's 50k coarse-to-fine annotations
     negative pairs: (I_j, d_i) for j != i within the same video, plus (I_i, d_j)
     loss: binary cross-entropy on E_phi; frozen during GRPO.
```

**Why this design** — We chose a per-interval counterfactual masker over a learned disentanglement head (Anti-pattern 1) because the necessity signal is defined by an intervention on the video input, not by a statistical independence assertion over coupled variables; the trade-off is that the masker must retain enough context for the policy to produce a non-degenerate p_out, which we handle by keeping K=4 in-interval frames. We chose a frozen entailment head trained on the existing 50k annotations over an end-to-end joint head because GRPO's reward must be stationary across policy updates, accepting the cost that E_phi cannot adapt to novel step phrasings the policy invents. We chose to augment rather than replace EgoCaptioner's alignment reward because alignment captures coarse temporal proximity that CIER alone would not enforce on long videos, accepting the cost of one extra reward weight to tune. A domain expert would not describe this as a variant of Process Reward Models (Lightman 2023) or VinePPO: those assign credit via Monte-Carlo prefix rollouts of the same policy, whereas CIER's necessity term is a counterfactual intervention on the visual input that no prefix rollout can produce.

**Why it measures what we claim** — The necessity score nec_i measures causal necessity of interval I_i for step d_i because we assume that if d_i is grounded in I_i, then removing I_i's distinctive frames should reduce the policy's likelihood of d_i relative to the text-only prior; this assumption fails when the policy's p_out is dominated by language priors (e.g., d_i is a generic verb like 'walk'), in which case nec_i reflects prior-driven fluency rather than visual necessity. We explicitly state this load-bearing assumption (A) and its failure mode (F) here: A = the policy's p_out − p_in gap is dominated by visual evidence rather than language priors (i.e., masking removes distinctive motion while retaining enough context for non-degenerate p_out); F = generic/habitual verbs where p_out stays high. To mitigate F, we subtract the text-only prior p_prior(d_i) from both p_in and p_out (which cancels in the difference but makes the prior correction explicit) and calibrate tau per step-type on 512 held-out examples. The sufficiency score suf_i measures interval-step entailment because we assume E_phi's positive/negative training pairs from EgoCaptioner's annotations separate true (interval, step) matches from shuffled ones; this assumption fails when negative pairs are too easy (different scenes), in which case suf_i reflects scene classification rather than fine-grained action entailment. The combined r_i^CIER measures grounded step-level captioning because we assume that a step passing both the counterfactual and entailment tests cannot be a brief-action miss or order confusion; this assumption fails when the masker removes too much context, in which case nec_i reflects general video degradation rather than interval-specific evidence.

## Contribution

['A counterfactual interval entailment reward (CIER) that operationalizes per-step temporal grounding as a necessity-plus-sufficiency score computed by masking the assigned interval and querying a frozen entailment head, usable as a dense GRPO reward without changing caption format.', 'A design principle for egocentric captioning rewards: step-level temporal alignment must be paired with a counterfactual intervention on the visual input, because timestamp-overlap rewards are satisfied by order-confused and brief-action captions.', "An offline entailment-head training procedure that reuses EgoCaptioner's 50k coarse-to-fine annotations with within-video shuffled negatives."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Ego4D-based EgoCaptioner 50k annotations, five egocentric benchmarks | Matches target task and provides timed-step supervision |
| Primary metric | Step-level factual accuracy + temporal alignment on brief-action subset | Directly probes the miss/order-confusion failure CIER targets |
| Baseline | EgoCaptioner (GRPO, no CIER) | Isolates the contribution of the CIER reward term |
| Baseline | MMEgo-style egocentric MLLM | Tests whether QA-tuned understanding transfers to captioning |
| Baseline | Process-reward / prefix-rollout credit assignment | Tests the counterfactual-vs-rollout distinction claim |
| Baseline | Likelihood-ratio reward (p_in − p_out without entailment head and without K-frame context retention) | Tests whether context-preserving masker + frozen entailment head drives gains |
| Ablation-of-ours | CIER with necessity term only (drop sufficiency) | Checks whether entailment adds beyond counterfactual masking |
| Ablation-of-ours | CIER with sufficiency term only (drop necessity) | Checks whether counterfactual masking adds beyond entailment |
| Ablation-of-ours | CIER with raw log-prob difference (no prior correction, no tau calibration) | Tests the load-bearing assumption repair |

### Why this setup validates the claim

The central claim is that a per-step reward combining counterfactual necessity and interval entailment reduces brief-action misses and order confusion beyond what EgoCaptioner's factual/coverage/alignment/redundancy rewards already achieve. EgoCaptioner is the right primary baseline because it is the exact host into which CIER plugs: any gain must come from the added reward term, not from a different backbone, data, or optimizer. The MMEgo-style baseline tests a different sub-claim — that egocentric video-QA competence does not automatically yield fine-grained timed narration — so if it trails on brief-action accuracy despite strong QA numbers, that supports the premise that captioning needs step-grounded supervision. The prefix-rollout baseline tests the design argument that counterfactual visual intervention captures something Monte-Carlo prefix credit cannot; if CIER only matches it, the novelty claim is empty. The likelihood-ratio baseline (p_in − p_out without entailment head and without K-frame context retention) directly tests whether the specific intervention design — context-preserving masker plus frozen entailment head — is what drives gains, rather than generic masking. The necessity-only and sufficiency-only ablations isolate whether each component contributes beyond the other, guarding against redundancy. The raw-log-prob ablation tests whether the prior correction and tau calibration are necessary for the necessity signal to track grounding. The primary metric is chosen to be diagnostic rather than aggregate: step-level factual accuracy restricted to brief-action and order-sensitive subsets is precisely where the predicted failure mode lives, so a real effect should be visible there rather than smeared across an overall score.

### Expected outcome and causal chain

**vs. EgoCaptioner (no CIER)** — On a case where a brief hand-object action occurs inside a two-second interval amid a long walk, the baseline produces a fluent but generic step ("the wearer moves through the room") because its coverage and alignment rewards are satisfied by coarse temporal proximity and do not penalize omission of the interval's distinctive motion. Our method instead assigns low nec_i to that generic step because masking the interval barely changes its likelihood, and low suf_i because the interval does not entail the generic verb, so GRPO is pushed toward steps whose likelihood collapses under masking. We expect a concentrated accuracy gain on the brief-action subset with near-parity on long static segments.

**vs. MMEgo-style egocentric MLLM** — On a case requiring ordered multi-step narration of a recipe-like sequence, the baseline produces steps in plausible but shuffled order or merges two steps, because its memory-pointer design optimizes question answering over whole-video recall rather than per-interval timing. Our method instead rewards each step against its own interval via suf_i, so a step placed on the wrong interval receives a low entailment score even when the text is globally plausible. We expect a larger gap on order-sensitive and multi-step subsets than on single-action clips, consistent with the timing-specific mechanism.

**vs. Process-reward / prefix-rollout credit assignment** — On a case where the policy's language prior strongly predicts a step regardless of visual content, the baseline assigns high credit because prefix rollouts from the same policy also produce that step fluently, so the reward cannot distinguish grounded from prior-driven text. Our method instead computes p_out under a masked visual context, so a prior-dominated step shows a small p_out − p_in gap and receives low necessity. We expect CIER to outperform rollout-based credit specifically on prior-confusable steps (generic verbs, habitual actions) with parity elsewhere.

**vs. Likelihood-ratio reward (p_in − p_out without entailment head and without K-frame context retention)** — On a case where the step is a specific action (e.g., "open the drawer") inside a short interval, the baseline likelihood-ratio reward uses a masker that removes all in-interval frames, causing p_out to collapse for any step and producing a noisy, high-variance signal that does not discriminate grounded from ungrounded steps. Our method retains K=4 in-interval frames and adds the frozen entailment head, so nec_i and suf_i are both non-degenerate and jointly require the step to be both unpredictable under masking and entailed by the interval. We expect CIER to outperform the likelihood-ratio baseline on brief-action and order-sensitive subsets, with the gap concentrated where the baseline's masker destroys context.

### What would falsify this idea

If CIER's gain over EgoCaptioner is uniform across brief-action, order-sensitive, and prior-confusable subsets rather than concentrated on them, the counterfactual-and-entailment mechanism is not doing the claimed work. Likewise, if the necessity-only ablation matches full CIER everywhere, the sufficiency term adds nothing and the two-component claim is wrong; if the raw-log-prob ablation matches full CIER, the prior correction and tau calibration are unnecessary.

## References

1. EgoCaptioner: Towards Comprehensive and Fine-Grained Egocentric Video Captioning
