# Compositional Closure Verification for Interleaved Multimodal Chain-of-Thought Training

## Motivation

Interleaved-reasoning training recipes such as ThinkMorph-SSL supervise each step-wise visual thought in isolation: stage-one pretext traces and stage-two demonstrations reward locally plausible image generation/selection without ever checking that the accumulated sequence of visual thoughts plus textual rationales composes into a globally coherent chain. The same structural gap recurs in attention-driven selection methods like ICoT, where each region insertion is chosen independently at each step, so a chain of individually high-quality visual selections can drift, contradict, or fail to entail the next sub-goal while every local reward stays high. The root cause is that supervision is defined per-step rather than over the closure of the chain, leaving the composition operator (how step t's visual thought constrains what step t+1 may validly assume) entirely unregularized.

## Key Insight

A reasoning chain is valid only if each generated visual thought is entailed by the accumulated prefix rather than merely plausible in isolation, so a chain-level closure reward that is zero exactly when every step's visual content is derivable from the prefix-plus-subgoal turns per-step supervision into a compositional invariant the model cannot satisfy by locally good guesses.

## Method

**What it is.** COCO-VT (Compositional Closure Verification for Visual Thoughts) augments ThinkMorph-SSL's two-stage interleaved training with a chain-level closure signal. Inputs: an interleaved trace of (visual thought v_t, textual rationale r_t) pairs produced by the policy MLLM, plus the question-derived sub-goals g_t. Outputs: a scalar chain-closure reward R_closure used in the stage-two objective, and a per-step closure residual that gates which steps receive gradient.

**How it works.**

```pseudocode
# Stage 1: unchanged pretext reasoning (ThinkMorph-SSL)
# Stage 2: question-driven instruction tuning WITH closure reward

for each training example (question q, image I, sub-goals g_1..g_T):
    prefix = [q]                       # textual+visual context accumulator
    for t in 1..T:
        (v_t, r_t) = MLLM.generate_step(prefix, g_t)   # visual thought + rationale
        prefix.append((v_t, r_t))

    # ---- closure verification pass (frozen lightweight verifier V) ----
    # V is a 2-layer cross-attention probe over MLLM hidden states (frozen backbone)
    # V outputs a grounding alignment between v_t and prefix: a region/entity match
    # score s_align in [0,1] computed as the max cosine similarity between v_t's
    # projected region embedding and the prefix's entity/region embeddings, plus a
    # scalar confidence c in [0,1] from a 2-layer MLP head on the cross-attention output.
    # p_entail is defined from s_align via a held-out-calibrated threshold theta=0.65:
    #   p_entail = 1 if s_align >= theta else 0
    # p_goal is defined analogously for the next sub-goal g_t.
    # A calibration/abstention gate: if c < c_min=0.5, the step contributes zero reward.
    R_closure = 0
    for t in 1..T:
        s_align, c = V(prefix_{<t}, v_t, r_t, g_t)   # grounding alignment + confidence
        s_goal, c_goal = V(prefix_{<t}, g_t)          # alignment for next sub-goal
        if c < c_min or c_goal < c_min:
            residual_t = 0   # abstain: no reward signal for this step
        else:
            p_entail = 1.0 if s_align >= theta else 0.0
            p_goal   = 1.0 if s_goal >= theta else 0.0
            residual_t = max(0, tau - (p_entail + p_goal)/2)   # tau = 0.7 margin
        R_closure += -residual_t

    # ---- chain-level consistency reward ----
    R_consistency = -KL( V(prefix_{<T}) || V(prefix_{<T} with v_T replaced by v_T^') )
    # v_T^' = counterfactual visual thought sampled from a distractor pool of
    # same-question, wrong-region crops; KL large => chain depends on v_T

    total_reward = R_task + lambda_c * R_closure + lambda_k * R_consistency
    # lambda_c = 0.5, lambda_k = 0.2 (tuned on held-out split)
    update MLLM with policy-gradient on total_reward
    # verifier V trained jointly on (prefix, v_t, g_t) -> grounding alignment label
    # labels from a held-out 5k human-annotated closure set, where annotators mark
    # whether v_t's region is entailed by the prefix under g_t (binary) and provide
    # a bounding-box alignment when entailed.
```

**Why this design.** We chose a frozen-backbone cross-attention verifier over a full second MLLM judge because a judge re-introduces the same per-step plausibility bias we are trying to escape, accepting the cost that V may miss long-range entailment failures beyond its 2-layer receptive field. We chose a margin-based hinge residual (tau=0.7) over a soft log-prob reward because log-probs are known to be poorly calibrated for correctness (Kadavath et al.), accepting the cost that the hinge saturates and gives no gradient once closure is comfortably satisfied. We chose a counterfactual-swap KL term over a simple entropy penalty because entropy rewards diversity without testing whether the chain actually depends on its visual content, accepting the cost of an extra forward pass per example. A domain expert would not describe this as a variant of Process Reward Models (Lightman 2023) or VinePPO: those assign per-step values from rollouts of the same policy, whereas COCO-VT's closure residual is computed by an independent verifier over the accumulated prefix and a counterfactual intervention, so the signal is a composition invariant rather than a Monte-Carlo step value.

**Why it measures what we claim.** The quantity p_entail = V(prefix_{<t}, v_t, r_t, g_t) measures compositional closure (that step t's visual thought is derivable from the accumulated prefix) because we assume the verifier's grounding alignment score s_align (a region/entity match against the input image) is a faithful proxy for logical derivability under the sub-goal; this assumption fails when the verifier is fooled by surface lexical overlap between r_t and g_t, in which case p_entail reflects topical similarity rather than entailment. The hinge residual max(0, tau - p) measures per-step closure deficit because we assume a margin of 0.7 separates genuinely entailed from unentailed steps on the annotated closure set; this fails when the annotation distribution is skewed toward easy steps, in which case the residual under-penalizes hard multi-step drift. The counterfactual KL measures chain-level causal dependence on visual thoughts because we assume replacing v_T with a same-question wrong-region crop is a valid intervention that isolates the visual contribution; this fails when the distractor crop is trivially distinguishable from the true region, in which case the KL reflects low-level image mismatch rather than reasoning dependence.

## Contribution

(1) A chain-level compositional closure reward (COCO-VT) that replaces per-step plausibility supervision with an entailment-based residual over the accumulated visual-thought prefix, plus a counterfactual-swap consistency term that tests whether the chain causally depends on its visual content. (2) A lightweight frozen-backbone cross-attention verifier trained on a 5k human-annotated closure set that supplies step-level closure residuals for gating policy gradient. (3) The design principle that interleaved visual-thought training requires a composition operator regularizer distinct from per-step pretext supervision, instantiated as a hinge-margin closure reward with a counterfactual intervention.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Twelve interleaved-reasoning benchmarks from ThinkMorph-SSL | Matches baseline training distribution and reported evaluation suite |
| Primary metric | Accuracy on spatial/fine-grained visual-cue subsets | Closure failures concentrate where visual grounding is subtle |
| Baseline | Instruction tuning alone (same budget) | Isolates whether closure reward adds beyond demonstrations |
| Baseline | ThinkMorph-SSL two-stage recipe | Tests whether closure signal improves on pretext-only supervision |
| Baseline | Interleaved-MCoT (ADS) prompting | Tests whether training-time closure beats inference-time insertion |
| Baseline | Per-step process reward model (PRM) trained on same 5k closure set | Tests whether closure residual diverges from Monte-Carlo step values |
| Baseline | VinePPO with same rollout budget | Tests whether closure residual adds beyond per-step value estimation |
| Ablation-of-ours | COCO-VT without R_consistency (closure hinge only) | Isolates counterfactual KL from per-step closure residual |
| Ablation-of-ours | COCO-VT with verifier confidence gate disabled (c_min=0) | Tests whether abstention prevents reward hacking on low-confidence steps |
| Ablation-of-ours | COCO-VT with distractor pool of trivially distinguishable crops | Tests whether KL tracks reasoning dependence vs. low-level mismatch |

### Why this setup validates the claim

The central claim is that chain-level closure — not per-step plausibility — is what interleaved training lacks, and that an independent verifier plus a counterfactual swap supplies it. Instruction-tuning-alone tests sub-claim one: that demonstrations alone leave closure deficits unpenalized, so any gain over it must come from the reward, not the data. ThinkMorph-SSL tests sub-claim two: that pretext-derived visual skills are necessary but insufficient, so closure should add on top rather than substitute. Interleaved-MCoT tests sub-claim three: that closure is a training-time composition invariant, not an inference-time formatting trick, so a plug-and-play prompt method should not close the same gap. The PRM and VinePPO baselines test sub-claim four: that the closure residual is not merely a Monte-Carlo step value, so we report where the closure residual diverges from PRM/VinePPO step values on identical traces. The ablation removing R_consistency is the diagnostic: if removing the counterfactual KL collapses the gain, the effect is causal dependence on visual thoughts; if removing the hinge collapses it, the effect is per-step entailment. The confidence-gate ablation tests whether the abstention mechanism prevents reward hacking on low-confidence steps. The distractor-difficulty ablation tests whether the KL tracks reasoning dependence rather than low-level image mismatch. Accuracy on spatial and fine-grained subsets is the right metric because the method's stated failure mode — drift where visual cues are subtle — is exactly where closure residuals fire; uniform gains would indicate a generic regularization effect rather than closure.

### Expected outcome and causal chain

**vs. Instruction tuning alone** — On a case where a multi-step spatial question requires re-grounding on a small region after two textual steps, instruction tuning produces a fluent rationale that drifts to a plausible but wrong region because nothing penalizes a step that is locally fluent yet underivable from the accumulated prefix. Our method instead assigns a high closure residual to that step via the verifier and gates gradient there, so we expect a noticeable accuracy gap concentrated on spatial and fine-grained subsets with near-parity on OCR and general-scene subsets.

**vs. ThinkMorph-SSL two-stage** — On a case where pretext training has taught the model to generate coherent intermediate images but the chain's final answer does not actually depend on the last visual thought, ThinkMorph-SSL produces a correct-looking trace that would survive swapping the final crop for a wrong-region distractor because no signal tests dependence. Our method's counterfactual KL detects the flat response and penalizes it, so we expect the gain to appear specifically on questions where the last visual thought is load-bearing, visible as a larger delta on hard spatial items than on easy ones.

**vs. Interleaved-MCoT (ADS) prompting** — On a case where attention-driven region insertion picks a salient but goal-irrelevant crop, ADS produces an interleaved trace whose visual content is topically related yet does not entail the next sub-goal because selection is attention-based and untrained against closure. Our method instead trains the policy against an entailment verifier over the prefix, so we expect a moderate but consistent edge on fine-grained extraction items, with the largest separation on multi-step chains where drift compounds.

**vs. Per-step process reward model (PRM)** — On a case where a step is locally plausible but underivable from the accumulated prefix, a PRM trained on the same 5k closure set assigns a high step value because it scores each step in isolation, whereas our closure residual fires because it conditions on the prefix. We expect our method to outperform PRM on multi-step spatial items, with the largest gap on chains where early drift compounds, and we will report the fraction of steps where the closure residual and PRM value disagree on identical traces.

**vs. VinePPO** — On a case where Monte-Carlo rollouts from the same policy assign similar values to a correct and a drifted step because both lead to plausible continuations, VinePPO fails to penalize the drift, whereas our counterfactual KL detects the flat response to swapping v_T. We expect our method to outperform VinePPO on questions where the final visual thought is load-bearing, with the gain concentrated on hard spatial items.

### What would falsify this idea

If COCO-VT's gain over ThinkMorph-SSL is uniform across spatial, OCR, and general subsets rather than concentrated where closure residuals are predicted to fire, the closure mechanism is not doing the claimed work. Equally fatal: if the ablation removing R_consistency matches the full method, the counterfactual-dependence claim is unsupported and the effect reduces to per-step regularization; if the confidence-gate ablation (c_min=0) matches the full method, the abstention mechanism is not preventing reward hacking; if the distractor-difficulty ablation (trivially distinguishable crops) matches the full method, the KL reflects low-level image mismatch rather than reasoning dependence.

## References

1. ThinkMorph-SSL: Scaling Interleaved Multimodal Reasoning with Pretext Tasks
