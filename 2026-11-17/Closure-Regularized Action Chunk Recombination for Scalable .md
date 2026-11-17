# Closure-Regularized Action Chunk Recombination for Scalable Vision-Language-Action Policies

## Motivation

Vision-language-action models across the field — CoT-VLA's autoregressive visual-goal prediction, LLaRA's conversation-style action tokens, OTTER's text-aware visual feature extraction, and the knowledge-insulating diffusion/flow-matching action experts studied for ScaleVLA-style backbones — all assume that their action representation composes: that chunks learned on sub-tasks can be concatenated or recombined and remain executable. ScaleVLA in particular replaces LLaRA's discrete conversation-style action tokens with continuous action chunks produced by a trainable action DiT, discarding any explicit compositional structure and leaving the closure of the action-chunk set under sub-task recombination entirely unverified. The root cause is structural: no existing VLA training objective ever presents the model with a recombined action sequence and asks whether it is executable, so compositional generalization is hoped for rather than enforced, and failures surface only at deployment on novel task orderings.

## Key Insight

Executability of a recombined action chunk is a property of the action decoder's own reconstruction manifold, so a closure predicate trained as a margin-separated classifier on the decoder's reconstruction residual gives a checkable closure rule for the action-chunk set rather than a hoped-for generalization.

## Method

**What it is.** CReW (Closure-Regularized Action Chunk Recombination) is a training-time regularizer for ScaleVLA's action DiT. Its inputs are the demonstration dataset (segmented into sub-task chunks), the current VLM backbone representations, the action DiT decoder, and an outcome-grounded executability oracle (simulator rollout or learned forward/feasibility model). Its outputs are (i) a closure predicate head `c_phi(a_1:K, z) -> [0,1]` and (ii) an updated action DiT whose generated chunks are closed under sub-task recombination.

**How it works.**

```pseudocode
# Phase 0: segment demonstrations into sub-task chunks
for each demonstration d:
    segment d into sub-task chunks {a^(1), ..., a^(m)} via change-point
    detection on end-effector velocity (threshold tau_v = 0.02 m/frame)
    store (VLM repr z^(i), action chunk a^(i), seam state s^(i)) pairs in buffer B

# Phase 1: sample recombination pairs and build composed sequences
for each training step:
    sample two chunks (z^(1), a^(1), s^(1)), (z^(2), a^(2), s^(2)) ~ B from DIFFERENT sub-tasks
    compose a_comp = [a^(1); a^(2)]            # concatenate along time
    compose z_comp = concat(z^(1), z^(2))      # VLM representation context
    # label executability by outcome-grounded oracle (simulator or learned forward model)
    y = 1 if Oracle(a_comp, s^(1)) reaches s^(2) within tolerance eps_state = 0.02 m else 0
    # also compute decoder reconstruction residual for diagnostic (not used as label)
    a_hat = ActionDiT.decode(z_comp)
    r = ||a_hat - a_comp||^2

# Phase 2: closure regularizer + predicate loss
L_closure = BCE(c_phi(a_comp, z_comp), y)
L_margin  = max(0, score_pos + delta - score_neg)   # score = oracle executability score (e.g., 1 - normalized state error)
L_total   = L_action + lambda_c * L_closure + lambda_m * L_margin
# lambda_c = 0.5, lambda_m = 0.1, delta = 0.1, tuned on held-out recombinations

# Phase 3: rejection at inference
at deployment, for any proposed recombination:
    if c_phi(a_comp, z_comp) < 0.5: reject and fall back to single-chunk plan
    else: execute a_comp
```

**Why this design.** We chose to label executability from an outcome-grounded oracle (simulator rollout or learned forward/feasibility model) rather than from the action DiT's own reconstruction residual, because executability is a property of the closed-loop transition dynamics (does the arm reach the next sub-task's initial state?), which no open-loop reconstruction objective observes; the residual is blind to the state mismatch at the seam by construction. We accept the cost that the oracle introduces its own bias and requires either simulator access or a learned forward model. We chose a margin-separated BCE predicate over a plain threshold on the oracle score because a bare threshold gives no gradient signal to the DiT to *expand* its executable set, accepting the cost of an extra hyperparameter `delta`. We chose velocity-based change-point segmentation over fixed-length windows because sub-task boundaries are where recombination is semantically meaningful, accepting the cost that segmentation errors propagate into the pair sampler. A domain expert would not describe CReW as a variant of knowledge-insulating VLA training (the closest prior method, which isolates the VLM backbone from the action expert): knowledge insulation changes *which parameters receive gradient*, whereas CReW changes *which input sequences the decoder is trained to treat as in-distribution*, and the closure predicate is a new output surface that knowledge insulation does not produce.

**Why this measures what we claim.** The oracle executability label `y` measures *executability of a recombined chunk* because we assume the oracle (simulator or learned forward model) accurately predicts whether the composed chunk reaches the second sub-task's entry state within tolerance; this assumption fails when the oracle is inaccurate or the tolerance is mis-specified, in which case `y` reflects oracle error rather than compositional closure. The predicate `c_phi` measures *closure membership* because we assume the margin-separated BCE on oracle-derived labels yields a decision boundary that coincides with the executability threshold; this assumption fails when recombined sequences cluster near the boundary, in which case `c_phi` reflects label noise from the threshold rather than a stable closure rule. The margin loss `L_margin` measures *pressure to expand the executable set* because we assume increasing the oracle score gap between positive and negative recombinations corresponds to the DiT learning to decode more recombinations; this assumption fails when the DiT trades off single-chunk fidelity for recombination fidelity, in which case the margin reflects a fidelity shift rather than closure expansion. The reconstruction residual `r` is retained only as a diagnostic and is not used for labeling; we explicitly state the assumption A that the DiT's residual is monotone in distance from its training action manifold and that low residual implies successful execution, and the failure mode F where the decoder is underfit on both composed and neighboring sequences so `r` reflects capacity rather than closure. We validate A by correlating `r` with simulator success on held-out recombinations.

## Contribution

(1) CReW, a closure regularizer for action-DiT-based VLAs that turns sub-task recombination executability into a trainable, checkable predicate rather than an assumed generalization property. (2) A design principle: executability labels derived from the action decoder's own reconstruction residual, with a margin term that pressures the decoder to expand its executable set, distinguishing closure enforcement from parameter isolation. (3) A segmentation-and-recombination pipeline that converts existing behavior-cloning demonstrations into closure supervision without additional action annotations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LIBERO-PRO recombinations (primary); real-world charger insertion (secondary) | Primary: controlled simulation for margin ablation; secondary: real-world confirmation |
| Primary metric | Success rate on recombined sub-task sequences | Directly tests compositional closure claim |
| Baseline | ScaleVLA action DiT, no regularizer | Isolates CReW's contribution |
| Baseline | Knowledge-insulating VLA training | Closest prior method, different mechanism |
| Baseline | CoT-VLA visual chain-of-thought | Tests whether explicit reasoning substitutes closure |
| Ablation-of-ours | CReW without margin loss (plain BCE threshold) | Tests whether margin pressure expands executable set |
| Ablation-of-ours | CReW with margin loss backpropagating only into c_phi (not DiT) | Tests whether input-distribution shaping is distinct from gradient routing |
| Ablation-of-ours | CReW with residual-derived labels (original circular version) | Tests whether outcome-grounded labels are necessary |
| Diagnostic | Correlation between reconstruction residual r and simulator success on held-out recombinations | Validates assumption A (monotonicity) |
| Oracle agreement | Agreement between residual-derived label y and oracle label on 512 held-out recombinations | Establishes predicate validity against external ground truth |

### Why this setup validates the claim

The central claim is that CReW trains ScaleVLA's action DiT to treat recombined sub-task chunks as in-distribution, improving success on compositions the decoder would otherwise fail. The no-regularizer ScaleVLA baseline tests whether closure training is necessary at all: if recombination success is already high, CReW has no job. Knowledge insulation tests the specific contrast the method claims — gradient routing versus input-distribution shaping — so a gain concentrated on recombined sequences (not single-chunk fidelity) distinguishes the two mechanisms. CoT-VLA tests whether explicit visual reasoning already supplies compositional ability, which would make CReW redundant. The margin ablation is the load-bearing one: it tests whether the predicate alone suffices or whether the margin loss is what actually expands the executable set. The margin-backprop ablation (margin loss only into c_phi, not DiT) directly tests whether input-distribution shaping is a genuinely distinct mechanism from gradient routing, as the method claims. The residual-label ablation tests whether the original circular labeling (using DiT residual) is sufficient or whether outcome-grounded labels are necessary, addressing the load-bearing assumption. The diagnostic correlation between r and simulator success validates assumption A (monotonicity of residual), and the oracle agreement metric establishes that the predicate's labels are grounded in external ground truth rather than decoder capacity. Success rate on recombined sequences is the right metric because it is exactly the behavior the closure predicate is supposed to characterize; single-chunk fidelity would be blind to the predicted effect.

### Expected outcome and causal chain

**vs. ScaleVLA action DiT (no regularizer)** — On a case where a charger-insertion demonstration is segmented into approach and contact sub-tasks and the approach chunk from one demo is concatenated with the contact chunk from another, the baseline produces a discontinuous trajectory with an offset at the seam because its decoder never saw such concatenations and treats the boundary as out-of-distribution. Our method instead trains the decoder on sampled recombinations with the closure predicate and margin loss, so the seam region enters the training manifold, and we expect a noticeable success gap on recombined sequences but rough parity on single-demonstration rollouts.

**vs. Knowledge-insulating VLA training** — On the same seam instance, knowledge insulation produces a smooth-looking but semantically mismatched chunk because it only protects VLM semantics from gradient interference; it does nothing to shape which input sequences the action expert treats as in-distribution, so the composed context is still novel to the decoder. Our method instead directly labels and regularizes composed sequences via outcome-grounded executability, so the decoder's executable set is explicitly expanded at sub-task boundaries. We expect the gain to concentrate on recombination subsets while single-chunk accuracy stays comparable, since gradient routing and input-distribution shaping target different failure modes.

**vs. CoT-VLA visual chain-of-thought** — On a case requiring a mid-horizon plan switch, CoT-VLA produces a plausible predicted goal image but an action sequence that does not realize the concatenated sub-tasks, because its reasoning is over future frames rather than over the decoder's action manifold, leaving compositional closure unaddressed. Our method instead constrains the decoder's output surface directly through the closure predicate and margin, so recombined chunks are validated against measured executability. We expect a gap on recombined multi-sub-task cases with parity on single-goal tasks, since CoT-VLA's advantage is planning, not closure.

**vs. CReW without margin loss (plain BCE threshold)** — On recombined sequences near the executability boundary, the plain BCE variant yields a predicate that classifies but does not push the DiT to expand its executable set, because the margin loss is what provides gradient pressure to increase the oracle score gap between positive and negative recombinations. Our method with margin loss should show higher success on recombinations that were initially near the boundary, while single-chunk performance remains similar. We expect the margin ablation to underperform full CReW on recombined sequences, confirming that margin pressure is necessary for closure expansion.

**vs. CReW with margin loss backpropagating only into c_phi (not DiT)** — This ablation isolates whether input-distribution shaping (gradient into DiT) is distinct from gradient routing (gradient only into predicate). If the gain is purely from a better predicate, then this ablation should match full CReW; if full CReW outperforms, it demonstrates that the DiT itself must be shaped to treat recombinations as in-distribution. We expect full CReW to outperform this ablation on recombined sequences, supporting the claim that input-distribution shaping is a distinct mechanism.

**vs. CReW with residual-derived labels (original circular version)** — This ablation tests whether the original self-referential labeling (using DiT reconstruction residual) is sufficient. If the residual-derived labels yield similar performance to outcome-grounded labels, then the circularity concern is empirically benign; if outcome-grounded labels yield higher success on recombinations, it confirms that the residual measures decoder capacity rather than executability. We expect outcome-grounded labels to outperform, especially on contact-sensitive seams where smooth-but-wrong concatenations yield low residual.

### What would falsify this idea

The idea would be falsified if the CReW gain were uniform across recombined and single-chunk sequences rather than concentrated on recombinations, or if the margin ablation matched full CReW — either pattern would mean the closure regularizer is not doing the compositional work the claim attributes to it. Additionally, if the residual-derived label ablation matched outcome-grounded labels, the load-bearing assumption would be empirically validated as benign, but if oracle agreement is low, the predicate's validity would be undermined.

## References

1. ScaleVLA: Scaling Vision-Language-Action Models to 552 Billion Parameters
