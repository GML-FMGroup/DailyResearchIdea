# Counterfactual Intervention Certificates for Faithful Vision-Language Navigation Policies

## Motivation

Across the frozen-LLM-plus-projection pipeline instantiated by Uni-NaVid and the prompt-only chain-of-thought agents evaluated in VLNVerse, the reasoning-to-action pathway is assumed faithful: no interventional test establishes that the emitted action is causally driven by the claimed perception context rather than by spurious correlations absorbed into the projection/policy head. Existing benchmarks only score task success (VLNVerse's unified multi-task evaluation, SAME's seven-task success rates), so a policy that ignores the instruction and exploits a scene prior is indistinguishable from one that reasons correctly, and error propagation from a mis-grounded token to a wrong action is never identifiable. The structural gap is that faithfulness is a claim about a counterfactual (remove the token, does the action change as the trace predicts?) while every current training objective is purely observational.

## Key Insight

Because VLNVerse's full-kinematics simulator lets us physically realize the intervention the LLM's own reasoning trace predicts, we can turn faithfulness from an untestable assumption into a measurable do-calculus invariant: a perception token is certified only if its removal changes the action distribution in the direction the trace names, making uncertifiable actions identifiable as spurious rather than merely low-confidence.

## Method

## FARA: Faithfulness-Audited Reasoning-to-Action

**(A) What it is.** FARA wraps a frozen LLM + projection/policy head with a token-level *intervention certificate*: for each action, the head emits a set of perception tokens plus a predicted signed action-shift direction. Training enforces that this certificate is *necessary* (removing the tokens changes the action) and *sufficient* (the change matches the predicted direction) via a counterfactual consistency loss computed on paired, simulator-realized perception contexts. **Load-bearing assumption (stated explicitly):** the simulator-realized removal of a set of perception tokens is treated as a valid do-operation on the causal variable the reasoning trace names, so that the measured action shift reflects the named token's causal contribution rather than residual visual evidence, re-encoding, or off-manifold artifacts. Because removing an object re-renders the whole scene (shadows, occlusion boundaries, background fill, lighting), the raw removal effect conflates the named token's effect with the re-rendering confound; FARA therefore identifies the certificate on a *difference-in-differences* against a matched sham removal, as specified in step 2b.

**(B) How it works.**

```pseudocode
# Frozen LLM L, projection head P, policy head pi, trace decoder T
# Perception context C = [v_1..v_K] (visual tokens), instruction x
# Simulator S realizes interventions (object removal, occlusion, lighting)
# Sham selector Sham(C, cert) returns a distractor object with pixel footprint
#   within +/-10% of the named object and saliency score within +/-1 rank

for each episode step (C, x, a_gt):
    # 1. Forward pass: trace + action + certificate
    h      = L(P(C), x)                      # frozen LLM hidden states
    trace  = T(h)                            # reasoning trace naming tokens + direction
    a      = pi(h)                           # action distribution
    cert   = CertHead(h)                     # {(token_id, sign, magnitude)} set, |cert|<=M=8

    # 2. Realize the intervention the trace predicts
    C_cf   = S.intervene(C, cert.token_ids)  # physically remove/occlude those tokens
    h_cf   = L(P(C_cf), x)
    a_cf   = pi(h_cf)

    # 2b. Matched sham removal (difference-in-differences control)
    sham_ids = Sham(C, cert)                 # size- and saliency-matched distractor
    C_sham   = S.intervene(C, sham_ids)
    h_sham   = L(P(C_sham), x)
    a_sham   = pi(h_sham)
    # DiD action shift: named-token effect minus re-rendering confound
    delta    = (a - a_cf) - (a - a_sham)     # = a_sham - a_cf

    # 3. Counterfactual consistency loss (do-calculus style, on DiD shift)
    #    necessity: DiD shift MUST exceed margin (named removal moves action
    #    more than sham removal does)
    L_nec  = relu(margin_n - KL(a_sham || a_cf))     # margin_n = 0.5 nats
    #    sufficiency: DiD shift MUST match predicted signed direction
    proj   = <delta, dir(cert)>                      # dir from trace
    L_suf  = relu(margin_s - proj)                   # margin_s = 0.2
    L_cf   = L_nec + L_suf

    # 4. Task loss on original + counterfactual (counterfactual must NOT
    #    still succeed on the same action -> penalize spurious reliance)
    L_task = CE(a, a_gt)
    L_spur = relu(margin_p - KL(a_cf || a_gt_onehot))  # cf action should degrade
    loss   = L_task + lambda_cf * L_cf + lambda_sp * L_spur   # lambda_cf=1.0, lambda_sp=0.3

# 5. Inference: certificate gate
#    If cert is empty OR L_cf < tau (=0.1) -> action uncertifiable
#    -> route to fallback: re-query frozen LLM with minimal sufficient context
#       (tokens retained by cert across the last H=3 steps)
```

**(C) Why this design.** We chose to *physically realize* the intervention in VLNVerse's physics simulator rather than approximate it with token masking or prompt rewriting, because a masked token is still statistically present in the LLM's context window and thus does not constitute a true do-operation, accepting the cost that each training step requires a second simulator rollout (roughly 2x compute). We chose a *signed direction* from the reasoning trace over a scalar importance score because a scalar cannot distinguish "this token caused the action" from "this token is correlated with the action," accepting the cost that the trace decoder must produce a parseable direction and can itself be wrong (mitigated by the sufficiency loss penalizing mismatched directions). We chose a *hard certificate gate with a minimal-context fallback* over a soft confidence-weighted blend because the gate produces an identifiable binary flag (certified vs. spurious) that downstream error-propagation analysis can consume, accepting the cost that some borderline-correct actions get routed to the slower fallback. A domain expert would not describe FARA as a variant of Uni-NaVid's unified VLA training or SAME's mixture-of-experts routing: those methods optimize observational task loss and never compute a counterfactual, so they cannot produce the necessity/sufficiency certificate at all; FARA's loss is defined on a *paired* (original, intervened) context that no prior navigation training objective constructs.

**(D) Why it measures what we claim.** The KL divergence `KL(a_sham || a_cf)` between sham-removal and named-removal action distributions measures *causal necessity* of the certificate tokens because the difference-in-differences subtracts the re-rendering/off-manifold confound that a single physical removal would otherwise contaminate, making the divergence an estimate of the named token's interventional effect under the assumption that the sham removal is matched on pixel footprint and saliency so that its re-rendering effect is exchangeable with the named removal's; this assumption fails when no distractor of comparable footprint exists in the scene, in which case the divergence reflects residual mismatch between the two re-renderings rather than the token's causal contribution. The signed projection `<delta, dir(cert)>` measures *directional sufficiency* because the trace's predicted sign is a hypothesis about the DiD intervention's effect that the projection tests against, under the assumption that the policy head's output space is locally linear and unimodal in the direction of the action shift; this assumption fails near decision boundaries where the action distribution is multimodal, in which case the projection reflects the dominant mode's shift rather than the trace's specific prediction. The certificate emptiness flag at inference measures *spuriousness* of an action because an action with no token whose DiD removal changes the distribution cannot be causally attributed to the claimed perception context, under the assumption that the certificate head's top-M tokens cover the true causal parents; this assumption fails when the causal parent is a distributed pattern spanning more than M tokens, in which case the flag reflects certificate capacity rather than genuine spuriousness. The fallback's minimal sufficient context measures *error-propagation containment* because re-querying with only certified tokens removes the uncertified (potentially spurious) context, under the assumption that certified tokens from prior steps remain valid parents; this assumption fails after a large viewpoint change, in which case the fallback reflects stale context rather than a clean minimal sufficient set.

## Contribution

(1) A counterfactual intervention certificate mechanism that couples a frozen-LLM projection/policy head to simulator-realized do-operations, producing per-action necessity and directional-sufficiency scores that no purely observational navigation training objective (Uni-NaVid, SAME, NWM) can compute. (2) A paired-context counterfactual consistency loss with a spuriousness penalty that converts the faithfulness assumption into a measurable training invariant, yielding an identifiable certified-vs-spurious action flag. (3) A minimal-sufficient-context fallback protocol that re-queries the frozen LLM using only certified tokens, providing a concrete error-propagation containment procedure for embodied agents.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VLNVerse (GRBench tasks) | Realistic simulator supports do-operation interventions |
| Primary metric | Certified-action success rate | Directly measures gated, causally-attributed actions |
| Baseline | Uni-NaVid (unified VLA) | Tests whether counterfactual certificates beat observational training |
| Baseline | Classic VLN seq2seq | Tests whether certificate gating helps without LLM priors |
| Baseline | SAME (MoE routing) | Tests whether hard certificates beat soft routing confidence |
| Baseline | Attention-rollout / integrated-gradients saliency (soft certificate) | Tests whether physical do-operation beats saliency proxies |
| Ablation-of-ours | FARA w/o counterfactual loss | Tests whether necessity/sufficiency loss drives gains |
| Ablation-of-ours | FARA w/o sham control (single removal) | Tests whether DiD control removes re-rendering confound |
| Ablation-of-ours | FARA w/ M ∈ {2,4,8,16} | Tests top-M coverage assumption |
| Diagnostic | Local-linearity sweep (intervention magnitude) | Tests monotonicity of action shift in predicted direction |
| Diagnostic | Projection-vs-true-direction agreement on multimodal states | Tests sufficiency projection under multimodality |
| Diagnostic | Certificate transfer to one GRBench manipulation task | Tests significance beyond navigation |
| Diagnostic | Fraction of deployment actions certifiable | Quantifies practical utility |
| Compute analysis | Wall-clock and GPU-hour cost of full pipeline vs. cached/surrogate variant | Tests feasibility |

### Why this setup validates the claim

The central claim is that physically realized counterfactual certificates let FARA distinguish causally necessary perception tokens from spurious correlations, so its certified actions succeed more often than actions chosen by methods that only optimize observational task loss. VLNVerse is the right substrate because its physics simulator can actually remove or occlude named objects, making the do-operation the certificate names a real graph surgery rather than a token mask. Uni-NaVid tests the sub-claim that a unified VLA trained only on task loss cannot produce necessity/sufficiency certificates; SAME tests the sub-claim that soft mixture-of-experts routing confidence is not the same as a hard causal flag; the classic seq2seq baseline tests whether the gain requires LLM hidden-state structure or whether any policy with a certificate head would suffice. The attention-rollout / integrated-gradients saliency baseline tests the sub-claim that a physical do-operation yields certificates that saliency proxies cannot, isolating the novel component. The ablation removes the counterfactual consistency loss while keeping the certificate head and gate, so any gap between FARA and the ablation isolates the loss's contribution from architecture; the no-sham ablation isolates the difference-in-differences control's contribution to removing the re-rendering confound. Certified-action success rate is the right metric because it conditions on the gate firing, so it directly measures whether the certificate's causal claim holds at deployment; a method that certifies everything or nothing will score poorly, and a method that certifies the wrong tokens will show high un-certified fallback rates or failures on the certified subset. The M-sweep and local-linearity sweep directly probe the top-M coverage and local-linearity assumptions stated in (D), reporting where each breaks; the transfer diagnostic tests whether the certificate extends to a non-navigation GRBench task; the certifiable-fraction diagnostic quantifies practical utility; and the compute analysis reports wall-clock and GPU-hour cost of the full pipeline versus a cached/surrogate variant.

### Expected outcome and causal chain

**vs. Uni-NaVid** — On a case where the instruction says "turn left at the red chair" and a red chair appears in the periphery while the actual turn cue is a doorway, Uni-NaVid's unified VLA attends to the salient red chair and turns early because its observational task loss never asks which token the action depends on. Our method instead computes the difference-in-differences action shift after simulator-removing the chair tokens versus a matched sham removal; the DiD shift barely moves, so the certificate head is forced toward the doorway token, and the gated action uses that certified token. We expect a noticeable gap on distractor-object subsets but parity on clean-instruction subsets.

**vs. Classic VLN seq2seq** — On a case where a lighting change darkens the target object, the seq2seq baseline's fixed visual encoder loses the object and produces a stalled or random action because it has no mechanism to name which perception tokens its action depends on. Our method's trace names the lighting-sensitive tokens, the simulator intervenes on them, and the sufficiency loss forces the predicted direction to match the observed DiD action shift; at inference the certificate gate flags the action as uncertifiable and routes to minimal-context fallback, which re-queries with only the still-certified doorway tokens. We expect a larger gain over seq2seq on lighting/occlusion subsets than on standard subsets.

**vs. SAME (MoE routing)** — On a case where two objects are equally salient and the correct one is only weakly correlated with the instruction, SAME's soft routing blends experts with high confidence on both and commits to the wrong object because routing confidence is not a causal test. Our method's hard certificate gate requires the DiD KL to exceed the necessity margin and the signed projection to exceed the sufficiency margin; when the wrong object's tokens are removed the DiD action shift does not exceed the sham-matched baseline, so the gate rejects that certificate and the fallback uses the instruction-aligned tokens. We expect SAME and FARA to be close on easy instances but FARA to show a clear certified-success advantage on ambiguous-salience instances.

**vs. Attention-rollout / integrated-gradients saliency (soft certificate)** — On a case where a distractor object is highly salient but causally irrelevant, saliency proxies assign it high importance because they measure correlation with the model's attention or gradient, not the effect of removing it. Our method's DiD removal shows the action does not move relative to the sham, so the certificate rejects the distractor; the saliency baseline, lacking a do-operation, certifies the distractor and fails on the certified subset. We expect FARA to outperform the saliency baseline specifically on distractor-object and ambiguous-salience subsets, isolating the physical do-operation as the novel component.

**vs. FARA w/o counterfactual loss** — Without the necessity/sufficiency loss, the certificate head is trained only by task loss and cannot learn which tokens are causally necessary; the gate either certifies everything or nothing. We expect the ablation to match FARA on clean subsets but to show a large gap on distractor, occlusion, and lighting subsets, confirming the loss drives the certificates.

**vs. FARA w/o sham control (single removal)** — Without the difference-in-differences control, the measured action shift conflates the named token's effect with the re-rendering confound; the certificate head is misled by off-manifold artifacts. We expect the no-sham ablation to show inflated necessity scores on cluttered scenes and a higher false-certification rate, with the gap to FARA concentrated on scenes with large re-rendering changes (occlusion boundaries, lighting).

**Diagnostic: local-linearity sweep** — We sweep intervention magnitude (fraction of named object removed: 25%, 50%, 75%, 100%) and measure whether the action shift is monotone in the predicted direction. We expect monotonicity to hold for small-to-moderate magnitudes and to break near decision boundaries, reporting the magnitude at which projection-vs-true-direction agreement drops below 0.7.

**Diagnostic: top-M coverage** — We vary M ∈ {2,4,8,16} and measure certificate-emptiness rate versus ground-truth causal-parent size (estimated by exhaustive removal on a held-out subset). We expect emptiness rate to decrease with M and to plateau when M exceeds the true parent size, reporting the M at which coverage saturates.

**Diagnostic: transfer to manipulation** — We apply FARA's certificate gate to one GRBench manipulation task and report certified-action success rate versus Uni-NaVid, testing whether the certificate extends beyond navigation.

**Diagnostic: certifiable fraction** — We report the fraction of deployment actions that are certifiable (gate fires) across all subsets, quantifying practical utility.

**Compute analysis** — We report wall-clock and GPU-hour cost of the full pipeline (two simulator rollouts per step) versus a cached/surrogate variant that batches interventions across steps and approximates the second rollout with a learned surrogate validated against the true simulator on a held-out subset.

### What would falsify this idea

If FARA's certified-action success rate gain over Uni-NaVid is uniform across distractor, occlusion, and clean subsets rather than concentrated on the subsets where the predicted causal failure mode occurs, then the certificate is not doing causal work and the central claim is wrong. Likewise, if the ablation without the counterfactual loss matches FARA on certified-action success, the loss is not what produces the certificates; and if the no-sham ablation matches FARA, the difference-in-differences control is not removing the re-rendering confound.

## References

1. VLNVerse: A Benchmark for Vision-Language Navigation with Versatile, Embodied, Realistic Simulation and Evaluation
2. Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks
3. GRUtopia: Dream General Robots in a City at Scale
4. Navigation World Models
5. Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method
6. SAME: Learning Generic Language-Guided Visual Navigation with State-Adaptive Mixture of Experts
7. Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots
8. Qwen Technical Report
