# Cross-Signal Consistency Projection for Compositional Video Control

## Motivation

Controllable video generation systems such as Invoker-Render in Code World Model 2.0 accept multiple independent control sub-signals (structural proxies, appearance references, language) through a shared conditioning interface, but they assume those sub-signals are mutually consistent by construction — nothing in the architecture checks whether a supplied camera trajectory, an object motion trajectory, and an instance-subpart mask actually agree on the same scene state. AnimateAnything converts all control into frame-by-frame motion flows and improves temporal coherence via frequency-domain feature computation, yet it never verifies that independently specified controls remain mutually compatible; IM-Zero similarly enforces layout and trajectory alignment only through coarse-to-refined generation without an explicit cross-signal conflict measure. The structural root cause is that each control branch is projected into the conditioning space separately (or concatenated), so conflicts between sub-signals are absorbed silently into the latent and surface only as flicker or drift that pixel-level regularizers cannot attribute to a specific conflict.

## Key Insight

Independently supplied control sub-signals are mutually consistent if and only if their projections into a shared scene-state space agree on the per-element scene state, so a pairwise residual in that shared space is a direct, attributable measure of cross-element conflict that can be penalized before conditioning and re-measured after rendering.

## Method

**What it is.** We introduce CSP (Cross-Signal Projection), a compositional control-consistency regularizer. Inputs: a set of K=4 independently supplied control sub-signals (camera trajectory, object motion trajectory, instance-subpart mask, language prompt) plus the rendered video. Outputs: a per-element consistency score vector and a training-time penalty that reshapes the conditioning latents before they enter the renderer.

**Load-bearing assumption (stated explicitly).** CSP assumes a single shared projection head P maps heterogeneous sub-signals into a common scene-state slot space such that cosine agreement between slot assignments is monotone in true scene-state agreement — i.e., P does not collapse distinct scene states onto nearby slots. This is a cross-modal alignment claim with no supervision signal enforcing it: nothing in the loss ties q_k to a ground-truth scene state, only to render reconstruction and the consistency residual itself, so P is free to satisfy the objective by collapsing all sub-signals to a constant slot assignment (C_jk→0 everywhere), which trivially minimizes consistency_pre. This is the standard representation-collapse failure documented for joint-embedding and contrastive objectives (BYOL/VICReg literature; Jing et al., 'Understanding Dimensional Collapse in Contrastive Self-supervised Learning', ICLR 2022). The structural problem is worse than generic collapse: camera pose, dense optical flow, and instance masks have different intrinsic dimensionalities and invariances (FloVD, arXiv:2502.08244, explicitly uses background optical flow as the 3D-correlated signal precisely because camera and object motion are not separable in a shared flow space), so a single 128-slot soft assignment cannot be simultaneously injective on pose and on masks without an explicit disentanglement constraint that CSP does not provide. We therefore anchor P with a supervised per-slot scene-state target and a collapse diagnostic, and report probe accuracy as a precondition for trusting C_jk.

**How it works.**

```pseudocode
# Phase 1: Shared scene-state projection (anchored)
# Per-signal encoders E_k: 2-layer MLP, hidden=512, GeLU, output dim d=256
# Shared projection head P: linear R^d -> R^S, S=128 scene-state slots, softmax over slots
for each sub-signal s_k in {camera, object_traj, subpart_mask, language}:
    z_k = E_k(s_k)                      # per-signal encoder, dim d=256
    q_k = P(z_k)                        # shared projection head P: R^d -> R^S, S=128 scene-state slots
    # q_k is a soft assignment of sub-signal k onto S scene-state slots

# Phase 1b: Anchored scene-state target and collapse diagnostic
# Per-slot scene-state label y_k derived from the proxy corpus:
#   camera -> per-element 6-DoF pose (3 translation + 3 rotation, binned to S slots)
#   object_traj -> per-instance flow vector (binned to S slots)
#   subpart_mask -> per-instance mask occupancy (binned to S slots)
#   language -> per-element attribute token (binned to S slots)
# Supervised alignment loss: cross-entropy between q_k and y_k
L_align = mean_k CE(q_k, y_k)
# Collapse diagnostic: VICReg-style variance/covariance term on q_k
L_var = mean_k [ mean_s max(0, 1 - std(q_k[:,s])) + sum_{s!=s'} cov(q_k[:,s], q_k[:,s'])^2 ]
# Per-signal-type linear probe: must decode its own sub-signal from q_k
# Probe: linear R^S -> dim(s_k), trained jointly; report probe accuracy as precondition
L_probe = mean_k CE(Probe_k(q_k), s_k)

# Phase 2: Pairwise conflict tensor (pre-conditioning)
for each pair (j,k):
    C_jk = 1 - cosine(q_j, q_k)         # per-slot disagreement in [0,2]
    w_jk = softmax_over_k( -C_jk / tau ) # tau=0.1, reweights sub-signals by agreement
consistency_pre = mean_{j<k} C_jk
z_cond = sum_k w_k * q_k                # conflict-weighted conditioning latent

# Phase 3: Render with conflict-aware conditioning
video = Renderer(z_cond, appearance_ref)   # Invoker-Render-style interface

# Phase 4: Post-render verification (realized sub-signals)
for each sub-signal s_k:
    s_hat_k = V_k(video)                # frozen verifier: pose estimator / optical-flow tracker / mask segmenter
    q_hat_k = P(E_k(s_hat_k))           # SAME projection head P
consistency_post = mean_{j<k} (1 - cosine(q_hat_j, q_hat_k))

# Phase 5: Regularizer
L = L_render + lambda_pre * consistency_pre + lambda_post * consistency_post
      + lambda_align * L_align + lambda_var * L_var + lambda_probe * L_probe
# lambda_pre=0.5, lambda_post=0.5, lambda_align=0.1, lambda_var=0.1, lambda_probe=0.1
# consistency_post uses stop-gradient on V_k
```

**Why this design.** We chose a *shared* projection head P over per-signal conflict classifiers because a shared space makes pairwise cosine disagreement comparable across heterogeneous signal types (camera poses vs. masks vs. text), accepting the cost that P must be trained jointly and may under-fit rare signal combinations. We chose conflict-weighted reweighting of conditioning latents (Phase 2) over simply penalizing the loss, because penalizing alone leaves the renderer free to ignore the penalty under a strong reconstruction gradient, accepting the cost that reweighting can suppress a correct-but-minority sub-signal when the majority disagree. We chose a post-render verification stage with frozen verifiers (Phase 4) over a learned discriminator because frozen pose/flow/mask estimators give a signal anchored to observable video content rather than to the renderer's own latent statistics, accepting the cost that verifier error propagates into the consistency score. This differs from the frequency-domain coherence regularizer in AnimateAnything, which operates on latent flicker and cannot attribute a violation to a named sub-signal pair; a domain expert would not describe CSP as a variant of it because CSP operates on *pre-conditioning signal projections* and produces a per-pair attribution that a latent-frequency penalty structurally cannot produce.

**Why it measures what we claim.** `C_jk = 1 - cosine(q_j, q_k)` measures cross-element control consistency because it assumes that a shared projection head P maps each sub-signal to a scene-state slot assignment whose cosine agreement is monotone in whether the sub-signals describe the same scene state; this assumption fails when P collapses distinct scene states onto nearby slots (representation collapse), in which case C_jk reflects projection-space proximity rather than actual scene-state agreement. `consistency_post` measures *realized* control consistency because it assumes the frozen verifiers V_k recover the sub-signal actually present in the rendered video with bounded error; this assumption fails when the verifier itself is confused by occlusion or motion blur, in which case the score reflects verifier failure rather than renderer inconsistency. The conflict-weighted latent `z_cond` measures the renderer's effective control input because it assumes softmax reweighting by agreement preserves the dominant mutually-consistent signal; this assumption fails when all sub-signals are pairwise inconsistent, in which case z_cond reflects an arbitrary average rather than a coherent scene state. The anchored alignment loss `L_align` and variance/covariance term `L_var` measure whether P is injective on scene states because they assume the per-slot scene-state labels y_k are a faithful discretization of the true scene state; this assumption fails when the binning of continuous pose/flow/mask values onto S=128 slots merges distinct states, in which case L_align can be satisfied without true injectivity and the probe accuracy must be reported as a precondition for trusting C_jk.

## Contribution

['CSP, a compositional control-consistency regularizer that projects heterogeneous control sub-signals into a shared scene-state space, computes a pairwise conflict tensor before conditioning, and re-verifies realized sub-signals in the rendered output to produce per-element consistency scores.', 'A conflict-weighted conditioning mechanism that reweights sub-signals by mutual agreement, making cross-element consistency an explicit measurable property of the conditioning interface rather than an assumed-by-construction one.', 'An attribution protocol that localizes inconsistency to specific sub-signal pairs, enabling per-element consistency diagnostics on a structural-proxy video generation interface.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Multi-granularity proxy corpus with camera + trajectory + mask + text, plus programmatically perturbed conflict cases (one sub-signal perturbed per scene) | Matches Invoker-Render conditioning interface and supplies all sub-signals; perturbation synthesizes conflict without new data curation |
| Primary metric | Per-pair conflict-attributed consistency error | Directly measures CSP's claimed per-sub-signal-pair attribution |
| Secondary metric | Projection-space collapse diagnostic: effective rank of P and pairwise slot cosine distribution on held-out scenes, correlated with per-pair attribution error | Tests the load-bearing monotonicity assumption empirically |
| Secondary metric | Per-signal linear probe accuracy on q_k (precondition for trusting C_jk) | Detects representation collapse |
| Baseline | Invoker-Render (unregularized) | Tests whether CSP's pre/post regularizer adds control consistency |
| Baseline | AnimateAnything-style frequency-domain coherence penalty | Tests whether latent-flicker penalty can substitute for signal-pair attribution |
| Baseline | IM-Zero zero-shot motion control | Tests whether training-free layout/trajectory control closes the gap |
| Baseline | CLIP-style contrastive projection trained on the same sub-signal pairs | Tests whether generic cross-modal alignment substitutes for shared-slot soft-assignment |
| Baseline | Concatenation-plus-MLP conflict classifier | Tests whether a simple classifier substitutes for shared-slot structure |
| Ablation-of-ours | CSP without Phase 4 post-render verification | Isolates whether frozen-verifier anchoring drives realized consistency |
| Ablation-of-ours | CSP without L_align / L_var / L_probe (unsupervised P) | Isolates whether anchoring prevents collapse and drives attribution |

### Why this setup validates the claim

The central claim is that CSP's *pre-conditioning signal-projection* conflict tensor plus *post-render frozen-verifier* attribution produces consistency gains that a latent-frequency penalty cannot, because only CSP can name which sub-signal pair is violated. The dataset must therefore contain scenes where heterogeneous sub-signals (camera, object trajectory, subpart mask, language) genuinely conflict, so that a per-pair attribution metric is meaningful; a corpus with only mutually-consistent signals would saturate every method. The unregularized Invoker-Render baseline tests the sub-claim that conflict-aware reweighting of `z_cond` matters at all; the AnimateAnything-style frequency penalty tests the sharper sub-claim that attribution—not merely smoothness—is what closes the gap; IM-Zero tests whether the gain is just a zero-shot artifact. The CLIP-style contrastive and concatenation-plus-MLP baselines test whether the shared-slot soft-assignment structure—not generic cross-modal alignment—is what produces the gain. The collapse diagnostic and probe accuracy test the load-bearing monotonicity assumption directly: if effective rank is low or probe accuracy is near chance, C_jk cannot be trusted. The ablation without Phase 4 isolates whether realized-consistency improvement comes from verifier anchoring or purely from pre-conditioning reweighting; the ablation without anchoring isolates whether L_align/L_var/L_probe prevent collapse. The per-pair conflict-attributed consistency error is the right primary metric because it is the only one that can distinguish "video looks smooth" from "the named sub-signal pair agrees," which is exactly the distinction CSP claims to make and its competitors structurally cannot.

### Expected outcome and causal chain

**vs. Invoker-Render (unregularized)** — On a case where a language prompt requests an object to exit frame while the supplied object trajectory keeps it centered, the baseline produces a video that silently favors one signal (likely the stronger trajectory conditioning) because its shared conditioning interface has no mechanism to detect or reweight inter-signal disagreement. Our method instead computes `C_jk` in the shared projection space, down-weights the disagreeing pair via `w_jk`, and post-render verifiers flag the realized mismatch, so we expect a concentrated gain on conflict-heavy subsets and near-parity on mutually-consistent subsets.

**vs. AnimateAnything-style frequency-domain coherence penalty** — On a case where camera motion and object motion are individually plausible but mutually inconsistent, the frequency-domain baseline produces a temporally smooth but semantically contradictory video because its penalty operates on latent flicker and cannot attribute a violation to a named sub-signal pair. Our method instead localizes the conflict to the (camera, object_traj) pair through `q_j`, `q_k` cosine disagreement and penalizes that specific pair, so we expect a clear gap on the per-pair attribution metric while both methods look similar on raw flicker metrics.

**vs. IM-Zero zero-shot motion control** — On a case requiring subpart-level mask control combined with a fine-grained language prompt, the zero-shot baseline produces motion-consistent but layout-drifting video because its keyframe-extraction stage has no cross-signal consistency term to reconcile mask and text. Our method instead projects all four sub-signals onto shared scene-state slots and reweights by agreement, so we expect a moderate gap on combined mask+text subsets and parity on single-signal subsets.

**vs. CLIP-style contrastive projection** — On a case where camera and object trajectory conflict, the contrastive baseline aligns sub-signals in a generic embedding space without per-slot scene-state assignment, so its conflict score cannot attribute the violation to a specific pair and may collapse under the same objective. Our method instead uses shared-slot soft-assignment with anchored labels and variance/covariance regularization, so we expect lower per-pair attribution error and higher probe accuracy, demonstrating that the shared-slot structure—not generic contrastive alignment—drives the gain.

**vs. Concatenation-plus-MLP conflict classifier** — On a case with rare signal combinations, the concatenation-plus-MLP baseline overfits to common pairs and cannot generalize to held-out combinations, so its per-pair attribution error is high on rare pairs. Our method instead projects each sub-signal onto shared slots with a linear probe per signal type, so we expect better generalization to rare combinations and lower attribution error on held-out pairs.

**vs. CSP without Phase 4 (ablation)** — On a case where the renderer satisfies the reweighted `z_cond` in latent space but the verifier would reveal a realized mismatch, the ablation produces a video whose *pre*-consistency looks good but whose *post*-consistency degrades because nothing anchors the conditioning to observable content. Our method instead uses frozen pose/flow/mask verifiers with stop-gradient to penalize realized inconsistency, so we expect the ablation to match CSP on `consistency_pre` but lag on `consistency_post`, isolating verifier anchoring as the causal driver.

**vs. CSP without anchoring (ablation)** — On a case with rare signal combinations, the unsupervised-P ablation collapses q_k to a constant slot assignment (C_jk→0 everywhere), trivially minimizing consistency_pre but yielding near-chance probe accuracy and low effective rank. Our method instead uses L_align, L_var, and L_probe to prevent collapse, so we expect higher effective rank, higher probe accuracy, and lower per-pair attribution error, isolating anchoring as the driver of trustworthy attribution.

### What would falsify this idea

If CSP's gain over the unregularized baseline is uniform across conflict-heavy and mutually-consistent subsets rather than concentrated where the predicted failure mode occurs, then the per-pair conflict attribution is not doing the claimed work. Likewise, if the Phase 4 ablation matches full CSP on `consistency_post`, the frozen-verifier anchoring claim is wrong and the improvement is purely pre-conditioning reweighting; and if the collapse diagnostic shows low effective rank or near-chance probe accuracy, the monotonicity assumption fails and C_jk cannot be trusted.

## References

1. Code World Model 2.0: Towards Infinite Controllable Video Generation
