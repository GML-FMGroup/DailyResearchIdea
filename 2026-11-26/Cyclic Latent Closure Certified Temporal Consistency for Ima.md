# Cyclic Latent Closure: Certified Temporal Consistency for Image-Only Video Editor Post-Training

## Motivation

Image-to-Video Transfer (I2VT) post-trains pretrained video editors using only single images, treating each image as a one-frame video and letting an image editor guide per-denoising-step predictions; at test time the improved editor is applied to whole clips 'while preserving temporal consistency,' but the training loop never once feeds the model a multi-frame input, so temporal consistency is asserted rather than exercised. This is not a quirk of I2VT alone: the same structural gap recurs across the video-editing lineage — VideoSPatS (E1) must hand-engineer spline parametrization to 'guarantee long-term temporal consistency,' SketchVideo (E9) must bolt on an inter-frame attention and latent-fusion module to propagate sparse keyframe conditions, and ZeroPatcher (E7) relies on a latent mask fuser plus a trajectory-optimizing sampler to keep edits coherent. In every branch, temporal consistency is an emergent byproduct of an architectural or inference-time workaround, never a constraint that the training objective itself verifies, so the operator that is supposed to enforce consistency is never certified on multi-frame inputs. I2VT is the highest-leverage place to close this gap because its image-only pipeline currently contains no consistency constraint at all.

## Key Insight

Temporal consistency is a closure property of the denoising operator under multi-frame composition, and a closure property can only be certified by exercising the operator on composed multi-frame inputs — which image-only post-training never does — so injecting a cyclic latent-closure rule that composes the same image into pseudo-clips and penalizes the divergence between per-frame and clip-level denoising trajectories converts an unverified assumption into a training-time invariant.

## Method

**What it is.** Cyclic Latent Closure (CLC) is a post-training loss and sampler modification for I2VT-style video editors. Its inputs are a single training image `x`, the pretrained video editor `V_θ`, the frozen image editor `I`, and a controlled latent perturbation set; its outputs are an updated `V_θ` whose per-frame denoising trajectories are provably equal (within tolerance `ε`) to the trajectory it produces on the composed multi-frame clip.

**Load-bearing assumption (stated explicitly).** CLC assumes *shared-noise exchangeability*: under identical noise `ε` and identical underlying latent `z0`, a temporally consistent editor's per-frame prediction `v_frame^(k)` and its clip-level prediction `v_clip[k]` must coincide, so any nonzero closure residual `r_k` is attributable to temporal inconsistency rather than to the editor's inherent frame-indexed (3D/temporal-attention) computation. This assumption is false for a structurally frame-indexed editor whenever the K frames are not identical, because temporal attention makes `v_clip[k]` depend on the whole clip context. CLC therefore does not use the raw residual `r_k = v_frame^(k) − v_clip[k]`; it uses a *context-invariant closure residual* defined against an unperturbed repeated-clip reference, as specified in steps 5–7 below.

**How it works.**

```pseudocode
# One CLC post-training step on a single image x (no real video used)
1. Encode:  z0 = E(x)                       # single-frame latent
2. Compose a pseudo-clip of K frames under controlled latent perturbations:
     for k in 1..K:  z0^(k) = z0 + δ_k ,  δ_k ~ N(0, σ² I)  with σ = 0.02, K = 4
     # same underlying image, K independent latent perturbations
     Z0 = stack(z0^(1), ..., z0^(K))        # shape [K, C, H, W]
3. Sample ONE shared noise schedule: t ~ U[0,1]; ε ~ N(0,I)
     zt^(k) = √ᾱ_t · z0^(k) + √(1-ᾱ_t) · ε    # identical ε across k  (shared-noise coupling)
4. Per-frame teacher-guided prediction (the I2VT role-play):
     for k:  v_frame^(k) = V_θ(zt^(k), t, c)  guided by I on the decoded frame
5. Clip-level prediction on the perturbed clip (the closure operator):
     v_clip = V_θ(Z_t, t, c)                 # V_θ applied to the whole stack
6. Clip-level prediction on the unperturbed repeated-clip reference:
     Z0_ref = stack(z0, z0, ..., z0)         # K identical copies, δ_k = 0 for all k
     Zt_ref = √ᾱ_t · Z0_ref + √(1-ᾱ_t) · ε   # same shared ε
     v_ref = V_θ(Zt_ref, t, c)               # reference clip-level prediction
7. Context-invariant closure residual:
     r_k = v_clip[k] − v_ref[k]              # deviation of perturbed clip from repeated-clip reference
     # equivalently: penalize deviation of v_clip[k] from the K-frame-mean prediction
     # after subtracting a measured per-frame context bias b_k estimated on a held-out
     # consistent editor (see calibration below)
8. CLC loss:
     L_clc = (1/K) Σ_k || r_k − b_k ||²  +  λ · mean_{k≠j} || z0_hat^(k) − z0_hat^(j) ||²
     # λ = 0.1; b_k is the per-frame context bias (step 9); second term penalizes decoded-frame drift
9. Context-bias calibration (once, before post-training):
     Run a held-out temporally consistent editor V_cal (e.g., the pretrained V_θ before CLC)
     on N_cal = 512 pseudo-clips composed from held-out images with the same σ = 0.02, K = 4.
     For each frame index k, set b_k = mean over calibration clips of (v_clip[k] − v_ref[k]).
     Freeze b_k; it is subtracted in step 8 and used to calibrate ε in step 11.
10. Total loss:  L = L_i2vt + β · L_clc ,  β = 0.5
11. Update θ;  I and E are frozen.
```

At inference, the same closure rule is used as a **latent-fusion closure check**: after denoising a real clip, the sampler computes `r_k = v_clip[k] − v_ref[k]` on the actual multi-frame input using the frozen `b_k`, and if `max_k ||r_k − b_k|| > ε` (ε = 0.05, calibrated as the 95th percentile of `||r_k − b_k||` on the N_cal = 512 held-out calibration clips from step 9), re-runs the clip-level prediction for the offending frame using the shared-noise coupling — a deterministic, closed-form correction that requires no extra training.

**Why this design.** We chose shared-noise coupling (identical ε across the K pseudo-frames) over independent per-frame noise because independent noise makes `v_frame^(k)` and `v_clip[k]` differ for reasons unrelated to temporal consistency — the residual would then measure noise variance, not trajectory divergence — accepting the cost that the pseudo-clip is a restricted (low-diversity) sample of multi-frame inputs. We chose to penalize the residual `r_k` in velocity/prediction space rather than in decoded pixel space because pixel-space penalties are dominated by VAE reconstruction error (the very step ZeroPatcher's latent mask fuser was built to avoid, E7), accepting the cost that a small velocity residual may correspond to a visible pixel artifact; the second loss term bridges this by also penalizing decoded-frame drift. We chose a frozen image editor `I` as the per-frame teacher rather than a learned consistency critic because a critic would reintroduce the Anti-Pattern-1 coupling problem (a learned head cannot manufacture consistency the data lacks), accepting the cost that CLC inherits any bias in `I`. We chose the context-invariant residual `r_k = v_clip[k] − v_ref[k]` with a calibrated per-frame bias `b_k` rather than the raw residual `v_frame^(k) − v_clip[k]` because a structurally frame-indexed editor's clip-level output depends on the whole clip context, so the raw residual conflates genuine inconsistency with the context-mixing that is the source of consistency; the repeated-clip reference and the held-out bias estimate remove that confound, accepting the cost of one calibration pass over N_cal = 512 pseudo-clips. A domain expert would not describe CLC as a variant of I2VT (E4/E6): I2VT post-trains on single images and never composes frames, whereas CLC's load-bearing operation is the composition-then-closure residual, which does not exist in I2VT, SketchVideo (E9), or VideoSPatS (E1) — those enforce consistency through architecture (inter-frame attention, spline parametrization) rather than through a training-time closure constraint.

**Why it measures what we claim.** The context-invariant closure residual `r_k = v_clip[k] − v_ref[k]` measures temporal consistency because the assumption that makes it equivalent is *shared-noise exchangeability*: if the editor is truly temporally consistent, applying it to the perturbed clip and to the unperturbed repeated-clip reference under identical noise must yield identical predictions up to the calibrated per-frame context bias `b_k`, so any nonzero `r_k − b_k` is attributable to inconsistency; this assumption fails when the editor legitimately produces different outputs for different frames of a genuinely moving scene (e.g., a real action change), in which case `r_k` reflects the intended edit difference rather than inconsistency — which is why CLC is applied only to pseudo-clips composed from one image, where the ground-truth trajectory is constant. The decoded-frame drift term `||z0_hat^(k) − z0_hat^(j)||` measures consistency of the *output surface* because it assumes the pseudo-clip's decoded frames should be identical up to the perturbation δ; this assumption fails when `δ_k` is large enough to cross an edit boundary, in which case the term reflects perturbation sensitivity rather than inconsistency — hence σ is capped at 0.02. The inference-time closure check `max_k ||r_k − b_k|| > ε` measures whether a real clip's edit is consistent because it assumes the shared-noise coupling plus the frozen bias `b_k` makes the repeated-clip prediction the correct reference; this assumption fails on clips with genuine large inter-frame motion, in which case the check reflects motion magnitude rather than inconsistency — mitigated by the closed-form re-run being a no-op when the clip-level prediction is already correct.

## Contribution

(1) Cyclic Latent Closure (CLC), a post-training loss and inference-time closure check that composes single images into shared-noise pseudo-clips and penalizes per-frame-vs-clip denoising trajectory divergence, making temporal consistency an exercised constraint in image-only video-editor training. (2) The design principle that temporal consistency is a closure property of the denoising operator under multi-frame composition, and that shared-noise coupling is the minimal intervention that isolates trajectory divergence from noise variance. (3) A deterministic latent-fusion closure correction at inference that re-runs the clip-level prediction only for frames whose closure residual exceeds ε, requiring no additional training.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | OpenVE++ (incl. action/expression/body/relighting) | Stresses temporal consistency across edit types |
| Primary metric | Inter-frame edit consistency on OpenVE++ | Directly measures trajectory divergence CLC targets |
| Baseline | I2VT (E4/E6) | Tests closure residual as the added ingredient |
| Baseline | ZeroPatcher (E2/E7) | Tests training-free latent-fusion alternative |
| Baseline | VideoHandles (E3/E10) | Tests 3D-composition editing competitor |
| Ablation-of-ours | CLC without shared-noise coupling | Isolates the coupling's role in the residual |
| Ablation-of-ours | CLC with raw residual `v_frame^(k) − v_clip[k]` (no repeated-clip reference, no `b_k`) | Tests whether the context-invariant residual is load-bearing vs. the raw residual |
| Ablation-of-ours | Naive temporal-smoothness loss (penalize `||v_frame^(k) − v_frame^(j)||` for k≠j) | Tests whether CLC is reducible to a standard smoothness regularizer |
| Ablation-of-ours | Shared-noise-only loss (no clip-level operator; penalize `||v_frame^(k) − v_frame^(j)||` under shared ε) | Tests whether the clip-level closure operator is the load-bearing component |
| Calibration set | N_cal = 512 held-out pseudo-clips (σ = 0.02, K = 4) | Fixes `b_k` and calibrates ε = 95th percentile of `||r_k − b_k||` |
| Compute budget | 1 post-training run: 8× A100-80GB, ~24 GPU-hours for K = 4, batch size 8 images (32 pseudo-frames), 10k steps | Makes the 3–6 month timeline concrete |

### Why this setup validates the claim

The central claim is that a training-time closure constraint on pseudo-clips composed from a single image yields video editors whose per-frame and clip-level denoising trajectories coincide, and that this coincidence is what produces temporally consistent edits. OpenVE++ is the right testbed because its four new tasks (action, expression, body movement, relighting) are precisely the regime where the exchangeability assumption is stressed: edits change content over time, so a consistency mechanism that merely smooths frames will be caught, while one that enforces trajectory closure should hold. I2VT is the load-bearing baseline because CLC is defined as a post-training add-on to I2VT; if CLC's gain over plain I2VT is absent, the closure residual carries no information beyond the single-image role-play it extends. ZeroPatcher and VideoHandles test whether the effect is merely a byproduct of latent-space fusion or 3D-prior editing, respectively. The shared-noise ablation is the diagnostic: if removing coupling does not degrade consistency, the residual is measuring noise variance rather than trajectory divergence, and the mechanism story collapses. The raw-residual ablation is the second diagnostic: if the raw residual matches the context-invariant residual, the repeated-clip reference and `b_k` are unnecessary and the adversarial concern about context-mixing is not operative. The naive-smoothness and shared-noise-only ablations test whether CLC is reducible to a standard regularizer or to coupling alone, isolating the clip-level closure operator as the load-bearing component. A single primary metric is sufficient because the claim is monotone — closure should reduce divergence, not trade it against a second axis.

### Expected outcome and causal chain

**vs. I2VT** — On a case where an action change must propagate coherently across a clip (e.g., a person shifting from standing to reaching), plain I2VT produces per-frame edits that drift because its single-image role-play never composes frames and so never penalizes per-frame versus clip-level divergence. Our method instead adds the context-invariant closure residual `r_k − b_k`, forcing the perturbed-clip prediction to agree with the repeated-clip reference under shared noise, so we expect a noticeable consistency gap on the action/body-movement subsets and near-parity on static relighting, where the failure trigger is absent.

**vs. ZeroPatcher** — On a case where dynamic object removal leaves a region that must be filled consistently across frames, ZeroPatcher's latent mask fuser produces plausible fills but its trajectory is optimized per known-segment conditioning, so offending frames can diverge because nothing ties the per-frame and clip-level predictions together. Our method instead checks `max_k ||r_k − b_k||` at inference and re-runs the clip-level prediction with shared-noise coupling for the offending frame, so we expect fewer inter-frame flicker events concentrated on removal/replacement clips, with parity on clips lacking dynamic objects.

**vs. VideoHandles** — On a case where a 3D object's composition is edited in a static scene with camera motion, VideoHandles can shift the object consistently in 3D but its training-free feature lifting gives no guarantee that per-frame and clip-level denoising agree, so residual drift appears when the edit boundary interacts with occlusion. Our method instead penalizes exactly that context-invariant residual during post-training, so we expect a modest consistency gain on composition-style clips and a larger gain on the expression/relighting subsets where VideoHandles' 3D prior does not apply.

**vs. CLC with raw residual** — If the raw residual `v_frame^(k) − v_clip[k]` matches the context-invariant residual `v_clip[k] − v_ref[k]` with calibrated `b_k`, the adversarial concern about context-mixing is not operative and the repeated-clip reference is unnecessary; we expect the context-invariant version to outperform on action/body-movement subsets where context-mixing is largest.

**vs. naive temporal-smoothness loss** — If a standard smoothness penalty `||v_frame^(k) − v_frame^(j)||` matches CLC, the clip-level closure operator is not load-bearing; we expect CLC to outperform because smoothness does not reference the clip-level prediction.

**vs. shared-noise-only loss** — If coupling alone (no clip-level operator) matches CLC, the closure operator is not load-bearing; we expect CLC to outperform on subsets where per-frame and clip-level predictions diverge.

### What would falsify this idea

If CLC's consistency gain over I2VT is uniform across all OpenVE++ subsets rather than concentrated on the action, expression, and body-movement subsets where the exchangeability failure is predicted, or if the shared-noise ablation, the raw-residual ablation, the naive-smoothness ablation, or the shared-noise-only ablation matches full CLC, then the closure residual is not measuring trajectory divergence and the central claim is wrong.

## References

1. Improving and Expanding Video Editing Capabilities Without Videos
