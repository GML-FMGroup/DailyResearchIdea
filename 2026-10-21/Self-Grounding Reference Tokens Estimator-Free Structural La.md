# Self-Grounding Reference Tokens: Estimator-Free Structural Latents for Control-LoRA Video Generation

## Motivation

Across the current video-generation literature, structure is always exogenous: SolarRenderer injects structural videos as native reference tokens at described frame positions, Video-As-Prompt feeds a reference video through a separate MoT expert, and per-step structure-map conditioning pipelines supply depth/pose/edge maps at every denoising step. The shared structural failure is that none of these systems can produce those structural tokens from the target video's own latent state — they must be estimated externally and frozen before generation, so the generator can never jointly refine structure with appearance, and any estimator error is baked in irreversibly. SolarRenderer in particular exposes the native reference-token interface but only ever feeds it externally supplied structural videos, leaving the positional-interface closure of the reference pathway unestablished.

## Key Insight

Frame-position equivariance between the injected structural tokens and the backbone's own reference pathway makes the structural latent identifiable from the denoising trajectory alone, without any depth/pose/edge estimator, because the equivariance constraint uniquely pins the token at each frame position to the trajectory's own structural content.

## Method

**What it is.** We introduce **SGRT** (Self-Grounding Reference Tokens), a module that replaces the control-LoRA's externally supplied structural reference tokens with a per-frame structural latent inferred from the target video's own denoising trajectory. Inputs: the target latent state $z_t$, the denoising step index $t$, and the frozen backbone's reference-pathway slot layout. Output: a set of structural reference tokens $\hat{s}_{1:F}$ injected at the native reference-token positions of the control LoRA. No estimator (depth, pose, edge) is ever invoked.

**How it works.**

```pseudocode
# Frozen: MiniMax-H3 backbone B, control LoRA L_ctrl, four-step LoRA L_4step
# Trainable: SGRT encoder E_phi (2-layer MLP, hidden=256, GeLU), injected at native reference-token slots
# Reference-slot positional encoding: relative/convolutional (see note below)

for each training clip (target video v_{1:F}, text prompt c):
    z_T ~ q(z_T)                       # noised target latent, F frames
    for t in {T, ..., 1} (four-step schedule):
        # 1. Infer per-frame structural latent from CURRENT trajectory state
        h_t = B.denoiser_features(z_t, t, c)          # frozen backbone hidden states, layer 12 of 24
        s_hat_{1:F} = E_phi(h_t, t)                    # (F, d_ref=128) structural tokens

        # 2. Frame-position equivariance constraint (estimator-free grounding)
        #    Re-inject s_hat at the NATIVE reference slots the control LoRA expects,
        #    then require the denoised prediction to be invariant to a frame-shift
        #    permutation pi applied jointly to s_hat and to the reference-slot indices.
        s_perm = pi(s_hat_{1:F})                       # frame-shift by k (k ~ Uniform{1..F-1})
        eps_a = B.L_ctrl(z_t, t, c, ref_tokens=s_hat_{1:F})
        eps_b = B.L_ctrl(z_t, t, c, ref_tokens=s_perm, ref_slots=pi(slots))
        L_equiv = || eps_a - pi^{-1}(eps_b) ||^2       # equivariance residual

        # 3. Structure-guided distribution matching (SolarRenderer objective)
        #    keeps the learned control while SGRT supplies the tokens
        L_dm = DMD(eps_a, teacher=B(z_t, t, c))        # four-step DM objective

        # 4. Slot-consistency: injected tokens must lie in the control LoRA's
        #    native reference subspace, else the LoRA ignores them
        L_slot = || Proj_ref(s_hat_{1:F}) - s_hat_{1:F} ||^2

        loss = L_dm + lambda_eq * L_equiv + lambda_slot * L_slot
        # lambda_eq = 0.5, lambda_slot = 0.1 (fixed, no tuning)
        update E_phi only (backbone, L_ctrl, L_4step frozen)
```

**Note on reference-slot positional encoding.** To make the equivariance constraint well-posed, we replace the control LoRA's absolute positional encoding of reference slots with a relative/convolutional encoding: each slot index $i$ is encoded as a 1D convolution over the frame index with kernel size 3 and circular padding, so that shifting the slot indices by $k$ commutes with the encoding up to the shift. This is a specification of the existing slot layout, not a new module; it ensures that $\pi$ applied to slots is a true symmetry of the reference pathway.

**Why this design.** We chose an equivariance constraint over a structural-estimator regression head because a regression head would reintroduce exactly the exogenous-estimator dependency we are removing, accepting the cost that equivariance alone does not constrain absolute structural scale (only relative frame-to-frame structure) — a limitation we mitigate by anchoring scale through the frozen backbone's own reference-slot normalization. We chose to inject at the native control-LoRA reference slots rather than a parallel conditioning pathway because SolarRenderer's interface already exposes those slots and reusing them preserves the trained control LoRA's structure-following behavior, accepting the cost that SGRT is tied to SolarRenderer's slot layout rather than being backbone-agnostic. We chose to train only $E_\phi$ with the backbone and both LoRAs frozen because full fine-tuning would destroy the four-step distillation, accepting the cost that SGRT's capacity is bounded by a small encoder and cannot correct backbone-level structural errors. We chose the slot-consistency projection over an unconstrained token space because the control LoRA only responds to tokens in its native reference subspace, accepting the cost that SGRT cannot express structural content outside that subspace. A domain expert would **not** describe SGRT as a variant of Video-As-Prompt: VAP processes an external reference video through a separate MoT expert as a semantic prompt, whereas SGRT infers structural tokens from the target's own denoising trajectory and grounds them by frame-position equivariance against the backbone's native reference pathway — there is no external reference video and no separate expert.

**Why it measures what we claim.** The equivariance residual $L_\text{equiv}$ measures **structural self-consistency of the injected latent** because the assumption that the backbone's reference pathway is frame-position equivariant (i.e., shifting the reference slot indices and the tokens jointly commutes with the denoised prediction) makes the residual zero exactly when the inferred tokens carry the trajectory's own per-frame structure rather than an arbitrary exogenous signal; this assumption fails when the backbone's attention is not strictly frame-position local (e.g., long-range temporal attention leaks across the shift), in which case $L_\text{equiv}$ reflects attention non-locality rather than structural mis-grounding. The slot-consistency term $L_\text{slot}$ measures **interface compatibility of the inferred tokens** because the assumption that the control LoRA's native reference subspace is the only channel through which structural tokens influence generation makes the projection residual zero exactly when the tokens are actually consumed by the control pathway; this assumption fails when the LoRA partially attends outside its native subspace, in which case $L_\text{slot}$ under-reports the true influence of the tokens. The distribution-matching term $L_\text{dm}$ measures **fidelity of the generated video under self-supplied structure** because the assumption that the frozen teacher's distribution is the correct target makes DM a valid divergence; this assumption fails on out-of-distribution clips, in which case $L_\text{dm}$ reflects teacher bias rather than SGRT's structural quality.

## Contribution

(1) SGRT, a self-grounding module that replaces the control-LoRA's exogenous structural reference tokens with a per-frame structural latent inferred from the target video's own denoising trajectory and grounded by a frame-position equivariance constraint, requiring no depth/pose/edge estimator. (2) A design principle — native-reference-slot closure — showing that structural tokens become a jointly-inferred latent variable when the equivariance constraint ties them to the backbone's own reference pathway, rather than an externally supplied input. (3) A slot-consistency projection that keeps inferred tokens inside the control LoRA's native reference subspace, enabling the frozen control LoRA to consume self-inferred structure without retraining.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | SolarRenderer frame-aligned corpus (game-engine clips) | Provides ground-truth per-frame structure without external estimators |
| Primary metric | Frame-shift equivariance residual on held-out clips | Directly probes whether tokens carry trajectory-own structure |
| Baseline | SolarRenderer w/ estimator-supplied reference tokens | Tests whether removing exogenous estimators costs structural fidelity |
| Baseline | Video-As-Prompt (external reference video) | Tests whether self-inference beats external semantic prompting |
| Baseline | SGRT w/ frozen random encoder (no training) | Isolates whether learned inference matters vs. slot injection alone |
| Ablation-of-ours | SGRT minus L_equiv (DM + slot only) | Tests whether equivariance is what grounds structure |
| Calibration | Ground-truth structural tokens (from corpus) fed to L_ctrl | Measures attention-locality floor of L_equiv |

### Why this setup validates the claim

The central claim is that structural reference tokens can be inferred from the target video's own denoising trajectory and grounded without any exogenous estimator, while remaining consumable by the frozen control LoRA. The SolarRenderer corpus is the right dataset because it supplies frame-aligned ground-truth structure, letting us compute the equivariance residual against a known reference rather than a proxy. The estimator-supplied baseline tests the sub-claim that self-inference does not sacrifice structural fidelity: if SGRT matches it, removing the estimator is free; if it lags only on scale-sensitive subsets, that isolates the predicted scale limitation. The VAP baseline tests the sub-claim that trajectory-inferred structure is preferable to external semantic prompting on the same clips. The random-encoder baseline and the no-equivariance ablation together test the causal sub-claim: if the gain comes from learned inference and equivariant grounding, removing either should collapse the residual gap, not merely shave it. The equivariance residual is the right primary metric because it is zero exactly under the mechanism we claim, and a baseline that ignores frame structure cannot drive it low by accident. The ground-truth-token calibration provides the attention-locality floor: any residual above this floor on ground-truth tokens is attributable to attention non-locality, not to SGRT's inference, so we can compare SGRT's residual against the floor to isolate the grounding signal.

### Expected outcome and causal chain

**vs. SolarRenderer w/ estimator-supplied reference tokens** — On a clip where the external depth estimator mislabels a thin occluder (e.g., a fence against sky), the baseline injects a structurally wrong reference token and the control LoRA faithfully renders the wrong geometry, because its structure comes from an estimator that fails on that region. Our method instead infers tokens from the trajectory state itself, so the occluder's per-frame motion is already encoded and the equivariance constraint forces consistency across the shift, so we expect parity on well-estimated clips but a visible residual gap on occluder-heavy subsets, with SGRT lower.

**vs. Video-As-Prompt (external reference video)** — On a clip whose target motion is not present in any retrieved reference video, VAP conditions on a semantically similar but structurally mismatched prompt and produces plausible-but-wrong camera motion, because its structure is borrowed from an external video rather than the target. Our method instead derives tokens from the target's own denoising trajectory, so the motion prior is the target's, and we expect a clear gap on out-of-distribution motion clips with near-parity on clips that resemble the reference pool.

**vs. SGRT w/ frozen random encoder** — On any clip, the random encoder emits tokens that carry no trajectory structure, so even with slot injection the control LoRA receives noise-like references and the equivariance residual stays high, because the grounding signal is absent. Our method's trained encoder maps backbone hidden states to structure-bearing tokens, so the residual drops specifically where the encoder saw motion, and we expect a large residual gap on all motion subsets, confirming the gain is from learned inference not from slot reuse.

**vs. Ground-truth structural tokens (calibration)** — On any clip, feeding the corpus's ground-truth structural tokens to the control LoRA yields a residual that reflects only the backbone's attention non-locality and positional-encoding mismatch, because the tokens are perfectly structural. Our method's residual should approach this floor on well-estimated subsets, and any excess over the floor quantifies SGRT's inference error; we expect SGRT's residual to be within 10% of the floor on occluder-free clips and to exceed it on occluder-heavy clips, confirming that the residual measures grounding rather than attention artifacts.

### What would falsify this idea

If SGRT's equivariance residual is uniformly lower than every baseline across all subsets — including static and occluder-free clips where the estimator baseline should already be correct — rather than concentrated on the subsets where the predicted estimator failure and scale limitation occur, then the residual is not measuring structural self-grounding and the central claim is wrong. Likewise, if removing L_equiv leaves the residual unchanged, equivariance is not the mechanism doing the grounding.

## References

1. SolarRenderer: Few-Step Generative Reference Rendering
2. Video-As-Prompt: Unified Semantic Control for Video Generation
3. Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield
