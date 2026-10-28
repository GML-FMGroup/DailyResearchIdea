# Motion-Counterfactual Identifiability for Triplet-Supervised Video Style Transfer

## Motivation

Triplet-supervised video style transfer, as instantiated in VISTA, minimizes a reconstruction loss on (content, style, stylized) triples but imposes no constraint that the inferred style code is non-predictive of motion. Because style references and content videos in VISTA-1000 share correlated motion statistics, the model can achieve low triplet loss while entangling motion into the style branch, which then breaks temporal consistency whenever motion changes. This is a structural failure of factor separability assumptions inherited from Composer: neither tree formalizes an identifiability condition on the learned style, content, and motion representations, so low loss does not imply the style code is a sufficient statistic for style alone.

## Key Insight

A style representation is identifiable as motion-invariant exactly when it is invariant under a known counterfactual intervention on the motion field while the motion code changes by the intervened amount, a condition that triplet loss cannot express because it never holds motion fixed.

## Method

**What it is.** MC-ID (Motion-Counterfactual Identifiability) is a training-time constraint for VISTA-style diffusion-transformer video style transfer. It takes as input a content video $x_c$, a style reference $x_s$, and their stylized output $\hat{y}=G_\theta(x_c,x_s)$, and augments the standard triplet loss with a counterfactual consistency loss computed on a motion-perturbed twin $\hat{y}'$ of $\hat{y}$. The twin is synthesized by warping $\hat{y}$ with the style reference's motion field, and the loss requires that the style code inferred from $\hat{y}'$ equals that inferred from $\hat{y}$, while the inferred motion code changes by exactly the known warp.

**Load-bearing assumption (stated explicitly).** The method assumes that warping $\hat{y}$ by the style reference's RAFT motion field $m_s$ is a *style-preserving pure-motion intervention*: it changes only the motion factor and leaves the style content — and hence the style code $A_\phi(\hat{y})$ — unchanged, so that any measured change in $\hat{s}$ is attributable to motion entanglement rather than to the warp itself. This assumption is not free: forward warping produces occlusion/disocclusion holes, boundary tearing, and resampling blur, and RAFT flow is unreliable in high-motion, occlusion-heavy regions. To make the assumption testable and to gate the loss on its valid support, the method includes (i) an occlusion-aware, cycle-consistent warp and (ii) a calibration control that verifies near-zero $L_{id}$ on a style-only reference before the signal is trusted.

**How it works.**

```pseudocode
Input: content video x_c (T frames), style reference x_s (T frames), stylized output y_hat = G_theta(x_c, x_s)
Hyperparameters: lambda_id = 0.5, lambda_mot = 0.1, flow estimator F (pre-trained RAFT, frozen),
                 occlusion threshold tau_occ = 0.5 (forward-backward consistency), calibration set size = 512 triplets

1.  # Extract motion fields (forward and backward for cycle consistency)
    m_c        = F(x_c)              # content motion field, T-1 frames
    m_s_fwd    = F(x_s)              # style reference forward flow, T-1 frames
    m_s_bwd    = F(reverse(x_s))     # style reference backward flow, T-1 frames
    m_y        = F(y_hat)            # stylized output motion field

2.  # Build occlusion-aware, cycle-consistent counterfactual twin
    # Forward-backward consistency check to mask unreliable flow regions
    fb_err     = || m_s_fwd + Warp(m_s_bwd, m_s_fwd) ||_2   # per-pixel flow cycle error
    occ_mask   = (fb_err < tau_occ)                         # 1 = valid (non-occluded), 0 = occluded
    y_hat_prime = Warp(y_hat, m_s_fwd) * occ_mask           # motion-perturbed twin, masked
    m_y_prime   = F(y_hat_prime)                            # motion field of the twin

3.  # Infer style codes via the frozen style adapter A_phi (from VISTA)
    s_hat        = A_phi(y_hat)
    s_hat_prime  = A_phi(y_hat_prime)

4.  # Infer motion codes via a small motion encoder M_psi (trained jointly)
    # M_psi: 2-layer MLP, hidden=256, GeLU activation, output dim = 128
    z_m          = M_psi(m_y)
    z_m_prime    = M_psi(m_y_prime)

5.  # Counterfactual identifiability losses (L_id computed only on valid support)
    L_id  = || (s_hat - s_hat_prime) * occ_mask ||_2^2 / sum(occ_mask)   # style code invariant under motion intervention
    # Encode: linear projection of the known warp difference to motion-code space
    # Encode(d) = W_enc * d, where W_enc is a fixed 128 x 2 linear map (identity on first two channels)
    L_mot = || (z_m_prime - z_m) - Encode(m_s_fwd - m_c) ||_2^2

6.  # Standard triplet reconstruction loss (VISTA)
    L_trip = DiffusionLoss(G_theta(x_c, x_s), y_hat)

7.  # Total loss: L = L_trip + lambda_id * L_id + lambda_mot * L_mot

8.  # Calibration control (run once before training, on 512 triplets)
    # For each triplet, construct a style-only reference x_s0 with zero motion (temporally constant style frame)
    # Compute L_id on (y_hat, Warp(y_hat, F(x_s0))) and require mean L_id < 1e-3
    # If calibration fails, increase tau_occ or exclude high-motion triplets from L_id
```

**Why this design.** We chose to intervene on the *motion field of the style reference* rather than resample a new style or content video because the style reference's motion is the exact nuisance factor that triplet loss leaves uncontrolled: VISTA-1000 aligns style and content motion, so the model can cheat by reading motion into the style code, and only a targeted intervention on that nuisance factor can expose the cheat. We chose a *frozen pre-trained flow estimator* (RAFT) over a jointly learned motion encoder for the warp itself because the counterfactual must be a known, externally specified intervention—if the motion field were learned, the intervention would be endogenous and the identifiability constraint would collapse into the same entanglement it is meant to remove, accepting the cost that RAFT's errors propagate as noise into $L_{id}$. We chose an $L_2$ penalty on style-code difference over a KL divergence or adversarial discriminator because the style adapter $A_\phi$ is a deterministic encoder, so a mean-squared invariance penalty is the correct sufficient-statistic constraint; a discriminator would reintroduce the anti-pattern of a learned independence head over coupled variables, accepting the cost that $L_2$ is less sample-efficient than a contrastive objective. We chose to require the motion code to change by the *known* warp amount (via $L_{mot}$) rather than merely to change, because without the known target the model could satisfy $L_{id}$ by collapsing the style code to a constant; the motion-side constraint is what keeps the style code informative. A domain expert would not describe this as a variant of RelationAdapter or Composer: those methods *compose* factors at inference, whereas MC-ID *intervenes* on a factor during training to impose an identifiability condition that composition alone never tests.

**Why it measures what we claim.** The quantity $\|(s_\text{hat} - s_\text{hat}') \cdot \text{occ\_mask}\|_2^2 / \sum \text{occ\_mask}$ measures *motion-invariance of the style code* because the only difference between $\hat{y}$ and $\hat{y}'$ is the warp by $m_s$ on the non-occluded support, under the assumption that the warp is a valid intervention on motion that leaves style content unchanged; this assumption fails when the style reference's motion field is not a pure motion transformation (e.g., it contains appearance changes from occlusion or disocclusion), in which case $L_{id}$ reflects warp artifacts rather than style entanglement. The quantity $\|(z_m' - z_m) - \text{Encode}(m_s - m_c)\|_2^2$ measures *motion-code sensitivity to the known intervention* because the motion encoder $M_\psi$ is trained to linearly track the difference between motion fields, under the assumption that $M_\psi$ is expressive enough to represent the warp difference; this assumption fails when the warp difference lies outside $M_\psi$'s range, in which case $L_{mot}$ penalizes encoder capacity rather than motion-code correctness. The triplet loss $L_{trip}$ measures *stylization fidelity* because it is the diffusion reconstruction objective of VISTA, under the assumption that the VISTA-1000 triplets are motion-aligned; this assumption fails for real-world style references with independent motion, in which case $L_{trip}$ and $L_{id}$ conflict and the method degrades to VISTA.

## Contribution

["A counterfactual motion-intervention training objective (MC-ID) that turns the informal goal of style-motion disentanglement into an explicit identifiability constraint: the style code must be invariant under a known warp of the style reference's motion field, while the motion code must shift by exactly that warp.", "A concrete construction of the motion-perturbed twin via warping the stylized output with the style reference's motion field, which requires no new data and can be applied on top of any triplet-supervised video style transfer model that exposes a style adapter.", 'An empirical design principle: the motion-side constraint (known shift of the motion code) is necessary to prevent style-code collapse under the invariance penalty, and the two constraints together dominate triplet loss alone on temporal consistency when motion statistics are shifted at test time.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VISTA-1000 triplets | Motion-aligned triplets where nuisance factor is controlled |
| Primary metric | Style-code motion-invariance error (SMITE) | Directly measures the quantity L_id constrains |
| Baseline | VISTA | Tests whether MC-ID improves on standard triplet training |
| Baseline | RelationAdapter | Tests whether intervention beats inference-time composition |
| Baseline | Composer | Tests whether training-time identifiability beats factor composition |
| Ablation-of-ours | MC-ID w/o L_mot | Tests whether motion-side constraint is necessary |
| Ablation-of-ours | MC-ID w/o occlusion mask | Tests whether occlusion-aware gating is necessary |
| Ablation-of-ours | MC-ID w/o calibration control | Tests whether calibration gating is necessary |

### Why this setup validates the claim

The central claim is that triplet loss leaves style-motion entanglement uncontrolled, and that a targeted counterfactual intervention on the style reference's motion field removes it. VISTA-1000 is the right dataset because its triplets are motion-aligned, so any residual style-code sensitivity to motion is attributable to the entanglement the triplet loss fails to penalize—exactly the failure mode MC-ID targets. The primary metric, style-code motion-invariance error, directly operationalizes the quantity L_id constrains: it measures how much the frozen style adapter's output changes when only the motion field is intervened on. VISTA tests the sub-claim that MC-ID improves on the standard triplet objective; RelationAdapter and Composer test the sub-claim that training-time intervention beats inference-time factor composition; the ablation tests the sub-claim that the motion-side constraint is load-bearing rather than decorative. The additional ablations (w/o occlusion mask, w/o calibration control) test the load-bearing assumption that the warp is a style-preserving pure-motion intervention: if removing the occlusion mask or calibration control does not degrade SMITE, then the intervention-validity machinery is not necessary and the identifiability claim collapses into an ungrounded invariance penalty. If the metric does not separate these conditions in the predicted pattern, the identifiability argument fails.

### Expected outcome and causal chain

**vs. VISTA** — On a case where the style reference contains strong independent motion (e.g., a stylized waterfall reference with fast flow), VISTA produces a style code that drifts with the reference's motion because its triplet loss never penalizes motion leakage into the style adapter. Our method instead warps the stylized output by the style reference's motion field and penalizes style-code change, because L_id directly constrains the sufficient statistic to be motion-invariant. We expect a noticeable gap in style-code motion-invariance error on high-motion-reference subsets but parity on low-motion subsets, with a smaller but consistent gap in temporal consistency.

**vs. RelationAdapter** — On a case where the style reference's motion is correlated with its appearance (e.g., a flickering candle reference), RelationAdapter composes style and motion factors at inference, so its style code inherits the correlation because composition never tests invariance. Our method instead intervenes on motion during training and requires the style code to be unchanged, because L_id enforces the counterfactual. We expect RelationAdapter to show elevated style-code motion-invariance error specifically on correlated-motion references, while our method stays flat across correlation levels, producing a crossover pattern rather than a uniform gain.

**vs. Composer** — On a case where the content and style motions are misaligned (e.g., a static content video with a dynamic style reference), Composer's inference-time factor composition produces a stylized output whose style code still varies with the injected style motion, because it has no training signal separating the factors. Our method instead trains the style adapter to be invariant under exactly this intervention, because L_id is computed on the warped twin. We expect Composer's style-code motion-invariance error to grow with the magnitude of motion misalignment, while our method's error remains roughly constant, yielding a divergence that widens on the most misaligned subset.

**vs. MC-ID w/o L_mot** — On a case where the style code is allowed to collapse to a constant, the ablation without L_mot can satisfy L_id trivially because there is no motion-side constraint keeping the style code informative. Our method includes L_mot to require the motion code to change by the known warp amount, because without it the model could collapse the style code. We expect the ablation to show lower SMITE (apparent invariance) but also lower stylization fidelity (higher L_trip) and worse temporal consistency, revealing that the motion-side constraint is load-bearing for keeping the style code informative.

**vs. MC-ID w/o occlusion mask** — On a case where the style reference has high-motion, occlusion-heavy regions, the ablation without the occlusion mask computes L_id over all pixels, so warp artifacts (holes, tearing) leak into the style-code difference and the loss penalizes appearance change rather than motion entanglement. Our method masks out occluded/disoccluded regions using forward-backward consistency (tau_occ = 0.5), because the intervention is only valid on the non-occluded support. We expect the ablation to show higher SMITE on high-occlusion subsets and a weaker separation from VISTA, confirming that occlusion-aware gating is necessary for the intervention to be style-preserving.

**vs. MC-ID w/o calibration control** — On a case where the style reference is temporally constant (zero motion), the ablation without calibration control cannot verify that the warp itself introduces near-zero style-code change, so any residual L_id on such controls is misattributed to motion entanglement. Our method runs a calibration control on 512 triplets with style-only references and requires mean L_id < 1e-3 before trusting the signal. We expect the ablation to show inflated SMITE on low-motion subsets (false positives) and no improvement on high-motion subsets, confirming that calibration gating is necessary to ground the intervention-validity assumption.

### What would falsify this idea

The idea would be falsified if MC-ID's gain in style-code motion-invariance error is uniform across motion-correlation and misalignment subsets rather than concentrated where the predicted entanglement occurs, or if the ablation without L_mot matches MC-ID's performance, indicating the motion-side constraint is not load-bearing. Additionally, if the ablations without occlusion mask or without calibration control match MC-ID's performance, then the intervention-validity machinery is not necessary and the identifiability claim collapses into an ungrounded invariance penalty.

## References

1. VISTA: Triplet-Supervised Video Style Transfer with Diffusion Transformers
2. FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space
3. Composer: Creative and Controllable Image Synthesis with Composable Conditions
4. RelationAdapter: Learning and Transferring Visual Relation with Diffusion Transformers
5. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
6. In-Context LoRA for Diffusion Transformers
7. OmniGen: Unified Image Generation
8. Boosting Latent Diffusion with Flow Matching
