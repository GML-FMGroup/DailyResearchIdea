# QuotientFlow: Encoder-Invariant Semantic Conditioning for OCT Latent Flow Matching via Null-Space Projection

## Motivation

OCTFlow conditions its latent Flow-Matching sampler on a frozen foundation encoder's embedding, so the conditioning surface inherits encoder-specific and acquisition-specific directions. This is the same structural failure that DeCLIP addresses with paired perceptual/semantic text supervision, but OCT lacks such paired corpus and CLIP's language directions do not span scanner, contrast, and speckle nuisance. The field's trajectory is toward encoder-agnostic generative augmentation, yet no existing method defines a conditioning signal that is invariant to which frozen encoder produced it and to acquisition appearance variation.

## Key Insight

The semantic code is the quotient of the frozen embedding by the joint group of encoder identity and acquisition augmentation, and this quotient is identifiable from encoder-swap and augmentation pairs because the group action is exactly the variation that must be discarded, making the conditioning signal invariant by construction rather than by learned independence.

## Method

**What it is** — QuotientFlow is a learned projection that maps a frozen OCT encoder embedding into a semantic code used as the sole conditioning input to OCTFlow's latent Flow-Matching sampler; it is trained on paired embeddings from the same image under encoder swaps and acquisition augmentations so that the semantic code is invariant to both nuisance factors while a residual perceptual code absorbs their variation. **Load-bearing assumption (stated explicitly):** the difference between two independently pretrained frozen OCT encoders (RETFound vs. DINOv2-OCT) lies, approximately, in a low-dimensional linear subspace that a rank-128 linear projection's null space can absorb, so that a single linear P_theta can map both encoders' embeddings of the same image to the same semantic code while preserving class-relevant content. This assumption is not asserted as a guarantee; it is calibrated and verified empirically before any FID interpretation (see Step 0 and Step 6 below).

**How it works** —
```pseudocode
Input: OCT image x; frozen encoders E1 (RETFound), E2 (DINOv2-OCT);
       augmentation set A = {contrast (0.7-1.3), gamma (0.7-1.4),
       speckle (Gaussian sigma 0.02-0.10), scanner-blur (Gaussian kernel 3-7 px)}
Output: semantic code s(x) in R^d_s used to condition the Flow-Matching sampler

Step 0 — Calibration of the encoder-swap operator (before training P_theta):
  Draw a calibration set C of N_cal = 512 OCT images from the training split.
  Compute Z1 = [E1(x)]_{x in C}, Z2 = [E2(x)]_{x in C} in R^{N_cal x d}.
  Estimate the linear encoder-swap operator R in R^{d x d} by orthogonal Procrustes:
     R = argmin_{Q^T Q = I} ||Z1 Q - Z2||_F^2  (closed form via SVD of Z1^T Z2).
  Report the Procrustes residual r_cal = ||Z1 R - Z2||_F / ||Z2||_F on a held-out
  calibration split (N_cal/2). This residual is the empirical measure of how
  linear the encoder-swap map is; it is logged and reported alongside all FID numbers.
  Also compute the top-k singular values of (Z2 - Z1 R) to record the rank of the
  non-linearizable residual.

Step 1 — Embedding pairs (training):
  z1 = E1(x), z2 = E2(x)
  z1a = E1(a(x)), z2a = E2(a(x)) for a ~ A

Step 2 — Semantic projection P_theta: R^d -> R^d_s (linear, rank d_s = 128 < d)
  s(x) = P_theta z1;  s_a(x) = P_theta z1a
  s_swap(x) = P_theta z2
  P_theta is parameterized as P_theta = U V^T with U in R^{d x d_s}, V in R^{d_s x d};
  U is initialized to the top-d_s left singular vectors of Z1 (from Step 0) and V to
  the corresponding right singular vectors, so training starts from a content-preserving
  subspace rather than the collapse solution P_theta -> 0.

Step 3 — Null-space constraint:
  Loss_inv = ||s(x) - s_swap(x)||^2 + ||s(x) - s_a(x)||^2
  Loss_res = ||(I - P_theta^+ P_theta) z1 - (I - P_theta^+ P_theta) z2||^2
  Loss_recon = ||P_theta^+ P_theta z1 - z1||^2
  L = Loss_inv + lambda_res Loss_res + lambda_rec Loss_recon
  Collapse guard: Loss_recon is computed on the full z1 (not only the semantic span),
  and lambda_rec = 0.1 is kept non-zero throughout; if ||P_theta||_F drops below 0.5 *
  ||P_theta^{(0)}||_F during training, lambda_rec is doubled for the next epoch
  (logged as a collapse event).

Step 4 — Train P_theta with lambda_res = 0.5, lambda_rec = 0.1, d_s = 128, 50 epochs,
  Adam lr = 1e-3, batch size 64, one encoder-swap pair and one augmentation pair per
  image per step.

Step 5 — Condition OCTFlow: s(x) replaces the frozen-encoder embedding in the
  Flow-Matching velocity network v_t(z_t, t, s(x)).

Step 6 — Null-space diagnostic (run before interpreting any FID gap):
  On a held-out diagnostic set D of 256 images, compute the measured encoder-swap
  difference vector delta(x) = E2(x) - E1(x) R (using R from Step 0).
  Project delta(x) onto the learned null space of P_theta:
     delta_null(x) = (I - P_theta^+ P_theta) delta(x)
  Report the absorption ratio rho = mean_x ||delta_null(x)|| / mean_x ||delta(x)||.
  The residual loss is interpreted as measuring nuisance absorption only when rho is
  high (rho >= 0.8); if rho is low, the FID gap is reported as reflecting encoder-
  specific semantic drift, not nuisance absorption.
```

**Why this design** — We choose a linear projection with an explicit null-space penalty over a nonlinear disentanglement head because the nuisance group (encoder identity and acquisition augmentation) is not guaranteed to be linearly separable in the embedding, but the linear projection's null space provides a well-defined subspace that can be constrained to absorb variation; accepting the cost that nonlinear residual nuisance may leak into the semantic code. We choose paired encoder-swap and augmentation losses over text supervision because OCT lacks a paired perceptual/semantic text corpus, and CLIP's language directions do not span scanner/speckle nuisance; accepting the cost that the invariance is limited to the augmentations we simulate. We choose to condition the Flow-Matching sampler only on the semantic code over conditioning on the full embedding, because the full embedding entangles encoder-specific geometry with class/mask semantics; accepting the cost that we discard perceptual information that could aid generation fidelity. A domain expert would not describe this as a variant of DeCLIP because DeCLIP relies on text-guided disentanglement with a paired corpus, whereas our mechanism defines the semantic code as a quotient by a group estimated from encoder-swap and augmentation pairs, with no text involved. The Procrustes calibration in Step 0 and the absorption diagnostic in Step 6 are included because the linear-separability assumption is load-bearing and must be measured, not assumed.

**Why it measures what we claim** — The invariance loss ||s(x) - s_swap(x)||^2 measures encoder-invariance of the semantic code because we assume that the two frozen encoders differ only by a nuisance transformation group; this assumption fails when the encoders have different semantic biases (e.g., one is trained on a different disease taxonomy), in which case the loss reflects encoder-specific semantic drift rather than nuisance — the Procrustes residual r_cal and absorption ratio rho in Step 6 are the direct diagnostics of whether this assumption holds. The residual loss ||(I - P^+P)z1 - (I - P^+P)z2||^2 measures that the discarded subspace absorbs encoder-identity variation because we assume the null space of P spans the encoder-identity direction; this assumption fails when the encoder difference has components outside the null space, in which case the residual retains encoder-specific structure — again measured by rho. The augmentation invariance loss ||s(x) - s_a(x)||^2 measures acquisition-invariance because we assume the augmentation set spans the acquisition nuisance group; this assumption fails when real scanner variation includes transformations not in A, in which case the semantic code still carries unmodeled acquisition appearance.

## Contribution

['A quotient-projection mechanism that defines the semantic conditioning signal for OCTFlow as the quotient of a frozen encoder embedding by the joint group of encoder identity and acquisition augmentation, estimated from encoder-swap and augmentation pairs without text supervision.', 'A training objective that couples an invariance loss on the semantic code with a residual perceptual loss on the discarded subspace, making encoder identity and appearance nuisance live in the same null space.', 'A conditioning surface for latent Flow Matching that is invariant to which frozen encoder produced the embedding, enabling encoder-agnostic conditional OCT synthesis.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RETOUCH/OCT-CL multi-scanner OCT; training split N=8,000 images, calibration set N_cal=512, diagnostic set N=256, test split N=2,000 | Multi-scanner, multi-encoder nuisance is present |
| Primary metric | Class-conditional FID under encoder swap (E1→E2), reported alongside the Step-6 absorption ratio rho and Procrustes residual r_cal | Directly probes encoder-invariance of conditioning |
| Baseline | OCTFlow full-embedding conditioning | Tests whether quotient code beats entangled embedding |
| Baseline | DeCLIP-style text-disentangled conditioning | Tests whether text-free quotient is competitive |
| Baseline | DisQUE appearance/content disentanglement | Tests linear null-space vs appearance disentanglement |
| Ablation-of-ours | QuotientFlow without null-space residual loss (lambda_res = 0) | Isolates contribution of residual absorption term |
| Ablation-of-ours | QuotientFlow with soft-penalty only (no Procrustes calibration, no Step-6 diagnostic) | Tests whether calibration changes the interpretation |

### Why this setup validates the claim

The central claim is that conditioning OCTFlow on a code that is invariant to encoder identity and acquisition nuisance yields more faithful and more controllable conditional generation than conditioning on the raw embedding, and that the gain is attributable to the quotient construction rather than to extra capacity. The full-embedding OCTFlow baseline tests the first sub-claim: if our code is truly nuisance-free, generated samples should be stable across encoder swaps where the baseline drifts. The DeCLIP-style and DisQUE baselines test whether our text-free, linear-null-space mechanism is a genuine alternative rather than a relabeling of existing disentanglement. The ablation removes the residual term, so any residual gap isolates the null-space mechanism rather than the invariance loss alone. Class-conditional FID under encoder swap is the right metric because it directly quantifies the predicted effect — invariance should show up as low variance across encoder conditions, not merely low absolute FID. Crucially, the Step-6 absorption ratio rho and the Step-0 Procrustes residual r_cal are reported for every condition: if rho is low, the FID gap cannot be attributed to nuisance absorption and is instead reported as encoder-specific semantic drift, which is the falsification condition stated below.

### Expected outcome and causal chain

**vs. OCTFlow full-embedding conditioning** — On a case where the same image is encoded by RETFound versus DINOv2-OCT, the baseline's conditioning vector shifts with encoder-specific geometry, so the sampler produces class-consistent but structurally drifted samples whose FID degrades when the conditioning encoder is swapped. Our method maps both embeddings through the same quotient projection, so the sampler sees a near-identical semantic code and produces structurally stable samples. We expect a noticeable FID gap on the encoder-swap subset but near-parity on the single-encoder subset, and we expect rho >= 0.8 on the diagnostic set; if rho < 0.8 the gap is reported as semantic drift, not nuisance absorption.

**vs. DeCLIP-style text-disentangled conditioning** — On a case where scanner speckle and gamma shift the appearance, the DeCLIP baseline's language directions do not span these nuisance axes, so its conditioning retains speckle-correlated variation and generated samples inherit scanner-specific texture. Our method instead constrains the null space to absorb augmentation-pair variation, so the semantic code stays fixed while the residual absorbs appearance. We expect our method to show lower FID variance across acquisition augmentations but similar mean FID to DeCLIP on clean single-scanner data.

**vs. DisQUE appearance/content disentanglement** — On a case where a nonlinear appearance transform (e.g., strong gamma plus blur) is applied, the DisQUE baseline's appearance-content split assumes a smoother decomposition, so appearance leaks into the content code and conditioning drifts. Our linear null-space penalty explicitly absorbs the paired augmentation direction, so the semantic code remains stable. We expect a concentrated gain on the strong-augmentation subset, with parity on mild augmentations.

**vs. QuotientFlow without null-space residual loss** — On a case where encoder identity has components outside the semantic span, the ablation's invariance loss alone cannot absorb them, so the residual retains encoder structure and conditioning drifts under encoder swap. Our full method routes that variation into the null space, so we expect the ablation to recover most of the single-encoder performance but lose specifically on the encoder-swap subset, isolating the residual term's contribution. The Step-6 rho diagnostic is reported for both the ablation and the full method so the gap can be attributed to measured absorption rather than to the loss term by name.

**vs. QuotientFlow with soft-penalty only (no calibration)** — On the same encoder-swap subset, the soft-penalty-only variant starts from a random U, V initialization and has no Procrustes calibration, so it is more likely to collapse toward P_theta -> 0 and its rho is expected to be lower. We expect the calibrated full method to show higher rho and a larger encoder-swap FID gap over this variant, which tests whether the calibration step is doing the work the quotient construction claims.

### What would falsify this idea

If our gain over the full-embedding baseline is uniform across encoder-swap and single-encoder subsets rather than concentrated where encoder identity changes, or if the Step-6 absorption ratio rho is below 0.8 on the diagnostic set, the quotient mechanism is not doing the causal work we claim. Likewise, if the null-space ablation matches the full method on the encoder-swap subset, the residual absorption term is not the source of invariance and the central claim fails.

## References

1. Language-Guided Visual Perception Disentanglement for Image Quality Assessment and Conditional Image Generation
2. Joint Quality Assessment and Example-Guided Image Processing by Disentangling Picture Appearance from Content
3. AesExpert: Towards Multi-modality Foundation Model for Image Aesthetics Perception
4. OCTFlow: Unified conditional generation for balanced classification and annotated segmentation in retinal OCT.
