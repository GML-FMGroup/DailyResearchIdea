# Type-Codebook Coupling for Identifiable Text-Conditioned Structural Control

## Motivation

Unified structural-control branches such as SolarRenderer and CtrlVDiff collapse heterogeneous structural modalities (depth, edges, segmentation) into a single control LoRA and rely on plain text to carry the modality distinction, but neither establishes that different structural types induce distinguishable reference-token distributions. Structurally, this fails because the text encoder is never constrained to preserve a type-sufficient statistic: two different structural type labels can map to text embeddings whose projections onto the LoRA's reference-token insertion space are nearly identical, so the single control LoRA cannot recover the modality from text alone. The result is cross-type interference and silent type confusion whenever the text description is ambiguous or the type is under-specified.

## Key Insight

If the text encoder's structural-type subspace is made injective on the type label and coupled to the LoRA's reference-token insertion distribution through a shared contrastive type-codebook, then plain-text type labels become a sufficient statistic for selecting the correct structural reference-token distribution without any per-modality encoder.

## Method

**What it is.** TCC (Type-Codebook Coupling) is a training-time coupling mechanism for a frozen video backbone with a single control LoRA. It takes as input a plain-text structural description (which implicitly names a structural type) and a structural condition video, and outputs an insertion distribution over native reference tokens whose type-conditional mean is forced to match a learned type-codebook entry shared with the text encoder's type subspace.

**Load-bearing assumption (stated explicitly).** TCC assumes that the expected inserted token embedding `mu_r = softmax(r) @ V` is a sufficient statistic for structural type — i.e., the control LoRA's reference-token insertion distribution carries recoverable type identity that can be aligned to a text-side type codebook. This assumption fails when the LoRA inserts tokens in a type-agnostic spatial pattern (e.g., always at frame boundaries regardless of type), in which case `mu_r` reflects insertion position rather than type. To make this assumption testable and to guard against it, we add a type-probe calibration step (step 0 below) and a distribution-level fallback coupling (step 3b) that uses the full insertion distribution `softmax(r)` rather than only its mean when the probe shows the mean is insufficient.

**How it works.**

```pseudocode
# Frozen backbone B, single control LoRA L, text encoder E (frozen except type-projection P_t)
# Structural condition video s, plain-text description t (contains type label y in {depth, edge, seg, ...})
# Learnable type codebook C = {c_y} in R^d, one vector per structural type
# Type-probe calibration set: 512 held-out (s, t, y) triples, stratified across the 8 types

# Step 0 (calibration, run once before training):
#   Train a linear probe g_mean: mu_r -> y on the calibration set using frozen L (initial LoRA weights).
#   Train a linear probe g_dist: softmax(r) -> y on the same set.
#   If accuracy(g_mean) < 0.8 * accuracy(g_dist), set USE_DIST_COUPLING = True; else False.
#   Report both accuracies in the paper as the sufficiency diagnostic.

for each training step (s, t, y):
    # 1. Text-side type subspace
    h_t = E(t)                          # token embeddings
    z_t = P_t(h_t)                      # type-projection head -> d-dim (2-layer MLP, hidden=256, GeLU)
    # 2. Reference-token insertion distribution from control LoRA
    r = L(s)                            # reference-token insertion logits over K slots (K=64)
    mu_r = softmax(r) @ V               # expected inserted token embedding, d-dim
    # 3. Contrastive type-codebook coupling
    L_type = -log( exp(sim(z_t, c_y)/tau) / sum_{y'} exp(sim(z_t, c_{y'})/tau) )
    L_ins  = || mu_r - c_y ||^2         # insertion mean matched to same codebook entry
    L_cb   = -log( exp(sim(mu_r, c_y)/tau) / sum_{y'} exp(sim(mu_r, c_{y'})/tau) )
    # 3b. Distribution-level fallback (only if USE_DIST_COUPLING):
    #     L_dist = -sum_k softmax(r)_k * log( p_y(k) ), where p_y is a type-conditional categorical prior
    #     over K slots, initialized as uniform and updated by EMA from softmax(r) for samples of type y.
    #     When USE_DIST_COUPLING, replace L_ins with L_dist and keep L_cb on mu_r for codebook alignment.
    # 4. Injectivity regularizer on type subspace (per-batch)
    L_inj  = sum_{y != y'} max(0, margin - ||z_t(y) - z_t(y')||)^2
    # 5. Structure-guided distribution matching (frozen backbone)
    L_dm   = DMD_loss(B, L, s, t)
    L = L_dm + lambda_1 * L_type + lambda_2 * L_ins + lambda_3 * L_cb + lambda_4 * L_inj
```

Hyperparameters: tau = 0.07, margin = 0.5, lambda_1 = lambda_2 = lambda_3 = 0.1, lambda_4 = 0.05, codebook size |C| = number of structural types (8 in our corpus), K = 64 insertion slots, d = 256 (matching the LoRA reference-token embedding dimension), P_t is a 2-layer MLP with hidden width 256 and GeLU activation. The codebook C is updated by exponential moving average (decay 0.99) from z_t, not by gradient, to avoid collapse. Training uses batch size 8 with type-stratified sampling (at least one sample per type per batch when possible), LoRA rank 16, learning rate 1e-4 for P_t and 1e-5 for LoRA, 20k steps. DMD loss follows the standard configuration from the frozen backbone's reference implementation (same as SolarRenderer); if DMD is unstable (loss diverges or FVD plateaus before 10k steps), fall back to a standard denoising diffusion loss with the same LoRA and P_t. Estimated compute: 4×A100-80GB, ~120 GPU-hours per full run including calibration.

**Why this design.** We chose a shared codebook between the text type-projection and the LoRA's insertion mean over a per-modality classifier because the codebook forces both surfaces to live on the same type-sufficient statistic, accepting the cost that the codebook must be pre-specified in size and can under-cover rare types. We chose an injectivity margin regularizer over a pure contrastive loss because contrastive loss alone can collapse when the text encoder is frozen and type labels are rare, accepting the cost of an extra hyperparameter (margin) that must be tuned per corpus. We chose to keep the backbone and text encoder frozen except for a small type-projection head over fine-tuning the text encoder, because fine-tuning the encoder would destroy general text understanding and because the injectivity condition is a property of the subspace, not the full embedding, accepting the cost that the type-projection head must be trained from scratch. We added the type-probe calibration (step 0) and the distribution-level fallback (step 3b) because the load-bearing assumption that `mu_r` is type-sufficient is not guaranteed by construction; the probe makes it measurable on a 512-example calibration set, and the fallback provides a concrete alternative coupling when the mean is insufficient, without changing the core codebook-coupling training procedure. A domain expert would not describe TCC as a variant of SolarRenderer's plain-text control or of a standard contrastive retrieval head, because TCC couples two distributions (text type subspace and LoRA insertion mean) through a shared codebook and enforces injectivity, which neither SolarRenderer nor a retrieval baseline does.

**Why it measures what we claim.** The quantity `mu_r` measures the structural-type identity of the reference-token distribution because the assumption that the expected inserted token embedding is a sufficient statistic for the type; this assumption fails when the LoRA inserts tokens in a type-agnostic spatial pattern (e.g., always at frame boundaries regardless of type), in which case `mu_r` reflects insertion position rather than type. The quantity `z_t` measures the text-side type label because the assumption that the type-projection head is injective on the type label; this assumption fails when two type labels are paraphrased identically in the corpus, in which case `z_t` reflects lexical overlap rather than type. The contrastive loss `L_cb` measures type identifiability because the assumption that the codebook entries are well-separated; this assumption fails when the EMA codebook collapses, in which case `L_cb` reflects codebook geometry rather than type separation. The type-probe accuracy on `mu_r` (step 0) measures whether the sufficiency assumption holds in practice because it directly tests whether type is recoverable from `mu_r` alone; this assumption fails when the probe accuracy is low relative to the probe on the full distribution `softmax(r)`, in which case the distribution-level fallback (step 3b) is activated and the mean-based coupling `L_ins` is replaced.

## Contribution

(1) A modality-agnostic structural type identifiability condition for text-conditioned structural control, operationalized as an injectivity constraint on the text encoder's type subspace. (2) A contrastive type-codebook coupling mechanism that ties the text type subspace to the control LoRA's reference-token insertion distribution through a shared codebook, requiring no per-modality encoders. (3) A training procedure that combines the coupling with structure-guided distribution matching and an injectivity margin regularizer, yielding a single control LoRA whose reference-token distributions are type-separable under plain-text descriptions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Frame-aligned multi-type structural corpus (depth/edge/seg) | Holds type labels needed for codebook coupling |
| Primary metric | Type-conditional structural fidelity (per-type FVD) | Detects type-specific gains, not aggregate blur |
| Diagnostic metric | Type-probe accuracy on `mu_r` vs. on full `softmax(r)` (512-example calibration set, stratified across 8 types) | Tests the load-bearing sufficiency assumption before training |
| Diagnostic metric | Cosine similarity matrix of `z_t` across types and of `mu_r` across types | Demonstrates the claimed type-collapse in baselines |
| Baseline | SolarRenderer plain-text control | Tests whether shared codebook beats text-only control |
| Baseline | Standard contrastive retrieval head | Tests whether injectivity regularizer matters |
| Baseline | Per-modality type classifier | Tests whether shared codebook beats separate classifiers |
| Ablation-of-ours | TCC without injectivity regularizer (λ₄=0) | Isolates the injectivity sub-claim |
| Ablation-of-ours | TCC with distribution-level coupling (step 3b) forced on | Tests whether the fallback changes per-type FVD when the mean is insufficient |

### Why this setup validates the claim

The central claim is that coupling the text type subspace and the LoRA insertion mean through a shared, injective codebook makes the structural type identifiable from either surface, and that this identifiability is what drives type-correct rendering. The plain-text-control baseline (SolarRenderer) tests the sub-claim that a shared codebook is necessary: if text-only control already matches TCC per type, the coupling buys nothing. The standard contrastive retrieval head tests the sub-claim that injectivity matters: it has a codebook-like objective but no margin regularizer, so any gap concentrated on rare or lexically-overlapping types isolates the injectivity contribution. The per-modality classifier tests whether sharing one codebook across surfaces beats two independent classifiers, which is the coupling claim proper. The λ₄=0 ablation is the direct diagnostic: if removing injectivity does not degrade rare-type fidelity, the injectivity argument is empty. The type-probe diagnostic (step 0) directly tests the load-bearing sufficiency assumption: if the probe on `mu_r` is much less accurate than the probe on the full distribution `softmax(r)`, the mean-based coupling is not measuring type, and the distribution-level fallback is the appropriate specification. The cosine similarity matrices of `z_t` and `mu_r` across types provide the direct evidence that type-collapse exists in the baselines before the fix. Per-type FVD is the right metric because the predicted effect is type-conditional; an aggregate metric would average away exactly the signal the mechanism predicts.

### Expected outcome and causal chain

**vs. SolarRenderer plain-text control** — On a case where the text description names a rare type (e.g. segmentation on a cluttered scene) whose lexical form overlaps a common type, the baseline produces blurred or type-drifting structure because its text encoder has no type-sufficient statistic and no insertion-side anchor, so the type signal is diluted by lexical overlap. Our method instead forces both `z_t` and `mu_r` onto the same codebook entry, so the insertion mean is pulled toward the correct type even when the text is ambiguous, because the coupling term supplies a second, video-grounded path to the type. We expect a noticeable per-type FVD gap on rare/overlapping types but near-parity on common, lexically distinct types. The cosine similarity matrix of `z_t` across types for SolarRenderer is expected to show high similarity (>0.9) for the overlapping pair, while TCC's is expected to be lower (<0.5).

**vs. Standard contrastive retrieval head** — On a case where two type labels are paraphrased similarly in the corpus, the baseline produces collapsed type representations because contrastive loss alone with a frozen encoder cannot separate near-duplicate labels, so its retrieval and conditioning drift between the two types. Our method instead adds the injectivity margin, which pushes `z_t(y)` and `z_t(y')` apart by at least the margin, so the two types remain distinguishable downstream. We expect the gap to be concentrated on the lexically-overlapping type pairs, with the baseline's per-type FVD degrading sharply there while ours stays roughly flat.

**vs. Per-modality type classifier** — On a case where the structural condition video is noisy (e.g. depth with holes) so the insertion pattern is spatially irregular, the baseline's separate classifier produces a type estimate that disagrees with the text-side estimate because the two surfaces are trained independently and can drift apart. Our method instead ties both surfaces to one EMA codebook, so a noisy insertion mean is regularized toward the shared entry rather than free to drift. We expect lower variance in per-type FVD across noisy clips and a smaller text-video type-disagreement rate, with the largest gap on the noisiest subset.

**vs. TCC with distribution-level coupling (step 3b) forced on** — On the calibration set, if the type-probe on `mu_r` is insufficient (accuracy < 0.8 × accuracy on `softmax(r)`), the mean-based coupling `L_ins` is expected to underperform the distribution-level coupling on rare types whose insertion geometry is shared with common types. We expect the distribution-level variant to show lower per-type FVD on those rare types, confirming that the full distribution carries type information that the mean discards.

### What would falsify this idea

If TCC's gain over the plain-text-control and retrieval baselines is uniform across all types rather than concentrated on rare, lexically-overlapping, and noisy subsets where the predicted failure modes occur, then the coupling and injectivity mechanisms are not doing the claimed causal work. Equally, if the λ₄=0 ablation shows no degradation on those same subsets, the injectivity sub-claim is falsified. If the type-probe on `mu_r` achieves accuracy comparable to the probe on `softmax(r)` (ratio ≥ 0.95) and the cosine similarity matrices show no type-collapse in the baselines, then the load-bearing sufficiency assumption and the claimed gap are both unsupported.

## References

1. SolarRenderer: Few-Step Generative Reference Rendering
2. Video-As-Prompt: Unified Semantic Control for Video Generation
3. Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield
