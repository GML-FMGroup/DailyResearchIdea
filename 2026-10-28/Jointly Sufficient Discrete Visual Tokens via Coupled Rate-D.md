# Jointly Sufficient Discrete Visual Tokens via Coupled Rate-Distortion-Identification

## Motivation

Unified multimodal systems such as Janus-Pro and its descendants (e.g., ILLUME, MetaMorph, LMFusion) inherit a decoupled visual encoding assumption: the understanding pathway uses its own encoder and the generation pathway uses another, so no single discrete tokenization is ever forced to be sufficient for both pixel reconstruction and semantic discrimination. Even the strongest discrete-token work, Factorized Vector Quantization, is evaluated only on multimodal understanding and never tests whether its tokens can also reconstruct pixels under the same codebook, because the two sufficiency conditions are never jointly constrained. This meta-gap persists because cross-pathway token compatibility is treated as a design choice rather than a property to be enforced by the training objective.

## Key Insight

A single discrete codebook can be forced to be a jointly sufficient statistic for pixels and semantics if and only if its tokens are trained under a coupled rate-distortion-identification objective in which the semantic decoder is a fixed linear probe on the token embedding, because any capacity allocated to one pathway necessarily reduces the other, making joint sufficiency a non-trivial constraint that cannot be satisfied by concatenating separate VQ and classifier modules.

## Method

**What it is.** We introduce **JSRDI** (Jointly Sufficient Rate-Distortion-Identification tokenizer), a single-codebook discrete visual tokenizer whose encoder maps each image patch to one codebook entry, trained so that every token simultaneously (i) reconstructs its patch within a bounded distortion and (ii) is linearly decodable to the patch's semantic label by a fixed linear probe on the token embedding. Inputs: image patches; outputs: discrete token indices, reconstructed patches, and a fixed linear semantic classifier.

**Load-bearing assumption (stated explicitly).** JSRDI assumes that a fixed linear probe on the discrete token embedding is a sufficient statistic for the patch's semantic label, so that minimizing CE(W^T e_k, y) with W fixed (not re-fit) forces the tokens to carry intrinsic semantic content rather than merely being linearly separable in a low-dimensional space that the probe can exploit. This assumption is load-bearing: if it fails, L_sem reflects probe capacity to extract linearly available nuisance structure (color, texture, position) rather than semantic sufficiency. To make the assumption testable rather than assumed, the probe W is fixed once on a disjoint reference codebook (see below) and never re-fit, and L_sem is reported as selectivity above a Hewitt-Liang-style control task (probe trained on randomized labels / random embeddings).

**How it works.**
```pseudocode
# Given: image x, patch grid P, semantic label map y (per patch)
# Hyperparameters: codebook size K=1024, dim d=64, lambda_rec=1.0,
#                  lambda_sem=0.5, lambda_commit=0.25, margin=0.1
#                  L semantic classes; reference set R of 4096 patches
#                  (disjoint from training patches)

Encoder E_phi: patch p -> continuous feature z in R^d
Codebook C = {e_1,...,e_K} in R^{d x K}  # single shared codebook
Decoder D_theta: token embedding e_k -> reconstructed patch p_hat

# --- Probe construction (once, before training) ---
# Build a disjoint reference codebook C_ref by running a randomly-initialized
# encoder E_phi0 (same architecture, no training) on reference set R and
# taking the K nearest-neighbor centroids (k-means, 20 iterations).
# Fit W in R^{d x L} by closed-form least squares on (e_k^ref, y) pairs
# from C_ref and R. W is then FROZEN for the entire training run.
# Control probe W_ctrl: same closed-form least squares but on (e_k^ref, y_shuffled)
# where y_shuffled is a random permutation of labels within R.

for each training batch of patches:
  z = E_phi(p)
  k = argmin_j || z - e_j ||_2          # nearest-neighbor quantization
  e_k = C[k]
  p_hat = D_theta(e_k)
  sem_logits = W^T e_k                  # FIXED linear probe on TOKEN EMBEDDING
  ctrl_logits = W_ctrl^T e_k            # control probe (random-label fit)

  L_rec  = || p - p_hat ||_2^2
  L_sem  = CE( sem_logits, y )          # cross-entropy on patch label
  L_ctrl = CE( ctrl_logits, y )         # control-task cross-entropy
  L_commit = || z - sg(e_k) ||_2^2 + beta * || sg(z) - e_k ||_2^2  # beta=0.25
  L_rate = || z - e_k ||_2^2            # rate proxy: quantization distortion

  # Coupled constraint: enforce margin between reconstruction and semantic gradients
  # by projecting codebook updates to satisfy both sufficiency conditions.
  # Concrete coupling operator (projected gradient on codebook):
  #   g_rec  = grad_C L_rec
  #   g_sem  = grad_C L_sem
  #   g_coupled = lambda_rec * g_rec + lambda_sem * g_sem
  #   # margin projection: if cosine(g_rec, g_sem) < -margin (conflicting),
  #   # project g_coupled onto the half-space where both directional
  #   # derivatives are non-negative:
  #   if dot(g_rec, g_sem) < -margin * ||g_rec|| * ||g_sem||:
  #       g_coupled = g_coupled - (dot(g_coupled, u) / ||u||^2) * u
  #       where u = g_rec/||g_rec|| - g_sem/||g_sem||
  #   C <- C - lr * g_coupled

  L_total = lambda_rec * L_rec + lambda_sem * L_sem + lambda_commit * L_commit + L_rate

  # Update E_phi, D_theta via gradient descent on L_total.
  # Update C via the projected-gradient coupling operator above.
  # W and W_ctrl are FROZEN: never updated, so tokens cannot cheat by
  # allocating representational capacity to the semantic pathway independently.

# After training, report L_sem and L_ctrl on held-out patches.
# Semantic selectivity = L_ctrl - L_sem (Hewitt-Liang control).
# Joint sufficiency gap is computed against a random-codebook baseline
# (C initialized randomly and frozen) to rule out probe-capacity artifacts.
```

**Why this design.** We chose a single shared codebook over factorized parallel codebooks (as in Factorized Vector Quantization) because factorized codebooks allow the model to allocate separate capacity to separate views, which structurally evades the joint sufficiency constraint; a single codebook forces every token to serve both purposes, accepting the cost that the effective bit budget is lower and reconstruction fidelity may be worse than a dedicated VQ tokenizer. We chose a fixed linear probe over a learned semantic head because a learned head can co-adapt with the encoder and effectively become a second encoder, whereas a linear probe on the token embedding is the minimal structural coupling that cannot manufacture separability the tokens do not contain; this accepts the cost that semantic accuracy may be lower than with a deep classifier. We chose to fix the probe once on a disjoint reference codebook and never re-fit it, rather than re-estimating it by closed-form least squares each epoch, because re-fitting would let the probe track whatever linear structure the current codebook happens to expose and would make the two-timescale scheme a form of co-adaptation, just slower; fixing W once on a disjoint reference codebook removes this co-adaptation path and makes the probe a genuine fixed measurement, accepting the cost that the probe may be suboptimal for the final codebook and that semantic accuracy may be lower than with a re-fit probe. We chose a coupled rate-distortion-identification objective with explicit rate proxy rather than a standard VQ objective because standard VQ has no term that penalizes tokens for being semantically uninformative, accepting the cost that the loss landscape is more complex and requires careful weighting of lambda_rec and lambda_sem. We chose a projected-gradient coupling operator on the codebook rather than a plain weighted-sum loss because a weighted sum can be dominated by one term and does not enforce that both sufficiency conditions improve together; the projection explicitly detects gradient conflict (cosine < -margin) and removes the conflicting component, accepting the cost of an extra projection step per batch. A domain expert would not describe this as a variant of Factorized Vector Quantization because that method uses multiple parallel codebooks to scale bit budget for understanding alone, whereas JSRDI uses one codebook and a fixed linear probe to enforce joint sufficiency across two sufficiency conditions that prior work treats as independent.

**Why it measures what we claim.** The computational quantity `L_sem = CE(W^T e_k, y)` measures the semantic identifiability of token `e_k` because the assumption that a fixed linear probe on the token embedding is a sufficient statistic for the patch's semantic label makes `L_sem` equivalent to the negative log-likelihood of the label given the token; this assumption fails when the semantic label is not linearly decodable from the token embedding (e.g., when the label depends on spatial context outside the patch), in which case `L_sem` reflects the linear separability of the token space rather than true semantic sufficiency. The control quantity `L_ctrl = CE(W_ctrl^T e_k, y)` measures the probe-capacity floor because `W_ctrl` is fit to randomized labels on the same reference codebook, so `L_ctrl - L_sem` (semantic selectivity) isolates intrinsic semantic content from probe-exploitable nuisance structure; this assumption fails when the control probe happens to capture the same nuisance structure as the real probe, in which case selectivity is underestimated. The computational quantity `L_rec = ||p - D(e_k)||^2` measures the pixel reconstruction sufficiency of token `e_k` because the assumption that the decoder `D` is expressive enough to invert the tokenization makes `L_rec` equivalent to the distortion of the token as a sufficient statistic for the patch; this assumption fails when the decoder is under-parameterized or the patch is out-of-distribution, in which case `L_rec` reflects decoder capacity rather than token sufficiency. The coupling of `L_rec` and `L_sem` under a single codebook measures joint sufficiency because the assumption that capacity allocated to one pathway reduces capacity available to the other makes the Pareto frontier of `(L_rec, L_sem)` equivalent to the set of jointly sufficient tokenizations; this assumption fails when the codebook is large enough that both losses can be driven to zero independently (i.e., when the bit budget is not binding), in which case the coupling is vacuous and the method reduces to standard VQ with a linear classifier. The rate proxy `L_rate = ||z - e_k||^2` measures the information bottleneck because the assumption that quantization distortion is monotone in bit rate makes `L_rate` equivalent to the rate term in rate-distortion theory; this assumption fails when the encoder distribution is non-stationary, in which case `L_rate` reflects encoder drift rather than rate.

## Contribution

['A single-codebook discrete visual tokenizer (JSRDI) trained under a coupled rate-distortion-identification objective with a frozen linear probe on token embeddings, which enforces joint sufficiency for pixel reconstruction and semantic discrimination without allowing capacity to be allocated independently to either pathway.', 'A two-timescale optimization procedure in which the linear probe is frozen within each epoch and re-estimated by closed-form least squares on held-out tokens, preventing probe-encoder collusion and providing a non-trivial joint sufficiency constraint.', 'An empirical design principle: joint sufficiency is only binding when the codebook bit budget is below the sum of the bit budgets required for independent pixel reconstruction and semantic discrimination, which we operationalize as a measurable Pareto frontier of (reconstruction distortion, semantic accuracy) across codebook sizes.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | COCO panoptic patches with per-patch labels | Provides patch-level semantic labels and reconstruction targets |
| Primary metric | Joint sufficiency gap: (L_rec, L_sem) Pareto distance | Directly measures coupling of reconstruction and semantic sufficiency |
| Secondary metric | Semantic selectivity: L_ctrl − L_sem (Hewitt-Liang control) | Controls for probe capacity to exploit nuisance structure |
| Baseline 1 | Factorized VQ tokenizer | Tests whether separate codebooks evade joint sufficiency |
| Baseline 2 | Standard VQ-VAE + linear probe | Tests whether frozen probe alone suffices without coupling |
| Baseline 3 | Jointly trained VQ + deep semantic head | Tests whether learned head manufactures separability |
| Baseline 4 | Random-codebook JSRDI (C frozen at random init) | Rules out probe-capacity artifacts in the gap |
| Ablation-of-ours | JSRDI with jointly trained probe | Tests whether fixing probe is necessary for joint sufficiency |
| Ablation-of-ours | JSRDI with plain weighted-sum loss (no projection) | Tests whether coupling operator drives the gain |
| Bit-budget sweep | K ∈ {64, 256, 1024}, d ∈ {16, 32, 64} | Identifies binding-budget regime where coupling is non-trivial |

### Why this setup validates the claim

This design forms a falsifiable test of the claim that a single-codebook tokenizer with a fixed linear probe enforces joint sufficiency across reconstruction and semantics. Factorized VQ tests the sub-claim that parallel codebooks structurally evade joint sufficiency: if it achieves a better Pareto frontier than JSRDI, the single-codebook coupling is not necessary. Standard VQ-VAE with a linear probe tests whether coupling the losses adds anything beyond post-hoc probing: if it matches JSRDI, the coupled objective is vacuous. The deep-head baseline tests whether a learned semantic head can manufacture linear separability that tokens do not intrinsically contain: if it matches JSRDI on the fixed-probe evaluation, the fixed probe is not the minimal coupling we claim. The random-codebook baseline tests whether the joint sufficiency gap can be attributed to probe capacity rather than token content: if random-codebook JSRDI shows the same gap, the metric is not measuring joint sufficiency. The semantic selectivity metric (L_ctrl − L_sem) directly controls for the Alain & Bengio (2016) linear-probe-as-evidence fallacy by subtracting the probe's capacity to fit randomized labels. The bit-budget sweep identifies the regime where the codebook bit budget is binding (K·d below the entropy needed for both tasks); outside this regime both losses can go to zero independently and the coupling is vacuous, so the sweep is necessary to locate where the claim is non-trivial. The ablation with a jointly trained probe tests whether fixing is essential to prevent collusion. The ablation with plain weighted-sum loss tests whether the projection operator, not just the loss terms, drives the gain. The joint sufficiency gap metric directly measures the trade-off between L_rec and L_sem across the codebook, which is exactly the quantity our central claim predicts will be tighter for JSRDI. A uniform improvement across all subsets would not be diagnostic; the predicted effect is a gap concentrated where the baseline's mechanism fails.

### Expected outcome and causal chain

**vs. Factorized VQ tokenizer** — On a case where a patch contains both fine texture and a clear semantic category, the baseline produces low reconstruction error but poor semantic linear decodability because its parallel codebooks allocate separate capacity to reconstruction and semantics, so the semantic codebook can ignore reconstruction fidelity. Our method instead forces the single codebook to serve both purposes because every token must reconstruct and be linearly decodable, so we expect a noticeable gap on patches with high texture-semantic conflict but parity on simple patches where both losses are easy. This effect should be largest in the binding-budget regime (K=64, d=16) and shrink as K and d grow.

**vs. Standard VQ-VAE + linear probe** — On a case where the VQ codebook collapses to a few codes that reconstruct well but are semantically ambiguous, the baseline produces low L_rec but high L_sem because the standard VQ objective has no term penalizing semantically uninformative tokens, and the post-hoc probe cannot recover information absent from the tokens. Our method instead penalizes semantic unidentifiability during training via L_sem, so we expect a tighter joint Pareto frontier, especially on patches whose semantics are not trivially predictable from low-level features. The semantic selectivity metric should be positive for JSRDI and near zero for the baseline on these patches.

**vs. Jointly trained VQ + deep semantic head** — On a case where the semantic label is only weakly correlated with patch appearance, the baseline produces high semantic accuracy on its own deep head but poor accuracy when a fixed linear probe is applied to the same tokens, because the deep head co-adapts with the encoder to create separability not present in the token embedding. Our method instead uses a fixed linear probe, so the tokens must be intrinsically linearly separable, and we expect the fixed-probe accuracy gap to be small for JSRDI but large for the baseline, with reconstruction parity.

**vs. Random-codebook JSRDI** — On all patches, the random-codebook baseline should show a joint sufficiency gap no better than Standard VQ-VAE + linear probe, because its tokens carry no learned structure. If JSRDI's gap is not better than this baseline, the measured coupling is attributable to probe capacity rather than token content.

**vs. JSRDI with jointly trained probe (ablation)** — On patches where the semantic label is weakly correlated with appearance, the jointly trained probe should achieve lower L_sem than the fixed probe because it co-adapts with the encoder, but its semantic selectivity (L_ctrl − L_sem) should be no better or worse, indicating the gain is probe capacity rather than token content. We expect the fixed probe to show higher selectivity.

**vs. JSRDI with plain weighted-sum loss (ablation)** — On patches with high texture-semantic conflict, the plain weighted-sum loss should be dominated by L_rec (since lambda_rec=1.0 > lambda_sem=0.5) and produce a gap closer to Standard VQ-VAE, while the projection operator should maintain both sufficiency conditions. We expect the projection ablation to lose the gap advantage specifically on conflict patches.

### What would falsify this idea

If JSRDI's joint sufficiency gap is no better than Factorized VQ's, or if its advantage over Standard VQ-VAE + linear probe is uniform across all patches rather than concentrated on patches where semantic and reconstruction demands conflict, or if its semantic selectivity (L_ctrl − L_sem) is no better than the random-codebook baseline, then the central claim that single-codebook coupling enforces joint sufficiency is wrong.

## References

1. Can Discrete Tokens Serve as Semantically Lossless Representations of Visual Signals?
2. Emerging Properties in Unified Multimodal Pretraining
3. ILLUME: Illuminating Your LLMs to See, Draw, and Self-Enhance
4. MetaMorph: Multimodal Understanding and Generation via Instruction Tuning
5. LMFusion: Adapting Pretrained Language Models for Multimodal Generation
