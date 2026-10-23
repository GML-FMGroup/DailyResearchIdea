# Compositional Closure Certificates for Factor-Span Video Editing

## Motivation

Factor-span replacement in FAPL (Making MLLM Latents Matter) and its branches assumes that editing one factor direction in a frozen MLLM latent space leaves the complementary factor subspace invariant, but this is never guaranteed for factor combinations unseen during paired training. The structural problem across these paper-trees is that preservation is enforced only empirically through paired supervision, with no compositional closure condition that holds out-of-distribution, so the generator can silently entangle factors when the frozen encoder's latents are not perfectly disentangled.

## Key Insight

A learned scalar certificate that is zero exactly when the edit lies in the span of the target factor direction can be trained as a conditional independence test on frozen encoder latents, turning preservation into a checkable algebraic closure condition that generalizes to unseen factor compositions rather than an empirical hope.

## Method

**What it is** — We introduce **CLOSURE** (Compositional Latent-Subspace Invariance Certificate), a training objective and inference check for factor-span edits in FAPL-style video generators. Given a frozen MLLM latent pair (z, z') and a factor direction f, CLOSURE trains a scalar certificate network C_psi(z, z', f) that is zero iff the edit delta = z' - z lies in span(f), i.e., the complementary subspace is invariant. The load-bearing assumption is stated explicitly: **C_psi is trained only on synthetic deltas drawn from span(f) and its orthogonal complement, and we assume this synthetic distribution covers the real generator delta distribution and that factor directions are approximately orthogonal; if real deltas lie off the synthetic support or factors are non-orthogonal, C_psi reflects its training error rather than true subspace membership, and projection may corrupt the intended factor.** To make this assumption checkable, we add a geometric calibration diagnostic (step 8) that measures the distribution shift between synthetic and real deltas and the pairwise cosine similarity of estimated factor directions, and we report the certificate's calibration curve (score vs. true off-span mass) on held-out deltas.

**How it works** —
```pseudocode
# Inputs: frozen MLLM encoder E, Video DiT G, factor directions {f_k}, paired latents (z, z')
# Hyperparameters: lambda_cert=0.1, tau=0.05, K_neg=4, rho=0.9, freeze_steps=0.1*T, calib_size=512

1. Factor directions: extract f_k as top principal directions of paired latent differences for factor k (PCA on 10k paired deltas per factor).
2. Certificate network C_psi: 2-layer MLP, hidden=256, GeLU activation, input dim = 2*d_z + d_f, output two heads -> (s, u):
     s = scalar invariance score (linear output), u = unit vector in orthogonal complement of f (L2-normalized linear output).
3. Positive pairs (edit in span(f)): delta_+ = alpha * f (alpha ~ N(0,1)); z'_+ = z + delta_+.
4. Negative pairs (edit leaks into complement): delta_- = alpha * f + beta * u, u ~ Uniform(unit sphere in f^perp), beta ~ N(0, tau^2); z'_- = z + delta_-.
5. Certificate loss: L_cert = E[ C_psi(z, z'_+, f)^2 + max(0, 1 - C_psi(z, z'_- , f))^2 ].
6. Generator loss: L_gen = L_diffusion(G(E(text), z')) + lambda_cert * C_psi(z, G_latent, f)^2.
   Freeze C_psi for the first freeze_steps=10% of training steps; then unfreeze and continue joint training.
7. Inference: accept edit iff C_psi(z, z', f) < rho; else project delta onto span(f) and re-run G.
8. Geometric calibration diagnostic (run every 1000 steps on a held-out set of 512 real (z, z') pairs from unseen factor compositions):
     a. Estimate local linear/Jacobian model of the factor manifold around z using K=20 nearest neighbors in paired data; compute the local tangent basis T_f(z) and its orthogonal complement.
     b. Compute geometric ground-truth off-span mass: m_true = || (I - P_{T_f(z)}) (z' - z) ||_2.
     c. Compute certificate score s = C_psi(z, z', f).
     d. Report calibration curve: bin m_true into 10 quantile bins and plot mean s vs. mean m_true; compute Spearman correlation and expected calibration error (ECE) between s and m_true.
     e. Distribution-shift diagnostic: compute maximum mean discrepancy (MMD) between synthetic delta distribution (steps 3-4) and real delta distribution on held-out compositions; report pairwise cosine similarity matrix of estimated factor directions {f_k}.
```

**Why this design** — We chose a learned scalar certificate over a fixed linear projection because the frozen MLLM latents are only approximately factorized, so a linear test would be too brittle; the certificate can absorb residual nonlinearity while still being trained to vanish exactly on the span, accepting the cost that the certificate is only as reliable as its training distribution. We chose to train the certificate jointly with the generator (step 6) rather than post-hoc because it forces the generator to produce edits that the certificate can verify, accepting the risk of co-adaptation which we mitigate by freezing the certificate for the first 10% of steps. We chose the negative sampling scheme with orthogonal complement vectors (step 4) over random perturbations because it directly targets the failure mode of factor leakage, accepting the cost that it requires estimating the complement basis. This design is not a variant of standard disentanglement heads (e.g., the independence penalty in FAPL's factor-span replacement) because the certificate is a function of the edit and the latent pair that is trained to be zero iff the edit is in the span, not a statistical independence penalty on the latents themselves; a domain expert would not describe it as a repackaged independence head because the certificate operates on the edit delta, not on the latent distribution.

**Why it measures what we claim** — The certificate value C_psi(z, z', f) measures the degree to which the edit delta = z' - z lies outside span(f) because we assume that the certificate network can learn the indicator function of the subspace span(f) from positive and negative samples; this assumption fails when the frozen encoder's latents have nonlinear coupling between factors, in which case the certificate reflects the network's training error rather than true subspace membership. The projection step at inference measures preservation of the complementary subspace because we assume that projecting delta onto span(f) removes all components outside the span; this assumption fails when the factor directions are not orthogonal, in which case projection may alter the intended factor. The generator loss term lambda_cert * C_psi^2 measures the generator's adherence to the closure condition because we assume that minimizing the certificate on generated latents enforces invariance; this assumption fails when the generator exploits certificate blind spots, in which case the loss reflects certificate overfitting rather than true invariance. The geometric calibration diagnostic (step 8) measures the certificate's calibration to true off-span mass because we assume that the local linear/Jacobian model around z provides an accurate ground-truth tangent; this assumption fails when the local neighborhood is too sparse or the manifold is highly curved, in which case the diagnostic reflects estimation error rather than certificate miscalibration.

## Contribution

['A learned scalar certificate network that is trained to be zero exactly when an edit lies in the span of a target factor direction, providing a checkable compositional closure condition for factor-span edits.', 'A joint training objective that couples the certificate with the video generator, forcing generated edits to satisfy the certificate and enabling inference-time rejection or projection of non-compliant edits.', 'An empirical demonstration that the certificate generalizes to unseen factor combinations and reduces factor leakage compared to standard factor-span replacement.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Cinematic multi-factor video benchmark with sparse guidance | Tests factor-span control where leakage is likely |
| Primary metric | Factor-leakage rate on held-out complementary-subspace probe | Directly measures the invariance claim |
| Secondary metric | Certificate calibration: Spearman correlation and ECE between C_psi score and geometric off-span mass (step 8d) | Tests whether certificate measures subspace membership |
| Diagnostic | MMD between synthetic and real deltas; pairwise cosine similarity of factor directions (step 8e) | Checks the load-bearing assumption of synthetic-to-real coverage |
| Baseline | FAPL factor-span replacement | Closest disentanglement mechanism without certificate |
| Baseline | Native MLLM conditioning pathway | Tests whether certificate adds over plain conditioning |
| Baseline | Linear projection test on delta | Tests whether learned certificate beats fixed subspace test |
| Ablation-of-ours | CLOSURE without joint generator loss (post-hoc certificate) | Isolates co-adaptation vs verification benefit |
| Ablation-of-ours | CLOSURE with certificate trained only on synthetic deltas (no calibration diagnostic) | Tests whether calibration diagnostic is necessary for OOD generalization |

### Why this setup validates the claim

The central claim is that a learned scalar certificate trained on edit deltas can both detect and suppress factor leakage in FAPL-style generators. The primary metric — leakage rate on a held-out complementary-subspace probe — is the right detector because it measures exactly the quantity CLOSURE asserts is controlled: mass of the edit delta outside span(f). The secondary calibration metric (Spearman correlation and ECE between C_psi score and geometric off-span mass from step 8d) directly tests the load-bearing assumption that the certificate generalizes from synthetic to real deltas: if the certificate is well-calibrated, its score should monotonically track true off-span mass on held-out compositions. The distribution-shift diagnostic (MMD and cosine similarity) quantifies whether the synthetic training distribution covers real deltas and whether factor directions are orthogonal, providing a concrete check on the assumption's premises. FAPL factor-span replacement tests whether a statistical independence penalty suffices; if CLOSURE beats it on leakage while matching on intended-factor fidelity, the claim that edit-level certification matters is supported. The native conditioning baseline tests whether the gain is merely from MLLM conditioning rather than the certificate. The linear-projection baseline tests the design choice of a learned certificate over a fixed test; if the linear test is equally good, our motivation collapses. The post-hoc ablation isolates whether joint training with the generator (step 6) is necessary or whether verification alone suffices, which is the key sub-claim about co-adaptation being manageable. The synthetic-only ablation tests whether the calibration diagnostic is needed to detect OOD failure. Together these form a falsifiable contrast: CLOSURE should win specifically where complementary-subspace leakage occurs, not uniformly, and its certificate should be calibrated to geometric ground truth.

### Expected outcome and causal chain

**vs. FAPL factor-span replacement** — On a case where the prompt edits character identity while holding camera fixed, the baseline produces camera drift because its independence penalty acts on latent statistics rather than on the edit delta, so residual coupling slips through. Our method instead certifies the delta against span(f_character) and rejects or reprojects leakage, because C_psi is trained to vanish only on the span. We expect a noticeable leakage-rate gap on coupled-factor subsets but near parity on single-factor edits. Additionally, we expect the calibration diagnostic to show Spearman correlation > 0.7 between C_psi score and geometric off-span mass on held-out compositions, supporting the certificate's validity.

**vs. Native MLLM conditioning pathway** — On a case with sparse guidance where only the action factor is specified, the baseline produces uncontrolled scene changes because nothing constrains the unmentioned factors in latent space. Our method instead applies the certificate to the generated delta and projects out complementary components, because the closure condition is enforced at the edit level regardless of guidance sparsity. We expect the gap to widen as guidance gets sparser, with parity when all factors are fully specified. The MMD diagnostic should show low distribution shift (MMD < 0.1) on sparse-guidance subsets, indicating synthetic deltas cover real ones.

**vs. Linear projection test on delta** — On a case where the frozen encoder's latents are nonlinearly coupled between scene and camera, the baseline produces false accepts because a fixed linear projection cannot capture curved factor manifolds. Our method instead absorbs residual nonlinearity in C_psi, because it is trained on positive and negative delta samples rather than assuming orthogonality. We expect lower leakage on the nonlinear-coupling subset specifically, with comparable results where coupling is approximately linear. The calibration curve should show that C_psi tracks geometric off-span mass even when factor directions have cosine similarity up to 0.3, whereas the linear test's false accepts increase with cosine similarity.

**vs. CLOSURE without joint generator loss (post-hoc certificate)** — On a case where the generator produces edits that are slightly off-span, the post-hoc certificate may accept them because it was not trained to push the generator toward verifiable edits. Our method instead jointly trains the generator to minimize C_psi on its outputs, so the generator learns to produce edits that the certificate can verify. We expect the joint training to yield lower leakage on coupled-factor subsets, with the gap increasing as the generator's initial edit distribution deviates from span(f). The freeze-for-10% mitigation should prevent co-adaptation from degrading calibration, as measured by stable Spearman correlation across training.

**vs. CLOSURE with certificate trained only on synthetic deltas (no calibration diagnostic)** — On held-out factor compositions, the synthetic-only certificate may overfit to its training distribution and assign low scores to real off-span edits. Our method includes the calibration diagnostic (step 8) to detect this, and we expect the synthetic-only ablation to show higher ECE and lower Spearman correlation on held-out compositions, confirming the diagnostic's necessity.

### What would falsify this idea

If CLOSURE's leakage reduction is uniform across all subsets rather than concentrated on the coupled-factor and sparse-guidance subsets where the predicted failure mode lives, the certificate is not measuring subspace membership but acting as a generic regularizer. Likewise, if the post-hoc certificate ablation matches joint training everywhere, the co-adaptation argument for step 6 is wrong. Additionally, if the calibration diagnostic shows low Spearman correlation (< 0.5) or high ECE (> 0.2) between C_psi score and geometric off-span mass on held-out compositions, the load-bearing assumption that the certificate generalizes from synthetic to real deltas is falsified.

## References

1. Making MLLM Latents Matter: Factor-Aware Predictive Learning for Cinematic Video Generation
2. Revisiting Feature Prediction for Learning Visual Representations from Video
3. HunyuanVideo: A Systematic Framework For Large Video Generative Models
4. Masked Siamese Networks for Label-Efficient Learning
5. Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language
