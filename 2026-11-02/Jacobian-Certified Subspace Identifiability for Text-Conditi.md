# Jacobian-Certified Subspace Identifiability for Text-Conditioned Generation (JCSI)

## Motivation

Across both the disentangled-CLIP and appearance/content trees, the factor decomposition of a conditioning embedding is assumed rather than identified: DeCLIP (Language-Guided Visual Perception Disentanglement) learns perceptual vs. semantic text-supervised subspaces whose purity is bounded by I&2T annotation accuracy, and DisQUE similarly assumes content and appearance separate cleanly under self-supervision. The structural root cause is that neither method imposes any testable condition on the generator's response to conditioning edits, so perceptual and semantic directions can remain linearly mixed inside the conditioning embedding and no user (or training loss) can detect the mixing. This meta-gap recurs wherever a downstream model is conditioned on an embedding whose factor split is asserted, making the split unverifiable and unbounded by supervision quality.

## Key Insight

If the generator's Jacobian with respect to the conditioning embedding is required to be block-diagonal across the two subspaces (cross-subspace invariance) and to have orthogonal row spaces on each block (intra-subspace orthogonality), then the split is identifiable up to permutation from the conditioning embedding alone, independent of annotation fidelity.

## Method

**What it is.** JCSI (Jacobian-Certified Subspace Identifiability) is a training-time constraint plus a test-time certificate for a conditional generator G(z, e) that maps noise z and a conditioning embedding e = [e_perc ; e_sem] to an image. It outputs (i) a generator whose Jacobian J_e = ∂G/∂e is approximately block-structured, and (ii) a scalar certificate κ(e) that reports, per sample, whether the perceptual/semantic split is algebraically identifiable.

**Load-bearing assumption (stated explicitly).** The certificate is valid only if the PCA-estimated subspaces P_perc and P_sem, computed from the edit-difference sets {e_p - e0} and {e_s - e0}, span the true, mutually orthogonal perceptual and semantic factor directions in the conditioning embedding. This assumption fails when the edit distribution is degenerate (e.g., all perceptual edits change only brightness) or when the two edit-difference sets share a dominant direction (as CLIP text embeddings are known to do), in which case the residual ||P_sem^T J_p||_F reflects edit-distribution bias or subspace overlap rather than generator entanglement. JCSI therefore does not take this assumption for granted: it measures the principal angles between P_perc and P_sem on every batch and gates the certificate on those angles being small (see Phase 2b and Phase 5).

**How it works.**

```pseudocode
Input: generator G_theta, paired text edits (t_perc, t_sem) per image from existing I&2T-style data
       (no new annotation; edits are re-used as intervention pairs)
Hyperparams: lambda_inv = 1.0, lambda_orth = 0.5, eps = 1e-3, K = 4 probes per edit,
             theta_max = 15 degrees (max allowed principal angle between P_perc and P_sem),
             kappa_min = 0.9 (certificate acceptance threshold)

Phase 1 - Intervention probes (per training step):
  e0 = E(x)                              # base conditioning embedding
  e_p = E(edit_perceptual(x))            # perceptual-only text edit
  e_s = E(edit_semantic(x))              # semantic-only text edit
  # finite-difference Jacobian columns, K random directions per subspace
  for k in 1..K:
     d_p^k ~ unit sphere in perceptual subspace; J_p[:,k] = (G(z, e0+eps*d_p^k) - G(z,e0))/eps
     d_s^k ~ unit sphere in semantic subspace;   J_s[:,k] = (G(z, e0+eps*d_s^k) - G(z,e0))/eps

Phase 2 - Subspace estimation with orthogonality diagnostic:
  # Estimate P_perc and P_sem jointly from the edit-difference covariance
  # via a canonical-correlation / orthogonal Procrustes decomposition:
  #   stack D = [e_p - e0 ; e_s - e0] across the batch (2B x d)
  #   compute the two leading subspaces by CCA between the perceptual and semantic blocks
  #   P_perc, P_sem = orthonormal bases of the resulting canonical directions
  # Report the principal angles theta_1 >= theta_2 >= ... between P_perc and P_sem.
  # If max(theta_i) > theta_max, mark the batch as edit-degenerate and SKIP the
  # L_inv / L_orth update for that batch (do not backprop through a biased residual).

Phase 2b - Cross-subspace invariance loss:
  # perceptual edits must not move semantic output directions and vice versa
  L_inv = || P_sem^T J_p ||_F^2 + || P_perc^T J_s ||_F^2
  where P_perc, P_sem are the CCA-estimated projection matrices from Phase 2.

Phase 3 - Intra-subspace orthogonality loss:
  L_orth = || J_p^T J_p - I ||_F^2 + || J_s^T J_s - I ||_F^2
  # forces each block to be an isometry, so the two blocks are linearly independent

Phase 4 - Standard generation loss (flow-matching / diffusion) L_gen; total
  L = L_gen + lambda_inv * L_inv + lambda_orth * L_orth

Phase 5 - Certificate at test time:
  kappa(e) = 1 - ( ||P_sem^T J_p||_F + ||P_perc^T J_s||_F ) / ( ||J_p||_F + ||J_s||_F )
  # Gate on the edit-geometry diagnostic: if max principal angle between
  # P_perc and P_sem exceeds theta_max, the certificate is undefined for this
  # sample; abstain and report 'edit-degenerate' rather than a kappa value.
  accept split if kappa(e) > kappa_min AND max_angle(P_perc, P_sem) <= theta_max;
  else flag as entangled (or edit-degenerate) and abstain from fine control.
```

**Why this design.** We chose a *finite-difference Jacobian probe* over a learned disentanglement head because the head would only assert statistical independence (the anti-pattern), whereas the Jacobian probe directly measures the generator's response to conditioning edits, accepting the cost of K extra forward passes per step. We chose *projection-based cross-subspace invariance* over a reconstruction-based swap loss because projection gives a closed-form algebraic residual ||P_sem^T J_p||_F that is zero iff the subspaces are orthogonal in the output tangent space, accepting the cost that subspace estimation via CCA must be re-run each batch. We chose *intra-subspace orthogonality* (isometry) over a simple L2 penalty because isometry makes the two blocks linearly independent by construction (rank additivity), accepting the cost that the generator's output scale is constrained and we must tune lambda_orth. We chose a *per-sample certificate* over a single global metric because identifiability is a per-embedding property, accepting the cost that at test time we run the probe and may abstain on some inputs. We added the *principal-angle gate* because the projection residual is only meaningful when the two estimated subspaces are themselves nearly orthogonal; without the gate, a shared dominant direction in the edit distribution would inflate or deflate κ by construction. This is not a variant of DeCLIP or DisQUE: those methods train a disentanglement loss on *features*, while JCSI constrains the *generator Jacobian* and emits a testable algebraic certificate, which neither prior method produces.

**Why it measures what we claim.** The quantity ||P_sem^T J_p||_F measures *cross-subspace invariance* because it equals the norm of the generator's response to perceptual edits projected onto semantic output directions, under the assumption that the CCA-estimated subspaces span the true factor directions and are themselves nearly orthogonal (max principal angle <= theta_max); this assumption fails when the edit distribution is degenerate or the two edit-difference sets share a dominant direction, in which case the residual reflects edit-distribution bias or subspace overlap rather than true entanglement, and the Phase 2b gate causes the batch to be skipped and the Phase 5 certificate to abstain. The quantity ||J_p^T J_p - I||_F measures *intra-subspace orthogonality* because it is zero iff the perceptual block is an isometry, under the assumption that the generator is locally linear in e (small eps); this assumption fails for highly nonlinear generators, in which case the isometry residual reflects curvature rather than factor mixing. The certificate κ(e) measures *identifiability up to permutation* because it is a normalized sum of the two residuals, under the assumption that the two subspace projections are mutually orthogonal and the principal-angle gate has passed; this assumption fails when the edit-induced directions overlap, in which case κ is not reported and the sample is flagged as edit-degenerate rather than entangled.

## Contribution

(1) A training-time Jacobian block-structure constraint (cross-subspace invariance + intra-subspace isometry) that makes the perceptual/semantic split of a conditional generator identifiable from the conditioning embedding alone, without new annotations. (2) A per-sample identifiability certificate κ(e) computed from finite-difference Jacobian probes, turning subspace separation from an assumed property into a testable algebraic condition. (3) A reusable protocol that repurposes existing paired text descriptions as intervention probes for the invariance test, demonstrating that disentanglement quality can be certified rather than merely trained.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | I&2T perceptual/semantic edit pairs | Provides matched intervention pairs isolating each factor |
| Primary metric | Cross-subspace leakage ratio (κ on held-out edits) | Directly measures the algebraic quantity we certify |
| Secondary metric | Principal angle between P_perc and P_sem (per batch) | Tests the load-bearing subspace-recovery assumption |
| Baseline | DeCLIP | Feature-space disentanglement; no Jacobian constraint |
| Baseline | DisQUE | Appearance/content disentanglement without per-sample certificate |
| Baseline | Standard conditional diffusion (no invariance loss) | Untrained Jacobian baseline for entanglement |
| Ablation-of-ours | JCSI without L_orth (invariance only) | Tests whether isometry block is necessary for identifiability |
| Ablation-of-ours | JCSI with PCA subspaces (no CCA, no angle gate) | Tests whether the orthogonality diagnostic is load-bearing |

### Why this setup validates the claim

The central claim is that constraining the generator Jacobian to be block-structured yields a per-sample algebraic certificate that tracks true conditional controllability. I&2T is the right testbed because it supplies paired perceptual-only and semantic-only edits, so the Jacobian probe and the ground-truth intervention share the same factor directions — meaning κ can be checked against actual edit-response behavior rather than a proxy. The secondary metric — the principal angle between P_perc and P_sem — is included because the certificate's validity is conditional on the two estimated subspaces being nearly orthogonal; if the edit distribution is degenerate, the angle will be large and the certificate will abstain, and we can verify that the abstention rate tracks the angle rather than the generator's true entanglement. DeCLIP tests the sub-claim that feature-space disentanglement does not transfer to generator controllability: if DeCLIP's features are decoupled but its generator still leaks perceptual edits into semantic directions, our Jacobian residual should separate from its feature metric. DisQUE tests the sub-claim that appearance/content separation without a per-sample certificate cannot flag entangled inputs, so its errors should concentrate on the same samples our κ rejects. The no-invariance baseline isolates the contribution of the loss itself, the L_orth ablation tests the causal role of isometry in making the two blocks linearly independent, and the PCA-vs-CCA ablation tests whether the orthogonality diagnostic is load-bearing — if removing it inflates κ on degenerate-edit batches, the assumption story is confirmed. The cross-subspace leakage ratio is the correct metric because it is exactly the normalized residual the certificate reports, so a gap in κ is a gap in the claimed quantity, not a downstream correlate.

### Expected outcome and causal chain

**vs. DeCLIP** — On a case where a perceptual edit (e.g., brightness) is applied to a textured semantic object, DeCLIP produces decoupled text features yet its generator still shifts object-category appearance because feature independence does not constrain the output Jacobian. Our method instead penalizes ||P_sem^T J_p||_F directly, so perceptual edits are projected away from semantic output directions. We expect a noticeable κ gap on high-texture subsets but near-parity on flat-background subsets, where both methods are already near-orthogonal. On batches where the principal angle between P_perc and P_sem exceeds theta_max = 15 degrees, our method abstains rather than reporting an inflated κ, so the gap is measured only on batches that pass the edit-geometry gate.

**vs. DisQUE** — On a case where appearance and content are statistically separable across the dataset but entangled for a specific embedding, DisQUE produces good average tone-mapping quality while silently failing on that sample because it has no per-sample diagnostic. Our method instead emits κ(e) and abstains when the residual is high, so entangled samples are flagged rather than mis-generated. We expect DisQUE to match us on the aggregate metric but diverge sharply on the subset our certificate rejects — a concentrated, not uniform, gap. We also expect DisQUE's errors to concentrate on samples where our principal-angle diagnostic passes (so the entanglement is real, not edit-degenerate), confirming that the certificate is measuring generator structure rather than edit bias.

**vs. Standard conditional diffusion** — On a case where a semantic edit (e.g., "make it a cat") is applied, the unconstrained generator produces a correlated shift in brightness and style because nothing forces the semantic Jacobian block to be orthogonal to perceptual directions. Our method instead drives ||P_perc^T J_s||_F toward zero via L_inv, so semantic edits leave perceptual attributes intact. We expect a large leakage-ratio gap on semantic-edit samples and a smaller one on perceptual-edit samples, since the perceptual block is lower-dimensional and easier to isolate. The gap should be largest on batches that pass the principal-angle gate, and should shrink toward zero on edit-degenerate batches where both methods are forced to abstain.

### What would falsify this idea

If the κ gap over DeCLIP and DisQUE is uniform across subsets rather than concentrated on the samples our certificate rejects, the claim that κ tracks per-sample identifiability is wrong. Likewise, if removing L_orth leaves κ equally discriminative, then isometry is not the causal mechanism and the algebraic story collapses to a generic invariance penalty. Finally, if the PCA-vs-CCA ablation shows no difference in κ on batches with large principal angles, then the orthogonality diagnostic is not load-bearing and the subspace-recovery assumption is not the binding constraint.

## References

1. Language-Guided Visual Perception Disentanglement for Image Quality Assessment and Conditional Image Generation
2. Joint Quality Assessment and Example-Guided Image Processing by Disentangling Picture Appearance from Content
3. AesExpert: Towards Multi-modality Foundation Model for Image Aesthetics Perception
4. OCTFlow: Unified conditional generation for balanced classification and annotated segmentation in retinal OCT.
