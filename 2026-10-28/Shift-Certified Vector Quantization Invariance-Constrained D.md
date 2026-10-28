# Shift-Certified Vector Quantization: Invariance-Constrained Discrete Visual Tokens with Equivalence-Class Codebooks

## Motivation

Across the factorized-tokenizer paper-tree (Can Discrete Tokens Serve as Semantically Lossless Representations of Visual Signals?) and the unified-generation trees (BAGEL, ILLUME, MetaMorph, LMFusion), discrete visual token fidelity is validated only through in-distribution multimodal understanding accuracy, so token assignment stability under safety, robustness, or long-tail shift is never a constraint on the quantization objective. Factorized VQ in particular shows quantization error does not correlate with understanding performance yet still treats benchmark accuracy as the proxy for semantic fidelity, meaning a tokenizer can silently remap a perturbed-but-semantically-identical image to a different code sequence and no existing method detects or penalizes this. The root cause is structural: the codebook assignment is trained purely by reconstruction/commitment objectives with no certificate tying assignment equality to semantic equivalence.

## Key Insight

Constraining codebook assignment to be constant on each learned semantic-equivalence class of perturbations (and provably non-constant across classes) converts the tokenizer's assignment map into an invariant of the semantic content rather than a function of the pixel distribution, so shift-robustness becomes a property enforced at quantization time instead of an accuracy outcome measured after the fact.

## Method

**What it is.** Shift-Certified VQ (SC-VQ) is a discrete visual tokenizer whose encoder–codebook assignment map is trained under an explicit invariance certificate: for a perturbation δ drawn from a learned null-perturbation generator, tokens must satisfy `assign(x) == assign(x+δ)` when δ is semantically null, and `assign(x) != assign(x+δ)` when δ crosses a semantic boundary. Inputs: visual feature `f` (and perturbed `f+δ`); outputs: a discrete token sequence plus a per-token certificate flag (null-invariant vs. boundary-crossing).

**Load-bearing assumption (stated explicitly).** The certificate is meaningful only if the learned generator `G_phi` and discriminator `D_psi` recover the true semantic-equivalence class: deltas judged null by `D_psi` are genuinely semantically null, and the adversarial argmax delta is a genuine semantic-boundary crossing. Because `G_phi` and `D_psi` co-adapt, this assumption can fail in two directions: `G_phi` can exploit `D_psi` blind spots to produce deltas that are actually semantic but misjudged (so `L_inv` reflects discriminator error), and the inner maximization can find a delta that fools `D_psi` without crossing a true boundary (so `L_sep` reflects adversarial over-sensitivity). We therefore anchor the equivalence relation to an external, non-learned semantic signal and report the certificate's agreement with that frozen oracle as a calibration term; without this anchor the certificate is indistinguishable from discriminator agreement.

**How it works.**

```pseudocode
# Encoder E, factorized symmetric codebooks {C_1..C_K} (as in Factorized VQ),
# perturbation generator G_phi (2-layer MLP, hidden=256, GeLU),
# semantic-boundary discriminator D_psi (frozen pretrained semantic encoder + 2-layer MLP head, hidden=256),
# frozen external semantic oracle O (pretrained CLIP ViT-B/32 image encoder, not co-trained)
for each batch of features f:
    z      = E(f)                       # continuous latent, dim d
    # --- null perturbations: learned generator proposes semantically-null deltas ---
    delta_null = G_phi(f, eps)          # eps ~ N(0, sigma^2), sigma=0.05
    # --- boundary perturbations: adversarial deltas that D_psi judges semantic ---
    delta_bnd  = argmax_delta  D_psi(f, f+delta)  s.t. ||delta||<=eps_bnd (eps_bnd=0.1)
                 # inner maximization: 3-step gradient ascent, step size 0.03
    # --- quantize each view with parallel symmetric codebooks ---
    q(f)       = concat_k  nearest(C_k, view_k(z))          # token sequence
    q(f+dnull) = concat_k  nearest(C_k, view_k(E(f+delta_null)))
    q(f+dbnd)  = concat_k  nearest(C_k, view_k(E(f+delta_bnd)))
    # --- certificate losses ---
    L_inv  = CE( q(f+dnull), stopgrad(q(f)) )               # pull null -> same token
    L_sep  = margin_hinge( sim(q(f),q(f+dbnd)), m=0.3 )     # push boundary -> different token
    L_vq   = ||z - sg[q(f)]||^2 + beta*||sg[z]-q(f)||^2     # beta=0.25, standard VQ
    L_gen  = -log D_psi(f, f+delta_null)                    # generator makes null look semantic-null
    L_disc = BCE(D_psi(f,f+delta_bnd), 1) + BCE(D_psi(f,f+delta_null), 0)
    # --- external calibration term (frozen oracle O, no gradient to G_phi/D_psi) ---
    L_cal  = | sim_O(f, f+delta_null) - 1 | + | sim_O(f, f+delta_bnd) - 0 |
             # sim_O = cosine similarity of O embeddings; target 1 for null, 0 for boundary
    L = L_vq + lambda_inv*L_inv + lambda_sep*L_sep + L_gen + L_disc + lambda_cal*L_cal
        # lambda_inv=1.0, lambda_sep=0.5, lambda_cal=0.2
```

The certificate is emitted at inference: a token is flagged *null-invariant* if `assign(f)==assign(f+δ_null)` for all sampled δ_null (5 samples), and *boundary-crossing* if some δ_bnd flips it. The generator and discriminator are trained jointly with the quantizer in an alternating schedule (1 generator/discriminator step per 2 quantizer steps) so the equivalence relation co-adapts with the codebook; `D_psi` is initialized from the frozen oracle `O` and its MLP head is updated, while `O` itself remains frozen and is used only in `L_cal`.

**Why this design.** We chose a *learned* null-perturbation generator over a fixed augmentation set (e.g., Gaussian noise, color jitter) because the semantic-null equivalence class is image-dependent and cannot be enumerated a priori, accepting the cost that the generator can collapse to trivial perturbations (mitigated by the discriminator's BCE term forcing null deltas to be judged non-semantic). We chose *adversarial* boundary perturbations over random large perturbations because random large deltas may be semantically null in ways the tokenizer should ignore, accepting the cost of an inner maximization step per batch (3-step gradient ascent). We chose to constrain the *assignment* (post-codebook) rather than the continuous latent because Factorized VQ's parallel codebooks admit many-to-one latent-to-token maps, so latent-space consistency does not imply token consistency; constraining assignment directly is the only surface where the certificate is meaningful, accepting the cost that gradients through `nearest` require straight-through estimation. We added the external calibration term `L_cal` against a frozen pretrained semantic encoder `O` so the certificate is not self-certifying: `L_cal` penalizes disagreement between the learned equivalence relation and the frozen oracle's judgment, providing an anchor to a non-learned semantic signal. An expert would not describe this as "Factorized VQ + a consistency loss" because the discriminator-defined equivalence relation makes the *sign* of the constraint data-dependent (same-class pulls, different-class pushes), whereas a standard consistency loss is sign-fixed and collapses to augmentation averaging.

**Why this measures what we claim.** `L_inv` measures *semantic-null invariance* because the assumption that `G_phi`-generated deltas judged non-semantic by `D_psi` are semantically null makes token equality under those deltas equivalent to invariance; this assumption fails when the generator exploits discriminator blind spots to produce deltas that are actually semantic but misjudged, in which case `L_inv` reflects discriminator error rather than true invariance. `L_sep` measures *boundary sensitivity* because the assumption that the adversarial `argmax` delta is the minimal semantic-crossing perturbation makes token inequality under it equivalent to provable token change at a semantic boundary; this assumption fails when the inner maximization finds a delta that fools `D_psi` without crossing a true boundary, in which case `L_sep` reflects adversarial over-sensitivity. The per-token certificate flag measures *assignment stability* because the assumption that sampling δ_null at inference covers the null equivalence class makes the flag equivalent to a stability guarantee; this fails under out-of-distribution shifts not represented by `G_phi`, in which case the flag reflects in-generator coverage rather than global invariance. `L_cal` measures *agreement with the frozen oracle* because the assumption that `O`'s cosine similarity is a faithful proxy for semantic equivalence makes the calibration term equivalent to an external anchor; this assumption fails when `O` itself is insensitive to the semantic distinctions of interest, in which case `L_cal` reflects oracle limitations rather than true semantic alignment.

## Contribution

(1) SC-VQ, a discrete visual tokenizer whose codebook assignment is trained under an explicit invariance certificate defined by a learned null-perturbation generator and an adversarial semantic-boundary discriminator, emitting per-token certificate flags at inference. (2) A design principle showing that constraining the post-codebook assignment map (rather than the continuous latent) is necessary for token-level invariance in factorized parallel-codebook tokenizers, with a joint alternating training schedule that co-adapts the equivalence relation with the codebook. (3) A perturbation-equivalence benchmark protocol that reports token-assignment flip rates under safety, robustness, and long-tail shifts alongside in-distribution understanding accuracy, decoupling semantic fidelity from accuracy as a proxy.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | COCO-derived null/boundary probe set (512 images, each with one benign edit and one semantic edit verified by human annotators) | Contains images with known benign and semantic edits |
| Primary metric | Token-flip rate on null vs. boundary subsets | Directly reads out invariance/separation tradeoff |
| Secondary metric | Certificate-oracle agreement rate: fraction of tokens where the emitted certificate flag matches the frozen CLIP ViT-B/32 oracle's judgment (cosine similarity > 0.9 for null, < 0.5 for boundary) | Tests whether the certificate is anchored to an external semantic signal |
| Baseline | Factorized VQ (no certificate) | Isolates the effect of the certificate losses |
| Baseline | Gaussian-augmentation consistency VQ | Tests learned vs. fixed null perturbations |
| Baseline | Continuous encoder representation | Tests whether discretization adds anything |
| Baseline | Continuous-latent adversarial invariance (same `L_inv`/`L_sep` applied to `z` before quantization) | Isolates whether the assignment-level constraint is necessary |
| Ablation-of-ours | SC-VQ without adversarial boundary term | Isolates the role of the discriminator-driven push |
| Ablation-of-ours | SC-VQ without external calibration term `L_cal` | Tests whether the frozen-oracle anchor changes certificate-oracle agreement |
| Ablation-of-ours | SC-VQ with certificate flag disabled at inference (training losses only) | Tests whether the flag provides value beyond the training losses |

### Why this setup validates the claim

The central claim is that a discriminator-defined equivalence relation makes the invariance/separation sign data-dependent, so tokens become stable under semantically null perturbations while remaining sensitive at true semantic boundaries. A probe set with matched null-edit and boundary-edit subsets lets us read both halves of that claim from a single metric: token-flip rate. Factorized VQ without the certificate tests whether the certificate losses are doing the work versus the codebook alone; the Gaussian-consistency baseline tests whether a learned generator beats a fixed augmentation class, which is the specific design choice we defended; the continuous representation tests whether the discrete assignment surface is where the guarantee lives; the continuous-latent adversarial baseline tests whether the assignment-level constraint is necessary versus applying the same losses to the latent. The ablation removing the adversarial boundary term tests whether separation gains persist without it, so if they do, our causal story about the discriminator is wrong. The ablation removing `L_cal` tests whether the external anchor changes certificate-oracle agreement, directly addressing the load-bearing assumption. The certificate-flag-disabled ablation tests whether the flag provides value beyond the training losses. Token-flip rate is the right primary metric because the certificate is literally a statement about assignment equality/inequality, so any other metric would be a proxy; certificate-oracle agreement is the right secondary metric because it directly measures whether the certificate is anchored to an external semantic signal rather than to the co-adapting discriminator.

### Expected outcome and causal chain

**vs. Factorized VQ (no certificate)** — On an image with a benign photometric jitter, plain Factorized VQ assigns a different token because its codebook boundaries are placed by reconstruction alone and nothing penalizes assignment drift under null deltas. Our method pulls the perturbed view to the same token via `L_inv`, so we expect a markedly lower null-subset flip rate, with the gap largest on images whose latents sit near codebook cell boundaries.

**vs. Gaussian-augmentation consistency VQ** — On an image where the semantic-null class is anisotropic (e.g., small object translations are null but color shifts are not), a fixed Gaussian augmentation applies the wrong perturbation shape, so consistency training either over-stabilizes semantic changes or under-stabilizes null ones. Our learned generator adapts δ per image, so we expect parity on isotropic cases but a clear advantage on the anisotropic subset, concentrated exactly where the fixed augmentation mismatches the true null class.

**vs. Continuous encoder representation** — On a case where two latents are close but map to different tokens under many-to-one parallel codebooks, a continuous representation shows small distances and appears stable, yet the discrete assignment flips. Our method constrains the assignment surface directly, so we expect the continuous baseline to look invariant on the null subset while its downstream discrete tokens still flip, revealing that latent-space stability does not transfer—our method should close that specific gap.

**vs. Continuous-latent adversarial invariance** — On a case where the continuous latent is stable under null perturbations but the nearest-codebook assignment still flips due to many-to-one parallel codebooks, the continuous-latent baseline will show low latent drift but high token-flip rate. Our method constrains the assignment directly, so we expect a lower token-flip rate on the null subset, isolating the necessity of the assignment-level constraint.

**vs. SC-VQ without external calibration term `L_cal`** — Without the frozen-oracle anchor, the certificate is self-certifying and may drift from the true semantic class as `G_phi` and `D_psi` co-adapt. We expect the no-`L_cal` variant to show lower certificate-oracle agreement, especially on boundary-crossing cases where the adversarial delta fools `D_psi` without crossing a true boundary.

**vs. SC-VQ with certificate flag disabled** — The training losses alone may improve token-flip rates, but the flag provides per-token stability information at inference. We expect the flag-enabled variant to show higher certificate-oracle agreement on the null subset, demonstrating that the flag adds value beyond the training losses.

### What would falsify this idea

If our null-subset flip-rate gain is uniform across anisotropic and isotropic cases rather than concentrated where fixed augmentations mismatch the null class, the learned-generator mechanism is not doing what we claim. If removing the adversarial boundary term leaves separation performance unchanged, the discriminator-defined equivalence relation is not the source of the effect. If certificate-oracle agreement is not higher with `L_cal` than without it, the external anchor is not doing the work we claim.

## References

1. Can Discrete Tokens Serve as Semantically Lossless Representations of Visual Signals?
2. Emerging Properties in Unified Multimodal Pretraining
3. ILLUME: Illuminating Your LLMs to See, Draw, and Self-Enhance
4. MetaMorph: Multimodal Understanding and Generation via Instruction Tuning
5. LMFusion: Adapting Pretrained Language Models for Multimodal Generation
