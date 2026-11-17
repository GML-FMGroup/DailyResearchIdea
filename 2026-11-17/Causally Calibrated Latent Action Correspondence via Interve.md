# Causally Calibrated Latent Action Correspondence via Intervention Anchors and Learned Appearance-Shift Equivariance

## Motivation

Latent action models such as SALA and LAPA leave the human-to-robot latent mapping underdetermined whenever appearance and language are confounded: SALA's soft neighborhoods are built from language-description similarity of unpaired clips, which is a purely correlational signal that cannot distinguish a latent that clusters by visual appearance from one that clusters by underlying action, and LAPA's reconstruction-only VQ objective inherits the same appearance-clustering bias. Structurally, this is because neither method ever observes an intervention that holds appearance fixed while changing the action (or vice versa), so the appearance-shift direction in latent space is never identified and the correspondence is only defined up to an arbitrary appearance-dependent reparameterization. This is a meta-level identifiability gap that recurs across every branch of the human-video latent-action tree, not a local tuning issue.

## Key Insight

Latent action semantics become identifiable under appearance shift exactly when the latent is required to be equivariant to a learned appearance-shift operator and simultaneously pinned by a small set of intervention-anchored human-robot pairs, because the equivariance constraint removes the appearance-dependent reparameterization freedom that language-similarity alone leaves unconstrained.

## Method

**What it is.** CALIC (Causally CAlibrated Latent action Correspondence) takes as input unpaired human and robot video clips with language descriptions, a small set of paired or causally-intervened human-robot action anchors (N_anchor = 100 by default, swept over {50, 100, 200}), and outputs a latent-action encoder plus a human-to-robot latent transport map such that latent actions are equivariant to appearance shift and anchored to the robot action manifold.

**Load-bearing assumption (stated explicitly).** CALIC assumes that the appearance-shift operator A_phi transports human clips along a pure appearance direction that is orthogonal to action-relevant content, so that enforcing z_{A_phi(h)} = T_phi(z_h) identifies action semantics rather than merely absorbing whatever content A_phi happens to alter. This assumption is load-bearing: if A_phi alters action-relevant content (e.g., changes object pose during style transfer), the equivariance loss reflects action distortion rather than appearance leakage. To make this assumption verifiable rather than asserted, A_phi is constrained to be action-preserving by construction (Phase 1 below) and its action-invariance is externally validated by an optical-flow consistency check (Phase 1b).

**How it works.**

```pseudocode
Phase 1: Appearance-shift operator A_phi (frozen, action-preserving by construction)
  Input: unpaired human clip h, robot clip r (no correspondence assumed)
  A_phi is a frozen photometric/texture-only perturbation stack applied to h:
    - random color jitter (brightness, contrast, saturation, hue) with strength in [0.2, 0.8]
    - random Gaussian blur (sigma in [0.5, 2.0]) and additive Gaussian noise (std in [0.01, 0.05])
    - random low-frequency texture overlay (Perlin noise, amplitude in [0.05, 0.2])
    - NO geometric warping, NO pose change, NO object replacement
  A_phi(h) -> h'  # style-transport human->robot appearance, action content preserved
  # A_phi is NOT trained jointly with E_theta; it is a fixed stochastic operator.
  # A_psi is the identity (no inverse needed; cycle loss is not used).
  # Discriminator D is removed; lambda_adv = 0.

Phase 1b: Action-preservation validation of A_phi (external check, run once before training)
  For a held-out set of 200 human clips, compute optical flow F(h) and F(A_phi(h))
  using RAFT (frozen). Report mean endpoint error ||F(h) - F(A_phi(h))||_2.
  Accept A_phi only if mean EPE < 0.5 pixels; otherwise reduce perturbation strength.
  This validates that A_phi does not alter action-relevant motion.

Phase 2: Latent action encoder E_theta (VQ-VAE, codebook K=64, code dim=32)
  z_h = E_theta(h), z_r = E_theta(r)  # discrete latent action sequences
  Reconstruction: decode(z_h) ~ h, decode(z_r) ~ r
  VQ commitment loss weight beta=0.25; codebook reset if perplexity < 5.

Phase 3: Equivariance objective (regularizer, weight lambda_eq=0.1)
  z_{A_phi(h)} = E_theta(A_phi(h))
  Equivariance loss: || z_{A_phi(h)} - T_phi(z_h) ||^2
  where T_phi is a learned latent transport head (2-layer MLP, hidden=256, GeLU)
  conditioned on phi (the perturbation parameters, 8-dim vector).
  # forces latent action to be invariant to the appearance direction A_phi moves in
  # lambda_eq is validated by the anchor ablation (Phase 4); if anchors alone suffice,
  # lambda_eq is reduced to 0 and the equivariance term is demoted to a regularizer.

Phase 4: Intervention anchors (N_anchor = 100 by default, swept over {50, 100, 200})
  For each anchor (h_i, r_i) paired by construction or by causal intervention
  (same action, appearance shifted):
  Anchor loss: || T_phi(z_{h_i}) - z_{r_i} ||^2
  # pins the equivariant latent to the actual robot action manifold
  # T_phi is the SOLE source of the transport map; equivariance (Phase 3) only regularizes it.

Phase 5: Language-similarity soft neighborhoods (SALA inherited)
  Soft neighborhood loss over unpaired clips using language-description similarity
  (cosine similarity of frozen CLIP text embeddings, temperature tau=0.1)
  weight lambda_lang=0.5 (ablated to 0 in the lambda_lang=0 condition)

Total loss = recon + lambda_eq * equivariance + lambda_anchor * anchor + lambda_lang * lang
  with lambda_anchor=1.0, lambda_eq=0.1, lambda_lang=0.5.
```

**Why this design.** We chose a frozen, action-preserving appearance-shift operator over a jointly-trained adversarial translator because the appearance gap between human and robot is embodiment-specific and not captured by generic image translation, and because a jointly-trained A_phi can collapse onto action-bearing content to trivially satisfy the equivariance loss; restricting A_phi to photometric/texture-only perturbations (color jitter, blur, noise, low-frequency texture) with an optical-flow consistency check (Phase 1b) makes the action-invariance of A_phi externally validated rather than assumed. We chose discrete VQ latent actions over continuous latents because the equivariance constraint is only meaningful on a quantized codebook where transport is a permutation-like map, accepting the cost of codebook collapse risk (mitigated by codebook reset at perplexity < 5). We chose a small set of intervention anchors (N_anchor = 100, swept over {50, 100, 200}) over large-scale paired data because paired human-robot data is expensive and the identifiability argument only needs enough anchors to pin the equivariant subspace, accepting the cost that anchor coverage must span the action manifold; the anchor set is the sole source of the transport map T_phi, and equivariance is demoted to a regularizer whose weight is validated by the anchor ablation. A domain expert would not describe CALIC as a variant of SALA because SALA's supervision is a correlational language-similarity neighborhood, whereas CALIC's is an equivariance constraint plus interventional anchors that SALA does not contain; nor is it a variant of LAPA because LAPA has no appearance-shift operator and no anchor set.

**Why it measures what we claim.** The equivariance loss ||z_{A_phi(h)} - T_phi(z_h)||^2 measures appearance-invariance of the latent action because the assumption that A_phi moves only along the appearance direction and not the action direction (validated by the optical-flow consistency check in Phase 1b) makes the two latents equal iff the encoder ignores appearance; this assumption fails when A_phi inadvertently alters action-relevant content (e.g., changing object pose during style transfer), in which case the loss reflects action distortion rather than appearance leakage. The anchor loss ||T_phi(z_h) - z_r||^2 measures correspondence to the robot action manifold because the assumption that each anchor pair shares an action and differs only in appearance makes the transported latent equal to the robot latent iff the transport map is action-faithful; this assumption fails when anchor pairs are mislabeled or interventions do not fully control appearance, in which case the loss reflects anchor noise rather than correspondence. The language-similarity soft neighborhood measures semantic action similarity because the assumption that language descriptions are informative and action-discriminative makes similar descriptions imply similar actions; this assumption fails when descriptions are generic or confounded with appearance, in which case the neighborhood reflects description clustering rather than action clustering.

## Contribution

(1) A latent-action training framework that couples a learned appearance-shift operator with an equivariance constraint and a small intervention-anchor set, giving an identifiable human-to-robot latent correspondence without requiring large paired data. (2) The design principle that appearance-equivariance plus interventional anchoring is sufficient to remove the appearance-dependent reparameterization freedom that language-similarity supervision alone leaves. (3) A procedure for constructing intervention anchors from unpaired clips via cycle-consistent appearance transport, reducing anchor collection cost.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Unpaired human + robot manipulation clips with language; N_anchor = 100 intervention anchors (swept over {50, 100, 200}) | Matches CALIC's stated input regime and anchor budget |
| Primary metric | Success rate on held-out real-world manipulation tasks | Directly reflects action-faithfulness of transferred latents |
| Secondary metric | Appearance-leakage probe: linear classifier accuracy from z to held-out appearance attribute (object color, lighting, viewpoint) on robot clips | Grounds 'latents cluster by appearance' in a measured quantity |
| Baseline (correlational) | SALA (language-similarity soft neighborhoods) | Tests whether equivariance + anchors beat correlational alignment |
| Baseline (reconstruction-only) | LAPA (VQ latent actions, no appearance operator) | Tests whether appearance-shift operator is necessary |
| Baseline (domain adaptation) | Human-robot contrastive visual pretraining (E3-style) | Tests whether latent-action transport beats representation alignment |
| Ablation-of-ours | CALIC minus anchor loss (equivariance + language only) | Isolates the interventional-anchor contribution |
| Ablation-of-ours | CALIC with lambda_lang = 0 (equivariance + anchors only) | Tests whether correlational term is load-bearing or harmful |
| Ablation-of-ours | CALIC with lambda_eq = 0 (anchors + language only) | Tests whether equivariance regularizer is load-bearing |

### Why this setup validates the claim

The central claim is that latent actions become appearance-invariant and anchored to the robot action manifold only when an equivariance constraint is pinned by a small set of interventional anchors. Each baseline removes exactly one ingredient of that argument. SALA retains language-similarity neighborhoods but has no equivariance or anchor term, so it tests whether correlational semantic alignment alone suffices; if CALIC beats SALA, the gain must come from the equivariance-plus-anchor structure. LAPA retains VQ latent actions and reconstruction but has no appearance-shift operator, so it tests whether equivariance to A_phi is doing real work rather than being decorative. The E3-style contrastive baseline aligns human and robot representations without any latent-action transport, so it tests whether the transport map T_phi adds anything beyond representation-level domain adaptation. The anchor ablation is the diagnostic centerpiece: if CALIC-without-anchors already matches CALIC, then the identifiability argument is wrong and anchors are unnecessary. The lambda_lang = 0 ablation tests whether the correlational term is load-bearing or actively harmful once anchors are present; the lambda_eq = 0 ablation tests whether the equivariance regularizer is load-bearing. The appearance-leakage probe provides a direct, measured signal: if CALIC's latents are truly appearance-invariant, the linear classifier accuracy from z to held-out appearance attributes should be near chance, whereas SALA and LAPA should show above-chance accuracy. Success rate on held-out real-world tasks is the right primary metric because the claim is about action faithfulness, not reconstruction fidelity; a latent that reconstructs well but clusters by appearance would still fail downstream, and success rate is the only signal that distinguishes these. The anchor budget sweep over {50, 100, 200} establishes the sensitivity of the identifiability argument to N_anchor.

### Expected outcome and causal chain

**vs. SALA** — On a case where a human clip shows a hand manipulating a red mug and the robot clip shows a gripper on a blue mug, SALA's language-similarity neighborhoods pull the human latent toward robot latents that share description words, but because its supervision is correlational, latents still cluster partly by visual appearance, so the transported action drifts on appearance-confounded pairs. CALIC instead forces z_{A_phi(h)} = T_phi(z_h) and pins T_phi with anchors, so appearance is explicitly factored out. We expect a noticeable success gap on appearance-shifted object subsets but near-parity on appearance-matched subsets, and we expect the appearance-leakage probe to show above-chance accuracy for SALA and near-chance accuracy for CALIC.

**vs. LAPA** — On a case where human and robot execute the same grasp under different lighting and viewpoint, LAPA's VQ latents are trained purely by reconstruction, so the codebook absorbs appearance variation into the latent codes and the human-to-robot mapping is ill-defined. CALIC's A_phi moves the human clip along the appearance direction before encoding, so the equivariance loss forces the codebook to ignore that direction. We expect LAPA to fail on cross-appearance transfer and CALIC to succeed there, with the gap concentrated on clips whose appearance differs most from the robot domain, and we expect the appearance-leakage probe to show high accuracy for LAPA and near-chance accuracy for CALIC.

**vs. Human-robot contrastive visual pretraining** — On a fine-grained insertion task where the decisive signal is a small motion, contrastive pretraining aligns global visual semantics but discards the motion detail, so the policy stalls at contact. CALIC's latent action carries the motion because the anchor loss ties transported latents to robot action latents on the same action. We expect a clear gap on fine-grained insertion and rough parity on coarse pick-and-place, where appearance alignment alone is sufficient.

**vs. CALIC minus anchor loss** — Without anchors, the equivariance loss alone can be satisfied by A_phi collapsing onto action-bearing content (the degenerate shortcut), so the transport map T_phi is underdetermined and the latent action is not pinned to the robot action manifold. We expect CALIC-without-anchors to underperform full CALIC on appearance-shifted subsets, with the gap shrinking as N_anchor increases from 50 to 200.

**vs. CALIC with lambda_lang = 0** — If the language-similarity term is load-bearing, removing it should hurt performance on tasks where descriptions are action-discriminative; if it is harmful, removing it should improve performance by eliminating appearance-confounded supervision. We expect lambda_lang = 0 to match or slightly improve full CALIC on appearance-shifted subsets, indicating that anchors alone suffice once present.

**vs. CALIC with lambda_eq = 0** — If the equivariance regularizer is load-bearing, removing it should hurt performance on appearance-shifted subsets; if anchors alone suffice, lambda_eq = 0 should match full CALIC. We expect lambda_eq = 0 to underperform full CALIC on appearance-shifted subsets, confirming that the equivariance term is doing real work beyond anchors.

### What would falsify this idea

If CALIC's gain over SALA and LAPA is uniform across appearance-shifted and appearance-matched subsets rather than concentrated where the predicted appearance-leakage failure occurs, the equivariance mechanism is not doing what we claim. Equally, if the anchor ablation matches full CALIC, the interventional-anchor identifiability argument is wrong and the central claim collapses. Additionally, if the appearance-leakage probe shows near-chance accuracy for SALA and LAPA, the premise that latents cluster by appearance is false and the motivation for CALIC is undermined.

## References

1. SALA: Semantically Aligned Latent Actions Enable Dual-Axis Scaling of Embodied Foundation Models
