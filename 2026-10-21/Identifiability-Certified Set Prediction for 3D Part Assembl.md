# Identifiability-Certified Set Prediction for 3D Part Assembly

## Motivation

Transformer-based set predictors such as PART output a single variable-length part set from an image-derived structure graph, assuming a unique well-posed mapping from structural evidence to part poses. This assumption fails structurally under occlusions, symmetric parts, and missing edges, where multiple part-pose assignments are equally consistent with the observed evidence. No existing method represents this equivalence class or provides a checkable witness of non-identifiability, leaving downstream assemblers unable to distinguish a confident prediction from an arbitrary choice among indistinguishable alternatives.

## Key Insight

The set of part-pose assignments consistent with an under-determined structure graph forms a finite equivalence class under the prior's symmetry and occlusion relations, and this class can be exactly enumerated and certified by a witness assignment whose structural residuals vanish, making ambiguity a checkable property rather than an epistemic caveat.

## Method

**What it is.** We introduce **CERTIS** (Certified Equivalence-class Representer for Identifiable Sets), a transformer that takes as input an image-derived structure graph (nodes = part slots with observed geometry, edges = observed structural relations) and a part library, and outputs a finite set of part-pose assignments together with a certificate: a witness assignment that is structurally indistinguishable from the primary prediction under the prior's equivalence relation. The certificate is checked at inference by evaluating structural residuals against the observed graph.

**Load-bearing assumption (stated explicitly).** The prior equivalence relation ~_P, instantiated from the structure graph via a fixed generator set G_prior (symmetry orbits declared by the PartNet symmetry annotation and occlusion-induced unconstrained directions declared by the occlusion annotation), exactly enumerates the true set of part-pose assignments indistinguishable under the observed evidence, so that structural_residual(A,G)=0 if and only if A is genuinely non-identifiable given G. This assumption is load-bearing: the certificate certifies consistency with the declared prior, not with the true ambiguity. It fails when G_prior is incomplete (rare symmetry types, undeclared occlusions) or when the edge-residual model is misspecified; in that case the residual reflects prior incompleteness rather than true ambiguity. We therefore (i) report certificate soundness and completeness against an independent held-out evidence channel (see evaluation), and (ii) require the witness to be distinguishable from the primary under at least one held-out observation, so a zero-residual witness that is actually distinguishable is scored as a certificate failure rather than a success.

**How it works.**

```pseudocode
Input: structure graph G = (V, E_obs), part library L, prior equivalence relation ~_P
       with generator set G_prior = {g_1, ..., g_m} (symmetry orbits + occlusion directions)
Output: equivalence class C = {A_1, ..., A_k} of part-pose assignments, certificate W

1. Encode G with a 6-layer graph transformer (hidden=256, 8 heads, GeLU)
   -> node features H, edge features R.
2. Predict primary assignment A_0 = Decoder(H, L) via set-prediction head
   (3-layer MLP, hidden=512, output = per-slot 6-DoF pose + part-identity logits).
3. Predict a residual field Delta = ResidualHead(H, R) over part slots
   (2-layer MLP, hidden=256, GeLU, output = per-slot 6-DoF perturbation).
4. Enumerate candidate alternative assignments:
   for each generator g in G_prior consistent with G:
       A_g = g · A_0 + Delta_g   (Delta_g = Delta restricted to slots moved by g)
   keep A_g if structural_residual(A_g, G) <= tau   (tau = 0.02 in normalized units;
       calibrated per-instance from the observed edge-residual noise distribution
       at a 95% confidence level, see evaluation).
5. Certificate W = argmin_{A_g kept} structural_residual(A_g, G).
6. Train with loss L = L_set(C, C_gt) + lambda * L_cert(W, G)   (lambda = 0.5)
   where L_set is a set-matching loss (Hungarian matching over assignments,
   per-assignment L2 on 6-DoF poses + cross-entropy on part identities) enforcing
   that C equals the set of assignments consistent with ~_P, and L_cert enforces
   structural_residual(W, G) = 0 (L2 on edge residuals).
```

The prior equivalence relation ~_P is instantiated from the structure graph: two assignments are equivalent if they differ only by a permutation of part identities within a symmetry orbit declared by the prior, or by a pose adjustment that leaves all observed edge residuals unchanged (occlusion-induced). The residual head Delta is trained to predict the minimal pose perturbation that maps the primary assignment to each alternative in the class, so the class is generated rather than enumerated by brute force. The generator set G_prior is fixed at training time from the PartNet symmetry and occlusion annotations; no generator is learned.

**Why this design.** We chose to generate the equivalence class by a learned residual field over symmetry generators rather than by sampling multiple set predictions, because sampling produces assignments that are merely diverse, not certified to lie in the prior's equivalence class, accepting the cost that the residual field must be trained with a structural-residual loss that requires differentiable edge residuals. We chose a certificate witness that is a single assignment with minimal structural residual rather than a scalar confidence score, because a scalar cannot be checked against the prior without a calibration story (the failure mode of confidence gating), accepting the cost that the witness must be stored and verified at inference. We chose to enforce the class-equality loss L_set against the ground-truth equivalence class computed from the structure graph rather than against a single ground-truth pose set, because the single-pose supervision is exactly what makes PART and its descendants assume a unique mapping, accepting the cost that the ground-truth class must be computed at training time by applying the prior's generators to the annotated assembly. This design is not a variant of PART: PART outputs one set; CERTIS outputs a certified finite class and a witness, and the training signal is class equality rather than set regression.

**Why it measures what we claim.** The structural residual `structural_residual(A, G)` measures **non-identifiability** because under the named assumption that the prior's edge constraints are the only evidence linking poses to the graph, two assignments have equal residual if and only if they are indistinguishable under that evidence; this assumption fails when the prior is misspecified (e.g., an occlusion is not declared), in which case the residual reflects prior incompleteness rather than true ambiguity. The class-equality loss `L_set(C, C_gt)` measures **exact equivalence-class coverage** because it is computed as a set-matching loss over the ground-truth class generated by the same prior, under the assumption that the prior's generators enumerate all symmetries and occlusions present in the data; this assumption fails for rare symmetry types not in the generator set, in which case the loss penalizes a correct assignment as if it were missing. The certificate witness `W` measures **checkable non-identifiability** because it is the assignment with minimal residual among the generated alternatives, under the assumption that the residual head's perturbations span the tangent space of the equivalence class; this assumption fails when the residual head underfits, in which case the witness reflects the head's capacity rather than the true class.

## Contribution

['A transformer architecture (CERTIS) that outputs a finite equivalence class of part-pose assignments together with a structural-residual certificate, replacing single-set prediction with certified ambiguity representation.', "A training objective that enforces class equality against a prior-generated ground-truth equivalence class, making the predicted class exactly the set of assignments consistent with the structure graph's symmetry and occlusion relations.", 'A checkable certificate mechanism that verifies non-identifiability at inference by evaluating structural residuals, enabling downstream assemblers to distinguish certified ambiguity from arbitrary prediction.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | PartNet-style assemblies with annotated symmetries and occlusions (curated subset: 1 symmetry type — mirror pairs — and 1 occlusion type — single unobserved edge — for milestone 1; full generator set in phase 2) | Provides ground-truth equivalence classes for class-matching |
| Primary metric | Equivalence-class coverage at calibrated certificate residual ≤ tau (tau per-instance from observed edge-residual noise at 95% confidence) | Directly measures certified non-identifiability, not pose error |
| Secondary metric | Certificate soundness (false-certification rate) and completeness (missed-alternative rate) against brute-force enumeration on a held-out subset | Tests whether the certificate is falsifiable, not tautological |
| Baseline | PART | Single-set output; tests whether class coverage beats point regression |
| Baseline | Rectified Point Flow | Ad-hoc symmetry handling; tests whether explicit generators certify better |
| Baseline | Multi-sample set predictor (k=8 samples) | Tests whether diversity alone yields certified equivalence classes |
| Ablation-of-ours | CERTIS without residual head (brute-force generators) | Isolates contribution of learned residual field to class coverage |

### Why this setup validates the claim

The central claim is that CERTIS produces a *certified* equivalence class of assignments rather than a single best guess, and that this class is exactly the prior-consistent set. The dataset choice is load-bearing: PartNet-style assemblies come with annotated symmetry orbits and occlusion structure, so the ground-truth equivalence class can be computed by applying the prior's generators to the annotation — exactly the signal L_set is trained against. The primary metric, class coverage at a calibrated residual threshold, is the right detector because it is only high when the predicted class both matches the ground-truth class and the certificate witness passes the structural-residual check; a method that merely regresses poses well but misses class members scores low. To avoid the tautology of evaluating against the same generators used in training, we additionally compute a held-out equivalence class from an independent evidence channel (re-rendered depth from a held-out view) and report certificate soundness (false-certification rate) and completeness (missed-alternative rate) against brute-force enumeration on a 512-instance held-out subset. PART tests the sub-claim that single-set output cannot represent the class; Rectified Point Flow tests whether implicit symmetry handling can certify membership; the multi-sample predictor tests whether diversity without certification suffices. The ablation isolates the residual head, testing whether learned perturbations are needed to span the class or whether brute-force generator application is enough.

### Expected outcome and causal chain

**vs. PART** — On an assembly with a symmetric part pair (e.g., two interchangeable brackets), PART outputs a single pose set and commits to one identity assignment because its set-prediction head is trained with single-pose supervision, so it cannot represent the alternative labeling. Our method instead predicts the primary assignment, then applies the symmetry generator with a learned residual to generate the alternative and certifies it via structural residual. We expect a large coverage gap on symmetric-part subsets, with parity on asymmetric subsets where the class is a singleton.

**vs. Rectified Point Flow** — On an occluded part whose pose is under-constrained by observed edges, Rectified Point Flow regresses a point estimate and its ad-hoc symmetry handling does not declare the occlusion-induced equivalence, so it produces one pose with no witness. Our method's prior declares the occlusion, the residual head perturbs along the unconstrained tangent direction, and the witness achieves near-zero structural residual. We expect our certificate residual to sit near the calibrated tau while the baseline's implicit symmetry handling yields no checkable witness, showing a gap concentrated on occlusion-heavy instances.

**vs. Multi-sample set predictor** — On a case with both a symmetry orbit and an occlusion, a sampling-based set predictor emits diverse assignments but they drift outside the prior's equivalence class because diversity is unconstrained by structural residuals. Our method generates alternatives by composing generators with a residual field trained under the structural-residual loss, keeping every kept member inside the class. We expect the sampler to show high diversity but low class coverage, while CERTIS shows coverage near the ground-truth class size with residual below tau.

### What would falsify this idea

If CERTIS's coverage gain over PART is uniform across symmetric and asymmetric subsets rather than concentrated where the prior declares non-identifiability, or if the certificate witness residual is no lower than a randomly sampled alternative's residual, or if the false-certification rate against the held-out evidence channel is high (witness certified as indistinguishable but distinguishable under held-out view), the claim that the class is certified rather than merely diverse is wrong.

## References

1. PART: Learning 3D Part Assembly and Retrieval with Transformers
2. Rectified Point Flow: Generic Point Cloud Pose Estimation
3. PartNeXt: A Next-Generation Dataset for Fine-Grained and Hierarchical 3D Part Understanding
4. PuzzleFusion++: Auto-agglomerative 3D Fracture Assembly by Denoise and Verify
5. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
6. Point-SAM: Promptable 3D Segmentation Model for Point Clouds
7. SAMPart3D: Segment Any Part in 3D Objects
8. Boosting Latent Diffusion with Flow Matching
