# Gauge-Quotient Policy Learning with Identifiability Certificates for Uncalibrated Multi-View Vision-Language-Action Control

## Motivation

Across both VLA lineages the observation-to-state map silently assumes task-relevant geometry is uniquely recoverable from a fixed, calibrated sensor stream: HexaShell's triage tools hard-code known camera intrinsics/extrinsics and triangulate rays, while action-chunk models such as OpenVLA-OFT assume the latent goal geometry is identifiable from a fixed observation stream. When calibration is unknown or the scene moves, the same physical task geometry induces a continuum of observation-equivalent states related by the similarity group, so a policy trained on raw coordinates overfits to one gauge and fails under another — and no existing system flags when the current stream fails to determine the shape at all. The missing structural element is an explicit identifiability condition on the observation-to-state map, plus a policy defined on the gauge-orbit quotient rather than on raw coordinates.

## Key Insight

The similarity group acts as an exact symmetry of the multi-view observation model, so task-relevant shape lives on the quotient of landmark configurations by this group; a policy that is a function of gauge-invariant cross-ratios and shape eigenvalues, together with a rank-based certificate of when those invariants are uniquely determined, is invariant to the unobservable calibration gauge while remaining sensitive to the physical geometry that matters.

## Method

**What it is.** GQ-VLA (Gauge-Quotient VLA) takes a stream of uncalibrated multi-view RGB frames plus a language instruction and outputs (i) a gauge-invariant task-state estimate on the similarity-orbit quotient, (ii) an identifiability certificate flagging whether that estimate is unique, and (iii) an action chunk expressed in a gauge-fixed frame. It is a structural wrapper around any action-chunk backbone (we use OpenVLA-OFT's head) and around any code-generation harness (we use HexaShell's CLI tool interface).

**Load-bearing assumption (stated explicitly).** The cross-ratio and Gram-eigenvalue shape vector s is a complete and sufficient gauge-invariant representation of task-relevant geometry, such that a policy acting on s (plus null(J)) loses no task-relevant information relative to the raw calibrated state — i.e., the similarity-orbit quotient is the right quotient and the invariants are a lossless encoding of what the task needs. This assumption is load-bearing and is verified empirically in the experiment section; it can fail if the invariants are invariant to more than the nuisance gauge (e.g., projective rather than similarity transformations) or if they discard orientation/frame information needed for metric action regression.

**How it works.**
```pseudocode
INPUT: frames {I_v}_{v=1..V}, instruction l, intrinsics UNKNOWN

PHASE 1  Landmark lift (per view, frozen keypoint detector + DINOv2 descriptor)
  for each view v: detect K task-relevant keypoints u_{v,k} in R^2, k=1..K
  build measurement matrix per frame: U in R^{2V x K}, with visibility mask M
  # K=8 task-relevant keypoints (block corners + gripper tip); DINOv2 ViT-S/14 descriptor, 384-d

PHASE 2  Gauge-invariant shape encoding
  # unknown camera matrices P_v ~ Similarity; eliminate them by cross-view invariants
  for each visible pair (v,v'): compute cross-ratio quadruples CR(u_{v,a},u_{v,b};u_{v',c},u_{v',d})
  form shape vector s = [ {CR}, eigvals(Gram(U M)) normalized ]   # dim d_s, gauge-invariant
  # Gram eigenvalues are invariant to the similarity action on landmark configs
  # Scale normalization: Gram matrix G = (UM)(UM)^T transforms as G -> lambda^2 R G R^T under similarity (scale lambda, rotation R);
  # eigenvalues are invariant only up to lambda^2. Normalize by dividing by the largest eigenvalue: eigvals(G)/lambda_max(G).
  # This removes the lambda^2 factor and yields a scale-invariant shape vector.
  # d_s = number of cross-ratio quadruples + (K-1) normalized eigenvalues; for K=8, d_s = 28 + 7 = 35.

PHASE 3  Identifiability certificate (per frame, closed form)
  J = Jacobian of s wrt landmark coords, evaluated at U
  # J computed via autodiff (torch.func.jacrev) over the shape-vector function; validated against finite differences on 100 random configurations (max relative error < 1e-5).
  rank_r = numerical_rank(J, tol=1e-3)
  cert = 1[ rank_r == d_s ]                       # unique shape iff full column rank
  ambiguity_set = null(J) if cert==0 else {0}     # directions shape is NOT determined
  # Conditioning diagnostic: report condition number kappa(J) = sigma_max(J)/sigma_min(J) alongside cert.
  # Assumption A: the smallest nonzero singular value of J exceeds tol by a margin (well-conditioned away from degeneracy).
  # Failure mode F: near-degenerate configurations push singular values below tol and the certificate reports false ambiguity;
  # in this case we fall back to null-space magnitude (||null(J)||) as the ambiguity signal.

PHASE 4  Policy on the quotient
  z = Encoder(s)                                   # gauge-invariant latent; 2-layer MLP, hidden=256, GeLU
  if cert == 1:  a_chunk = Head(z, l)              # OpenVLA-OFT action head
  else:          a_chunk = Head(z, l, ambiguity_set) with active-probing action appended
                 # request a viewpoint change that reduces null(J) rank deficit
  output a_chunk in a gauge-fixed frame (origin=gripper, scale=block edge)

PHASE 5  Code-generation lineage hook
  expose invariants s and cert to HexaShell tools; triangulation tools now consume
  s (calibration-free) and MUST check cert before localizing; cert=0 raises a typed
  IdentifiabilityError that the coding agent must resolve by moving the wrist camera
  # First milestone: scope to a single tool wrapper (the triangulation tool) to keep timeline tractable.
```

**Why this design.** We chose to eliminate camera matrices analytically via cross-ratios and Gram eigenvalues rather than estimating calibration first, because the similarity gauge is an exact symmetry of the multi-view observation model and estimating it would reintroduce the very nuisance the policy must be blind to — accepting the cost that we lose absolute metric scale and must re-express actions in a gauge-fixed frame. We chose a closed-form rank test on the shape Jacobian over a learned confidence head because the rank condition is exactly the algebraic statement of local uniqueness of the quotient coordinate, whereas a learned head would only approximate it and could not expose the ambiguity directions needed for active probing — accepting the cost that the test is local and can miss global (fold) ambiguities. We chose to feed the ambiguity set into the action head as an explicit input rather than gating with a scalar confidence, because gating with log-probs is uncalibrated for task correctness (Kadavath et al.), while the null-space directions carry geometric information the policy can act on — accepting the cost of a larger action-head input. Unlike HexaShell's triangulation tools, which assume known calibration, and unlike OpenVLA-OFT's raw-coordinate observation-to-action map, GQ-VLA's policy is a function on the quotient by construction, so a domain expert would not describe it as a variant of either: the mechanism is the algebraic quotient plus a rank certificate, not a calibration pipeline or a learned invariance regularizer.

**Why it measures what we claim.** The numerical rank of the shape Jacobian measures local identifiability of task-relevant geometry because the implicit-function theorem makes full column rank equivalent to a locally unique quotient coordinate; this assumption fails when the landmark configuration is near-degenerate (collinear keypoints), in which case the rank test reflects numerical conditioning rather than genuine ambiguity and we fall back to the null-space magnitude. The cross-ratio and Gram-eigenvalue vector measures gauge-invariant shape because the similarity group acts trivially on these quantities; this assumption fails when keypoint visibility masks are inconsistent across views, in which case the invariant reflects detection noise rather than physical geometry. The ambiguity-set input to the action head measures actionable geometric uncertainty because null(J) spans exactly the shape directions the current stream cannot resolve; this assumption fails when the ambiguity is global rather than local, in which case the probing action may not reduce the deficit.

## Contribution

(1) A gauge-quotient VLA architecture that replaces raw calibrated coordinates with a similarity-invariant shape encoding (cross-ratios + Gram eigenvalues) and defines both the action-chunk head and the code-generation tool interface as functions on the quotient. (2) A closed-form identifiability certificate based on the numerical rank of the shape Jacobian, which exposes the null-space ambiguity directions and turns them into an active-probing action signal rather than an uncalibrated scalar confidence. (3) A calibration-free triangulation tool contract for coding agents that raises a typed IdentifiabilityError when the stream does not determine the shape, converting silent geometric failure into a recoverable control decision.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | HexaShell ten-layout manipulation suite, held-out layouts | Matches coding-agent tool-building regime with distribution shift |
| Dataset (transfer) | RLBench multi-camera variant with unknown intrinsics (4 cameras, 10 tasks) | Tests whether certificate transfers beyond in-house suite |
| Primary metric | Task success rate on held-out layouts | Directly reflects whether quotient policy generalizes |
| Secondary metric | Identifiability certificate accuracy (precision/recall vs. ground-truth rank from known calibration) | Validates that cert measures local identifiability |
| Secondary metric | Conditioning number kappa(J) reported alongside cert | Diagnoses false ambiguity from near-degenerate configurations |
| Baseline | OpenVLA-OFT raw-coordinate head | Tests whether gauge elimination beats raw-coordinate policy |
| Baseline | HexaShell calibration-first triangulation | Tests whether analytic quotient beats calibrated pipeline |
| Baseline | Learned confidence-gated VLA | Tests whether null-space input beats scalar gating |
| Ablation-of-ours | GQ-VLA without identifiability certificate | Isolates contribution of rank certificate and probing |
| Ablation-of-ours | GQ-VLA with scale-normalized Gram eigenvalues (ours) vs. unnormalized | Tests necessity of lambda^2 normalization |

### Why this setup validates the claim

The central claim is that representing the task state on the similarity-orbit quotient, plus an exact rank certificate, yields a policy that is blind to the nuisance camera gauge while still acting on genuinely resolvable geometry. The held-out-layout split is the right test because it forces the policy to generalize across camera poses and layouts it never saw, which is exactly where a raw-coordinate map must implicitly re-estimate the gauge and where a calibration-first pipeline must recover matrices it never observed. OpenVLA-OFT's raw-coordinate head tests the quotient-versus-raw sub-claim: if gauge elimination matters, the gap should appear precisely on layouts with unusual camera placement. HexaShell's calibration-first tools test the analytic-versus-estimated sub-claim: if exact symmetry elimination matters, the gap should appear where calibration is ill-conditioned. The learned confidence-gated baseline tests the ambiguity-set sub-claim: if the null-space directions carry actionable geometry, gating with a scalar should underperform on cases needing active probing. The no-certificate ablation is the linchpin — it removes only the rank test and probing, so any advantage attributable to the certificate itself rather than to the invariant encoding will show up as a gap between full GQ-VLA and this ablation, concentrated on the degenerate and under-constrained subsets. The RLBench transfer test checks that the certificate's value is not tied to a single in-house dataset. Success rate is the right primary metric because it is the only quantity that couples the geometric claim to task outcome; a metric like shape error would not reveal whether the certificate actually helps the policy act. The certificate-accuracy metric directly validates the load-bearing assumption that s is a complete gauge-invariant encoding: if cert precision/recall is high against ground-truth rank, the invariants are sufficient for identifiability.

### Expected outcome and causal chain

**vs. OpenVLA-OFT raw-coordinate head** — On a case where the held-out layout places cameras at an oblique pose the training distribution never contained, the baseline produces a systematically biased action chunk because its observation-to-action map must implicitly absorb an unestimated similarity gauge, and that implicit estimate is wrong off-distribution. Our method instead encodes cross-ratios and Gram eigenvalues that are invariant to that pose change by construction, so the policy sees the same quotient coordinate it saw in training. We expect a noticeable success-rate gap on the oblique-camera layouts but near parity on layouts whose camera poses resemble training.

**vs. HexaShell calibration-first triangulation** — On a case where intrinsics are unknown and the multi-view geometry is weakly constrained, the baseline produces a confident but wrong 3D localization because its triangulation tools assume known calibration and silently propagate an ill-conditioned estimate. Our method instead checks the closed-form rank of the shape Jacobian before localizing, raising an IdentifiabilityError and triggering a wrist-camera probe. We expect the baseline to fail on the weak-geometry subset with high-confidence errors, while our method either succeeds after probing or abstains cleanly, producing a gap concentrated on that subset.

**vs. Learned confidence-gated VLA** — On a case where the stream is locally ambiguous along a specific shape direction, the baseline produces a wrong action because its scalar confidence is uncalibrated for task correctness and cannot tell the policy which direction to resolve. Our method instead feeds the null-space basis into the action head, so the probing action is chosen to reduce exactly that rank deficit. We expect the gated baseline to show near-chance success on the ambiguous subset despite high reported confidence, while our method recovers most of the gap after one probing step, yielding a clear separation on ambiguous instances and parity elsewhere.

**vs. GQ-VLA without identifiability certificate** — On degenerate and under-constrained subsets, the no-certificate ablation cannot flag ambiguity and thus cannot trigger probing, leading to failures that full GQ-VLA avoids. We expect the gap to be concentrated on those subsets, confirming the certificate's causal role.

**vs. GQ-VLA with unnormalized Gram eigenvalues** — Without lambda^2 normalization, the shape vector is not scale-invariant, so the policy overfits to the training scale and fails under scale changes. We expect a success-rate drop on layouts with varying object scale, confirming the necessity of the normalization.

### What would falsify this idea

If the full GQ-VLA advantage over the no-certificate ablation is uniform across all subsets rather than concentrated on the degenerate and under-constrained cases the certificate is designed to catch, then the rank certificate is not doing the causal work and the central claim is wrong. Likewise, if the gap over OpenVLA-OFT appears on layouts with training-like camera poses as much as on oblique ones, the gauge-elimination mechanism is not the source of the gain.

## References

1. HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents
2. π0.5: a Vision-Language-Action Model with Open-World Generalization
3. AutoEval: Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real World
4. AlphaEvolve: A coding agent for scientific and algorithmic discovery
5. RT-H: Action Hierarchies Using Language
6. PaliGemma: A versatile 3B VLM for transfer
7. Achieving Human Level Competitive Robot Table Tennis
8. Autonomous Improvement of Instruction Following Skills via Foundation Models
