# Modality-Calibrated Routing Priors for Sparse Mixture-of-Experts Scientific Language Models

## Motivation

Sparse MoE backbones that ingest heterogeneous scientific encoders (molecular graphs, biological sequences, spectra) via modality-specific projectors, such as the Sage S1.5 Flash 320B model, feed a single shared gating network token distributions that differ systematically by modality, so the router's expert-count and top-k decisions inherit a modality-dependent bias that no shared gate can correct. DynMoE (E1) makes top-k adaptive per token but assumes a homogeneous input distribution and never tests whether its learned expert-count selection stays stable when the token stream is a mixture of projector outputs; the abstract-only evidence for Sage (E5) reports no routing diagnostics at all, so the instability is unmeasured rather than absent. The root cause is structural: the gate is trained to minimize task loss, not to make its assignment statistics invariant to the modality label, so any modality-correlated shift in projector output scale or sparsity is absorbed directly into routing entropy and expert count.

## Key Insight

Routing decisions become modality-invariant not by adding a separate router per modality but by constraining the shared gate's output distribution to match a modality-marginalized reference distribution, which makes the modality label a quantity the gate is provably unable to exploit as a routing cue.

## Method

**What it is.** Modality-Calibrated Routing Priors (MCRP) is a training-time calibration procedure for a shared sparse-MoE gate: it takes per-token projector outputs plus a modality label (molecular, biological, spectral, text) and returns a calibrated gate that produces expert-count and top-k decisions whose distribution is invariant to that label, while leaving the underlying DynMoE-style adaptive expert-count mechanism intact.

**How it works.**

```pseudocode
# Per MoE layer l, shared gate G_l(theta) -> router logits z in R^N_experts (N_experts = 64)
# Modality label m in {mol, bio, spec, text}; calibration heads H_m (2-layer MLP, width 64, GeLU)
# H_m outputs (a_m, b_m) in R^N_experts: c_i = a_m ⊙ z_i + b_m  (modality-conditioned affine)
# Reference distribution r is an EMA of the pooled marginal, updated each step with decay 0.99

for each training batch B of 256 tokens (x_i, m_i):
    z_i = G_l(x_i)                              # shared router logits (float32)
    (a_i, b_i) = H_{m_i}(z_i)                   # per-modality affine calibration
    c_i = a_i ⊙ z_i + b_i                       # affine offset (scale + location)
    p_i = softmax((z_i + c_i) / tau)            # tau = 1.0 fixed
    k_i = DynMoE_count(p_i)                     # inherited adaptive expert count
    top_k experts routed by p_i; standard MoE forward

# Routing-consistency loss (added to task loss, weight lambda = 0.1)
L_rc = sum_m  KL( mean_{i: m_i=m} p_i  ||  r )
#   -> each modality's mean assignment distribution matches the EMA reference r
L_ent = sum_m  max(0, H_floor - H( mean_{i: m_i=m} p_i ))  # H_floor = 0.5 * log(N_experts)
L_cal = lambda * L_rc + mu * L_ent               # mu = 0.01

# Calibration heads trained only on L_cal; shared gate trained on task loss + L_cal
# Reference r updated as: r <- 0.99 * r + 0.01 * mean_{i} p_i (detached, no gradient)
# Initial r = uniform(1/N_experts)
```

A per-modality prior network H_m is a two-layer MLP (width 64, GeLU) that maps the shared logits z to an affine offset (a_m, b_m); it cannot see x directly, only the shared gate's logits, so it can only re-shape, not re-route from scratch. The reference distribution r is an exponential moving average (decay 0.99) of the batch-pooled mean assignment, so the consistency term has a stable target that does not move with the current batch's modality mixture.

**Why this design.** We chose modality-conditioned affine offsets (a_m ⊙ z_i + b_m) over independent per-modality routers because independent routers would fragment the expert pool into modality-specialized silos, destroying the cross-modal transfer that motivates a shared backbone in the first place, accepting the cost that the affine transform cannot fully undo a projector whose output geometry is qualitatively different from text. We chose an EMA reference over a frozen uniform prior because a uniform target would force every modality toward maximum entropy and erase DynMoE's adaptive expert-count advantage, accepting the cost that the EMA reference is still an estimate and may lag at the start of training. We chose to train calibration heads only on L_cal while the shared gate sees task loss plus L_cal because letting the heads see task loss would let them become a second router that re-introduces modality leakage, accepting the cost that the heads' gradients are weaker and converge more slowly. We chose a per-modality entropy floor (H_floor = 0.5 * log(N_experts)) rather than a global entropy penalty so that legitimate narrow specialization is not penalized. A domain expert would not describe this as DynMoE plus a router-prior head, because the consistency loss constrains the *distributional relationship between modality-conditional and EMA-pooled routing statistics*, a quantity DynMoE's per-token adaptive top-k objective never touches.

**Why this measures what we claim.** The routing-consistency loss L_rc measures modality-invariance of expert assignment because KL between the modality-mean assignment and the EMA reference is zero exactly when the modality label is uninformative about the expected routing distribution; this assumption fails when a modality genuinely warrants a different expert mix (e.g., spectra always route to the same two experts for legitimate reasons), in which case L_rc penalizes correct specialization and the metric reflects forced uniformity rather than calibration. The entropy term L_ent measures preservation of adaptive expert-count capacity because the entropy of the modality-mean distribution is monotone in the effective number of experts that modality activates; this assumption fails when a modality has genuinely low entropy due to a narrow but correct expert preference, in which case L_ent reflects legitimate concentration rather than lost adaptivity. The calibration offset c_i measures the modality-induced shift in router logits because an affine transform (a_m ⊙ z_i + b_m) is the sufficient statistic for a location-and-scale family shift in logit space; this assumption fails when the shift is rank-changing (e.g., a full covariance rotation), in which case c_i reflects only the diagonal scale and location components and residual modality bias leaks through.

## Contribution

(1) A modality-calibrated routing-prior framework (MCRP) that augments a shared sparse-MoE gate with per-modality additive offset heads trained under a routing-consistency loss, preserving DynMoE-style adaptive expert-count selection while constraining routing statistics to be modality-invariant. (2) A design principle: routing stability under heterogeneous projector inputs is governed by the distributional relationship between modality-conditional and pooled assignment statistics, not by per-modality router capacity, which motivates distribution-matching constraints over architectural separation. (3) A routing-diagnostics protocol that reports per-modality assignment entropy, expert-count variance, and cross-modality KL as first-class metrics alongside task performance.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Mixed-modality routing benchmark (text + molecular + biological + spectral) built from public sources: text from OLMoE pretraining mixture (E2), molecular from QM9, biological from UniProt sequences, spectral from NIST/IR spectra; 256-token batches with balanced modality sampling | Forces shared gate to face genuinely heterogeneous projector geometries |
| Primary metric | Modality-conditional routing divergence (KL from EMA reference) | Directly measures the invariance claim, not downstream accuracy |
| Secondary metric | Modality-mean assignment entropy (effective expert count) | Measures preservation of DynMoE adaptivity |
| Baseline | Uncalibrated shared sparse-MoE gate | Tests whether calibration is needed at all |
| Baseline | Independent per-modality routers | Tests whether affine offset beats silo fragmentation |
| Baseline | Uniform-prior regularization on gate | Tests whether EMA reference beats fixed target |
| Baseline | Gradient-reversal modality classifier on router logits (domain-adversarial invariance) | Tests whether affine-offset-plus-KL is more than a reparameterization of known invariance regularization |
| Ablation-of-ours | MCRP without L_ent term | Isolates adaptivity-preservation from invariance |
| Ablation-of-ours | MCRP with additive-only offset (b_m only, a_m=1) | Tests whether affine scale correction is necessary vs. location-only |

### Why this setup validates the claim

The central claim is that a shared sparse-MoE gate can be made modality-invariant in its expected routing distribution without sacrificing DynMoE's adaptive expert-count capacity. The uncalibrated gate baseline tests the first sub-claim negatively: if modality-conditional routing divergence is already near zero without calibration, the method has nothing to fix. The independent-per-modality-router baseline tests the design choice: it should drive divergence toward zero too, but at the cost of cross-modal transfer, so downstream accuracy on shared-modality-transfer subsets separates "invariant because calibrated" from "invariant because siloed." The uniform-prior baseline tests the reference-distribution choice: it should also reduce divergence while collapsing entropy, so the entropy of modality-mean distributions distinguishes our EMA reference from a fixed target. The gradient-reversal baseline tests whether the affine-offset-plus-KL design is more than a reparameterization of known invariance regularization: if it matches MCRP on both divergence and entropy, the specific calibration mechanism is not doing additional work. The additive-only ablation tests the load-bearing assumption that the shift is location-and-scale rather than location-only. The no-L_ent ablation isolates whether the entropy term is what preserves adaptivity. The primary metric, modality-conditional routing divergence, is the right detector because it is exactly the quantity the loss constrains; a downstream-accuracy-only evaluation could not tell calibration from siloing from entropy collapse.

### Expected outcome and causal chain

**vs. Uncalibrated shared sparse-MoE gate** — On a case where spectral tokens arrive through a projector whose output geometry is shifted relative to text, the baseline produces a modality-clustered expert assignment because the shared gate's logits inherit the projector-specific location and scale shift. Our method instead adds a per-modality affine offset learned from the gate's own logits, which recenters and rescales each modality's mean assignment toward the EMA reference because the affine transform is the sufficient statistic for a location-and-scale shift. We expect a clear divergence gap on the spectral and molecular subsets, with near-parity on text where the shift is smallest.

**vs. Independent per-modality routers** — On a case where a molecular token's correct expert is one that text tokens also populate, the baseline produces a hard silo assignment because each router only ever sees its own modality and cannot route across the shared pool. Our method instead keeps one shared gate and only re-shapes its logits per modality, so the molecular token can still reach text-populated experts because the affine offset cannot re-rank from scratch. We expect comparable routing divergence but a visible downstream gain on cross-modal-transfer subsets for our method, with the silo baseline losing precisely there.

**vs. Uniform-prior regularization on gate** — On a case where DynMoE would legitimately activate two experts for a narrow spectral token, the baseline produces a near-maximum-entropy assignment because a fixed uniform target pulls every modality toward uniform routing. Our method instead uses the EMA reference, so a genuinely narrow modality keeps its concentration because the EMA mean itself is not uniform. We expect similar routing divergence but a markedly higher modality-mean entropy under the uniform baseline, and better adaptive-expert-count behavior (fewer wasted activations) for our method on the narrow-modality subset.

**vs. Gradient-reversal modality classifier on router logits** — On a case where the modality label is linearly decodable from router logits, the baseline removes the label signal by adversarial training but does not constrain the expected assignment distribution to a stable reference, so it can achieve low divergence while still allowing modality-specific mean shifts that cancel in the classifier. Our method instead directly matches each modality-mean to the EMA reference, so we expect lower routing divergence and more stable entropy across modalities, with the gradient-reversal baseline showing higher variance in modality-mean entropy.

**vs. Additive-only offset (a_m=1)** — On a case where the projector shift includes a scale change (e.g., spectral tokens have larger logit variance), the additive-only ablation cannot correct the scale and leaves residual divergence. Our method with affine offset should show lower divergence on the spectral subset, confirming that scale correction is necessary.

### What would falsify this idea

If the routing-divergence gain over the uncalibrated gate is uniform across all modality subsets rather than concentrated on the subsets predicted to carry projector-induced location-and-scale shifts, the calibration is not doing what we claim and the central mechanism is wrong. Likewise, if the no-L_ent ablation matches full MCRP on both divergence and modality-mean entropy, the entropy term is inert and the adaptivity-preservation claim collapses; and if the additive-only ablation matches full MCRP, the affine scale correction is unnecessary and the load-bearing assumption reduces to location-only.

## References

1. Sage S2.1 Flash Technical Report
