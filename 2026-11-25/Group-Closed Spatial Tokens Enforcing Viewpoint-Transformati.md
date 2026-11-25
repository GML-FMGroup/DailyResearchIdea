# Group-Closed Spatial Tokens: Enforcing Viewpoint-Transformation Composition Consistency in Multimodal LLM Scene Representations

## Motivation

S-Mind-style spatial-token representations re-emit fused multi-view geometry per reasoning viewpoint, but there is no constraint that the token operator forms a well-defined group action: composing two viewpoint transformations applied to the fused tokens need not equal the tokens produced by the composed transformation. This is a structural defect recurring across the spatial-token lineage — S-Mind reports strong per-viewpoint transformation accuracy (82.17%) yet the representation is only a per-viewpoint re-emitter, so multi-step reasoning that chains viewpoint changes (as in maze navigation, video-based reasoning, and multi-view tasks that Spark and ORIGAMISPACE probe) accumulates drift because the algebra the model implicitly uses is not closed. The root cause is that supervision targets each viewpoint independently, leaving the composition law unconstrained.

## Key Insight

If the spatial-token operator is trained to satisfy the group composition law on sampled transformation triples, then chained viewpoint reasoning becomes path-independent — the tokens reached by any sequence of transformations depend only on the net transformation, eliminating composition drift that per-viewpoint re-emission cannot detect.

## Method

**What it is.** Group-Closed Spatial Tokens (GCST) is a training objective and lightweight architecture addition on top of an S-Mind-style spatial-token MLLM. Inputs: a fused scene token set $Z_0 \in \mathbb{R}^{k\times d}$ produced from multi-view observations, and a set of viewpoint transformations $g$ drawn from a parametrized group $G$ (rotations, translations, and discrete viewpoint switches). Outputs: re-emitted token sets $Z_g = T_g(Z_0)$ that are required to satisfy $T_{g_2}(T_{g_1}(Z_0)) = T_{g_2 \circ g_1}(Z_0)$ for all sampled pairs, plus the usual per-viewpoint re-expression tokens used for downstream QA.

**Load-bearing assumption (stated explicitly).** The method assumes that the token-space operator $T_\theta$ can be made to satisfy the group composition law $T_{g_2}\circ T_{g_1} = T_{g_2\circ g_1}$ as a faithful (or at least drift-free) representation of viewpoint transformations, so that minimizing a relative Frobenius residual in the latent token space yields path-independent, algebraically consistent spatial reasoning rather than merely shrinking latent distances. This assumption is load-bearing because the entire closure penalty is defined in token space; if the token geometry is only loosely tied to the decoded scene, the penalty can be satisfied by representational slack (collapse toward a low-variance, near-constant token set) rather than by genuine algebraic consistency. We therefore anchor the closure penalty to a decoded/geometric observable (a frozen pose probe) and add an explicit anti-collapse term, as detailed below.

**How it works.**

```pseudocode
# Phase 1: base spatial-token supervision (inherited from S-Mind)
for each multi-view scene S with fused tokens Z0:
    for each viewpoint g in G:
        Z_g = T_theta(Z0, g)
        L_view += || Decode(Z_g) - target_view(g) ||^2

# Phase 2: closure-consistency training (new)
# Frozen pose probe P: 2-layer MLP (hidden=256, GeLU) trained on Phase-1 tokens to regress
# relative orientation (quaternion, 4-d) and translation (3-d) from Z_g; frozen during Phase 2.
for each scene S, sample M triples (g1, g2) from G:
    Z_g1      = T_theta(Z0, g1)
    Z_g2      = T_theta(Z0, g2)
    Z_g2g1    = T_theta(Z0, g2 o g1)          # composed transformation
    Z_chain   = T_theta(Z_g1, g2)              # compose in token space
    L_closure += || P(Z_chain) - P(Z_g2g1) ||^2 / || P(Z_g2g1) ||^2   # relative Frobenius on probe readout
    L_anti    += -log softmax( P(Z_g1) )_{g1}  # viewpoint-discrimination term (cross-entropy on probe logits)

# Phase 3: joint optimization
L = L_view + lambda_closure * L_closure + lambda_anti * L_anti + lambda_qa * L_qa
# lambda_closure = 0.5, lambda_anti = 0.1, lambda_qa = 1.0, M = 8 triples/scene
```

The transformation operator $T_\theta$ is a 2-layer equivariant MLP (hidden=256, GeLU activation) with a group-conditioned FiLM modulation on each spatial token; it is the ONLY operator applied to tokens, so closure is enforced on the same surface that produces the reasoning representation. Triples are sampled with stratified coverage over transformation magnitude (5 bins: <15°, 15–45°, 45–90°, 90–180°, >180° rotation; translation bins analogously) so small and large compositions are both supervised. The frozen pose probe $P$ is a 2-layer MLP (hidden=256, GeLU) trained on Phase-1 tokens to regress relative orientation (quaternion, 4-d) and translation (3-d) from $Z_g$; it is frozen during Phase 2 and provides a metric-meaningful readout for the closure penalty. The anti-collapse term $L_anti$ is a cross-entropy loss on the probe's viewpoint-classification logits, ensuring that $T_g(Z_0)$ remains discriminative across viewpoints and excluding the trivial constant-token solution.

**Why this design.** We chose to enforce closure as a relative Frobenius penalty on the frozen pose probe readout rather than on raw token space because decoding introduces a perceptual bottleneck that can mask token-level drift, and the probe provides a space with metric meaning; we accept the cost that the probe itself must be trained and frozen, and that its accuracy bounds the meaningfulness of the closure signal. We chose a single shared equivariant operator $T_\theta$ over separate per-transformation heads because a shared operator is the minimal architecture in which the composition law is even expressible — separate heads would make $T_{g_2}\circ T_{g_1}$ undefined — accepting the cost that the operator must generalize across the whole group rather than specialize. We chose stratified triple sampling over uniform sampling because uniform sampling over-weights small transformations where closure is nearly trivial, accepting the cost of a slightly more complex sampler. We added the anti-collapse term $L_anti$ because without it the closure penalty admits a trivial constant-token solution; this term is a direct specification of the existing viewpoint-discrimination requirement implicit in $L_view$, not a new mechanism. A domain expert would not describe this as 'S-Mind + a consistency regularizer' (the obvious-composition failure mode) because the closure penalty is not a generic smoothness prior: it constrains an algebraic law on the operator itself, and S-Mind's per-viewpoint supervision is provably (by construction) invariant to violations of that law, so no amount of S-Mind-style training can produce it.

**Why it measures what we claim.** The relative Frobenius residual $\|P(T_{g_2}(T_{g_1}(Z_0))) - P(T_{g_2\circ g_1}(Z_0))\| / \|P(T_{g_2\circ g_1}(Z_0))\|$ measures group-action closure because it assumes the frozen pose probe $P$ is a faithful readout of the token's geometric content, i.e., that probe-space composition mirrors scene-space composition; this assumption fails when the probe's tolerance for token perturbation is large relative to the residual, in which case the residual reflects representational slack rather than genuine algebraic inconsistency. The anti-collapse term $L_anti$ measures viewpoint discrimination because it assumes the probe's classification logits are sensitive to viewpoint differences; this fails when the probe saturates, in which case $L_anti$ provides no gradient against collapse. The stratified triple sampler measures coverage of the composition law because it assumes transformation magnitude correlates with closure difficulty; this fails when the group has non-commuting elements whose difficulty is driven by order rather than magnitude, in which case coverage reflects magnitude distribution rather than algebraic stress. The shared equivariant operator measures path-independence of chained reasoning because it assumes all reasoning viewpoints lie in the span of sampled transformations; this fails for out-of-distribution viewpoints, in which case the metric reflects interpolation quality rather than closure. To calibrate the probe's sensitivity, we run a calibration experiment: decode $Z_g$ plus $\epsilon$-noise (with $\epsilon$ scaled to the measured residual) and record view error, establishing the residual-to-scene-error mapping; we report closure residual alongside decoded-view error so the token-space metric is anchored to observable geometry.

## Contribution

(1) A group-closure training objective and equivariant token operator that make an MLLM's spatial-token representation a well-defined group action over viewpoint transformations, with a stratified triple sampler for composition supervision. (2) The design principle that closure in token space — rather than per-viewpoint decoding accuracy — is the correct supervision target for chained spatial reasoning, supported by an ablation isolating closure loss from view loss. (3) A diagnostic probe suite measuring composition residual, path-independence, and out-of-distribution viewpoint transfer for spatial-token MLLMs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | VSI-Bench video visual-spatial suite | Tests spatial memory and viewpoint reasoning |
| Primary metric | Group-closure residual (on frozen pose probe) + QA accuracy | Directly measures algebraic consistency and task payoff |
| Baseline | S-Mind (Qwen3.5 backbone) | Isolates gain from closure over inherited supervision |
| Baseline | Open-weight MLLMs of comparable scale (e.g., Qwen3.5-VL, LLaVA-OneVision) | Tests whether closure beats scale alone without inaccessible frontier models |
| Baseline | S-Mind + generic token smoothness prior | Distinguishes algebraic law from generic regularizer |
| Ablation-of-ours | GCST without closure term (L_view + L_qa only) | Removes the algebraic constraint, keeps architecture |
| Ablation-of-ours | GCST without anti-collapse term (L_view + L_closure + L_qa) | Tests whether anti-collapse is necessary to avoid trivial solution |
| Calibration | Decoder sensitivity: decode Z_g + ε-noise, record view error | Establishes residual-to-scene-error mapping |

### Why this setup validates the claim

The central claim is that closure is an algebraic property that per-viewpoint supervision cannot produce, so the design must separate "more spatial training" from "enforcing a composition law." S-Mind on Qwen3.5 is the right baseline because it already has the strongest inherited spatial-token supervision; any gain over it cannot be attributed to better per-viewpoint re-expression. The open-weight MLLM baselines test whether the effect is merely a scale artifact without relying on inaccessible frontier models. The smoothness-prior baseline is the crucial discriminator: it isolates whether the gain comes from constraining an algebraic law on $T_\theta$ rather than from generic token regularization. The no-closure ablation confirms the architecture alone is insufficient, and the no-anti-collapse ablation tests whether the closure penalty collapses without the discrimination term. The primary metric is the group-closure residual on the frozen pose probe because it is the only quantity that directly measures violation of $T_{g_2}\circ T_{g_1} = T_{g_2\circ g_1}$ in a space with metric meaning; QA accuracy is secondary because it can improve for reasons unrelated to closure. The calibration experiment anchors the token-space residual to decoded-view error, ensuring the residual reflects genuine algebraic inconsistency rather than representational slack. Together, a residual drop concentrated in the intended subsets, paired with QA gains that the smoothness prior does not reproduce, and a residual-to-scene-error mapping that confirms the probe's sensitivity, forms a falsifiable test.

### Expected outcome and causal chain

**vs. S-Mind (Qwen3.5)** — On a maze-navigation case requiring an intermediate viewpoint then a rotation, S-Mind produces tokens that decode plausibly per view but drift under chained re-expression because its loss never couples $T_{g_1}$ and $T_{g_2\circ g_1}$. Our method instead penalizes the probe-space composition residual directly, so the operator is pushed toward a consistent representation. We expect a substantial closure-residual reduction on multi-step, non-commuting subsets and near-parity on single-step subsets.

**vs. Open-weight MLLMs (Qwen3.5-VL, LLaVA-OneVision)** — On multi-view fusion cases with large rotation-then-translation compositions, these models answer from surface linguistic priors and lose track of relative orientation because they lack an explicit equivariant token surface. Our method forces path-independence on that surface, so composed viewpoints remain coherent. We expect a concentrated advantage on high-composition subsets and parity on simple single-view questions.

**vs. S-Mind + generic smoothness prior** — On cases where two transformations nearly cancel, a smoothness prior shrinks token differences and appears to help, but on large non-commuting compositions it cannot enforce order because it has no notion of $g_2\circ g_1$. Our method encodes the composition explicitly, so order-sensitive subsets remain correct. We expect the smoothness baseline to match us on small-magnitude triples but fall behind sharply on stratified large-magnitude and non-commuting triples.

### What would falsify this idea

If the closure residual drops uniformly across all transformation-magnitude and commutativity strata rather than concentrating where the predicted composition failure occurs, or if the smoothness-prior baseline reproduces the same residual and QA gains, or if the calibration experiment shows the probe's tolerance for token perturbation exceeds the residual (so the residual reflects slack rather than algebraic inconsistency), then the closure penalty is acting as a generic regularizer and the central algebraic claim is wrong.

## References

1. S-Mind: Internalizing Multi-View Geometry for Spatial Intelligence
