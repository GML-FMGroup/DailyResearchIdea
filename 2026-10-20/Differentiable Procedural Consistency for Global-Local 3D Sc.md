# Differentiable Procedural Consistency for Global-Local 3D Scene Editing

## Motivation

InfiniDream's end-to-end pipeline treats per-asset symbolic constraints and global scene layout as separate stages, so local asset edits can silently violate the global layout and text semantics without any joint feedback. The same residual gap appears across Set-the-scene, CityCraft, and SceneDreamer: symbolic asset generation is never jointly constrained with the global placement objective, because procedural parameters are non-differentiable with respect to rendered geometry. This structural separation makes semantic-to-geometric consistency unenforceable under global-local editing.

## Key Insight

A differentiable proxy of the procedural renderer, trained to match the exact symbolic renderer on a distribution of parameter perturbations, makes the symbolic-to-geometric map differentiable, so a single gradient of a joint semantic-layout loss can flow into both procedural parameters and global layout without breaking the symbolic constraints.

## Method

**What it is.** We introduce **DPC (Differentiable Procedural Consistency)**, a joint consistency objective that couples text semantics, procedural asset parameters, and global scene layout through a differentiable proxy of InfiniDream's symbolic renderer. Inputs: a text prompt, a symbolic scene graph with per-asset procedural parameters, and a global layout (object poses/scales). Output: updated procedural parameters and layout that jointly minimize a semantic-to-geometric consistency loss.

**Load-bearing assumption (stated explicitly).** The method assumes that a proxy trained to low forward MSE on perturbations around the current scene yields gradients that are faithful enough to the symbolic renderer's true finite-difference response that, after K gradient steps, projection back onto the symbolic constraint set, and re-rendering with R_sym, the updated parameters and layout still satisfy the semantic and layout thresholds. This assumption is load-bearing because the entire optimization loop queries the proxy off its training distribution (the K steps deliberately move away from the initial scene). We therefore make the assumption testable and enforce it with a trust-region / on-policy proxy loop, described in Phase 2.

**How it works.**
```pseudocode
# Phase 1: Build differentiable proxy
Given symbolic renderer R_sym (InfiniDream's procedural pipeline)
Collect N=4096 (params, layout) perturbation pairs around the current scene,
  with perturbation radius r ~ Uniform(0, r_max), r_max = 0.1 * (param_range + layout_range)
For each, render R_sym to image I_sym and depth D_sym
Train proxy R_prox (small U-Net, 4 down/up blocks, 16M params) to minimize
    ||R_prox(params, layout) - (I_sym, D_sym)||^2 + lambda_smooth * TV(R_prox)
with lambda_smooth = 0.01; train to MSE < 1e-3 on held-out perturbations

# Phase 2: Joint consistency optimization with trust-region and on-policy refresh
For each edit request (text prompt t, local asset a to edit):
  Encode t with CLIP text encoder -> e_t
  Initialize trust-region radius delta = 0.05 * (param_range + layout_range)
  For K=8 optimization steps:
    Render scene through R_prox -> I, D
    Compute semantic loss L_sem = 1 - cos(CLIP_image(I), e_t)
    Compute layout loss L_lay = sum_i ||proj(R_prox(params_i, layout_i)) - proj(R_sym(params_i^0, layout_i^0))||_1
      over all assets i != a (preserve global layout)
    Compute parameter prior L_par = ||params_a - params_a^0||_2^2 (stay near symbolic manifold)
    Total L = L_sem + alpha * L_lay + beta * L_par, alpha=1.0, beta=0.1
    Compute gradient g = dL/d(params_a, layout_a) through R_prox
    Propose step: (params_a', layout_a') = (params_a, layout_a) - lr * g, lr = 0.01
    Clip step to trust region: if ||step|| > delta, scale step to ||step|| = delta
    Evaluate true loss L_true' = L_sem(R_sym(params_a', layout_a')) + alpha * L_lay(R_sym(params_a', layout_a'))
    Evaluate proxy-predicted loss L_prox' = L_sem(R_prox(params_a', layout_a')) + alpha * L_lay(R_prox(params_a', layout_a'))
    If |L_true' - L_prox'| > tau * max(L_true', L_prox'), tau = 0.1:
        reject step; shrink delta = 0.5 * delta; continue
    Else:
        accept step: (params_a, layout_a) = (params_a', layout_a')
        Add (params_a, layout_a) to proxy training set; every 2 accepted steps, fine-tune R_prox for 100 iterations on the accumulated trajectory points (on-policy refresh)
        Optionally expand delta = 1.2 * delta if |L_true' - L_prox'| < 0.5 * tau * max(L_true', L_prox')
    # Gradient-fidelity gate (checked every step before acceptance):
    Compute finite-difference gradient g_fd of L_true w.r.t. (params_a, layout_a) using central differences with epsilon = 1e-3 * (param_range + layout_range)
    Compute cosine similarity cos(g, g_fd)
    If cos(g, g_fd) < 0.7: reject step; shrink delta = 0.5 * delta; continue
  Project updated params_a back onto symbolic constraint set (clip to valid ranges)
  Re-render final scene with R_sym and verify L_sem + alpha*L_lay below threshold (same threshold as success criterion)
```

**Why this design.** We chose a learned differentiable proxy over analytically differentiating the symbolic renderer because InfiniDream's procedural algorithms contain discrete branching (e.g., conditional mesh assembly) that has no gradient; the proxy accepts the cost that it introduces approximation error, which we control by training on perturbations concentrated around the current scene rather than the full parameter space, and by the trust-region / on-policy refresh loop that rejects steps where the proxy's predicted loss disagrees with the true renderer's loss beyond tolerance tau=0.1 and where the proxy gradient's cosine agreement with finite differences falls below 0.7. We chose to freeze all non-edited assets' parameters and penalize layout drift (L_lay) rather than re-optimizing the whole scene because global-local editing semantics require that a local edit not silently move other objects; this accepts the cost that some edits requiring global rearrangement will be rejected by the consistency check. We chose a parameter prior (L_par) over unconstrained optimization because procedural parameters live on a symbolic manifold (valid ranges, discrete seed choices) and drifting off it destroys editability; the cost is reduced expressivity for edits that would require changing the asset's symbolic structure. A domain expert would not describe this as a variant of InfiniDream because InfiniDream has no gradient path from rendered geometry back to procedural parameters — its symbolic constraints are enforced only at generation time, not jointly with layout during editing.

**Why it measures what we claim.** The semantic loss `1 - cos(CLIP_image(I), e_t)` measures text-semantic alignment because CLIP's image-text embedding space is trained so that cosine similarity is monotone in human-judged semantic match; this assumption fails when the text describes spatial relations not captured by global CLIP embeddings, in which case the loss reflects coarse appearance similarity rather than relational semantics. The layout loss `||proj(R_prox) - proj(R_sym^0)||_1` measures global layout preservation because projecting rendered geometry to 2D and comparing to the pre-edit projection is equivalent to preserving object screen-space positions under the assumption that the camera is fixed and depth ordering is preserved so screen-space equality implies world-space layout equality; this assumption fails when the edit legitimately requires a camera change or occlusion reordering, in which case the loss reflects camera-induced displacement or occlusion-induced projection drift rather than object drift. The parameter prior `||params_a - params_a^0||^2` measures symbolic-manifold adherence because Euclidean distance in parameter space is a proxy for staying within the procedural algorithm's valid range under the assumption that valid ranges are convex; this assumption fails for non-convex ranges (e.g., seed-dependent branches), in which case the prior reflects distance to the initial seed rather than validity. The proxy MSE measures the fidelity of the differentiable surrogate because low MSE on held-out perturbations implies the proxy's gradient approximates the symbolic renderer's finite-difference gradient under the assumption that the perturbation distribution covers the edit trajectory; this assumption fails for edits far from the initial scene, in which case the proxy reflects interpolation error rather than true procedural response. The trust-region acceptance test `|L_true' - L_prox'| <= tau * max(L_true', L_prox')` measures proxy-vs-truth agreement along the actual edit trajectory because it compares the true renderer's loss to the proxy's predicted loss at each proposed step; this assumption fails when the proxy is queried far from its training distribution, in which case the test rejects the step and shrinks the trust region. The gradient-fidelity gate `cos(g, g_fd) >= 0.7` measures gradient faithfulness because cosine similarity between the proxy gradient and the finite-difference gradient of the true renderer is a direct measure of directional agreement; this assumption fails when the proxy's gradient field is unconstrained off-manifold, in which case the gate rejects the step and triggers on-policy proxy refresh.

## Contribution

["A differentiable proxy of InfiniDream's symbolic procedural renderer that enables gradient-based joint optimization of procedural parameters and global layout while preserving symbolic constraints via post-hoc projection.", 'A joint semantic-to-geometric consistency objective (DPC) that couples text semantics, per-asset procedural parameters, and global layout in a single loss, with explicit layout-preservation and parameter-prior terms that prevent local edits from violating global structure.', 'An editing protocol that verifies consistency by re-rendering with the exact symbolic renderer after proxy-based optimization, providing a concrete mechanism for rejecting edits that the proxy falsely certifies.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | InfiniDream edit suite (object + scene tasks) | Matches procedural pipeline and edit semantics |
| Primary metric | Edit success rate under consistency check | Directly tests joint semantic-layout feasibility |
| Baseline | InfiniDream regenerate-then-select | Tests whether gradient coupling beats sampling |
| Baseline | VLM-guided procedural edit (no proxy) | Tests value of differentiable proxy over API calls |
| Baseline | Unconstrained proxy optimization (no L_lay/L_par) | Tests whether priors are load-bearing |
| Ablation-of-ours | DPC with L_par removed | Isolates symbolic-manifold prior contribution |
| Ablation-of-ours | DPC with trust-region and on-policy refresh disabled (single in-distribution proxy) | Tests whether the trust-region / on-policy loop is load-bearing for proxy-gradient fidelity |
| Diagnostic metric | Cosine agreement between R_prox gradient and R_sym finite-difference gradient along the edit trajectory | Directly measures the load-bearing assumption; report correlation with edit success rate across perturbation radii r_max ∈ {0.05, 0.1, 0.2} |
| Diagnostic metric | Proxy-vs-truth loss gap |L_true' - L_prox'| / max(L_true', L_prox') at each accepted step | Quantifies trust-region effectiveness; report distribution and rejection rate |
| Compute block | Proxy data collection: 4096 renders at ~0.5 s each on 1 GPU = ~0.6 GPU-hours; proxy training: 16M-param U-Net, 100 epochs on 4096 samples, batch size 16, ~2 GPU-hours on 1 A100; Phase 2 per edit: K=8 steps with R_sym evaluation and finite differences, ~8 × (0.5 s render + 0.5 s finite-diff) = ~8 s per edit; on-policy fine-tuning: 100 iterations every 2 accepted steps, ~0.2 s per iteration = ~0.8 s per edit. Total per edit ~10 s. Sensitivity to N: report edit success rate for N ∈ {512, 1024, 4096} to show proxy data efficiency. Fallback if R_sym cannot be batch-parallelized: amortized proxy retraining schedule (fine-tune every 4 accepted steps instead of every 2) with reported degradation. | Demonstrates 3-6 month feasibility and resource requirements |

### Why this setup validates the claim

The central claim is that coupling text semantics, procedural parameters, and layout through a differentiable proxy yields edits that are both semantically aligned and globally consistent, where existing approaches trade one for the other. The regenerate-then-select baseline tests the sub-claim that discrete sampling cannot jointly satisfy semantic and layout constraints without combinatorial search. The VLM-guided procedural baseline tests the sub-claim that gradient signal from rendered geometry is necessary, not merely better prompting. The unconstrained-proxy baseline tests the sub-claim that the layout and parameter priors are load-bearing rather than decorative. The L_par ablation isolates whether symbolic-manifold adherence is doing real work or whether semantic loss alone suffices. The trust-region/on-policy ablation directly tests the load-bearing assumption: if the single in-distribution proxy (no trust region, no refresh) performs as well as full DPC, then the assumption holds and the loop is unnecessary; if it degrades, the loop is load-bearing. The gradient-fidelity diagnostic makes the assumption empirically grounded by measuring cosine agreement between proxy and finite-difference gradients along the actual edit trajectory and correlating it with edit success rate across perturbation radii; this directly tests whether low forward MSE implies faithful gradients. Edit success rate under the consistency check is the right primary metric because it is exactly the predicate the method optimizes and verifies: an edit counts only if it passes the same L_sem + alpha*L_lay threshold, so the metric cannot be gamed by sacrificing layout for semantics or vice versa. This makes the design falsifiable: if the coupling mechanism is real, gains should concentrate on edits where semantic and layout pressures conflict, and gradient-fidelity agreement should predict success.

### Expected outcome and causal chain

**vs. InfiniDream regenerate-then-select** — On a case where the text asks to change one asset's material while keeping neighboring objects fixed, regenerate-then-select produces many candidates where the target asset changes but neighbors drift, because sampling explores the joint space without a layout-preservation term and selection must trade off semantic match against drift. Our method instead backprops semantic loss through the proxy while L_lay penalizes neighbor projection drift, so we expect a clear gap on conflict-heavy edits and near-parity on edits where the target asset is isolated.

**vs. VLM-guided procedural edit (no proxy)** — On a case requiring fine-grained parameter adjustment (e.g., subtle roof pitch change) that the VLM must translate into API calls, the baseline produces coarse or rejected edits because the VLM reasons in discrete token space without seeing the rendered consequence of a parameter delta. Our method instead receives gradient from rendered image to parameter, so we expect the largest gains on edits whose correct parameter change is small and non-obvious from text alone, with parity on edits the VLM can express as a single obvious API call.

**vs. Unconstrained proxy optimization (no L_lay/L_par)** — On a case where the semantic objective can be satisfied by moving the camera or deforming unrelated geometry, the unconstrained variant produces semantically matching but globally inconsistent scenes because nothing anchors non-edited assets or keeps parameters on the symbolic manifold. Our method instead freezes non-edited assets and clips to valid ranges, so we expect the gap to concentrate on scenes with many objects and on edits whose semantic target is ambiguous, with parity on single-object scenes.

**vs. DPC with trust-region and on-policy refresh disabled** — On edits that require moving parameters far from the initial scene (e.g., large material or shape changes), the single in-distribution proxy will be queried off its training distribution and its gradient will be unfaithful, leading to rejected or inconsistent edits. Our method instead uses the trust-region acceptance test and on-policy refresh to reject steps where proxy-vs-truth loss gap exceeds tau=0.1 or gradient cosine agreement falls below 0.7, so we expect the disabled variant to have lower edit success rate concentrated on large-perturbation edits, with parity on small edits near the initial scene. The gradient-fidelity diagnostic should show that cosine agreement drops with perturbation radius for the disabled variant but remains above 0.7 for full DPC.

### What would falsify this idea

If DPC's gain over regenerate-then-select and VLM-guided editing is uniform across conflict-heavy and isolated edits rather than concentrated where semantic and layout pressures conflict, the coupling mechanism is not doing the claimed work. Likewise, if removing L_par produces no degradation on edits requiring symbolic validity, the manifold-adherence claim is unsupported. Additionally, if the trust-region/on-policy ablation shows no degradation and gradient-fidelity cosine agreement is uniformly high (>0.9) across perturbation radii, then the load-bearing assumption holds trivially and the trust-region loop is unnecessary; conversely, if gradient-fidelity agreement is low (<0.5) even with the loop and edit success rate does not correlate with agreement, the proxy-gradient mechanism is fundamentally unfaithful and the method fails.

## References

1. InfiniDream: Open-Ended Photorealistic 3D Worlds Through Procedural Generation
