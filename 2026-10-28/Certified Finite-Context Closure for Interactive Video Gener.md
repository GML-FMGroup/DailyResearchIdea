# Certified Finite-Context Closure for Interactive Video Generation via Scene-Consistency Drift Bounds

## Motivation

Existing interactive world models such as Hy-Nomas condition on a bounded, append-only causal KV cache and assume that this finite window is sufficient for long-horizon scene consistency, but they never verify this assumption. The structural problem — finite-context sufficiency — recurs across multiple paper-trees (Hy-Nomas, Yume-1.5, and others), where scene consistency is an unverified side-effect of window size rather than a checked invariant. Without a measurable certificate that the cached window is adequate for the current scene, drift accumulates silently and cannot be detected or corrected until visible degradation occurs.

## Key Insight

The cached KV window satisfies a scene-consistency invariant if and only if a drift statistic computed from the window's own content is monotone in the residual inconsistency of the generated segment; this monotonicity follows from the Lipschitz continuity of the scene-consistency functional with respect to the cached context, which is a structural property of the diffusion backbone's causal attention and not a learned proxy.

## Method

```pseudocode
# Certified Finite-Context Closure (CFCC)
# Inputs: diffusion backbone F with causal attention, current cached KV window W_t,
#         current text/camera condition c_t, scene-consistency functional S(·) (CLIP ViT-B/32 image encoder; descriptor = L2-normalized CLIP embedding of the middle frame of each segment),
#         per-step local Lipschitz estimate L_S(t) (see step 3), tolerance ε (calibrated per scene type, see below),
#         calibration set D_cal of 512 held-out (W, V) pairs with human consistency ratings.
# Output: generated video segment V_t, updated cache W_{t+1}, certificate status cert_t ∈ {valid, invalid}.

# --- Offline calibration (once, before rollout) ---
# 1. Estimate a global reference Lipschitz constant L_S^global by finite-difference perturbation of W on D_cal:
#    L_S^global = max_{(W,V) in D_cal} ||S(V) - S(V')||_2 / ||W - W'||_2, where W' = W + δ, δ ~ N(0, σ^2 I), σ = 0.01 * ||W||_2.
# 2. Calibrate ε via conformal prediction on D_cal: for each scene type g ∈ {static, fast-pan, slow-zoom},
#    compute nonconformity scores s_i = L_S^global * d_i for calibration examples, set ε_g = the ⌈(1-α)(n+1)⌉-th smallest score
#    with target miscoverage α = 0.1 (i.e., 90% coverage). Report ε_g per scene type; also report a single ε* = max_g ε_g.

# --- Per-step rollout ---
1. Generate segment V_t = F(· | W_t, c_t) via diffusion sampling (fixed number of steps, e.g., 50).
2. Compute scene-consistency drift statistic:
   d_t = max_{k in W_t} || S(V_t) - S(V_k) ||_2 / (1 + ||t - k||_2)   # weighted max over cached segment descriptors
3. Compute per-step local Lipschitz estimate L_S(t) over the actual perturbation ball used at inference:
   - Draw N=16 perturbations δ_j ~ N(0, σ_t^2 I) with σ_t = 0.01 * ||W_t||_2.
   - For each j, compute W_t^{(j)} = W_t + δ_j and V_t^{(j)} = F(· | W_t^{(j)}, c_t) (same diffusion steps).
   - L_S(t) = max_j ||S(V_t) - S(V_t^{(j)})||_2 / ||W_t - W_t^{(j)}||_2.
   - Also compute a probabilistic bound: L_S^{prob}(t) = L_S(t) + κ * std_j(ratio_j), with κ = 2.0 (≈95% one-sided).
4. Compute certificate bound:
   cert_t = L_S^{prob}(t) * d_t   # probabilistic bound with explicit failure probability ≤ 5% under the perturbation model
5. If cert_t <= ε_g (where g is the scene type detected from c_t via a fixed rule: static if camera motion < 5°/frame, fast-pan if > 30°/frame, else slow-zoom):
      cert_t = valid; W_{t+1} = append(W_t, V_t)   # standard append-only update
   Else:
      cert_t = invalid; trigger context-window refresh:
         a. Re-encode the last M segments (M = 4) from raw frames with the current text/camera condition to produce a refreshed cache W'_t (discard old KV entries).
         b. Re-generate V_t' = F(· | W'_t, c_t) with the same diffusion steps.
         c. Recompute cert_t' = L_S^{prob}(t) * d_t' with W'_t; if still invalid, expand window size by factor 1.5 and repeat (max 2 expansions).
         d. Set W_{t+1} = append(W'_t, V_t') if valid, else fallback to W_t with a warning.
6. Return V_t (or V_t'), W_{t+1}, cert_t.
```

**Why this design**
We chose a drift statistic based on the maximum weighted descriptor difference over the cached window rather than a learned router or a simple average because the maximum is the sufficient statistic for the worst-case violation of the scene-consistency invariant under the Lipschitz assumption, accepting the cost that it is sensitive to outlier segments (which we mitigate by the temporal weighting). We chose to estimate the Lipschitz constant L_S per-step via finite-difference perturbation of the cache rather than learning it end-to-end because a learned constant would reintroduce the unverified assumption we aim to eliminate, accepting the cost that L_S may be conservative for out-of-distribution scenes. We chose a refresh mechanism that re-encodes raw frames with the current condition rather than simply truncating the cache because truncation discards information without verifying that the remaining window is adequate, accepting the cost of extra computation only when the certificate fails. We chose a hard threshold ε rather than a soft penalty because the certificate must yield a binary decision to trigger refresh, accepting the cost that ε requires tuning per scene type (calibrated via conformal prediction on 512 held-out pairs). This design is not a variant of Hy-Nomas's passive cache or Yume-1.5's compression because those methods never compute a certificate from the window content; our mechanism couples the window to an invariant via a monotone statistic, which is absent in all prior interactive world models.

**Why it measures what we claim**
The computational quantity d_t (maximum weighted descriptor difference) measures residual scene inconsistency because the scene-consistency functional S is Lipschitz continuous with respect to the cached context under the assumption that the diffusion backbone's causal attention preserves the metric structure of the context; this assumption fails when the backbone's attention is not Lipschitz (e.g., due to softmax saturation), in which case d_t reflects only descriptor-space drift, not true inconsistency. The quantity cert_t = L_S^{prob}(t) * d_t measures the certified bound on scene-consistency violation because L_S^{prob}(t) is estimated by finite-difference perturbation over the actual inference perturbation ball and includes a 2-std margin, which assumes local linearity of S; this assumption fails when the perturbation magnitude is too large, in which case L_S^{prob}(t) overestimates and cert_t becomes conservative. The refresh trigger (cert_t > ε_g) measures the necessity of context revision because ε_g is set by conformal prediction to achieve 90% coverage on held-out human ratings; this assumption fails when the scene's natural variability exceeds ε_g, in which case the trigger fires spuriously. Together, these quantities operationalize the meta-level concept of finite-context sufficiency: the window is sufficient iff cert_t ≤ ε_g, and the monotonicity of d_t in residual inconsistency follows from the Lipschitz property.

## Contribution

['A certified finite-context closure mechanism (CFCC) that computes a per-scene context-adequacy certificate from the cached KV window and triggers context refresh only when the certificate fails, converting the finite-context sufficiency assumption into a checkable invariant.', "A concrete design principle: the certificate is a monotone function of the cached window's scene-consistency drift, derived from the Lipschitz continuity of the scene-consistency functional with respect to the context, and the refresh mechanism re-encodes raw frames rather than truncating the cache.", 'An open-source implementation of CFCC integrated with the Hy-Nomas backbone, including the offline Lipschitz estimation procedure and the certificate computation, enabling reproducible evaluation of finite-context sufficiency in interactive video generation.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Interactive scene navigation with scripted camera/text prompts (3 scene types: static, fast-pan, slow-zoom; 50 sequences per type, 200 frames each) | Provides controlled drift triggers and ground-truth consistency |
| Primary metric | Certified violation rate vs. human scene-consistency rating (5-point Likert, 3 annotators, majority vote) | Directly tests certificate validity and practical relevance |
| Secondary metric | False-positive rate (certificate invalid but human rating ≥4) and false-negative rate (certificate valid but human rating ≤2) per scene type | Tests calibration and scene-specificity |
| Baseline | Hy-Nomas (passive cache) | Tests whether certification improves over no verification |
| Baseline | Yume-1.5 (compressed history) | Tests whether certification beats lossy compression |
| Baseline | Truncation-only refresh | Tests whether re-encoding raw frames adds value over truncation |
| Ablation-of-ours | CFCC without refresh (certificate only) | Isolates the contribution of the refresh mechanism |
| Ablation-of-ours | CFCC with global L_S (offline finite-difference) instead of per-step L_S^{prob}(t) | Tests whether per-step local estimate is necessary |

### Why this setup validates the claim

The central claim is that a Lipschitz-certified drift statistic on the cached window can detect and repair scene-consistency violations that passive or compressed-cache baselines miss. The dataset supplies controlled interactive sequences with known drift triggers (camera rotations, text condition shifts) across three scene types, so we can measure both the certificate's binary decisions and the actual consistency of generated frames. The primary metric—certified violation rate versus human scene-consistency rating—directly tests whether the certificate's valid/invalid decisions align with perceptible inconsistency, which is the operational definition of the invariant. Hy-Nomas tests the sub-claim that passive caching without any certificate accumulates undetected drift; Yume-1.5 tests whether compression alone suffices; truncation-only refresh tests whether re-encoding raw frames with current condition is necessary. The ablation (certificate without refresh) isolates whether the refresh mechanism actually repairs violations or whether the certificate merely flags them. The second ablation (global L_S) tests whether the per-step local Lipschitz estimate is needed to maintain calibration. If the certificate is valid, then on subsets where baselines fail, our method should show a measurable reduction in certified violation rate that correlates with human ratings, and the ablation should show the certificate fires but consistency remains low—proving the refresh is essential.

### Expected outcome and causal chain

**vs. Hy-Nomas (passive cache)** — On a case where a rapid camera pan introduces a new scene region not present in the cached window, Hy-Nomas appends the new segment without any verification, so the KV cache accumulates inconsistent descriptors and subsequent generations drift. Our method computes d_t over the cached window and triggers a refresh when cert_t exceeds ε_g, re-encoding the last M segments with the current condition. We expect a noticeable gap in certified violation rate on high-drift subsets (e.g., fast pans) but parity on low-drift subsets (static scenes).

**vs. Yume-1.5 (compressed history)** — On a case where the scene gradually changes over many steps (slow zoom), Yume-1.5's compression discards fine-grained context that is necessary to maintain consistency, leading to subtle but accumulating descriptor drift that its compression scheme cannot detect. Our method's d_t is computed over the full cached window (before any compression), so the certificate fires when drift exceeds ε_g, and the refresh re-encodes raw frames to restore lost detail. We expect our method to show lower violation rates on gradual-change subsets, with the gap widening as the number of steps since last refresh increases.

**vs. Truncation-only refresh** — On a case where the cache contains an outlier segment (e.g., a transient occlusion), truncation simply removes the oldest entries but does not verify that the remaining window is sufficient, so the outlier may persist or the truncation may discard useful context. Our method's refresh re-encodes the last M segments from raw frames with the current condition, producing a cache that is both consistent and sufficient. We expect truncation-only to have higher violation rates on outlier-heavy subsets, while our method maintains lower rates, with the gap concentrated on sequences containing transient occlusions.

**vs. CFCC without refresh (certificate only)** — On a case where the certificate correctly fires (cert_t > ε_g) but no refresh is performed, the cache remains inconsistent and subsequent generations continue to drift. Our method's refresh repairs the cache by re-encoding raw frames with the current condition. We expect the ablation to show that the certificate fires but consistency remains low, confirming that certification must be coupled with repair.

**vs. CFCC with global L_S** — On a case where the local Lipschitz constant of S∘F varies significantly across the rollout (e.g., due to softmax saturation in fast-pan scenes), a single global L_S may be too conservative (causing spurious refreshes) or too loose (missing violations). Our per-step L_S^{prob}(t) adapts to the local perturbation ball. We expect the global-L_S ablation to have higher false-positive and false-negative rates, especially on fast-pan scenes, demonstrating the necessity of per-step estimation.

### What would falsify this idea

If the certified violation rate does not correlate with human scene-consistency ratings (e.g., certificate says valid but humans perceive inconsistency, or vice versa) on the controlled drift subsets, then the Lipschitz bound fails to operationalize scene consistency. Additionally, if the ablation without refresh achieves the same consistency as full CFCC, then the refresh mechanism is unnecessary and the certificate alone is sufficient—contradicting the claim that certification must be coupled with repair.

## References

1. Hy-Nomas: Towards Free-form Interactive World Generation
2. Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield
3. Yume-1.5: A Text-Controlled Interactive World Generation Model
