# Certified Baking: Residual-Corrected MLP Distillation with Held-Out Novel-View Error Certificates

## Motivation

TTB bakes cross-view attention aggregation into a per-scene fast-weight MLP but provides no approximation-error bound, so fidelity is not guaranteed to converge as baking capacity increases — a shared residual gap across TTT3R, Long-LRM++, and PixelSplat, where the baked mapping's deviation from the original decoder-only aggregation is never measured or controlled. The root cause is structural: baking is trained only on the observed context-token pairs with a pointwise regression loss, and the attention weights that define the true aggregation are never exposed to the loss, so worst-case weight mismatch on unseen novel views is invisible to training and unmeasurable at validation.

## Key Insight

Because the baked MLP's deviation from the original cross-view attention aggregation is dominated by attention-weight mismatch on held-out novel views, a residual head trained against the true attention-weight distribution plus a held-out PAC-style tolerance bound yields a per-scene certificate that is monotone in actual novel-view error.

## Method

**What it is.** CBAKE (Certified Baking) is a wrapper around TTB's attention-to-MLP bake that outputs, per scene, (i) a baked MLP augmented with a residual correction head and (ii) a scalar certificate ε_cert with a distribution-shift-aware tolerance bound on the deviation between the baked MLP's novel-view output and the original decoder-only cross-view attention aggregation. The certificate is computed on a held-out novel-view split V_hold that is partitioned by viewpoint displacement from the context trajectory; the bound is an error-vs-displacement envelope (or weighted conformal prediction with likelihood-ratio weights under a bounded covariate-shift assumption) evaluated at the test view's displacement, rather than a single exchangeability-based PAC slack. This makes the certificate degrade gracefully under viewpoint extrapolation instead of silently under-estimating.

**How it works.**
```pseudocode
# Inputs: per-scene context tokens (camera-only r_i, ground-truth g_i),
# decoder-only attention module A (frozen), held-out novel-view split V_hold (|V_hold|=m).
# 1. Bake base MLP f_theta on context pairs (as in TTB):
for step in range(N_bake):
    theta <- theta - lr * grad_theta || f_theta(r_i) - g_i ||^2

# 2. Residual correction head h_phi (2-layer, width 64, GeLU activation), trained on context pairs
#    against the *attention-weight mismatch*, not just output error:
for step in range(N_res):
    # per-context attention weights from A on context tokens
    w_true = softmax(Q(r_i) K(c)^T / sqrt(d))          # A's true aggregation weights
    w_bake = softmax(Q(r_i) K(f_theta(c))^T / sqrt(d)) # weights induced by baked tokens
    L_res = || h_phi(r_i) - (A(r_i) - f_theta(r_i)) ||^2 \
          + lambda * KL(w_true || w_bake)              # lambda=0.5, worst-case weight mismatch
    phi <- phi - lr * grad_phi L_res

# 3. Held-out certificate on V_hold (novel views, not used in steps 1-2):
#    Partition V_hold by viewpoint displacement d_v = distance(r_v, nearest context camera).
#    Fit an error-vs-displacement envelope: for each bin of d_v, compute the max deviation e_v.
#    Alternatively, use weighted conformal prediction with likelihood-ratio weights
#    w_v = p_test(d_v)/p_hold(d_v) estimated from the displacement distributions.
for each v in V_hold:
    y_hat(v) = f_theta(r_v) + h_phi(r_v)
    y_ref(v) = A(r_v)                                  # true cross-view aggregation
    e_v = || y_hat(v) - y_ref(v) ||
    d_v = || r_v - nearest_context_camera ||
# Fit envelope: eps_env(d) = max_{v: d_v <= d} e_v + slack(d), where slack(d) = sqrt(log(1/delta) / (2 * n_d))
# and n_d is the number of held-out views with displacement <= d.
# For a test view with displacement d_test, report eps_cert = eps_env(d_test).
# Also report the empirical max eps_emp = max_v e_v for reference.

# 4. Deploy: render novel views with f_theta + h_phi; report eps_cert per scene (as a function of test displacement).
```

**Why this design.** We chose a residual correction head over retraining the base MLP with a larger capacity because the residual head isolates the *systematic* deviation that TTB's pointwise regression cannot see — the attention-weight mismatch — while leaving the efficient base MLP intact, accepting the cost that the head adds a small per-view forward pass (2-layer, width 64, negligible vs. the 23x speedup TTB reports). We chose a KL penalty on attention weights over output-only residual loss because output error can be small while weights are badly mismatched (cancellation), and the KL term forces the baked tokens to induce the same aggregation geometry as the true decoder, accepting the cost that the KL term requires access to the frozen decoder's Q/K projections at bake time (a one-time cost). We chose a distribution-shift-aware held-out bound over a training-loss proxy because training loss on context pairs cannot certify novel views — the held-out split V_hold is the only surface where the deviation is measured on the target distribution, and partitioning by viewpoint displacement makes the certificate degrade gracefully under extrapolation rather than assuming exchangeability. We accept the cost that ε_cert is conservative when |V_hold| is small (we report the empirical max alongside it). A domain expert would NOT describe this as TTT or TTB with a bigger MLP: the matched prior method (TTB) trains only on context pairs with pointwise regression and reports no deviation bound; CBAKE's load-bearing mechanism is the attention-weight-mismatch penalty plus a held-out distribution-shift-aware certificate, which TTB does not contain.

**Why it measures what we claim.** The KL(w_true || w_bake) term measures *worst-case attention-weight mismatch* because the assumption that the baked tokens' induced softmax weights are a sufficient statistic for the aggregation deviation holds when the decoder's value projection is Lipschitz in token space; this assumption fails when the baked MLP collapses distinct context tokens (modality aliasing), in which case the KL reflects token collision rather than aggregation geometry. The residual head's output h_phi(r_v) measures the *systematic deviation* between baked and true aggregation because the assumption that the deviation is a smooth function of the camera-only rendering token holds when the residual is dominated by low-frequency scene structure; this assumption fails at occlusion boundaries, where the residual becomes high-frequency and h_phi under-corrects. The held-out ε_cert measures *novel-view fidelity with a tolerance bound* because the assumption that the error-vs-displacement envelope fitted on V_hold generalizes to test views with similar displacement holds when the test displacement distribution is covered by V_hold; this assumption fails under large viewpoint extrapolation beyond the held-out displacement range, in which case ε_cert under-estimates true deviation (and we report the empirical max alongside to flag this).

## Contribution

(1) CBAKE, a certified distillation wrapper for TTB's attention-to-MLP bake that adds a residual correction head trained against attention-weight mismatch and emits a per-scene PAC-style tolerance bound on novel-view deviation from the original cross-view attention aggregation. (2) The design principle that penalizing worst-case attention-weight mismatch (not just output regression) is what makes the baked MLP's deviation controllable and the held-out certificate monotone in actual novel-view error. (3) A per-scene certificate protocol with an empirical max and a PAC tolerance term, reported alongside rendering quality and speed, enabling fidelity-aware deployment of baked view synthesizers.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | RealEstate10K novel-view splits | Standard decoder-only NVS benchmark with held-out trajectories |
| Primary metric | PSNR on held-out novel views | Directly reflects novel-view fidelity deviation |
| Secondary metric | Certificate tightness: ratio of ε_cert to empirical max deviation on test views, and coverage of the envelope | Tests whether the distribution-shift-aware bound tracks actual error under extrapolation |
| Baseline | TTB (pointwise bake) | Matched prior method without certificate or KL |
| Baseline | Decoder-only LSVM (full attention) | Reference upper bound on fidelity |
| Baseline | TTT (test-time training) | Alternate efficiency baseline with no bake |
| Ablation-of-ours | CBAKE minus KL penalty | Isolates attention-weight-mismatch contribution |
| Ablation-of-ours | CBAKE with exchangeability-based PAC slack (original bound) | Tests whether the distribution-shift-aware envelope improves certificate validity under extrapolation |

### Why this setup validates the claim

The central claim is that CBAKE's attention-weight-mismatch penalty plus a held-out distribution-shift-aware certificate yield novel-view fidelity closer to the full decoder than TTB's pointwise bake, and that the improvement is attributable to correcting aggregation geometry rather than raw capacity. RealEstate10K novel-view splits provide held-out trajectories that are partitioned by viewpoint displacement, which is exactly the structure the certificate relies on, so PSNR on those views is the right surface to detect the predicted effect. TTB is the matched prior: it tests whether the residual head plus KL actually closes the gap that pointwise regression leaves open. The full decoder-only LSVM establishes the fidelity ceiling, so we can see whether CBAKE narrows the bake-to-decoder gap rather than merely beating TTB. TTT tests whether the gain is specific to the bake-plus-correction design or generic to any efficiency method. The CBAKE-minus-KL ablation is load-bearing: if removing the KL term does not degrade PSNR where attention weights are most mismatched, the mechanism claim is unsupported. The exchangeability-based PAC slack ablation tests whether the distribution-shift-aware envelope is necessary for the certificate to remain valid under extrapolation. The certificate's empirical max is reported alongside so the conservative bound is interpretable.

### Expected outcome and causal chain

**vs. TTB (pointwise bake)** — On a scene with several overlapping context views whose aggregation weights are sharply peaked, TTB's pointwise regression fits each context pair well but induces baked tokens whose softmax weights are diffuse, so novel views show blurred or ghosted structure at depth discontinuities. CBAKE's KL penalty forces the baked tokens to reproduce the decoder's aggregation geometry, and the residual head absorbs the remaining low-frequency deviation, so we expect a visible PSNR gap concentrated on scenes with high weight peakedness, with near-parity on simple single-dominant-view scenes.

**vs. Decoder-only LSVM (full attention)** — On a scene requiring fine cross-view reasoning, the full decoder attends over all context tokens and resolves occlusion correctly, while any bake compresses that aggregation into per-token MLP outputs. CBAKE does not claim to match the decoder; we expect CBAKE to sit between TTB and the decoder, with the bake-to-decoder gap noticeably smaller than TTB's, and the certificate ε_cert to track that residual gap rather than vanish.

**vs. TTT (test-time training)** — On a scene where test-time optimization is given limited steps, TTT underfits and produces noisy renders, while CBAKE's base bake plus small residual head converges quickly. We expect CBAKE to match or exceed TTT fidelity at comparable or lower per-view cost, with the largest advantage on scenes where TTT's optimization is unstable.

**vs. CBAKE with exchangeability-based PAC slack** — On test views with large viewpoint displacement (extrapolation), the exchangeability-based PAC slack is expected to under-estimate the true deviation because it assumes i.i.d. sampling from the deployment distribution, whereas the distribution-shift-aware envelope is expected to provide a valid upper bound (or at least a less anti-conservative estimate) by evaluating the error-vs-displacement envelope at the test displacement. We expect the envelope to have higher coverage of the empirical max on extrapolated views, at the cost of being more conservative on near-trajectory views.

### What would falsify this idea

If CBAKE's PSNR gain over TTB is uniform across scenes rather than concentrated on scenes with high attention-weight peakedness, or if the CBAKE-minus-KL ablation matches full CBAKE, then the attention-weight-mismatch mechanism is not what drives the improvement and the central claim is wrong. Likewise, if ε_cert systematically fails to bound the observed held-out deviation on exchangeable splits, or if the distribution-shift-aware envelope does not improve coverage under extrapolation compared to the exchangeability-based PAC slack, the certificate claim is falsified.

## References

1. TTB: Test-time Baking for Efficient Rendering of Decoder-only View Synthesis Models
