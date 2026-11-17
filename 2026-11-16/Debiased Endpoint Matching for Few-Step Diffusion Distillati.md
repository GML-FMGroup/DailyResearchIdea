# Debiased Endpoint Matching for Few-Step Diffusion Distillation

## Motivation

Endpoint distribution matching distillation (eDMD) replaces intermediate-state fake-model training with a single shared fake score model trained only on the student's final outputs, so that the matching term vanishes only when the student's endpoint distribution equals the teacher's. However, that guarantee is asymptotic: the shared fake model is fit to the same finite batch of student endpoints used to compute the student's update, so the fake score inherits the batch's empirical sampling noise and the matching signal is biased exactly where it matters — the tail of the endpoint distribution that few-step students under-cover. DisBack (Distribution Backtracking) tries to fix score mismatch by importing the teacher's full convergence trajectory, but it likewise never establishes identifiability of the teacher endpoint distribution from finite student samples, so both branches of the field leave the distillation target only asymptotically identified.

## Key Insight

The finite-sample bias of eDMD's shared fake score is a plug-in bias that is structurally removable: splitting student endpoints into a fake-fitting fold and a disjoint matching fold, then adding a cross-fold correction term, yields an endpoint matching functional whose expectation is stationary at the teacher endpoint distribution for any finite sample size, so identifiability of the teacher endpoint no longer depends on the fake model being asymptotically correct.

## Method

**What it is.** Debiased Endpoint Matching Distillation (DEMD) is a few-step distillation procedure that takes a pretrained multistep teacher `T`, a student generator `G_θ`, and a finite batch of student endpoint samples, and returns an updated `θ` together with a shared fake score model `s_φ`. Its distinguishing object is a sample-split, cross-fold-corrected endpoint matching loss that replaces eDMD's plug-in matching term.

**How it works.**

```pseudocode
input: teacher T, student G_θ, fake score s_φ (2-layer MLP, hidden=256, GeLU),
       batch size B=256, split ratio ρ=0.5 (so |A|=|C|=128),
       kernel bandwidth h=0.1 (Gaussian kernel k(a,b)=exp(-||a-b||²/(2h²))),
       correction weight λ (default 1.0), steps K=50k
for each distillation step:
    # 1. Draw student endpoints (one or few student steps)
    z ~ prior;  x_G = G_θ(z)                      # B endpoints
    x_T = teacher_endpoint(z)                     # paired teacher endpoints

    # 2. Sample-splitting: disjoint folds
    (A, C) = split({1..B}, ρ)                     # |A| = |C| = B/2 = 128

    # 3. Fit shared fake score on fold A ONLY
    φ ← φ - η ∇_φ  mean_{i∈A} || s_φ(x_G^i) - ∇log p_G(x_G^i) ||²

    # 4. Endpoint matching on fold C with cross-fold correction
    for i∈C:
        r_i      = s_φ(x_G^i) - s_T(x_G^i)        # fake-teacher discrepancy
        r_i^corr = r_i - (1/|A|) Σ_{j∈A} k(x_G^i, x_G^j) · r_j
        g_i      = r_i^corr · ∂x_G^i/∂θ
    θ ← θ - γ · mean_{i∈C} g_i

    # 5. Recompute correction weight λ by one-step held-out check on A
    λ ← argmin_λ  mean_{i∈A} || s_φ(x_G^i) - λ·r_i^corr - s_T(x_G^i) ||²
```

**Why this design.** We chose strict sample splitting (fold A fits the fake model, fold C drives the student) over eDMD's single shared batch because the plug-in bias of a fake model evaluated on its own training points is first-order in the batch size, accepting the cost that the fake model sees only half the endpoints per step and therefore has higher variance; the cross-fold correction recovers the lost first-order accuracy without reintroducing the bias. We chose a kernel-weighted residual correction over a learned bias head because the bias here is a known function of the empirical endpoint measure (the fake score is a plug-in estimator of the student score), accepting the cost that the kernel bandwidth must be set but gaining a correction whose expectation is analytically stationary at the teacher endpoint. We chose to re-estimate the scalar correction weight λ on the held-out fold A rather than fix it at 1 because the magnitude of the plug-in bias depends on how far the student currently is from the teacher, accepting a small extra forward pass per step; a domain expert would not describe this as a variant of eDMD or DisBack because neither method separates the fold used to fit the fake reference from the fold used to update the student, and the cross-fold correction term has no counterpart in either.

**Why it measures what we claim.** The cross-fold-corrected residual `r_i^corr` measures the teacher–student endpoint discrepancy net of finite-sample plug-in bias because the kernel-weighted mean over fold A estimates the conditional expectation of the fake-score residual given the endpoint, and subtracting it removes the component of the residual that is explained by the fake model's own fitting noise; this assumption fails when the kernel bandwidth is too small relative to the endpoint spread, in which case `r_i^corr` reflects nearest-neighbor noise rather than bias. The split-ratio ρ measures the bias–variance trade-off of the fake reference because fold A's size controls the variance of `s_φ` while fold C's size controls the variance of the student update; this assumption fails when B is so small that either fold is under-powered, in which case the correction itself becomes the dominant noise source. The scalar λ measures the residual bias magnitude because it is the least-squares coefficient of the correction term against the held-out residual; this assumption fails when the bias is not approximately scalar-multiplicative across the endpoint distribution, in which case λ absorbs only the mean bias and leaves distributional bias uncorrected.

**Load-bearing assumption (explicit).** The correction assumes that the fake-score fitting noise is approximately a conditional mean function of the student endpoint `x_G` that a Gaussian kernel smoother with bandwidth `h=0.1` over `|A|=128` samples can identify and cancel. This assumption fails when the endpoint space is high-dimensional (e.g., ImageNet 64×64, effectively hundreds of dimensions) and the kernel bandwidth is too small relative to the endpoint spread, in which case the correction reflects nearest-neighbor noise rather than bias; it also fails when the student under-covers the teacher's support, because the fold-A kernel regression then estimates the student-conditional mean of the residual, not the teacher-conditional mean, so subtracting it removes signal rather than only fitting noise. We do not claim this assumption is verified; it is the premise that the calibration experiment below is designed to test.

## Contribution

(1) A debiased endpoint matching estimator for few-step diffusion distillation that combines sample splitting with a kernel-weighted cross-fold correction and a held-out scalar recalibration, giving a matching signal whose expectation is stationary at the teacher endpoint for finite student batches. (2) The design principle that the fake reference and the student update must be computed on disjoint student-endpoint folds, with the correction term carrying the bias information across folds, so that endpoint identifiability does not require an asymptotically accurate fake model. (3) A diagnostic protocol that reports the gap between plug-in and debiased matching signals as a function of batch size, exposing when eDMD-style distillation is bias-limited rather than variance-limited.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | ImageNet 64×64 | Standard few-step distillation benchmark with dense coverage |
| Primary metric | FID at fixed few-step budget (1, 2, 4 steps) | Directly measures endpoint distribution match |
| Baseline | eDMD (endpoint matching) | Isolates the plug-in bias we claim to correct |
| Baseline | DisBack (trajectory backtracking) | Tests whether trajectory methods already absorb this bias |
| Baseline | CED (contrastive energy distillation) | Tests simple one-step alternative without shared fake model |
| Ablation-of-ours | DEMD without cross-fold correction (λ=1, no kernel term) | Isolates contribution of the correction term |
| Ablation-of-ours | DEMD with fixed λ=1 but kernel correction on | Isolates the λ re-estimation from the kernel correction |
| Ablation-of-ours | DEMD with kernel bandwidth h ∈ {0.01, 0.05, 0.1, 0.5, 1.0} | Sensitivity to bandwidth; default h=0.1 |
| Ablation-of-ours | DEMD with split ratio ρ ∈ {0.25, 0.5, 0.75} | Sensitivity to fold sizes; default ρ=0.5 |
| Diagnostic | Plug-in bias estimate: mean_{i∈A} ||s_φ(x_G^i) - s_T(x_G^i)||² vs. mean_{i∈C} same, as function of B ∈ {64, 128, 256, 512} and distillation step | Quantifies whether bias is non-trivial fraction of total matching error |
| Compute metric | Wall-clock per distillation step and peak GPU memory for DEMD vs. eDMD | Bounds feasibility and tuning burden |

### Why this setup validates the claim

The central claim is that eDMD's plug-in matching term is first-order biased because the shared fake score is evaluated on the same endpoints it was fit on, and that sample-splitting plus a cross-fold correction removes that bias. eDMD is the direct test: it shares our endpoint-matching target but lacks the fold separation, so any gain over it isolates the bias correction rather than the matching paradigm. DisBack tests a competing account — that trajectory information, not bias correction, drives gains — so if we beat eDMD but not DisBack, the claim is weakened. CED tests whether a simpler one-step objective already avoids the bias, guarding against attributing a generic effect to our mechanism. The ablation (λ fixed at 1, no kernel correction) isolates the correction term from the splitting itself, separating the two design choices we bundled. The bandwidth and split-ratio sweeps test the load-bearing assumption that the kernel smoother can identify the bias: if performance is flat across h and ρ, the correction is not doing the work we claim; if it degrades sharply at small h, the correction reflects nearest-neighbor noise as predicted. The plug-in bias diagnostic directly measures whether the bias is a non-trivial fraction of the total matching error, which is the premise for the significance of the correction. FID at a fixed few-step budget is the right metric because the predicted effect is a distributional endpoint mismatch: bias in the matching direction should manifest as residual distributional error that FID captures, and it should shrink specifically where the bias is largest — when the student is far from the teacher early in distillation and when the fake model is under-fit. A metric measuring only sample quality per image, without distributional comparison, would miss the predicted signature.

### Expected outcome and causal chain

**vs. eDMD** — On a case where the student is early in distillation and far from the teacher, eDMD's fake score is fit and evaluated on the same endpoints, so its residual carries the fake model's own fitting noise, and the student update chases a distorted matching direction. Our method fits the fake score on fold A and corrects fold C residuals by the kernel-weighted mean over A, removing the component explained by fitting noise. We expect a noticeable FID gap concentrated in early-to-mid distillation steps, narrowing to near-parity in late steps once the student is close to the teacher and the bias is small. The plug-in bias diagnostic should show a larger gap between fold-A and fold-C residuals for eDMD than for DEMD, with the gap shrinking as B increases.

**vs. DisBack** — On a case where the teacher's convergence trajectory has sharp intermediate transitions, DisBack trains the student to backtrack intermediate distributions, which can pull the student toward states the endpoint objective does not require, leaving endpoint FID above what pure endpoint matching could achieve. Our method ignores intermediate distributions entirely and corrects only the endpoint residual, so we expect parity or a modest gain on smooth trajectories but a clearer advantage on instances where intermediate and endpoint objectives conflict — visible as a gap on the high-variance subset and parity elsewhere.

**vs. CED** — On a case where the multistep teacher's score is poorly approximated by the contrastive energy objective's auxiliary model, CED's implicit KL optimization inherits that approximation error, producing one-step samples with distributional drift. Our method uses the teacher score directly as the reference and corrects only finite-sample bias, so we expect CED to show larger FID degradation on harder teacher-distribution regions while our method stays closer to the teacher, with the gap widening as the teacher's distribution becomes more multimodal.

### What would falsify this idea

If the gain over eDMD is uniform across distillation stages and endpoint subsets rather than concentrated where the plug-in bias is predicted to be largest (early distillation, under-fit fake model), the bias-correction account is wrong and the improvement must come from something else. Likewise, if removing the cross-fold correction (λ=1, no kernel term) leaves performance unchanged, or if the plug-in bias diagnostic shows the bias is a negligible fraction of total matching error, the correction term is not doing the work we claim.

## References

1. eDMD: Endpoint Distribution Matching for Few-Step Diffusion Distillation
