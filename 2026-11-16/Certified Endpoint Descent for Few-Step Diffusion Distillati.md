# Certified Endpoint Descent for Few-Step Diffusion Distillation

## Motivation

Endpoint distribution matching distillation (eDMD, arXiv:2611.17194) replaces intermediate-state fake-model training with a single shared fake model trained only on the student's final outputs, and argues that the matching term vanishes if and only if the final output distributions align. However, that argument is asymptotic: at finite optimization steps, the per-step correction computed from the shared fake score model is never checked to be a descent direction for the endpoint distribution matching objective, so the correction can increase the endpoint divergence while still appearing to reduce a noisy local estimate. The structural root cause is that the fake score model's outputs are treated as an unbiased reference for the endpoint distribution, but they are only a finite-sample estimate whose variance is never accounted for when the correction is applied.

## Key Insight

The per-step correction in endpoint distribution matching is a valid descent direction for the endpoint KL if and only if it lies inside a variance-normalized trust region around the shared fake score model's current endpoint estimate; enforcing this region per step converts the asymptotic endpoint guarantee into a finite-step monotone descent property without changing the fake model's training distribution.

## Method

```pseudocode
# CED: Certified Endpoint Descent (plug-in for eDMD)
# Inputs: teacher score s_T, student G_theta, shared fake score s_phi (trained ONLY on G_theta(z) final outputs)
# Hyperparams: trust radius rho (e.g., 0.05), variance floor sigma_min^2 (e.g., 1e-4), EMA decay beta=0.99, step count K,
#             certificate batch size M=64 (held-out minibatch for batch-level certificate),
#             confidence level alpha=0.05 (one-sided), shrinkage temperature tau=1.0

for each distillation step:
    z ~ N(0, I); x_hat = G_theta(z)                     # student final output
    # 1. Endpoint matching objective estimate on a fresh minibatch B (size |B|=128)
    L_end = mean_{x_hat in B} [ KL_est( s_T(x_hat) || s_phi(x_hat) ) ]
    # 2. Per-sample correction direction from fake-vs-teacher discrepancy
    for i in B:
        g_i = s_T(x_hat_i) - s_phi(x_hat_i)              # eDMD correction term
        v_i = || g_i ||^2 / (sigma_phi^2(x_hat_i) + sigma_min^2)   # variance-normalized magnitude
    # 3. Batch-level variance-controlled descent certificate (replaces per-sample sign test)
    #    Compute aggregated correction g_bar = mean_i g_i
    #    Compute endpoint objective gradient on a held-out minibatch C (size M=64, disjoint from B)
    #    grad_L_end_C = mean_{x_hat in C} grad_{x_hat} KL_est( s_T(x_hat) || s_phi(x_hat) )
    #    Inner product: ip = <g_bar, grad_L_end_C>
    #    Estimate standard error of ip via bootstrap (B_boot=100 resamples of C)
    #    One-sided lower confidence bound: LCB = ip - z_{1-alpha} * SE_boot
    #    Accept batch update if LCB > 0 AND mean_i v_i <= rho^2
    accept_batch = (LCB > 0) and (mean_i v_i <= rho^2)
    # 4. Apply soft shrinkage weight per sample (replaces hard per-sample mask)
    #    w_i = 1 / (1 + v_i / tau)   # down-weights high-variance samples
    #    If accept_batch is False, set all w_i = 0 (no update)
    L_student = mean_i [ w_i * <g_i, x_hat_i - sg(x_hat_i)> ]
    update theta via grad L_student
    # 5. Refresh fake model ONLY on final outputs (preserves eDMD endpoint reference)
    update phi via score-matching loss on {G_theta(z)} with EMA(beta)
    # 6. Update variance estimate sigma_phi^2 with EMA over accepted samples
    #    sigma_phi^2(x_hat) = EMA_beta( ||s_phi(x_hat) - s_T(x_hat)||^2 )  # residual variance proxy
```

**Why this design.** We chose a batch-level variance-controlled certificate over a per-sample sign test because per-sample gradient signs are dominated by minibatch noise in high-dimensional score matching, making a single-sample sign an unreliable indicator of population descent; the batch-level inner product with a bootstrap confidence bound aggregates evidence across samples and directly tests the expected descent direction. We chose a soft shrinkage weight w_i = 1/(1+v_i/tau) over a hard accept/reject mask because a hard mask discards many corrections early in training when sigma_phi^2 is unstable, while soft weighting down-weights noisy samples without fully discarding them, preserving gradient signal. We chose to keep the fake model trained exclusively on final outputs rather than re-introducing intermediate-state training because that is exactly the structural property eDMD establishes and that we want to certify; re-introducing intermediate states would collapse us back to standard DMD and destroy the endpoint guarantee we are trying to make finite-step. We chose the batch-level inner product with a confidence bound over a pure magnitude test because magnitude alone cannot distinguish a large correct-direction correction from a large wrong-direction one, and the confidence bound accounts for estimation variance. A domain expert would not describe this as a variant of eDMD: eDMD defines the correction and asserts an asymptotic endpoint property, whereas CED leaves the correction unchanged and adds a separate, batch-level certificate that gates whether that correction is admitted, so the mechanism being contributed is the certificate, not the correction.

**Why it measures what we claim.** The variance-normalized magnitude v_i measures the reliability of the fake score model's endpoint reference because the assumption that s_phi(x_hat_i) is an unbiased estimator of the endpoint score with variance sigma_phi^2(x_hat_i) makes v_i a standardized residual; this assumption fails when the fake model has not yet converged on a region of the output manifold, in which case v_i reflects fake-model training error rather than correction reliability. The batch-level inner product <g_bar, grad_L_end_C> with bootstrap confidence bound measures whether the aggregated eDMD correction is a descent direction for the endpoint objective because the assumption that L_end is locally differentiable and the held-out minibatch gradient is an unbiased estimate of the full endpoint KL gradient makes a positive lower confidence bound equivalent to first-order descent with high probability; this assumption fails under high minibatch variance or when the held-out set C is not representative, in which case the bound reflects estimation noise rather than true descent. The accept_batch flag measures certified descent because the conjunction of the confidence-bound test and the mean variance threshold under the stated assumptions implies L_end decreases on the accepted batch; this fails when the trust radius rho is set larger than the local Lipschitz constant of the endpoint score, in which case the certificate reflects the radius choice rather than actual descent. The soft weight w_i measures per-sample correction reliability because it is a monotone decreasing function of v_i; this fails when sigma_phi^2 is systematically biased (e.g., due to self-referential fake model training), in which case w_i reflects fake-model bias rather than reliability.

## Contribution

["A per-sample variance-aware trust-region certificate that gates eDMD's endpoint correction, turning the asymptotic endpoint matching guarantee into a finite-step accepted-correction set with an explicit accept/reject rule.", "A design principle for endpoint distribution matching: the shared fake score model's variance must be estimated and used to normalize corrections, otherwise the correction signal is not a reliable descent direction at finite optimization steps.", 'An online variance estimator for the shared fake score model that is updated only on accepted final outputs, preserving the endpoint-only training distribution of eDMD.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ImageNet 64×64 | Matches eDMD/DisBack endpoint-fidelity benchmarks |
| Dataset | ImageNet 256×256 | Tests scalability and significance per reviewer |
| Primary metric | FID at fixed few-step budget (1, 2, 4, 8 steps) | Detects endpoint distribution mismatch directly |
| Secondary metric | Number of distillation steps to reach target FID (e.g., FID=2.0 on ImageNet 64) | Measures sample-efficiency of distillation |
| Baseline | eDMD (uncertified correction) | Isolates effect of gating the correction |
| Baseline | DisBack | Tests trajectory-based alternative to endpoint gating |
| Baseline | Standard DMD | Tests whether endpoint reference alone suffices |
| Baseline | eDMD + trust-region gate on intermediate-state correction | Tests whether endpoint reference is necessary for certification |
| Ablation-of-ours | CED without batch-level certificate (always accept) | Isolates effect of certificate from soft weighting |
| Ablation-of-ours | CED with hard per-sample mask (original design) | Tests batch-level vs per-sample certificate |
| Ablation-of-ours | CED without soft shrinkage (w_i=1 for all accepted) | Isolates soft weighting from certificate |

### Why this setup validates the claim

The central claim is that gating eDMD's correction with a batch-level variance-controlled descent certificate and soft shrinkage yields certified endpoint descent that an ungated correction cannot guarantee. Comparing against eDMD isolates exactly the certificate, since the correction term is held fixed; any gap must come from the certificate and weighting. DisBack tests whether a trajectory-based remedy achieves the same endpoint fidelity, distinguishing our endpoint-local certificate from trajectory recording. Standard DMD tests whether the endpoint reference alone (without our certificate) is sufficient. The eDMD + trust-region gate on intermediate-state correction baseline tests whether the endpoint reference is what makes certification meaningful, as suggested by the reviewer. FID on a fixed few-step budget is the right metric because the predicted failure mode — accepted-but-non-descending corrections inflating endpoint mismatch — manifests as distributional error. The number of steps to target FID measures sample-efficiency, framing significance beyond marginal FID delta. The no-certificate ablation isolates whether the certificate itself drives any gain, making the mechanism falsifiable. The hard-mask ablation tests whether the batch-level certificate with soft weighting is necessary over the original per-sample design. The no-shrinkage ablation isolates the effect of soft weighting from the certificate. Additionally, a diagnostic experiment logs per-sample v_i, batch acceptance rate, and the actual change in held-out endpoint KL after each accepted update, reporting the fraction of accepted batches that increased endpoint KL to empirically bound the false-accept rate.

### Expected outcome and causal chain

**vs. eDMD (uncertified correction)** — On a case where the fake score is still poorly fit in a high-variance region of the student's output manifold early in training, eDMD applies the full correction g_i unconditionally, so a large wrong-direction update pushes the student away from the endpoint target and FID stalls or oscillates because its mechanism has no per-sample reliability gate. Our method instead down-weights high-variance samples via w_i and rejects batches with low confidence, so only certified corrections update theta, and we expect a visible FID gap concentrated in early-training and high-variance subsets that narrows to parity once sigma_phi^2 stabilizes. The diagnostic false-accept rate should be lower for CED than for eDMD's implicit acceptance.

**vs. DisBack** — On a case where the student's intermediate distributions diverge from the teacher's convergence trajectory, DisBack spends capacity recording and backtracking a path that is only an implicit proxy for the endpoint, so endpoint fidelity improves slowly because its mechanism optimizes trajectory alignment rather than the endpoint KL directly. Our method instead measures and certifies descent on the endpoint objective itself, so we expect faster FID reduction at equal step budget and a smaller residual endpoint mismatch, with the gap largest where trajectory recording is least representative of the endpoint. The step-efficiency metric should show CED reaching target FID in fewer steps.

**vs. Standard DMD** — On a case where the fake model is trained across intermediate states, the student's update signal is entangled with intermediate distributions, so even after final outputs match the teacher the matching term remains nonzero and FID plateaus above the teacher's endpoint error because its mechanism cannot certify endpoint-only descent. Our method keeps the fake model on final outputs and gates corrections, so we expect a lower FID floor and a flatter plateau, with the advantage appearing as a lower asymptote rather than a steeper early curve.

**vs. eDMD + trust-region gate on intermediate-state correction** — On a case where the fake model is trained on intermediate states, the certificate may accept corrections that reduce intermediate-state mismatch but not endpoint KL, so FID improvement is limited because the gate is applied to a non-endpoint objective. Our method applies the same certificate to the endpoint correction, so we expect better endpoint fidelity, demonstrating that the endpoint reference is necessary for certification to be meaningful.

**vs. CED without batch-level certificate (always accept)** — On a case where the fake score is noisy, always accepting corrections leads to divergent updates and higher FID, because the soft weighting alone cannot prevent wrong-direction updates. Our method's certificate rejects batches with low confidence, so we expect lower FID and a lower false-accept rate, confirming the certificate's role.

**vs. CED with hard per-sample mask (original design)** — On a case where per-sample sign tests are noisy, the hard mask discards many corrections and may accept spurious ones, leading to slower convergence or instability. Our batch-level certificate aggregates evidence and uses soft weighting, so we expect faster convergence and lower FID variance across seeds.

**vs. CED without soft shrinkage (w_i=1 for all accepted)** — On a case where some samples have high v_i but the batch is accepted, applying full weight to all samples may still introduce noise. Our soft shrinkage down-weights high-variance samples, so we expect slightly better FID and more stable training.

### What would falsify this idea

If the FID gain over eDMD is uniform across low- and high-variance subsets rather than concentrated where the predicted fake-model reliability failure occurs, the certificate is not doing what we claim. Likewise, if the diagnostic false-accept rate (fraction of accepted batches that increased held-out endpoint KL) is not lower for CED than for eDMD, the certificate is not effectively gating descent.

## References

1. eDMD: Endpoint Distribution Matching for Few-Step Diffusion Distillation
