# Trajectory-Aware Curvature Tracking for Preservation-Constrained Adaptation

## Motivation

Preservation-constrained adaptation methods such as CurvDelta construct a low-curvature subspace from the pre-adaptation geometry alone, so the preservation-loss invariance they certify is local and static; as the update moves away from the pre-adaptation point, the K-FAC factors and the factorized projection no longer reflect the curvature actually experienced, and the preservation loss can drift outside the intended tolerance. This is a meta-level failure recurring across preservation-aware PEFT: the constraint is certified at a single point rather than along the optimization path, so the guarantee silently expires as soon as the parameters move.

## Key Insight

The preservation-loss drift along an adaptation trajectory is governed by the curvature of the preservation constraint evaluated at the current parameters, so re-estimating the K-FAC factors and reprojecting at each step converts a one-shot local invariance certificate into a trajectory-level invariance certificate whose radius shrinks exactly when the observed curvature drift indicates the old subspace is stale.

## Method

**What it is.** TACT (Trajectory-Aware Curvature Tracking) is a preservation-constrained fine-tuning procedure that augments a CurvDelta-style low-curvature parameterization with online K-FAC re-estimation, a reprojection trigger, and a certified invariance radius computed from the measured curvature drift. Inputs: pretrained weights $\theta_0$, preservation loss $\mathcal{L}_{\text{pres}}$ (e.g., next-token loss on a retention corpus), adaptation loss $\mathcal{L}_{\text{task}}$, target tolerance $\epsilon$, tracking interval $T=100$ steps, drift threshold $\delta=0.01$ (calibrated on a held-out retention batch as described below), rank $r=8$. Outputs: adapted weights $\theta^\star$ and a per-step certificate $r_t$ bounding the preservation-loss increase attributable to the update.

**How it works.**

```pseudocode
# Initialization (t=0)
F_0 = KFAC_Factors(theta_0, L_pres)          # block-diagonal Kronecker factors, block size = layer
P_0 = FactorizedProjection(F_0, rank=r=8)    # low-curvature subspace, rank r
r_0 = CertifiedRadius(F_0, P_0, epsilon)     # initial invariance radius

# Calibration of delta (once, before training):
#   Draw 512 examples from retention corpus, compute ||(I-P_0) grad L_pres|| at theta_0,
#   set delta = 95th percentile of this residual norm over the 512 examples.

for t = 1..N:
    # 1. Task step in the current low-curvature subspace
    delta_task = -lr * P_{t-1} @ grad(L_task(theta_{t-1}))
    theta_t = theta_{t-1} + delta_task

    # 2. Curvature drift measurement (every T=100 steps)
    if t mod T == 0:
        F_t = KFAC_Factors(theta_t, L_pres)
        # Direct projection-residual staleness measure (replaces Frobenius factor drift):
        g_pres = grad(L_pres(theta_t))            # true preservation gradient, full model
        residual = ||(I - P_{t-1}) @ g_pres||_2   # component outside current subspace
        drift = residual                          # trigger quantity

        # 3. Reprojection when residual exceeds calibrated threshold delta
        if drift > delta:
            P_t = FactorizedProjection(F_t, rank=r=8)
            # 4. Certified radius from residual and current curvature
            r_t = CertifiedRadius(F_t, P_t, epsilon, drift)
            # 5. If radius below tolerance, roll back and shrink step
            if r_t < epsilon:
                theta_t = theta_{t-1}       # reject step
                lr = lr * 0.5                # shrink learning rate
        else:
            P_t = P_{t-1}
            r_t = r_{t-1}
    else:
        P_t = P_{t-1}
        r_t = r_{t-1}

return theta_N, {r_t}
```

**Why this design.** We chose online K-FAC re-estimation every $T=100$ steps over a fully static pre-adaptation estimate because the preservation-loss curvature is a function of the current parameters and a static estimate becomes stale as soon as the update leaves the local neighborhood, accepting the cost of $O(1/T)$ extra curvature evaluations and factorizations. We chose a drift-triggered reprojection over unconditional reprojection at every step because unconditional reprojection would make the projection itself a moving target and destabilize the task objective, accepting the cost that between triggers the subspace may be slightly stale. We chose a rollback-and-shrink rule over a penalty term when the certified radius falls below tolerance because a penalty term only softly discourages drift whereas rollback enforces the tolerance hard, accepting the cost of occasionally discarding a productive step. We chose the projection-residual norm $\|(I-P_{t-1})\nabla\mathcal{L}_{\text{pres}}(\theta_t)\|_2$ as the drift measure over Frobenius drift on the K-FAC factors because the residual directly measures the component of the true preservation gradient that the current subspace fails to capture, is invariant to eigenbasis rotation and factor rescaling, and remains valid even when K-FAC's block-diagonal approximation is inaccurate, accepting the cost of one full preservation-gradient evaluation per tracking interval. A domain expert would not describe TACT as a variant of static CurvDelta because CurvDelta's central object — a single pre-adaptation subspace — is replaced by a time-indexed sequence of subspaces with an explicit drift-triggered update rule and a per-step certificate, which is a different algorithmic contract.

**Why it measures what we claim.** The projection-residual norm $\|(I-P_{t-1})\nabla\mathcal{L}_{\text{pres}}(\theta_t)\|_2$ measures subspace staleness because, under the assumption that the preservation loss is locally quadratic with block-diagonal Kronecker structure, the change in the projection residual is bounded by the change in the factors; this assumption fails when the preservation loss has strong off-block coupling (e.g., cross-layer attention interactions), in which case the residual reflects only the component of the true gradient outside the current subspace and may underestimate staleness if the gradient itself is small; however, because the residual is evaluated on the true preservation gradient rather than on the approximate factors, it remains a direct measurement of the quantity the certificate depends on. The certified radius $r_t$ measures the trajectory-level invariance of the preservation loss because, under the assumption that the preservation loss is $L$-smooth with $L$ estimated by the K-FAC factors, the loss increase is bounded by $\tfrac{L}{2}\|\Delta\theta\|^2$ projected onto the complement of the low-curvature subspace; this assumption fails when the smoothness constant changes faster than the tracking interval, in which case $r_t$ reflects the smoothness at the last measurement rather than the worst case over the interval. The reprojection trigger $\text{residual} > \delta$ measures whether the current subspace still certifies the tolerance because, under the assumption that the certificate is monotone in residual, exceeding $\delta$ implies the certificate has degraded; this assumption fails when the residual is driven by directions orthogonal to the update, in which case the trigger fires unnecessarily and the method pays reprojection cost without a corresponding preservation benefit.

## Contribution

['A trajectory-aware preservation-constrained adaptation procedure (TACT) that re-estimates K-FAC factors at a fixed interval, triggers reprojection on measured curvature drift, and emits a per-step certified invariance radius for the preservation loss.', 'A drift-triggered reprojection and rollback rule that converts a static local invariance certificate into a trajectory-level tolerance-maintenance mechanism, with an explicit coupling between measured curvature drift and the certified radius.', 'An empirical characterization of how curvature drift along the adaptation trajectory relates to preservation-loss degradation across capability-learning and selective-unlearning tasks.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Knowledge unlearning + math reasoning retention corpus | CurvDelta's own preservation-constrained tasks; drift is measurable |
| Primary metric | Retention-loss increase at matched task gain | Directly tests the certified-radius claim |
| Baseline | Static CurvDelta | Tests whether online re-estimation beats one-shot subspace |
| Baseline | LoRA-Dash | Tests against task-direction PEFT lacking preservation certificate |
| Baseline | LowRA | Tests against quantization-based efficiency without curvature tracking |
| Ablation-of-ours | TACT with drift threshold δ = ∞ (never reproject) | Isolates the reprojection trigger's contribution |
| Ablation-of-ours | TACT with Frobenius factor drift trigger (original) | Tests whether projection-residual trigger is necessary |
| Diagnostic | Off-block coupling measurement on retention corpus | Quantifies K-FAC approximation error to validate residual trigger |

### Why this setup validates the claim

The central claim is that a time-indexed, drift-triggered low-curvature subspace certifies preservation better than a single static subspace at comparable task performance. Static CurvDelta is the sharpest test of that sub-claim: it shares our low-curvature parameterization but never re-estimates, so any gap between it and TACT is attributable specifically to online K-FAC re-estimation and reprojection rather than to the curvature machinery both share. LoRA-Dash tests a different sub-claim — that preservation-aware curvature tracking beats maximizing task-specific directions — and LowRA tests whether the preservation benefit survives under aggressive efficiency pressure. The δ = ∞ ablation is the crucial control: it keeps our parameterization and certificate but removes the trigger, so if TACT's gain over static CurvDelta survives this ablation, the gain is not coming from reprojection at all and our causal story collapses. The Frobenius-trigger ablation isolates whether the projection-residual trigger (which directly measures staleness) is necessary over the original factor-drift heuristic. The off-block coupling diagnostic measures the Frobenius norm of the neglected inter-block terms in the K-FAC approximation on the retention corpus, providing a quantitative check on when the residual trigger is expected to outperform the factor-drift trigger. Retention-loss increase at matched task gain is the right primary metric because it holds task utility fixed, forcing the comparison onto the preservation axis where our mechanism is supposed to act; a metric that rewarded task fit would confound the two.

### Expected outcome and causal chain

**vs. Static CurvDelta** — On a case where the unlearning update pushes parameters far enough that the pre-adaptation low-curvature subspace no longer aligns with the current preservation Hessian, static CurvDelta keeps projecting onto a stale subspace and leaks preservation loss along newly curved directions, because its single pre-adaptation factorization cannot see curvature drift. Our method re-estimates K-FAC factors every T=100 steps, computes the projection residual $\|(I-P_{t-1})\nabla\mathcal{L}_{\text{pres}}(\theta_t)\|_2$, detects when it exceeds the calibrated δ, and reprojects onto the current low-curvature subspace, so we expect static CurvDelta to show a visibly larger retention-loss increase concentrated in the later portion of adaptation, while TACT stays near the tolerance band with parity on early steps.

**vs. LoRA-Dash** — On a case where task-specific directions require updates that happen to lie along high-curvature preservation directions, LoRA-Dash amplifies those directions because its objective rewards task-direction impact without a preservation constraint, producing strong task gain but disproportionate retention degradation. Our method projects each task step through the certified low-curvature subspace and rolls back steps whose radius falls below ε, so we expect LoRA-Dash to lead on raw task fit but lose on the retention-at-matched-gain metric, with the gap widening on the unlearning subset where preservation pressure is highest.

**vs. LowRA** — On a case where ultra-low-bit quantization coarsens the effective update geometry, LowRA's fine-grained quantization mapping introduces preservation drift that its precision assignment does not track, because it optimizes a performance–precision trade-off rather than a curvature certificate. Our method measures the projection residual directly and rejects steps that violate the certified radius, so we expect LowRA to match TACT on memory reduction while showing a larger retention-loss increase at matched task gain, with the divergence concentrated where quantization error is largest.

**vs. TACT with Frobenius factor drift trigger** — On a case where the K-FAC block-diagonal approximation has significant off-block coupling (as measured by the diagnostic), the Frobenius factor drift can be large due to eigenbasis rotation or factor rescaling without corresponding change in the projection residual, or small when an eigenvalue crosses the rank-r truncation boundary causing subspace flip. Our projection-residual trigger directly measures the component of the true preservation gradient outside the current subspace, so we expect the residual trigger to yield fewer unnecessary reprojections and better retention-loss control, with the advantage growing with the measured off-block coupling magnitude.

### What would falsify this idea

If TACT's retention-loss advantage over static CurvDelta is uniform across adaptation steps and subsets rather than concentrated late in adaptation and on high-preservation-pressure subsets, then drift-triggered reprojection is not the causal driver and the central claim is wrong. Likewise, if the δ = ∞ ablation matches full TACT, the reprojection trigger contributes nothing and the design's core mechanism is unsupported. Additionally, if the projection-residual trigger does not outperform the Frobenius factor drift trigger on datasets with high off-block coupling, then the residual measure does not provide the claimed improvement in staleness detection.

## References

1. CurvDelta: Low-Curvature Delta Modeling For Surgical Learning and Unlearning
