# Nested-Grid Posterior Matching with Refinement-Invariant Consistency

## Motivation

Distribution-based diffusion distillation methods, including Posterior Matching Distillation (PMD) and Distribution Backtracking (DisBack), assume a fixed discrete timestep grid during training: PMD's any-step guarantee holds only within the trained grid, and DisBack records its degradation trajectory on one grid. When a deployed student is sampled on a finer or coarser grid than the one used in training, its conditional posteriors at the new grid's noise levels are unconstrained, so the any-step property degrades from a guarantee to an empirical hope. The structural cause is that no existing distillation objective ties the student's posteriors across nested grids, leaving refinement invariance untrained.

## Key Insight

Because Gaussian-corrupted posteriors at a coarse timestep are exact mixtures of posteriors at any finer sub-grid, enforcing that the student's fine-grid posteriors aggregate to its coarse-grid posteriors under the known Gaussian transition kernel makes refinement invariance a trainable consistency condition rather than an untested extrapolation.

## Method

**What it is.** Nested-Grid Posterior Matching (NGPM) distills a pretrained diffusion teacher into a stochastic student that is trained simultaneously on K=3 nested timestep grids G_1={8 steps} ⊂ G_2={16 steps} ⊂ G_3={32 steps} and is constrained so that its conditional posteriors on a finer grid are consistent, under the Gaussian corruption kernel, with its posteriors on the next coarser grid. Inputs: teacher score network eps_phi, K=3 nested grids, noise schedule beta_t. Outputs: a student denoiser s_theta(x_t, t, g) that supports any step count in the union of the grids with a monotone consistency property across grid refinement.

**How it works.**
```pseudocode
# Grids G_1 ⊂ G_2 ⊂ G_3 (coarse to fine), shared noise schedule beta_t
# Student s_theta(x_t, t, g) conditioned on grid id g; teacher eps_phi
# Hyperparameters: M=2 fine samples, lambda_refine=0.5, lambda_mono=0.1, margin=0.01,
#   warmup steps W=500, confidence threshold tau=0.1
for each training step:
    sample batch x0 ~ data; sample grid index k ~ Uniform(1..K)
    # --- Anchor branch: PMD-style posterior matching on grid G_k ---
    sample t in G_k; sample x_t = sqrt(alpha_t) x0 + sqrt(1-alpha_t) eps
    target_mean = teacher_posterior_mean(eps_phi, x_t, t)        # single teacher eval at shifted input
    L_anchor = || s_theta(x_t, t, G_k) - target_mean ||^2
    # --- Refinement-consistency branch: coarse grid must be a Gaussian mixture of fine ---
    if k > 1:
        t_coarse = nearest_coarse(G_k, t)                          # t_coarse in G_{k-1}, t_coarse < t
        # Draw M fine-grid samples from the TEACHER's posterior at t (stop-gradient),
        # not from the student, to avoid encoding the student's own sampling bias.
        {x_t^{(m)}} = { teacher_posterior_sample(eps_phi, x_t, t) }  # M=2 samples, stop-grad
        # Push each teacher fine sample through the known Gaussian transition
        # N(x_t; sqrt(a) x_{t_coarse}, (1-a) I) to t_coarse
        {x_{t_coarse}^{(m)}} = { sqrt(a) * x_t^{(m)} + sqrt(1-a) * eps_m }
        # Student coarse mean evaluated at the SAME pushed-forward states
        student_coarse_mean = mean_m s_theta(x_{t_coarse}^{(m)}, t_coarse, G_{k-1})
        # Teacher coarse target evaluated at the SAME pushed-forward states
        teacher_coarse_mean = mean_m teacher_posterior_mean(eps_phi, x_{t_coarse}^{(m)}, t_coarse)
        L_refine = || student_coarse_mean - teacher_coarse_mean ||^2
        # Confidence gate: only active after warmup and when anchor error is small
        gate = 1.0 if (step > W and L_anchor.detach() < tau) else 0.0
    else:
        L_refine = 0
        gate = 0.0
    # --- Monotonicity hinge: finer grid must not increase matching error ---
    L_mono = max(0, L_anchor - stopgrad(L_refine) + margin)
    loss = L_anchor + gate * lambda_refine * L_refine + lambda_mono * L_mono
    update theta
```

**Why this design.** We chose nested grids over independent per-grid students because a single grid-conditioned student shares parameters across grids, forcing the refinement-consistency loss to act as a coupling constraint rather than as K independent distillation problems; the cost is that capacity must be shared, which we accept because the shared structure is exactly the invariant we want. We chose the Gaussian-transition mixture identity for the coarse-grid target over an unrolled multi-step student rollout because the mixture is exact under the known corruption kernel and avoids backpropagating through M student steps, accepting the cost that the consistency signal is a moment-matching signal rather than a full distributional one. We chose a hinge monotonicity term over a symmetric penalty because the direction of refinement (finer should be at least as accurate) is the deployment-relevant monotonicity, accepting the cost that the hinge can saturate and stop providing gradient once the margin is satisfied. A domain expert would not describe this as PMD plus a regularizer: PMD trains each posterior independently at a fixed grid, whereas NGPM's refinement-consistency term ties posteriors across grids through the Gaussian kernel, which is the structural property PMD's per-level matching does not produce.

**Why this measures what we claim.** The refinement-consistency residual `L_refine` measures grid-refinement invariance because the Gaussian-transition mixture identity makes the coarse-grid posterior an exact expectation of fine-grid posteriors under the corruption kernel; this assumption fails when the student's fine-grid samples are not drawn from the true fine-grid posterior (early training), in which case `L_refine` reflects the student's own sampling bias rather than a teacher-student mismatch. To mitigate this, we draw the fine-grid samples from the teacher's posterior (stop-gradient) and evaluate both the student and teacher coarse means at the same pushed-forward states, so that `L_refine` is a teacher-anchored consistency signal; we additionally gate `lambda_refine` by a warmup/confidence weight tied to `L_anchor` so the consistency term is only active once the fine-grid posterior is approximately correct. The monotonicity hinge `L_mono` measures the claimed monotone any-step guarantee because the stop-gradient target makes it a one-sided constraint that the anchor error not exceed the refinement error; this assumption fails when `L_refine` is itself large, in which case the hinge is vacuous and reflects only that both grids are poorly fit. The grid-conditioning input `g` measures the student's ability to generalize across sampling budgets because shared parameters force the student to represent a family of posteriors indexed by grid; this assumption fails when grids are too sparse for the shared trunk to interpolate, in which case `g` acts as a task identifier rather than a budget parameter.

## Contribution

["A nested-grid distillation objective that couples the student's posteriors across nested timestep grids via an exact Gaussian-transition mixture identity, turning refinement invariance into a trainable consistency condition.", 'A one-sided monotonicity hinge that constrains finer-grid matching error to not exceed coarser-grid matching error, operationalizing a monotone any-step property rather than a within-grid any-step property.', 'A grid-conditioned student architecture and training protocol that supports sampling at any step count in the union of nested grids without retraining.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | ImageNet 64×64 | Standard distillation benchmark with reported FID anchors |
| Primary metric | FID at each grid step count (8, 16, 32) | Directly measures quality across budgets |
| Baseline | PMD (per-grid posterior matching) | Tests whether cross-grid coupling adds value |
| Baseline | CED (one-step contrastive energy) | Tests multi-budget flexibility vs single-step |
| Baseline | DisBack (trajectory backtracking) | Tests refinement-consistency vs trajectory recording |
| Baseline | Consistency-model-style student (continuous-time consistency loss across the same nested grids) | Tests whether Gaussian-mixture moment-matching coupling is equivalent to existing consistency training |
| Ablation-of-ours | NGPM without L_refine and L_mono | Isolates the coupling and monotonicity contributions |
| Ablation-of-ours | NGPM with student-generated fine samples (original L_refine) | Tests the effect of teacher-anchored vs student-generated consistency signal |

### Why this setup validates the claim

The central claim is that tying posteriors across nested grids through the Gaussian corruption kernel produces a student whose quality is monotone in grid refinement and whose any-step behavior exceeds what independent per-grid distillation achieves. ImageNet 64×64 is the right testbed because prior work (DisBack, CED) reports FID there, giving us concrete anchors for what per-grid and one-step distillation achieve. PMD is the critical baseline: it matches posteriors at a fixed grid without cross-grid coupling, so if NGPM's gain over PMD is concentrated at intermediate grids and on refinement-sensitive samples, the coupling mechanism is doing the work rather than the anchor loss alone. CED tests whether one-step-only distillation already captures the multi-budget benefit, and DisBack tests whether trajectory recording is a cheaper route to the same refinement consistency. The consistency-model-style baseline directly tests whether our Gaussian-mixture moment-matching coupling is equivalent to existing consistency training; if it matches NGPM, the novelty claim collapses. The ablation removes both coupling terms, so any gap between it and full NGPM localizes the effect to the refinement-consistency and hinge mechanisms rather than to the shared trunk or anchor branch. The additional ablation with student-generated fine samples isolates the effect of the teacher-anchored repair.

### Expected outcome and causal chain

**vs. PMD** — On a case where the deployed budget is an intermediate grid (e.g., 16 steps) that PMD never trained at its own grid, PMD produces a student whose posterior at that budget drifts from the teacher's because its per-grid matching has no constraint linking adjacent budgets. Our method instead enforces that the coarse-grid posterior equals the Gaussian-mixture expectation of fine-grid posteriors, so the intermediate grid inherits a consistent target from its neighbors. We expect a noticeable FID gap over PMD at intermediate step counts, with parity at the grids PMD was explicitly trained on.

**vs. CED** — On a case requiring multiple sampling budgets from one checkpoint, CED produces a one-step generator whose quality degrades sharply when queried at, say, 8 or 32 steps because it was distilled for a single budget and has no grid-indexed posterior family. Our method instead conditions the student on grid id and ties posteriors across the nested grids, so each budget has an explicit, coupled target. We expect CED to be competitive at one step but to fall behind NGPM across the multi-step union, with the gap widening as the queried budget moves away from one step.

**vs. DisBack** — On a case where the student's early-training samples are far from the teacher's intermediate distributions, DisBack produces a degraded backtracking signal because its recorded degradation path was built from the untrained student and does not track the student's evolving bias. Our method instead computes the coarse-grid target directly from the teacher at the shifted input, so the consistency signal is anchored to the teacher rather than to a stale recorded path. We expect a faster early-training drop in FID for NGPM and a smaller final gap on refinement-sensitive samples, with parity on easy samples.

**vs. Consistency-model-style student** — On a case where the student is trained with a standard continuous-time consistency loss across the same nested grids, that baseline enforces self-consistency between adjacent timesteps but does not tie the coarse-grid posterior to the teacher's Gaussian-mixture expectation. Our method instead uses the teacher-anchored mixture identity, so the coarse-grid target is explicitly the teacher's posterior mean at the pushed-forward state. We expect NGPM to outperform the consistency-model-style student at intermediate grids, demonstrating that the Gaussian-mixture coupling is not equivalent to existing consistency training.

### What would falsify this idea

If NGPM's gain over PMD is uniform across all grid step counts rather than concentrated at intermediate grids and on refinement-sensitive samples, the cross-grid coupling is not the causal mechanism and the central claim is wrong. If the no-coupling ablation matches full NGPM at every budget, the refinement-consistency and monotonicity terms are inert.

## References

1. Posterior Matching Distillation
