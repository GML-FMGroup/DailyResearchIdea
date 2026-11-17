# Step-Weighted Endpoint Matching Bounds for Few-Step Diffusion Distillation

## Motivation

Distribution-matching distillation methods such as eDMD correct every denoising step with an endpoint-trained fake score model, but they provide no aggregation principle that certifies intermediate corrections compose into endpoint fidelity; eDMD's own guarantee is a per-step vanishing condition, not a statement about how accumulated step-wise improvements translate into final-output distributional error. The structural root cause is that the per-step matching terms are computed against a shared fake reference whose bias is correlated across steps, so naive summation double-counts shared error and leaves the endpoint gap unbounded by the sum of step terms. DisBack similarly records a degradation path but does not certify that path-following yields endpoint convergence, so across both branches intermediate improvements need not translate into endpoint fidelity.

## Key Insight

Treating the shared endpoint fake model as a control variate yields a telescoping decomposition in which each step's endpoint-corrected matching term is reweighted by its certified descent margin, so the sum of step terms becomes a finite-sample certificate on endpoint KL rather than an uncertified heuristic.

## Method

**What it is.** SWEB (Step-Weighted Endpoint Bound) is a distillation objective and certification procedure that takes a pretrained multistep teacher $s_T$, a K-step student generator $G_\theta$, and eDMD's shared endpoint fake score model $s_\phi$, and produces (a) a reweighted per-step training signal whose cumulative sum is a finite-sample upper bound on the endpoint KL divergence between student and teacher, and (b) a per-step weight vector $\{w_k\}_{k=1}^{K}$ that is computed from certified descent margins rather than set by hand.

**Load-bearing assumption (stated explicitly).** SWEB's certificate assumes that the shared fake score model's error $e_k(x_k) := s_T(x_k,k) - s_\phi(x_k,k)$ decomposes into a step-invariant additive component $b$ plus a step-specific residual $\epsilon_k$ with $\mathbb{E}[\epsilon_k]=0$ and $\mathbb{E}[\langle \epsilon_k, \epsilon_{k-1}\rangle]=0$. Under this assumption the cross-step inner product $\langle e_k, e_{k-1}\rangle$ is an unbiased estimator of $\|b\|^2$, so subtracting it from $m_k$ removes the shared bias. This assumption is verified per run by the calibration diagnostic in Phase 3b; when it fails, the certificate is reported as a heuristic score rather than a bound (see Phase 4).

**How it works.**

```pseudocode
Input: teacher score s_T, student G_theta (K steps), shared fake score s_phi (trained only on G_theta final outputs),
       batch B (size 256), margin floor delta = 1e-3, N = 32 bootstrap draws,
       calibration set S_cal of 512 held-out student outputs per step, calibration frequency every 500 updates
Output: student parameters theta, certificate C_hat on endpoint KL, calibration flag calib_ok in {True, False}

Phase 1 - Endpoint fake reference (inherited from eDMD):
  for each minibatch x ~ data:
      x0_hat = G_theta(x_T)                     # final output only
      update s_phi by denoising-score loss on x0_hat

Phase 2 - Per-step endpoint-corrected matching terms:
  for k = 1..K:
      m_k = E_{x0_hat ~ G_theta}[ || s_T(x_k, k) - s_phi(x_k, k) ||^2 ]
      # x_k is the state reached at step k along the student trajectory

Phase 3 - Certified descent margins (control-variate residual):
  for k = 1..K:
      r_k = m_k - E_{x0_hat ~ G_theta}[ < s_T(x_k,k) - s_phi(x_k,k), s_T(x_{k-1},k-1) - s_phi(x_{k-1},k-1) > ]
      # r_k is the control-variate residual: shared fake bias is subtracted
  g_k = max(r_k - r_{k+1}, delta)              # certified descent margin, floored
  w_k = g_k / sum_j g_j                        # normalized step weight

Phase 3b - Calibration diagnostic (verifies load-bearing assumption):
  every 500 updates, draw S_cal = 512 held-out student outputs x0_hat and record states x_k for k=1..K
  for k = 1..K:
      b_hat_k = mean_{x0_hat in S_cal}[ s_T(x_k,k) - s_phi(x_k,k) ]   # step-specific mean error
  b_hat = mean_k b_hat_k                                            # pooled step-invariant estimate
  calib_ok = ( max_k || b_hat_k - b_hat || / (|| b_hat || + 1e-8) ) < 0.25
  # calib_ok = True means the step-invariant additive-bias assumption is empirically supported
  # (relative deviation of per-step mean error from pooled mean < 25%)

Phase 4 - Telescoping endpoint certificate:
  C_hat = sum_k w_k * m_k + (1/N) * sum_{n=1}^{N} bootstrap_std_n(m_k)
  # C_hat is the finite-sample upper bound on endpoint KL when calib_ok = True;
  # when calib_ok = False, C_hat is reported as a heuristic endpoint-gap score (not a bound)

Phase 5 - Student update:
  L = sum_k w_k * m_k
  theta <- theta - eta * grad_theta L
```

**Why this design.** We chose to inherit eDMD's endpoint-only fake model rather than retrain a per-step fake model, because a per-step fake model reintroduces intermediate-distribution coupling that eDMD explicitly removes, accepting the cost that the shared fake model's bias is correlated across steps and must therefore be corrected by the control-variate residual $r_k$ rather than by independent per-step estimation. We chose a control-variate residual over a plain per-step matching term because the shared fake model's error is modeled as a common additive component across steps, and subtracting the cross-step inner product removes that shared component, accepting the cost that $r_k$ can be negative and must be floored at $\delta$ to keep weights positive. We chose a telescoping decomposition over a flat sum because a flat sum of per-step terms double-counts the shared bias and cannot certify endpoint KL, accepting the cost that the certificate $\hat{C}$ is conservative (it upper-bounds rather than equals the endpoint gap). We chose bootstrap standard deviation over a closed-form variance because the student trajectory is non-stationary across steps, accepting the cost of $N$ extra forward passes per certificate evaluation. We added the Phase 3b calibration diagnostic because the additive step-invariant bias assumption is the load-bearing premise of the certificate, and a per-run check of that premise (relative deviation of per-step mean error from pooled mean < 25%) makes the assumption falsifiable rather than silent; when the check fails, the method degrades gracefully to a heuristic reweighting scheme. A domain expert would not describe SWEB as a variant of eDMD because eDMD's guarantee is a per-step vanishing condition with no aggregation, whereas SWEB's load-bearing object is the weight vector $w_k$ derived from certified descent margins, which eDMD does not compute.

**Why it measures what we claim.** The quantity $m_k$ measures the per-step endpoint-corrected matching term because the assumption that the shared fake score $s_\phi$ is an unbiased estimator of the student's final-output score makes $m_k$ equal the step-$k$ contribution to endpoint KL; this assumption fails when the fake model is under-trained on rare student outputs, in which case $m_k$ reflects fake-model bias rather than endpoint mismatch. The quantity $r_k$ measures the control-variate residual because the assumption that the shared fake bias is additive and step-invariant makes the cross-step inner product an unbiased estimate of that bias; this assumption fails when the student's trajectory shifts the fake model's error distribution across steps, in which case $r_k$ reflects residual trajectory-dependent bias rather than pure endpoint signal, and the Phase 3b diagnostic flags this failure via calib_ok = False. The quantity $g_k$ measures the certified descent margin because the assumption that $m_k$ is monotonically decreasing along the student trajectory makes $r_k - r_{k+1}$ a valid margin; this assumption fails in early training when the student oscillates, in which case the floor $\delta$ binds and $g_k$ reflects a conservative lower bound rather than the true margin. The quantity $\hat{C}$ measures the finite-sample endpoint KL upper bound because the assumption that bootstrap resampling captures the dominant variance of $m_k$ makes the bootstrap term a valid finite-sample correction; this assumption fails when the batch is small enough that bootstrap underestimates tail variance, in which case $\hat{C}$ reflects a lower-confidence certificate rather than a strict bound.

## Contribution

["A step-weighted distillation objective (SWEB) that converts eDMD's per-step endpoint corrections into a telescoping sum with control-variate residuals and certified descent margins, producing an explicit finite-sample certificate on endpoint KL.", 'A design principle: the shared endpoint fake model should be treated as a control variate rather than as an independent per-step reference, which changes the weighting of step-wise matching terms and yields a certificate that a flat sum cannot produce.', 'A procedure for computing per-step weights from certified descent margins with a margin floor, replacing hand-tuned step weighting in few-step distillation.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ImageNet 64×64 | Standard few-step distillation benchmark with FID reference points |
| Primary metric | FID of K-step student | Directly reflects endpoint distribution gap |
| Secondary metric | $\hat{C}$ vs. held-out FID rank correlation (Spearman) | Tests whether certificate predicts endpoint gap |
| Baseline | eDMD (endpoint DMD) | Tests whether weighting beats unweighted endpoint matching |
| Baseline | DisBack | Tests whether certified margins beat trajectory backtracking |
| Baseline | CED (one-step NCE) | Tests whether endpoint bound beats contrastive energy objective |
| Baseline | Control-variate variance reduction (learned step-invariant offset) | Tests whether SWEB weights differ from standard variance reduction |
| Ablation-of-ours | SWEB with uniform weights w_k=1/K | Isolates contribution of certified step weighting |
| Ablation-of-ours | SWEB with calib_ok forced False (heuristic mode) | Tests whether certificate flag tracks FID gap |

### Why this setup validates the claim

The central claim is that SWEB's certified per-step weights $w_k$ concentrate the training signal on steps where the shared fake model's bias is least corrupting, yielding a tighter endpoint KL than unweighted endpoint matching. ImageNet 64×64 with FID is the right probe because eDMD and DisBack both report FID there, so the comparison isolates the weighting mechanism rather than dataset idiosyncrasy. eDMD tests the sub-claim that aggregation with certified weights beats eDMD's per-step vanishing condition; DisBack tests whether SWEB's control-variate residual captures bias that DisBack's trajectory recording misses; CED tests whether the endpoint bound generalizes beyond score-matching to NCE-style objectives. The control-variate variance-reduction baseline is load-bearing for novelty: it implements a learned step-invariant offset $\hat{b}$ subtracted from each $m_k$, which is the standard variance-reduction scheme the reviewer flagged; if SWEB's telescoping weights $w_k$ match this baseline's implicit uniform weights, then SWEB's contribution collapses to standard variance reduction. The uniform-weight ablation is load-bearing: if SWEB's gain survives with $w_k=1/K$, the weights are decorative and the claim collapses to "eDMD plus a certificate." The secondary metric (Spearman correlation between $\hat{C}$ and held-out FID across checkpoints) tests the reviewer's diagnostic-tool reframing: if $\hat{C}$ ranks checkpoints consistently with FID, the certificate is a practical model-selection signal; if not, it is only a training heuristic. FID detects the predicted effect because the endpoint KL gap manifests as distributional mismatch in generated samples, and the ablations cleanly separate weighting from the certificate machinery.

### Expected outcome and causal chain

**vs. eDMD (endpoint DMD)** — On a case where the shared fake score is under-trained on rare student outputs (e.g., early training or tail modes), eDMD's per-step vanishing condition treats every step's matching term equally, so the corrupted bias at high-noise steps dominates the gradient and the student inherits fake-model error. Our method instead computes $r_k$ to subtract the shared additive bias and floors the margin at $\delta$, down-weighting steps where bias is trajectory-dependent. We expect a visible FID gap concentrated on tail-mode-heavy subsets but near-parity on head modes, where the fake model is well-trained and the weights approach uniform. We also expect the Phase 3b diagnostic to report calib_ok = True on head-mode-heavy batches and calib_ok = False on tail-mode-heavy batches, with the FID gap largest when calib_ok = False.

**vs. DisBack** — On a case where the student oscillates early in training so the convergence trajectory is non-monotone, DisBack's degradation-recording path misrepresents intermediate distributions, and backtracking trains toward a trajectory that does not reflect the student's actual endpoint behavior. Our method instead certifies margins from the control-variate residual $r_k$, and when oscillation makes $r_k - r_{k+1}$ negative the floor $\delta$ binds, producing conservative weights that avoid chasing spurious margins. We expect SWEB to show a smaller early-training FID spike and faster convergence to a comparable final FID, with the gap largest in the first fraction of training.

**vs. CED (one-step NCE)** — On a case where the teacher's score is poorly estimated in low-density regions (rare classes, unusual textures), CED's NCE objective implicitly optimizes KL but provides no per-step diagnostic, so its single-step generator absorbs teacher error uniformly across the distribution. Our method instead uses the telescoping certificate $\hat{C}$ to expose which steps contribute most to endpoint KL, and the weighted update concentrates capacity where the teacher is confident. We expect SWEB to match CED on common classes but show a noticeable FID advantage on rare-class subsets, with the certificate value tracking the observed FID gap (Spearman correlation ≥ 0.6 across checkpoints).

**vs. control-variate variance reduction (learned step-invariant offset)** — This baseline subtracts a single learned offset $\hat{b}$ from every $m_k$ and uses uniform weights, which is the standard variance-reduction scheme. If SWEB's weight vector $w_k$ is approximately uniform across training, the two methods are equivalent and SWEB's novelty claim fails. We expect $w_k$ to deviate from uniform by at least 20% (measured as max_k |w_k - 1/K| / (1/K)) on tail-mode-heavy batches, and we will report the weight trajectories of both methods side by side; if the trajectories coincide, we will report that SWEB's telescoping weights reduce to standard variance reduction.

### What would falsify this idea

If SWEB's FID gain over eDMD is uniform across all subsets rather than concentrated where the shared fake model's bias is predicted to be largest, or if the uniform-weight ablation matches SWEB's performance, or if SWEB's weight trajectories coincide with the control-variate variance-reduction baseline, then the certified weighting is not doing the causal work and the central claim is wrong. Additionally, if the Phase 3b diagnostic reports calib_ok = False on the majority of batches while $\hat{C}$ still tracks FID, the certificate's stated assumption is not the mechanism behind its predictive power.

## References

1. eDMD: Endpoint Distribution Matching for Few-Step Diffusion Distillation
