# Certified Domain of Applicability for Latent-Feedback Amplification via Identifiable Regime Boundaries

## Motivation

The single latent-feedback branch (Latent Feedback Is a Compute and Data Amplifier) fits one scaling law that models latent feedback as a monotone effective-parameter multiplier, then extrapolates savings beyond its tested grid — but the paper itself concedes latent feedback stops helping when unique data is scarce relative to model size. Because the multiplier is assumed monotone, the scarce-data inversion point is never made identifiable from the fitted law, so the amplifier's domain of applicability cannot be certified before training; practitioners either over-apply it in the scarce regime or discard it despite real savings. This recurs structurally across data-efficiency work (DataRater's compute-efficiency claim, OTQMS's transfer-quantity optimum): a monotone efficiency law is reported without an identifiable boundary at which the claimed benefit inverts.

## Key Insight

The inversion of latent-feedback savings is identifiable as a breakpoint in unique-data-per-parameter where the fitted effective-parameter multiplier's marginal loss reduction crosses zero, so a piecewise (rather than monotone) law yields a pre-training, held-out-validated certificate of when the amplifier is data-efficient.

## Method

**A. What it is.** We introduce **CAB-LF** (Certified Amplifier Boundary for Latent Feedback): a procedure that takes a small calibration grid of (model size N, unique data D, epochs E) runs of both a latent-feedback model and a vanilla Transformer, fits a *smooth, sign-changing* scaling law with an explicit crossover location in unique-data-per-parameter r = D/N, and outputs a certified threshold r* below which latent-feedback savings are declared invalid, together with a held-out bracket test that either certifies or refutes r*. The load-bearing assumption, stated plainly: latent-feedback savings change sign at a crossover in r = D/N that is identifiable from a small calibration grid, and a smooth multiplier g(r) = g_plus * tanh(kappa * log(r/r*)) with an interior r* is the correct functional form; its zero-crossing equals the true marginal-loss-reduction sign change. This assumption is tested, not presumed, by the calibration and verification steps below.

**B. How it works.**
```pseudocode
INPUT: model sizes N in {N1..Nk} (k=3, spanning >=1 decade, e.g. 60M, 160M, 410M),
       unique-data multipliers d in {d1..dm} (m=5, D = d*N, e.g. d in {0.5, 1, 2, 4, 8}),
       epochs E in {E1..E2} (2 settings, e.g. E=1 and E=4)
RUN for each (N,d,E):
    L_vanilla(N,d,E)  # tuned at every config, as in the source paper
    L_latent(N,d,E)   # latent feedback: top-layer state -> next-token input
# Step 1: effective-parameter fit per regime
for each config: fit  L = A/N_eff^alpha + B/D^beta + C   with N_eff = N*(1 + g(r))
# Step 2: smooth sign-changing multiplier with crossover
model g(r) = g_plus * tanh(kappa * log(r/r*))    # r = D/N, r* = crossover, kappa = sharpness
fit (g_plus, r*, kappa, alpha, beta, A,B,C) by minimizing pooled squared error over BOTH models
# Step 3: identifiability check via likelihood-ratio / sup-test for the breakpoint
compute profile likelihood of r* over a grid; keep r* only if the likelihood-ratio statistic
  LR(r*) = 2*(logL_full - logL_r*_fixed) exceeds the Andrews (1993) / Davies (1987) sup-test
  critical value at level 0.05 (tau set by held-out residual variance)
# Step 4: certified held-out bracket
hold out the 2 smallest-r and 2 largest-r configs; refit on the rest; predict held-out L_latent - L_vanilla sign
declare CERTIFIED iff sign predictions correct on all held-out configs AND r* lies strictly inside the held-out bracket
OUTPUT: r*, g_plus, kappa, certificate flag, predicted savings curve S(r) = L_vanilla - L_latent
```

**C. Why this design.** We chose a *smooth, sign-changing* multiplier with an explicit crossover over the source paper's monotone effective-parameter multiplier because a monotone form cannot represent the scarce-data inversion the source itself acknowledges, accepting the cost that the fit has one more free parameter (r*) and therefore needs a likelihood-ratio / sup-test identifiability check. We chose a likelihood-ratio / sup-test for the breakpoint over a bootstrap confidence interval for r* because the sup-test directly tests whether the data pin down the crossover rather than merely whether a resample is stable, accepting the cost that it assumes a locally quadratic profile near r* (which fails when the true transition is smooth; the tanh form and the sup-test are chosen to survive a non-quadratic profile). We chose a held-out *bracket* (smallest and largest r configs withheld) over random cross-validation because the scientific claim is specifically about the boundary, and only bracketing configs can falsify an interior r*; the cost is that we spend the most extreme grid points on validation rather than fitting. This is not a variant of DataRater's meta-gradient data valuation or OTQMS's transfer-quantity optimization: those learn per-example or per-source values, whereas CAB-LF estimates a single architecture-level regime boundary and certifies it, a quantity neither prior method produces.

**D. Why it measures what we claim.** The crossover r* measures the *identifiable scarce-data boundary* because the smooth multiplier's zero-crossing is equivalent to the marginal loss reduction of latent feedback changing sign, under the assumption that the pooled scaling law is correctly specified up to the crossover; this assumption fails when the true transition is smooth rather than sharp, in which case r* reflects the midpoint of a gradual crossover rather than a hard boundary. The likelihood-ratio / sup-test statistic measures *certifiability* because a high LR statistic at the optimum is equivalent to the boundary being pinned by the data, under the assumption that residuals are homoscedastic across the grid; this fails when small-N configs have much higher variance, in which case the LR statistic reflects noise structure rather than identifiability. The held-out bracket sign test measures *domain-of-applicability validity* because correct sign prediction on withheld extreme-r configs is equivalent to the smooth law generalizing across the boundary, under the assumption that held-out configs are drawn from the same regime family; this fails if the held-out grid sits entirely on one side of r*, in which case the test is vacuous rather than confirming.

## Contribution

(1) A piecewise scaling law for latent feedback with an explicit, fitted unique-data-per-parameter breakpoint r*, replacing the source paper's monotone effective-parameter multiplier. (2) A pre-training certification procedure (profile-likelihood curvature plus held-out bracket sign test) that outputs a binary certificate and a predicted savings curve, giving a measurable decision rule for when latent feedback is data-efficient. (3) An analysis artifact: the estimated r* and its certificate status across the model-size ladder, which quantifies how far today's models sit from the scarce-data regime.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Unique-data × epoch × model-size ladder (source grid): N ∈ {60M, 160M, 410M}, d ∈ {0.5, 1, 2, 4, 8}, E ∈ {1, 4} | Directly spans the r = D/N range the boundary lives in. |
| Primary metric | Sign-correctness of held-out (L_latent − L_vanilla) at extreme-r configs | Falsifiable domain-of-applicability test, not just loss parity. |
| Baseline | Monotone effective-parameter fit (source paper's form) | Tests whether a breakpoint is actually needed. |
| Baseline | Unconstrained piecewise fit without likelihood-ratio gate | Tests whether certification adds anything over naive fitting. |
| Baseline | Random-K-fold cross-validation of r* | Tests whether bracketing extremes is necessary for the claim. |
| Ablation-of-ours | CAB-LF with the likelihood-ratio identifiability check removed | Isolates the certification step from the fitting step. |
| Diagnostic | Heteroscedasticity check: residual variance vs. N; flag uncertified if variance ratio > 2 | Gates the curvature-based certificate on homoscedasticity. |

### Why this setup validates the claim

The central claim has two parts that must be tested separately: that latent feedback's savings genuinely invert near a scarce-data boundary (a *fitting* claim), and that this boundary is *certifiable* from a small grid rather than an artifact of flexible curves (a *certification* claim). The monotone effective-parameter baseline tests the first part: if a monotone multiplier already predicts held-out extreme-r signs as well as our smooth sign-changing form, then the crossover is unnecessary and the scarce-data inversion is not a real regime change. The unconstrained-piecewise baseline and the random-K-fold baseline test the second part: the former can fit any curve and will overfit a spurious r*, while the latter validates interior points that cannot falsify an interior boundary. Our ablation removes only the likelihood-ratio gate, isolating whether certification—not fitting—drives held-out sign accuracy. The heteroscedasticity diagnostic ensures that the likelihood-ratio statistic reflects identifiability rather than noise structure. Sign-correctness on withheld extreme-r configs is the right metric because the scientific claim is directional (latent feedback helps here, hurts there), so a method that merely matches average loss but flips signs at the boundary has failed the actual claim.

### Expected outcome and causal chain

**vs. Monotone effective-parameter fit** — On a grid configuration with very small unique data per parameter, the monotone baseline is forced to predict latent feedback still helps (or still hurts) uniformly, because its single smooth multiplier cannot change sign. Our smooth multiplier lets g(r) cross zero at r*, so the predicted savings curve flips. We expect the baseline to mispredict held-out sign at the smallest-r configs while roughly matching us at large r — a gap concentrated at the extreme, not uniform.

**vs. Unconstrained piecewise fit without likelihood-ratio gate** — On a sparse calibration grid, the unconstrained fit places r* wherever residual noise is lowest, often at a grid edge, because nothing penalizes a flat profile. Our likelihood-ratio gate rejects such fits or flags them uncertified. We expect the baseline to report a confident-looking r* that fails the held-out bracket sign test, while our certified r* passes on the same held-out configs.

**vs. Random-K-fold cross-validation of r*** — On a grid whose folds all sit on one side of the true boundary, random K-fold never withholds a bracketing pair, so it cannot detect a wrong interior r*. Our smallest- and largest-r holdout forces a bracketing test. We expect random-K-fold to show deceptively low held-out error while our bracket test exposes sign errors when r* is misplaced.

### What would falsify this idea

If our held-out sign accuracy is no better than the monotone baseline's and is uniform across r rather than concentrated at extreme-r configs, then the scarce-data inversion is not a real regime boundary and the crossover claim is wrong. If the likelihood-ratio gate rejects nearly all fits or certifies r* values that fail the bracket test, then certification adds nothing over naive fitting.

## References

1. Latent Feedback Is a Compute and Data Amplifier
