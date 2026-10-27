# Gauge-Fixed Finite-Difference Certificates for Flatness-Regularized RLHF

## Motivation

Flatness-regularized RLHF penalizes an observable proxy for an unobservable quantity: the finite-difference gradient-norm estimate in Gradient Regularization (T_0) carries truncation and roundoff error never bounded against the true reward-model gradient norm, while the contrastive IRL reconstruction in IR3 (T_1) recovers a latent reward only up to gauge and partial observability. Across both trees the mitigation guarantee is asserted at the level of the proxy, so a policy can be certified 'flat' while the true reward gradient norm remains large. The structural missing piece is a coupling that turns the behavior-implied reward estimate into a gauge-fixed anchor against which finite-difference error can actually be bounded.

## Key Insight

A contrastive IRL reconstruction from paired behavior fixes the reward's scale gauge, and once the gauge is fixed the finite-difference truncation error becomes a computable function of the reconstruction's joint smoothness with the behavior policy, so the proxy-to-truth gap is itself an estimable quantity rather than an unverified assumption.

## Method

**What it is.** GFFD (Gauge-Fixed Finite-Difference certificate) takes as input a reward model r_phi, a behavior-policy dataset of paired responses D = {(x, y+, y-)}, and a finite-difference step h, and outputs (i) a gauge-fixed latent reward estimate r_hat, (ii) a certified radius U such that the true gradient norm at any evaluated state is bracketed by the finite-difference proxy plus/minus U, and (iii) a flatness penalty computed on the *upper* bracket rather than the raw proxy. The certificate is issued only on states inside the behavior-data support; off-support states are declared uncertified and receive no finite radius (see Phase 3b).

**How it works.**

```pseudocode
Input: reward model r_phi, paired behavior data D = {(x, y+, y-)}, step h, smoothness constants L_r, L_pi, gauge anchor set G

Phase 1 - Gauge-fixed reconstruction:
  Fit contrastive IRL head g_psi on D by minimizing Bradley-Terry loss
      L_IRL = -log sigmoid( g_psi(x,y+) - g_psi(x,y-) )
  Fix scale gauge by normalizing g_psi on anchor set G:
      r_hat(x,y) = g_psi(x,y) / mean_{(x,y) in G} |g_psi(x,y)|
  (anchor set = 512 held-out prompts with known unit-variance reward labels;
   if such labels are unavailable, use the empirical variance of g_psi over D itself:
   r_hat(x,y) = g_psi(x,y) / sqrt( Var_{(x,y) in D} g_psi(x,y) ) )

Phase 2 - Joint smoothness estimation:
  Estimate L_r = sup over sampled (s,a) of ||grad_s r_phi(s,a) - grad_s r_phi(s',a)|| / ||s - s'||
      via finite differences on a grid of spacing delta (delta >> h)
  Estimate L_pi = Lipschitz constant of the behavior policy log-prob w.r.t. state,
      via regression of log pi_b(a|s) on s over D

Phase 3 - Certified finite-difference error bound (on-support only):
  For each evaluated state s and action a, compute proxy gradient
      G_h(s,a) = ( r_phi(s + h*u, a) - r_phi(s - h*u, a) ) / (2h)   for random unit u
  Compute truncation term  T(s,a) = (L_r * h^2) / 6
  Compute roundoff term     R(s,a) = eps_mach * |r_phi(s,a)| / h
  Compute reconstruction-disagreement term  D(s,a) = ||grad_s r_hat(s,a) - G_h(s,a)||
      (grad_s r_hat computed by the same central difference with step h)
  Certified radius  U(s,a) = T(s,a) + R(s,a) + D(s,a)

Phase 3b - Support gate and abstention:
  For each evaluated state s, compute support score
      kappa(s) = min over (x,y) in D of ||s - x||
  If kappa(s) > kappa_max (kappa_max = 95th percentile of kappa over D), mark s as UNCERTIFIED:
      U(s,a) = +inf, Penalty(s,a) = 0, and record an abstention flag
  Else issue the finite U(s,a) from Phase 3.

Phase 4 - Certified flatness penalty:
  Penalty(s,a) = ( ||G_h(s,a)|| + U(s,a) )^2   if s is certified
  Penalty(s,a) = 0                              if s is uncertified (abstain)
  Add Penalty to the RLHF policy-gradient loss with weight lambda

Output: r_hat, U, Penalty, abstention flags
```

Hyperparameters: h = 1e-3 (finite-difference step), delta = 1e-2 (smoothness grid), lambda = 0.1 (flatness weight), anchor set size |G| = 512, kappa_max = 95th percentile of kappa over D (support gate).

**Why this design.** We chose a gauge-fixed contrastive reconstruction over a raw reward-model gradient because the contrastive head supplies the scale normalization that makes the reconstruction-disagreement term D computable, accepting the cost that D requires paired behavior data which may be scarce in some RLHF settings. We chose an additive certified radius U over a multiplicative correction because the truncation, roundoff, and reconstruction-disagreement errors arise from independent sources (Taylor remainder, floating point, and reconstruction bias respectively) and additivity follows from the triangle inequality without requiring them to be correlated, accepting the cost that U may be conservative when errors partially cancel. We chose to penalize the upper bracket ||G_h|| + U rather than the raw proxy because the upper bracket is the quantity that actually controls the true gradient norm under the joint smoothness condition, accepting the cost that the penalty is stricter than necessary when U is loose. We chose to gate the certificate on behavior-data support and abstain off-support rather than assign a spurious finite radius, because contrastive IRL is only identified on the support of the behavior data and off-support reward values are arbitrary; accepting the cost that some evaluation states receive no flatness penalty. A domain expert would NOT describe this as a variant of Gradient Regularization (T_0) because T_0 penalizes the raw finite-difference proxy with no error term, whereas GFFD's penalty is dominated by U when the reconstruction is poor and by ||G_h|| only when the certificate is tight — the two methods coincide only in the limit U -> 0, which T_0 never checks.

**Why it measures what we claim.** The certified radius U(s,a) measures the proxy-to-truth gap because the joint smoothness condition (L_r and L_pi finite) makes the Taylor truncation, floating-point roundoff, and reconstruction-disagreement errors additive upper bounds on ||G_h - grad r_true||; this assumption fails when the reward model has a discontinuity or the behavior policy is non-Lipschitz (e.g., a hard argmax), in which case U underestimates the true gap and the penalty reverts to the uncertified proxy. The gauge-fixed reconstruction r_hat measures the latent reward up to scale because the anchor-set normalization removes the multiplicative gauge freedom of contrastive IRL; this assumption fails when the anchor set is not representative of the evaluation distribution, in which case the reconstruction-disagreement term D absorbs the distribution shift as a spurious penalty. The reconstruction-disagreement term D(s,a) measures the reconstruction bias component of the proxy-to-truth gap because it directly compares the finite-difference gradient of the gauge-fixed reconstruction to the finite-difference gradient of the reward model; this assumption fails when r_hat and r_phi share a gradient direction but differ in magnitude (D then under-reports the bias), in which case D reflects only directional disagreement. The upper bracket ||G_h|| + U measures the true reward-model gradient norm because the triangle inequality plus the joint smoothness condition gives ||grad r_true|| <= ||G_h|| + U; this assumption fails when the finite-difference direction u is adversarially aligned with the gradient, in which case the bracket is loose but still valid (never violated), which is the safety property we rely on. The support gate kappa(s) measures whether the certificate is valid because contrastive IRL is only identified on the behavior-data support; this assumption fails when the behavior data is sparse in a region the policy visits, in which case the state is marked uncertified and receives no penalty rather than a spurious finite radius.

## Contribution

(1) A gauge-fixed finite-difference certificate (GFFD) that couples contrastive IRL reconstruction with finite-difference gradient estimation to produce a computable, per-state certified radius on the proxy-to-truth gap in flatness-regularized RLHF. (2) A certified flatness penalty that penalizes the upper bracket of the true gradient norm rather than the raw proxy, with the certificate's tightness (ratio U / ||G_h||) reported as a diagnostic for when the penalty is dominated by reconstruction error versus genuine flatness. (3) An open-source implementation of the joint smoothness estimator and certificate computation that plugs into standard PPO-based RLHF pipelines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Paired preference data with held-out anchor prompts (|G| = 512) | Supplies contrastive pairs plus gauge anchor set |
| Primary metric | Reward accuracy under policy shift | Detects hacking as proxy-reward divergence |
| Secondary metric | Abstention rate (fraction of evaluation states with kappa(s) > kappa_max) | Quantifies how often the certificate is withheld off-support |
| Baseline | KL penalty toward reference model | Standard anti-hacking baseline |
| Baseline | Raw finite-difference gradient regularization (T_0) | Tests value of the certified radius U |
| Baseline | Reference Resets of KL penalty | Tests whether flatness alone explains gains |
| Ablation-of-ours | GFFD with U set to zero | Isolates the certificate's contribution |
| Ablation-of-ours | GFFD with support gate disabled (kappa_max = +inf) | Tests whether off-support abstention matters |
| Ablation-of-ours | GFFD with anchor set size |G| in {64, 128, 256, 512, 1024} | Sensitivity to anchor scarcity |

### Why this setup validates the claim

The central claim is that penalizing the *certified upper bracket* ||G_h|| + U, rather than the raw finite-difference proxy, controls the true reward-model gradient norm and thereby suppresses reward hacking. Each baseline probes a distinct sub-claim. The KL baseline tests whether flatness regularization is even needed beyond standard trust-region pressure. The raw finite-difference baseline (T_0) tests the core claim directly: if T_0 matches GFFD, then the certified radius U contributes nothing and the whole construction is decorative. Reference Resets test whether the gain is just a byproduct of finding flatter regions by any means. The U=0 ablation is the sharpest control: it is GFFD minus its certificate, so any gap between it and full GFFD is attributable to U alone. The support-gate ablation tests whether the abstention mechanism is load-bearing: if disabling the gate does not change reward accuracy, then off-support states are not where hacking occurs and the gate is unnecessary. The anchor-size sweep tests whether the gauge-fixing procedure is robust to anchor scarcity, which is the practical concern for RLHF settings where labeled anchors are expensive. Reward accuracy under policy shift is the right primary metric because reward hacking manifests precisely as proxy reward rising while true reward accuracy degrades; a flatness metric alone could be gamed, but accuracy under shift is the outcome the certificate is supposed to protect. The abstention rate is the right secondary metric because it makes the certificate's honesty measurable: a method that certifies everything is either correct everywhere or lying, and the abstention rate distinguishes these.

### Expected outcome and causal chain

**vs. Raw finite-difference gradient regularization (T_0)** — On a case where the reward model has a region of high curvature near a spurious feature, T_0 produces a penalty that underestimates the true gradient norm there because its finite-difference proxy carries uncorrected truncation and reconstruction-disagreement error, so the policy drifts into that region. Our method instead inflates the penalty by U exactly where the reconstruction is poor, because the reconstruction-disagreement term D grows with ||grad r_hat - G_h||, so we expect a visible reward-accuracy gap concentrated on the high-curvature subset and near-parity on smooth subsets.

**vs. KL penalty toward reference model** — On a case where the reference model itself assigns high probability to a hacking-prone action, the KL baseline produces a policy that stays close to the reference and therefore inherits its hacking behavior, because KL constrains distance not gradient flatness. Our method instead penalizes the certified gradient norm independently of the reference, because the flatness penalty is computed on r_phi's geometry rather than on divergence from a possibly-compromised reference, so we expect KL to match us on benign prompts but trail us markedly on prompts where the reference is itself misaligned.

**vs. Reference Resets of KL penalty** — On a case where resetting the reference repeatedly lands the policy in a flat but low-accuracy basin, the reset baseline produces flatness without accuracy because flatness is necessary but not sufficient and the reset has no signal about which flat regions are good. Our method instead ties flatness to the certified reward geometry via r_hat, because the gauge-fixed reconstruction anchors the flatness penalty to the latent reward scale, so we expect the reset baseline to show flat gradients yet no accuracy gain, while GFFD shows both flat gradients and accuracy gain on the same subset.

**vs. GFFD with U set to zero** — On a case where the reconstruction is poor (high D), the U=0 ablation produces a penalty that ignores the reconstruction-disagreement term and therefore under-penalizes exactly where the certificate would have inflated U, because U=0 is the uncertified proxy. Our method instead adds D to the bracket, so we expect the U=0 ablation to match full GFFD on smooth, well-reconstructed subsets and to trail it on high-D subsets, with the gap scaling with the magnitude of D.

**vs. GFFD with support gate disabled** — On a case where the evaluation distribution includes states far from the behavior data (high kappa), the gate-disabled ablation assigns a finite U to off-support states based on an unidentifiable reconstruction, because contrastive IRL is only identified on-support. Our method instead abstains on those states, so we expect the gate-disabled ablation to show spurious penalties (or spurious confidence) on the high-kappa subset and to trail full GFFD on reward accuracy there, while matching on the on-support subset.

**vs. GFFD with anchor set size |G| in {64, 128, 256, 512, 1024}** — On a case where the anchor set is small (|G| = 64), the gauge normalization is noisy and r_hat's scale is poorly fixed, because the mean absolute value over 64 prompts has high variance. Our method with |G| = 512 instead uses a larger anchor set, so we expect reward accuracy to improve monotonically with |G| up to 512 and to plateau thereafter, with the |G| = 64 condition showing the largest degradation on the high-D subset.

### What would falsify this idea

If GFFD's reward-accuracy advantage over raw finite-difference regularization is uniform across all subsets rather than concentrated on the high-curvature, high-reconstruction-disagreement subset where U is predicted to be large, then the certificate is not doing the causal work we claim and the gain must come from something else. Equivalently, if the U=0 ablation matches full GFFD everywhere, the central claim that the certified radius controls the proxy-to-truth gap is false.

## References

1. Gradient Regularization Mitigates Reward Hacking in Reinforcement Learning from Human Feedback and Verifiable Rewards
2. IR3: Contrastive Inverse Reinforcement Learning for Interpretable Detection and Mitigation of Reward Hacking
