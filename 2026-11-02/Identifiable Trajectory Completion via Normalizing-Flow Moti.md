# Identifiable Trajectory Completion via Normalizing-Flow Motion Priors

## Motivation

Dense 3D tracking methods like ReLift repair occluded trajectory segments by applying a learned corrector that treats the motion prior as a soft regularizer; because the prior is not an identifiable constraint, the completed trajectory in occluded regions is one of many plausible completions with no certificate of uniqueness. This structural underdetermination recurs across trajectory-repair approaches and means that even a correct prior cannot guarantee that the repaired trajectory is the one that actually occurred, leaving a persistent gap between plausible completion and identifiable completion.

## Key Insight

A normalizing flow defines an invertible mapping between latent codes and trajectories, so the maximum a posteriori completion is unique whenever the observed-point likelihood is log-concave and the flow's Jacobian is bounded away from zero, turning trajectory repair into a constrained optimization whose solution is certified by the invertibility of the prior.

## Method

**What it is.** We introduce **FlowCert**, a trajectory-repair module that takes a lifted 3D trajectory with per-point reliability masks and outputs a completed trajectory together with a uniqueness certificate. It uses a normalizing flow $f_\theta$ as a generative motion prior over trajectory windows and solves a constrained MAP problem whose solution is certified by the flow's invertibility.

**Load-bearing assumption (stated explicitly).** The certificate is valid only if the curvature of the negative log posterior in *trajectory space* $Y$ — not latent space $z$ — is bounded below at the MAP point. Because the posterior density over trajectories is $p_Y(Y) = p_Z(f_\theta(Y))\,|\det \partial f_\theta/\partial Y|$, the trajectory-space Hessian carries an extra term from the log-Jacobian of the flow, $\partial^2 \log|\det \partial f_\theta/\partial Y|/\partial Y^2$, that a latent-space Hessian omits. We therefore compute the certificate in trajectory space and do not use the latent-space Hessian $H = (1/\sigma^2) J_R^T J_R + I$, whose $+I$ term would trivially force $\lambda_{\min} \ge 1 > \kappa$ for any flow and any observation pattern.

**How it works.**

```pseudocode
Input: lifted trajectory X ∈ R^{T×3}, reliability mask m ∈ {0,1}^T, flow f_θ (trained on clean trajectories)
Output: completed trajectory X*, certificate flag cert ∈ {true, false}

1. Partition X into reliable points R = {t : m_t = 1} and missing points M = {t : m_t = 0}.
2. Define the observation model: X_t = Y_t + ε_t for t ∈ R, with ε_t ~ N(0, σ² I), σ = 0.05 m (fixed from lifting-noise calibration).
3. MAP objective over the latent code z:
     L(z) = (1/2σ²) Σ_{t∈R} ||Y_t(z) − X_t||² − log p_Z(z)
   where Y(z) = f_θ^{-1}(z) is the decoded trajectory and p_Z = N(0, I).
4. Solve z* = argmin_z L(z) by L-BFGS with 50 iterations, initializing z from f_θ(X).
5. Compute the trajectory-space Hessian of the negative log posterior at Y* = Y(z*):
     H_Y = (1/σ²) P_R^T P_R + ∇²_Y [ − log p_Z(f_θ(Y)) − log|det ∂f_θ/∂Y| ]_{Y=Y*}
   where P_R selects reliable points, and the second term is evaluated by automatic differentiation through the flow's log-Jacobian. (Equivalently, H_Y = J^{-T} H_z J^{-1} + ∇²_Y log|det J|, with J = ∂Y/∂z at z*.)
6. Certificate: cert = true iff λ_min(H_Y) ≥ κ, with κ = 0.1 (chosen so that the posterior is strongly log-concave in trajectory space).
7. Return X* = Y(z*) and cert.
```

**Why this design.** We chose a normalizing flow over an energy-based model because the flow provides an explicit invertible map whose Jacobian yields a computable trajectory-space Hessian for the uniqueness certificate, accepting the cost that flows require architectural constraints (e.g., coupling layers) that may limit expressiveness compared to unconstrained EBMs. We chose a hard constrained optimization over a soft attention-based corrector (as in ReLift) because soft priors cannot certify uniqueness, accepting the cost of iterative optimization at inference (≈50 L-BFGS steps) instead of a single feed-forward pass. We chose a fixed observation noise σ calibrated from lifting statistics rather than learning it, because a learned noise model would couple the certificate to training data and undermine the assumption that the likelihood is log-concave, accepting the cost that σ may be suboptimal when lifting errors are heteroscedastic. We chose to restrict the observation term to reliable points rather than all points, because missing points contribute no likelihood curvature and would artificially inflate the certificate, accepting the cost that the certificate becomes conservative when few reliable points are available. We chose to compute the Hessian in trajectory space including the log-Jacobian term rather than in latent space, because the latent-space Hessian omits the change-of-variables curvature and its $+I$ term would make the certificate trivially true for any flow, accepting the cost of an extra automatic-differentiation pass through the log-Jacobian at inference. A domain expert would not describe this as a variant of ReLift because ReLift's corrector is a feed-forward attention network that outputs a single trajectory without any uniqueness guarantee, whereas FlowCert solves a constrained MAP problem and returns a certificate that is false when the prior does not sufficiently constrain the missing segment.

**Why this measures what we claim.** The trajectory-space minimum eigenvalue λ_min(H_Y) measures the uniqueness of the MAP completion because the assumption that the flow is invertible and the observation likelihood is log-concave makes the posterior over trajectories strictly log-concave with curvature bounded below by λ_min(H_Y); this assumption fails when the flow's Jacobian is near-singular on the missing segment (e.g., the flow collapses a mode), in which case λ_min(H_Y) reflects numerical conditioning of the flow rather than genuine posterior uncertainty. The certificate flag cert measures identifiability because the assumption that reliable points provide sufficient curvature (P_R full rank) makes λ_min(H_Y) ≥ κ equivalent to a unique MAP estimate in trajectory space; this assumption fails when reliable points are collinear or too few, in which case cert may be false even though the true trajectory is unique, reflecting a limitation of the observation pattern rather than the prior.

## Contribution

['A trajectory-repair framework, FlowCert, that formulates completion as a constrained MAP problem with a normalizing-flow motion prior and returns a per-trajectory uniqueness certificate based on the Hessian of the negative log-posterior.', 'A design principle showing that invertible generative priors enable computable identifiability certificates for trajectory completion, whereas soft attention-based correctors (e.g., ReLift) cannot provide such certificates.', 'An empirical demonstration that the certificate flag correlates with completion accuracy on synthetic and real occlusion benchmarks, providing a practical tool for flagging unreliable completions.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Dense 3D tracking benchmark with occlusion-heavy sequences | Occlusion creates missing points where repair is testable |
| Primary metric | Trajectory error on missing/corrupted points | Directly measures repair quality where claim applies |
| Baseline | ReLift | Feed-forward corrector without uniqueness guarantee |
| Baseline | Lifted 2D trackers + geometry | Tests whether repair beats raw lifting |
| Baseline | Dedicated 3D tracker | Tests against end-to-end tracking alternative |
| Ablation-of-ours | FlowCert without certificate (repair only) | Isolates value of constrained MAP vs certificate |
| Ablation-of-ours | FlowCert with latent-space Hessian certificate (original H = (1/σ²)J_R^T J_R + I) | Tests whether trajectory-space certificate adds value |
| Calibration set | 512 synthetic trajectories with known ground truth and controlled occlusions | Validates certificate calibration on known truth |

### Why this setup validates the claim

The central claim is that FlowCert's constrained MAP over an invertible flow prior produces completions that are both accurate and certifiably unique, and that this advantage concentrates precisely where lifting fails: occlusion gaps and corrupted positions. Comparing against ReLift tests the sub-claim that a hard constrained MAP with an explicit prior beats a soft feed-forward corrector on repair accuracy; comparing against lifted 2D trackers plus geometry tests the sub-claim that repair recovers motion the lifting pipeline drops entirely; comparing against a dedicated 3D tracker tests whether the modular repair approach is competitive end-to-end. The ablation — FlowCert with repair but no certificate — isolates whether the constrained MAP itself, rather than the certificate machinery, drives gains, and whether certificate-flagged trajectories are more accurate than unflagged ones. The additional ablation comparing the trajectory-space certificate to the latent-space certificate tests whether the change-of-variables correction is necessary for the certificate to be meaningful. The calibration set of 512 synthetic trajectories with known ground truth allows us to measure the certificate's false positive and false negative rates under controlled occlusions and noise, directly validating that the certificate measures what it claims. Trajectory error on missing and corrupted points is the right metric because it is exactly the quantity the MAP objective minimizes on reliable points and extrapolates through the prior on missing points; a metric on reliable points would be insensitive to the predicted effect. The design is falsifiable because if FlowCert's gains are uniform rather than concentrated on occlusion/corruption subsets, the causal story about constrained prior-based completion is wrong.

### Expected outcome and causal chain

**vs. ReLift** — On a case where a limb is occluded for a long window so lifting produces a drifting, low-confidence segment, ReLift's attention corrector produces a smooth but prior-agnostic interpolation that drifts toward the mean of its training distribution because it has no explicit motion prior or uniqueness constraint. Our method instead solves for a latent code under the flow prior with the observation likelihood anchoring reliable points, because the invertible flow supplies a structured motion manifold and the MAP objective balances fit against prior plausibility. We expect a noticeable gap on long-occlusion subsets but parity on short gaps, where ReLift's feed-forward bias is already adequate.

**vs. Lifted 2D trackers + geometry** — On a case where 2D lifting fails because depth ambiguity corrupts a fast-moving segment, the lifted baseline produces a trajectory with large position errors and no mechanism to correct them because it treats each frame's lift independently and has no temporal motion model. Our method instead completes and corrects the trajectory by conditioning on reliable frames through the flow prior, because the MAP problem couples reliable observations to missing points via the learned trajectory distribution. We expect FlowCert to show a large error reduction on corrupted segments and near-parity on clean segments, with the gap widening as the fraction of corrupted points grows.

**vs. Dedicated 3D tracker** — On a case where a dedicated tracker loses track under heavy occlusion, it produces a plausible but unconstrained continuation because it has no explicit uniqueness certificate and its internal state can drift. Our method instead returns a completion whose uniqueness is certified by the trajectory-space Hessian curvature, because the flow's Jacobian yields a computable lower bound on posterior curvature in output space. We expect comparable accuracy overall but a diagnostic pattern: FlowCert's certified trajectories should be more accurate than its uncertified ones, while the dedicated tracker shows no such internal signal, revealing the certificate as a meaningful reliability indicator.

**vs. FlowCert with latent-space certificate** — On a case where the flow is ill-conditioned in output space but well-conditioned in latent space, the latent-space certificate returns true for any flow and any observation pattern because the $+I$ term forces $\lambda_{\min} \ge 1 > \kappa$, so it cannot flag unreliable completions. Our method instead computes the certificate in trajectory space including the log-Jacobian term, so it can return false when the prior does not constrain the missing segment. We expect the trajectory-space certificate to have lower false positive rate on the calibration set and to better separate accurate from inaccurate completions than the latent-space certificate.

### What would falsify this idea

If FlowCert's accuracy gain over ReLift is uniform across occlusion length and corruption fraction rather than concentrated on long-occlusion and high-corruption subsets, the constrained-MAP-over-flow-prior causal story is wrong. If certified trajectories are no more accurate than uncertified ones on the calibration set, or if the trajectory-space certificate does not outperform the latent-space certificate in separating accurate from inaccurate completions, the certificate does not measure what we claim.

## References

1. ReLift: Dense 3D Tracking as Lifted Trajectory Repair
