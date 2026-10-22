# Coupled Trust-Region Value Caching for Online Robot Post-Training

## Motivation

Online value-guided post-training methods such as VaM and GR-RL replace the external stabilizer of a world model or human intervention with a learned value signal used as the sole policy-improvement signal, yet neither provides a closure rule that bounds cached-value error or ties the value-update rate to the policy-update rate. Structurally, the cached value is treated as ground truth while the policy is allowed to move arbitrarily far from the distribution on which that cache was certified, so mixed offline/off-policy/on-policy data injects three different bias terms into the cache that nothing bounds; the compounding error is invisible to the advantage-conditioned refinement loop because the loop has no notion of a certified value-error radius.

## Key Insight

A data-source-aware pair of mutual trust-region radii — the value-cache KL radius being a monotone function of the policy radius and the policy step being clipped by the cache's certified error — makes the coupled value-policy operator a contraction on a joint ball, so cached-value error cannot outrun the policy's own movement.

## Method

**What it is.** CTVC (Coupled Trust-region Value Caching) is an online post-training loop for robot policies that maintains a value cache together with a per-source error radius and updates value and policy inside mutually constrained trust regions. Inputs: offline demonstrations D_off, off-policy interventions D_int, on-policy rollouts D_on, a sparse reward. Outputs: an updated policy π_θ, an updated value cache V_φ, and per-source radii (δ_off, δ_int, δ_on).

**Load-bearing assumption (stated explicitly).** The empirical TD residual on a source s, plus a Hoeffding term, is treated as an upper bound on that source's contribution to the cached value's error on the current policy's support. This holds only under assumption A: (i) the source's state distribution is stationary within an iteration, and (ii) the advantage is Lipschitz in θ with constant L_A. It fails (failure mode F) when on-policy rollouts shift the state distribution mid-iteration or when the advantage is non-Lipschitz, in which case eps_s reflects transient distribution shift rather than cache bias. The method below makes A explicit, adds a diagnostic that flags when A breaks, and uses a lagged, measured policy KL to break the g/h feedback loop.

**How it works.**

```pseudocode
initialize π_θ from offline BC; V_φ from offline TD on D_off
initialize radii: r_pi <- r0=0.01, r_V <- g(r_pi)   # g monotone increasing
initialize lagged KL: KL_prev <- 0.0
for iteration t = 1..T:
    # 0. measure realized policy KL from previous step (lagged, verified quantity)
    KL_prev <- E_{D_mix}[ KL( π_θ_old || π_θ ) ]   # measured, not intended

    # 1. per-source value-cache error certificates (assumption A explicit)
    for s in {off, int, on}:
        eps_s <- empirical_TD_residual(V_φ, D_s) + beta_s * sqrt(log(1/alpha) / n_s)
        # beta_s = source bias coefficient (off: largest, on: smallest)
    eps_bar <- max_s (eps_s / w_s)             # w_s = mixture weight of source s

    # 1b. diagnostic for assumption A: stationarity / Lipschitz flag
    #     compute per-source state-distribution shift between iterations
    for s in {off, int, on}:
        shift_s <- MMD( D_s^{t-1}, D_s^{t} )   # kernel two-sample statistic
    A_flag <- ( max_s shift_s > tau_shift=0.05 ) OR ( L_A_est > L_max=10.0 )
    #     L_A_est = max_{s,a} ||∇_θ A(s,a)|| estimated on a 256-sample minibatch
    if A_flag:
        eps_bar <- eps_bar * (1 + gamma_flag=2.0)   # inflate certificate when A breaks

    # 2. coupled radius update (value radius monotone in *measured* policy KL)
    r_V     <- g(KL_prev) = c * KL_prev + eps_bar    # c > 0 coupling constant
    r_pi    <- h(r_V) = min(r_max, kappa * r_V / (1 + eps_bar))

    # 3. value-cache update inside KL ball of radius r_V, source-weighted
    V_φ <- argmin_V  sum_s w_s * E_{D_s}[TD(V)^2]
           s.t.  E_{D_mix}[ KL( V_φ(a|s) || V_{φ_old}(a|s) ) ] <= r_V

    # 4. policy update clipped by certified cache error
    A(s,a) <- V_φ(s,a) - V_φ(s, π_θ(s))       # advantage from cache
    L <- -E[ min( rho_t A, clip(rho_t, 1-r_pi, 1+r_pi) A ) ]
         + lambda * E[ KL( π_θ || π_θ_old ) ]
    theta <- theta - eta * grad L

    # 5. log (r_pi, r_V, eps_bar, KL_prev, A_flag, shift_s) for the contraction monitor
```

Hyperparameters set inline: coupling constant c=0.5, contraction factor kappa=0.9, radius cap r_max=0.1 (nats), source bias coefficients beta_off=1.0, beta_int=0.5, beta_on=0.1, confidence alpha=0.05, KL penalty lambda=0.01, learning rate eta=3e-4, initial policy radius r0=0.01, shift threshold tau_shift=0.05, Lipschitz cap L_max=10.0, flag inflation gamma_flag=2.0.

**Why this design.** We chose a monotone coupling function g(KL_prev)=c*KL_prev+eps_bar over an independent value-update schedule because the certified cache error eps_bar is exactly the quantity that should set the *floor* of the value radius, so tying the radius to the *measured* policy KL (not the intended r_pi) prevents the cache from drifting while the policy is stationary and breaks the positive feedback loop where a small eps_bar permits a large policy step that invalidates the certificate — accepting the cost that a stalled policy also freezes cache refinement. We chose a per-source bias coefficient beta_s over a single pooled residual because offline demos and on-policy rollouts contribute structurally different bias (covariate shift vs. reward sparsity), so a pooled estimate would under-certify the offline term; the cost is three extra hyperparameters that must be tuned per task family. We chose to clip the policy advantage by the certified cache error r_V (via h) rather than by a fixed PPO clip because a fixed clip cannot shrink when the cache is unreliable, whereas h(r_V) automatically tightens the policy step when eps_bar is large; the cost is that early training with noisy caches is slower. The diagnostic A_flag inflates eps_bar when the stationarity or Lipschitz premise is violated, making the certificate conservative rather than stale. A domain expert would not describe this as PPO-with-value-caching (VaM) because VaM has no coupling between the value radius and the policy radius — here the two radii are a single closed-loop object, and the clip is a function of the certificate, not a constant.

**Why it measures what we claim.** The empirical TD residual eps_s measures *per-source cache bias* because the assumption that TD residuals on a fixed data source upper-bound that source's contribution to cache error holds only when the source's state distribution is stationary within the iteration (assumption A-i); this assumption fails when on-policy rollouts shift the state distribution mid-iteration, in which case eps_on reflects transient distribution shift rather than cache bias — the A_flag diagnostic detects this via MMD shift and inflates eps_bar. The quantity r_V=g(KL_prev) measures *value-policy coupling strength* because the assumption that a monotone map from measured policy KL to value radius yields a joint contraction holds only when the policy KL is bounded by r_pi (enforced by the clip and the KL penalty) and the advantage is Lipschitz in θ (assumption A-ii); this assumption fails when the advantage is non-Lipschitz in the policy parameters, in which case r_V reflects a loose upper bound rather than the actual coupling — the L_A_est diagnostic flags this and inflates eps_bar. The certified error bound r_V used as the policy clip measures *policy-improvement safety* because the assumption that the cache advantage error is bounded by r_V on the current policy's support holds only when the policy stays inside the value KL ball; this assumption fails when the policy escapes the ball between iterations, in which case the clip reflects a stale certificate rather than current safety — using lagged KL_prev instead of intended r_pi ensures the certificate is conditioned on the distribution actually visited.

## Contribution

(1) A coupled trust-region value-caching algorithm (CTVC) in which the value-cache KL radius is a monotone function of the policy trust-region radius and the policy step is clipped by the cache's certified per-source error bound, forming a single closed-loop object rather than two independent schedules. (2) A data-source-aware error certificate that assigns distinct bias coefficients to offline, off-policy, and on-policy data, yielding a per-iteration contraction monitor that flags when the coupled iteration is no longer contracting. (3) An empirical design principle: the policy clip radius should be a function of the certified cache error rather than a fixed constant, which makes early-training policy steps automatically conservative when the cache is unreliable.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Long-horizon contact-rich manipulation suite (deformable + precise insertion) | Matches regime where value error compounds over horizon |
| Primary metric | Success rate on held-out long-horizon rollouts | Directly reflects cache reliability driving policy safety |
| Secondary metric | A_flag rate and eps_bar trajectory | Tests whether diagnostic flags assumption violations |
| Baseline | Matched offline BC + offline TD (no online loop) | Isolates whether online coupled post-training adds anything |
| Baseline | VaM-style value caching, uncoupled radii | Tests whether radius coupling is the causal ingredient |
| Baseline | Fixed-clip PPO with value cache | Tests whether certificate-driven clip beats constant clip |
| Ablation-of-ours | CTVC with independent value schedule (g removed) | Isolates the coupling function's contribution |
| Ablation-of-ours | CTVC without A_flag inflation (gamma_flag=1.0) | Isolates the diagnostic's contribution |

### Why this setup validates the claim

The central claim is that coupling the value radius to the *measured* policy KL through a certified per-source error floor is what makes online post-training safe on long-horizon tasks, rather than value caching or online RL per se. The dataset is chosen so that value error compounds across many decision steps, which is exactly the regime where a stale or drifting cache should hurt and where a certificate-driven clip should help. The matched offline baseline tests the sub-claim that online post-training is needed at all; the uncoupled VaM-style baseline tests the sub-claim that coupling, not caching, is the active ingredient; the fixed-clip PPO baseline tests the sub-claim that the clip must be a function of the certificate rather than a constant. The ablation removes only the coupling map g, so any gap between it and full CTVC is attributable to the mechanism we claim is load-bearing. The A_flag ablation tests whether the diagnostic that flags assumption violations (stationarity/Lipschitz) actually improves safety by inflating eps_bar when needed. Success rate on held-out long-horizon rollouts is the right metric because it integrates the compounding of value error into behavior; a metric like TD residual alone could improve while behavior degrades, and a short-horizon metric would not expose the compounding we predict. The secondary A_flag rate metric directly tests whether the diagnostic fires when assumption A breaks, providing evidence for the load-bearing assumption.

### Expected outcome and causal chain

**vs. Matched offline BC + offline TD** — On a case where the offline demonstration coverage leaves a narrow contact-rich segment under-represented, the offline baseline produces confident but systematically biased value estimates there because its TD targets never see corrective on-policy data, so the policy commits to a wrong contact strategy. Our method instead runs the online loop with per-source certificates, inflating eps_off and thus the value radius floor, which tightens the policy clip exactly on the under-covered segment because h(r_V) shrinks. We expect a clear success-rate gap concentrated on the long-horizon contact segments, with near-parity on the well-covered early segments.

**vs. VaM-style value caching, uncoupled radii** — On a case where the policy stalls briefly during a hard sub-task, the uncoupled baseline produces a value cache that keeps refining against a frozen policy, because its value radius is scheduled independently and never tied to policy motion, so the cache drifts toward off-policy targets and the advantage becomes miscalibrated. Our method instead ties r_V to KL_prev via g, so a stationary policy freezes cache refinement and the advantage stays anchored, and when the policy moves the clip reopens proportionally. We expect the uncoupled baseline to show a characteristic dip in success right after stall-then-resume transitions, while our method holds steady, producing a gap localized to those transitions rather than uniform.

**vs. Fixed-clip PPO with value cache** — On a case where early training has a noisy cache, the fixed-clip baseline produces large harmful policy steps because its constant clip cannot shrink when the cache is unreliable, so it takes gradient steps on miscalibrated advantages and destabilizes. Our method instead derives the clip from the certified error through h, so a large eps_bar automatically tightens the step and the policy improves slowly but monotonically. We expect the fixed-clip baseline to show high variance and occasional collapse in early training, while our method shows a lower but steadier success curve that overtakes it after the cache stabilizes, i.e. a crossover pattern rather than a uniform advantage.

**vs. CTVC without A_flag inflation** — On a case where on-policy rollouts shift the state distribution mid-iteration (violating stationarity), the no-flag ablation uses an uninflated eps_bar that underestimates cache error, so the policy clip is too loose and the policy takes unsafe steps on stale advantages. Our method inflates eps_bar when A_flag fires, tightening the clip and preventing those unsafe steps. We expect the no-flag ablation to show a higher rate of catastrophic failures (e.g., large drops in success) precisely on iterations where the MMD shift exceeds tau_shift, while full CTVC maintains steadier performance.

### What would falsify this idea

If the gain over the uncoupled VaM-style baseline and over the g-removed ablation is uniform across all task segments rather than concentrated on stall-then-resume transitions and under-covered contact segments, then the coupling is not the causal ingredient and the central claim is wrong. Likewise, if the fixed-clip baseline matches our method even in early noisy-cache training, the certificate-driven clip is doing no work. Additionally, if the A_flag diagnostic never fires when MMD shift exceeds tau_shift or when L_A_est exceeds L_max, then the diagnostic is not detecting assumption violations and the load-bearing assumption is unsupported.

## References

1. Value as Memory: Online Value-Guided Post-training for General Robot Policies
2. GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation
3. RL-100: Performant Robotic Manipulation with Real-World Reinforcement Learning
4. One-Step Diffusion Policy: Fast Visuomotor Policies via Diffusion Distillation
5. Diffusion Policy Policy Optimization
6. UFOGen: You Forward Once Large Scale Text-to-Image Generation via Diffusion GANs
7. Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models
8. One-Step Diffusion with Distribution Matching Distillation
