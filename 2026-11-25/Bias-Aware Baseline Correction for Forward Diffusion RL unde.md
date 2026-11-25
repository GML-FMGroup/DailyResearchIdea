# Bias-Aware Baseline Correction for Forward Diffusion RL under Behavior-Policy Mismatch

## Motivation

Forward-diffusion RL methods such as iTDM (E1) anchor their sample-level gradient at a behavior velocity tracked by an online auxiliary denoiser, and then subtract a baseline assuming that this auxiliary velocity equals the true behavior marginal velocity. That equality is an idealized infinitesimal-limit identity; under classifier-free-guidance (CFG) rollouts, offline/off-policy updates, and imperfect online self-consistency, the auxiliary velocity is biased, so baseline subtraction no longer cancels and injects gradient drift that destabilizes training. The field's forward-RL line therefore has no finite-sample account of when baseline invariance survives, and no mechanism that restores it once the denoiser is imperfect.

## Key Insight

The residual of the auxiliary-denoiser consistency constraint is a direct, observable estimate of the behavior-policy velocity mismatch, so it can be used as the argument of a bias-corrected baseline that is invariant to any denoiser error orthogonal to the score direction rather than requiring the idealized zero-error limit.

## Method

**What it is.** We propose BIC-iTDM (Bias-Invariant Corrected iTDM), a drop-in correction layer for forward-diffusion RL that takes (i) ODE rollouts from the current policy under a CFG scale, (ii) an online auxiliary denoiser, (iii) a frozen reference denoiser (the pre-RL checkpoint, or the CFG-unconditional branch), and (iv) a batch of offline/off-policy transitions, and outputs a bias-corrected baseline and a corrected sample-level policy gradient together with a scalar drift diagnostic.

**Load-bearing assumption (stated explicitly).** The correction assumes that the score direction used to project the velocity residual is unbiased, i.e., that the frozen reference denoiser provides a score direction aligned with the true behavior-marginal score, and that baseline cancellation fails only along that direction. This assumption fails when the reference denoiser is itself biased (e.g., under strong CFG or distribution shift), in which case the projected residual under-reports the drift. We do not claim this assumption holds universally; we verify it empirically on a synthetic benchmark with known ground-truth velocity (see experiment) and report the residual projection error.

**How it works.**

```pseudocode
# per training step, batch B of (x0, prompt, reward r)
# hyperparams: cfg_scale w, aux_lr eta, lam_cons, lam_base, kappa, ema_beta, ref_frozen=True

1. SAMPLE trajectories: for each x0, run deterministic ODE with CFG scale w
   -> states x_t, behavior velocity v_b(x_t) recorded at every visited t

2. AUXILIARY DENOISER UPDATE (online, EMA target):
   eps_aux <- eps_aux - eta * grad E[ || eps_aux(x_t,t,c) - eps_target(x_t,t,c) ||^2 ]
   eps_aux_ema <- ema_beta * eps_aux_ema + (1-ema_beta) * eps_aux

3. VELOCITY RESIDUAL (observable mismatch signal):
   for each visited (x_t,t):
     v_aux(x_t,t) = (x_t - sqrt(1-a_t) eps_aux_ema(x_t,t,c)) / sqrt(a_t)
     r_vec(x_t,t) = v_aux(x_t,t) - v_b(x_t,t)          # vector residual
     delta(x_t,t) = || r_vec(x_t,t) ||                  # scalar norm

4. BIAS-CORRECTED BASELINE (externally anchored projection):
   b_raw = mean over batch of r                       # standard RWR baseline
   # score direction from FROZEN reference denoiser (not eps_aux_ema)
   s_ref = score_normal_ref(x_t,t)                    # unit score direction from frozen eps_ref
   delta_par = < r_vec , s_ref >                      # signed component along fixed axis
   b_corr = b_raw - lam_base * tanh(kappa * delta_par) * (r - b_raw)
   # when delta_par = 0, b_corr = b_raw (reduces to iTDM baseline)

5. COUPLED AUXILIARY-CONSISTENCY LOSS:
   L_cons = lam_cons * mean( delta(x_t,t)^2 )
   L_total = L_policy(v_aux, b_corr, r) + L_cons

6. DRIFT DIAGNOSTIC (logged, not optimized):
   # independent measurement: compare gradients under b_corr vs b_raw on the SAME batch
   g_corr = mean_batch[ (r - b_corr) * grad_theta log pi ]
   g_raw  = mean_batch[ (r - b_raw)  * grad_theta log pi ]
   drift = || g_corr - g_raw ||
```

**Why this design.** We chose a residual-derived, signed baseline correction over simply lowering the CFG scale or freezing the auxiliary denoiser because the mismatch is intrinsic to the forward-RL objective (E1 reports it arises from CFG rollouts and off-policy updates themselves), accepting the cost that the correction must be recomputed every step and adds one projection per sample. We chose an EMA target denoiser over a second learned critic because a critic would introduce its own bias that the correction cannot observe, accepting slower adaptation of the auxiliary velocity. We chose a tanh-shrunk signed projection over a full regression of the bias because the projection is what enters the baseline-invariance identity (only the score-parallel component breaks cancellation), accepting that off-score-parallel residual is left uncorrected. We chose to couple the consistency loss with the policy loss rather than train the denoiser separately because joint training keeps the residual a faithful estimate of the current mismatch, accepting that the two losses must be balanced by lam_cons. This is not a variant of the signed reward-weighted regression baseline that iTDM (E1) already critiques: RWR subtracts a fixed reward baseline with no access to denoiser error, whereas BIC-iTDM makes the baseline a function of the observable velocity residual, so a domain expert would describe it as a bias-identification correction, not as RWR with a tuned baseline.

**Why it measures what we claim.** The computational quantity `delta` (the velocity residual norm) measures behavior-policy mismatch because the auxiliary denoiser is trained to match the behavior marginal velocity, so under the assumption that the EMA target is an unbiased estimator of that marginal, `delta` equals the mismatch norm; this assumption fails when the EMA lags under nonstationary CFG rollouts, in which case `delta` reflects lag plus mismatch rather than mismatch alone. The signed projection `delta_par` measures the component of mismatch that breaks baseline invariance because, under the assumption that the frozen reference score direction is unbiased and that baseline cancellation fails only along it, orthogonal residual components are invariant by construction; this assumption fails when the reference score estimate itself is biased, in which case `delta_par` under-reports the drift. The quantity `drift` measures the actual gradient perturbation because it is the difference between the corrected and uncorrected batch gradients under the same samples; this assumption fails when the batch is too small, in which case `drift` reflects sampling noise rather than systematic bias.

## Contribution

['A bias-corrected baseline operator for forward-diffusion RL that uses the auxiliary-denoiser velocity residual as an observable mismatch signal and reduces exactly to the iTDM baseline when the residual vanishes.', 'A coupled auxiliary-consistency plus policy loss that keeps the residual a faithful per-step estimate of behavior-policy mismatch under CFG and off-policy regimes, together with a logged gradient-drift diagnostic.', 'A controlled empirical protocol that sweeps CFG scale and offline/on-policy ratio to characterize when baseline invariance holds and when it fails.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Synthetic 2D diffusion model with analytic reward and known ground-truth marginal velocity (primary); Stable Diffusion v1.5 prompt-reward alignment (secondary) | Synthetic allows exact computation of true gradient drift and verification of the load-bearing assumption; SD v1.5 matches E1/E7 CFG-rollout setting |
| Primary metric | Reward improvement per unit gradient-drift (drift computed as ||g_corr - g_raw|| on same batch) | Directly couples alignment gain to the predicted bias mechanism |
| Baseline | Signed RWR / vanilla iTDM (E1) | Tests whether residual-derived baseline beats fixed baseline |
| Baseline | Low-CFG-scale forward RL | Tests whether correction beats simply shrinking CFG mismatch |
| Baseline | Frozen auxiliary denoiser | Tests whether joint consistency training matters |
| Baseline | BIC-iTDM with self-referential projection (s_hat from eps_aux_ema) | Tests whether external anchoring is necessary vs. circular projection |
| Ablation-of-ours | BIC-iTDM with lam_base = 0 (no correction) | Isolates the baseline-correction term from the rest |
| Ablation-of-ours | BIC-iTDM with kappa = 0 (tanh disabled) | Tests sensitivity to shrinkage strength |

### Why this setup validates the claim

The central claim is that behavior-policy mismatch under CFG rollouts injects baseline-induced gradient drift, and that a residual-derived signed baseline correction removes it. The synthetic benchmark is essential because it provides ground-truth marginal velocity and true gradient drift, allowing direct verification of the load-bearing assumption that the frozen reference score direction is unbiased and that delta_par measures the baseline-invariance-breaking component. Without this controlled check, the SD v1.5 experiments cannot distinguish whether the correction works or merely reweights gradients. The SD v1.5 fine-tuning setting of E1 and E7 then tests scalability under realistic CFG rollouts and offline/off-policy updates. Vanilla iTDM (E1) is the decisive baseline because it is the method our correction modifies: if our gain over it is real, it must come from the baseline term, not from the policy loss, which is why the lam_base = 0 ablation is essential — it holds everything else fixed and removes only the correction. Low-CFG-scale forward RL tests the competing hypothesis that the mismatch is simply a scale artifact, and the frozen-denoiser baseline tests whether the coupled consistency loss matters. The self-referential projection baseline directly tests the adversarial alert: if external anchoring is unnecessary, the self-referential variant should match ours; if it fails, the anchoring is load-bearing. The primary metric, reward per unit drift, is chosen because drift is exactly the diagnostic our method claims to reduce; a method that improved reward while drift stayed flat would be succeeding for a different reason, which the metric would expose.

### Expected outcome and causal chain

**vs. Signed RWR / vanilla iTDM** — On a case where CFG scale is high and the offline batch is stale, the baseline subtracts a fixed reward mean, so its gradient retains a score-parallel component that does not cancel, producing oscillating or drifting updates. Our method instead projects the velocity residual onto the frozen reference score direction and shrinks the baseline by that signed component, so the residual drift term is attenuated where it actually lives. We expect a noticeable reward gap on high-CFG, off-policy-heavy subsets but near-parity on low-CFG on-policy subsets, with drift diagnostic roughly halved on the former. On the synthetic benchmark, we expect the true gradient drift (computed with ground-truth velocity) to be reduced by at least 50% relative to vanilla iTDM when the auxiliary denoiser is deliberately biased.

**vs. Low-CFG-scale forward RL** — On a case where lowering CFG reduces mismatch but also weakens prompt adherence, the baseline gains stability at the cost of alignment quality, so its reward saturates below the guided regime. Our method keeps the high CFG scale and corrects the induced bias directly, so it should retain guidance strength. We expect the baseline to show lower drift but also lower terminal reward, while ours shows high reward at comparable drift — a divergence in the reward-drift plane rather than a uniform win.

**vs. Frozen auxiliary denoiser** — On a case where the policy shifts during training, a frozen denoiser's velocity estimate lags the behavior marginal, so the residual `delta` increasingly reflects lag rather than mismatch and the correction mis-fires. Our method's coupled consistency loss keeps the denoiser tracking the current marginal. We expect the frozen variant to degrade over training steps while ours stays flat, visible as a widening gap in later training rather than at initialization.

**vs. BIC-iTDM with self-referential projection** — On the synthetic benchmark with a deliberately biased auxiliary denoiser, the self-referential variant uses a rotated score axis, so delta_par measures bias projected onto a biased axis and the correction under-reports or mis-signs the drift. Our externally anchored variant uses the frozen reference score direction, which is unbiased by construction in the synthetic setup. We expect the self-referential variant to show higher true gradient drift and lower reward than ours, with the gap increasing with the magnitude of denoiser bias. If the self-referential variant matches ours, the external anchoring is not necessary and the load-bearing assumption is weaker than claimed.

### What would falsify this idea

If the gain over vanilla iTDM is uniform across CFG-scale and on/off-policy subsets rather than concentrated where the score-parallel residual is predicted to be large, the baseline-correction mechanism is not what produces the improvement. Likewise, if the drift diagnostic stays flat while reward rises, the claimed causal link between mismatch and drift is wrong. On the synthetic benchmark, if the externally anchored correction does not reduce true gradient drift relative to the self-referential variant when the denoiser is biased, the load-bearing assumption fails.

## References

1. Reinforcing Diffusion Models by Instantaneous Trajectory Distribution Matching
