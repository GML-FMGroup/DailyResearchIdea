# Coverage-Certified Value Cache with Certificate-Triggered Exploration for Online Robot Policy Post-training

## Motivation

Online value-guided post-training methods such as VaM (Value as Memory) and GR-RL cache a progress/value signal and update it online, but both inherit the assumption that this signal is trained on a fixed task distribution and single embodiment, with no formal condition stating when the cached estimate remains valid under distribution shift. Structurally, the cache is passively trained on whatever the current task distribution provides, so it has no mechanism to detect that the state-action distribution has drifted outside its training support — precisely the regime where the cached progress signal silently becomes misleading and the online update propagates corrupted credit. This is a forest-level gap: the field already performs online cache updates and has the data needed to estimate density ratios, but lacks the coverage closure condition that would make the cached signal trustworthy off-distribution.

## Key Insight

A density-ratio coverage certificate estimated from the cache's own training support yields a checkable per-query validity region, and coupling the cache's update/exploration schedule to the certificate's margin keeps the trusted region closed under the evolving state-action distribution rather than assuming it.

## Method

**What it is.** COVE-VC (COVerage-cErified Value Cache) augments an online value-cache post-training loop (VaM-style) with a per-query coverage certificate. Inputs: current state-action query (s,a), the cached value estimate V_cache(s,a), a density-ratio estimator r(s,a) trained adversarially against a **frozen reference support buffer D_cache_ref** (a held-out subset of the cache's training transitions, never updated online), and a conservative lower bound V_LB(s,a) from an ensemble of K=5 pessimistic Q-functions. Outputs: a trusted value signal V_used(s,a) fed to the policy, plus an exploration target when the certificate margin is thin.

**Load-bearing assumption (stated explicitly).** The certificate assumes that the density ratio r(s,a) produced by the discriminator, when thresholded at a calibrated value RHO_MAX, is a per-query indicator of whether V_cache(s,a) is trustworthy. This assumption is made checkable by (i) freezing the reference distribution D_cache_ref so that r(s,a) is not self-confirming against the drifting online distribution, and (ii) calibrating RHO_MAX on a held-out calibration set with known cached-value error, as detailed below.

**How it works.**

```pseudocode
# Buffers: D_cache = transitions the value cache was trained on
#          D_cache_ref = fixed held-out subset of D_cache (20% of transitions), NEVER updated online
#          D_cal = calibration set of 512 queries with measured cached-value error |V_cache - V_true|
#          D_online = fresh on-policy/intervention data
# Ensemble of K=5 pessimistic critics {Q_k}; density-ratio net r_phi (2-layer MLP, hidden=256, GeLU)

# One-time calibration (before online loop):
#   Train r_phi on D_cache_ref (label 1) vs D_online_initial (label 0) via logistic loss.
#   For each query in D_cal, compute rho = r_phi(s,a)/(1 - r_phi(s,a)).
#   Set RHO_MAX = the largest threshold such that
#       P(|V_cache - V_true| > epsilon | rho <= RHO_MAX) <= delta,
#   with epsilon = 0.1 * std(V_cache) and delta = 0.1.
#   (If no threshold satisfies this, set RHO_MAX to the 90th percentile of rho on D_cal and flag calibration failure.)

for each online iteration t:
    # 1. Certificate estimation (every C=200 steps, on a minibatch of 256 queries)
    #    Train r_phi to discriminate D_cache_ref (label 1) vs D_online (label 0)
    #    via logistic loss; r(s,a) = r_phi(s,a)/(1 - r_phi(s,a))  # density ratio
    #    NOTE: D_cache_ref is frozen; only D_online changes.
    # 2. Per-query certified value
    for each query (s,a) in policy batch:
        rho = r(s,a)
        certified = (rho <= RHO_MAX)          # coverage certificate holds (calibrated threshold)
        V_used = V_cache(s,a) if certified else V_LB(s,a)
    # 3. Cache update schedule tied to certificate
    if fraction of batch with rho > RHO_MAX > TAU_FAIL (0.25):
        # certificate about to fail broadly -> targeted expansion
        launch exploration rollouts maximizing rho(s,a) (density-ratio bonus weight beta=0.5)
        add their transitions to D_cache and refit cache on D_cache
    else:
        standard VaM online update on mixed D_cache + D_online
    # 4. Policy update uses V_used as the progress signal
    update policy with policy gradient on V_used; update critics on returns
```

Hyperparameters: RHO_MAX (coverage threshold, calibrated as above; typical value ~2.0), TAU_FAIL (batch failure fraction, 0.25), C (certificate refresh period, 200 steps), K (critic ensemble size, 5), exploration bonus weight beta (0.5), calibration set size 512, epsilon=0.1*std(V_cache), delta=0.1.

**Why this design.** We chose an adversarial density-ratio estimator over a parametric Gaussian/KDE support model because the cache's support lives in high-dimensional visuomotor state-action space where parametric densities are misspecified, accepting the cost that the ratio estimate has finite-sample bias that RHO_MAX must absorb. We chose a per-query certificate over a single global drift statistic because a global test cannot localize where the cache is untrustworthy, accepting the cost of per-query inference overhead. We chose to couple the cache update schedule to the certificate (triggering targeted exploration when the batch failure fraction exceeds TAU_FAIL) rather than passively training on the current distribution as VaM does, because passive training cannot close the coverage gap it is supposed to close, accepting the cost that exploration rollouts consume real-robot budget. We chose a pessimistic ensemble lower bound V_LB for out-of-certificate queries rather than extrapolating V_cache, because extrapolation is exactly the unverified regime, accepting the cost that V_LB is conservative and slows early progress. A domain expert would not describe this as VaM plus a density estimator: VaM's value cache is a passive memory trained on whatever data arrives, whereas here the certificate is a load-bearing control variable that both gates the value signal and drives the data-collection schedule — the cache is actively expanded where the certificate margin is thin, which is a closed-loop invariant VaM does not establish. The frozen reference buffer D_cache_ref and calibrated RHO_MAX are what make the certificate a fixed reference rather than a self-confirming statistic.

**Why it measures what we claim.** The density ratio r(s,a) measures distributional coverage validity of the cached progress signal because the assumption that the adversarial discriminator's logit recovers the true density ratio (the f-divergence variational bound, exact when the discriminator is Bayes-optimal) makes rho<=RHO_MAX equivalent to the query lying in the cache's training support; this assumption fails when the discriminator overfits the finite support buffer, in which case rho reflects discriminator memorization rather than coverage and the certificate becomes over-permissive. To mitigate this, we freeze the reference distribution D_cache_ref and calibrate RHO_MAX against measured cached-value error on D_cal, so that the threshold is empirically grounded rather than assumed from the variational bound. The batch failure fraction (fraction with rho>RHO_MAX) measures imminent coverage collapse because the assumption that the online state-action distribution evolves continuously relative to the refresh period C makes a rising failure fraction a leading indicator of support exit; this assumption fails under abrupt task switches, in which case the fraction jumps without warning and exploration is triggered one iteration late. The pessimistic lower bound V_LB measures a conservative progress signal outside the certified region because the assumption that the critic ensemble brackets the true value under the pessimism penalty makes min_k Q_k a lower bound on achievable progress; this assumption fails when all ensemble members share a correlated bias, in which case V_LB reflects shared bias rather than a conservative bound.

## Contribution

(1) A coverage-certified value-cache mechanism (COVE-VC) that attaches a per-query density-ratio validity certificate to every cached progress signal and substitutes a pessimistic ensemble lower bound outside the certified region. (2) A certificate-coupled update schedule in which the cache's expansion is triggered by the batch coverage-failure fraction rather than passive training, closing the coverage loop between value estimation and data collection. (3) A design principle that value-cache validity under distribution shift should be treated as a maintained invariant with an explicit failure threshold, not an inherited assumption.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Long-horizon deformable + contact-rich real-robot tasks (VaM suite) | Where value-cache drift and support exit are most likely |
| Primary metric | Success rate on queries stratified by coverage ratio rho (bins: rho <= RHO_MAX, RHO_MAX < rho <= 2*RHO_MAX, rho > 2*RHO_MAX) | Directly diagnoses the certificate's predictive claim |
| Secondary metric | AUROC of rho > RHO_MAX against empirically measured value-signal error | Tests the certificate's predictive validity as a standalone diagnostic |
| Baseline | Matched offline training (VaM baseline) | Tests whether online cache post-training helps at all |
| Baseline | VaM (passive value cache, no certificate) | Isolates the certificate's gating + schedule contribution |
| Baseline | Pessimistic-ensemble-only (V_LB everywhere, no cache) | Tests whether the cache adds value over pure conservatism |
| Ablation-of-ours | COVE-VC without certificate-triggered exploration (gate only) | Isolates the closed-loop data-expansion claim |
| Ablation-of-ours | COVE-VC with uncalibrated RHO_MAX (fixed at 2.0) | Tests the contribution of calibration |

### Why this setup validates the claim

The central claim is that the coverage certificate is a load-bearing control variable: it both gates the value signal and drives data collection, and this closed loop is what keeps the cached progress signal trustworthy as the policy drifts. The stratified success-rate metric is chosen because the claim is inherently conditional — the certificate asserts that queries inside support are trustworthy and queries outside are not, so a uniform success metric cannot distinguish a correct certificate from an inert one. The AUROC metric directly measures whether the certificate's per-query prediction of value-signal error is better than chance, providing a standalone diagnostic that does not depend on the downstream policy. VaM (passive cache) tests the sub-claim that gating alone matters; the pessimism-only baseline tests whether the cache contributes anything beyond conservative lower bounds; the gate-only ablation tests the specific sub-claim that coupling exploration to the certificate, rather than merely gating, is what closes the coverage gap; and the uncalibrated-RHO_MAX ablation tests whether the calibration procedure is necessary. Together these form a falsifiable design: if the certificate is load-bearing, gains must be concentrated on the high-rho stratum and the ablation must lose exactly there.

### Expected outcome and causal chain

**vs. Matched offline training (VaM baseline)** — On a long-horizon cloth-folding segment where the policy drifts into states absent from the offline cache, the baseline's cached progress signal silently extrapolates and the policy receives a misleadingly confident value, producing stalls and mistimed grasps. Our method instead substitutes the pessimistic lower bound once rho exceeds RHO_MAX, so the policy gets a conservative but directionally correct signal and the batch failure fraction trips exploration. We expect a modest overall gap but a pronounced gap on the late-horizon, high-rho subset where the baseline's extrapolation dominates.

**vs. VaM (passive value cache)** — On a contact-rich unscrewing instance where the online distribution slowly exits the cache's support, VaM keeps training on whatever arrives and its cache drifts with the policy, so the value signal becomes self-confirming and the policy over-commits to a failing grasp mode. Our method detects the rising rho fraction and launches density-ratio-bonus rollouts that re-anchor the cache before collapse. We expect VaM to match us early and diverge sharply after the first coverage-exit event, with our advantage concentrated in the post-exit horizon window.

**vs. Pessimistic-ensemble-only** — On a deformable-object task where the ensemble's pessimism penalty is uniformly applied, the baseline is safe but slow, refusing to exploit well-covered states where the cache is actually reliable. Our method uses the cache inside the certified region, so it recovers the aggressive progress signal exactly where it is valid. We expect the pessimism-only baseline to show parity with us on high-rho queries but a clear deficit on low-rho queries, the mirror image of the VaM comparison.

**vs. COVE-VC without certificate-triggered exploration (gate only)** — On a task where the policy drifts into a new region, the gate-only ablation substitutes V_LB but never expands the cache, so the trusted region shrinks over time and the policy is stuck with conservative values. Our method triggers exploration and expands the cache, keeping the trusted region closed. We expect the gate-only ablation to match us initially but fall behind as the high-rho fraction grows, with the gap concentrated in the high-rho stratum.

**vs. COVE-VC with uncalibrated RHO_MAX** — With a fixed threshold of 2.0, the certificate may be over-permissive or over-conservative depending on the discriminator's scale. Our calibrated threshold adapts to the measured value error. We expect the uncalibrated version to show either more failures on high-rho queries (if over-permissive) or slower progress on low-rho queries (if over-conservative), and the calibrated version to dominate on the stratified success metric.

### What would falsify this idea

If our gain over VaM is uniform across the low-rho and high-rho strata rather than concentrated where the certificate predicts, the certificate is not carrying the causal load and the effect is just generic conservatism. Likewise, if the gate-only ablation matches full COVE-VC on the high-rho stratum, the certificate-triggered exploration loop is not doing the work we claim. Additionally, if the AUROC of rho > RHO_MAX against measured value error is not significantly above 0.5, the certificate is not a valid per-query diagnostic.

## References

1. Value as Memory: Online Value-Guided Post-training for General Robot Policies
2. GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation
3. RL-100: Performant Robotic Manipulation with Real-World Reinforcement Learning
4. One-Step Diffusion Policy: Fast Visuomotor Policies via Diffusion Distillation
5. Diffusion Policy Policy Optimization
6. UFOGen: You Forward Once Large Scale Text-to-Image Generation via Diffusion GANs
7. Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models
8. One-Step Diffusion with Distribution Matching Distillation
