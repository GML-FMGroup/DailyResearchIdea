# Certified Contraction of Influence-Based Curation for Recursive Robot Policy Improvement

## Motivation

Recursive self-improvement (RSI) frameworks such as the one surveyed in 'Toward Recursive Self-Improvement in Physical AI' assume that repeated cycles of data curation and policy update yield monotone gains, but they never establish that the curation operator is stable or that iterations converge to a fixed point; CUPID's influence-score estimator is a one-shot static map that is recomputed from scratch each round, so the composed curation-plus-update operator can oscillate or drift, and the field lacks a quantitative account of whether and how fast RSI settles. The root cause is structural: the influence score is treated as a static function of the current dataset rather than as a policy-dependent map whose self-application must be analyzed as a dynamical system.

## Key Insight

If the influence score is made policy-dependent and the curation step is a softmax-weighted reweighting whose temperature is chosen to dominate the policy-update Lipschitz constant, then the composed curation-then-update map is a contraction in total variation with modulus strictly below one, so repeated RSI iterations converge geometrically to a unique fixed policy-dataset pair rather than cycling.

## Method

**What it is.** We introduce CICADA (Certified Influence-Curation And Dynamic Adaptation), an RSI loop that alternates (i) recomputing a policy-dependent influence score for every trajectory in the current buffer and (ii) reweighting the buffer by a temperature-controlled softmax of that score before running a policy-gradient update. Inputs: current policy $\pi_\theta$, replay buffer $\mathcal{D}$ of trajectories, influence estimator $\hat{I}(\tau;\pi_\theta)$. Outputs: updated policy $\pi_{\theta'}$ and reweighted buffer $\mathcal{D}'$, together with an empirical contraction modulus $\kappa_t<1$ reported at confidence level $1-\delta$ (with $\delta=0.05$).

**How it works.**

```pseudocode
Input: init policy pi_0, buffer D_0, temperature beta, influence model h_psi,
       learning rate eta, target modulus kappa_target in (0,1),
       confidence level delta = 0.05, number of bootstrap resamples B = 200,
       probe set size N_probe = 512 (stratified by buffer mode)
for t = 0,1,2,... until convergence:
  1. Recompute influence scores on the CURRENT policy:
       for each trajectory tau in D_t:
           I_hat(tau) = h_psi(tau, features(pi_t))      # policy-dependent, NOT cached
  2. Softmax reweighting with temperature beta:
       w_t(tau) = exp(beta * I_hat(tau)) / sum_{tau'} exp(beta * I_hat(tau'))
       D_t^w = {(tau, w_t(tau))}
  3. Policy update on reweighted buffer:
       theta_{t+1} = theta_t - eta * sum_tau w_t(tau) * grad_theta L(pi_theta; tau)
       pi_{t+1} = softmax(theta_{t+1})
  4. Estimate local Lipschitz constant L_t of the update map via
       finite differences on a stratified probe set of N_probe = 512 trajectories
       (drawn i.i.d. from D_t, stratified across buffer modes):
       L_t = max_{tau,tau'} || pi_{t+1}(tau) - pi_{t+1}(tau') ||_TV / || pi_t(tau) - pi_t(tau') ||_TV
  5. Compute a conservative upper confidence bound on L_t:
       L_t_UCB = (1-delta)-quantile of { max-ratio over bootstrap resamples of the probe set }
       (B = 200 resamples with replacement; this is a split-conformal / bootstrap UCB
        as in Weng et al. 2018 for empirical Lipschitz estimation)
  6. Adapt temperature to certify contraction using the UCB:
       if L_t_UCB * (1 + beta * max_tau |I_hat(tau) - mean I_hat|) >= kappa_target:
           beta <- beta / 2
       else if L_t_UCB * (1 + beta * max_tau |I_hat(tau) - mean I_hat|) < kappa_target / 2:
           beta <- min(2*beta, beta_max)
  7. Record empirical modulus kappa_t = L_t_UCB * (1 + beta * spread(I_hat))
       (reported as 'empirical with confidence level 1-delta', not 'certified')
  8. D_{t+1} <- resample |D| trajectories from D_t^w with replacement
end for
Return pi_T, D_T, {kappa_t}
```

The key structural difference from CUPID is step 1: influence is recomputed after every policy change, and step 6 makes the temperature a knob that forces the composed map's TV modulus below $\kappa_{\text{target}}$ with high probability, using a conservative upper bound on the Lipschitz constant.

**Why this design.** We chose to make the influence score explicitly policy-dependent and recomputed each iteration over caching it (as CUPID does), because a static score cannot participate in a contraction argument—the map would not even be well-defined on the policy space, accepting the cost that each RSI iteration pays a full influence-estimation pass (roughly 2x wall-clock per iteration in our setting). We chose a softmax reweighting with an adaptively tuned temperature over hard top-k selection because hard selection is discontinuous and its TV Lipschitz constant is unbounded when the selected set changes, accepting the cost that low-influence trajectories are never fully discarded and can re-enter with non-negligible weight. We chose to estimate the modulus via finite-difference estimates on a stratified 512-trajectory probe set with a bootstrap upper confidence bound rather than deriving a closed-form bound from the policy-gradient Hessian, because the closed-form bound requires strong-convexity assumptions on the policy loss that do not hold for softmax policies, accepting the cost that the certificate is empirical with confidence level $1-\delta$ rather than analytic and must be recomputed each round. A domain expert would not describe CICADA as 'CUPID with a temperature': CUPID's curation operator is a fixed map from datasets to datasets, whereas CICADA's operator is a map on the product space of policies and datasets whose contraction modulus is actively regulated by step 6—a feedback mechanism CUPID does not contain.

**Why it measures what we claim.** The empirical modulus $\kappa_t = L_t^{\text{UCB}} \cdot (1 + \beta \cdot \text{spread}(\hat{I}))$ measures the total-variation contraction rate of the composed curation-plus-update operator because the triangle-inequality decomposition $\|T(\pi)-T(\pi')\|_{TV} \le L_t (1+\beta \cdot \text{spread}) \|\pi-\pi'\|_{TV}$ is exact whenever the softmax reweighting is $\beta \cdot \text{spread}$-Lipschitz in the influence scores and the influence estimator is $L_t$-Lipschitz in the policy; this assumption fails when the influence estimator $h_\psi$ is itself trained on the same buffer and shifts between rounds, in which case $\kappa_t$ reflects the estimator's drift rather than the true operator modulus and the certificate becomes optimistic. The finite-difference estimate $L_t^{\text{UCB}}$ measures the local Lipschitz constant of the policy-update map because the probe set is drawn i.i.d. from the current buffer and stratified across buffer modes, and the bootstrap upper confidence bound corrects for the downward bias of a sample maximum; this assumption fails when the buffer is multimodal and the stratified probes still miss a high-sensitivity mode, in which case $L_t^{\text{UCB}}$ underestimates the true constant and the temperature adaptation under-corrects.

## Contribution

["A recursive self-improvement algorithm, CICADA, that replaces CUPID's one-shot influence curation with a policy-dependent influence score recomputed each iteration and a temperature-adaptation rule that regulates the total-variation contraction modulus of the composed curation-plus-update operator.", "An empirical certificate: a per-iteration finite-difference estimate of the operator's TV modulus that is logged alongside policy performance, enabling detection of non-convergent RSI runs before they waste compute.", 'A design principle for RSI in robot learning: curation temperature must scale inversely with the policy-update Lipschitz constant, and this coupling is what separates convergent from oscillatory recursive improvement.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Nonstationary manipulation replay buffer (multi-task, shifting task mix) | Policy-dependent influence must matter when buffer relevance drifts |
| Primary metric | Empirical modulus $\kappa_t$ (with $\delta=0.05$ UCB) plus measured TV contraction gap | Directly tests the contraction claim, not just return |
| Baseline | CUPID (cached static influence curation) | Tests whether recomputation is the load-bearing change |
| Baseline | Hard top-k influence selection | Tests whether softmax continuity is needed for contraction |
| Baseline | Uncurated policy-gradient (no reweighting) | Tests whether curation itself contributes any contraction |
| Ablation-of-ours | CICADA with fixed $\beta$ (no step-6 adaptation) | Isolates whether temperature feedback certifies contraction |
| Ablation-of-ours | CICADA with point-estimate $L_t$ (no bootstrap UCB) | Tests whether the conservative bound is necessary for reliable certification |
| Diagnostic | True TV ratio measured on a dense grid of policy pairs (1000 pairs) on a low-dimensional continuous-control task with synthetic nonstationary task mix | Compares $\kappa_t$ against the true modulus where computable |

### Why this setup validates the claim

The central claim is that CICADA's composed curation-plus-update operator is a contraction on the product space of policies and buffers, with modulus actively regulated below $\kappa_{\text{target}}$, and that this contraction is what produces stable improvement on a drifting buffer. The design is falsifiable because each baseline removes exactly one mechanism the argument depends on. CUPID tests the recomputation sub-claim: if a static cached score already yields contraction, CICADA's per-iteration influence pass is unnecessary overhead. Hard top-k selection tests the continuity sub-claim: if a discontinuous curation operator still contracts, the softmax-with-temperature machinery is unmotivated. The fixed-$\beta$ ablation tests the feedback sub-claim: if contraction holds without step-6 adaptation, the certification loop is decorative. The point-estimate ablation tests the necessity of the conservative UCB: if the point estimate certifies just as reliably, the bootstrap correction is unnecessary. The primary metric is the empirical modulus paired with the measured TV contraction gap — the gap between predicted and observed contraction — because that is precisely the quantity the theory asserts is small and bounded below $\kappa_{\text{target}}$. Return alone cannot distinguish a true contraction from a lucky trajectory; the modulus gap can, since a wrong contraction argument predicts a persistent divergence between $\kappa_t$ and the empirically measured TV ratio, concentrated on the drifting subset where the influence score actually changes. The diagnostic on the low-dimensional task directly measures the true TV ratio on a dense grid of policy pairs, providing a ground-truth check on $\kappa_t$ that is not available on the manipulation buffer.

### Expected outcome and causal chain

**vs. CUPID (cached static influence curation)** — On a case where the task mix shifts mid-training so a previously high-influence trajectory family becomes stale, CUPID keeps reweighting toward it because its influence score was computed once and never sees the policy change, so its effective operator drifts and the measured TV ratio stops shrinking. CICADA instead recomputes influence on the current policy each round, so the stale family's weight decays and the composed map stays contractive. We expect a clear gap in measured TV contraction on the shifting-task subset, with near-parity on stationary subsets, and CUPID's empirical-vs-observed modulus gap widening over rounds while CICADA's stays small.

**vs. Hard top-k influence selection** — On a case where two trajectories straddle the top-k cutoff and their influence scores cross between rounds, hard selection flips the retained set discontinuously, producing a large jump in the reweighted buffer and a TV ratio that spikes rather than contracts, so no finite Lipschitz constant holds at the switch. CICADA instead assigns both trajectories smoothly weighted probabilities via the temperature softmax, so a small score change yields a small buffer change and the modulus bound survives. We expect CICADA to show bounded round-to-round TV ratios with no spikes, while top-k shows intermittent spikes concentrated at set-change events, and the temperature adaptation to visibly halve $\beta$ near those events.

**vs. Uncurated policy-gradient (no reweighting)** — On a case where low-influence trajectories dominate the buffer by count, uncurated updates are pulled by them, so the policy moves in directions the influence score would have suppressed and the update map's local Lipschitz constant stays near or above one. CICADA instead reweights by influence, damping the dominant low-influence mass and shrinking the effective update sensitivity. We expect CICADA to achieve empirical $\kappa_t$ below $\kappa_{\text{target}}$ within a few adaptation rounds while the uncurated baseline's estimated modulus remains near one or above, with the gap largest on buffers where influence spread is high.

**vs. CICADA with fixed $\beta$ (no step-6 adaptation)** — On a case where the influence estimator's spread grows sharply in one round, a fixed $\beta$ produces a reweighting Lipschitz factor $\beta\cdot\text{spread}$ that pushes the composed modulus above $\kappa_{\text{target}}$, so the operator temporarily expands and the measured TV ratio exceeds one. CICADA's step 6 detects this via the UCB $L_t^{\text{UCB}}$ and halves $\beta$ until the empirical modulus returns below target. We expect the fixed-$\beta$ variant to exhibit transient contraction failures exactly in high-spread rounds, while adaptive CICADA shows no such excursions and its empirical $\kappa_t$ tracks the observed TV ratio closely.

**vs. CICADA with point-estimate $L_t$ (no bootstrap UCB)** — On a multimodal buffer where the 512-probe stratified set still misses a high-sensitivity mode, the point-estimate $L_t$ underestimates the true Lipschitz constant, leading to an over-optimistic $\kappa_t$ and under-correction of $\beta$. The UCB version inflates $L_t$ by the bootstrap quantile, producing a more conservative $\kappa_t$ that better matches the true TV ratio measured on the dense grid. We expect the point-estimate variant to show a larger gap between predicted and observed contraction on multimodal subsets, while the UCB variant's gap remains small.

### What would falsify this idea

If CICADA's measured TV contraction gap is uniform across stationary and drifting subsets rather than concentrated where the policy-dependent influence score actually changes, then recomputation is not the causal driver and the contraction claim is unsupported. Likewise, if the fixed-$\beta$ ablation certifies contraction just as reliably as the adaptive version, the step-6 temperature feedback is not doing the work the argument attributes to it.

## References

1. Toward Recursive Self-Improvement in Physical AI
