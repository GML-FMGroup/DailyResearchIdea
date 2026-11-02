# Contraction-Mapped Curriculum RL for Latent Multimodal Reasoning

## Motivation

Latent reasoning policies in unified multimodal models (e.g., LARC) are trained by shifting from token-level to latent-space chain-of-thought via curriculum SFT and information-gain RL, but the field lacks any argument that the latent policy converges as the curriculum schedule changes. The gap is structural: curriculum shift is a non-stationary process, and without a contraction property the latent policy can oscillate or drift, so scale-emergence is assumed rather than derived. LARC's own limitation—sensitivity to reward design in self-evolved latent reasoning—is a symptom of this missing stability condition.

## Key Insight

A curriculum schedule that contracts the latent policy's Bellman operator at a rate faster than the schedule's non-stationarity induces a unique stable fixed point, so convergence follows from the geometry of the shift rather than from model scale.

## Method

**What it is.** CM-CRL (Contraction-Mapped Curriculum RL) trains a latent reasoning policy in a unified multimodal model by treating curriculum shift as a time-inhomogeneous Markov decision process whose transition kernel is indexed by a scalar curriculum stage $s_t \in [0,1]$ (0 = pure token-level CoT, 1 = pure latent reasoning). It outputs a latent policy $\pi_{\theta}(z \mid x, s)$ and a contraction modulus $\kappa_t$ that upper-bounds the stage-to-stage change in the latent Bellman operator. **Load-bearing assumption (stated explicitly):** the convergence argument requires a *uniform* contraction over the entire stage-$s_t$ latent support; the quantity we compute, $\kappa_t$, is a batch-wise max of Lipschitz ratios on a finite, stale replay buffer $B$, and we assume $B$ covers the stage-$s_t$ latent support with refresh rate faster than support drift. This assumption fails when the latent support shifts faster than the buffer refresh rate, in which case $\kappa_t$ reflects buffer staleness rather than operator contraction; we therefore report $\kappa_t$ as a *diagnostic* and gate the certificate on an explicit support-coverage test (step 4b below), rather than treating $\kappa_t \le \rho$ alone as the certificate.

**How it works.**
```pseudocode
Input: unified multimodal model M, token CoT dataset D_tok, image prompts D_img,
       schedule s_t = min(1, t / T_shift), T_shift = 20k steps,
       target contraction rate rho = 0.9, tolerance eps = 1e-3,
       buffer capacity |B| = 50k transitions, refresh window W = 500 steps,
       coverage threshold delta = 0.05, coverage probe size N_probe = 256
Initialize theta (SFT on D_tok), replay buffer B, stage-0 latent anchors A_0
for t = 1 .. T_train:
    s_t <- min(1, t / T_shift)                      # curriculum stage
    # 1. Latent rollout at current stage
    z_1..z_H <- M.sample_latent(x, s_t)              # H = 8 latent steps
    # 2. Stage-consistent reward: information gain toward image target
    r_t <- InfoGain(z_H, target_image) - lambda * KL(pi_theta(.|s_t) || pi_theta(.|s_{t-1}))
    # 3. Estimate latent Bellman operator T_{s_t} on batch from B
    Q_{s_t} <- r_t + gamma * E_{z'~pi_theta(.|s_t)}[V(z')]
    # 4. Empirical contraction modulus (DIAGNOSTIC, not certificate):
    #    sup over batch of Lipschitz ratio, batch size 1024 drawn from B
    kappa_t <- max_{(z,z') in B} ||T_{s_t}V(z) - T_{s_t}V(z')|| / ||V(z) - V(z')||
    # 4b. Support-coverage test (gates the certificate):
    #     draw N_probe = 256 fresh latents from pi_theta(.|s_t);
    #     coverage_rate_t <- fraction whose nearest neighbor in B is within
    #     radius r_cov = 0.1 (L2 in latent space);
    #     certificate_valid_t <- (coverage_rate_t >= 1 - delta)
    # 5. Enforce kappa_t <= rho via spectral-normalized latent head
    #    (only when certificate_valid_t; otherwise skip and log)
    if certificate_valid_t and kappa_t > rho:
        theta <- theta - eta * grad_theta (kappa_t - rho)_+^2
    # 6. Policy update (PPO clip, eps_clip = 0.2) with stage-conditioned advantage
    theta <- theta + eta * grad_theta L_PPO(pi_theta, s_t)
    # 7. Anchor drift check: distance from stage-(t-1) fixed point
    if ||V_t - V_{t-1}|| < eps and s_t == 1: break
    # 8. Buffer refresh: every W = 500 steps, evict oldest transitions
    #    and refill with fresh stage-s_t rollouts to bound staleness
return theta, {kappa_t}, {coverage_rate_t}, {certificate_valid_t}
```

**Why this design.** We chose a *stage-indexed* MDP over a single stationary MDP because the curriculum is the object whose stability we must certify, accepting the cost that the value function must be re-estimated per stage. We chose an *empirical contraction modulus* $\kappa_t$ computed by Lipschitz ratios on the replay buffer over a learned regularizer because the modulus is the quantity that enters the convergence argument, accepting the cost of batch-wise max estimation noise; because this estimator is a supremum over a finite, stale buffer, we report it as a diagnostic and gate the certificate on the explicit support-coverage test in step 4b, so the theorem's hypothesis is checked rather than assumed. We chose *spectral normalization on the latent head* over clipping rewards because contraction is a property of the operator, not of the reward scale, accepting the cost that the policy class is slightly restricted. We chose *PPO with stage-conditioned advantage* over information-gain REINFORCE as in LARC because PPO's clipped surrogate gives a bounded per-step policy change that composes with the contraction bound, accepting the cost of a critic. A domain expert would not describe this as LARC plus PPO: LARC's information-gain RL has no stage-indexed operator and no contraction modulus, so it cannot certify the shift; CM-CRL's load-bearing object is the certified $\kappa_t \le \rho$ condition together with the coverage gate, which LARC does not compute.

**Why it measures what we claim.** The empirical contraction modulus $\kappa_t$ measures the *stability of the latent policy under curriculum shift* because we assume the replay-buffer Lipschitz ratio is a consistent estimator of the true operator modulus when the buffer covers the stage-$s_t$ latent support (checked by the coverage test in step 4b); this assumption fails when the latent support shifts faster than the buffer refresh rate, in which case $\kappa_t$ reflects buffer staleness rather than operator contraction. The stage-conditioned KL penalty measures *non-stationarity of the curriculum* because we assume the KL between consecutive stage policies is proportional to the schedule's step size $1/T_{\text{shift}}$; this assumption fails when the latent head's spectral norm changes abruptly, in which case the KL reflects head re-initialization rather than schedule drift. The anchor-drift check $\|V_t - V_{t-1}\| < \epsilon$ measures *fixed-point convergence* because we assume the value iterate is a Cauchy sequence when $\kappa_t \le \rho$ and the coverage gate passes; this assumption fails when the stage index is still moving ($s_t < 1$), in which case the check reflects transient stability rather than the terminal fixed point.

## Contribution

(1) A curriculum-shift RL objective for latent multimodal reasoning that is parameterized by a scalar stage index and a per-stage empirical contraction modulus, with a spectral-normalized latent head that enforces $\kappa_t \le \rho$. (2) A design principle: the latent policy's fixed point is certified by the ratio of the schedule's step size to the contraction modulus, so curriculum length $T_{\text{shift}}$ and target rate $\rho$ must satisfy $1/T_{\text{shift}} < 1-\rho$ for stability. (3) A stage-conditioned PPO training recipe with an anchor-drift stopping criterion for latent reasoning in unified multimodal models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Text-to-image benchmarks with compositional prompts (e.g., T2I-CompBench compositional subset, 1,200 prompts; single-object subset, 800 prompts) | Stresses latent support shift across curriculum stages |
| Primary metric | InfoGain toward target image at stage s=1 | Directly measures the quantity the curriculum optimizes |
| Secondary metric | Coverage rate (fraction of fresh stage-$s_t$ latents within $r_{\text{cov}}=0.1$ of a buffer neighbor) and $\kappa_t$ estimation variance across buffer sizes $\{|B| = 10k, 50k, 200k\}$ | Validates the buffer-coverage assumption and the diagnostic's stability |
| Baseline | LARC (information-gain RL, no stage index) | Tests whether curriculum indexing itself adds value |
| Baseline | Token-level CoT UMM (no latent reasoning) | Tests whether latent reasoning beats discrete CoT |
| Baseline | PPO on stationary latent MDP (no curriculum) | Tests whether the stage schedule drives stability |
| Ablation-of-ours | CM-CRL without contraction enforcement ($\kappa$ unconstrained) | Isolates the certified contraction as load-bearing |
| Ablation-of-ours | CM-CRL with coverage gate disabled (certificate always on) | Tests whether the support-coverage check is doing work |
| Scale sweep | 1B / 3B / 7B unified models, same schedule | Tests whether stability tracks the contraction condition rather than scale |

### Why this setup validates the claim

The central claim is that a certified contraction modulus $\kappa_t \le \rho$ over a stage-indexed curriculum is what makes latent reasoning stable under shift, so the design must isolate (a) the curriculum index, (b) the contraction certificate together with its coverage gate, and (c) latent vs. token reasoning. LARC tests (a): it has information-gain RL but no stage-indexed operator, so any gain of ours over it is attributable to the curriculum-plus-contraction machinery rather than to the reward. Stationary-PPO tests the same sub-claim from the other side — it has the latent MDP and PPO but no shifting kernel, so it should degrade exactly where the latent support moves. Token-CoT tests (c): if latent reasoning adds nothing over discrete CoT on compositional prompts, the whole premise collapses. The $\kappa$-unconstrained ablation is the sharpest test: it holds dataset, reward, PPO, and schedule fixed and removes only the spectral-normalized contraction enforcement, so if our gain survives its removal, the certificate is decorative rather than load-bearing. The coverage-gate-disabled ablation tests whether the support-coverage check in step 4b is doing work, since without it the certificate is asserted rather than checked. The scale sweep tests the motivation-level claim that stability follows from the geometry of the shift rather than from scale: if the contraction condition predicts which scales exhibit stable latent reasoning, the claim is supported. InfoGain at s=1 is the right primary metric because it is the quantity the stage-indexed reward is defined on and the quantity the contraction argument is supposed to protect — a stability certificate that does not show up as terminal information gain is not doing work.

### Expected outcome and causal chain

**vs. LARC** — On a compositional prompt where the target requires a latent intermediate that only becomes reachable after several rollout steps (e.g., a scene with a specific spatial relation), LARC's stationary information-gain RL has no stage index, so its policy gradient chases terminal reward from step one and its latent support drifts before the value estimate catches up, producing high-variance rollouts and mode collapse onto easy prompts. Our method instead ramps $s_t$ from 0 to 1, so early training stays near token-level CoT anchors while the latent head's spectral norm is progressively tightened, keeping the effective operator contractive at each stage. We expect a noticeable InfoGain gap concentrated on the compositional subset and near-parity on single-object prompts, where the support does not shift.

**vs. Token-level CoT UMM** — On a case where the reasoning chain is long (many intermediate spatial or attribute constraints), the token-CoT baseline must emit every intermediate as discrete tokens, so its error compounds per token and it loses the ability to revise a committed discrete decision, producing chains that are locally fluent but globally inconsistent with the image target. Our method instead carries the intermediate state as a latent $z$ that is updated by a contractive operator, so a bad early latent step is pulled back toward the fixed point rather than frozen into the token stream. We expect our method to lead by a clear margin on long-chain compositional prompts and to be roughly matched on short prompts, with the gap growing with chain length.

**vs. Stationary PPO on latent MDP** — On a case where the curriculum would need to hand off from token-level to latent reasoning partway through training, the stationary baseline has a single kernel and no $s_t$, so as the latent head's effective behavior shifts it experiences a non-stationary target it cannot track, and its critic lags, yielding oscillation in the value loss and a plateau in InfoGain. Our method instead re-estimates the stage-indexed operator per stage and enforces $\kappa_t \le \rho$ when the coverage gate passes, so each stage's value iterate is a contraction toward that stage's fixed point and the anchor-drift check fires only when $s_t = 1$. We expect the stationary baseline's value loss to oscillate visibly while ours decays monotonically per stage, and a terminal InfoGain gap that opens at the handoff and persists.

**vs. CM-CRL without contraction enforcement ($\kappa$ unconstrained)** — On the compositional subset where the latent support is predicted to shift, the unconstrained ablation has no spectral-normalized latent head and no $\kappa_t \le \rho$ penalty, so its operator modulus can exceed 1 and its value iterate can oscillate; we expect it to degrade relative to full CM-CRL exactly on this subset and to match on single-object prompts. If it matches on the compositional subset, the contraction certificate is not load-bearing.

**vs. CM-CRL with coverage gate disabled** — On stages where the buffer is stale, the gate-disabled variant enforces $\kappa_t \le \rho$ on a stale estimate, so the certificate can hold while the true modulus exceeds 1; we expect it to show higher $\kappa_t$ estimation variance and worse InfoGain than full CM-CRL on the compositional subset, with the gap concentrated at stages where coverage rate falls below $1-\delta$.

**Scale sweep (1B / 3B / 7B)** — We expect the contraction condition (coverage rate $\ge 1-\delta$ and $\kappa_t \le \rho$) to predict which scales exhibit stable latent reasoning, with unstable scales showing oscillating value loss and low terminal InfoGain regardless of parameter count; if stability instead tracks scale monotonically, the geometry-of-shift claim is not supported.

### What would falsify this idea

If the $\kappa$-unconstrained ablation matches CM-CRL on the compositional subset — i.e., removing the certified contraction enforcement produces no degradation exactly where the latent support is predicted to shift — then the contraction modulus is not the load-bearing object and the central claim is wrong. Likewise, if our gain over LARC is uniform across easy and compositional subsets rather than concentrated on the compositional subset, the curriculum-shift mechanism is not what is producing the improvement; and if the coverage-gate-disabled variant matches full CM-CRL even when coverage rate falls below $1-\delta$, the support-coverage assumption is not doing work.

## References

1. Latent Reasoning in Continuous Space for Unified Multimodal Models
