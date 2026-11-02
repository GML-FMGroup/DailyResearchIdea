# Belief-Monotone Latent Reasoning: Supermartingale Certificates for Continuous Multimodal Chains

## Motivation

Latent reasoning in unified multimodal models (LARC) replaces discrete chain-of-thought tokens with continuous hidden-state trajectories, which recovers test-time scaling but discards the only property that made discrete step verification work: an explicit, checkable unit at which each reasoning step can be certified as progress. LARC's information-gain RL rewards latent steps for informativeness but imposes no ordering constraint across the trajectory, so a latent step can increase posterior entropy over the multimodal context and still receive reward — the optimizer has no structural reason to prefer a refinement sequence over a wandering one. The field's trajectory toward continuous reasoning therefore inherits a meta-level gap: monotone consistency of latent refinement under belief update, which no existing continuous-reasoning method enforces and which discrete CoT enforced only by construction of its token boundaries.

## Key Insight

If the latent step is treated as an observation that updates a posterior over the multimodal context, then a supermartingale condition on the expected information gain makes belief refinement monotone by construction, supplying a per-step certificate that is checkable in continuous space exactly as token-boundary verification was checkable in discrete space.

## Method

```pseudocode
# BM-LR: Belief-Monotone Latent Reasoning
# Inputs: unified multimodal model M with latent policy pi_theta, context c = (x_text, x_img)
# Outputs: trained pi_theta, per-step belief states b_t, monotonicity certificate C

# [ASSUMPTION A — stated explicitly]
# The entropy H(b_t) of the learned posterior head q_phi is treated as a faithful,
# policy-independent measure of genuine belief refinement: a decrease in H(b_t) across
# latent steps certifies real progress rather than an artifact of q_phi's misspecification
# or drift. This assumption is load-bearing for the supermartingale certificate C.
# It is verified (not merely asserted) by the calibration diagnostic in step 8.

1. Curriculum SFT (unchanged from LARC): convert token-level CoT -> latent steps z_1..z_T
2. Initialize belief head q_phi(b_t | z_{<=t}, c)  # posterior over context tokens/regions
   # Architecture: 2-layer MLP, hidden=256, GeLU, dropout=0.1, output dim = |context tokens| + |image regions|
3. For each rollout:
   a. Sample latent trajectory z_1..z_T ~ pi_theta(.|c)
   b. For t = 1..T: compute b_t = q_phi(.|z_{<=t}, c); record entropy H(b_t)
   c. Compute step info-gain  g_t = H(b_{t-1}) - H(b_t)   # b_0 = prior q_phi(.|c)
   d. Compute supermartingale residual  r_t = g_t - lambda * (H(b_{t-1}) - H_target)   # lambda = 0.5
   e. Certificate  C = 1[ min_t g_t >= -epsilon ]   # epsilon = 1e-3 tolerance
4. Reward: R = R_task + beta * sum_t g_t - gamma * sum_t max(0, -g_t)   # beta=0.1, gamma=1.0
   # gamma penalizes entropy-increasing steps; beta rewards cumulative refinement
5. RL update (policy gradient on pi_theta) with reward R; freeze q_phi during policy step
6. Every K=500 steps, refit q_phi by MLE on (z_{<=t}, c) -> context reconstruction
7. Emit certificate C per rollout; discard rollouts with C=0 from the policy-gradient batch
8. Calibration diagnostic (run every K=500 steps alongside q_phi refit, on a held-out
   calibration set of 512 examples):
   a. Compute ECE of q_phi over context-token predictions (15 bins).
   b. Compute mutual information I(z_t; c) between latent step z_t and context c
      (via a k-NN estimator on 512 samples) to detect entropy collapse: if I(z_t; c)
      falls below 0.1 * H(b_0) while H(b_t) also falls, the policy is emitting
      uninformative latents that q_phi trivially fits (failure mode F).
   c. Report g_t vs. downstream correctness correlation on the calibration set.
   # If ECE > 0.15 or I(z_t; c) < 0.1 * H(b_0), the certificate C is reported as a
   # diagnostic only for that training window, not as a load-bearing invariant.
```

**Why this design.** We chose a learned posterior head q_phi over the multimodal context rather than a scalar confidence head because the supermartingale condition requires a *distribution* whose entropy is well-defined; a scalar score cannot express information gain, accepting the cost that q_phi must be refit periodically and can drift. We chose a hard certificate gate (discard C=0 rollouts) over a soft penalty alone because the monotonicity invariant is the load-bearing claim, accepting the cost of reduced sample efficiency when the policy is early in training. We chose a per-step entropy-difference g_t over a cumulative KL-to-prior because entropy differences localize which step violates monotonicity, accepting the cost that entropy is only a proxy for refinement quality when the posterior is misspecified. We chose to freeze q_phi during each policy step over joint alternation because joint updates let the policy game the belief head (collapse entropy by making z uninformative), accepting slower convergence. A domain expert would not describe this as LARC-plus-a-penalty: LARC's information-gain RL is a per-step reward shaping with no cross-step ordering constraint, whereas here the supermartingale residual r_t couples consecutive belief states and the certificate C is a trajectory-level invariant that LARC cannot produce.

**Why it measures what we claim.** H(b_t) measures belief refinement because assumption A holds: q_phi is calibrated (ECE ≤ 0.15 on the 512-example calibration set) and its entropy is a proper scoring proxy for context reconstruction quality; this assumption fails when the policy collapses q_phi entropy by emitting uninformative latents (failure mode F), in which case H(b_t) reflects q_phi's ease of fitting rather than genuine progress. g_t measures per-step information gain because the entropy difference localizes which step violates monotonicity; this assumption fails when the posterior is misspecified, in which case g_t reflects q_phi's drift rather than refinement. The certificate C measures trajectory-level monotonicity because the supermartingale residual r_t couples consecutive belief states; this assumption fails when q_phi is frozen and the policy games it, in which case C reflects a self-referential invariant of the auxiliary head rather than of the reasoning trajectory — which is why step 8 reports C as a diagnostic when ECE > 0.15 or I(z_t; c) < 0.1 * H(b_0).

## Contribution

(1) A belief-state supermartingale certificate for continuous latent reasoning that converts per-step posterior entropy differences into a trajectory-level monotonicity invariant, turning discrete step verification into a checkable continuous-space condition. (2) A certificate-gated policy-gradient training procedure in which rollouts violating the monotonicity invariant are excluded from updates, yielding a design principle that latent reasoning trajectories can be filtered by an information-theoretic ordering rather than by task reward alone. (3) A per-step belief head refit schedule that keeps the posterior calibrated to the evolving latent policy, with the entropy-difference signal exposed as a diagnostic for where latent refinement breaks down.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multimodal CoT benchmarks requiring image-grounded multi-step inference (e.g., ScienceQA, A-OKVQA, MMMU) | Forces belief refinement across steps, not single-hop recall |
| Primary metric | Task accuracy stratified by reasoning-chain depth (T≤2, 3≤T≤5, T≥6) | Detects whether gains concentrate where monotonicity matters |
| Baseline | LARC (information-gain RL, no ordering constraint) | Isolates value of cross-step supermartingale coupling |
| Baseline | Reasoning-augmented UMM (token-level CoT) | Tests latent-vs-token reasoning under same model |
| Baseline | Base UMM without latent RL | Floor: measures RL contribution alone |
| Ablation-of-ours | BM-LR without certificate gate (soft penalty only) | Tests whether hard invariant, not reward shaping, drives gains |
| Diagnostic | Calibration set of 512 examples: ECE of q_phi, mutual information I(z_t; c), g_t–correctness correlation | Verifies assumption A and detects failure mode F |

### Why this setup validates the claim

The central claim is that enforcing a belief-monotone supermartingale over latent steps yields reasoning trajectories whose information gain is non-increasing in entropy, and that this ordering—not merely per-step reward shaping—is what improves multimodal reasoning. The LARC baseline is the critical test: it shares the information-gain reward but lacks any cross-step ordering constraint, so any gap between LARC and BM-LR isolates the value of the supermartingale coupling and certificate gate. The token-level CoT and base UMM baselines establish that gains are not merely from latent reasoning or RL in general. The ablation—soft penalty without the hard certificate—distinguishes whether the load-bearing element is the invariant itself or the reward term. Stratifying accuracy by reasoning-chain depth is the right metric because the predicted effect is depth-dependent: shallow chains give little room for monotonicity to matter, so a uniform gain across depths would contradict the mechanism, while a gain concentrated on deeper chains confirms it. The calibration diagnostic (ECE ≤ 0.15, I(z_t; c) ≥ 0.1·H(b_0)) is necessary to verify assumption A: if q_phi is miscalibrated or the policy collapses its entropy, the certificate C is reported as a diagnostic only, and any accuracy gap cannot be attributed to genuine belief refinement.

### Expected outcome and causal chain

**vs. LARC (information-gain RL, no ordering constraint)** — On a deep multimodal chain where an early latent step commits to a wrong image region, LARC's per-step reward still credits later steps that raise information gain locally, so the trajectory drifts into entropy-increasing corrections that never consolidate. Our method instead penalizes each entropy-increasing step via the supermartingale residual and discards certificate-failing rollouts, forcing the policy toward trajectories where belief entropy falls monotonically. We expect a noticeable accuracy gap on deep-chain subsets (T≥6) and near-parity on shallow ones (T≤2). The calibration diagnostic must confirm ECE ≤ 0.15 and I(z_t; c) ≥ 0.1·H(b_0) on the 512-example set; otherwise the gap is reported as inconclusive.

**vs. Reasoning-augmented UMM (token-level CoT)** — On a case requiring fine-grained visual grounding across several steps, token-level CoT emits discrete rationales that can silently contradict earlier ones because no distributional belief state is tracked, so errors compound undetected. Our method maintains an explicit posterior q_phi whose entropy is measured at each step, catching contradictions as entropy increases and gating them out. We expect our advantage to grow with chain depth and be modest on single-hop grounding items. The mutual-information check I(z_t; c) ≥ 0.1·H(b_0) guards against the failure mode where q_phi entropy falls because latents are uninformative, not because grounding improved.

**vs. BM-LR without certificate gate (soft penalty only)** — On an instance where the policy learns to game the reward by producing high-variance latent steps that occasionally spike information gain, the soft-penalty variant tolerates these because the cumulative reward stays positive, letting entropy-increasing steps survive into the batch. Our hard certificate discards such rollouts entirely. We expect the ablation to match us on easy items but show higher variance and lower worst-case accuracy on adversarial deep chains, confirming the invariant—not the reward—is load-bearing. The ECE and mutual-information diagnostics are reported for both variants to rule out entropy collapse as the source of any difference.

### What would falsify this idea

If BM-LR's accuracy gain over LARC were uniform across reasoning-chain depths rather than concentrated on deep chains, the monotonicity mechanism would not be doing the causal work claimed. Likewise, if the soft-penalty ablation matched BM-LR everywhere, the hard certificate invariant would be unnecessary and the central claim wrong. Additionally, if the calibration diagnostic showed ECE > 0.15 or I(z_t; c) < 0.1·H(b_0) throughout training, assumption A would be violated and the certificate C would be a self-referential artifact, falsifying the claim that it certifies genuine belief refinement.

## References

1. Latent Reasoning in Continuous Space for Unified Multimodal Models
