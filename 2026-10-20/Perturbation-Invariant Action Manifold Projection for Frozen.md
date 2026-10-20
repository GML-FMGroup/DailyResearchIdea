# Perturbation-Invariant Action Manifold Projection for Frozen VLM Navigation Policies

## Motivation

Frozen vision-language models deployed zero-shot in continuous navigation (e.g., LightZeroNav, Fast-SmartWay, NaVid) are assumed to remain reliable, yet no mechanism checks whether their instruction-conditioned action ranking is stable under physically-realizable nuisance changes such as viewpoint jitter or partial occlusion. This is a structural gap across the whole family of zero-shot VLN agents: they treat a single forward pass's action logits as trustworthy, so when the decision surface silently degrades under distribution shift (exactly the visually ambiguous/occluded scenes LightZeroNav's authors flag as a limitation), the agent commits to a degraded action with no internal signal that anything went wrong. The meta-level problem is that no test-time calibration closure exists for a frozen decision policy that requires neither parameter updates, replay, nor an auxiliary decision model.

## Key Insight

A correct navigation decision is invariant to nuisance observation perturbations, so the frozen policy's own action-distribution geometry under paired physically-realizable perturbations forms a self-referential residual whose zero set defines a locally-consistent action manifold onto which any degraded action distribution can be projected in closed form — no labels, no learning, no auxiliary model required.

## Method

**What it is.** PIAMP (Perturbation-Invariant Action Manifold Projection) takes as input the frozen VLM policy's action logits at the current step, plus logits under a small set of paired, physically-realizable observation perturbations, and outputs a corrected action distribution obtained by a closed-form projection onto the locally-consistent manifold. It never updates parameters, keeps no replay buffer, and trains no auxiliary decision model.

**Load-bearing assumption (stated explicitly).** PIAMP assumes that the paired-perturbation residual r[a] is a monotone, action-specific measure of decision-policy degradation: low residual on action a means the policy's ranking of a is trustworthy, high residual means it is not. This assumption is verified at test time by a calibration gate (step 4b) before the projection is trusted; when the gate fails, PIAMP falls back to the raw policy (lambda=0).

**How it works.**

```pseudocode
Input: frozen policy pi_theta, instruction I, observation o_t, action set A (K discrete actions)
Hyperparams: n_pert = 4 (paired perturbations), eps_jitter = 2 deg viewpoint, m_occ = 3 occlusion masks, tau = 0.5 temperature, lambda = 0.7 manifold weight, q = 0.5 (quantile threshold for invariant subset), n_cal = 64 (calibration steps for gate)

1. Base distribution:
   z0 = pi_theta(I, o_t)                 # logits over A
   p0 = softmax(z0 / tau)

2. Paired perturbations (each pair = symmetric nuisance):
   for j in 1..n_pert:
       o_plus  = perturb(o_t, +eps_jitter, mask=m_occ[j])   # e.g. yaw +2deg + mask
       o_minus = perturb(o_t, -eps_jitter, mask=m_occ[j])   # yaw -2deg + same mask
       z_plus[j]  = pi_theta(I, o_plus)
       z_minus[j] = pi_theta(I, o_minus)

3. Self-consistency residual per action a:
   r[a] = mean_j | softmax(z_plus[j]/tau)[a] - softmax(z_minus[j]/tau)[a] |
   # r[a] in [0,1]; high r[a] => action a's ranking is NOT invariant to nuisance

4. Invariant subset via quantile threshold:
   S = { a : r[a] <= quantile(r, q) }   # q=0.5 recovers median; q swept in ablation

4b. Calibration gate (verifies residual-to-correctness monotonicity):
   # On a held-out calibration set of n_cal steps from the training split (no labels used at test time),
   # compute for each step the argmax under each paired perturbation: a_plus[j] = argmax z_plus[j], a_minus[j] = argmax z_minus[j].
   # Define flip indicator f[a] = 1 if a appears as argmax in any perturbed pass but not in z0, else 0.
   # Compute Spearman rank correlation rho between r[a] and f[a] across the calibration set.
   # Gate: if rho < 0.3 (residual does not predict ranking flips), set lambda = 0 for this episode.
   # This gate is computed once per episode using the first n_cal steps; it does not require oracle labels.

5. Locally-consistent manifold M (closed form):
   # M = { p in simplex : p agrees with p0 on the invariant subspace }
   # Projection of p0 onto M:
   p_proj = argmin_{p in simplex, p[S]=p0[S]} KL(p || p0)
          = renormalize( p0 restricted to S, mass = p0[S] )
   # i.e., redistribute probability mass AWAY from non-invariant actions

6. Corrected action distribution:
   p_corr = (1 - lambda) * p0 + lambda * p_proj
   a_t = argmax_a p_corr[a]

Output: a_t (and p_corr for logging)
```

**Why this design.** We chose paired symmetric perturbations (yaw ±eps, same occlusion mask) over random independent perturbations because the paired structure cancels first-order nuisance sensitivity and isolates the residual to genuine ranking instability, accepting the cost that we cannot detect asymmetric degradations (e.g., only-left-occlusion failures). We chose a closed-form simplex projection over a learned calibration head because the manifold is defined by a linear invariance constraint on the invariant action subset, so the KL projection has an analytic solution (renormalization), accepting the cost that the invariant-subset threshold (quantile q) is a heuristic rather than an optimized boundary; the calibration gate (step 4b) mitigates this by disabling projection when the residual fails to predict ranking flips. We chose to blend p_corr = (1-lambda)p0 + lambda p_proj rather than replace p0 outright because full projection can over-correct when the base policy is already consistent (all residuals low => S = all actions => p_proj = p0, so blending is a no-op), accepting the cost that lambda introduces a knob whose value trades off correction strength against over-smoothing. A domain expert would NOT describe this as a variant of Fast-SmartWay's Uncertainty-Aware Reasoning or MapGPT's adaptive planning: those use the model's own confidence/reasoning to re-prompt or re-plan, whereas PIAMP never re-queries the model for a decision and never uses a confidence score — it operates purely on the geometry of the action-logit surface under nuisance perturbations, a signal that confidence-based methods (which look at a single forward pass) structurally cannot access.

**Why it measures what we claim.** The residual r[a] measures *decision-policy degradation* because the named assumption is that a correct instruction-conditioned action ranking is invariant to physically-realizable nuisance perturbations (viewpoint jitter, partial occlusion) that do not change the task-relevant scene content; this assumption fails when a perturbation genuinely changes task-relevant information (e.g., an occlusion mask hides the target landmark), in which case r[a] reflects legitimate scene ambiguity rather than model degradation. The invariant subset S measures *which actions the frozen policy can be trusted on* because the assumption is that actions with below-quantile residual lie on the locally-consistent manifold where the policy's ranking is stable; this fails when all actions have uniformly high residual (fully degraded step), in which case S is arbitrary and p_proj collapses toward a near-uniform distribution rather than a meaningful correction. The projection p_proj measures *the closest consistent action distribution* because the assumption is that the consistent manifold is the linear constraint set {p : p[S]=p0[S]}, whose KL projection is exact renormalization; this fails when the true consistent manifold is non-linear (e.g., actions are coupled through a shared sub-goal), in which case p_proj is a first-order approximation. The blend weight lambda measures *how much to trust the invariance signal over the raw policy* because the assumption is that residual magnitude is monotone in degradation severity; this fails when a small residual coincides with a large but symmetric bias, in which case lambda under-corrects. The calibration gate (step 4b) directly tests the monotonicity assumption by checking whether r[a] predicts argmax flips on held-out steps; when it does not (rho < 0.3), lambda is set to 0, so the method falls back to the raw policy rather than trusting an unverified signal.

## Contribution

(1) A closed-form test-time calibration operator (PIAMP) that projects a frozen VLM navigation policy's action distribution onto a perturbation-invariance manifold using only the policy's own logits under paired physically-realizable perturbations, requiring no parameter updates, replay buffer, or auxiliary decision model. (2) The design principle that paired symmetric nuisance perturbations yield a self-referential consistency residual whose zero set is a linear constraint on the action simplex, admitting an exact KL projection — establishing that invariance-based calibration is structurally distinct from confidence-based gating. (3) An analysis of when the invariance assumption holds (task-relevant content preserved under perturbation) and its failure mode (occlusion hiding task-relevant landmarks), yielding a concrete diagnostic for when to trust the correction.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VLN-CE (R2R-CE) | Continuous, zero-shot, matches nuisance perturbations |
| Primary metric | Success Rate (SR) | End-task correctness, not proxy confidence |
| Baseline | GPT-4o zero-shot VLN-CE agent | Strong single-forward-pass confidence-style reference |
| Baseline | Fast-SmartWay | Panoramic-free zero-shot pipeline, latency-sensitive |
| Baseline | MossVLN | Memory-based continuous VLN, no invariance check |
| Baseline | Naive majority-vote (9 forward passes) | Latency-matched control isolating projection from extra compute |
| Ablation-of-ours | PIAMP w/o projection (λ=0) | Isolates gain from manifold projection itself |
| Ablation-of-ours | PIAMP with independent random perturbations (not paired) | Tests whether paired symmetry is necessary |
| Ablation-of-ours | PIAMP with single-pass entropy/confidence score | Tests whether confidence alone reproduces the signal |
| Ablation-of-ours | PIAMP with quantile q ∈ {0.3, 0.5, 0.7} | Tests sensitivity to invariant-subset threshold |
| Diagnostic | Residual-correctness AUC (oracle next-action labels on held-out 200 steps) | Directly tests r[a] as degradation detector |
| Diagnostic | Spearman rho between r[a] and argmax-flip indicator on calibration set | Tests calibration gate validity |
| Cost metric | Forward passes per step (1 + 2*n_pert = 9) and wall-clock latency | Quantifies deployment overhead |
| Safety metric | Fraction of catastrophic failures (collisions, off-path >1m) prevented per episode | Measures safety signal value |

### Why this setup validates the claim

The central claim is that nuisance-invariance residuals computed from paired perturbations expose decision-policy degradation that single-forward-pass confidence methods cannot see, and that projecting onto the locally-consistent action subset recovers correct actions. VLN-CE is the right testbed because it is zero-shot and continuous, so viewpoint jitter and partial occlusion are physically realizable and naturally occurring, making the paired-perturbation signal meaningful rather than synthetic. Success Rate is the correct metric because PIAMP claims to fix the *action* chosen, not to produce better-calibrated probabilities; a calibration metric would not distinguish our mechanism from a confidence head. GPT-4o tests whether a strong single-pass policy already implicitly handles nuisance (if it does, our gain should vanish); Fast-SmartWay tests whether a latency-optimized pipeline is more brittle to nuisance; MossVLN tests whether memory alone compensates for per-step instability. The naive majority-vote baseline (9 forward passes, same compute as PIAMP) isolates the projection mechanism from the extra forward passes, so any gain over it is attributable to the manifold geometry rather than to compute. The λ=0 ablation is the falsification anchor: if removing the projection does not change outcomes, the manifold geometry carries no signal and the central claim is dead regardless of absolute scores. The residual-correctness AUC diagnostic directly tests the load-bearing assumption that r[a] is monotone in degradation severity: if AUC is near 0.5, the residual does not track correctness and the method's premise is falsified. The calibration gate diagnostic (Spearman rho) tests whether the gate correctly disables projection when the residual is uninformative.

### Expected outcome and causal chain

**vs. GPT-4o zero-shot VLN-CE agent** — On a case where the instruction says "turn left at the kitchen" and a 2° yaw jitter or a partial occlusion shifts the model's top action from left to forward, the baseline commits to the wrong action because it reads only one forward pass and has no signal that its ranking is nuisance-sensitive. Our method detects that the left action's paired residual is low while forward's is high, and renormalizes mass onto the invariant subset, so we expect a modest but consistent SR gain concentrated on jitter/occlusion-heavy episodes with near-parity on clean open-corridor segments. The calibration gate (step 4b) ensures that when the residual does not predict argmax flips on held-out steps (rho < 0.3), lambda is set to 0, so gains are not expected on episodes where the gate fires.

**vs. Fast-SmartWay** — On a case where a viewpoint perturbation near a doorway flips the predicted waypoint, Fast-SmartWay's latency-optimized single-pass pipeline propagates the flip without any consistency check, because its mechanism is a fast forward pass with no paired-observation comparison. Our method runs the same frozen policy under ±jitter and renormalizes away the unstable action, so we expect a larger SR gap than against GPT-4o specifically on doorway/corner subsets, where nuisance sensitivity is highest, and parity on long straight traversals.

**vs. MossVLN** — On a case where memory is intact but the current observation is partially occluded, MossVLN retrieves relevant history but still acts on a degraded per-step logit surface, because memory does not repair the current-step ranking instability. Our method leaves memory untouched and corrects only the current action distribution via the invariance residual, so we expect gains on occlusion-triggered failures while matching MossVLN on memory-dependent re-localization tasks where per-step logits are already stable.

**vs. Naive majority-vote (9 forward passes)** — On a case where the base policy's top action is correct but a perturbation flips it, majority-vote over 9 passes can be swayed by correlated perturbations (e.g., all occlusion masks share a bias) and does not use the paired symmetry to cancel first-order nuisance. Our method's paired residual isolates ranking instability and renormalizes mass onto the invariant subset, so we expect PIAMP to outperform majority-vote specifically on episodes where the residual-correctness AUC is high (i.e., where the residual is informative), and to match it when the gate sets lambda=0.

### What would falsify this idea

If the SR gain over the λ=0 ablation is uniform across clean and nuisance-heavy subsets rather than concentrated where paired residuals are high, the invariance signal is not doing the causal work we claim. Equally fatal: if the λ=0 ablation matches full PIAMP, the manifold projection is inert and the central mechanism is falsified. Additionally, if the residual-correctness AUC on held-out oracle labels is near 0.5 (no better than chance), the load-bearing assumption that r[a] is monotone in degradation severity is false, and the method's premise is falsified regardless of SR.

## References

1. LightZeroNav: Zero-Shot Vision Language Navigation in Continuous Environments Based on Lightweight VLMs
2. MossVLN: Memory-Observation Synergistic System for Continuous Vision-Language Navigation
3. Fast-SmartWay: Panoramic-Free End-to-End Zero-Shot Vision-and-Language Navigation
4. MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision-and-Language Navigation
5. Open-Nav: Exploring Zero-Shot Vision-and-Language Navigation in Continuous Environment with Open-Source LLMs
6. NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation
7. LangNav: Language as a Perceptual Representation for Navigation
8. NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models
