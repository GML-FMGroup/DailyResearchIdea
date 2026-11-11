# Closed-Loop Action-Spatial Consistency for Dexterous Policies

## Motivation

DexForesight trains an action-conditioned spatial world model, then freezes it and aligns the policy's latent queries to it one-way at training time, so the world model never receives the policy's actual actions as input during deployment. Structurally, the spatial foresight is a fixed prior that cannot be interrogated or corrected by the actions the policy is about to emit, so any mismatch between predicted hand-point motion and the action being generated is invisible to the policy. This is an instance of the broader open-loop coupling bottleneck in world-action models: prediction and action are aligned only offline, never reconciled online.

## Key Insight

A fixed-point condition requiring the spatial foresight induced by the policy's own emitted action to match the foresight the policy conditioned on turns the one-way alignment into a contraction-style consistency constraint that is checkable and correctable at inference without retraining the world model.

## Method

**What it is.** CLAS (Closed-Loop Action-Spatial consistency) wraps a frozen DexForesight-style spatial world model and a diffusion action policy in an inference-time refinement loop: it emits a candidate action, rolls the world model forward under that action, measures the discrepancy between the resulting foresight and the foresight the policy used as conditioning, and back-propagates that discrepancy into the denoising trajectory until the action-induced foresight and the conditioning foresight agree.

**Load-bearing assumption (stated explicitly).** The frozen world model's action-conditioned rollout is a faithful, differentiable predictor of hand-point motion for the candidate actions the policy emits during inference, so that the residual `||z_act - z_pred||^2` reflects genuine action–foresight mismatch rather than world-model extrapolation error. This assumption is checked at every refinement step by a support-validity score `v(o_t, a)` (see step 3b below); when `v` is low the residual is treated as extrapolation error and the update is gated accordingly.

**How it works.**

```pseudocode
inputs: obs o_t, language l, state s, frozen world model W, diffusion policy pi_theta,
        support classifier C (trained offline on (o_t, a) pairs, see calibration below)
hyperparams: K=4 refinement iters, lambda=0.5, tau=0.02 tolerance, eta=0.1 step,
             delta=0.05 trust-region radius on ||z_pred - z_pred_0||, v_min=0.5 support threshold

1. z_pred_0 = W.encode_foresight(o_t)          # conditioning foresight (frozen), anchored
   z_pred   = z_pred_0
2. a_0 = pi_theta.sample(o_t, l, s | z_pred)   # initial action from diffusion prior
3. for k in 1..K:
3a.    z_act = W.rollout(o_t, a_{k-1})         # foresight induced by candidate action
3b.    v = C(o_t, a_{k-1})                     # support-validity score in [0,1]
3c.    L_cons = || z_act - z_pred ||^2         # action-spatial consistency residual
3d.    if L_cons < tau or v < v_min: break
4.     g = dL_cons / d a_{k-1}                 # gradient through frozen W
5.     a_k = a_{k-1} - eta * v * g             # refine action, scaled by support validity
6.     z_pred_candidate = (1-lambda) * z_pred + lambda * z_act
7.     z_pred = z_pred_candidate if ||z_pred_candidate - z_pred_0|| <= delta else z_pred
       # trust-region bound: conditioning foresight may drift at most delta from its anchor
8. return a_K, L_cons, v
```

The loop alternates two corrections: the action is nudged so that rolling the world model forward under it reproduces the conditioning foresight (steps 3a–5), and the conditioning foresight is nudged toward the action-induced foresight (step 6), so the two quantities converge to a shared fixed point rather than one being held rigid. The tolerance `tau` and iteration cap `K` bound inference cost; `lambda` controls how much the policy is allowed to revise its own spatial prior versus revise its action. The support-validity score `v` gates both the gradient step (step 5) and early stopping (step 3d), and the trust-region radius `delta` bounds how far the conditioning foresight may drift from its anchor `z_pred_0` (step 7), so the residual is not minimized by co-drift of two unanchored latents.

**Support classifier C (concrete).** A 2-layer MLP (hidden=256, GeLU) trained offline on 10k (o_t, a) pairs labeled in-support if `a` lies within the 95th percentile of the offline action distribution conditioned on `o_t` (estimated via a k-NN distance in action space, k=20); output is a sigmoid validity score. Training takes ~1 GPU-hour on a single A100. At inference, `C` adds <1 ms per call.

**Why this design.** We chose to keep the world model frozen and differentiate through it rather than fine-tune it online, because DexForesight's offline training distribution is the only source of calibrated foresight and online fine-tuning would drift it, accepting the cost that gradients must flow through a frozen network and the correction is local rather than a full replan. We chose a bidirectional update (both `a` and `z_pred` move, step 6) over correcting only the action, because a rigid conditioning foresight can be unreachable by any action in the current diffusion support, in which case action-only correction stalls; the trade-off is that the conditioning signal is no longer a pure prior and may drift toward a locally consistent but globally suboptimal prediction. We chose a fixed small iteration budget `K=4` with early stopping at `tau` over an unrolled optimization to convergence, because dexterous control runs at control rate and convergence guarantees are unavailable for a non-convex residual; the cost is that some steps return before full consistency. A domain expert would not describe this as DexForesight plus a refinement wrapper, because the load-bearing object is the fixed-point residual `L_cons` coupling two quantities DexForesight treats as independent (the conditioning latent and the action-induced rollout), and the method's behavior is defined by the joint update that drives their difference to zero, not by either module alone.

**Why it measures what we claim.** The consistency residual `L_cons = ||z_act - z_pred||^2` measures action–spatial consistency because it is the squared distance between the foresight the policy conditioned on and the foresight the world model actually produces when rolled forward under the policy's emitted action, under the assumption that the frozen world model's rollout is a faithful action-conditioned predictor within its training distribution; this assumption fails when the candidate action leaves the world model's offline support, in which case `L_cons` reflects world-model extrapolation error rather than genuine action–foresight mismatch. The support-validity score `v` measures whether the candidate action is within the world model's calibrated support because it is a classifier trained on the offline action distribution, under the assumption that the 95th-percentile k-NN distance threshold captures the model's reliable region; this assumption fails for tasks with multimodal action distributions where the k-NN threshold is too coarse, in which case `v` may misclassify in-support actions as out-of-support. The bidirectional update weight `lambda` measures the trust placed in the action-induced foresight relative to the conditioning prior because it interpolates the two estimates, under the assumption that the action-induced rollout is more reliable once the action is locally consistent; this assumption fails under large distribution shift, where the conditioning prior should dominate and `lambda` should shrink, and otherwise the metric reflects prior drift. The trust-region radius `delta` measures how much the conditioning foresight is allowed to drift from its anchor `z_pred_0` because it bounds `||z_pred - z_pred_0||`, under the assumption that the anchor is a ground-truth-consistent foresight; this assumption fails when the initial foresight is itself erroneous, in which case the bound prevents correction of that error. The early-stopping tolerance `tau` measures convergence of the closed loop because it thresholds the residual that defines the fixed point, under the assumption that residuals below `tau` are indistinguishable from exact consistency at control resolution; this assumption fails for tasks with sub-centimeter precision requirements, where `tau` masks residual error that still degrades success.

## Contribution

(1) A closed-loop inference-time refinement algorithm, CLAS, that establishes a fixed-point consistency condition between a diffusion action prior and a frozen action-conditioned spatial world model via a bidirectional residual-minimizing update. (2) The design principle that action–spatial consistency is best enforced as a differentiable residual through a frozen world model with joint action-and-prior correction, rather than as a one-way latent alignment loss at training time. (3) An analysis of when the consistency residual is a valid proxy for action–foresight agreement versus world-model extrapolation error, characterized by the support condition on candidate actions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | DexJoCo 11-task suite | Standard dexterous benchmark with spatial structure |
| Primary metric | Task success rate | Directly reflects manipulation completion, not proxy loss |
| Baseline | DexForesight (frozen, no refinement) | Isolates value of closed-loop consistency correction |
| Baseline | Action-only refinement (λ=0) | Tests whether bidirectional update is load-bearing |
| Baseline | Diffusion policy without foresight | Tests whether foresight conditioning matters at all |
| Baseline | Test-time guidance (classifier-guided diffusion) | Tests whether CLAS is just guidance |
| Baseline | MPC with learned model (random shooting, 100 samples) | Tests whether CLAS is just model-predictive control |
| Ablation-of-ours | CLAS with K=1 iteration | Tests whether iterative fixed-point convergence helps |
| Ablation-of-ours | CLAS without support classifier (v=1 always) | Tests whether support gating prevents extrapolation error |
| Ablation-of-ours | CLAS without trust-region (delta=∞) | Tests whether anchoring prevents latent co-drift |
| Ablation-of-ours | CLAS with finite-difference gradient (no backprop) | Tests whether exact gradient is necessary |

### Why this setup validates the claim

The central claim is that coupling the conditioning foresight and the action-induced rollout through a shared fixed-point residual produces actions that are more spatially consistent with the policy's own prior, and that this consistency translates into higher task success. DexJoCo's 11 tasks give a spread of spatial-precision demands, so a success-rate metric can reveal where consistency matters. The frozen DexForesight baseline tests the sub-claim that unrefined foresight conditioning is insufficient: if it already succeeds everywhere, the residual is doing nothing. The action-only (λ=0) baseline tests the sub-claim that the bidirectional update is load-bearing rather than decorative — if it matches CLAS, the step-6 re-conditioning is unnecessary. The no-foresight diffusion baseline tests whether foresight conditioning itself contributes anything, guarding against the possibility that our gains come purely from the diffusion prior. The test-time guidance baseline tests whether CLAS is just a form of classifier guidance applied to the diffusion policy; if guidance matches CLAS, the fixed-point residual is not the mechanism. The MPC-with-learned-model baseline tests whether CLAS is just model-predictive control with a learned model; if MPC matches CLAS, the specific closed-loop consistency update is not necessary. The K=1 ablation tests whether the iterative fixed point matters versus a single correction. The support-classifier ablation tests whether gating the gradient by validity prevents the world model from being exploited out of support. The trust-region ablation tests whether anchoring the conditioning foresight prevents co-drift of two latents. The finite-difference ablation tests whether exact backprop through the frozen model is necessary or whether a Jacobian-free approximation suffices. Success rate is the right metric because our mechanism is supposed to reduce action–foresight mismatch that manifests as task failure, not as a lower training loss; a proxy residual metric could improve while success stalls, which would itself be diagnostic.

### Expected outcome and causal chain

**vs. DexForesight (frozen, no refinement)** — On a task requiring the gripper to approach a contact-rich object from a specific spatial configuration, the frozen baseline emits an action from its diffusion prior whose induced rollout diverges from the conditioning foresight, because the policy never checks whether its action actually reproduces the spatial latent it conditioned on. Our method rolls the world model forward under the candidate action, measures that divergence, and nudges the action until the induced foresight matches, so we expect a clear success-rate gap concentrated on the contact-rich, spatially constrained tasks, with near-parity on tasks where the prior already lands in a consistent region.

**vs. Action-only refinement (λ=0)** — On a task where the conditioning foresight is itself slightly off the manifold reachable by any action in the diffusion support, the action-only baseline stalls: it keeps pushing the action toward an unreachable target and either exhausts K iterations with a large residual or returns a distorted action, because the conditioning prior is held rigid. Our method lets the conditioning foresight drift toward the action-induced rollout via the λ interpolation, so the two converge to a shared fixed point, and we expect the action-only variant to fail disproportionately on the subset where the initial foresight is unreachable, while matching us where the prior is already reachable.

**vs. Diffusion policy without foresight** — On a task requiring anticipatory spatial reasoning (e.g., pre-shaping the hand before contact), the no-foresight baseline emits actions that are locally plausible but globally misaligned, because it has no spatial latent to condition on and no consistency signal to correct against. Our method inherits the foresight conditioning and then enforces that the action reproduces it, so we expect the largest gap on anticipatory tasks and a smaller gap on reactive tasks where immediate observation dominates.

**vs. Test-time guidance (classifier-guided diffusion)** — On a task where the guidance classifier is trained on the same offline data, the guidance baseline may improve over frozen DexForesight but cannot correct for the world model's extrapolation error because it does not roll the model forward under the emitted action. Our method measures the action-induced foresight and gates the update by support validity, so we expect CLAS to outperform guidance on tasks where the action leaves the offline support, and to match guidance where the action remains in support.

**vs. MPC with learned model (random shooting, 100 samples)** — On a task with a high-dimensional action space, random shooting MPC may fail to find a consistent action within its sample budget, while our gradient-based refinement uses the world model's Jacobian to move directly toward consistency. We expect CLAS to outperform MPC on high-dimensional tasks, but to be comparable on low-dimensional tasks where random shooting is sufficient.

**vs. CLAS with K=1 iteration** — On a task requiring multiple corrections to reach the fixed point, the K=1 variant performs a single gradient step and may not converge, leaving a large residual. Our method with K=4 iterates until the residual falls below τ or the budget is exhausted, so we expect a success-rate gap on tasks where the initial action is far from consistency, and near-parity on tasks where one step suffices.

**vs. CLAS without support classifier (v=1 always)** — On a task where the refinement loop pushes the action out of the world model's offline support, the no-classifier variant will exploit the model's extrapolation error, producing physically implausible actions that minimize the residual but fail the task. Our method gates the gradient by v and stops when v < v_min, so we expect the no-classifier variant to fail on tasks that require large action corrections, while matching us on tasks where the action stays in support.

**vs. CLAS without trust-region (delta=∞)** — On a task where the conditioning foresight is erroneous, the no-trust-region variant allows z_pred to drift arbitrarily far from its anchor, potentially converging to a locally consistent but globally wrong fixed point. Our method bounds the drift by delta, so we expect the no-trust-region variant to fail on tasks where the initial foresight is noisy, while matching us on tasks where the initial foresight is accurate.

**vs. CLAS with finite-difference gradient** — On a task with a high-dimensional action space, finite-difference gradient estimation requires many forward passes and may be too slow for the control rate, while exact backprop is efficient. We expect the finite-difference variant to have lower success rate due to latency-induced control delays, but to match exact backprop on low-dimensional tasks where the approximation is accurate.

### What would falsify this idea

If CLAS's success-rate gain over frozen DexForesight is uniform across all 11 tasks rather than concentrated on the spatially constrained and contact-rich subsets where the residual is predicted to matter, the central claim that action–spatial consistency drives the improvement is wrong. Likewise, if the action-only (λ=0) baseline matches CLAS everywhere, the bidirectional fixed-point update is not load-bearing and the mechanism reduces to a standard action correction. Additionally, if the support-classifier ablation matches CLAS everywhere, the load-bearing assumption that the world model's rollout is faithful within its support is not being tested, and the residual may reflect extrapolation error rather than genuine mismatch.

## References

1. DexForesight: Grounding Dexterous Policies with Implicit Spatial Foresight
