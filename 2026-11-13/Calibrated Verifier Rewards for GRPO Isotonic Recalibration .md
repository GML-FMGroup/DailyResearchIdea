# Calibrated Verifier Rewards for GRPO: Isotonic Recalibration of Per-Step Judge Scores with a Monotonicity Identifiability Constraint

## Motivation

Per-step judge models in long-horizon agent RL are used as drop-in reward oracles, but their scalar outputs are never checked for identifiability against actual action quality — the judge's helpfulness score is treated as ground truth even though the same paper (Actor, Judge, or Teacher?) shows judges are systematically biased by trajectory context and subgoal salience. This is a meta-level problem: every branch that trains a policy with judge-provided step rewards (JEV-style teachers, process reward models, LLM-as-judge reward shaping) inherits an uncalibrated oracle whose score-to-quality mapping is unconstrained, so GRPO optimizes a distorted reward surface. The root cause is structural: the judge is a decision model whose per-action helpfulness answer is a classification, not a calibrated regression onto verified action quality, and no held-out anchor ties the two.

## Key Insight

A monotone, held-out-verified calibration map from judge scalar scores to empirical action quality is identifiable when the judge is at least rank-consistent on a small human-validated calibration set, so post-hoc isotonic recalibration removes the affine and saturation distortions that make raw judge rewards unidentifiable without altering their ordering.

## Method

**What it is.** CVR-GRPO (Calibrated Verifier Rewards for GRPO) takes as input a frozen per-step judge J that scores each action a_t in a trajectory with scalar s_t, plus a small human-validated calibration set D_cal of (trajectory, action, verified quality q in [0,1]) tuples, and outputs a calibrated reward function r_cal(s) = f(s) that feeds GRPO. f is a monotone isotonic map fit on D_cal, and a held-out monotonicity constraint rejects any f whose Spearman correlation with verified quality on D_held falls below a threshold tau. **Load-bearing assumption (stated explicitly):** the judge's per-step scalar scores are rank-consistent with verified action quality on the calibration distribution, so that a single monotone map f fitted on D_cal transfers to the GRPO rollout distribution and remains valid as the policy drifts. This assumption is unverified and is the target of the identifiability check below.

**How it works.**

```pseudocode
Input: judge J, calibration set D_cal (n=500 verified steps), held-out D_held (n=200)
       GRPO policy pi_theta, threshold tau=0.7, bin width w=0.05

Phase 1 - Calibration fit:
  for each (traj, a_t, q) in D_cal:
      s_t = J(traj, a_t)                       # raw judge scalar
  Bin s_t into B equal-frequency bins (B = n/20 = 25)
  Fit isotonic regression f: R -> [0,1] minimizing sum (f(s_t) - q)^2
      subject to f monotone non-decreasing (PAVA algorithm)

Phase 2 - Identifiability check:
  for each (traj, a_t, q) in D_held:
      r_hat = f(J(traj, a_t))
  Compute Spearman rho(r_hat, q) on D_held
  if rho < tau:  # judge is not rank-consistent -> calibration unidentifiable
      reject and report; fall back to D_cal augmentation
  else accept f as r_cal

Phase 3 - GRPO training with calibrated rewards:
  for each episode:
      rollout trajectory tau under pi_theta
      for each step t: s_t = J(tau, a_t);  r_t = f(s_t)
      group-normalize r_t across the GRPO group (G=8 rollouts)
      GRPO update on pi_theta with advantage A_t from normalized r_t
  # f is frozen during training; no judge re-querying

Output: calibrated reward function f, trained policy pi_theta
```

**Why this design.** We chose isotonic regression over temperature scaling because isotonic makes no parametric assumption about the judge's distortion shape (saturation, floor effects, context-dependent compression) while temperature scaling forces a single global slope, accepting that isotonic needs more calibration points to avoid overfitting (n=500 vs n=50). We chose a held-out monotonicity constraint (Spearman rho >= tau=0.7) over a training-set fit check because in-sample monotonicity is guaranteed by PAVA construction and therefore vacuous as an identifiability test, accepting that some judges will be rejected and require more calibration data. We chose to freeze f during GRPO rather than jointly update it because a jointly-updated calibrator would co-adapt to the policy's reward-hacking direction and destroy the identifiability guarantee, accepting that f cannot track distribution shift in judge scores over training. A domain expert would not describe this as Process Reward Model training (Lightman 2023) or as VinePPO-style Monte-Carlo step rewards: PRMs learn a new reward model from scratch, whereas CVR-GRPO leaves the judge untouched and only fits a monotone post-hoc map whose validity is separately certified on held-out data — the contribution is the identifiability certificate, not the reward model.

**Why it measures what we claim.** The isotonic map f(s) measures calibrated action quality because the assumption that the judge is rank-consistent with verified quality on the calibration distribution makes the monotone fit equivalent to a quantile-matching map; this assumption fails when the judge's ranking inverts on a subpopulation (e.g., it over-credits fluent but unhelpful actions), in which case f reflects the judge's biased ranking rather than true quality and the Spearman gate on D_held should reject it. The held-out Spearman rho measures identifiability of the calibration because the assumption that D_held is exchangeable with the GRPO rollout distribution makes held-out rank correlation a valid proxy for in-training calibration validity; this assumption fails when the policy drifts far from the calibration distribution mid-training, in which case rho overstates calibration quality on the new distribution. The frozen-f design measures reward stability because the assumption that the judge's score distribution is stationary over the training horizon makes a single f valid throughout; this assumption fails under severe policy shift, in which case the calibrated reward silently reverts toward the raw judge's distortion.

## Contribution

(1) CVR-GRPO, a post-hoc calibration framework that fits a monotone isotonic map from frozen per-step judge scalars to human-verified action quality and admits it into GRPO only when a held-out Spearman identifiability gate passes. (2) The identifiability-gate design principle: calibration is only meaningful when the judge is rank-consistent on held-out verified steps, yielding a concrete reject/accept criterion that prior judge-as-teacher pipelines (JEV) lack. (3) A calibration-set construction protocol for per-step action quality that yields a reusable small human-validated benchmark of judge-score-vs-quality pairs on ALFWorld and WebShop trajectories.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ALFWorld + WebShop long-horizon tasks | Matches baseline and exposes step-level judge distortion |
| Primary metric | Success rate on held-out task split | Directly measures downstream policy quality, not judge agreement |
| Baseline | GRPO with raw JEV judge scores | Tests whether calibration alone adds value over raw judge |
| Baseline | GRPO with temperature-scaled judge scores | Tests whether monotone non-parametric fit beats global slope |
| Baseline | GRPO with no judge (sparse task reward) | Tests whether any dense judge signal helps at all |
| Ablation-of-ours | CVR-GRPO without held-out Spearman gate | Tests whether identifiability certificate drives the gain |

### Why this setup validates the claim

The central claim is that a monotone post-hoc calibration of a frozen per-step judge, certified by a held-out rank-correlation gate, yields a reward signal that improves GRPO policy quality specifically where the raw judge's distortion would mislead advantage estimation. The dataset and metric are chosen so that the effect is visible at the level that matters — task success — rather than at judge-agreement level, which could be trivially satisfied by any monotone map. The raw-judge baseline isolates the value of calibration; the temperature-scaling baseline isolates the value of non-parametric monotonicity over a parametric global slope; the no-judge baseline establishes that dense step rewards are useful at all, so a gain over it is not merely an artifact of reward density. The no-gate ablation is the decisive test of the identifiability-certificate sub-claim: if the gate is doing real work, removing it should degrade performance on the subpopulation where judge ranking inverts, while leaving easy subpopulations roughly unchanged. Success rate is the right metric because it is downstream of the entire reward pipeline and cannot be gamed by a calibrator that merely fits D_cal well.

### Expected outcome and causal chain

**vs. GRPO with raw JEV judge scores** — On a case where the judge over-credits a fluent but unhelpful ALFWorld action (e.g., a plausible-sounding "look" that wastes a step), the raw-judge baseline assigns it high reward, inflating its advantage and reinforcing the fluent-but-wasteful pattern because the judge's saturation region compresses true quality differences. Our method instead maps that saturated high score down through the isotonic fit, because D_cal contains verified examples where such actions scored low in quality, so the calibrated reward no longer rewards fluency alone. We expect a modest but consistent success-rate gain concentrated on tasks requiring many steps, with near-parity on short tasks where the judge is already well-behaved.

**vs. GRPO with temperature-scaled judge scores** — On a case where the judge exhibits a floor effect (many distinct bad actions all score near the minimum), temperature scaling applies one global slope and cannot separate them, so the baseline assigns near-identical advantages and the policy receives no gradient distinguishing mildly-bad from very-bad actions. Our method instead uses isotonic regression to fit a piecewise-constant map that resolves the floor into distinct calibrated levels, because PAVA makes no parametric assumption about distortion shape. We expect the gain over temperature scaling to be largest on WebShop, where judge score distributions are more skewed, and small on ALFWorld, where the distortion is closer to linear.

**vs. GRPO with no judge (sparse task reward)** — On a case where a long ALFWorld episode fails only at the final step, the sparse-reward baseline assigns zero credit to all preceding correct actions because only terminal success is rewarded, so the policy cannot learn which intermediate actions were good. Our method instead provides calibrated per-step rewards that preserve the judge's correct ranking of good versus bad intermediate actions while correcting its distortion, because the isotonic map is monotone and thus cannot invert a correctly-ranked pair. We expect a large gain on long-horizon tasks and near-parity on short tasks, with the gap widening as episode length grows.

### What would falsify this idea

If CVR-GRPO's gain over the raw-judge baseline is uniform across easy and hard task subsets — rather than concentrated on the long-horizon, high-distortion subset where the predicted failure mode lives — then the calibration is not doing what we claim and the improvement is attributable to some unrelated regularization effect. Likewise, if removing the held-out Spearman gate produces no degradation on the subpopulation where judge ranking inverts, the identifiability-certificate sub-claim is falsified.

## References

1. Actor, Judge, or Teacher? Understanding Decision Models in Long-Horizon Agents
