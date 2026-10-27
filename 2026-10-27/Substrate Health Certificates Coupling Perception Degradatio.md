# Substrate Health Certificates: Coupling Perception Degradation to Decision Validity for VLM Agents

## Motivation

Across both HexaShell and LightZeroNav, the perception-and-evaluation substrate is treated as an oracle: HexaShell's success detector reads the robot's own sensor logs under exactly the degraded conditions it is trying to certify, and its geometric triangulation silently fails under occlusion or poor views, while LightZeroNav's frozen VLM has no calibration mechanism to detect when its action logits are unreliable. The structural gap is that neither tree couples substrate degradation (intrinsics drift, occlusion, viewpoint, distribution shift) to the validity of the agent's downstream decision, so a confident-looking action can be produced from a substrate that has silently stopped being identifiable. What is missing is a certificate that propagates substrate quality into a monotone decision-validity predicate, gating actions by shrinking the admissible set rather than thresholding a confidence score.

## Key Insight

Because substrate degradation and decision validity are both monotone functions of a single shared latent 'substrate health' variable that is recoverable from the agent's own action-observation stream, the health estimate yields a calibrated interval on the success signal that is provably tight exactly when the substrate is identifiable, so shrinking the admissible action set by this interval is equivalent to a validity predicate that no point-estimate confidence score can provide.

## Method

```pseudocode
INPUT: action-observation stream {(a_t, o_t)}, frozen VLM policy pi, geometric tool set G (triangulation/ray-plane)
OUTPUT: certified admissible action set A_cert(t) and calibrated success interval [L_t, U_t]

PHASE 1 — Substrate health estimation (shared latent z_t)
  For each observation o_t:
    - Extract geometric residuals: triangulation reprojection error e_t, ray-plane
      consistency r_t, epipolar residual under estimated intrinsics k_t
    - Extract photometric residuals: occlusion mask coverage c_t, viewpoint entropy v_t
    - Extract distribution-shift score d_t = MMD(o_t, reference_buffer) with kernel bandwidth h=0.1
  Fit z_t via EM on the monotone link: (e_t, r_t, c_t, v_t, d_t) ~ f(z_t) + noise, z_t in [0,1]
  where f is constrained monotone decreasing in z_t (isotonic regression, 5 bins)
  [LOAD-BEARING ASSUMPTION, stated explicitly: z_t is a sufficient statistic that monotonically and jointly determines both the residuals and the true task-success probability. This holds only when all active degradation modes are co-monotone in the same direction; it is violated when modes (e.g., specular reflection) corrupt residuals without reducing success. We therefore treat z_t as a scalar summary and verify the assumption via the saturation diagnostic below.]

PHASE 2 — Monotone validity predicate
  Estimate success signal s_t = success_detector(o_t) (HexaShell-style log reader)
  Fit isotonic calibration s_t ~ g(z_t), g monotone nondecreasing, via PAV algorithm
  Compute conformal interval [L_t, U_t] on s_t using held-out calibration set of size n=200,
  width w_t = quantile_{1-alpha}(|s - g(z)|), alpha=0.1
  Tightness flag: interval is 'tight' iff identifiability margin m_t = |dg/dz| * Var(z_t) > tau=0.05
  [SATURATION DIAGNOSTIC: flag saturation when |dg/dz| < 0.01 at the operating point z_t. If saturated, fall back to a local linear fit of g in a window of ±0.1 around z_t and recompute m_t; if still saturated, mark the interval as non-tight regardless of m_t.]

PHASE 3 — Action-set shrinkage gate
  For candidate action a, compute predicted success s_hat(a) and admissible if L_t <= s_hat(a) <= U_t
  A_cert(t) = {a : s_hat(a) in [L_t, U_t]}  (shrink, not threshold)
  If |A_cert| = 0: trigger active perception (move wrist camera for second view,
  re-estimate k_t) and re-run Phase 1-2; max 3 retries
  Return A_cert(t), [L_t, U_t], tightness flag
```

**Why this design.** We chose a single shared latent z_t with a constrained monotone link over separate per-degradation detectors because the meta-gap requires coupling degradation to validity, and only a shared variable forces the coupling; the cost is that a single scalar cannot represent degradation modes that are non-monotone in the same direction, which we accept because the isotonic constraint is what makes the certificate interpretable. We chose conformal intervals over a learned variance head because conformal prediction gives finite-sample coverage without assuming the success detector is well-specified, accepting the cost of requiring a held-out calibration set that must be refreshed when the environment shifts. We chose action-set shrinkage over confidence thresholding because thresholding a point estimate (the anti-pattern of log-prob gating) discards the interval width that encodes substrate identifiability, accepting the cost of a combinatorial admissible-set computation that is tractable only for discrete action libraries. A domain expert would not describe this as a variant of AutoEval's automated success detection or of standard conformal prediction applied to a VLM, because the load-bearing object is the shared latent z_t whose monotone link to both residuals and success is what makes the interval tight only under identifiability — a property neither retrieval-based tool selection nor a standalone calibration head produces.

**Why it measures what we claim.** The isotonic-calibrated map g(z_t) measures decision validity because the monotone link assumption makes substrate health a sufficient statistic for success probability, and this assumption fails when a degradation mode (e.g., specular reflection) corrupts residuals without reducing actual task success, in which case g reflects a pessimistic validity estimate rather than true validity. The conformal width w_t measures certificate tightness because exchangeability of the calibration and test streams makes the interval coverage exact, and this assumption fails under distribution shift where the calibration buffer no longer matches the test distribution, in which case w_t reflects coverage under the stale reference rather than the current substrate. The identifiability margin m_t measures the provable-tightness condition because a nonzero derivative of the monotone link at the operating point makes the interval informative about z_t, and this assumption fails in saturated regions of the isotonic fit where dg/dz = 0, in which case m_t reflects flatness of the calibration curve rather than substrate identifiability. The action-set shrinkage |A_cert| measures admissible decision space because predicted success is monotone in z_t, and this assumption fails when the VLM policy's action logits are themselves miscalibrated independent of substrate, in which case the shrunk set reflects policy uncertainty rather than substrate validity.

## Contribution

["A substrate-validity certificate that couples perception degradation to a monotone decision-validity predicate via a single shared latent substrate-health variable estimated from the agent's own action-observation stream, with a conformal interval whose tightness is gated by an identifiability margin.", 'A design principle: gating decisions by shrinking the admissible action set according to a calibrated success interval, rather than by thresholding a point confidence score, preserves the interval width as a signal of substrate identifiability that thresholding discards.', 'An active-perception trigger that fires when the admissible set collapses, using wrist-camera re-viewing and intrinsics re-estimation to restore identifiability before committing to an action.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HexaShell held-out layouts, degraded-view splits | Tests generalization under controlled substrate corruption |
| Primary metric | Success-prediction agreement on certified set | Directly measures whether certificate tracks true validity |
| Baseline | AutoEval success detector | Tests whether shared-latent coupling beats standalone detection |
| Baseline | Point-estimate log-prob gating | Tests whether interval shrinkage beats thresholding anti-pattern |
| Baseline | Retrieval-based tool selection | Tests whether identifiability margin matters beyond retrieval |
| Ablation-of-ours | Per-degradation detectors, no shared z_t | Isolates the load-bearing shared-latent coupling |
| Ablation-of-ours | Saturation diagnostic disabled | Tests whether fallback estimator recovers tightness under saturation |

### Why this setup validates the claim

The central claim is that a shared latent z_t with a constrained monotone link to both geometric/photometric residuals and success is what makes the certificate tight exactly when the substrate is identifiable, and that shrinking the action set by the conformal interval preserves decisions that point-estimate gating discards. HexaShell's frozen-tool held-out layouts give a clean generalization test where substrate degradation (occlusion, specularity, viewpoint shift) is injected but task semantics are unchanged, so any agreement gap is attributable to validity estimation rather than task difficulty. AutoEval's success detector tests the sub-claim that standalone detection lacks the coupling that makes the interval informative; the log-prob gating baseline tests the sub-claim that thresholding a point estimate throws away width information; retrieval-based tool selection tests whether the identifiability margin adds anything beyond nearest-neighbor tool choice. The per-degradation ablation is the decisive control: if the shared z_t is truly load-bearing, removing it should collapse the tightness flag's correlation with actual identifiability while leaving raw success detection roughly intact. The saturation-diagnostic ablation tests whether the fallback estimator (local linear fit in a ±0.1 window) recovers tightness flags in saturated regions without inflating false positives. Success-prediction agreement on the certified set is the right metric because it directly asks whether actions admitted by [L_t, U_t] actually succeed — a metric that is insensitive to policy quality and sensitive only to certificate validity, which is exactly the object the method claims to produce.

### Expected outcome and causal chain

**vs. AutoEval success detector** — On a case where specular reflection corrupts triangulation residuals while the grasp still succeeds, the baseline produces a confident success prediction because its detector keys on visual appearance and has no channel linking residual corruption to validity. Our method instead widens [L_t, U_t] through the monotone link from z_t, because the shared latent absorbs the residual spike and the isotonic map pushes the interval up, so we expect a noticeable agreement gap on the specular subset but parity on clean-lighting subsets.

**vs. Point-estimate log-prob gating** — On a case where the VLM's action logits are sharply peaked but the substrate is degraded (high occlusion coverage c_t), the baseline admits the high-log-prob action because thresholding discards interval width. Our method instead shrinks A_cert because L_t rises with degradation and the action's predicted success falls below L_t, so we expect the baseline to admit a larger but less accurate set, showing lower precision on the certified set with a concentrated gap on high-occlusion instances.

**vs. Retrieval-based tool selection** — On a case where a novel degradation mode (unseen viewpoint entropy v_t) appears, the baseline retrieves the nearest tool from its buffer and applies it with no validity estimate because retrieval has no notion of substrate health. Our method instead flags the interval as non-tight (m_t below tau) and triggers active perception, so we expect the baseline to fail silently on novel-degradation instances while our method either recovers after retries or correctly reports non-tightness, producing a gap in silent-failure rate rather than in average success.

**vs. Per-degradation detectors, no shared z_t** — On a case where specular reflection and occlusion co-occur, the baseline fits separate isotonic maps per residual and combines them by a fixed rule (e.g., average), which cannot represent the joint monotone link and thus produces intervals that are either too wide (over-conservative) or too narrow (over-confident) depending on the mode. Our method instead uses the shared z_t to couple the residuals, so the interval width reflects the joint identifiability; we expect the ablation to show lower correlation between the tightness flag and actual identifiability (measured by ground-truth z_t on synthetic data) and a larger gap on mixed-mode instances.

**vs. Saturation diagnostic disabled** — On a case where the isotonic fit is saturated (dg/dz ≈ 0) despite an identifiable substrate, the baseline (no diagnostic) reports m_t ≈ 0 and marks the interval non-tight, triggering unnecessary active perception. Our method instead applies the local linear fallback in a ±0.1 window, recomputes m_t, and correctly flags tightness when the local slope is nonzero, so we expect the ablation to show a higher false non-tight rate on saturated-but-identifiable instances without a corresponding gain in agreement.

### What would falsify this idea

If the shared-latent method's agreement gain over the per-degradation ablation is uniform across clean and degraded subsets rather than concentrated on the subsets where the monotone link predicts interval widening, the claim that z_t's coupling is load-bearing is wrong. Likewise, if the tightness flag m_t fails to correlate with actual identifiability (i.e., non-tight intervals occur equally often on identifiable and saturated substrates), the identifiability-margin mechanism is falsified.

## References

1. HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents
2. AutoEval: Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real World
3. Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents
4. AlphaEvolve: A coding agent for scientific and algorithmic discovery
5. Autonomous Improvement of Instruction Following Skills via Foundation Models
6. ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI
7. Multi-objective Evolution of Heuristic Using Large Language Model
8. Amplifying human performance in combinatorial competitive programming
