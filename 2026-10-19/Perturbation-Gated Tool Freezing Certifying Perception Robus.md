# Perturbation-Gated Tool Freezing: Certifying Perception Robustness of VLM-Written Robot Tools Before Deployment

## Motivation

Across the current tool-building paradigm, VLM coding agents such as HexaShell synthesize and freeze robot perception tools after observing only ten clean, calibrated layouts, which structurally guarantees that the frozen tools inherit the harness's assumption of clean observations. This is the same structural failure that recurs across the broader manipulation stack — LIBERO-Plus, MoGe-2, and GR00T N1 all report a residual generalization gap to perception perturbations — because none of these systems ever exposes the tool-selection or tool-synthesis process to viewpoint, lighting, texture, or sensor-noise variation during development. The root cause is not insufficient data but the absence of any mechanism that injects perturbations into the development loop and gates tool freezing on measured robustness, so tools are frozen on a distribution they will never see again.

## Key Insight

A perception tool's robustness is a property of its code structure (which geometric primitives and fallbacks it invokes), so injecting a calibrated perturbation ensemble at the tool-development stage and requiring a per-tool robustness certificate before freezing converts an untestable post-hoc generalization claim into a structural precondition that the VLM agent must satisfy by rewriting its own tool.

## Method

```pseudocode
# PERTURB-GATE: perturbation-injection layer over the HexaShell harness

INPUT:  task T, VLM coding agent A, base harness H (3 calibrated RGB cams, 2-arm robot),
        perturbation family P = {viewpoint, lighting, texture, sensor_noise},
        robustness threshold tau = 0.80 (success-rate ratio perturbed/clean),
        n_dev = 10 clean layouts, n_pert = 6 perturbations x 3 severities x 2 seeds

PHASE 1 — Clean tool synthesis (unchanged from HexaShell)
  for layout in n_dev:
      A observes 3 RGB views, issues CLI commands, reads failure logs,
      writes/repairs tool code t using triangulation + ray-plane intersection
  -> candidate tool t_0

PHASE 2 — Perturbation injection layer
  for each perturbation p in P:
      for severity s in {low, med, high}:
          for seed in {1,2}:
              render observation o' = p(o, s, seed)   # camera pose jitter, gain/exposure, texture swap, Gaussian+shot noise
              run t_k on o' on the SAME layout, log success/failure + failure cause
  -> robustness profile R(t_k) = [r_p,s for each cell], and per-cell failure-cause histogram

PHASE 3 — Robustness certificate + gating
  rho(t_k) = min over p of (mean_{s,seed} success_p / mean success_clean)
  if rho(t_k) >= tau:  FREEZE t_k  -> deploy on n_test=10 unseen layouts
  else:
      A reads the failure-cause histogram, identifies the dominant fragile primitive
      (e.g., 'triangulation relies on a fixed pixel threshold that noise breaks'),
      rewrites t_k -> t_{k+1},  goto PHASE 2   (max K=5 iterations)

OUTPUT: frozen tool t* with certificate rho(t*) >= tau, evaluated on held-out layouts
```

**Why this design.** We chose a *per-perturbation-family minimum* robustness ratio rho(t) over an average success rate because averaging lets a tool hide a catastrophic weakness in one family (e.g., viewpoint) behind strong performance in another (e.g., lighting), which is exactly the failure mode that produces the residual gap reported across LIBERO-Plus and GR00T N1; the cost is that rho is conservative and rejects some tools that would have deployed fine. We chose to *route the failure-cause histogram back to the VLM agent* rather than have the agent blindly retry, because the agent's only reliable diagnostic signal in HexaShell is the structured log, and a per-cell histogram tells it which geometric primitive (triangulation threshold, plane-intersection tolerance) to repair; the cost is that the histogram is coarse and may misattribute a multi-cause failure. We chose a *fixed severity ladder* {low, med, high} with 2 seeds over a continuous severity distribution, because a discrete ladder is reproducible and cheap enough to run inside the tool-development loop (6 perturbations x 3 severities x 2 seeds = 36 runs per tool iteration); the cost is that the certificate does not interpolate to severities between rungs. We chose to gate *before* freezing rather than to fine-tune the tool after freezing, because the HexaShell protocol freezes code and the whole point is to make freezing conditional on a measured property; the cost is that tools which cannot reach tau within K=5 rewrites are discarded rather than salvaged.

**Why this measures what we claim.** The quantity rho(t) — the minimum over perturbation families of the perturbed-to-clean success ratio — is claimed to measure *perception robustness*, and it does so because we assume that the perturbation ensemble P spans the deployment distribution shift (i.e., the real-world perturbations the tool will face are drawn from the same families and severities we inject); this assumption fails when deployment contains a perturbation family not in P (e.g., motion blur from fast arm movement), in which case rho reflects robustness only to the injected families and overstates deployment robustness. The failure-cause histogram is claimed to measure *which code primitive is fragile*, and it does so because we assume that a failure recurring under a specific perturbation family is caused by the geometric primitive that family stresses (viewpoint -> triangulation, noise -> pixel thresholds); this assumption fails when two primitives interact (a noise-induced pixel error that only matters under viewpoint jitter), in which case the histogram attributes the failure to one primitive and the agent's repair is incomplete. The rewrite loop's convergence to tau is claimed to measure *that the agent can structurally repair robustness*, and it does so because we assume that the agent's edits are localized to the flagged primitive and do not regress other families (we re-run all 36 cells each iteration to enforce this); this assumption fails when a repair trades one family's robustness for another's, in which case rho stays flat and the tool is discarded rather than deployed.

## Contribution

(1) PERTURB-GATE, a perturbation-injection layer that wraps a VLM robot-coding harness and turns tool freezing into a conditional operation gated on a per-family minimum robustness certificate. (2) A closed-loop repair protocol in which a per-perturbation-family failure-cause histogram is fed back to the VLM agent so that tool rewrites target the specific geometric primitive that the perturbation stresses, yielding tools whose robustness is measured rather than assumed. (3) An empirical design principle: robustness of VLM-written perception tools is governed by which geometric primitive the tool invokes and whether it has a perturbation-aware fallback, not by the amount of clean development data.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HexaShell 10-layout dev + 10-layout test per task | Same frozen-tool protocol isolates robustness from tool synthesis |
| Primary metric | ρ(t) = min over perturbation families of perturbed/clean success ratio | Detects hidden catastrophic weakness that averaging masks |
| Baseline | HexaShell frozen tool (no perturbation gating) | Tests whether gating adds robustness beyond clean-only synthesis |
| Baseline | Clean-only retry loop (K=5, no perturbation feedback) | Tests whether perturbation histogram drives repair vs blind retry |
| Baseline | Darwin Gödel Machine self-modification | Tests whether open-ended code search achieves robustness without certificate |
| Ablation-of-ours | PERTURB-GATE with mean (not min) robustness ratio | Tests whether min-over-families is the causal driver of gains |

### Why this setup validates the claim

The central claim is that a per-perturbation-family minimum robustness certificate, fed back as a failure-cause histogram to the VLM coding agent, produces frozen tools that survive deployment shifts better than tools frozen after clean-only development. The HexaShell dataset is the right substrate because it already separates tool synthesis (10 dev layouts) from deployment (10 unseen layouts) under a freeze protocol, so any gain on test layouts cannot come from continued adaptation. The primary metric ρ(t) is chosen because it is the exact quantity the method gates on: if the method works, the distribution of ρ(t) at freeze time should shift upward relative to baselines, and that shift should predict test-layout success. The HexaShell frozen-tool baseline tests the sub-claim that perturbation gating is necessary at all; the clean-only retry baseline tests the sub-claim that the histogram, not just extra iterations, drives repair; the Darwin Gödel Machine baseline tests the sub-claim that certificate-gated repair beats open-ended self-modification for this robustness target. The mean-ratio ablation is the critical control: if min-over-families is doing the causal work, replacing it with a mean should reintroduce the hidden-weakness failure mode and collapse the test-layout gain, even though the loop structure is identical.

### Expected outcome and causal chain

**vs. HexaShell frozen tool (no perturbation gating)** — On a layout where the tool's triangulation uses a fixed pixel threshold, the baseline freezes after clean-only synthesis and produces a tool that fails under sensor noise because the threshold was never stressed during development. Our method instead injects noise perturbations during Phase 2, the histogram flags the threshold primitive, and the agent rewrites it with a noise-adaptive tolerance, so we expect a large gap on the noise-perturbed test subset and near-parity on clean test layouts.

**vs. Clean-only retry loop (K=5, no perturbation feedback)** — On a case where viewpoint jitter shifts the ray-plane intersection, the baseline retries clean layouts up to K=5 but never observes the failure because clean layouts do not stress viewpoint, so it freezes a tool with a brittle intersection tolerance. Our method instead routes the viewpoint-cell failure histogram back to the agent, which localizes the repair to the intersection primitive, so we expect the baseline to show repeated identical failures across iterations while our method shows monotone improvement in the viewpoint cell and a corresponding test-layout gain concentrated on layouts with camera-pose variation.

**vs. Darwin Gödel Machine self-modification** — On a task where a robustness fix in one primitive (e.g., noise threshold) regresses another (e.g., lighting gain), the baseline's open-ended search archives variants without a per-family certificate and drifts toward variants that trade families against each other, so it freezes a tool with high average but low minimum robustness. Our method instead re-runs all 36 cells each iteration and gates on the minimum, rejecting family-trading repairs, so we expect the baseline to show higher variance across perturbation families and a lower test-layout minimum, while our method shows a tighter per-family profile and higher worst-family test success.

### What would falsify this idea

If the test-layout gain over the HexaShell frozen-tool baseline is uniform across perturbation subsets rather than concentrated on the subsets the certificate was designed to protect (noise, viewpoint, lighting, texture), then the gating is not the causal driver and the central claim is wrong. Similarly, if the mean-ratio ablation matches the min-ratio method on test layouts, then min-over-families is not doing the work the argument claims.

## References

1. HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents
2. Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents
3. π0.5: a Vision-Language-Action Model with Open-World Generalization
