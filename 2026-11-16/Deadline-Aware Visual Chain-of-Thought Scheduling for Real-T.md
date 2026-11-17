# Deadline-Aware Visual Chain-of-Thought Scheduling for Real-Time Vision-Language-Action Control

## Motivation

Existing vision-language-action systems treat visual chain-of-thought planning and real-time serving as separate concerns: CoT-VLA (E1) autoregressively predicts future image goals with no bound on how many reasoning steps it consumes per control tick, while Gamer (E2) engineers a low-latency cloud-edge inference engine but leaves the autoregressive visual-goal loop unbounded, so its latency guarantee holds only for the serving path and not for the reasoning path. Structurally, neither work exposes the number of visual CoT steps as a schedulable resource, so under a hard control deadline the planner either overruns the budget or is truncated by an arbitrary fixed cap that ignores which intermediate goal is actually best for the current action.

## Key Insight

Because each intermediate visual goal in an autoregressive visual-CoT loop is a self-contained, action-decoding-sufficient state, the loop admits a monotone quality-versus-latency frontier on which the best-available goal at any deadline is a well-defined function of elapsed wall-clock time rather than of an external step cap, which is the invariant CoT-VLA and Gamer both lack.

## Method

**What it is.** DAVIS (Deadline-Aware Visual CoT Scheduling) is a closed-loop controller that wraps a CoT-VLA-style autoregressive visual-goal predictor and a Gamer-style cloud-edge serving layer, taking as input the current observation, language instruction, a hard per-step deadline $D$ (e.g. 50 ms for 20 Hz control), and a per-step latency model, and outputting a robot action decoded from the best visual goal available at deadline expiry together with a per-step latency ledger.

**Load-bearing assumption (stated explicitly).** The scheduling law assumes that each completed autoregressive visual goal is action-decodable by $A_\phi$ and that goal quality is monotonically non-decreasing in step index $k$, so that 'last completed goal at deadline' is always the best-available goal and step count $k^*$ equals usable planning depth. This assumption is tested, not asserted: the edge node runs a deadline-budgeted action-compatibility check (step 8 below) that rejects a candidate goal whose decoded action falls outside the action head's feasible set, and the experiment reports the fraction of completed goals that are action-incompatible on the target suite.

**How it works.**

```pseudocode
Input: obs o_t, instruction l, deadline D, latency model L(.),
       visual-goal predictor G_theta, action head A_phi,
       feasible-set check F(.) (bounds on joint/EE deltas from A_phi),
       fallback goal g_prev (last tick's committed goal)
Output: action a_t, ledger (k*, t_elapsed, slack, n_rejected)

1. t0 <- now(); g_0 <- encode(o_t, l)              # prefill, cost L_pre measured online
2. k <- 0; goals <- [g_0]; best <- g_0; n_rejected <- 0
3. while now() - t0 + L_step(goals) < D - L_act - L_check:  # L_step from rolling p50 over last 32 steps; L_check = p50 of F(.) over last 32 checks
4.     g_{k+1} <- G_theta(g_k | o_t, l)            # one autoregressive visual-goal step
5.     k <- k + 1; goals.append(g_{k+1})
6.     a_cand <- A_phi(g_{k+1})                   # decode candidate action
7.     if F(a_cand) is feasible: best <- g_{k+1}  # commit only action-compatible goal
8.     else: n_rejected <- n_rejected + 1        # keep previous best; do not commit
9.     if now() - t0 + L_step(goals) >= D - L_act - L_check: break
10. if best == g_0 and g_prev exists: best <- g_prev  # fallback to previous tick's goal when no new goal is compatible
11. a_t <- A_phi(best)                             # decode action from best-available compatible goal
12. ledger <- (k, now() - t0, D - (now() - t0), n_rejected)
13. update L_step and L_check with observed latencies (EMA, alpha=0.3)
```

A cloud-edge split mirrors Gamer: the cloud node runs $G_\theta$ steps and streams completed goals to the edge node, which holds the latest goal and fires $A_\phi$ at deadline expiry; if the cloud stream stalls, the edge decodes from the most recent goal it holds, so the deadline is enforced on the edge clock, not the cloud clock. The action-compatibility check $F(\cdot)$ runs on the edge and is a fixed-threshold feasibility test on the decoded action (e.g. per-joint delta within $\pm 0.1$ rad of the current joint state, end-effector delta within $\pm 5$ cm), costing $L_\text{check}$ measured online; it does not add a learned scoring head.

**Why this design.** We chose a wall-clock-driven loop condition over a fixed step cap (as in CoT-VLA's fixed autoregressive budget) because a fixed cap cannot adapt to per-step latency variance across cloud-edge hops, accepting the cost that the controller must maintain a calibrated latency model $L_\text{step}$. We chose 'last completed goal is best-available' over an intermediate-goal scoring head because scoring heads reintroduce an uncalibrated confidence signal (the failure mode of log-prob gating), accepting the cost that the controller forgoes selecting an earlier goal that might be better for the current action; the deadline-budgeted feasibility check $F(\cdot)$ is a fixed-threshold reachability test, not a learned confidence head, so it does not reintroduce that failure mode. We chose an edge-side deadline enforcer over a cloud-side one because the edge clock is the only clock that bounds the actuator command, accepting the cost that the edge may occasionally decode from a stale goal when the cloud stalls. A domain expert would not describe DAVIS as 'CoT-VLA plus Gamer': CoT-VLA has no latency model and Gamer has no visual-CoT loop, and neither exposes step count as a schedulable resource with an edge-enforced deadline; DAVIS's contribution is the scheduling law, not the composition.

**Why it measures what we claim.** The quantity $k^*$ (number of completed visual-goal steps at deadline) measures 'reasoning depth actually delivered under the latency budget' because the assumption that each completed goal is action-decodable makes $k^*$ equivalent to the number of usable planning steps; this assumption fails when a goal is visually plausible but action-incompatible (e.g. a goal image the action head cannot reach), in which case $k^*$ reflects visual-step count rather than usable planning depth, and the ledger's $n_\text{rejected}$ counts exactly those steps so the gap is observable. The quantity $t_\text{elapsed}$ measures 'deadline adherence' because the assumption that the edge clock is the actuator clock makes elapsed edge time equivalent to control-tick latency; this assumption fails under clock skew between cloud and edge, in which case $t_\text{elapsed}$ reflects edge-local time only. The quantity $L_\text{step}$ (rolling p50) measures 'per-step reasoning cost' because the assumption that step latency is locally stationary over a 32-step window makes the rolling median equivalent to the next step's cost; this assumption fails during thermal throttling or network congestion, in which case $L_\text{step}$ reflects recent rather than imminent cost, and the calibration study below reports the p50/p95 gap between predicted and realized next-step latency.

## Contribution

(1) DAVIS, a deadline-aware scheduling wrapper that turns the number of visual-CoT steps into a schedulable resource with an edge-enforced hard deadline and a rolling latency model. (2) A design principle: 'best-available visual goal at deadline' is a well-defined decode target that requires no confidence calibration, unlike step-cap or log-prob gating. (3) A latency-accounting ledger and evaluation protocol that reports action latency and task success as a function of reasoning budget on long-horizon tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Deadline-stressed real-robot manipulation suite (CoT-VLA real tasks) plus a deterministic latency-injection replay harness (tc/netem bandwidth and delay shaping + CPU-throttling script) that replays recorded latency traces | Provides hard per-step deadlines and real actuator clock, and makes high-variance episodes reproducible rather than dependent on incidental congestion |
| Primary metric | Task success rate at fixed deadline D (20 Hz) | Directly measures usable planning depth under budget |
| Secondary metrics | (i) $k^*$ distribution per latency-variance stratum; (ii) fraction of completed goals that are action-incompatible ($n_\text{rejected}/k^*$); (iii) $L_\text{step}$ calibration: p50 and p95 of |predicted − realized| next-step latency, and its correlation with success; (iv) deadline-adherence rate ($t_\text{elapsed} \le D$) | Operationalizes the decodability and stationarity assumptions so they are measured, not asserted |
| Baseline | CoT-VLA (fixed autoregressive step cap) | Tests whether deadline-aware scheduling beats fixed budget |
| Baseline | Gamer-style cloud-edge serving (no visual CoT) | Tests whether visual-goal loop adds value under deadline |
| Baseline | Log-prob-gated early-exit CoT-VLA | Tests whether our last-completed-goal rule beats confidence gating |
| Ablation-of-ours | DAVIS with fixed step cap instead of wall-clock loop | Isolates the scheduling law from the CoT-VLA backbone |
| Ablation-of-ours | DAVIS without the action-compatibility check $F(\cdot)$ (unconditional last-completed-goal) | Isolates the contribution of the decodability check to the scheduling law |

### Why this setup validates the claim

The central claim is that treating visual-CoT step count as a schedulable resource under an edge-enforced deadline yields more usable planning depth per control tick than any fixed budget or confidence-gated alternative. The dataset choice matters because the claim is about latency variance, not raw capability: a deadline-stressed real-robot suite with cloud-edge hops, replayed through a deterministic latency-injection harness, is the only setting where per-step latency actually varies enough to make the scheduling law bite while remaining reproducible. Success rate at a fixed deadline is the right primary metric because it collapses reasoning depth and deadline adherence into the single quantity the method claims to improve; a latency-only metric would not distinguish deeper reasoning from faster but shallower decoding. The secondary metrics directly test the load-bearing assumption: the action-incompatibility fraction measures whether completed goals are action-decodable, and the $L_\text{step}$ calibration error measures whether the rolling p50 is a valid predictor of imminent step cost. CoT-VLA tests the fixed-cap sub-claim, Gamer-style serving tests the no-visual-CoT sub-claim, and log-prob gating tests the last-completed-goal design choice. The fixed-cap ablation on our own backbone isolates whether any gain comes from the scheduling law rather than from the CoT-VLA predictor itself, and the no-$F(\cdot)$ ablation isolates whether the decodability check is what makes the scheduling law select on usable planning depth; a win over CoT-VLA that vanishes in the fixed-cap ablation would falsify the contribution claim.

### Expected outcome and causal chain

**vs. CoT-VLA (fixed autoregressive step cap)** — On a case where per-step latency spikes mid-episode (cloud congestion or thermal throttling, replayed deterministically by the latency-injection harness), the baseline produces a fixed number of visual-goal steps regardless of wall clock, so it either overruns the deadline and drops control ticks or, if capped conservatively, reasons too shallowly on fast steps because its mechanism cannot observe latency at all. Our method instead spends exactly the steps that fit before D and stops, because the wall-clock loop condition reads the rolling latency model, so we expect a noticeable success gap concentrated on high-latency-variance episodes and near-parity on low-variance ones.

**vs. Gamer-style cloud-edge serving (no visual CoT)** — On a case requiring multi-step spatial planning (e.g. a goal image behind an occluder), the baseline produces a direct action from the current observation because it has no visual-goal loop to build intermediate targets, so it fails on tasks whose correct action is not locally inferable. Our method instead decodes from the latest completed visual goal, which encodes the intermediate spatial target, so we expect a clear gap on multi-step manipulation subsets and parity on single-step reaching tasks.

**vs. Log-prob-gated early-exit CoT-VLA** — On a case where the model's confidence is miscalibrated (a visually plausible but action-incompatible goal gets high log-prob), the baseline produces an early exit on a bad goal because its gating signal is uncalibrated confidence. Our method instead decodes from the last completed goal that passes the fixed-threshold feasibility check $F(\cdot)$, because it uses completion order plus a reachability test rather than a confidence head, so we expect fewer catastrophic failures on the miscalibrated subset with a modest cost on the subset where early exit was correct.

**vs. DAVIS without the action-compatibility check $F(\cdot)$** — On a case where a visually plausible but action-incompatible goal is generated late in the loop, the no-check variant commits that goal because it uses completion order alone, so it decodes an infeasible action and fails; our full method rejects the goal (incrementing $n_\text{rejected}$) and falls back to the previous compatible goal, so we expect the full method to dominate on the subset with non-zero action-incompatibility fraction and to match the no-check variant when that fraction is near zero.

### What would falsify this idea

If DAVIS's gain over CoT-VLA is uniform across latency-variance strata rather than concentrated on high-variance episodes, the scheduling-law explanation is wrong and the gain is just a better backbone. If the fixed-cap ablation of DAVIS matches full DAVIS, the wall-clock loop contributes nothing and the central claim collapses; likewise, if the action-incompatibility fraction is high and the no-$F(\cdot)$ ablation matches full DAVIS, the decodability premise fails and the scheduling law selects on completion order rather than usable planning depth.

## References

1. Gamer: A Generalist Vision-Language-Action Foundation Model for Long-Horizon Real-Time Game Agent
