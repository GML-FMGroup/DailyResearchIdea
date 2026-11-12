# Pareto-Invariant Chunked Scheduling for Non-Stationary Agent Contexts

## Motivation

Agentic serving systems such as Evolving Infra adapt compute allocation and KV management at runtime, but they do so through heuristics (online planner matching observed PD composition, agent-signal-driven eviction) with no stated invariant tying batching decisions to the latency-throughput frontier. The consequence is structural: Sarathi-Serve's chunked-prefill Pareto frontier is derived under a stationary request mixture, so when an agent's context grows or its tool-use pattern shifts mid-session, the batching policy that was Pareto-optimal for the previous context distribution silently becomes interior to the new frontier, and no existing system detects or corrects this because the planner has no representation of 'which frontier am I on'. Evolving Infra's reported gains over Dynamo are measured as aggregate latency reductions, not as frontier preservation, so the non-stationarity failure is invisible to its own evaluation.

## Key Insight

Chunk size is a monotone control on the prefill-decode interference term, so the Pareto-optimal chunk size is a monotone function of a single scalar — the current context-length-induced decode stall pressure — which means frontier membership can be tracked online by a one-dimensional monotone estimator rather than re-derived from a stationary workload model.

## Method

**What it is.** PARIS (Pareto-Aware Refinement-Invariant Scheduler) is a chunked-prefill batching policy for LLM serving that outputs, per scheduling tick, a chunk size $c_t$ and a token budget split between prefill and decode. Its inputs are per-request context lengths, observed decode inter-token stalls, the current KV occupancy, and the prefill token count issued in the previous tick; its output replaces the fixed/adaptive chunk size in Sarathi-Serve-style schedulers.

**Load-bearing assumption (stated explicitly).** PARIS assumes that the normalized excess decode inter-token gap $s_t$ is monotone non-decreasing in context length within an agent session, so that a one-dimensional isotonic floor estimator can track the Pareto-optimal chunk size without a workload model. This assumption holds when decode is prefill-contention-bound; it fails when decode is KV-bandwidth-bound (Pope et al., MLSys 2023; Williams et al., CACM 2009), in which case $s_t$ reflects KV pressure rather than batching interference. PARIS therefore decomposes $s_t$ into a prefill-contention component and a KV-pressure component (step 1b) and uses only the prefill coefficient as the pressure signal, and it replaces the running-max isotonic floor with a two-state estimator that can decrease when the prefill coefficient decreases (step 2).

**How it works.**

```pseudocode
state: prefill-pressure estimate p_hat (scalar), chunk ladder C = [64,128,256,512,1024]
       per-tick: requests R with context lengths L_i, decode set D
       rolling window W_fit = 32 ticks of (s_t, prefill_tokens_t, kv_occupancy_t)

# Calibration (offline, once per model/hardware):
#   Run a controlled microbenchmark: for context length L in {512,1024,2048,4096,8192}
#   and KV occupancy in {25%,50%,75%,100%}, measure s_t with prefill disabled (decode-only).
#   Fit s_t = a * prefill_tokens + b * kv_occupancy + c.  Store (a,b,c) as calibration.
#   This validates or refutes the monotonicity premise before deployment.

every tick t:
  # 1. Measure decode stall pressure (the invariant's observable)
  s_t = mean over i in D of (observed_inter_token_gap_i - roofline_gap_i)
        normalized by roofline_gap_i          # dimensionless stall ratio

  # 1b. Decompose s_t into prefill-contention and KV-pressure components
  #     using the calibration coefficients (a,b,c) and the current prefill_tokens_t, kv_occupancy_t
  s_prefill_t = a * prefill_tokens_t
  s_kv_t      = b * kv_occupancy_t + c
  #     Use only the prefill component as the pressure signal:
  p_obs = s_prefill_t

  # 2. Two-state estimator (replaces running-max isotonic floor)
  #     p_hat can decrease when p_obs decreases; it is not a running maximum.
  if p_obs >= p_hat:
      p_hat <- p_obs                         # rise immediately
  else:
      p_hat <- p_hat * exp(-lambda * dt) + (1 - exp(-lambda * dt)) * p_obs   # decay toward p_obs
  #     lambda = 0.5/s, dt = tick duration in seconds

  # 3. Map pressure to chunk size via a monotone ladder selector
  #    higher stall pressure => smaller chunk (less decode interference)
  idx = clamp(round(log2(p_hat / p_ref) * kappa), 0, len(C)-1)   # kappa = 2.0, p_ref = 0.1
  c_t = C[len(C)-1-idx]

  # 4. Two-sided Pareto-tracking probe: perturb chunk size every W=8 ticks
  if t mod W == 0:
     c_probe_up   = C[idx_neighbor_up]    # one ladder step toward larger chunk
     c_probe_down = C[idx_neighbor_down]  # one ladder step toward smaller chunk
     measure throughput and P95 TTFT under c_probe_up and c_probe_down for this tick
     # Frontier membership is confirmed by local dominance in BOTH directions:
     if (throughput, 1/P95TTFT) at c_t dominates both probe points:
         keep c_t; freeze idx for 2W ticks   # frontier move detected
     else:
         # move toward the dominating probe (if any), else keep c_t
         if c_probe_up dominates c_t: idx <- idx_neighbor_up
         elif c_probe_down dominates c_t: idx <- idx_neighbor_down
         else: keep c_t
```

The two-state estimator in step 2 is the load-bearing element: it encodes the structural claim that decode stall pressure cannot decrease while contexts are growing, so any observed decrease must be decay (contexts shrank / evicted), not noise — but unlike a running maximum, it can follow a genuine decrease when the prefill coefficient drops. The two-sided probe in step 4 is a local finite-difference on the Pareto surface, not a re-solve; it costs two ticks per 8 and detects when the agent's tool-use shift has moved the frontier in either direction.

**Why this design.** We chose a one-dimensional prefill-pressure estimator over Evolving Infra's multi-signal online planner because the planner's inputs (PD composition, agent context signals) are high-dimensional and non-stationary, so any planner fitted to them is chasing a moving target, accepting the cost that we ignore cross-request heterogeneity that a full planner could exploit. We chose a fixed geometric chunk ladder with a monotone index map over continuous optimization because the ladder makes the probe a single-step comparison with a known neighbor, accepting the cost of quantization error up to one ladder rung. We chose a periodic two-sided perturb-and-revert probe over a learned bandit over chunk sizes because the bandit's exploration would itself perturb the latency distribution the agent observes, accepting the cost that frontier moves between probes are detected with up to W-tick lag. A domain expert would not describe this as a variant of Evolving Infra's planner: the planner *predicts* a good split from workload features, whereas PARIS *asserts and maintains* a monotonicity invariant and only probes to confirm frontier membership — the planner has no invariant, and PARIS has no workload model.

**Why it measures what we claim.** The stall ratio $s_t$ measures decode-interference pressure because the assumption that roofline inter-token gap is context-independent makes the normalized excess equal the prefill-induced stall; this assumption fails when an agent's decode is memory-bandwidth-bound by KV growth rather than prefill contention, in which case $s_t$ reflects KV pressure instead of batching interference. The prefill coefficient $a$ from the calibration regression measures the prefill-contention component because the assumption that $s_t$ is linear in prefill tokens and KV occupancy makes the coefficient isolate the prefill effect; this fails when the two are collinear (e.g., prefill tokens and KV occupancy grow together), in which case the coefficient is biased. The two-state estimator $\hat p$ measures frontier drift because the assumption that context length is non-decreasing within an agent session makes pressure monotone, so any frontier move must appear as a monotone shift in $\hat p$; this fails when tool calls truncate context, in which case the decay term attributes the drop to time rather than eviction. The two-sided probe's dominance test measures Pareto-frontier membership because the assumption that the local throughput/TTFT tradeoff is unimodal in chunk size makes a non-dominating neighbor in both directions evidence of frontier movement; this fails under bimodal interference, where a non-dominating probe may reflect measurement noise rather than a frontier shift.

## Contribution

(1) A context-dependent Pareto-invariant scheduler (PARIS) that replaces heuristic agent-adaptive allocation with a monotone pressure estimator plus a perturb-and-revert frontier probe, giving chunked-prefill serving an explicit invariant under non-stationary agent contexts. (2) The empirical finding that decode stall pressure is monotone in context growth within agent sessions, which is what licenses a one-dimensional controller in place of a multi-signal planner. (3) A probe-based frontier-membership test that detects when an agent's tool-use shift has moved the Sarathi-Serve frontier, with quantified detection lag as a function of probe period.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Agentic coding + computer-use traces (GLM-5.2, Kimi K2.6) | Non-stationary context growth stresses frontier drift |
| Primary metric | P95 TTFT at fixed decode SLO | Directly reflects prefill-decode interference tradeoff |
| Secondary metric | Distance to offline per-tick Pareto frontier (frontier preservation) | Directly measures the invariant's target, not just a latency delta |
| Baseline | Dynamo | Production fixed-policy reference point |
| Baseline | Evolving Infra planner (commit pinned; fallback reimplementation) | Multi-signal predictive planner competitor |
| Baseline | Sarathi-Serve fixed chunk | The scheduler PARIS directly replaces |
| Baseline | Learned bandit over same chunk ladder | Isolates the invariant from the ladder |
| Ablation-of-ours | PARIS w/o monotone projection (two-state estimator replaced by running max) | Isolates the invariant's contribution |
| Ablation-of-ours | PARIS w/o prefill/KV decomposition (uses raw s_t) | Tests the load-bearing assumption directly |

### Why this setup validates the claim

The central claim is that a one-dimensional prefill-pressure invariant, plus cheap two-sided probing, tracks the throughput/TTFT Pareto frontier under agent-driven context drift better than either a static policy or a high-dimensional predictive planner. Dynamo and Sarathi-Serve test the static-policy half: if PARIS only matched them, the invariant buys nothing over a fixed chunk. The Evolving Infra planner tests the predictive half: it is the strongest competitor precisely where workload features are informative, so any PARIS advantage must come from not modeling the workload at all. The learned bandit over the same ladder tests whether the invariant, not the ladder, is the novel contribution: if the bandit matches PARIS, the ladder alone suffices. The ablation without monotone projection isolates whether the invariant, rather than the probe or the ladder, is load-bearing; the ablation without prefill/KV decomposition tests the load-bearing assumption directly. P95 TTFT at a fixed decode SLO is the right primary metric because the predicted effect is a latency-tail improvement from reduced prefill-decode interference, not a throughput gain; a metric that averaged over the tail would hide exactly the interference the invariant is meant to suppress. The secondary metric — distance to an offline per-tick Pareto frontier computed by sweeping chunk sizes on each tick's request set — directly measures frontier preservation, so the significance claim rests on a new evaluation axis rather than a P95 delta that may be within noise.

### Expected outcome and causal chain

**vs. Dynamo** — On a case where an agent's context grows steadily across a long tool-use session, Dynamo's fixed batching keeps a large chunk size, so prefill bursts repeatedly interrupt in-flight decodes and the latency tail inflates. Our method instead shrinks the chunk as the prefill-pressure estimate rises, because the two-state estimator tracks the prefill coefficient rather than averaging it away, so we expect a visible P95 TTFT gap concentrated in the late-session, long-context subset and near-parity on short sessions.

**vs. Evolving Infra planner** — On a case where the workload shifts mid-session from short coding turns to long computer-use contexts, the planner's high-dimensional features go stale and it keeps predicting the pre-shift split, so its chunk choice lags the frontier. Our method instead detects the shift through the two-sided probe's dominance test and moves to a smaller chunk within one probe window, because the prefill-pressure estimate makes any frontier move appear as a monotone shift, so we expect a modest but consistent P95 advantage on shift-heavy traces and parity on stationary ones.

**vs. Sarathi-Serve fixed chunk** — On a case where decode stalls spike transiently while contexts are still growing, a fixed chunk cannot react, so the stall persists until the burst ends. Our method instead lowers the chunk on the rising prefill-pressure estimate, because the two-state estimator preserves the spike rather than decaying it as noise, so we expect a sharper reduction in P95 TTFT during stall bursts with little change in mean throughput.

**vs. learned bandit over same ladder** — On stationary traces, the bandit and PARIS should match, because both explore the same ladder. On shift-heavy traces, the bandit's exploration perturbs the latency distribution and its reward signal is non-stationary, so it lags the frontier; PARIS's invariant gives it a monotone target, so we expect PARIS to have lower frontier distance on shift-heavy traces and parity on stationary ones.

**vs. PARIS w/o monotone projection** — Replacing the two-state estimator with a running max makes $\hat p$ monotone by construction, so it cannot follow a genuine decrease when tool calls truncate context; we expect this ablation to over-chunk (too-small chunks) after truncation events, inflating P95 TTFT on traces with frequent tool-call truncation, and to match full PARIS on traces without truncation.

**vs. PARIS w/o prefill/KV decomposition** — Using raw $s_t$ instead of the prefill coefficient makes the pressure signal reflect KV-bandwidth stall when decode is memory-bound; we expect this ablation to shrink chunks unnecessarily under high KV occupancy, reducing throughput without improving P95 TTFT, and to match full PARIS only when KV occupancy is low.

### What would falsify this idea

If PARIS's P95 TTFT gain over the Evolving Infra planner is uniform across stationary and shift-heavy traces rather than concentrated where the predicted frontier drift occurs, the invariant-and-probe mechanism is not what produces the gain. Equally, if the ablation without monotone projection matches full PARIS, or if the ablation without prefill/KV decomposition matches full PARIS, the load-bearing structural claim is wrong.

## References

1. Evolving Infra: Agent-Adaptive LLM Serving at Scale
