# Deadline-Closure KV Scheduling for LLM Serving via Drift-Plus-Penalty

## Motivation

Evolving Infra's agent-aware KV manager and online planner assume workload stationarity and treat per-request latency SLOs as soft objectives to be traded off, so when the workload shifts away from agentic patterns (e.g., non-agentic high-throughput batch inference) the agent-signal-driven eviction/migration policy has no mechanism that preserves per-request deadlines. The structural root cause is that the planner and KV manager are coupled only through heuristic signals from agent context management, not through any invariant that binds current scheduling decisions to future deadline feasibility; thus the shared residual gap — decoupled compute/KV scheduling failing to preserve deadlines under distribution shift — recurs across all descendant branches of Evolving Infra.

## Key Insight

Per-request deadline feasibility can be encoded as a per-queue virtual-queue backlog whose drift-plus-penalty minimization yields a closed-form eviction/migration rule that keeps the backlog bounded under any workload distribution, because the bound depends only on realized arrivals and service rates rather than on the stationarity or agentic structure of the arrival process.

## Method

**What it is.** We introduce Lyra-KV, a workload-agnostic SLO-closure controller that sits between the prefill/decode compute allocator and the distributed KV manager. Its inputs are per-request remaining deadline slack, per-request KV block residency state, and per-node memory/service capacity; its outputs are, at each scheduling epoch, a prefill-decode split ratio, an eviction set of KV blocks, and a migration set of KV blocks.

**How it works.**
```pseudocode
# Epoch t, duration tau (tau = 20 ms; control horizon T = 5 epochs)
# State per request i: residual deadline d_i(t), remaining tokens r_i(t),
#   KV blocks B_i(t) resident on node n(i)
# Virtual queues (per node n, per request class c):
#   Q_nc(t+1) = max(0, Q_nc(t) + A_nc(t) - mu_nc(t))   # deadline-backlog queue
#   Z_n(t+1)   = max(0, Z_n(t) + sum_i in n |B_i(t)| - C_n)  # memory-overflow queue
# Per-request deadline-violation queue (new, replaces class-aggregated Q_nc for feasibility):
#   Y_i(t+1) = max(0, Y_i(t) + 1[slack_i(t) < 0] * r_i(t) / mu_hat_i(t) - 1[slack_i(t) >= 0] * mu_hat_i(t) * tau)
#   where slack_i(t) = d_i(t) - r_i(t) / mu_hat_i(t)
# Weight V = 8.0 (penalty-vs-drift trade-off), kappa = 1.2 (deadline pressure gain)
# Class granularity: 4 classes (by request type: agentic, chat, batch, interactive)

for each epoch t:
  # (1) Observe arrivals A, service rates mu, and per-request slack
  for each request i:
      slack_i = d_i(t) - r_i(t) / mu_hat_i(t)      # estimated completion margin
      pressure_i = kappa * max(0, -slack_i) / tau    # deadline-pressure weight

  # (2) Closed-form PD split: choose prefill fraction f in {0.1,...,0.9}
  #     minimizing drift-plus-penalty of Q_nc + Z_n + Y_i over the epoch
  f* = argmin_f [ sum_nc Q_nc * (A_nc - mu_nc(f)) + sum_n Z_n * (|B_n| - C_n) + sum_i Y_i * (1[slack_i<0] * r_i/mu_hat_i(f) - 1[slack_i>=0]*mu_hat_i(f)*tau) + V * sum_i pressure_i * r_i / mu_hat_i(f) ]

  # (3) Eviction: for each node, sort resident blocks by marginal drift contribution
  #     delta_b = Q_nc(b) * 1[b needed within T] + Z_n * 1 + Y_i(b) * 1[b needed within T] - pressure_b * reuse_gain_b
  #     evict blocks with smallest delta_b until Z_n projected <= 0
  evict_set = greedy_evict_by_delta(node, T)

  # (4) Migration: move blocks whose delta_b is negative on source and positive on dest
  #     (i.e., moving reduces total drift), subject to migration bandwidth cap B_mig = 25 GB/s
  migrate_set = argmin_{M subset feasible} sum_{b in M} (delta_b(src) - delta_b(dst))

  # (5) Apply f*, evict_set, migrate_set; update Q, Z, Y with realized service
```

**Why this design.** We chose a Lyapunov drift-plus-penalty formulation over a learned or heuristic planner because the drift term gives a per-epoch closed-form objective whose minimizer can be computed by sorting blocks, whereas Evolving Infra's agent-signal heuristic requires the workload to be agentic to produce useful signals; accepting the cost that we must estimate service rates mu_hat per epoch (a lightweight exponential average over the last 5 epochs). We chose per-request-class virtual queues Q_nc over per-request queues because per-request queues would make the state dimension grow with concurrency and make the split decision intractable; accepting the cost that class-level aggregation can mask individual deadline misses within a class, which we mitigate by the per-request pressure weight in the penalty term and by the per-request deadline-violation queue Y_i that directly tracks individual slack. We chose greedy eviction by marginal drift contribution over an optimal knapsack because the drift objective is separable across blocks given the queues, so greedy is optimal for the linearized objective; accepting the cost that the linearization ignores second-order interactions between evictions on the same node, which we bound by capping evictions per epoch at 10% of resident blocks. We chose to keep the PD split, eviction, and migration decisions in one joint objective rather than a two-module controller because the drift term couples them through the shared queues Q_nc, Z_n, and Y_i; accepting the cost that the argmin over f must be evaluated over a discrete grid of 9 values per epoch, which is negligible. A domain expert would not describe Lyra-KV as a variant of Evolving Infra's agent-aware KV manager because the decision rule is derived from the drift inequality rather than from agent context signals, and would not describe it as a variant of standard deadline scheduling (e.g., EDF) because the objective jointly prices memory overflow and deadline backlog through the same virtual-queue state, which EDF does not.

**Why it measures what we claim.** The virtual queue Q_nc(t) measures per-class deadline-backlog because the drift inequality Q_nc(t+1)^2 - Q_nc(t)^2 <= 2 Q_nc(t)(A_nc - mu_nc) + (A_nc - mu_nc)^2 holds for any arrival process, so bounding the drift bounds the backlog regardless of stationarity; this assumption fails when the service-rate estimate mu_hat is biased (e.g., during a sudden model change), in which case Q_nc reflects estimation error rather than true deadline pressure. The per-request deadline-violation queue Y_i(t) measures individual deadline feasibility because it accumulates the required service r_i/mu_hat_i when slack_i < 0 and credits service mu_hat_i*tau when slack_i >= 0, so Y_i > 0 exactly when request i has missed its deadline; this assumption fails when the service-rate estimate mu_hat_i is inaccurate (e.g., due to heavy-tailed decode lengths), in which case Y_i reflects estimation error rather than true violation. The per-request pressure term pressure_i measures deadline urgency because slack_i is computed from the observed residual deadline and estimated service rate, which equals true urgency under the assumption that service rate is stable over the control horizon T; this assumption fails when a request's decode length is heavy-tailed and the estimate is from a short window, in which case pressure_i underweights long requests. The memory-overflow queue Z_n measures node capacity violation because it accumulates realized resident blocks minus capacity, which equals true overflow under the assumption that block sizes are known and fixed; this assumption fails when KV block size varies with sequence length, in which case Z_n reflects a size-weighted approximation. The penalty weight V measures the trade-off between deadline pressure and memory overflow because it multiplies the pressure term in the same objective as the drift terms, which equals the desired trade-off under the assumption that the pressure term is a valid proxy for end-to-end latency; this assumption fails when prefill and decode latency have different sensitivities to pressure, in which case V over- or under-weights one phase.

## Contribution

(1) Lyra-KV, a workload-agnostic SLO-closure controller that replaces agent-signal heuristics with a drift-plus-penalty objective over per-class deadline-backlog and memory-overflow virtual queues, yielding a closed-form per-epoch eviction/migration rule and a discrete PD-split search. (2) A design principle: coupling PD split, eviction, and migration through shared virtual queues removes the need for workload-stationarity assumptions, because the drift bound depends only on realized arrivals and service rates. (3) An open-source implementation of the controller as a drop-in replacement for the KV manager in a decoupled compute/KV serving stack, with a configurable control horizon and pressure gain.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Mixed agentic + chat workload trace, GLM-5.2 / Kimi K2.6, up to 256 GPUs | Spans agentic and non-agentic traffic to test workload-agnosticism |
| Primary metric | SLO attainment rate (fraction of requests meeting deadline) | Directly measures deadline closure, the claimed objective |
| Baseline | Dynamo | Production reference; tests whether drift control beats default scheduling |
| Baseline | EDF-style deadline scheduler | Tests whether joint memory+deadline pricing beats pure deadline ordering |
| Baseline | Evolving Infra agent-aware KV manager | Tests whether workload-agnostic drift control matches agent-signal heuristic |
| Ablation-of-ours | Lyra-KV without migration (split+evict only) | Isolates contribution of the migration decision |
| Ablation-of-ours | Lyra-KV with class granularity = 1 class (aggregate only) | Tests the masking effect of class aggregation |
| Ablation-of-ours | Lyra-KV with class granularity = per-request (no aggregation) | Tests the benefit of per-request queues |
| Ablation-of-ours | Lyra-KV without per-request Y_i queue (class Q_nc only) | Isolates the contribution of the per-request deadline-violation queue |
| Additional experiment | Distribution-shift trace with regime switch at t=50% (agentic to batch) | Directly tests non-stationarity claim |
| Additional experiment | Prefill-heavy request subset | Exposes failure mode F (prefill latency underweighted) |
| Prototype | 8-GPU single-node prototype with synthetic trace | Low-cost validation of core mechanism and overhead measurement |

### Why this setup validates the claim

The central claim is that a single Lyapunov drift-plus-penalty objective over shared deadline-backlog and memory-overflow queues closes SLOs across workloads without needing workload-specific signals. The baseline set is chosen to isolate each sub-claim. Dynamo tests the coarse claim that any deadline-aware controller beats a production default. EDF tests the sharper claim that pricing memory overflow jointly with deadline backlog matters: EDF optimizes deadline order but is blind to KV residency pressure, so it should degrade exactly where memory pressure and deadline pressure conflict. Evolving Infra tests the workload-agnosticism claim: its agent-signal heuristic should be competitive on agentic traffic but should lose signal on chat traffic, whereas Lyra-KV should hold parity or better across both. The no-migration ablation tests whether the migration term in the drift objective is load-bearing or decorative; if removing it changes nothing, the coupling argument is weakened. The class-granularity ablations (1 class, 4 classes, per-request) directly test the masking effect of aggregation: if per-request queues yield no improvement over 4 classes, the aggregation cost is bounded in practice; if they do, the per-request Y_i queue is necessary. The no-Y_i ablation isolates the contribution of the per-request deadline-violation queue. The distribution-shift experiment with a mid-run regime switch directly tests the non-stationarity claim by measuring SLO attainment in the transition window versus steady state. The prefill-heavy subset exposes failure mode F, where prefill latency is underweighted by the decode-slack proxy. The 8-GPU prototype validates the core mechanism at low cost and measures per-epoch wall-clock overhead. SLO attainment rate is the right primary metric because it is the quantity the drift bound directly controls; secondary latency percentiles would conflate queueing effects with the specific mechanism under test. A design that could not distinguish these hypotheses would not falsify the claim, so this combination is the minimal discriminating set.

### Expected outcome and causal chain

**vs. Dynamo** — On a case where a burst of long-decode agentic requests arrives while KV blocks from short chat requests still occupy nodes, Dynamo produces deadline misses because its scheduling is throughput-oriented and does not price residual deadline slack into the prefill/decode split. Our method instead raises the pressure weight on the long-decode requests and shifts the split toward decode while evicting low-reuse chat blocks, because the drift objective couples Q_nc, Z_n, and Y_i in one argmin. We expect a noticeable SLO-attainment gap concentrated in the burst window, with near-parity in steady-state low-load periods.

**vs. EDF-style deadline scheduler** — On a case where many requests share a class deadline but KV residency is near capacity, EDF produces misses because it orders by deadline without accounting for the memory-overflow queue, so it admits work that forces eviction of blocks needed within the horizon. Our method instead evicts by marginal drift contribution and migrates blocks whose delta is negative on source, because Z_n enters the same objective as Q_nc and Y_i. We expect EDF to match us when memory is slack and to fall behind sharply as occupancy approaches capacity, producing a gap that widens with load.

**vs. Evolving Infra agent-aware KV manager** — On a case of non-agentic chat traffic where agent-context signals are absent or uninformative, the Evolving Infra heuristic produces weaker KV decisions because its signal source is degenerate for that workload. Our method instead relies on per-request slack and queue state that exist for any request, because the drift rule needs no workload semantics. We expect parity on agentic subsets and a visible advantage on chat subsets, so the aggregate gain should be driven disproportionately by the non-agentic portion of the trace.

**vs. Lyra-KV without migration** — On a case where a node is memory-tight but has spare migration bandwidth, the no-migration variant cannot move blocks to relieve pressure, so it must evict blocks that would be needed within the horizon, causing deadline misses. Our method migrates those blocks to nodes with spare capacity, because the migration term in the drift objective reduces total delta. We expect the full method to outperform the no-migration variant primarily in memory-tight, bandwidth-available regimes.

**vs. Lyra-KV with class granularity = 1 class** — On a case where requests within the single class have heterogeneous deadlines (e.g., a mix of short chat and long batch), the aggregate queue Q_nc cannot distinguish urgent from non-urgent requests, so the controller may under-serve urgent requests. Our method with per-request Y_i queues tracks individual slack and prioritizes urgent requests, because Y_i directly penalizes deadline violations. We expect the 1-class variant to have lower SLO attainment, especially on heterogeneous traces, and the gap to shrink as granularity increases to 4 classes and per-request.

**vs. Lyra-KV with class granularity = per-request** — On a case with high concurrency, per-request queues increase state dimension and may make the split decision noisier, but they eliminate masking. We expect per-request granularity to yield the highest SLO attainment, with diminishing returns beyond 4 classes, and to incur higher per-epoch wall-clock overhead.

**vs. Lyra-KV without per-request Y_i queue** — On a case where class-level aggregation masks individual deadline misses (e.g., one request in a class is urgent but others are not), the variant without Y_i relies only on the soft pressure term, which may be outvoted by aggregate drift. Our method with Y_i directly penalizes individual violations, because Y_i accumulates required service when slack_i < 0. We expect the full method to outperform the no-Y_i variant on traces with high within-class deadline variance.

**Distribution-shift experiment** — On a trace that switches from agentic to batch workload at t=50%, the Evolving Infra heuristic loses signal after the switch, while Lyra-KV maintains SLO attainment because its queues depend only on realized arrivals and service rates. We expect Lyra-KV to show a smaller drop in SLO attainment in the transition window (t=50% to t=60%) compared to baselines, and to recover faster to steady-state performance.

**Prefill-heavy subset** — On a subset of requests with long prefill and short decode, the decode-slack proxy pressure_i may underweight prefill latency, causing deadline misses. We expect Lyra-KV to show lower SLO attainment on this subset compared to decode-heavy subsets, exposing failure mode F, and we will report the gap to quantify the limitation.

**Prototype overhead** — On an 8-GPU single-node prototype with a synthetic trace, we measure the per-epoch wall-clock overhead of the controller (observation, argmin over f, eviction sorting, migration selection). We expect the overhead to be less than 5% of the epoch duration (tau = 20 ms), validating feasibility.

### What would falsify this idea

If the SLO-attainment gain over EDF is uniform across memory-slack and memory-tight regimes rather than concentrated where occupancy approaches capacity, the joint-queue coupling claim is wrong. If the no-migration ablation matches full Lyra-KV on every subset, the migration term is not load-bearing and the coupling argument collapses. If the per-request Y_i queue yields no improvement over class-level Q_nc alone, the per-request feasibility claim is unsupported.

## References

1. Evolving Infra: Agent-Adaptive LLM Serving at Scale
