# Lock-Ordered Control Allocation for Asynchronous Multi-Agent Robot Teams

## Motivation

ACAP's hyper-agent dynamically reallocates control between persistent member agents during execution, but its allocation policy is a heuristic with no liveness guarantee: under asynchronous member reasoning, two agents can each hold a control group and request the other's, producing a circular wait that stalls the team. EMOS inherits the same gap at the planning layer, where a static hierarchical planner assigns embodiments without any protocol ensuring that concurrent task execution cannot deadlock on shared physical resources. Both branches treat coordination as an LLM planning problem rather than a resource-allocation protocol problem, so neither provides the structural property — an acyclic wait-for relation over physical dependencies — that makes concurrent execution safe.

## Key Insight

If every physical dependency between persistent agents is represented as a lockable resource and every control acquisition follows a global total order derived from the robot's kinematic tree, then the wait-for graph of the asynchronous team is acyclic by construction, so ACAP's concurrent execution can proceed without deadlock while retaining its time-cost advantage.

## Method

**What it is.** LOCA (Lock-Ordered Control Allocation) is a coordination layer that sits between ACAP's hyper-agent and its persistent member agents. Its inputs are (i) the robot's kinematic model, from which it derives a dependency DAG of control groups, (ii) the hyper-agent's task assignments, each annotated with the set of control groups it needs, and (iii) member agents' acquire/release requests. Its outputs are an ordered acquisition plan per task and a grant/queue decision for each request, such that no two agents can enter a circular wait.

**Load-bearing assumption (stated explicitly).** LOCA's acyclicity guarantee holds only if every physical dependency that can cause a circular wait between persistent agents is captured as a lockable resource node in the URDF-derived kinematic DAG, so a static total order over that DAG is sufficient to make the runtime wait-for graph acyclic. This assumption fails when a dependency is not a joint (e.g., shared power bus, thermal limit, cable/range constraint) or when the hyper-agent grants an ad-hoc control group outside `plan[t]`. To enforce the assumption at the allocation boundary, the hyper-agent's grant interface is the sole entry point for acquiring any control group: any ad-hoc grant whose rank is below a resource the requester already holds is rejected (or serialized until the requester releases the higher-ranked resource), and non-kinematic couplings that are known for the platform (power bus, range limit) are added as explicit nodes in the DAG before rank computation. Out-of-plan grants are therefore rank-checked acquisitions, not an escape hatch.

**How it works.**

```pseudocode
# Phase 1: build resource order (offline, once per platform)
D = kinematic_dependency_dag(urdf)          # edge u->v if v is a child of u
D = D + non_kinematic_couplings(platform)   # e.g., shared power bus, range limit as nodes
R = {control groups} = nodes(D)
rank[r] = topological_order(D)[r]           # total order, parents before children

# Phase 2: hyper-agent task assignment (per task)
for task t assigned to member m:
    need[t] = required_control_groups(t)     # from hyper-agent plan
    plan[t] = sort(need[t], key=lambda r: rank[r])   # ascending rank

# Phase 3: runtime acquisition (per member, asynchronous)
def acquire(m, t):
    held = []
    for r in plan[t]:
        while not try_lock(r):
            if would_violate_order(m, r):    # m holds s with rank[s] > rank[r]
                release_all(held); retry_from_scratch(t)
            else:
                wait(r)                      # queued, not spinning
        held.append(r)
    return held

def release(m, held):
    for r in reversed(held): unlock(r)

# Phase 4: hyper-agent reallocation under contention
def reallocate(m, r, timeout=tau):
    if wait_time(m, r) > tau:
        preempt_lowest_rank_holder(r)        # only if holder rank > requester rank
        reschedule(m, plan[t])               # re-derive order, never break rank

# Phase 5: grant interface (sole entry point for any control group)
def grant(m, r):
    if rank[r] < max(rank[s] for s in held[m]):
        reject_or_serialize(m, r)            # ad-hoc grant below a held resource
    else:
        try_lock(r)
```

Hyperparameters: `tau` (contention timeout before hyper-agent reallocation, set to 2x the median member reasoning latency measured during a warm-up rollout); `rank` is fixed by the kinematic DAG (augmented with non-kinematic couplings) and never re-ordered at runtime. The grant interface in Phase 5 enforces that any out-of-plan acquisition is rank-checked against the requester's currently held resources.

**Why this design.** We chose a static total order derived from the kinematic DAG over dynamic priority inheritance because the dependency relation between control groups is fixed by the robot's physical structure, so a runtime priority scheme would introduce unnecessary reordering and break the acyclicity argument; the cost is that tasks whose natural execution order conflicts with the kinematic order must buffer or release-and-retry, adding latency in those cases. We chose release-and-retry over blocking wait for order violations because blocking on a lower-ranked resource while holding a higher-ranked one is exactly the circular-wait condition; the cost is wasted work when a member must restart its acquisition sequence. We chose a hyper-agent-triggered preemption timeout over pure wait-free progress because wait-free progress would require every member to be able to complete without any other member's cooperation, which is false for tightly coupled manipulation; the timeout gives the hyper-agent a bounded intervention point while the rank order guarantees the intervention itself cannot create a cycle. This differs from ACAP's heuristic dynamic allocation, which has no ordering discipline, and from EMOS's static hierarchical planner, which assigns embodiments once and cannot react to runtime contention.

**Why it measures what we claim.** The wait-for graph `W` built from `held` and `wait(r)` edges measures deadlock-freedom because the rank order makes every edge in `W` point from a lower-ranked to a higher-ranked resource, so `W` is acyclic; this assumption fails when a member can acquire a resource outside its `plan[t]` (e.g., the hyper-agent grants an ad-hoc control group mid-task), in which case `W` may contain a back-edge and the metric reflects only the planned acquisitions. The quantity `wait_time(m, r)` measures contention pressure because it counts time a member is queued rather than reasoning or acting; this assumption fails when a member is slow for reasons unrelated to locking (long LLM reasoning), in which case the timeout fires spuriously and the metric reflects reasoning latency rather than resource contention. The quantity `rank[r]` measures physical dependency depth because it is the topological position of the control group in the URDF-derived DAG (augmented with non-kinematic couplings); this assumption fails when the URDF omits a physical coupling (e.g., a shared power bus not modeled as a joint), in which case the order does not capture the true dependency and the acyclicity claim is scoped to the modeled DAG only.

## Contribution

["A lock-ordered control-allocation protocol (LOCA) that converts ACAP's heuristic hyper-agent reallocation into a resource-acquisition discipline over a kinematic-dependency total order, giving the asynchronous member team an acyclic wait-for relation at runtime.", 'A hyper-agent reallocation rule with a contention timeout that preserves the rank order under preemption, so dynamic control transfer between lead and member agents cannot introduce a circular wait.', 'An empirical demonstration on real robot rollouts that the 28.2% time-cost reduction of ACAP is retained under LOCA while eliminating deadlock-induced stalls observed in the unconstrained ACAP baseline.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | 25 ACAP task/platform settings, 72 rollouts | Matches prior real-world multi-group control benchmark |
| Primary metric | Task-balanced time cost | Directly reflects coordination overhead and wasted work |
| Baseline | Single-agent Direct Policy | Tests whether parallelism gain survives ordering discipline |
| Baseline | Fixed Delegation | Tests whether runtime contention handling beats static assignment |
| Baseline | ACAP dynamic allocation (no ordering) | Tests whether lock ordering removes deadlock/starvation |
| Baseline | Priority inheritance on the same kinematic DAG | Tests whether static rank beats dynamic priority inheritance |
| Baseline | Wait-die on the same kinematic DAG | Tests whether static rank beats timestamp-based deadlock avoidance |
| Baseline | Wound-wait on the same kinematic DAG | Tests whether static rank beats preemptive timestamp scheme |
| Ablation-of-ours | LOCA with dynamic priority instead of static rank | Tests whether kinematic static order is the causal ingredient |
| Diagnostic | Wait-for graph instrumentation on ACAP baseline | Measures empirical cycle rate and stall duration |
| Simulator | Discrete-event lock simulator driven by logged ACAP latencies | Reproduces contention at scale cheaply |

### Why this setup validates the claim

The central claim is that a static total order derived from the kinematic DAG eliminates circular wait while preserving ACAP's concurrency gains on tightly coupled manipulation. The Single-agent Direct Policy baseline tests whether any parallelism gain survives once ordering is imposed — if LOCA cannot beat a serial policy on task-balanced time cost, the ordering discipline is pure overhead. The Fixed Delegation baseline tests the sub-claim that runtime contention handling matters: it assigns control groups once and cannot react, so any LOCA advantage concentrated in Collaborative settings isolates the value of dynamic reallocation under a fixed order. The un-ordered ACAP baseline is the sharpest test: it has the same parallelism as LOCA but no ordering, so any deadlock, starvation, or release-and-retry thrash it exhibits is attributable to the missing discipline. The three textbook deadlock-avoidance baselines (priority inheritance, wait-die, wound-wait) instantiated on the same kinematic DAG isolate what is genuinely new beyond textbook lock ordering: priority inheritance does not prevent cycles, wait-die and wound-wait prevent cycles but may abort or preempt differently, and comparing their time-cost and stall distributions to LOCA shows whether the kinematically derived static rank is the causal ingredient. The ablation replacing static rank with dynamic priority tests whether the causal ingredient is specifically the kinematically derived static order rather than ordering per se. Task-balanced time cost is the right metric because deadlocks and order-violation retries manifest as queued or wasted time, not as wrong final task outcomes; a metric that only scored task success would be blind to the predicted failure mode. The diagnostic wait-for graph instrumentation on the ACAP baseline grounds the motivation in observed cycles rather than a hypothesized failure mode: we log every wait edge and report the empirical cycle rate, mean cycle duration, and correlation with time-cost variance across the 72 rollouts. The discrete-event lock simulator, driven by logged ACAP reasoning latencies, reproduces the contention scenarios at scale so that deadlock/starvation statistics can be gathered cheaply and the real-robot rollouts are reserved for validating the simulator's fidelity (we require simulator cycle rate within 20% of real-robot cycle rate on the 72 rollouts).

### Expected outcome and causal chain

**vs. Single-agent Direct Policy** — On a Collaborative setting where one arm must stabilize a chassis-held object while another manipulates it, the Direct Policy serializes all reasoning and action through one agent, so the stabilization and manipulation phases cannot overlap and total time is roughly additive across phases. Our method splits these into separate member agents whose control groups are ordered by the kinematic DAG (chassis before arm), letting independent reasoning proceed concurrently while the order prevents the arm from grabbing a resource the chassis still needs. We expect a clear task-balanced time-cost reduction on Collaborative and Fully Parallel settings, with near-parity on Parallel-Sequential settings where there is little genuine concurrency to exploit.

**vs. Fixed Delegation** — On a case where a member's assigned control group becomes a bottleneck mid-task (e.g., the torso agent is idle while an arm agent queues for a torso-held resource), Fixed Delegation has already committed control groups and cannot reassign, so the arm waits until the torso's original task finishes, producing a long stalled interval. Our method's hyper-agent reallocation fires once wait time exceeds tau and preempts the lowest-ranked holder only when its rank exceeds the requester's, so the blocked arm is unblocked without breaking rank order. We expect the gain to be concentrated in the eight Collaborative settings, with a smaller or absent gap where no mid-task contention arises.

**vs. ACAP dynamic allocation (no ordering)** — On a case where two members each hold one control group and each request the other's (a classic two-resource cycle, e.g., arm-holds-gripper-wants-torso and torso-holds-chassis-wants-gripper), ACAP's heuristic dynamic allocation has no ordering discipline, so the two members can each wait on the other indefinitely or thrash through repeated preemption, inflating queued time. Our method's static rank makes every wait edge point from lower to higher rank, so this cycle cannot form; the lower-ranked member simply releases and retries in ascending rank order. We expect ACAP to show occasional long-tail stalls and higher variance in time cost on contention-heavy rollouts, while LOCA shows a tighter distribution and a lower mean concentrated exactly on those contention-heavy instances. The diagnostic wait-for graph instrumentation will report the empirical cycle rate on ACAP; if that rate is near zero, the contention-heavy instances may not exist in the 72 rollouts and the comparison reduces to overhead measurement.

**vs. Priority inheritance on the same kinematic DAG** — Priority inheritance raises the priority of a lock holder to that of the highest-priority waiter, but it does not impose a global order, so two members can still form a cycle if each holds a resource the other needs and priorities are symmetric. We expect priority inheritance to reduce starvation relative to un-ordered ACAP but to exhibit residual cycles on the two-resource-cycle case, giving higher time-cost variance than LOCA. LOCA's static rank prevents the cycle by construction, so its advantage should be concentrated on the same contention-heavy instances where priority inheritance still cycles.

**vs. Wait-die on the same kinematic DAG** — Wait-die allows an older transaction to wait and aborts a younger one that requests a held resource, preventing cycles by timestamp order. On ACAP-style asynchronous members, aborting a younger member means discarding its partial reasoning and restarting, which adds wasted work. We expect wait-die to avoid deadlock but to incur higher task-balanced time cost than LOCA on Collaborative settings because of restart overhead, while LOCA's release-and-retry only restarts acquisition, not reasoning. The gap should widen as the number of contended resources per task increases.

**vs. Wound-wait on the same kinematic DAG** — Wound-wait preempts the holder if the requester is older, otherwise the requester waits, also preventing cycles by timestamp order. It avoids some restarts relative to wait-die but can preempt a holder mid-manipulation, causing physical disruption. We expect wound-wait to show lower restart cost than wait-die but higher stall duration than LOCA on tightly coupled manipulation, because LOCA's preemption is rank-checked and only fires after tau, preserving the rank invariant. LOCA should therefore have a lower mean and tighter distribution of time cost on the eight Collaborative settings.

### What would falsify this idea

If LOCA's advantage over un-ordered ACAP is uniform across all task subsets rather than concentrated on contention-heavy Collaborative and Fully Parallel instances, then the static rank order is not the causal ingredient and the deadlock-freedom story is wrong. Likewise, if the dynamic-priority ablation matches LOCA's time cost and stall distribution, the specific kinematic static order — not ordering in general — fails to explain the result. Additionally, if the diagnostic instrumentation shows zero cycles on the ACAP baseline across all 72 rollouts, the motivation is unsupported and the comparison reduces to overhead measurement.

## References

1. ACAP: Asynchronous Coding Agents for Efficient Robot Control
