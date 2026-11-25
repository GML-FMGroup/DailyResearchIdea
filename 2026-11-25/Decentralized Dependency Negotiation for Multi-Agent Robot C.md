# Decentralized Dependency Negotiation for Multi-Agent Robot Control with Liveness Verification

## Motivation

ACAP (E4) resolves physical-dependency conflicts among member agents via a single LLM hyper-agent whose coordination policy is unverified, so liveness (deadlock-freedom) and bounded task time rest on an opaque centralized controller rather than on any checkable protocol invariant. This is the local instance of a field-level pattern: decentralized multi-agent control has formal learning-complexity characterizations (E1) and scalable planners (E2), yet deployed LLM-agent robot teams (E3, E4) still assume a centralized coordinator can resolve inter-agent coupling without a verified distributed protocol. The structural root cause is that ACAP's dependency resolution is a runtime decision of one agent, not a property of a protocol, so no invariant exists that could be verified or that would hold if the coordinator were removed.

## Key Insight

Physical-dependency conflicts in a robot team can be encoded as a wait-for graph over control groups whose acyclicity is a protocol-level invariant maintained by local, message-passing negotiation, so deadlock-freedom becomes a checkable property of the protocol rather than an emergent hope in a centralized coordinator.

## Method

**What it is.** DDNP (Decentralized Dependency Negotiation Protocol) is a distributed coordination layer that replaces ACAP's hyper-agent. Its inputs are (i) a set of member agents, each owning a configurable control group (e.g., arm+gripper, torso+chassis) with a declared set of physical resources, (ii) each member's pending task with its resource demand vector, (iii) a synchronous-round message channel among members, and (iv) a declared continuous coupling model: for each ordered pair of control groups (m, j), a kinematic interference predicate `interfere(m, j, q_m, q_j)` that returns true when m's planned motion from its current configuration q_m is physically blocked by j's held configuration q_j. The predicate is computed from the robot URDF (as parsed in E3's Robot Resume) using forward kinematics to obtain each group's reachable workspace as a set of swept volumes over the next control step, and returns true when the swept volumes intersect or when the groups share a rigid kinematic chain (e.g., torso and arm on the same platform). Its outputs are, per round, a set of granted resource locks, a set of deferred tasks, and a wait-for graph whose acyclicity certifies deadlock-freedom over both logical and physical dependencies.

**How it works.** Each member maintains a local wait-for graph restricted to its own resource requests and the requests it has observed. A round proceeds as:

```pseudocode
# State per member m: held locks H_m, held configuration q_m, pending request R_m (resource set, priority p_m, timestamp t_m)
# Global invariant target: wait-for graph G is acyclic at every round boundary
# Edge types: logical edge m->j if R_m ∩ H_j != empty; physical edge m->j if interfere(m, j, q_m, q_j)

for each round r = 1..R_max:              # R_max = 8 rounds default
    # Phase 1: broadcast intent
    m broadcasts (R_m, p_m, t_m, q_m) to all members
    # Phase 2: local graph update
    for each received (R_j, p_j, t_j, q_j):
        add logical edge m -> j to local G_m if R_m ∩ H_j != empty
        add physical edge m -> j to local G_m if interfere(m, j, q_m, q_j)
    # Phase 3: priority-based yielding (deadlock avoidance)
    if exists cycle in G_m containing m:
        let j* = member in cycle with lowest (p_j, t_j)   # lexicographic: priority then timestamp
        if m == j*:  m releases all locks in H_m, defers R_m, increments t_m
        else:        m keeps waiting (no release)
    # Phase 4: grant
    for each resource x in R_m:
        if no other member holds x and G_m has no incoming edge to m on x:
            H_m <- H_m ∪ {x}
    # Phase 5: verify
    m checks acyclicity of G_m via DFS; if cyclic after R_max, m escalates to
    a fallback serialization: all members release, resources granted in (p, t) order
```

Priority p_m is set to the number of downstream members blocked on m's control group (computed from the declared dependency DAG, a static input). Timestamp t_m is a Lamport counter incremented on every release. The protocol's liveness guarantee is stated as an explicit precondition: acyclicity of G certifies deadlock-freedom under the assumption that the declared continuous coupling model captures all physical interference between control groups; for platforms whose control groups are provably dynamically decoupled (no shared kinematic chain and non-intersecting swept volumes over the task horizon), this precondition holds by construction, and for platforms with unmodeled coupling the protocol's guarantee is restricted to logical resource deadlock and the physical-edge derivation is the mechanism that extends it.

**Why this design.** We chose a wait-for-graph protocol with priority-timestamp yielding over ACAP's single hyper-agent (E4) because the hyper-agent's dependency resolution is a runtime LLM decision with no invariant, so deadlock-freedom cannot be checked; the cost is that DDNP cannot exploit semantic task knowledge the LLM hyper-agent uses to reorder tasks opportunistically, so we accept potentially longer makespan on tasks where semantic reordering matters. We chose priority = downstream-blocked-count over fixed priority because fixed priority can starve a member whose control group gates many others, accepting the cost that priority must be recomputed when the static dependency DAG changes. We chose a bounded round count (R_max=8) with a serialization fallback over unbounded negotiation because unbounded rounds make worst-case time cost unbounded, accepting the cost that the fallback serializes work that could have proceeded in parallel. We chose to derive physical edges from a URDF-based kinematic interference predicate rather than from declared resource sets alone because declared locks are not the sole channel of physical coupling on rigidly coupled platforms, accepting the cost that the predicate must be recomputed each round from current configurations. A domain expert would not describe DDNP as a variant of ACAP: ACAP's mechanism is an LLM hyper-agent assigning tasks, whereas DDNP's mechanism is a graph-acyclicity invariant maintained by local message passing, with no central decision-maker and no LLM in the coordination loop.

**Why it measures what we claim.** The wait-for graph G measures physical-dependency conflict because an edge m→j is added when m's requested resource set intersects j's held set (logical edge) or when m's planned motion is physically blocked by j's held configuration under the declared kinematic interference predicate (physical edge), which under the assumption that the predicate captures all inter-group physical coupling is equivalent to a real dependency; this assumption fails when two control groups are coupled through unmodeled dynamics not represented in the URDF-derived swept volumes (e.g., a torso motion that perturbs an arm's grasp through compliance or payload shift), in which case G reflects declared logical and kinematic conflicts but misses that residual coupling, and deadlock incidence would be underestimated. The round count to acyclic G measures negotiation efficiency because acyclicity is reached exactly when no member waits on a lower-priority holder, under the assumption that priority-timestamp yielding terminates; this assumption fails when two members share equal priority and equal timestamps (possible if timestamps are not strictly incremented on release), in which case the round count reflects tie-breaking retries rather than genuine conflict resolution. Task-balanced time cost measures the end-to-end effect of coordination because it aggregates per-task wall-clock over rollouts, under the assumption that rollouts are comparable across platforms; this assumption fails when platform-specific actuation latency dominates, in which case the metric reflects platform speed rather than coordination quality.

## Contribution

['A decentralized dependency-negotiation protocol (DDNP) for multi-agent robot control that replaces a centralized LLM coordinator with local wait-for-graph maintenance and priority-timestamp yielding, making deadlock-freedom a checkable protocol invariant.', "A controlled empirical comparison isolating whether liveness in multi-agent robot control requires decentralization or merely verification, by pairing DDNP with a verified centralized variant of ACAP's hyper-agent protocol.", 'An open protocol specification and instrumentation for deadlock-incidence and time-cost measurement across parallel, parallel-sequential, and collaborative task settings.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | 25 task/platform rollouts from ACAP benchmark (E4) | Matches source setting for direct comparison |
| Dataset | Habitat-MAS benchmark (E3): 100 conflict-injected episodes across the four task categories (manipulation, perception, navigation, multi-floor rearrangement) | Tests liveness at benchmark scale on a public heterogeneous multi-robot platform |
| Primary metric | Task-balanced time cost | Aggregates coordination effect across heterogeneous tasks |
| Secondary metric | Deadlock-incidence rate with 95% confidence intervals (Wilson interval) over the 100 conflict-injected episodes | Directly measures the liveness claim rather than inferring it from time cost |
| Secondary metric | Rounds-to-acyclic-G (mean and max over episodes) | Measures negotiation efficiency of the protocol |
| Baseline | Single-agent Direct Policy | Tests whether multi-agent coordination itself helps |
| Baseline | Fixed Delegation | Tests whether dynamic yielding beats static assignment |
| Baseline | ACAP hyper-agent | Tests whether graph invariant beats LLM reordering |
| Baseline | Classical wound-wait deadlock avoidance over declared resource locks only (no physical edges) | Tests whether the physical-edge derivation is what closes the liveness gap relative to classical protocols |
| Ablation-of-ours | DDNP with fixed priority | Tests whether downstream-blocked priority matters |
| Ablation-of-ours | DDNP with logical edges only (physical-edge derivation disabled) | Tests whether the kinematic interference predicate is doing the work of covering physical coupling |
| Ablation-of-ours | DDNP with unmodeled-coupling injection (torso motion perturbing arm grasp, injected at 10% of episodes) | Quantifies the deadlock-underestimation rate when the declared coupling model is incomplete |

### Why this setup validates the claim

The central claim is that replacing ACAP's LLM hyper-agent with a wait-for-graph negotiation protocol yields deadlock-free coordination without sacrificing end-to-end time cost, and that the gain comes specifically from dependency-aware yielding over both logical and physical edges. The Single-agent Direct Policy baseline tests the weak sub-claim that any multi-agent decomposition helps at all; if DDNP cannot beat a single agent on parallelizable tasks, the coordination layer adds no value. Fixed Delegation tests whether the dynamic yielding mechanism matters relative to a static assignment that cannot resolve runtime conflicts; this is the closest non-LLM coordination baseline. ACAP's hyper-agent is the direct incumbent and tests whether a maintained acyclicity invariant is competitive with LLM-based semantic reordering. The classical wound-wait baseline over declared locks only isolates the contribution of the physical-edge derivation: if DDNP matches wound-wait on deadlock incidence, the kinematic interference predicate is not what closes the gap that classical protocols leave open. The logical-edges-only ablation isolates the same mechanism from the other direction, and the unmodeled-coupling injection quantifies the residual failure mode the authors themselves flag. The fixed-priority ablation isolates the downstream-blocked-count priority design: if priority choice is irrelevant, the ablation should match DDNP. Task-balanced time cost is the right primary metric because it is exactly what ACAP optimizes and what the coordination mechanism is supposed to improve; deadlock-incidence rate with confidence intervals is the right secondary metric because a time-cost-only comparison cannot show whether the protocol trades correctness for speed, and the 100-episode conflict-injected set gives enough samples to bound the incidence rate.

### Expected outcome and causal chain

**vs. Single-agent Direct Policy** — On a case where a torso-and-chassis control group must reposition while an arm-and-gripper group independently prepares a grasp, the single-agent policy serializes both because one policy must sequence all actions. Our method instead lets both members hold disjoint resource locks and proceed concurrently, with the wait-for graph certifying no conflict, so we expect a clear time-cost gap on Fully Parallel and Parallel-Sequential tasks and near-parity on tightly coupled Collaborative tasks where serialization is unavoidable.

**vs. Fixed Delegation** — On a case where a member gated by many downstream dependents requests a resource currently held by a low-priority member, Fixed Delegation has no mechanism to preempt the holder, so the gated member stalls until the holder finishes. Our method instead assigns priority from downstream-blocked count, so the low-priority holder yields and the gated subtree unblocks, so we expect lower mean time cost concentrated on Collaborative settings where dependency depth is high, with parity where all members are independent.

**vs. ACAP hyper-agent** — On a case where a runtime dependency emerges that the static DAG did not anticipate, the hyper-agent may reorder tasks semantically but offers no deadlock-freedom guarantee, so under adversarial interleavings it can stall or require an ad hoc reset. Our method instead maintains an acyclic wait-for graph by construction and falls back to priority-timestamp serialization after bounded rounds, so we expect comparable or slightly worse makespan on semantically reorderable tasks but strictly fewer stalls and bounded worst-case time on conflict-heavy tasks; on the 100 conflict-injected Habitat-MAS episodes we expect a deadlock-incidence rate at or near zero with a 95% Wilson upper bound below the hyper-agent's point estimate.

**vs. Classical wound-wait over declared locks only** — On a case where a torso motion physically blocks an arm's planned reach even though the two groups hold disjoint declared resource locks, wound-wait sees no conflict and grants both, leading to a physical deadlock or self-collision. Our method instead adds a physical edge from the kinematic interference predicate, so the cycle is detected and resolved by priority-timestamp yielding, so we expect a lower deadlock-incidence rate than wound-wait concentrated on episodes where physical coupling is present, with parity on episodes where all conflicts are logical.

### What would falsify this idea

If DDNP's time-cost advantage over Fixed Delegation and the hyper-agent is uniform across all task subsets rather than concentrated on the high-dependency-depth Collaborative and Parallel-Sequential settings the mechanism predicts, then the gain is not coming from dependency-aware yielding and the central claim is wrong. Likewise, if the fixed-priority ablation matches DDNP everywhere, the downstream-blocked priority design is not doing the work we claim; and if the logical-edges-only ablation matches full DDNP on deadlock incidence in the unmodeled-coupling-injected episodes, the physical-edge derivation is not covering the coupling the claim attributes to it.

## References

1. ACAP: Asynchronous Coding Agents for Efficient Robot Control
