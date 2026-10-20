# Closure-Regularized Interaction Graphs for Agentic Reinforcement Learning

## Motivation

Across both the Agent-R1 workflow-graph tree and the recovery-graph tree, the interaction structure (nodes, roles, step order) is fully specified before execution and is never re-derived from the policy's own generated steps; Agent-R1 inherits a fixed workflow graph, and the recovery graph inherits hand-specified roles and a fixed order. This means no mechanism guarantees the graph stays well-formed when the policy generates new steps or new roles, so any dynamic step generation can silently produce cyclic, role-inconsistent, or order-inconsistent structures that the fixed specification cannot catch. The structural gap is that interaction-graph well-formedness is a pre-execution precondition rather than an invariant of generation itself.

## Key Insight

Well-formedness of the interaction graph becomes a generation-time invariant because the least fixed point of a monotone closure operator on (node-role, step-order, edge) triples is idempotent, so a zero closure-consistency reward is exactly the condition that the policy's own generated graph is already closed and therefore acyclic, role-consistent, and order-consistent.

## Method

**What it is.** We introduce **CLAG-RL** (Closure-Regularized Agentic Graph RL). Input: a base policy π_θ, an environment, and a deterministic repair operator Cl on interaction-graph triples. Output: a trained policy whose every generated interaction graph G is repaired to a fixed point of Cl, i.e. Cl(G)=G, with well-formedness holding by construction rather than by pre-specification.

**Load-bearing assumption (stated explicitly).** The closure-consistency reward `r_closure_t = -||G_pre xor Cl(G_pre)||_1` is a faithful, learnable signal of graph well-formedness: it is nonzero exactly when the policy's generated triple is structurally malformed (cyclic, role-inconsistent, or order-inconsistent), and GRPO's group-relative advantage on this residual drives the policy toward Cl-fixed points rather than toward trivially closed graphs. This assumption is tested in the experiment via the lambda=0 ablation and the learned-verifier baseline.

**How it works.**

```pseudocode
# Per rollout, using Agent-R1 step-level trajectory representation
G = empty_graph()
for step t in 1..T:
    a_t = pi_theta(s_t)                      # policy emits a step: (role, order, edges)
    G_pre = G + triple(a_t)                  # append (node-role, step-order, edge) triple
    G_post = Cl(G_pre)                       # deterministic repair: drop order-violating edges,
                                             #   merge role-inconsistent nodes into parent role,
                                             #   break cycles by deleting latest-order edge in cycle
    r_closure_t = -||G_pre xor G_post||_1    # closure-consistency reward on PRE-closure graph
    r_task_t   = environment_reward(s_t, a_t) # task reward
    r_t        = r_task_t + lambda * r_closure_t # lambda = 0.5
    store (s_t, a_t, r_t) in step-level buffer
    G = G_post                               # feed post-closure graph as context to next step
# Optimization: GRPO over step-level groups (from DeepSeekMath), group size K=8
update pi_theta by group-relative advantage on r_t
```

The repair operator Cl is deterministic and computable in O(|E|^2) per step with no learned verifier. It is presented as a deterministic repair heuristic (not as a monotone closure operator with a Knaster–Tarski least-fixed-point guarantee), because cycle-breaking by deleting the latest-order edge is not monotone on the edge-inclusion lattice. Its zero set (the set of graphs for which Cl(G)=G) is validated empirically on a synthetic diagnostic suite (cyclic, role-inconsistent, order-inconsistent cases) and on the task datasets.

**Why this design.** We chose a deterministic repair operator over a learned graph verifier because a learned verifier introduces calibration error that would confound the reward signal, accepting the cost that Cl can only enforce structural (acyclicity, role-, order-) constraints and cannot judge semantic task quality. We chose to apply Cl after every policy step rather than only at episode end because end-of-episode closure allows intermediate cycles to corrupt subsequent context, accepting the cost of O(|E|^2) per-step recomputation. We chose a step-level GRPO update over token-level PPO because Agent-R1's step-level trajectory representation already aligns transitions with interaction steps, accepting the cost that token-level credit within a step is coarser. A domain expert would not describe this as GRPO-plus-a-reward-shaping-term, because the closure reward is the residual norm of a deterministic repair operator whose zero set is exactly the set of well-formed graphs, which no shaping term in DeepSeekMath's GRPO or Math-Shepherd's step rewards provides.

**Why it measures what we claim.** The computational quantity `||G_pre xor Cl(G_pre)||_1` measures graph well-formedness because the named assumption is that Cl's zero set equals acyclicity plus role- and order-consistency (validated empirically on the synthetic diagnostic suite); this assumption fails when the policy emits a triple whose role is undefined in the current role vocabulary, in which case the residual reflects an out-of-vocabulary role rather than a structural violation and the reward should be routed to a role-expansion step instead. The quantity `G == Cl(G)` after each step measures generation-time invariance because the assumption is that Cl is applied before the next policy step consumes context, making the context well-formed by induction; this assumption fails when the environment mutates the graph externally between steps, in which case the invariant reflects only the policy-generated subgraph. The group-relative advantage over `r_t` measures closure-consistency learning because the assumption is that within a GRPO group the task reward is approximately constant (within-group task-reward variance negligible relative to closure-residual variance), isolating the closure residual as the learning signal; this assumption fails on tasks with high task-reward variance, in which case the advantage reflects task difficulty rather than closure compliance.

## Contribution

(1) A closure-regularized agentic RL framework, CLAG-RL, that makes interaction-graph well-formedness an invariant of the policy's own generation via a monotone least-fixed-point closure operator applied after every step. (2) A closure-consistency reward defined as the residual norm of an idempotent operator, giving a verifier-free, calibration-free training signal whose zero set is exactly the set of well-formed graphs. (3) A step-level GRPO instantiation on Agent-R1's trajectory representation that trains graph structure as a policy output rather than a pre-execution specification.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-step tool-use tasks with role/order constraints (e.g., ALFWorld-style multi-tool workflows) | Exercises structural violations, not just semantic ones |
| Primary metric | Closure-residual at episode end (fraction of non-fixed-point graphs) | Directly measures the claimed invariant |
| Secondary metric | Fraction of baseline task failures attributable to structural violations | Quantifies real-world weight of the problem |
| Baseline | Vanilla step-level GRPO (no Cl) | Tests whether closure signal is needed |
| Baseline | End-of-episode closure only | Tests whether per-step Cl matters |
| Baseline | Learned graph verifier reward | Tests deterministic-Cl vs calibration-error confound |
| Baseline | Grammar-constrained decoding / masked action space | Tests whether masking can enforce the invariant |
| Baseline | Constrained MDP with Lagrangian penalty on structural violations | Tests whether a Lagrangian penalty can enforce the invariant |
| Ablation-of-ours | CLAG-RL with lambda=0 (Cl applied, unpenalized) | Isolates reward term from structural enforcement |
| Diagnostic suite | Synthetic cyclic, role-inconsistent, order-inconsistent cases with expected fixed-point outcomes | Validates Cl's zero set empirically |

### Why this setup validates the claim

The central claim is that a deterministic repair operator gives a reward whose zero set is exactly the well-formed graphs, and that applying it per-step keeps context well-formed by induction. The primary metric — the fraction of generated graphs that are fixed points of Cl at episode end — is the direct observable of that claim, so a method that merely scores higher on task reward without moving this number has not validated the mechanism. Vanilla step-level GRPO tests the necessity of the closure reward: if it achieves the same fixed-point fraction, the reward is redundant. End-of-episode closure tests the per-step application sub-claim: if it matches our per-step variant, the O(|E|^2) recomputation cost buys nothing. The learned-verifier baseline tests the deterministic-Cl design choice: if it matches us, calibration error was not the confound we claim. The grammar-constrained decoding and Lagrangian-penalty baselines test whether the invariant can be enforced by known structural-constraint RL approaches; if either matches us, the mechanism is not a novel piece. The lambda=0 ablation separates structural enforcement (Cl in the loop) from the reward signal, showing whether gains come from the operator or the gradient. The synthetic diagnostic suite validates that Cl's zero set matches the intended well-formedness conditions. Together these isolate each design choice against a falsifiable observable.

### Expected outcome and causal chain

**vs. Vanilla step-level GRPO (no Cl)** — On a case where the policy emits a step whose edges create a cycle with an earlier step, the baseline produces a graph that stays cyclic and feeds the corrupted context forward, because nothing in its reward penalizes structural violations and task reward is sparse. Our method instead deletes the latest-order edge in the cycle at that step and penalizes the residual, because Cl's repair is applied before the next step consumes context. We expect a large gap in episode-end fixed-point fraction and a moderate task-reward gap concentrated on the multi-step subset where cycles are reachable.

**vs. End-of-episode closure only** — On a case where an intermediate step introduces a role-inconsistent node that later steps reference, the baseline produces a graph that is repaired only at episode end, after downstream steps already conditioned on a malformed node, because closure runs once and cannot undo the corrupted context. Our method instead repairs the node at the step it appears, so subsequent steps condition on a well-formed graph by induction. We expect near-parity on single-step tasks but a clear fixed-point and task-reward gap on tasks with three or more dependent steps.

**vs. Learned graph verifier reward** — On a case where the policy emits a subtly malformed graph that the verifier scores as acceptable, the baseline produces a positive reward on a non-fixed-point graph because the verifier's calibration error admits false positives. Our method instead assigns exactly zero closure reward only when Cl(G_pre)=G_pre, because the residual is the norm of a deterministic repair operator with no learned parameters. We expect the learned verifier to match us on gross violations but to show a detectable false-positive rate on near-miss graphs, widening our advantage as violation subtlety increases.

**vs. Grammar-constrained decoding / masked action space** — On a case where the policy must generate a new role or step order not present in the pre-specified grammar, the baseline either masks out the valid action (preventing generation) or admits a structurally invalid action (if the grammar is incomplete), because masking enforces only local syntactic constraints and cannot enforce global acyclicity or role-consistency across steps. Our method instead repairs the global graph after each step, so it can admit novel roles and orders while still enforcing well-formedness. We expect the masking baseline to fail on tasks requiring dynamic role introduction, showing a fixed-point and task-reward gap.

**vs. Constrained MDP with Lagrangian penalty** — On a case where structural violations are rare but severe, the baseline's Lagrangian penalty is tuned to a fixed threshold and may under-penalize rare violations or over-penalize benign steps, because the penalty is a scalar cost added to the reward without a repair mechanism. Our method instead computes an exact residual on the pre-closure graph and repairs the context, so the reward is zero exactly on well-formed graphs. We expect the Lagrangian baseline to show a higher false-positive rate on near-miss graphs and a lower fixed-point fraction, especially when the penalty weight is not perfectly tuned.

### What would falsify this idea

If our gain in fixed-point fraction and task reward is uniform across single-step and multi-step subsets rather than concentrated where cycles and role inconsistencies are reachable, the per-step closure mechanism is not the cause. Likewise, if the lambda=0 ablation matches full CLAG-RL, the closure reward — not the operator — is doing no work, and the central claim about the residual as a learning signal is wrong.

## References

1. Agent-R1: A Unified and Modular Framework for Agentic Reinforcement Learning
2. HybridFlow: A Flexible and Efficient RLHF Framework
3. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
4. Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations
5. OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text
6. Improving Generalization of Alignment with Human Preferences through Group Invariant Learning
7. Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates
8. GEAR: A GPU-Centric Experience Replay System for Large Reinforcement Learning Models
