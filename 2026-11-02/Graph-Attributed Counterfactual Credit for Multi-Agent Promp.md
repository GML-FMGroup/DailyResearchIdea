# Graph-Attributed Counterfactual Credit for Multi-Agent Prompt Optimization

## Motivation

Existing multi-agent prompt optimizers treat the execution graph as a fixed scaffold and update role prompts against a single task-level scalar reward. CEPO, for instance, structures prompts into a shared coordinator and per-agent executors and then hands the search to an external optimizer that sees only final task success, so every role prompt receives the same undifferentiated signal regardless of which agent transition actually caused the failure. This is structurally unable to assign credit to the role whose prompt change would have altered the outcome, because the optimizer's objective never references the graph edge on which the role acts.

## Key Insight

A role prompt's causal contribution is identifiable as the counterfactual change in downstream transition success when that role's prompt is perturbed while its incoming state is held fixed, which makes the graph edge — not the task reward — the unit of credit.

## Method

**What it is** — GACCO (Graph-Attributed Counterfactual Credit Optimizer) takes a multi-agent execution graph G=(V,E) with role prompts θ_v, a task distribution, and a frozen coordinator/executor LLM, and returns updated role prompts θ'_v. It replaces the scalar task reward with a per-edge advantage computed from counterfactual role-prompt perturbations along each transition.

**How it works**
```pseudocode
Input: roles V, edges E, initial prompts {θ_v}, tasks D, perturbation set P (|P|=3 paraphrases per role), budget B (number of optimization iterations, default 20), calibration set D_cal (|D_cal|=64 held-out tasks), executor temperature=0 (deterministic)

# Phase 0: composition calibration (one-time, before iteration 1)
for each task d in D_cal:
  run G with prompts θ, record for each edge (u→v) the incoming state s_{u→v} and downstream outcome r_{u→v} (binary success of v's subgoal)
for each edge (u→v) in E and each perturbed prompt θ_v^p in P:
  replay v from frozen s_{u→v} with θ_v^p, record r_{u→v}^p
  compute edge delta δ_{u→v}^p = r_{u→v}^p - r_{u→v}
  replay full downstream subgraph from s_{u→v} to task termination with θ_v^p, record global task success R^p
  compute task delta Δ^p = R^p - R (R = baseline task success)
fit calibration coefficient c_{u→v} = Cov(δ_{u→v}^p, Δ^p) / Var(δ_{u→v}^p) over all p and d in D_cal (if Var=0, set c_{u→v}=0)

for iteration t = 1..B:
  # Phase 1: rollout and record graph
  for task d in D:
    run G with prompts θ, record for each edge (u→v) the incoming state s_{u→v} and the downstream outcome r_{u→v} (binary success of v's subgoal)
  # Phase 2: counterfactual edge perturbation
  for each edge (u→v) in E and each perturbed prompt θ_v^p in P:
    replay v from frozen s_{u→v} with θ_v^p, record r_{u→v}^p
    edge advantage A_{u→v} = mean_p [ r_{u→v}^p - r_{u→v} ]   # transition-level counterfactual delta
    calibrated edge advantage Ã_{u→v} = c_{u→v} · A_{u→v}   # composition-calibrated
  # Phase 3: graph attribution to roles
  for each role v:
    A_v = Σ_{u: (u→v)∈E} w_{u→v} · Ã_{u→v},  w_{u→v} = 1/indegree(v)  # incoming-edge attributed credit
    also propagate outgoing credit: A_v += λ · Σ_{w: (v→w)∈E} Ã_{v→w}   (λ=0.3)
  # Phase 4: prompt update guided by attributed signal
  for each role v:
    select top-k (k=2) perturbed prompts by A_v; θ_v ← best of {θ_v} ∪ {θ_v^p : A_v(θ_v^p) > A_v(θ_v)} using attributed advantage as selection criterion
  # Phase 5: coordinator prompt updated only if global task reward improves across all edges (shared signal)
  if mean task reward improved: update coordinator with task-level gradient
return {θ_v}
```

**Why this design** — We chose to perturb role prompts at frozen incoming states rather than re-running full rollouts because the frozen-state replay isolates the role's contribution from upstream noise, accepting the cost that replay requires the executor to be deterministic given a state (we fix temperature to 0). We chose to attribute credit via incoming-edge averaging plus a discounted outgoing term rather than a single global reward because the graph structure encodes which roles can causally affect which others, accepting that this attribution is only as good as the recorded graph topology (missing edges yield zero credit). We chose counterfactual delta over the task-level reward as the selection criterion for prompt updates because the task reward conflates all roles' contributions, accepting that the counterfactual estimate has higher variance per update and thus needs a larger perturbation set (|P|=3) to stabilize. Unlike VinePPO or process reward models, which estimate step values from Monte-Carlo returns of the same policy, GACCO intervenes on the prompt itself and measures the resulting change in transition success, so a domain expert would not describe it as a variant of per-step value estimation: the intervention target is the prompt, and the graph edge is the attribution unit. The composition-calibration coefficient c_{u→v} is a one-time linear rescaling fitted on a held-out calibration set (|D_cal|=64) that maps edge-level deltas to task-level deltas; it does not change the intervention or the attribution unit, but it makes explicit the assumption that local transition improvements compose monotonically to task success and provides a concrete verification step (Phase 0) that measures the empirical correlation between edge deltas and task deltas.

**Why it measures what we claim** — The edge advantage A_{u→v} measures the causal contribution of role v's prompt to the transition outcome because holding the incoming state s_{u→v} fixed while perturbing θ_v makes the transition outcome a function of θ_v alone under the assumption that the executor's behavior on a fixed state depends only on its prompt; this assumption fails when the executor's output is stochastic or when the incoming state is not a sufficient statistic for the role's decision context, in which case A_{u→v} reflects replay noise rather than prompt causality. The graph-attributed role signal A_v measures role-level credit because the incoming-edge average plus discounted outgoing term operationalizes the assumption that a role's influence flows through its incident edges with equal weight per incoming edge and discounted weight per outgoing edge; this fails when the graph has high-degree hubs whose outgoing influence is not uniform, in which case A_v over-credits hub roles. The selection of perturbed prompts by A_v measures prompt improvement because a higher counterfactual transition success under the perturbed prompt is assumed to transfer to the full task, which fails when local transition success does not compose to global task success (e.g., coordination failures across edges), in which case the update optimizes local subgoals at the expense of the task; the calibrated edge advantage Ã_{u→v} = c_{u→v} · A_{u→v} measures the expected change in global task success under the perturbed prompt because c_{u→v} is the empirical linear coefficient relating edge deltas to task deltas on held-out tasks, and this calibration fails when the relationship is non-linear or task-dependent, in which case Ã_{u→v} reflects a linear approximation of the true composition.

## Contribution

['A graph-conditioned credit assignment mechanism for multi-agent prompt optimization that computes transition-level counterfactual advantages from role-prompt perturbations at frozen incoming states and attributes them to roles via incident-edge aggregation.', 'A design principle showing that role prompts should be selected by their attributed edge advantage rather than by the global task reward, with the coordinator prompt updated only on global improvement — separating shared coordination from role-specific execution credit.', 'An open implementation of the frozen-state replay perturbation protocol that makes counterfactual prompt credit computable without full multi-agent re-rollouts.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-agent benchmarks with recorded execution graphs (e.g., document-understanding MAS, collaborative reasoning) | Provides graph topology for edge attribution |
| Primary metric | Task success rate vs. optimization budget curve (budget = number of LLM calls) | Tests whether counterfactual credit beats global reward per cost |
| Baseline | CEPO-wrapped prompt optimizer | Strongest prior multi-agent prompt optimization |
| Baseline | Global-reward prompt optimizer (no credit attribution) | Isolates value of edge-level attribution |
| Baseline | VinePPO-style per-step value estimation | Tests distinctness from step-value methods |
| Ablation-of-ours | GACCO without outgoing-credit propagation (λ=0) | Tests whether graph propagation adds signal |
| Ablation-of-ours | GACCO without composition calibration (c_{u→v}=1) | Tests whether calibration improves credit fidelity |
| Diagnostic metric | Correlation between A_v and true leave-one-role-out task-success delta | Validates attribution quantity directly |

### Why this setup validates the claim

The central claim is that attributing credit per graph edge via counterfactual prompt perturbation yields better role-prompt updates than conflating all roles into a scalar task reward. The global-reward baseline directly tests this: if our method only matched it, the credit-attribution machinery would be unjustified. CEPO-wrapped optimization tests whether we beat the strongest existing multi-agent prompt-optimization framework, not just a strawman. The VinePPO-style baseline tests our claim that prompt intervention is distinct from per-step value estimation — if it matched us, our framing would collapse. The λ=0 ablation isolates the graph-propagation sub-claim: if removing outgoing credit does not hurt, the graph structure is not doing work and the method reduces to per-node local optimization. The no-calibration ablation tests whether the composition-calibration coefficient c_{u→v} (fitted on 64 held-out tasks) improves the fidelity of edge credit to global task success; if it does not, the load-bearing composition assumption is unverified and the method's central claim is weakened. The diagnostic metric directly measures whether A_v ranks roles by their true causal contribution (leave-one-role-out task-success delta), providing a direct test of the attribution quantity. Task success rate as a function of optimization budget (measured in LLM calls) is the right metric because our mechanism predicts *faster* improvement per budget unit (better-aimed updates), not merely a higher asymptote; a metric that ignored budget would fail to distinguish credit-attribution efficiency from raw capacity. Together these choices form a falsifiable test: gains must concentrate where credit misattribution is predicted to hurt, and must disappear when graph attribution is removed.

### Expected outcome and causal chain

**vs. Global-reward prompt optimizer** — On a case where a downstream role fails because an upstream role's prompt produced an ambiguous handoff, the global-reward baseline assigns the same scalar penalty to every role, so it perturbs the downstream role's prompt even though the upstream prompt caused the failure. Our method instead measures the counterfactual transition delta at the upstream edge under a perturbed upstream prompt, so credit lands on the causal role. We expect a noticeable gap on tasks with long chains and asymmetric role influence, but near-parity on short two-role tasks where the global reward is already close to per-role. The composition-calibration coefficient c_{u→v} is expected to be positive and less than 1 on average, reflecting that local transition improvements only partially compose to task success; the no-calibration ablation is expected to show lower correlation between A_v and leave-one-role-out deltas.

**vs. CEPO-wrapped prompt optimizer** — On a case where the coordinator's prompt is fine but one executor role's prompt systematically degrades a specific transition, CEPO's framework-level optimization improves prompts jointly but cannot localize which role to edit, so it spends budget on roles that are not the bottleneck. Our method instead ranks perturbed prompts by attributed edge advantage, concentrating edits on the bottleneck role. We expect faster success-rate improvement per optimization iteration (measured in LLM calls), with the gap widening as the number of agents grows and narrowing when all roles contribute equally.

**vs. VinePPO-style per-step value estimation** — On a case where a step's Monte-Carlo return is high by luck despite a weak role prompt, the value-estimation baseline reinforces that prompt because it reads returns of the same policy rather than intervening. Our method instead replays the frozen state under a perturbed prompt and measures the *change* in transition success, isolating the prompt's causal effect. We expect our method to avoid reinforcing lucky-but-weak prompts, showing a gap concentrated on high-variance transitions and parity on low-variance ones.

### What would falsify this idea

If GACCO's gain over the global-reward baseline is uniform across graph topologies and role-degree subsets rather than concentrated on long chains, high-indegree bottlenecks, and asymmetric-influence tasks where credit misattribution is predicted to hurt, the central claim that edge-level counterfactual attribution drives the improvement is wrong. Likewise, if the λ=0 ablation matches full GACCO, the graph-propagation claim is falsified and the method reduces to independent per-role local search; if the no-calibration ablation matches full GACCO and the correlation between A_v and leave-one-role-out deltas is near zero, the composition assumption is unverified and the credit-attribution claim is unsupported.

## References

1. CEPO: A General Framework for Improving Prompt Optimization in Multi-Agent Systems
