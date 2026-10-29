# Procedure-Graph Sampling for Scalable, Valid, and Diverse Tool-Agent Evaluation

## Motivation

Existing procedure-aware benchmarks like TauIndianBankBench rely on manually curated tasks, so their coverage is fixed and their procedural validity is only as good as the curator's foresight; scaling to new domains or edge cases requires costly annotation and cannot systematically explore the space of compositional trajectories. This creates a structural evaluation gap: we cannot measure how agents generalize across the full space of procedurally valid action sequences, only across the subset someone thought to write down. TRACE and TRAJECT-Bench highlight trajectory-level metrics but do not solve the generation problem—their tasks are either manually designed or synthesized without a formal guarantee of procedural validity.

## Key Insight

A task is procedurally valid if and only if its required-step graph is a subgraph of a domain policy graph and its argument constraints are satisfied by the graph's path semantics; sampling such subgraphs directly yields valid tasks by construction, and the space of subgraphs provides a principled diversity measure that manual curation cannot achieve.

## Method

**What it is**: We introduce **Procedural Scenario Sampler (PSS)**, a generator that takes a domain policy graph (nodes = tool actions with pre/post-conditions, edges = allowed transitions) and produces a procedurally valid task specification consisting of a user request, a required-step graph (a subgraph of the policy graph), and argument constraints. Input: policy graph and a desired complexity profile (number of required steps, branching factor). Output: a task instance ready for a simulated user and a tool-using agent.

**Load-bearing assumption (stated explicitly)**: PSS assumes that a task's required-step graph H being a subgraph of the domain policy graph is sufficient for the task to be procedurally valid and for H to be the actually-required procedure — i.e., that graph-subgraph membership equals procedural necessity and executability under the sampled argument bindings. This assumption is load-bearing: if it fails, the 'validity by construction' claim is unverified. To bound this assumption, PSS's validity check (step 6) is an executable state-space reachability check, not merely a subgraph-membership check: it simulates H step-by-step against the policy graph's pre/post-condition semantics and rejects any H whose full conjunction of preconditions is not reachable from the start state under the sampled argument bindings. Conditionally-required nodes are marked explicitly so that 'required' means 'necessary given state', not merely 'present in a legal path'.

**How it works**:
```pseudocode
# Input: policy graph G = (V, E) with pre/post-conditions per node and
#        conditional-requirement flags; complexity profile (k_min, k_max, b_max)
# Output: task spec T = (user_request, required_steps, constraints, forbidden_actions)

1. Sample a connected subgraph H of G with k ~ Uniform(k_min, k_max) nodes and
   branching factor <= b_max, using a random walk with restart that respects
   edge directions and node preconditions.
2. For each node v in H, sample argument values from v's argument domains
   conditioned on the path from the root (ensuring argument consistency along
   the path, e.g., same account ID).
3. Identify a goal node g in H such that all nodes in H are on some path from
   the root to g (i.e., H is a directed acyclic graph rooted at start).
4. Generate a user request template that mentions only the goal g and any
   necessary context (e.g., "block my card"), deliberately omitting intermediate
   steps to test procedure adherence.
5. Add negative constraints: sample 1-2 nodes not in H that are plausible but
   unrequested, and record them as forbidden actions.
6. Verify procedural validity by executable state-space reachability:
   simulate H step-by-step from the start state under the sampled argument
   bindings, checking that each node's preconditions are satisfied by the
   accumulated state and that the full conjunction of preconditions is
   reachable; reject any H whose full conjunction is not reachable. Also
   check that H is a subgraph of G and that argument constraints are
   satisfiable; if any check fails, resample (rejection sampling).
7. Return T = (user_request, H, constraints, forbidden_actions).
```

**Why this design**: We chose a policy-graph subgraph sampler over a language-model-based task generator because the graph provides a formal guarantee of procedural validity (every required step is a legal action given the preconditions), accepting the cost that the policy graph must be manually specified per domain. We chose rejection sampling over a learned generative model because it ensures exact adherence to the graph's constraints without training data, accepting the cost that the acceptance rate may be low for dense graphs. We chose to sample negative constraints (forbidden actions) explicitly rather than relying the agent to avoid unrequested changes implicitly, because this makes the evaluation of unrequested changes precise and reproducible, accepting the cost that the task specification becomes more detailed. We chose a random-walk-with-restart subgraph sampler over uniform node sampling because it produces connected, realistic procedure paths, accepting the cost that the diversity of subgraphs may be biased toward high-degree nodes. This design is not a variant of TRACE or TRAJECT-Bench: TRACE evaluates trajectories post-hoc, while PSS generates tasks with a validity guarantee; TRAJECT-Bench synthesizes trajectories from manually designed tasks, while PSS generates the tasks themselves from a formal policy graph.

**Why it measures what we claim**: The subgraph H measures procedural validity because the policy graph encodes the domain's legal action sequences, and any subgraph of it corresponds to a sequence of actions that can be executed in order without violating preconditions; this assumption fails when the policy graph is incomplete or contains spurious edges, in which case H may include steps that are not actually required in practice. The complexity profile (k, b_max) measures task difficulty because the number of required steps and branching factor correlate with the cognitive and procedural load on the agent; this assumption fails when agents can shortcut steps via alternative tools not in the graph, in which case the measured difficulty overestimates the true difficulty. The forbidden actions measure the constraint against unrequested changes because they are explicitly sampled from plausible but unrequired nodes, and any execution of a forbidden action is a direct violation; this assumption fails when the agent's tool set includes actions outside the policy graph that achieve the same effect, in which case the constraint check may miss unrequested changes. The user request template measures the gap between stated goal and required procedure because it mentions only the goal and omits intermediate steps, forcing the agent to infer the procedure from the policy; this assumption fails when the agent has memorized the procedure from training data, in which case the task does not test procedure-following but recall.

## Contribution

["A procedure-graph sampling algorithm that generates procedurally valid and diverse evaluation tasks by sampling connected subgraphs from a domain policy graph, with a formal guarantee that every generated task's required steps are legal and executable in order.", 'A method for automatically injecting negative constraints (forbidden actions) and argument consistency checks into generated tasks, enabling precise measurement of unrequested changes and procedural adherence without manual annotation.', 'An empirical demonstration that the generated tasks cover a wider diversity of trajectory structures (in terms of required-step count, branching, and argument dependencies) than manually curated benchmarks, and that agent performance varies systematically with these structural properties.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | PSS-generated tasks from a retail-banking policy graph (30–50 nodes, authored per pre-registered protocol) | Formal validity guarantee enables controlled complexity sweeps |
| Primary metric | Procedure-adherence rate at matched outcome success | Isolates procedure-following from goal achievement |
| Baseline | Outcome-only scoring agent | Tests whether procedure signal is separable from outcome |
| Baseline | LLM-generated task benchmark (no graph guarantee) | Tests whether formal validity is what drives the effect |
| Baseline | Memorization-prone frontier model, zero-shot | Tests whether tasks test procedure vs. recall |
| Ablation-of-ours | PSS without forbidden-action constraints | Tests whether explicit negatives drive constraint measurement |
| Ablation-of-ours | PSS with spurious edges injected into policy graph | Bounds sensitivity to graph completeness |
| Ablation-of-ours | PSS with out-of-graph shortcut tools added | Bounds missed violations from alternative tools |

### Why this setup validates the claim

The central claim is that PSS produces tasks whose procedural validity is guaranteed by construction and whose complexity is controllable, so that procedure-adherence can be measured cleanly rather than confounded with outcome success or memorized recall. The outcome-only baseline tests the sub-claim that procedure adherence is a distinct, measurable signal: if PSS tasks are well-formed, an agent that reaches the goal but skips required steps should be scored differently under procedure-aware scoring than under outcome-only scoring, and the gap should be visible. The LLM-generated-task baseline tests the sub-claim that formal validity matters: on tasks generated without a policy-graph guarantee, some "required steps" will be procedurally illegal or unachievable, so measured procedure adherence will be noisy and will not track the complexity profile, whereas on PSS tasks adherence should degrade monotonically with k and b_max. The zero-shot frontier baseline tests the memorization confound: if PSS tasks are genuinely novel procedures, recall cannot rescue a model that does not follow the graph, so we expect a deficit concentrated on higher-complexity tasks rather than uniform failure. The ablation of forbidden-action constraints tests whether the explicit negative sampling is what makes unrequested-change detection precise: without it, violations should be under-counted because the agent's plausible-but-unrequested actions are not enumerated. The spurious-edge injection ablation tests the load-bearing assumption that subgraph membership equals procedural validity: by adding spurious edges to the policy graph, we measure how much adherence degrades when the graph is incomplete or contains illegal transitions, bounding the failure mode where H includes steps that are not actually required. The out-of-graph shortcut ablation tests the assumption that forbidden actions capture all unrequested changes: by adding alternative tools outside the policy graph that achieve the same effect, we measure missed violations and bound the failure mode where the constraint check misses unrequested changes. The primary metric, procedure-adherence rate at matched outcome success, is the right detector because it conditions on the agent having achieved the goal, so any residual variance is attributable to procedure-following rather than capability; a method that only improved outcome would show no effect here, and a method that only improved procedure would show it exactly here.

### Expected outcome and causal chain

**vs. Outcome-only scoring agent** — On a case where the user says "block my card" but the policy requires identity verification and a fraud-flag check before blocking, the outcome-only baseline blocks the card and is scored as a success because its mechanism checks only the final record state. Our method instead scores the same trajectory against the required-step subgraph H and records the skipped verification as a procedure violation, because the sampled H makes each required step explicit and checkable. We expect a large gap on tasks with k ≥ 4 required steps and near-parity on k = 1–2 tasks, since short procedures leave little room to skip.

**vs. LLM-generated task benchmark** — On a case where the LLM generator invents a "required" step that the policy graph does not permit given the account state, the baseline's task is procedurally invalid, so a correct agent is penalized for not taking an illegal step and the measured adherence is noise. Our method instead samples H as a subgraph of the policy graph and rejects any instance whose argument constraints are unsatisfiable or whose full conjunction of preconditions is not reachable from the start state under the sampled argument bindings, so every required step is executable in order. We expect the LLM-generated benchmark to show non-monotonic or flat adherence as complexity rises, while PSS shows a clean monotonic decline, with the divergence largest at high branching factor.

**vs. Memorization-prone frontier model, zero-shot** — On a case requiring a domain-specific procedure the model has seen in training, the baseline may reproduce the familiar sequence and appear to follow procedure, but on a PSS-sampled variant with a different required subgraph and different argument bindings it will substitute the memorized sequence for the required one. Our method's tasks are sampled from the graph with fresh argument constraints, so recall of a canonical sequence does not satisfy H. We expect the baseline to score well on low-complexity, canonical-looking tasks but to drop sharply on high-complexity or unusual-branch tasks, producing a steeper complexity slope than a procedure-following agent.

**vs. PSS without forbidden-action constraints** — On a case where the agent performs a plausible but unrequested action (e.g., changing the account's address while blocking a card), the ablated version has no explicit forbidden-action list, so the violation is not recorded and the agent is scored as compliant. Our method samples 1–2 forbidden actions from plausible but unrequired nodes, making the unrequested change directly checkable. We expect the ablation to under-count violations by a margin that grows with the number of plausible-but-unrequested actions in the domain, confirming that explicit negative sampling is what makes constraint measurement precise.

**vs. PSS with spurious edges injected into policy graph** — On a case where a spurious edge allows a step that is legal in the graph but not executable given the account state (e.g., a fraud-flag check that is only conditionally required), the injected graph admits H instances that are not actually required, so procedure adherence is measured against an incorrect required-step set. Our method's executable state-space reachability check rejects such H instances, so adherence remains stable. We expect adherence degradation to increase with the fraction of spurious edges, bounding the load-bearing assumption: when the graph is incomplete or contains illegal transitions, PSS's validity guarantee holds only to the extent that the reachability check filters out invalid H.

**vs. PSS with out-of-graph shortcut tools added** — On a case where the agent uses an alternative tool outside the policy graph to achieve the same effect as a required step (e.g., a direct database update instead of the sanctioned API call), the forbidden-action list does not include this tool, so the unrequested change is missed. Our method's constraint check is limited to the policy graph's nodes, so missed violations are expected to increase with the number of out-of-graph shortcuts. We expect this ablation to quantify the failure mode where the constraint check misses unrequested changes, bounding the assumption that forbidden actions capture all unrequested changes.

### What would falsify this idea

If procedure-adherence gains from PSS are uniform across complexity levels and branching factors rather than concentrated on high-k, high-b_max tasks where the predicted failure modes bite, then the tasks are not actually testing procedure-following and the central claim that PSS's validity guarantee and complexity control drive the effect is wrong. Likewise, if the forbidden-action ablation shows no difference in unrequested-change detection, the claim that explicit negative sampling is what makes constraint measurement precise is falsified. Additionally, if the spurious-edge injection ablation shows no degradation in adherence, the load-bearing assumption that subgraph membership equals procedural validity is not empirically bounded, and if the out-of-graph shortcut ablation shows no missed violations, the assumption that forbidden actions capture all unrequested changes is falsified.

## References

1. TauIndianBankBench: Procedure-Aware, Action-Oriented Evaluation of Tool-Using Conversational Agents
2. Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents
3. TRAJECT-Bench:A Trajectory-Aware Benchmark for Evaluating Agentic Tool Use
4. CRMArena-Pro: Holistic Assessment of LLM Agents Across Diverse Business Scenarios and Interactions
5. ACEBench: Who Wins the Match Point in Tool Usage?
6. τ2-Bench: Evaluating Conversational Agents in a Dual-Control Environment
7. Non-Collaborative User Simulators for Tool Agents
8. ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities
