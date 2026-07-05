# Causal Workflow Induction via Active Intervention in Structured Agent-Computer Interfaces

## Motivation

Current autonomous training systems like AutoTrainess rely on predefined workflows that assume tasks are fully decomposable into known sub-tasks. This structural assumption prevents automatic extension to unseen training scenarios because the workflow library cannot generate new sub-task structures. The root cause is that the system has no mechanism to infer sub-goal dependencies from scratch.

## Key Insight

The deterministic state transitions of structured agent-computer interfaces guarantee that interventional distributions are identifiable, enabling a causal graph over sub-goal states to be learned solely from the outcomes of atomic action executions.

## Method

We propose **Causal Workflow Learner (CAWL)**, an agent that interacts with a structured interface by executing atomic actions (interventions) to infer a causal Bayesian network over state variables, then constructs a workflow as a topological ordering of sub-goals that achieve the target.

(A) **What it is:** CAWL takes a target goal and a set of atomic actions (each with preconditions and deterministic effects on observable state variables). It outputs a workflow: a sequence of sub-goal states that, when achieved in order, realize the target.

(B) **How it works:** The agent maintains a causal graph over state variables, initialized as an empty graph with all variables observed. It iterates: (1) choose a variable with high uncertainty (e.g., highest variance of estimated causal coefficients), (2) intervene on that variable by executing an atomic action that sets it to a specific value (ensuring the action's preconditions hold), (3) observe the new state and update the causal graph (e.g., using a constrained PC algorithm or Bayesian score). After reaching a convergence criterion (e.g., graph structure unchanged for k interventions), the agent performs a topological sort on the causal graph to produce a sequence of sub-goals: for each variable, the sub-goal is its target value. The agent then executes this workflow and verifies the target is achieved; if not, it continues causal discovery.

```pseudocode
Initialize: G = empty graph over state variables V
Initialize: data D = []
Initialize: convergence_counter = 0
while not converged:
    # choose variable with max interventional variance
    v = argmax_{v in V} Var(E[parent_coefficients(v)])
    # select atomic action a that sets v to target value
    a = select_action(v, target_goal_v)
    execute a, observe new_state
    D.append( (pre_state, action, post_state) )
    # update G using constraint-based learning (e.g., PC with conditional independence tests at alpha=0.05)
    G = cpdag_learn(D, alpha=0.05)
    if G unchanged in last 3 interventions:
        convergence_counter += 1
    else:
        convergence_counter = 0
    if convergence_counter >= 5:
        converged = True
# compute topological ordering of G (any linear extension)
order = topological_sort(G)
# map each variable to sub-goal value
workflow = [ (v, target_goal_v) for v in order ]
return workflow
```

(C) **Why this design:** We chose intervention-based causal discovery over observational learning because the structured interface provides deterministic, reversible actions, making interventions efficient and ensuring identifiability; the cost is that interventions require additional environment steps (resource overhead). We used a constraint-based algorithm (PC) rather than a score-based one because the discrete state space of interfaces makes conditional independence tests computationally cheap and interpretable, at the cost of sensitivity to the chosen significance level (alpha=0.05) which may produce false edges. We prioritized interventional variance as the variable selection criterion over random selection because it targets variables whose causal role is most uncertain, accelerating convergence, though it may neglect variables with low variance but high causal impact in rare paths. Finally, we use topological sorting on the learned graph to produce a workflow rather than searching for an optimal sequence because the causal graph's acyclicity guarantees a valid ordering, but this ordering may be suboptimal in terms of execution cost; we accept this trade-off for simplicity and correctness.

(D) **Why it measures what we claim:** The conditional independence tests (e.g., Fisher's z-test) measure *causal necessity* between variable pairs because, under the causal Markov condition and faithfulness, independence in the interventional distribution implies absence of direct causal relation; this assumption fails when latent confounders exist (e.g., unobserved state variables), in which case the test reflects spurious correlation rather than causation. The interventional variance (Var(E[parent_coefficients(v)])) measures *epistemic uncertainty* about v's causal parents because high variance indicates conflicting evidence from past interventions; this assumption fails when the coefficient estimator is biased (e.g., due to small sample), in which case variance reflects noise rather than uncertainty. The topological ordering of the learned DAG measures *attributable sub-goal sequence* because acyclicity ensures that each sub-goal's achievement is causally prior to later ones; this assumption fails if the true causal graph contains cycles (e.g., feedback loops between state variables), in which case the ordering may not respect actual causal dependencies.

## Contribution

(1) A novel algorithm, CAWL, that automatically induces workflows for unseen training goals by learning causal graphs from active interventions in structured agent-computer interfaces. (2) The design principle that causal graph learning can replace manual workflow engineering, enabling generalization to new tasks without predefined sub-task decomposition. (3) An evaluation framework for measuring workflow correctness and efficiency in terms of intervention cost.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | WorkflowBench (50 structured training tasks) | Tests causal inference from interventions |
| Primary metric | Task success rate | Directly measures goal achievement |
| Baseline: CLI-only | Raw command-line agent | No structured workflow or causal model |
| Baseline: Random workflow | Random sub-goal ordering | Tests necessity of causal ordering |
| Baseline: Observational-only | Agent without interventions | Tests value of interventional data |
| Ablation: Random variable selection | CAWL with random intervention targets | Tests interventional variance heuristic |

### Why this setup validates the claim

WorkflowBench includes tasks with varying causal dependencies (e.g., learning rate adjustment must precede batch size change). Task success rate directly reflects whether the agent produces a valid workflow. The CLI-only baseline tests the benefit of any structured workflow. Random workflow tests if causal ordering matters. Observational-only tests the necessity of active interventions. The ablation isolates the impact of the variable selection heuristic. Together, these allow a falsifiable test: if CAWL outperforms all baselines, the claim that causal discovery from interventions yields better workflows is supported. Conversely, if a baseline matches CAWL, the corresponding mechanism gap is not critical.

### Expected outcome and causal chain

**vs. CLI-only** — On a task where hyperparameter order matters (e.g., set learning rate then decay scheduler), the CLI agent may set learning rate after scheduler due to trial-and-error, causing irrecoverable failure. CAWL infers causal graph (scheduler depends on learning rate) and orders correctly, so we expect CAWL to succeed on such tasks while CLI fails, yielding a >30% gap on causally structured tasks but parity on simple ones.

**vs. Random workflow** — On a task with a strict causal chain (e.g., model architecture must be chosen before fine-tuning), random ordering often violates dependencies, leading to preconditions false. CAWL's topological sort ensures ordering, so we expect CAWL to succeed >80% on all tasks, while random success rate is ≤50% on chain tasks, with large gap on complex workflows.

**vs. Observational-only** — On a task with latent confounders (e.g., two variables correlated but not causally linked), observational learning may infer spurious edges, producing incorrect workflow. CAWL's interventions break correlations, identifying true parents. Thus we expect CAWL to maintain >80% success while observational-only drops below 50% on tasks with spurious correlations.

### What would falsify this idea

If CAWL's success rate is not significantly higher than Random workflow on tasks with known causal dependencies, or if its gain is uniform across all task types (i.e., no concentration where causal inference is critical), the central claim that intervention-based causal discovery improves workflow learning would be refuted.

## References

1. AutoTrainess: Teaching Language Models to Improve Language Models Autonomously
2. PaperBench: Evaluating AI's Ability to Replicate AI Research
3. AlphaResearch: Accelerating New Algorithm Discovery with Language Models
4. Agent-as-a-Judge: Evaluate Agents with Agents
5. MLR-Copilot: Autonomous Machine Learning Research based on Large Language Models Agents
6. OpenHands: An Open Platform for AI Software Developers as Generalist Agents
7. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
8. MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation
