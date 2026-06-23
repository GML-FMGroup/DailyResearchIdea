# CausalAgentEval: Self-Validating Benchmark Evaluation via Causal Structure Learning and Automated Intervention

## Motivation

Existing agent benchmarks evaluate performance using human judgments or black-box judges, which introduces an evaluator evaluation regress where the judge itself requires costly validation. For instance, Automatically Benchmarking LLM Code Agents fine-tunes a judge on human annotations, but still relies on human oversight and cannot verify its own assessments. The root cause is that current evaluation frameworks lack a mechanism to test their own causal assumptions about agent behavior without external supervision.

## Key Insight

The causal graph of an agent's decision-making process is a testable model: if it correctly captures behavior, then interventions that perturb specific decision nodes will yield predicted outcomes, enabling fully automated self-validation of the evaluation metrics derived from the graph.

## Method

(A) **What it is**: CausalAgentEval is a self-validating evaluation framework that, given agent trajectories and an environment simulator, learns a causal graph of the agent's decisions using the FCI algorithm (which handles latent confounders), computes multi-dimensional quality metrics from graph properties, and automatically designs and executes interventions to validate its causal predictions.

(B) **How it works** (pseudocode):
```python
Input: Trajectories T = {τ_i}, environment simulator E with intervention API
Hyperparameters: α=0.05 (significance), k=10 (trials per edge), δ=0.1 (tolerance), β=0.8 (validation threshold)

# Phase 1: Causal Structure Learning (FCI algorithm to handle latent confounders)
G = empty PAG (Partial Ancestral Graph) with nodes = observed variables (states, actions, reward)
for each pair (X, Y) in variables:
    if conditional independence test (FCI algorithm) rejects H0 at α:
        add edge X → Y (respecting temporal order: state before action)
# FCI outputs a PAG representing causal relationships under possible latent confounders

# Phase 2: Metric Derivation
Goal-directedness = average causal effect of final reward on intermediate states via do-calculus (computed from PAG)
Efficiency = average shortest path length from initial to goal state in the underlying DAG representation (if a DAG can be extracted; otherwise, use PAG skeleton)
Robustness = average number of alternative paths when removing a random node (degree of connectivity in PAG)
Metrics M = (g, e, r)

# Phase 3: Automated Intervention Validation
valid_edges = 0
for each edge X → Y in the PAG's directed edges:
    for trial in 1..k:
        design intervention: set X to a value uniformly sampled from its observed domain
        execute intervention in E with same agent, get Y_observed
        query G using do(X=x) to compute predicted distribution P(Y|do(X=x)) (using do-calculus on PAG)
        if |Y_observed - expected[P(Y|do(X=x))]| < δ:
            valid_edges += 1/k
validation_score = valid_edges / |directed_edges|
if validation_score > β: accept metrics M as reliable; else, request more trajectories
```

(C) **Why this design**: We chose the FCI algorithm over PC because the PC algorithm fails in the presence of latent confounders, which are common in agent trajectories due to hidden internal states; FCI explicitly models and tests for such confounders, providing more robust causal discovery at the cost of increased computational complexity. We derive metrics from graph properties rather than raw trajectory statistics because graph properties like causal effect and path length directly operationalize intuitive notions of agency while raw statistics conflate correlation with causation. We select automated intervention over holdout validation because intervention directly tests causal predictions, breaking the regress of needing a separate evaluator; the trade-off is the requirement for an environment that supports interventions. We fix intervention trials k=10 to balance statistical power and computational cost. Unlike the fine-tuned judge approach, our framework does not require a separate training phase on human judgments; instead it self-validates using the environment itself.

(D) **Why it measures what we claim**: The average causal effect of final reward on intermediate states (computed via do-calculus) measures _goal-directedness_ because it quantifies how much the agent's actions causally drive progress toward the goal, under the assumption that the learned graph faithfully captures all relevant decision variables and that the FCI algorithm correctly accounts for latent confounders; this assumption fails when there are unmeasured confounders that FCI cannot fully recover (e.g., time-varying hidden states), in which case the effect may still reflect partial correlation rather than causation. The shortest path length from initial to goal state in G measures _efficiency_ because it captures the minimal number of decisions required to reach the goal, under the assumption that the graph encodes all necessary decisions; this assumption fails when there are redundant or missing decision nodes, in which case length misrepresents actual complexity. The validation score (fraction of edges passing intervention tests) measures _trustworthiness_ of the evaluation because a high score confirms the causal model is predictive under perturbations, under the assumption that the interventions cover the full causal structure; this assumption fails when the intervention set is incomplete (e.g., only tests easy-to-intervene edges), in which case the score overestimates reliability.

## Contribution

(1) A causal structure learning framework that extracts interpretable decision graphs from agent trajectories, generalizable across benchmark environments. (2) A self-validation mechanism that automatically designs and executes interventional experiments to test the learned graph's predictions, eliminating the need for human evaluator validation. (3) Multi-dimensional quality metrics (goal-directedness, efficiency, robustness) derived from graph properties that are causally grounded and interpretable.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Synthetic tool-use simulator with known DAG (DefaultSim-v0 with configurable DAG) | Enables ground truth verification |
| Dataset (real-world) | WebArena (simplified subset) | Show applicability beyond synthetic |
| Primary metric | MSE of derived vs true causal effects | Directly measures accuracy |
| Baseline 1 | Raw trajectory statistics | No causal modeling |
| Baseline 2 | Fine-tuned judge (PRDBench-style) | No causal validation |
| Baseline 3 | CausalBench (specific recent causal evaluation method) | Highlight distinct self-validation |
| Ablation | Without intervention validation | Isolates impact of validation |
| Diagnostic | Vary confounder strength (0, 0.3, 0.6) | Test robustness to hidden variables |

### Why this setup validates the claim

This setup directly tests the core claim that causal graph learning with automated intervention validation produces more accurate agent evaluation metrics than non-causal alternatives. By using a synthetic environment with known ground truth causal DAG, we can compute the true causal effects and compare against the metrics derived by each method. The raw trajectory baseline tests whether simply aggregating observed outcomes suffices; the fine-tuned judge baseline tests whether a trained evaluator can approximate causal reasoning without explicit structure learning; CausalBench tests whether an existing causal method without self-validation matches our approach. The ablation removes the intervention validation phase to isolate its contribution to accuracy. The diagnostic experiment varies confounder strength to test robustness: we inject hidden variables that affect both actions and rewards, and measure causal discovery precision/recall. The primary metric, mean squared error between derived and true causal effects, is the appropriate signal because it directly measures how well each method recovers the underlying causal relationships that define goal-directedness and efficiency. If our method's MSE is significantly lower, especially in the presence of unobserved confounders, then the claim is supported. Conversely, if the ablation performs similarly, then intervention validation is unnecessary.

### Expected outcome and causal chain

**vs. Raw trajectory statistics** — On a task where the agent makes suboptimal but causally effective detours (e.g., selecting a less optimal tool first but still achieving the goal), raw trajectory statistics (like average reward) would penalize the extra steps, giving low efficiency and goal-directedness. Our method, by constructing a causal graph and computing path lengths and causal effects, would correctly recognize the detour as causally necessary due to constraints, leading to higher fidelity metrics. We expect our method to show lower MSE on tasks with such causal complexities, but comparable MSE on tasks where trajectories align with causal structure.

**vs. Fine-tuned judge** — On a task with an unseen confounding variable (e.g., hidden tool failure rate), the fine-tuned judge, trained on human annotations from a different distribution, would misattribute success or failure to the visible actions, producing inaccurate evaluations. Our method explicitly tests for causal relationships through interventions; if the hidden confounder affects multiple actions, the intervention validation would detect inconsistencies, leading to lower validation scores and more conservative metric estimates. Thus, we expect our method's MSE to be lower than the judge's MSE, especially on tasks with strong confounders.

**vs. CausalBench** — On a task where the causal graph has latent confounders, CausalBench (which assumes causal sufficiency) may produce biased metrics. Our method, using FCI, explicitly handles latent confounders and self-validates via interventions. We expect our method to maintain lower MSE when confounder strength is high, while performing similarly on clean tasks.

**vs. Ablation (without intervention validation)** — On a task where causal discovery incorrectly orientates an edge due to finite sample noise, our full method would catch this via intervention and adjust its metrics, whereas the ablation would propagate the error. Therefore, we expect the full method to yield lower MSE than the ablation on such tasks, but similar performance on clean tasks.

### What would falsify this idea

If the ablation without intervention consistently matches or outperforms the full method across all confounding levels, then the intervention validation is not providing genuine benefit, contradicting the claim that it self-validates and improves reliability. Also, if our method's MSE on high-confounding tasks is not noticeably higher than on low-confounding (indicating failure to detect confounders), the central thesis would be falsified.

## References

1. Automatically Benchmarking LLM Code Agents through Agent-Driven Annotation and Evaluation
2. Automating Benchmark Design
3. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains
4. Automatic Generation of Benchmarks and Reliable LLM Judgment for Code Tasks
5. CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion
6. MetaTool Benchmark for Large Language Models: Deciding Whether to Use Tools and Which to Use
7. WebArena: A Realistic Web Environment for Building Autonomous Agents
8. AgentBench: Evaluating LLMs as Agents
