# Causal Intervention for Task Generation: Overcoming Tool-Bound Distribution in Language Agents

## Motivation

Self-Challenging (reference) generates tasks via tool interaction, producing tasks that are mere variations of tool-use patterns. This structurally limits distribution coverage because the task distribution is constrained by the agent's API set. To enable tasks requiring creative synthesis or causal understanding beyond tools, the agent must learn causal structures that underlie tool effects and generate tasks by intervening on those structures.

## Key Insight

Causal graphs capture invariant mechanisms that hold across different tool configurations; intervening on non-tool nodes yields tasks that require understanding these invariant causal relationships, thereby escaping the tool-bound distribution.

## Method

We present Causal Task Intervention (CTI).

(A) **What it is**: CTI is a task generation framework that inputs a learned causal graph of the environment and outputs novel tasks (instruction + verification function) requiring causal reasoning beyond tool-dependent patterns. It operates via intervention on the causal graph.

(B) **How it works**:

```python
# Pseudocode for CTI training loop (phases)
Phase 1: Collect environment interaction traces using existing agent (Llama-3.1-8B-Instruct, frozen).
Phase 2: Learn causal graph G from traces (PC algorithm, α=0.05; verify invariance of edges incident to tool-effect nodes using 100 calibration traces with swapped tools; remove edges that fail invariance).
Phase 3: For each task generation step:
  1. Sample a target variable V from G that is NOT directly a tool-effect node (rule: V not in set of nodes representing tool outputs).
  2. Select an intervention value v* that is counterfactual (statistically unlikely under current policy). Heuristic: pick value that maximizes change in V's parents' values.
  3. Perform a single-target do-intervention: set V=v* in the graph, and compute expected values of other nodes via causal inference (using G).
  4. Ground the intervention into a task instruction: e.g., "Achieve state where V=v* while using tools appropriately." (template: "Set {V} to {v*} by using any available tools, but without causing {conflict condition}").
  5. Generate verification function: check if V is v* and that tool usage is consistent with causal constraints (e.g., if V cannot be achieved via direct tool use, enforce that no tool directly changes V).
  6. Filter task: reject if any involved variable is causally disconnected from available actions (using graph distance threshold d_max=2) or if the intervention value is unreachable (e.g., outside observed range).
Phase 4: Train agent on generated tasks using reinforcement learning (as in Self-Challenging: PPO, 10K steps, lr=1e-5, same base model Llama-3.1-8B-Instruct).
```

(C) **Why this design**: We chose to learn a full causal graph rather than relying on observational correlations because tool-effect correlations are non-stationary (tools change), whereas causal mechanisms are invariant. The trade-off is learning quality: causal discovery from limited data is noisy, but we accept that imperfect graphs still yield better task diversity than tool-derived ones. We perform do-interventions rather than conditioning because interventions break statistical dependencies, ensuring tasks require understanding of causal mechanisms not just correlational patterns; the cost is that counterfactual values may be unrealistic, but we filter via feasibility. We train a separate agent on the generated tasks rather than using the same model for generation and execution because the generator's causal understanding may be stale; this duplicates training but avoids self-reinforcing bias. Finally, we filter tasks by causal connectivity to avoid impossible tasks, accepting that some potentially useful tasks with indirect connections may be discarded. The verification step (invariance testing) mitigates the risk of false edges, ensuring that only robust causal relationships are used for intervention targets.

(D) **Why it measures what we claim**: The computational quantity "intervention target variable set" measures "causal novelty" (i.e., tasks not derivable from tool-use patterns) because we select variables not directly affected by tools, under the assumption that tool-effect nodes are known; this assumption fails when a non-tool node is spuriously correlated with tool effects due to unobserved confounders, in which case the intervention may inadvertently target a tool-proxy and still yield tool-derived tasks. We test this by deliberately mislabeling 20% of tool-effect nodes and measuring performance degradation. The "intervention value v*" measures "counterfactual difficulty" because it is chosen to be statistically unlikely under current policy, under the assumption that policy behavior is drawn from a stationary distribution; this assumption fails when the policy has high variance, in which case unlikely values may still be reachable via tool usage. The "verification function" measures "causal faithfulness" because it checks consistency with causal graph predictions, under the assumption that the graph is correctly specified; this assumption fails when the graph contains false edges, in which case verification may reject valid tasks or accept invalid ones. The invariance verification step (Phase 2) additionally measures "causal robustness" by testing that edges involving tool-effect nodes remain stable under tool swaps.

## Contribution

(1) A novel task generation framework (CTI) that leverages learned causal graphs to produce tasks requiring understanding of underlying causal relationships, beyond tool-dependent patterns. (2) The design principle that intervention on non-tool causal nodes yields task distribution coverage that cannot be achieved by tool-interaction based generation. (3) A causal discovery and intervention pipeline integrated with reinforcement learning for tool-augmented agents.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | M3ToolEval | Multi-turn tool-use with diverse tasks. |
| Primary metric | Task success rate | Direct measure of task completion. |
| Secondary metric | Graph Structural Hamming Distance (SHD) vs. ground truth in toy simulator (5 tools, known graph) | Sanity check on causal discovery accuracy. |
| Baseline 1 | Original Llama-3.1-8B-Instruct | Untrained agent, same base model. |
| Baseline 2 | Self-Challenging (Llama-3.1-8B) | Prior task generation from own experience. |
| Baseline 3 | Random task generation | Non-causal task proposal baseline. |
| Baseline 4 | CTI with random causal graph (Erdos-Renyi, same density as learned) | Ablation: random structure. |
| Baseline 5 | CTI with correlation-based causal discovery (threshold r>0.3) | Ablation: simpler discovery. |
| Ablation | CTI without intervention | Uses conditioning instead of do-interventions. |

### Why this setup validates the claim

The dataset M3ToolEval includes tasks that require causal reasoning such as understanding tool effects. The primary metric, task success rate, directly measures whether the agent can complete the task. The secondary metric (SHD) quantifies how accurately the learned causal graph approximates the true causal structure in a controlled setting, thereby testing the load-bearing assumption of graph correctness. The baselines isolate key factors: Original Llama shows the effect of any training; Self-Challenging tests whether tasks derived from experience (rather than causal graph) suffice; Random task generation controls for task diversity; Random graph and correlation-based discovery ablate the causal structure and discovery method. The ablation (CTI without intervention) tests whether do-interventions are necessary for breaking tool-confounder correlations. This combination creates a falsifiable test: if CTI significantly outperforms Self-Challenging and the relevant ablations on tasks where causal reasoning is needed, the claim that causal task intervention improves causal understanding is supported.

### Expected outcome and causal chain

**vs. Original Llama-3.1-8B-Instruct** — On a case where the tool effect confounds a causal relation (e.g., tool output correlates with target variable V), the baseline fails to achieve the counterfactual state because it relies on shallow tool-use patterns learned from pretraining. Our method instead generates tasks requiring intervention on V that breaks the spurious correlation, so we expect a large gap (e.g., 30%+ absolute success rate improvement) on tasks that require decoupling tool from target.

**vs. Self-Challenging (Llama-3.1-8B)** — On a case where the agent's past experience overrepresents tool-specific correlations (e.g., always using a specific tool to achieve V), Self-Challenging reproduces the same biased tasks because it generates tasks from its own traces. Our method, using a learned causal graph, selects intervention values that are counterfactual under the current policy, so we expect a moderate gap (e.g., 15-20% improvement) on tasks that require causal reasoning orthogonal to tool-use patterns.

**vs. Random task generation** — On a case where a random task is either too easy (tool-derived) or too hard (causally disconnected), the baseline either fails to improve causal understanding or generates impossible tasks. Our CTI filters tasks by causal connectivity and ensures they are feasible yet require causal reasoning. Thus, we expect CTI to show higher success rate on the filtered task subset (perhaps 10% higher) while random tasks yield lower overall success due to infeasibility.

**vs. CTI with random graph** — On a case where the causal graph is random, intervention targets may not correspond to true causal mechanisms, so performance degrades. We expect CTI with learned graph to outperform random graph by 10-15% on tasks requiring genuine causal reasoning.

**vs. CTI with correlation-based discovery** — On a case where thresholding misses weak causal edges, the graph is incomplete; CTI with PC algorithm should outperform by 5-10% on tasks requiring those edges.

### What would falsify this idea

If CTI's advantage over Self-Challenging is uniform across all tasks rather than concentrated on tasks that require breaking tool-effect correlations (e.g., success improved equally on tool-derived simple tasks), then the claim that CTI specifically improves causal reasoning is false. Additionally, if the Graph SHD is high (>0.5) and CTI still performs well, the causal discovery assumption is not critical, undermining the motivation.

## References

1. Self-Challenging Language Model Agents
2. Proposer-Agent-Evaluator(PAE): Autonomous Skill Discovery For Foundation Model Internet Agents
3. Building Math Agents with Multi-Turn Iterative Preference Learning
4. You Only Look at Screens: Multimodal Chain-of-Action Agents
5. Bootstrap Your Own Skills: Learning to Solve New Tasks with Large Language Model Guidance
6. ProgPrompt: Generating Situated Robot Task Plans using Large Language Models
7. OPT: Open Pre-trained Transformer Language Models
