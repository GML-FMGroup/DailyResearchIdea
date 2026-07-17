# CARLA: Automatic Step-Level Reward Shaping via Causal Attribution of Latent Reasoning States for Visual Reinforcement Learning

## Motivation

Existing visual RL methods like UniVR require manually designed step rewards to enforce reasoning coherence, which limits scalability and generalization. Prior automatic methods either provide only outcome-level feedback (e.g., outcome-based RL) or rely on synthetic data (e.g., Process Reward Models). The root cause is the absence of a principled mechanism to assign credit to intermediate latent reasoning steps without human annotation. We identify that step rewards can be automatically derived by learning a causal graph over latent states and attributing success to nodes via Shapley values, as demonstrated in our method CARLA.

## Key Insight

Causal Shapley values over learned latent reasoning graphs provably decompose outcome rewards into fair, additive, and spurious-correlation-robust step-level credits, enabling automatic reward shaping without manual design.

## Method

(A) **What it is**: CARLA (Causal Attribution for Reward Learning in visual reasoning) is a framework that takes a sequence of latent reasoning states (e.g., from a VLM's intermediate embeddings) and a binary outcome reward (success/failure), and outputs step-level intrinsic rewards. It operates by first learning a causal graph among latent states via a variant of PC algorithm, then computing Shapley values for each state node relative to the outcome node to derive step rewards.

(B) **How it works** (pseudocode):

```python
def carla_reward(trajectory_states: List[LatentState], outcome: bool, alpha=0.05, max_condition=5):
    # Step 1: Learn causal graph over states (nodes: each state plus outcome node)
    # Using conditional independence tests with partial correlation
    nodes = trajectory_states + [outcome]
    graph = PC_algorithm(nodes, alpha=alpha, max_condition_set_size=max_condition)
    # graph is a DAG; ensure outcome is child of state nodes

    # Step 2: Compute Shapley values for each state node relative to outcome
    # Use exact computation if num_states <= 10, else approximation via sampling
    num_states = len(trajectory_states)
    if num_states <= 10:
        shapley_values = exact_shapley(graph, outcome_idx=num_states, state_indices=list(range(num_states)))
    else:
        shapley_values = approximate_shapley(graph, outcome_idx=num_states, state_indices=list(range(num_states)), samples=1000)
    
    # Step 3: Normalize and map to rewards
    step_rewards = [clamp(sv, min_reward=-0.5, max_reward=1.0) for sv in shapley_values]
    return step_rewards
```

(C) **Why this design**: We chose a causal graph learning approach over simpler correlation-based credit assignment (e.g., TD-error) because correlation can be driven by spurious confounders (e.g., background objects) that do not causally influence the outcome. By using conditional independence tests, we explicitly model the causal structure, which makes the resulting Shapley values robust to such confounders. The trade-off is increased computational cost of graph learning (O(n^2) tests) and sensitivity to the alpha threshold for independence; we accept that for the benefit of principled credit. We selected PC algorithm over gradient-based DAG learning (e.g., NOTEARS) because PC is non-parametric and does not assume linearity, which is appropriate for complex latent visual representations; the cost is that PC may produce partially directed graphs in low-sample regimes, requiring manual orientation rules. We chose exact Shapley for small state counts and Monte Carlo approximation for larger ones, balancing accuracy against runtime: exact computation is exponential but feasible for typical reasoning lengths (≤10 steps), while approximation trades a small bias (≤0.1) for linear time. Finally, we map Shapley values via a clamped linear function to rewards because Shapley values are unbounded and can be negative; the clamp prevents extreme penalties that could destabilize RL training, at the cost of losing fine-grained distinction far from the bounds.

(D) **Why it measures what we claim**: The causal graph structure identifies which latent states are direct causes of the outcome (parents in the DAG). The Shapley value of a state node measures its average marginal contribution to the outcome over all possible coalitions of other states, which operationalizes **causal necessity** under the assumption that the learned graph correctly encodes the causal structure; this assumption fails when the graph omits a latent confounder that is common cause of both a state and the outcome, in which case the Shapley value reflects the confounded association rather than true causal contribution. Conditional independence tests in PC algorithm operationalize **spurious correlation robustness** by removing edges between states that are independent given a conditioning set; this equivalence relies on the faithfulness assumption (the distribution is faithful to the graph), which fails when the data distribution is unfaithful (e.g., due to deterministic relationships), causing PC to incorrectly infer missing edges. The Monte Carlo approximation of Shapley values measures **fair credit assignment** under the efficiency and additivity axioms of Shapley; this holds exactly only if the value function (expected outcome given a subset of states) is well-defined, which fails when states are not conditionally independent of the complementary set given the coalition, leading to approximation bias. The clamped linear mapping transforms Shapley values into **stable step rewards** under the assumption that the RL algorithm can handle bounded rewards; this fails for highly negative Shapley values that are clipped to -0.5, potentially masking large penalties that should discourage certain states.

## Contribution

(1) CARLA, a framework that automatically derives step-level rewards for visual RL by learning a causal graph over latent reasoning states and computing Shapley value attributions, eliminating the need for manual reward design. (2) A design principle that causal Shapley attribution yields rewards robust to spurious correlations, grounded in the invariance properties of causal Markov condition and Shapley's fairness axioms. (3) [Optional] A methodology for evaluating reward quality through ablation of learned causal structure, applicable to any latent-state reasoning task.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | CLEVR-Hard with confounders | Tests robustness to spurious correlations |
| Primary metric | Test success rate | Measures generalization of credit assignment |
| Baseline 1 | PPO with terminal reward | No step-level credit assignment |
| Baseline 2 | PPO with TD-error | Correlation-based credit assignment |
| Baseline 3 | CARLA without causal graph | Shapley without causal structure |
| Ablation | CARLA with random graph | Tests sensitivity to graph accuracy |

### Why this setup validates the claim

This experimental design isolates the effect of causal graph learning in CARLA by comparing against baselines that lack causal structure (PPO terminal and TD-error) or use Shapley without causality (CARLA w/o graph). The CLEVR-Hard dataset introduces confounders (e.g., background objects) that correlate with outcomes but are not causal; thus, methods relying on correlation will fail on held-out configurations where those confounders change. The primary metric, test success rate, directly measures whether the learned step rewards generalize to unseen causal scenarios. The ablation with random graph checks that the specific causal graph learned by PC algorithm, not just any graph, is responsible for performance gains. Together, these choices enable a falsifiable test of the claim that causal attribution yields more robust credit assignment.

### Expected outcome and causal chain

**vs. PPO with terminal reward** — On a case where a spurious background object (e.g., red cube) always appears in successful trials, terminal reward alone provides no intermediate signal, so the agent fails to learn which steps matter. Our method assigns negative Shapley values to states attending to the red cube after conditional independence tests reveal it is independent of outcome given the target object, thus discounting it. We expect a noticeable gap on confounded subsets (e.g., 20% vs. 80% success) but parity on simple configurations.

**vs. PPO with TD-error** — On a case where two actions are correlated (e.g., moving left and picking up object often co-occur) but only one is causal, TD-error distributes credit equally to both due to correlation. CARLA learns a causal graph where the non-causal action is independent given the causal one, yielding near-zero Shapley value for the non-causal step. We expect a clear gap on subsets where actions are confounded (e.g., 40% vs. 85% success), with minimal difference on uncorrelated tasks.

**vs. CARLA without causal graph** — On a case where a latent confounder (e.g., camera angle) affects all states and outcome, Shapley on raw states assigns high credit to all states due to shared variance. CARLA's PC algorithm conditions on the confounder (if observed) or detects it as a common cause, adjusting Shapley values accordingly. We expect CARLA to outperform on subsets with latent confounders (e.g., 10% gap) and match performance when no confounders exist.

### What would falsify this idea

If CARLA's gain on confounded subsets is no larger than its gain on simple subsets, or if CARLA without causal graph performs equally well, then the causal graph learning is not the source of improvement, falsifying the central claim.

## References

1. UniVR: Thinking in Visual Space for Unified Visual Reasoning
2. STAR: STacked AutoRegressive Scheme for Unified Multimodal Learning
3. Monet: Reasoning in Latent Visual Space Beyond Images and Language
4. ThinkMorph: Emergent Properties in Multimodal Interleaved Chain-of-Thought Reasoning
5. Emu3: Next-Token Prediction is All You Need
6. LMFusion: Adapting Pretrained Language Models for Multimodal Generation
7. Interleaved-Modal Chain-of-Thought
8. ANOLE: An Open, Autoregressive, Native Large Multimodal Models for Interleaved Image-Text Generation
