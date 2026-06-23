# Causal Retrieval Agent Memory: Selecting Episodic Memories via Transfer Entropy for Improved Generalization

## Motivation

Current memory retrieval methods in LLM agents, such as the multi-dimensional similarity in Trajectory-Informed Memory Generation and the keyword-based linking in A-MEM, rely on surface-level feature matching. This fails when tasks share causal structure but differ superficially, because retrieved memories often lack causal relevance to the current situation. The root cause is that similarity in attribute space does not imply similarity in causal influence patterns, leading to interference rather than transfer.

## Key Insight

Transfer entropy provides a non-parametric measure of directed information flow that is invariant under surface transformations, enabling retrieval of memories with the same causal influence patterns even when their observable features are unrelated.

## Method

### (A) What it is
CRAM (Causal Retrieval Agent Memory) computes a causal signature for each episodic memory using transfer entropy (TE) from actions to outcomes, then retrieves memories with the most similar causal signatures for a given context.

### (B) How it works
```python
# CRAM: Causal Retrieval Agent Memory
# Input: Memory bank M = { (actions_i, outcome_i) }, new partial context (A_cur, O_cur)
# Hyperparameters: delay τ = 1, k = 3, TE estimator = Kraskov-Stögbauer-Grassberger

def compute_causal_signature(actions, outcome, τ=1):
    # actions: (T, num_action_dims), outcome: (T,)
    causal_vector = []
    for dim in range(actions.shape[1]):
        te = transfer_entropy(actions[:, dim], outcome, delay=τ, estimator='ksg')
        causal_vector.append(te)
    return normalize(causal_vector)  # unit L2 norm

# Offline: compute signatures for all memories
for mem in M:
    mem.causal_sig = compute_causal_signature(mem.actions, mem.outcome, τ=1)

# Online: retrieve top-k memories
def retrieve(A_cur, O_cur, M, k=3):
    query_sig = compute_causal_signature(A_cur, O_cur, τ=1)
    scores = [cosine_similarity(query_sig, mem.causal_sig) for mem in M]
    top_k = sorted(range(len(M)), key=lambda i: scores[i], reverse=True)[:k]
    return [M[i] for i in top_k]
```

### (C) Why this design
We chose transfer entropy over Granger causality or structural equation models because it is non-parametric and captures nonlinear dependencies, accepting a higher computational cost per memory. We compute a per-action-dimension causal vector rather than a single scalar TE to preserve which action dimensions are causally relevant, at the cost of increased dimensionality. We normalize the signature to unit length to focus on pattern shape rather than magnitude, accepting that two memories with the same relative causal strengths but different absolute TE values are treated as identical. We use cosine similarity instead of Euclidean distance to avoid sensitivity to overall TE magnitude, which may vary with trajectory length; the trade-off is ignoring absolute causal strength. We delay the outcome by 1 timestep to account for time lag in causation, accepting that the true delay may vary but a fixed delay is simpler. Unlike prior memory retrieval methods that use surface similarity (e.g., A-MEM's attribute linking or Trajectory-Informed Memory's multi-dimensional similarity), CRAM explicitly models causal influence, enabling retrieval that is robust to superficial variability.

### (D) Why it measures what we claim
The computational quantity `causal_vector` (vector of per-action-dimension TE from actions to outcome) measures the directed influence of each action type on the outcome because TE quantifies the reduction in uncertainty of the outcome given the past of that action, beyond the outcome's own past; this assumption fails when trajectories are too short (e.g., <20 steps) in which case TE estimates become biased and may reflect noise rather than true causality. The query causal signature similarly measures the causal influence pattern of the current context, assuming the partial trajectory is representative; this assumption fails when the current outcome is not observed or incomplete, in which case we cannot compute TE and must fall back to a heuristic, potentially retrieving irrelevant memories. The cosine similarity between query and memory signatures measures similarity in causal influence patterns because we normalize to unit vectors, making the angle correspond to the Pearson correlation of the causal patterns; this assumption fails when two memories have proportionally scaled causal vectors but different absolute strengths (e.g., one strongly causal, one weakly), in which case similarity overestimates relevance.

## Contribution

(1) A novel episodic memory retrieval mechanism that indexes memories by causal influence patterns computed via transfer entropy, enabling retrieval of causally relevant experiences irrespective of surface similarity. (2) Empirical demonstration that causal-based retrieval outperforms similarity-based retrieval on tasks with latent causal structure, showing improved generalization and decision-making efficiency. (3) A benchmark suite of causal reasoning agent tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | BabyAI (multi-task) | Diverse tasks with clear actions/outcomes |
| Primary metric | Task success rate | Direct measure of downstream utility |
| Baseline 1 | A-MEM | Attribute-based memory retrieval |
| Baseline 2 | TIM | Multi-dimensional similarity retrieval |
| Ablation of ours | CRAM w/o normalization | Isolate effect of normalization |

### Why this setup validates the claim
This experimental design directly tests the central claim that causal signatures improve memory retrieval for agents. By comparing against A-MEM (attribute linking) and TIM (multi-dimensional surface similarity), we isolate the effect of modeling causal influence. The ablation (removing normalization) tests whether normalization is essential. The primary metric, task success rate, measures whether retrieved memories actually help the agent complete tasks. The BabyAI dataset includes varied tasks where causal patterns differ from surface features, so if CRAM's causal signatures capture true relevance, it should outperform on tasks where superficial similarity is misleading. This setup is falsifiable: if CRAM only matches baselines on simple tasks, the causal claim is unsupported.

### Expected outcome and causal chain

**vs. A-MEM** — On a case where two different action types (e.g., "pick up" vs "open") both lead to the same outcome ("apple acquired") but have different causal roles, A-MEM retrieves memories based on attribute similarity like object type, so it retrieves irrelevant "open" memories when the context involves "pick up". Our method's causal signature captures the directed influence of each action dimension, so it retrieves only causally relevant memories. Thus we expect a noticeable gap on tasks with many causally distinct action types (e.g., +10% success rate) but parity on simple tasks with few actions.

**vs. TIM** — On a case where two trajectories have similar length and action sequence but different causal dependencies (e.g., one requires precise timing of a key action, the other does not), TIM's multi-dimensional similarity over temporal features retrieves the wrong memory because it ignores causality. CRAM uses transfer entropy to measure how much each action reduces uncertainty in the outcome, so it identifies the correct causal pattern. We expect CRAM to outperform on tasks with non-stationary transitions (e.g., +15% on those tasks) while performing similarly on stationary ones.

### What would falsify this idea
If CRAM shows uniform improvement across all task types rather than being concentrated on tasks where superficial similarity is misleading (e.g., tasks with high action-outcome variability), then the causal signature is not the driving factor; the improvement would be due to other confounding factors like increased representation size.

## References

1. Trajectory-Informed Memory Generation for Self-Improving Agent Systems
2. Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models
3. A-MEM: Agentic Memory for LLM Agents
4. Get Experience from Practice: LLM Agents with Record & Replay
5. StreamBench: Towards Benchmarking Continuous Improvement of Language Agents
6. TextGrad: Automatic "Differentiation" via Text
7. Agent Workflow Memory
8. AIOS: LLM Agent Operating System
