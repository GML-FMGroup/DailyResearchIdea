# Temporal Axioms for Evaluating Reasoning Dynamics in LLM Representations

## Motivation

The static axiomatic framework from 'Formalizing Latent Thoughts' evaluates representation properties like separability and stability but ignores temporal dynamics of reasoning flows. This is a structural gap because reasoning is inherently sequential; static metrics cannot capture how representations evolve across reasoning steps, leading to failures in distinguishing instance-level reasoning paths. We need temporal metrics that quantify continuity, velocity, and direction consistency of representation trajectories during token-by-token generation.

## Key Insight

The continuity and velocity of representation trajectories during autoregressive generation provide a measure of reasoning flow coherence that is independent of output accuracy, because they capture the smoothness and rate of representational change that underlies sequential reasoning.

## Method

**(A) What it is:** Temporal Representation Axioms (TRA) is a framework that evaluates reasoning dynamics by computing three metrics—continuity, velocity, and direction consistency—from the sequence of last-layer hidden states during autoregressive generation. Input: list of hidden vectors for each token in a reasoning trace. Output: three scalar scores. 

**(B) How it works:**
```python
def compute_temporal_metrics(layer_states):
    """
    layer_states: list of hidden vectors from the last transformer layer,
                  one per token in the reasoning trace (including intermediate CoT tokens).
    """
    # Continuity: average cosine similarity between consecutive states (hyperparameter: use cosine)
    cos_sims = [cosine_sim(layer_states[i], layer_states[i+1]) for i in range(len(layer_states)-1)]
    continuity = np.mean(cos_sims)
    
    # Velocity: average L2 norm of state differences (hyperparameter: use L2 norm)
    diffs = [np.linalg.norm(layer_states[i+1] - layer_states[i]) for i in range(len(layer_states)-1)]
    velocity = np.mean(diffs)
    
    # Direction consistency: average cosine similarity between consecutive difference vectors (hyperparameter: use cosine of diffs)
    diff_vectors = [layer_states[i+1] - layer_states[i] for i in range(len(layer_states)-1)]
    dir_consistency = np.mean([cosine_sim(diff_vectors[i], diff_vectors[i+1]) for i in range(len(diff_vectors)-1)])
    
    return {"continuity": continuity, "velocity": velocity, "dir_consistency": dir_consistency}
```

**(C) Why this design:** We chose cosine similarity over Euclidean distance for continuity because it normalizes magnitude, making the metric invariant to overall scaling of representations; this focus on angular change accepts the cost that continuity ignores changes in representational magnitude, which may be informative for some reasoning patterns. We chose L2 norm for velocity over alternative measures like cross-entropy change because it directly quantifies the geometric displacement in representation space, capturing the amount of representational shift per token; the trade-off is that L2 norm conflates directional and magnitude changes. We included direction consistency as a third metric because it captures whether the reasoning trajectory maintains a coherent direction over multiple steps, complementing the pairwise view of continuity and velocity; the cost is added complexity and sensitivity to noise in difference vectors. These three metrics together provide a multidimensional view of temporal dynamics that the static axioms lack. 

**(D) Why it measures what we claim:** Continuity measures the smoothness of reasoning flow because high cosine similarity between consecutive states implies that the representation evolves gradually, which assumes that reasoning progresses through small incremental updates; this assumption fails when the model makes large leaps or jumps between distinct latent regions (e.g., backtracking), in which case continuity reflects abruptness rather than smoothness. Velocity measures the speed of representational change per token, which approximates the rate of information acquisition under the assumption that each token contributes roughly equal representational shift; this assumption fails when tokens have varying semantic weight (e.g., a conjunction vs. a content word), causing velocity to reflect token-level variation rather than true reasoning speed. Direction consistency measures the coherence of the reasoning trajectory because high similarity between consecutive difference vectors indicates that the representation moves in a persistent direction; this assumes that reasoning follows a linear path, which fails when the model revisits previous states or oscillates, causing direction consistency to capture oscillation. Together, these metrics operationalize the concept of temporal reasoning dynamics by grounding it in geometric properties of the trajectory, explicitly naming the assumptions that link the computational quantities to the intended constructs.

## Contribution

(1) Introduction of three temporal axioms—continuity, velocity, and direction consistency—for evaluating reasoning dynamics in LLM representations, complementing the static axiomatic framework. (2) Empirical evidence that temporal metrics provide orthogonal signals to static metrics, revealing that models with high static separability may still exhibit disjointed reasoning flows. (3) Open-source implementation and standardized evaluation protocols for temporal representation analysis on chain-of-thought reasoning tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | 23 reasoning tasks | Covers diverse reasoning patterns |
| Primary metric | Spearman correlation with accuracy | Quantifies association with reasoning |
| Baseline | Random hidden states | Tests if structure is meaningful |
| Baseline | Input embedding states | Tests if deeper representations matter |
| Baseline | Static axiom metrics | Compare against representation quality |
| Ablation of ours | Remove direction consistency | Isolate contribution of direction |

### Why this setup validates the claim

The chosen combination of dataset, metric, baselines, and ablation forms a falsifiable test of the central claim that TRA captures reasoning dynamics beyond static properties. The 23 reasoning tasks provide a diverse set of traces that include both coherent and incoherent reasoning patterns. Using Spearman correlation with accuracy directly links TRA scores to reasoning success, testing whether temporal dynamics predict quality. The random hidden states baseline checks that observed correlations are not due to chance or random fluctuations; the input embedding states baseline tests whether hidden layer evolution adds value over static input similarities; the static axiom metrics baseline isolates the temporal dimension against known representation quality measures. The ablation of direction consistency isolates the contribution of the third metric, testing whether it captures trajectory coherence beyond continuity and velocity. This design ensures that if TRA correlates equally well across all baselines, the claim fails; but if TRA outperforms on subsets where temporal dynamics are critical, it validates the framework.

### Expected outcome and causal chain

**vs. Random hidden states** — On a reasoning trace that involves backtracking (e.g., a model corrects itself mid-CoT), random hidden states produce TRA metrics indistinguishable from random noise because they lack structure. Our method instead detects a drop in continuity (lower cosine similarity) when the model jumps back to an earlier state, capturing the abrupt shift. We expect a noticeable gap in Spearman correlation with accuracy on traces with self-corrections (high vs. low continuity) compared to random baselines.

**vs. Input embedding states** — On a trace where the model makes a subtle logical deduction (e.g., from "is a mammal" to "has lungs"), input embeddings remain static (same token type) while hidden states evolve significantly. The input embedding baseline thus misses this representational change, yielding low velocity values that do not reflect reasoning progress. Our method captures the increased velocity (L2 norm of hidden state differences) during the deduction step. We expect TRA velocity to correlate with accuracy on such deductive steps, while input embeddings show no correlation.

**vs. Static axiom metrics** — On a trace where the model oscillates between two competing hypotheses (e.g., vacillating between answers), static axioms like representational consistency may be high because each state is internally consistent, but the trajectory is incoherent. Our method's direction consistency (cosine similarity of consecutive difference vectors) becomes low, reflecting the oscillation. In contrast, static axioms would erroneously indicate high quality. We expect TRA direction consistency to correlate negatively with accuracy on such oscillating traces, while static axioms show positive or no correlation.

### What would falsify this idea

If TRA metrics show no significant Spearman correlation with accuracy across the 23 tasks, or if the correlation is uniform across all subsets (e.g., equally strong on simple vs. backtracking tasks) rather than concentrated on tasks where temporal dynamics are predicted to matter (e.g., self-correction, multi-step deduction, oscillation), then the claim that TRA captures reasoning dynamics is falsified.

## References

1. Formalizing Latent Thoughts: Four Axioms of Thought Representation in LLMs
