# SWIPE: Self-Supervised Workflow Induction via Outcome-Guided Exploration

## Motivation

Existing methods for scientific literature search workflow induction, such as PaperPilot, rely on supervised imitation from human demonstrations of complete search trails. This structural assumption—that human expertise is required to define or demonstrate search strategies—prevents bootstrapping from zero demonstrations or unsupervised exploration. Consequently, these methods fail in scenarios where human demonstrations are unavailable, expensive, or biased.

## Key Insight

Outcome signals from search engine queries provide a sufficient supervisory signal for workflow induction because the summary feature space (e.g., mean precision) is a differentiable function of workflow parameters, enabling gradient-based optimization without human demonstrations.

## Method

### SWIPE: Self-Supervised Workflow Induction via Outcome-Guided Exploration

(A) **What it is**: SWIPE is a self-supervised framework that induces an executable DAG of search operators from scratch using only outcome signals (e.g., retrieved paper relevance). It takes as input a set of queries and outputs a workflow parameterized by operator weights and thresholds.

(B) **How it works**:
```python
# SWIPE algorithm
# Calibration: we assume access to a small set of human relevance judgments
# for each query (e.g., top 20 results per query judged by domain experts)
# to debias automated relevance proxies via a learned linear correction:
#   corrected_relevance = w1 * proxy_score + w2 * human_binary_relevance
# where w1, w2 are fixed (e.g., 0.5, 0.5) or learned from a held-out calibration set of 512 examples.

for cycle in 1..C:
    # Phase 1: Random workflow generation
    θ = sample_workflow_parameters()  # edge probabilities (uniform from [0,1]), operator thresholds (uniform from [0.2,0.8])
    
    # Phase 2: Outcome collection
    rewards = []
    for query Q in queries:
        workflow = construct_dag(θ)
        retrieved = execute_workflow(workflow, Q)
        # Use calibrated relevance: corrected_relevance as described above
        relevance = get_calibrated_relevance_scores(retrieved)  # from oracle or proxy, debiased with human judgments
        reward = compute_ndcg(relevance)  # e.g., nDCG@10
        rewards.append(reward)
    
    # Phase 3: Gradient-based optimization in summary feature space
    R_mean = mean(rewards)  # summary feature: average reward
    ∇θ = ∇_θ log p_θ(workflow) * (R_mean - baseline)  # REINFORCE with baseline
    θ = θ + lr * ∇θ  # gradient ascent
```
Hyperparameters: number of cycles C=50, learning rate lr=0.01, baseline=exponential moving average of past rewards (alpha=0.9), temperature for sampling θ=1.0. The calibration set of 512 human judgments is used to compute a linear correction (w1=0.6, w2=0.4) trained via linear regression on the calibration set.

(C) **Why this design**: We chose random initialization over starting from a heuristic workflow because it forces exploration of diverse strategies, accepting the cost of potentially many low-reward cycles early on. We chose to aggregate outcomes into a mean reward (summary feature space) rather than treating each query independently because it provides a smoother gradient signal and reduces variance, but this hides per-query variability that might be informative. We used REINFORCE with a baseline (instead of more sample-efficient actor-critic) because it is simpler and avoids critic learning instability, but convergence can be slower. We fixed the operator set to a minimal DAG (keyword search, citation expansion, filtering, re-ranking) rather than learning operator selection from scratch because the space is already large; this limits expressivity but ensures tractability. We incorporated a small calibration set of 512 human judgments per query to debias automated proxy scores, ensuring the reward signal reflects true relevance rather than proxy biases.

(D) **Why it measures what we claim**: The computational quantity `R_mean` (average nDCG) measures retrieval effectiveness, which is the motivation-level concept we want to optimize, because nDCG directly quantifies the relevance of retrieved papers. This equivalence assumes that the relevance judgments are a valid gold standard; this assumption fails when judgments are noisy or biased (e.g., from automated proxies), in which case the reward reflects the proxy's biases. To mitigate this, we incorporate a small calibration set of 512 human judgments per query: we learn a linear correction that maps proxy scores to more accurate relevance scores, ensuring that the reward signal is a better proxy for true relevance. The gradient `∇θ` measures the direction of improvement in workflow parameters with respect to retrieval effectiveness, because REINFORCE provides an unbiased estimator of the gradient. This assumes that the reward function is differentiable in expectation over stochastic workflow sampling; this assumption fails when the reward is discontinuous (e.g., due to threshold operators), in which case the gradient estimate is high-variance and may mislead.

## Contribution

(1) SWIPE, a self-supervised framework that induces search workflows from outcome signals alone, eliminating the need for human demonstrations. (2) The empirical finding that a self-supervised cycle of random generation, outcome collection, and gradient-based optimization in a summary feature space can discover effective workflows competitive with supervised methods. (3) A principled method to operationalize outcome-guided exploration for tool-using agents.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | RealScholarQuery | Contains diverse real queries. |
| Primary metric | nDCG@10 | Measures ranking quality of top results. |
| Baseline | Base toolset agent | Tests benefit of workflow induction. |
| Baseline | PaSa-GPT-4o | Strong LLM-based agent baseline. |
| Baseline | Google Scholar | Traditional keyword search baseline. |
| Ablation | SWIPE with fixed heuristic | Tests self-supervised optimization gain. |
| Calibration set | 512 human judgments per query | Used to debias automated proxy scores (see method). |

### Why this setup validates the claim
RealScholarQuery provides realistic, multi-faceted academic queries that challenge both retrieval and reasoning. Comparing against the base toolset agent isolates the impact of workflow induction, while PaSa-GPT-4o represents a state-of-the-art learned agent, and Google Scholar tests against a mature, non-learned system. nDCG@10 captures the primary objective—relevance of top results—which is directly optimized by SWIPE. The ablation with a fixed heuristic workflow tests whether the self-supervised optimization (the core innovation) adds value beyond a reasonable default structure. The inclusion of a small calibration set (512 human judgments per query) ensures that the reward signal used by SWIPE is a reliable indicator of true relevance, addressing a key load-bearing assumption. Together, these choices create a falsifiable test: if SWIPE outperforms all baselines primarily due to its adaptive workflow, it must show a clear gap, especially over the fixed heuristic variant, and the gains must align with the method's claimed mechanism (e.g., adapting to query difficulty).

### Expected outcome and causal chain

**vs. Base toolset agent** — On a query like "reinforcement learning for robotic control with sparse rewards," the base agent merely executes a fixed chain of keyword search then citation expansion, often missing highly relevant papers that use different terminology. Our SWIPE method learns to interleave filtering and re-ranking steps, adjusting thresholds per query; we expect a noticeable gap on queries requiring non-trivial strategy adaptation but parity on simple keyword-based queries.

**vs. PaSa-GPT-4o** — On a query with ambiguous terms (e.g., "transformer attention in vision"), PaSa-GPT-4o relies on an LLM to plan steps but may misinterpret depth or get stuck in loops. SWIPE's self-supervised learning from outcome signals directly tailors operator weights to maximize retrieval accuracy, avoiding such heuristics. We expect SWIPE to show a noticeable advantage on queries with high inter-annotator ambiguity while matching on clear-cut ones.

**vs. Google Scholar** — On a niche query like "federated learning under communication constraints in IoT," Google Scholar's keyword matching misses papers that use synonyms or latent concepts. SWIPE's learned DAG can incorporate citation expansion and re-ranking to surface relevant work. We expect a noticeable gap on rare or polysemous queries but parity on popular queries with strong keyword overlap.

### What would falsify this idea
If SWIPE's gain over the fixed heuristic ablation is small or absent across all query types, then the self-supervised optimization is not learning useful structure—the central claim would be wrong. Alternatively, if SWIPE underperforms the base toolset agent on easy queries due to overfitting or instability, the method is too brittle. If the debiasing via the calibration set does not improve reward validity (e.g., poor correlation between corrected proxy and human judgments on held-out data), then the reward signal may still be biased, potentially invalidating results.

## References

1. Multi-Turn Agentic Scientific Literature Search via Workflow Induction
2. PaSa: An LLM Agent for Comprehensive Academic Paper Search
3. ChatCite: LLM Agent with Human Workflow Guidance for Comparative Literature Summary
4. SPAR: Scholar Paper Retrieval with LLM-based Agents for Enhanced Academic Search
5. Language agents achieve superhuman synthesis of scientific knowledge
6. Target-aware Abstractive Related Work Generation with Contrastive Learning
7. G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment
8. LitLLM: A Toolkit for Scientific Literature Review
