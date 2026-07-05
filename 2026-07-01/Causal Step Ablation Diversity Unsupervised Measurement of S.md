# Causal Step Ablation Diversity: Unsupervised Measurement of Strategic Variation in Mathematical Reasoning

## Motivation

Existing diversity metrics for mathematical reasoning rely on human-annotated LLM judges (e.g., 'Are We Measuring Strategy or Phrasing?') or semantic embeddings (e.g., DQO), both of which are expensive to calibrate and can be exploited by RL policies that learn to generate superficially diverse solutions (e.g., varying phrasing while keeping the same strategy). The core structural problem is that these metrics measure surface forms rather than causal reasoning structure, allowing policies to game the metric without genuinely expanding the set of strategies.

## Key Insight

The causal importance of individual reasoning steps—measured by the effect of ablating each step on the final answer—provides an invariant signature of a solution's strategy that cannot be manipulated without genuinely changing the reasoning approach.

## Method

### (A) What it is
**Causal Step Ablation Diversity (CSAD)** is an unsupervised metric that, given a set of solutions to a math problem, quantifies strategic diversity by comparing the sets of causally important steps across solutions. It does not require human annotations or learned models.

### (B) How it works
```python
def compute_causal_step_importance(solution_steps: List[str], final_answer: str, model: Callable) -> Dict[int, float]:
    """
    For each step index i, compute the causal importance I_i as the change in probability of reaching final_answer
    when step i is ablated (replaced with a neutral placeholder).
    """
    base_prob = model.get_probability_of_answer(final_answer, steps=solution_steps)
    importance = {}
    for i in range(len(solution_steps)):
        ablated_steps = solution_steps[:i] + ["[NEUTRAL]"] + solution_steps[i+1:]
        ablated_prob = model.get_probability_of_answer(final_answer, steps=ablated_steps)
        importance[i] = base_prob - ablated_prob  # positive means step is causally important
    return importance

def compute_CSAD(solutions: List[List[str]], model: Callable) -> float:
    """
    Input: list of solutions (each is a list of step strings) for the same problem.
    Output: diversity score = average pairwise Jaccard distance between sets of top-k important steps.
    """
    k = max(1, min(len(soln)//2 for soln in solutions))  # hyperparameter: fraction of steps
    sets = []
    for soln in solutions:
        imp = compute_causal_step_importance(soln, final_answer, model)
        top_k = set(sorted(imp, key=imp.get, reverse=True)[:k])
        sets.append(top_k)
    # pairwise Jaccard distance
    distance_sum = 0
    count = 0
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            intersection = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            distance_sum += 1 - (intersection / union)
            count += 1
    return distance_sum / count if count > 0 else 0.0
```

### (C) Why this design
We chose step-level causal importance over entire solution embeddings because embeddings conflate surface phrasing with strategic structure (as shown in 'Are We Measuring Strategy or Phrasing?'), accepting the cost that we require a model capable of answering counterfactual queries (e.g., probability of answer given ablated steps). We chose Jaccard distance on top-k important steps rather than full distribution comparison (e.g., KL on importance vectors) because the top-k set robustly captures the core strategy while ignoring steps that have small individual causal effect, accepting the cost that we lose information about differential importance weights. We set k as half the steps (hyperparameter) as a default, balancing between capturing meaningful structure and ignoring noise; this design choice may be suboptimal for very long solutions where only a few steps matter. We chose to ablate with a neutral placeholder rather than removing the step entirely, because removing steps may break coherence—accepting the cost that the neutral placeholder might inadvertently introduce new meaning.

### (D) Why it measures what we claim
`compute_causal_step_importance` measures the causal necessity of each step for a solution to reach the final answer because it quantifies the drop in answer probability when the step is replaced; this assumes the model's counterfactual probability reflects true causal dependence. When the model fails to compute accurate counterfactuals (e.g., if it is not faithful or the placeholder changes the reasoning path), then importance instead measures the model's sensitivity to surface perturbations, not true causal role. `top-k set` measures the set of steps that are most causally necessary across a solution; this assumes that a solution's strategy is characterized by which steps are indispensable. When the strategy is distributed across many equally important steps (e.g., gradual refinement), the top-k set may miss diffuse structure, instead reflecting only the most fragile steps. Finally, `Jaccard distance between top-k sets` measures the fraction of causally necessary steps that are not shared between two solutions; this assumes that strategy diversity is proportional to dissimilarity of essential steps. When two solutions use different steps but the same logical approach (e.g., different algebraic manipulations to isolate the same variable), Jaccard distance overstates diversity, falsely seeing difference in implementation as difference in strategy.

## Contribution

(1) A fully unsupervised causal intervention metric (CSAD) for measuring strategic diversity in mathematical reasoning without human annotations or learned judges. (2) A design principle: causal step importance provides a robust signal that RL policies cannot exploit because it directly probes the reasoning structure; optimizing it requires genuinely diverse strategies. (3) An empirical demonstration that CSAD correlates positively with human-defined approach-level diversity and that using it as a reward signal in RLVR preserves strategic diversity while avoiding the collapse observed with surface-based rewards.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | GSM8K | Standard benchmark for math reasoning. |
| Primary metric | Spearman ρ with human judgments | Direct validation of diversity measurement. |
| Baseline: Surface | ROUGE-L F1 | Naive surface-level baseline. |
| Baseline: Embedding | Sentence-BERT cosine similarity | Semantic similarity baseline. |
| Baseline: Prior metric | RPD (Reasoning Path Divergence) | Prior diversity metric for comparison. |
| Ablation | Remove step instead of neutral placeholder | Tests necessity of placeholder. |

### Why this setup validates the claim
This combination creates a falsifiable test. Dataset GSM8K provides diverse math problems with multiple solution strategies. The primary metric (Spearman correlation with human judgments) directly tests whether CSAD captures human-perceived strategic diversity, not just surface similarity. Baselines serve as controls: ROUGE-L checks if simple lexical overlap suffices; Sentence-BERT tests whether semantic embedding distance captures strategy; RPD tests a prior diversity metric. The ablation (removing steps) isolates the effect of the neutral placeholder—if CSAD relies on maintaining coherence, then removing steps should lower correlation. The setup ensures that if CSAD outperforms all baselines on correlation, the claim is supported; if not, it is falsified.

### Expected outcome and causal chain

**vs. ROUGE-L** — On a case where two solutions use synonyms or reorder steps, ROUGE-L (lexical overlap) yields low similarity, falsely indicating diversity because it ignores semantic equivalence. Our method instead identifies causally important steps; synonyms do not change causal role, so CSAD correctly sees high strategic similarity. We expect CSAD to show ρ > 0.6 vs. human ratings, while ROUGE-L correlations hover around ρ = 0.2–0.3 on such cases, producing a noticeable gap overall.

**vs. Sentence-BERT cosine** — On a case where two solutions differ in phrasing but follow identical logical steps (e.g., "add 5 to both sides" vs "subtract -5"), Sentence-BERT embeddings may assign medium similarity due to lexical mismatch, conflating phrasing with approach. CSAD isolates causal steps; if both solutions mention the same indispensable step (e.g., isolating x), the top-k sets overlap. We predict CSAD correlates ρ > 0.7 with human judgments, while Sentence-BERT ρ ≤ 0.5 on these specific instances.

**vs. RPD** — RPD measures divergence of next-token probabilities or reasoning paths. On problems where two solutions diverge early but converge late (e.g., different algebraic paths leading to same equation), RPD may inflate diversity because it focuses on early path differences. CSAD looks only at causally important steps—if both converge to the same key step, Jaccard distance remains low. We expect CSAD to agree with humans that such solutions have similar strategy (ρ > 0.6), while RPD shows low agreement (ρ < 0.4) on such convergent cases.

### What would falsify this idea
If CSAD's correlation with human judgments is no better than the best baseline (e.g., RPD) across diverse subsets, or if the gain is uniform across all problem types rather than concentrated on cases where surface and semantic metrics fail (e.g., synonyms, reordering, convergent paths), then the central claim that causal step importance captures strategic diversity is refuted.

## References

1. Are We Measuring Strategy or Phrasing? The Gap Between Surface- and Approach-Level Diversity in LLM Math Reasoning
2. Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)
3. Reasoning Path Divergence: A New Metric and Curation Strategy to Unlock LLM Diverse Thinking
4. Post-training Large Language Models for Diverse High-Quality Responses
5. Diversity-Incentivized Exploration for Versatile Reasoning
6. GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time
7. Benchmarking Linguistic Diversity of Large Language Models
8. One fish, two fish, but not the whole sea: Alignment reduces language models' conceptual diversity
