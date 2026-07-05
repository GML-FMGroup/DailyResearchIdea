# Causal Step Intervention (CSI): Unsupervised Approach-Level Diversity via Counterfactual Reasoning Step Swaps

## Motivation

Existing diversity metrics either rely on surface-level features (e.g., n-gram overlap) that miss strategic differences, or require human-calibrated LLM judges that are expensive and vulnerable to policy exploitation—as shown in 'Are We Measuring Strategy or Phrasing?' where the policy learns to game judge-specific preferences. A causal intervention on intermediate reasoning steps can reveal genuine approach-level variation because swapping a strategically critical step changes the final outcome, whereas surface variations do not, and this property is inherently resistant to exploitation since the policy would have to maintain correctness across all step variants.

## Key Insight

Step-level causal intervention isolates approach-level diversity because it measures the minimal difference in reasoning that alters the final answer, which surface metrics fail to capture and policies cannot systematically fake without collapsing correctness.

## Method

## (A) What it is
Causal Step Intervention (CSI) is an unsupervised metric that computes approach-level diversity between two solution trajectories by measuring the average change in final answer when replacing a step in one trajectory with the aligned step from the other, using the model’s own generation to simulate the counterfactual.

## (B) How it works
```python
def causal_step_intervention(traj_a: List[str], traj_b: List[str], model, problem: str, verifier) -> float:
    # Step 1: Align steps by semantic role using bipartite matching on sentence embeddings (Sentence-BERT).
    aligned_pairs = align_steps(traj_a, traj_b, model=embedder)
    total_effect = 0.0
    num_interventions = len(aligned_pairs)
    orig_final_a = extract_final_answer(generate_answer(model, problem, traj_a))
    for idx_a, idx_b in aligned_pairs:
        # Construct counterfactual prefix: replace traj_a[idx_a] with traj_b[idx_b]
        cf_prefix = "\n".join(traj_a[:idx_a]) + "\n" + traj_b[idx_b] + "\n"
        # Generate continuation from the model
        cf_full = generate_answer(model, problem, cf_prefix)
        cf_final = extract_final_answer(cf_full)
        # Measure effect: binary if verifier(s) differ
        effect = 1 if verifier(cf_final) != verifier(orig_final_a) else 0
        total_effect += effect
    diversity_score = total_effect / max(num_interventions, 1)
    return diversity_score
```
Hyperparameters: alignment threshold (0.7 cosine similarity), number of aligned pairs capped at min(len(traj_a), len(traj_b)).

## (C) Why this design
We designed CSI to overcome three key limitations of prior work. First, we choose step-level causal intervention over surface metrics (n-gram, embedding similarity) because surface metrics conflate phrasing with approach; causal effects isolate structural dependencies, accepting the computational cost of running multiple forward passes. Second, we use semantic alignment via bipartite matching rather than positional alignment, because steps may be reordered or omitted across solutions; this captures functional correspondence but introduces dependency on the embedding model (Sentence-BERT), which we accept for its off-the-shelf quality at the cost of potential domain mismatch. Third, we use a binary effect (verifier difference) instead of continuous probability changes, because binary outcomes are robust to model miscalibration—interventions that flip the answer are strong evidence of divergence, while small shifts may be noise; this trades granularity for reliability. Fourth, we average over multiple interventions per pair to reduce variance; the cost is linear in the number of aligned steps.

## (D) Why it measures what we claim
The average intervention effect over aligned steps measures approach-level diversity because it quantifies how much swapping a step from solution B into solution A changes the final outcome. The computational quantity `effect = 1 if verifier(cf_final) != verifier(orig_final) else 0` measures causal necessity of the step difference for correctness: if the step swap changes the answer, then that step carries approach-level information that is causally relevant. This assumption—that a change in answer implies a genuine difference in reasoning approach—fails when the step swap leads to an ambiguous continuation (e.g., model produces a random answer due to inconsistency), in which case the effect may reflect instability rather than divergence; we mitigate this by checking that the generated continuation passes a perplexity threshold (we use model perplexity < 1.5× baseline on training data). The alignment procedure `align_steps` operationalizes “corresponding steps” under the assumption that steps are functionally comparable if they have similar semantic content; this assumption fails when solutions use different high-level strategies that don't decompose into aligned steps, leading to few or no interventions, which we detect and flag as potential underestimation.

## Contribution

(1) A novel unsupervised metric for approach-level diversity, Causal Step Intervention (CSI), that uses counterfactual reasoning step swaps to measure divergence without human annotation. (2) The insight that causal effect of step intervention is robust to policy exploitation because the policy would need to maintain correctness across all step variants—an exponentially hard constraint. (3) A design principle for diversity measurement: leveraging the model's own generative capacity to simulate interventions, enabling scalability.

## Experiment

### Evaluation Setup
| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | MATH (Levels 1-5) | Wide range of reasoning complexity |
| Primary metric | Human-rated approach diversity | Direct measure of our target |
| Baseline | Standard RLVR (no diversity) | Tests effect of diversity reward |
| Baseline | Surface diversity reward (embedding) | Tests need for causal step intervention |
| Baseline | 1P1S training | Tests benefit of multiple solutions |
| Ablation-of-ours | CSI with positional alignment | Tests importance of semantic alignment |

### Why this setup validates the claim
This experimental design provides a falsifiable test of CSI's central claim that step-level causal intervention better captures approach diversity than surface metrics. MATH offers varied difficulty and multiple solution strategies, making it ideal to detect differences in diversity. Human-rated approach diversity directly measures the target construct. Comparing against standard RLVR isolates the effect of diversity reward. Surface diversity reward tests whether CSI's causal mechanism outperforms simple embedding similarity. 1P1S training assesses whether generating multiple solutions inherently yields diversity. The ablation of positional alignment tests whether semantic alignment is crucial. Together, these comparisons attribute any observed improvement to CSI's specific design choices.

### Expected outcome and causal chain

**vs. Standard RLVR (no diversity)** — On a problem where two solutions share surface phrasing but differ in key reasoning steps, standard RLVR treats them identically, failing to reward diversity. Our method identifies causal step differences via answer changes, so it rewards such pairs. We expect CSI-based RLVR to achieve a noticeable gap in human-rated diversity, especially on problems with multiple valid approaches.

**vs. Surface diversity reward (embedding)** — On a case where two solutions use different wording but identical reasoning (e.g., synonyms), embedding similarity may falsely flag them as diverse, introducing noise. CSI's causal intervention checks whether swapping a step actually changes the answer, avoiding false positives. Thus our method should produce more genuinely diverse sets, with higher correlation to human judgments.

**vs. 1P1S training** — On a problem with only one common solution, 1P1S produces a single trajectory; our method generates multiple, but without diversity reward they may be similar. CSI's reward encourages exploring different approaches even for simple problems. We expect our method to yield greater diversity on problems with multiple known strategies, while matching 1P1S on those with a single dominant approach.

### What would falsify this idea
If human-rated diversity of our method is not significantly higher than the surface diversity baseline, or if the improvement is uniform across all problem types rather than concentrated on those with multiple valid approaches, the central claim that causal step intervention captures approach-level diversity better than surface metrics would be falsified.

## References

1. Are We Measuring Strategy or Phrasing? The Gap Between Surface- and Approach-Level Diversity in LLM Math Reasoning
2. Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)
3. Reasoning Path Divergence: A New Metric and Curation Strategy to Unlock LLM Diverse Thinking
4. Post-training Large Language Models for Diverse High-Quality Responses
5. Diversity-Incentivized Exploration for Versatile Reasoning
6. GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time
7. Benchmarking Linguistic Diversity of Large Language Models
8. One fish, two fish, but not the whole sea: Alignment reduces language models' conceptual diversity
