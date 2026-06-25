# MACE: Multi-Judge Adversarial Counterfactual Evaluation for Unbiased Causal Reasoning in Text-to-Image Models

## Motivation

The CF-Eval benchmark in 'Are Text-to-Image Models Inductivist Turkeys? A Counterfactual Benchmark for Causal Reasoning' relies on a single VLM judge (e.g., CLIP) to evaluate generated images against counterfactual rules. However, any single VLM inherits systematic biases from its training data, leading to skewed evaluations that overestimate model performance on specific causal patterns while missing others. Furthermore, the fixed set of handcrafted scenarios cannot cover the open-ended space of possible counterfactual rules, limiting generalizability. These structural flaws—single-judge bias and static scenario coverage—are the meta-level bottleneck that the proposed multi-judge adversarial framework directly addresses.

## Key Insight

The ensemble of diverse VLMs, whose biases are uncorrelated due to different training objectives and data, combined with an adversarial scenario generator that explores the rule space via gradient-free optimization, provably reduces evaluation bias and expands coverage without requiring ground-truth labels.

## Method

**MACE: Multi-Judge Adversarial Counterfactual Evaluation**

(A) **What it is**: MACE is a framework that replaces the single VLM evaluator with a panel of Diverse Judges (DJ) and an adversarial scenario generator (ASG). Input: a text-to-image model M, a seed counterfactual rule R0 (e.g., "if an object is placed on a table, then it should be upright"). Output: an unbiased score S for M's causal reasoning ability, estimated over a dynamically expanded scenario set.

(B) **How it works** (pseudocode):

```python
# Hyperparameters: N_judges = 5 (e.g., CLIP, BLIP, GPT-4V, ViLT, ALIGN), 
# K_scenarios = 200 (target scenarios), T_steps = 50 (adversarial iterations),
# alpha = 0.1 (step size for rule perturbation), W = [w1,...,w5] (judge weights, learned via held-out validation).

# Phase 1: Adversarial Scenario Generation
scenario_pool = []
R = R0
for t in range(T_steps):
    # Generate perturbed rule R' by randomly flipping conditions or outcomes with probability alpha
    R' = perturb(R, alpha)  # e.g., change "on table" to "on chair" or "upright" to "tilted"
    # Use LLM (e.g., GPT-4) to create a textual scenario S describing R' (e.g., "A cup placed on a chair, tilted")
    S = LLM_verbalize(R')
    # Evaluate current model M on scenario S using DJ ensemble (see Phase 2)
    score = evaluate(M, S, DJ)  # returns agreement between judges
    # Accept scenario if ensemble score > threshold tau (ensures non-trivial rule) and scenario is novel (cosine similarity < 0.7)
    if score > 0.3 and is_novel(S, scenario_pool):
        scenario_pool.append(S)
        R = R'  # accept perturbation
    else:
        R = R  # revert
# Phase 2: Multi-Judge Evaluation on Final Scenario Pool
final_scores = []
for S in scenario_pool:
    # Each judge j outputs a score s_j in [0,1] (e.g., CLIP similarity)
    s_j = [judge_j(image_gen(M, S), S) for judge_j in DJ]
    # Weighted aggregation
    s_agg = sum(W_j * s_j for j in range(N_judges)) / sum(W)
    final_scores.append(s_agg)
S = mean(final_scores)  # overall score
```

(C) **Why this design**: We chose an ensemble of VLMs over a single VLM because the biases of different VLMs are empirically decorrelated (CLIP favors texture, BLIP favors layout), so averaging reduces systematic error—accepting the cost of increased inference time and requiring weight calibration on a small held-out set. The adversarial generator uses random rule perturbation rather than adversarial training on the model because we aim to cover the rule space independent of model weaknesses, avoiding overfitting to specific failure modes; the trade-off is slower convergence but better diversity. The LLM verbalization module converts abstract rules into textual prompts, which is necessary because T2I models require natural language input; we accept the risk that LLM may misinterpret rules, mitigated by manual validation of a subset. Novelty filtering via cosine similarity prevents scenario redundancy, ensuring coverage breadth at the cost of potentially missing near-duplicate nuances.

(D) **Why it measures what we claim**: The *ensemble disagreement* (variance across judges) measures the *bias uncertainty* of the evaluation because if all judges agree on a scenario, it reduces the chance of individual bias driving the score; this assumption fails when all judges share a common bias (e.g., all trained on LAION-2B), in which case the metric reflects consensus rather than ground truth. The *accepted scenario count* after adversarial generation measures *coverage breadth* because each accepted scenario represents a distinct counterfactual rule that the model must handle; this assumption fails when the perturbation step size is too small, producing scenarios that are semantically identical, in which case the metric overestimates diversity. The *weighted average score* measures *overall causal reasoning ability* because higher scores indicate better alignment with diverse rule expectations across multiple visual representations; this assumption fails if the weights are poorly calibrated (e.g., over-weighting an underperforming judge), in which case the metric drifts toward that judge's bias. By linking each computational quantity to these motivation-level concepts with explicit failure modes, MACE ensures interpretable evaluation.

## Contribution

(1) A multi-judge ensemble framework for debiased VLM-based evaluation of causal reasoning in T2I models, with a learnable weighting scheme that balances judge contributions. (2) An adversarial scenario generator that dynamically expands the counterfactual rule set via LLM-guided perturbation and novelty filtering, ensuring comprehensive coverage. (3) The MACE evaluation protocol that yields both a final score and interpretable diagnostic metrics (ensemble disagreement, coverage count).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|---|---|---|
| Dataset | CF-World counterfactual scenarios | Widely used causal reasoning benchmark |
| Primary metric | Agreement with human ratings | Measures alignment with ground truth |
| Baseline | Single VLM (CF-Eval) | Represents standard evaluator |
| Baseline | Non-adversarial fixed scenario set | Tests adversarial generation's benefit |
| Baseline | Random scenario sampling | Tests coverage vs random |
| Ablation | MACE without ensemble (single judge) | Isolates ensemble effect |

### Why this setup validates the claim
This experimental design directly tests MACE's two core claims: that the multi-judge ensemble reduces bias compared to a single VLM, and that adversarial generation increases coverage beyond fixed or random scenario sets. By comparing MACE to a single VLM baseline on the same scenarios, we can isolate the effect of the ensemble. The non-adversarial fixed set baseline tests whether adversarial generation produces a more diverse and challenging evaluation set than a static pre-defined set. Random scenario sampling serves as a naive baseline to quantify the benefit of targeted adversarial scenario generation. Finally, the ablation of removing the ensemble (i.e., using only one judge) reveals how much of the performance gain comes from the multi-judge aggregation. The primary metric—agreement with human ratings—directly probes whether MACE's scores better reflect true causal reasoning ability, as judged by humans, which is the ultimate validation criterion. If MACE improves agreement over all baselines, it supports the claim; otherwise, the idea is falsified.

### Expected outcome and causal chain

**vs. Single VLM (CF-Eval)** — On a case where the single VLM is texture-biased (e.g., a counterfactual scenario requiring object placement understanding), the baseline may misjudge the image because CLIP relies heavily on texture features. Our method, averaging opinions from five diverse judges, dilutes that bias (e.g., BLIP catches layout errors), so we expect MACE's score to correlate better with human ratings, yielding a noticeable improvement in agreement (e.g., +10-15% accuracy) on texture-critical subsets, but parity on simple scenarios.

**vs. Non-adversarial fixed scenario set** — On a rare counterfactual (e.g., "a cup placed on a chair, tilted") that is absent from a static set, the baseline cannot evaluate it at all. MACE’s adversarial generator actively seeks such novel scenarios by perturbing rules, so it includes them. We expect MACE to achieve broader coverage (e.g., at least 20% more unique accepted scenarios), and on the overlapping common scenarios, both methods should yield similar agreement scores, isolating the coverage benefit.

**vs. Random scenario sampling** — On a subtle counterfactual rule (e.g., "if an object is on a table, it should be upright"), random sampling may miss perturbed versions (e.g., "if on chair, then tilted") due to uniform randomness. MACE’s adversarial perturbation targets rule space systematically, so it captures more nuanced variations. We expect MACE to detect higher disagreement among judges on these nuanced cases, leading to a lower average score but higher correlation with human judgments (since humans also find these cases harder). The observable signal: MACE’s scenario pool has lower average score but higher per-scenario agreement with humans.

### What would falsify this idea
If MACE’s agreement with human ratings is not significantly higher than that of the single VLM baseline across the entire dataset, or if the coverage gain from adversarial generation is no greater than random sampling (i.e., both yield similarly sized and diverse scenario pools), then the central claim that ensemble and adversarial generation are both beneficial would be refuted. Specifically, if the improvement is concentrated on subsets where bias is not expected (e.g., simple scenarios), rather than on bias-prone ones, the idea fails.

## References

1. Are Text-to-Image Models Inductivist Turkeys? A Counterfactual Benchmark for Causal Reasoning
2. Commonsense-T2I Challenge: Can Text-to-Image Generation Models Understand Commonsense?
3. ImagenHub: Standardizing the evaluation of conditional image generation models
4. X-IQE: eXplainable Image Quality Evaluation for Text-to-Image Generation with Visual Large Language Models
5. T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-Image Generation
6. Scaling Instruction-Finetuned Language Models
