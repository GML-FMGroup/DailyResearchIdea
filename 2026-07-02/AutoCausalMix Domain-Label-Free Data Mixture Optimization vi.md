# AutoCausalMix: Domain-Label-Free Data Mixture Optimization via Additivity Testing

## Motivation

CausalMix requires predefined domain labels to define treatment proportions for causal effect estimation, which restricts applicability to unlabeled or dynamically forming data pools. This structural limitation recurs across data mixture optimization methods, as they assume data is partitioned into known domains. We relax this by automatically discovering latent groups such that the causal effects of group proportions on model performance are additive, enabling domain-label-free optimization.

## Key Insight

When the causal effects of data group proportions on model loss are additive (no cross-group interactions), the optimal mixture can be estimated via linear causal models without requiring domain labels, and additivity itself can be tested by checking interaction significance in a linear regression on small-scale proxy runs.

## Method

(A) **What it is**: AutoCausalMix is a two-stage framework that (1) discovers latent data groups by testing for additivity of causal effects across random partitions, and (2) applies causal mixture optimization (as in CausalMix) on the discovered groups. Input: unlabeled data pool, proxy model (e.g., 0.5B), compute budget. Output: optimal mixture weights for target model training.

(B) **How it works** (pseudocode):
```
Input: unlabeled data pool D, proxy model M0, number of proxy runs R per partition, number of random partitions P, significance threshold α=0.05
Output: optimal mixture weights w* for large model

best_partition = None
best_additivity_score = -inf

for i in 1..P:
    # Randomly partition D into K groups (K=5 default)
    groups = random_partition(D, K)
    
    # Step 1: Train R proxy models with varying group proportions
    losses = []
    for j in 1..R:
        mixture = sample_mixture(K)  # random proportions summing to 1
        sample_data = draw_mixture(D, groups, mixture)
        train M0 on sample_data for fixed steps
        record validation loss L_j and mixture proportions p_j
    
    # Step 2: Fit linear model with interactions
    # L ~ sum(beta_k * p_k) + sum(gamma_{k,l} * p_k * p_l) for k<l
    model = fit_linear_model(losses, p_j, include_interactions=True)
    
    # Step 3: Test for additivity (F-test on interaction terms)
    p_value = f_test_interactions(model)  # H0: all gamma=0
    if p_value > α:
        additivity_score = 1 - p_value  # higher is more additive
        if additivity_score > best_additivity_score:
            best_partition = groups
            best_additivity_score = additivity_score

# Step 4: Apply CausalMix on best_partition
# Use proxy runs (or additional runs) to estimate CATE of each group proportion
causal_model = fit_causal_model(best_partition, M0, num_runs=512)  # as in CausalMix
w* = find_optimal_mixture(causal_model, target_data_size)
return w*
```

(C) **Why this design**: We chose random partitions over learned clustering because learned clustering would introduce a complex optimization coupling with the causality test, risking overfitting to spurious additivity; the cost is that random partitions may miss a more coherent grouping. We chose linear additivity testing (F-test on interaction terms) over non-parametric tests (e.g., kernel independence tests) because linear test is computationally efficient and directly interprets interaction magnitude, accepting that nonlinear additive relationships may be misclassified. We chose to select the single most additive partition rather than averaging across partitions because averaging would dilute the guarantee of additivity for the chosen grouping, but this risks instability due to noise in the additivity score; the trade-off is reproducibility vs. potential overfit to a partition that is additive only by chance.

(D) **Why it measures what we claim**: The computational quantity `p_value` from the F-test on interaction coefficients measures additivity (the property that the causal effect of each group proportion on loss is independent of other proportions) because under the null hypothesis of additivity, the interaction terms are zero; the assumption is that linear interactions capture all relevant non-additivity. This assumption fails when the true loss function has non-linear interactions (e.g., multiplicative effects) but linear interactions are zero, in which case the test incorrectly accepts additivity. The `best_additivity_score` (1-p_value) measures the degree of evidence for additivity — close to 1 indicates strong evidence; this relies on the assumption that the F-test p-value monotonically reflects departure from additivity, which holds under standard linear model assumptions (homoscedastic, normal errors); when errors are heteroscedastic or non-normal, the p-value may not be calibrated. The causal model fit in step 4, as in CausalMix, uses the discovered groups as treatments and estimates CATE via linear regression or more sophisticated methods; this measures the causal effect of each group proportion on loss under the assumption of no unmeasured confounders — i.e., the mixture proportions are randomly assigned and there is no hidden structure beyond group membership; if the data generation process has confounders (e.g., topic correlation between groups), the CATE estimates may be biased.

## Experiment

### Evaluation Setup
| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Multi-domain pre-training corpus (C4 subsets) | Diverse domains cause non-additivity |
| Primary metric | Validation perplexity on held-out mixture | Direct measure of mixture quality |
| Baseline 1 | CausalMix (oracle groups) | Upper bound for group quality |
| Baseline 2 | REGMIX | State-of-the-art dynamic mixing |
| Baseline 3 | Uniform mixing | Naive baseline |
| Ablation | AutoCausalMix without partition selection | Isolates effect of additivity test |

### Why this setup validates the claim
This setup tests whether AutoCausalMix's automatic discovery of additive data groups and subsequent causal optimization outperforms alternatives. Validation perplexity on a multi-domain corpus captures how well the mixture generalizes across domains, directly reflecting the effectiveness of mixture weights. Comparing against CausalMix with oracle groups reveals the cost of automatic discovery; against REGMIX shows advantage over non-causal methods; against uniform establishes baseline. The ablation tests whether the additivity test is crucial. Together, they form a falsifiable test: if AutoCausalMix fails to outperform the ablation or oracle baseline, the central claim (that discovered additive groups improve mixture) is unsupported.

### Expected outcome and causal chain

**vs. CausalMix (oracle groups)** — On a case where oracle groups are misaligned with actual causal structure (e.g., domains with overlapping topics), CausalMix may optimize over non-additive groups, leading to suboptimal mixture. Our method, by testing many random partitions, may find a more additive grouping, yielding a mixture that better reduces perplexity. We expect AutoCausalMix to achieve perplexity comparable to or slightly better than CausalMix, with a noticeable gap when oracle groups are non-additive.

**vs. REGMIX** — On a case where domain interactions are strong (e.g., training on Wikipedia and Books together reduces perplexity more than independently), REGMIX treats all data uniformly, ignoring interactions, so it may converge to a poor static mixture. Our method captures these interactions via additivity testing, then optimizes accordingly. We expect AutoCausalMix to show lower perplexity than REGMIX, especially on validation subsets where non-additivity is prevalent.

**vs. Uniform mixing** — On a case where some domains are far more important for generalization (e.g., web text vs. specialized data), uniform mixing wastes capacity on irrelevant data, leading to high perplexity. Our method identifies domain importance through causal effects, adjusting proportions. We expect a large gap in perplexity, with AutoCausalMix achieving much lower values.

### What would falsify this idea
If AutoCausalMix's perplexity is not significantly lower than the ablation (without partition selection), then the additivity test provides no benefit, falsifying the claim that selecting an additive partition improves mixture optimization. Alternatively, if AutoCausalMix underperforms CausalMix with oracle groups, then automatic discovery fails to find adequate groups.

## References

1. CausalMix: Data Mixture as Causal Inference for Language Model Training
2. TiKMiX: Take Data Influence into Dynamic Mixture for Language Model Pre-training
3. Data Mixing Optimization for Supervised Fine-Tuning of Large Language Models
4. Aioli: A Unified Optimization Framework for Language Model Data Mixing
5. Scaling Laws for Predicting Downstream Performance in LLMs
6. TÜLU 3: Pushing Frontiers in Open Language Model Post-Training
7. What is Your Data Worth to GPT? LLM-Scale Data Valuation with Influence Functions
8. RegMix: Data Mixture as Regression for Language Model Pre-training
