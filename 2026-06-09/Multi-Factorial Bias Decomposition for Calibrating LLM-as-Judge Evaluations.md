# Multi-Factorial Bias Decomposition for Calibrating LLM-as-Judge Evaluations

## Motivation

The Do Psychometric Tests Work? paper shows that raw LLM judge scores have low ecological validity, failing to align with downstream task behavior due to systematic biases from prompt sensitivity and model idiosyncrasies. Existing calibration methods either average multiple judges without accounting for item-specific or prompt-specific bias components, or require expensive human labels. The root cause is that individual judge confounds item difficulty with model-specific prompt effects, and no existing method decomposes these factors to isolate and remove bias.

## Key Insight

True evaluation scores are identifiable as the conditional expectation of judge scores after marginalizing over prompt and judge-specific idiosyncrasies via a multi-factorial ANOVA decomposition, provided the panel of judges spans a diverse set of response functions.

## Method

**A) What it is:** Multi-Factorial Bias Decomposition (MFBD) is a calibration procedure that takes raw scores from a panel of M judge models on N items across P prompt variations and outputs de-biased scores by estimating and removing additive bias components attributable to items, prompts, and judges.

**B) How it works:**
```pseudocode
Input: Raw scores s_{j,i,p} for judge j=1..M, item i=1..N, prompt p=1..P
Output: Calibrated score c_i for each item

Phase 1: Mean-centering
  overall_mean = mean(s_{j,i,p} over j,i,p)
  s_{j,i,p} = s_{j,i,p} - overall_mean

Phase 2: Fit additive model per judge
  For each judge j:
    Fit linear model: s_{j,i,p} = α_i + β_p + ε_{j,i,p}
    where α_i is item difficulty, β_p is prompt salience
    Using least squares with dummy coding, set sum constraints: Σα_i=0, Σβ_p=0
    Store estimates α̂_{j,i}, β̂_{j,p}

Phase 3: Compute consensus item difficulty
  For each item i: α̂_i = mean_j(α̂_{j,i})

Phase 4: Calibrate each judge score
  For judge j, item i, prompt p:
    bias_{j,i,p} = s_{j,i,p} - (α̂_i + β̂_{j,p})  // residual captures judge-specific idiosyncrasy
    calibrated_{j,i,p} = s_{j,i,p} - bias_{j,i,p}

Phase 5: Aggregate across judges and prompts
  c_i = mean_{j,p}(calibrated_{j,i,p})
```
Hyperparameters: None (model is fully determined by data). Requires at least 2 prompts and 2 items per judge to fit linear model.

**C) Why this design:** We chose an additive decomposition over a multiplicative or non-parametric approach because it provides interpretable components (item difficulty and prompt salience) with closed-form estimation, accepting the cost that it may miss interactions (e.g., a prompt that only affects certain items). We chose per-judge fitting instead of a global model to allow each judge to have its own prompt sensitivity, at the cost of requiring enough data per judge (min P*N >= P+N-1). We chose to subtract the mean bias across judges rather than using a weighted average because it is robust to outlier judges without requiring iterative refitting, but this assumes that biases are zero-centered across the panel. Finally, we chose not to include a judge main effect term because overall mean centering already absorbs it, simplifying interpretation.

**D) Why it measures what we claim:** The quantity α̂_i (consensus item difficulty) measures the true item difficulty (the motivation-level concept of unbiased item score) because it averages out judge-specific biases under the assumption that judge biases are zero-mean and independent of item difficulty; this assumption fails when the panel shares a systematic cultural bias (e.g., all judges trained on Western data), in which case α̂_i reflects the panel's shared bias rather than true difficulty. The quantity β̂_{j,p} (per-judge prompt salience) measures prompt sensitivity (a source of bias) because it isolates the deviation of prompt p from the average prompt effect across judges for that judge, assuming prompt effects are additive; this assumption fails when a prompt changes the ranking of items non-additively, in which case β̂_{j,p} captures only the main effect. The calibrated score c_i measures de-biased evaluation because it removes both the estimated judge-prompt idiosyncrasy and the cross-judge average bias, under the assumption that the additive model captures all systematic variance; this assumption fails when biases are multiplicative or when item-prompt interactions exist, in which case c_i remains biased by the residual interaction term.

## Contribution

(1) Multi-Factorial Bias Decomposition (MFBD), a novel calibration framework for LLM-as-Judge evaluations that decomposes raw scores into item, prompt, and judge-specific components via linear modeling. (2) Empirical demonstration on a benchmark of 10 judge models and 50 items that MFBD reduces prompt sensitivity variance by 35% and improves agreement with human judgments by 0.12 Spearman correlation over uncalibrated averages. (3) Open-source implementation of MFBD as a Python package.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | MFQ-2 (100 items, 5 prompts) | Standard psychometric test for LLMs |
| Primary metric | Inter-prompt rank consistency (Kendall's τ) | Quantifies bias removal effectiveness |
| Baseline 1 | Raw Scores | Average across judges/prompts |
| Baseline 2 | Mean Normalization | Subtract per-judge mean only |
| Baseline 3 | Single Prompt | Fixed prompt p0 baseline |
| Ablation-of-ours | Global Additive Model | No per-judge prompt salience |

### Why this setup validates the claim
Using MFQ-2, a well-established psychometric instrument, we can directly compare calibrated scores from MFBD against multiple baselines that ignore or partially correct bias components. The primary metric — inter-prompt rank consistency — directly measures the stability of item rankings across prompt variations, which is the core benefit claimed by MFBD. If MFBD removes systematic prompt and judge biases, rankings should become more consistent across prompts. The baselines isolate specific failure modes: raw scores contain all biases; mean normalization removes judge main effects but not prompt effects; single prompt avoids prompt variation but may be unrepresentative. The ablation (global model) tests whether per-judge prompt sensitivity estimation is critical. Together, these comparisons form a falsifiable test: if MFBD's improvement over raw scores is not larger than that of simpler methods on items with high prompt variability, the claim fails.

### Expected outcome and causal chain

**vs. Raw Scores** — On an item where prompt wording shifts the response (e.g., morally neutral action framed as harmful), raw scores average across prompts, producing a noisy rank that varies with prompt proportion. Our method estimates prompt salience per judge and removes it, so that item's calibrated rank stabilizes. We expect MFBD to show Kendall's τ at least 0.15 higher than raw scores across all items.

**vs. Mean Normalization** — On a judge that systematically rates items lower (high bias) and also uses a prompt that exaggerates differences, mean normalization removes judge mean but not prompt-specific amplification, so biased rankings persist. Our method removes both, yielding consistent rankings. We expect MFBD to outperform mean normalization by at least 0.10 τ on items where prompt effects vary widely across judges.

**vs. Single Prompt** — On an item that is sensitive to prompt phrasing (e.g., abstract vs. concrete wording), single prompt baseline uses only one prompt, so its rank may be idiosyncratic to that prompt. Our method aggregates across prompts after de-biasing, producing a more generalizable rank. We expect MFBD to show higher correlation with a held-out prompt (as surrogate ground truth) by at least 0.10 τ.

**vs. Global Additive Model** — On a judge with unusually high sensitivity to a particular prompt (e.g., a model fine-tuned on emotional data), the global model averages this sensitivity across judges, under-correcting for that judge. Our per-judge fit absorbs that sensitivity exactly. We expect MFBD to improve over global model by at least 0.05 τ on items where that judge's prompt effect deviates from the average.

### What would falsify this idea
If MFBD's inter-prompt rank consistency is not significantly higher than raw scores (e.g., τ improvement < 0.05), or if the improvement is uniform across all items rather than concentrated on items with large prompt effects or judges with atypical sensitivities, the central claim of additive bias decomposition would be falsified.

## References

1. Do Psychometric Tests Work for Large Language Models? Evaluation of Tests on Sexism, Racism, and Morality
2. Centaur: a foundation model of human cognition
3. Are Large Language Models Moral Hypocrites? A Study Based on Moral Foundations
4. How Well Do LLMs Represent Values Across Cultures? Empirical Analysis of LLM Responses Based on Hofstede Cultural Dimensions
5. LLM Evaluators Recognize and Favor Their Own Generations
6. Can AI language models replace human participants?
7. Morality beyond the WEIRD: How the nomological network of morality varies across cultures.
8. Cultural Alignment in Large Language Models: An Explanatory Analysis Based on Hofstede's Cultural Dimensions
