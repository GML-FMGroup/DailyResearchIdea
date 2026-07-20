# Detecting AI-Assisted Code Reviews via Copula-Based Dependence Invariance Test

## Motivation

Current methods for identifying AI-assisted code reviews rely on explicit mentions of AI tools in pull request comments (e.g., From Human-Centric to Agentic Code Review), missing cases where developers do not disclose AI usage. This reliance on public metadata is a structural limitation: it assumes AI tool adoption is fully captured by observable references, which fails in corporate or private settings where AI usage is undocumented.

## Key Insight

The dependence structure between review features (e.g., comment length and response time) remains invariant for human-only reviews but shifts systematically when an AI assistant is involved, and this shift can be detected without labels by testing for copula invariance.

## Method

## Algorithm: Copula-based Dependence Invariance Test (CDIT)
### (A) What it is
CDIT is an unsupervised statistical test that detects AI-assisted code reviews by checking whether the copula of review features deviates from a reference copula estimated from human-only reviews.

### (B) How it works
```pseudocode
Input: Set of review threads R from project P
       Reference set R_ref (human-only PRs from same or similar projects)
Output: For each review thread r in R, a p-value indicating deviation from human-only copula

Step 1: Feature extraction
   For each review thread r:
       Extract per-comment features: comment length (chars), response time (seconds since previous comment), code churn (lines changed in diff)
       Form vectors of length n_r: X_r = [length_1, ..., length_n], Y_r = [time_1, ..., time_n], Z_r = [churn_1, ..., churn_n]
       Keep only complete cases.

Step 2: Estimate marginal distributions
   For each feature dimension d in {length, time, churn}:
       Compute empirical CDF F̂_d from reference set values.

Step 3: Transform to uniform margins
   For each feature vector V in {X,Y,Z} for each review:
       U_d = F̂_d(V_d)

Step 4: Estimate copula for each review
   For each review r, with N_r triplets (U_x, U_y, U_z):
       Empirical copula: Ĉ_r(u,v,w) = (1/N_r) Σ I(U_x ≤ u, U_y ≤ v, U_z ≤ w)

Step 5: Reference copula
   **Load-bearing assumption: R_ref contains exclusively human-only reviews. To verify and mitigate contamination, we use a trimmed copula estimator: first estimate a preliminary copula from all R_ref, then exclude the 5% of reviews with largest Cramér–von Mises deviation from that copula, and re-estimate on the remaining 95%. Alternatively, we restrict R_ref to reviews from a time period before AI tools were widely available (e.g., before 2021) sourced from the MSR Challenge dataset.**
   From reference set R_ref, pool all transformed triplets.
   Fit a parametric copula family (Clayton, Frank, Gumbel, Gaussian) via maximum likelihood using the 'copula' package in R, selecting the best by AIC.
   Obtain Ĉ_ref as the fitted parametric copula.

Step 6: Goodness-of-fit test
   For each review r, compute Cramér–von Mises statistic:
       T_r = ∫ (Ĉ_r - Ĉ_ref)^2 dĈ_ref
   Compute p-value via parametric bootstrap:
       For b in 1..B (B=1000):
           Sample N_r triplets from Ĉ_ref, compute empirical copula and T_b
       p = (1/B) Σ I(T_b ≥ T_r)

Step 7: Decision
   Reviews with p < α (e.g., 0.05) are flagged as likely AI-assisted.
```

### (C) Why this design
We chose copula modeling over direct joint density estimation because copulas separate marginal distributions from dependence structure, allowing us to isolate the dependence shift induced by AI while ignoring marginal shifts (e.g., overall longer comments). The cost is that copula estimation requires a good reference and is sensitive to small sample sizes. We fit a parametric copula to the reference set rather than a nonparametric empirical copula because parametric families provide a smooth model that generalizes to unseen patterns and enables straightforward bootstrap testing; the downside is potential model misspecification if the true dependence is not captured. We use bootstrap p-values instead of asymptotic approximations because empirical copula test statistics have complicated asymptotic distributions, especially with small N_r, making bootstrap valid under the null despite computational expense. We selected comment length, response time, and code churn as features because they are readily available from PR metadata and are plausibly affected by AI assistance (e.g., faster responses with longer comments); the limitation is that qualitative aspects are ignored.

### (D) Why it measures what we claim
The reference copula Ĉ_ref measures the dependence structure under the null hypothesis that the review is human-only, assuming the reference set is pure human reviews and the copula is invariant across time and contexts. This assumption fails when human review practices evolve (e.g., new tools change behavior) or when the reference set contains hidden AI reviews, in which case Ĉ_ref is contaminated and the test's sensitivity drops. The statistic T_r measures the squared difference between the review's empirical copula and Ĉ_ref, quantifying deviation from expected dependence; this deviation is causally linked to AI assistance under the assumption that AI introduces a distinct dependence pattern (e.g., negative correlation between response time and comment length becomes zero or positive). This assumption fails when human reviewers also produce similar patterns (e.g., a very fast human writes long comments), causing false positives. The bootstrap p-value calibrates the deviation magnitude under the null, ensuring only large deviations are flagged as significant; thus, each component operationalizes a step in the causal chain from dependence invariance to detection.

## Contribution

(1) A novel unsupervised method for detecting AI-assisted code reviews based on copula invariance testing, requiring no labels or explicit AI mentions. (2) An empirical finding that the copula structure of review features differs between human-only and AI-assisted reviews, validated on a large-scale GitHub dataset. (3) A publicly available reference copula dataset and evaluation framework for future research.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | CodeReview-AI dataset (Chen et al., 2024) with 10,000 labeled PRs (human/AI) from GitHub | Ensures ground truth for detection task |
| Primary metric | AUC-ROC | Threshold-free measure of ranking ability |
| Baseline 1 | Univariate threshold (comment length) | Tests if longer comments indicate AI |
| Baseline 2 | Logistic regression (feature means) | Baseline using summary statistics |
| Baseline 3 | Correlation difference test | Tests change in pairwise correlations |
| Ablation | CDIT with empirical copula | Removes parametric assumption to test necessity |
| Robustness check | CDIT on a synthetic dataset where human reviews are perturbed by time-of-day effects (confounding factor) | Tests whether CDIT flags AI-specific patterns rather than general behavioral shifts |

### Why this setup validates the claim

This experimental design creates a falsifiable test of CDIT’s central claim that it detects AI assistance by identifying shifts in dependence structure rather than marginal distributions or simple correlations. The dataset with known labels provides ground truth to compute detection accuracy. The univariate threshold tests whether marginal shifts are sufficient—if AI simply makes comments longer, a length threshold would work. Logistic regression on means tests whether summary statistics capture the AI signature. The correlation difference test checks if linear dependencies are enough. The ablation (empirical copula) tests whether parametric modeling is essential for robust detection. AUC-ROC is chosen because it aggregates performance across all decision thresholds, reflecting the test’s ability to rank deviation severity—a key property for a statistical test with adjustable α. The robustness check with synthetic confounding verifies that CDIT is not triggered by common human behavioral variations (e.g., faster reviews in the morning), isolating AI-specific dependence shifts. Together, these comparisons isolate whether only CDIT’s copula-based dependence invariance test yields unique signal.

### Expected outcome and causal chain

**vs. Univariate threshold** — On a review where AI generates comments of average length but responds unusually quickly, the threshold fails because it only looks at length marginals, flagging normal-length comments as human. Our method instead detects a dependence shift (e.g., response time and length become uncorrelated) because it models joint behavior via copula, so we expect a noticeable gap in AUC-ROC on reviews with near-identical length distributions but different timing patterns.

**vs. Logistic regression** — On a review where AI produces comments with the same means but increased variance or altered cross-feature dependencies (e.g., long comments no longer take longer), logistic regression on means sees no difference and declares it human. Our method captures the changed dependence structure because the copula isolates interactions, so we expect a clear advantage on reviews where marginal means are unchanged but copula deviates.

**vs. Correlation difference test** — On a review where AI introduces a nonlinear dependence (e.g., a U-shaped relation between comment length and response time), the linear correlation test misses it and outputs no change. Our method detects it because the empirical copula captures arbitrary dependencies. We expect CDIT to perform better on reviews with non-linear AI-induced patterns, while parity on linear-only shifts.

**vs. Ablation (empirical copula)** — On a review with few comments (N_r < 10), empirical copula is noisy and yields large p-values, missing AI presence. Our parametric copula smooths via a fitted family and produces stable p-values, so we expect higher sensitivity on small-review subsets for the parametric version.

**vs. Robustness check** — On synthetic confounding data where human reviews are perturbed by time-of-day effects (e.g., faster response times in the morning), CDIT should not flag these as AI-assisted because the dependence structure remains within the human-only reference. We expect AUC-ROC near 0.5 on this synthetic set, confirming the test is specific to AI patterns.

### What would falsify this idea
If CDIT’s AUC-ROC is not significantly higher than the correlation difference test’s in subsets where feature marginals are matched (e.g., stratified by comment length), this would indicate that the added copula complexity does not capture extra signal, contradicting the claim that dependence structure shifts are key. Also, if the ablation (empirical copula) matches or outperforms parametric CDIT, the parametric assumption is unnecessary. Additionally, if the robustness check yields high AUC-ROC (e.g., >0.7), then CDIT is sensitive to confounding factors, undermining its specificity to AI.

## References

1. From Human-Centric to Agentic Code Review: The Impact of Different Generations of Generative AI Technology on Review Quality
2. Developer-LLM Conversations: An Empirical Study of Interactions and Generated Code Quality
3. On the Use of Agentic Coding: An Empirical Study of Pull Requests on GitHub
4. A Performance Study of LLM-Generated Code on Leetcode
5. On the Taxonomy of Developers’ Discussion Topics with ChatGPT
6. WildChat: 1M ChatGPT Interaction Logs in the Wild
7. LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset
