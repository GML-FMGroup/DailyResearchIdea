# Causal Transportability of Hallucination Mitigation in Code Generation: A Multiplicative Decomposition Approach

## Motivation

Existing evaluations of hallucination mitigation techniques for code generation models rely on surface-level output metrics without causal analysis, assuming that observed effects generalize across model families. This assumption fails because model architectures, training data, and hyperparameters act as confounders. For example, the survey by Hallucination by Code Generation LLMs notes that mitigation effectiveness varies across models, but no methodology establishes causal transportability. This leaves practitioners unable to predict whether a technique that works on GPT-4 will have the same causal effect on CodeLLaMA without costly re-evaluation.

## Key Insight

The effect of any mitigation technique on hallucination rate decomposes multiplicatively into a model-specific susceptibility factor and a technique-specific efficacy factor, with the susceptibility factor estimable from a minimal set of baseline hallucination examples.

## Method

### Causal Transportability via Multiplicative Decomposition (CTMD)
**Load-bearing assumption (explicit):** The effect of a mitigation technique on hallucination rate decomposes multiplicatively into a model-specific susceptibility factor and a technique-specific efficacy factor that is invariant across models.

**How it works:**
```pseudocode
1. Reference Phase (once per technique set):
   - Select a reference model M_ref, e.g., GPT-4.
   - For each mitigation technique T_i, estimate its efficacy E_i:= ΔH(M_ref, T_i) / S_ref
     where ΔH(M_ref, T_i) is the observed reduction in hallucination rate under T_i vs. baseline, and S_ref is set to 1 as an identifiability constraint.
2. Target Phase (per new model):
   - Given target model M_t, collect a small set of N=50 prompts that are known to induce hallucinations in any model (e.g., ambiguous API calls from PackageHallucination test set covering 10 API categories: AWS, Azure, Google Cloud, Stripe, Twilio, etc.). N is fixed to 50.
   - Generate code from M_t with a reference technique T_ref (e.g., retrieval-augmented generation with fixed API docs) and without mitigation. Compute the observed effect ΔH_obs(M_t, T_ref).
   - Estimate susceptibility S_t := ΔH_obs(M_t, T_ref) / E_ref.
2.5. Validation of multiplicative assumption:
   - Estimate susceptibility using a second reference technique T_ref2 (e.g., prompt engineering with few-shot examples). Compute S_t2 = ΔH_obs(M_t, T_ref2) / E_ref2.
   - Compute ratio r = S_t / S_t2. If r ∉ [0.8, 1.2], the multiplicative assumption is violated for M_t. In that case, do not use CTMD for prediction; output a warning that predictions may be unreliable. (Default threshold: 0.2 deviation.)
3. Prediction:
   - For any technique T_i, predict effect on M_t as  ΔH_pred(M_t, T_i) = S_t * E_i.
```

**Why this design:** We chose a multiplicative decomposition over an additive one because prior work (e.g., Collu-Bench) shows that baseline hallucination rates differ by orders of magnitude across models, making an additive constant unrealistic. We chose to estimate susceptibility using a single reference technique rather than multiple to minimize data requirements, accepting the cost that if the reference technique is poorly calibrated on the target model (e.g., due to different retriever quality), the estimate may be biased. We fixed the reference model susceptibility to 1 for identifiability, which is a standard convention in factor models; the trade-off is that the absolute scale of effects is relative to the reference model, but relative comparisons across models remain valid. We selected a small fixed prompt set for baseline evaluation rather than a random sample to ensure that the prompts elicit at least some hallucinations in most models, avoiding zero baseline effects that would make susceptibility undefined; the risk is that the set may not be representative of the deployment distribution, but we mitigate this by selecting prompts from diverse API categories.

**Why it measures what we claim:** The computational quantity [observed effect ΔH_obs on reference model] measures [technique efficacy E_i] because we assume a stable causal relationship where the mitigation mechanism operates similarly across models after scaling by susceptibility; this assumption fails when the technique interacts with model-specific properties (e.g., a retriever that works poorly for some models' tokenization), in which case E_i reflects the effect only on the reference model rather than an invariant property. The quantity [estimated susceptibility S_t] measures [model's inherent hallucination tendency relative to reference] because we assume the multiplicative decomposition holds exactly; this assumption fails when there is effect modification by unobserved confounders like instruction-tuning quality, making S_t an average effect rather than a true susceptibility. The predicted effect ΔH_pred measures [transportable mitigation impact] because it operationalizes the decomposition under the causal invariance assumption; when the decomposition is violated, ΔH_pred instead reflects a biased estimate that may mislead practitioners.

## Contribution

(1) A causal framework that decomposes the effect of hallucination mitigations into model susceptibility and technique efficacy, enabling transportability across model families. (2) A minimal-data procedure to estimate model susceptibility from a small number of hallucination-prone prompts, reducing the need for full re-evaluation. (3) An analysis of the conditions under which the multiplicative decomposition holds, identifying model characteristics that may violate invariance.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|-------|--------|------------------------|
| Dataset | PackageHallucination (Py/JS) – 50 curated prompts across 10 API categories (AWS, Azure, Google Cloud, Stripe, Twilio, etc.) | Realistic package hallucination across models. |
| Primary metric | MAE of predicted vs measured ΔH | Directly evaluates prediction accuracy. |
| Baseline 1 | Direct transfer (use E_i directly, i.e., ΔH_pred = E_i) | Tests necessity of susceptibility parameter. |
| Baseline 2 | Additive decomposition (ΔH_pred = E_i + b, where b = ΔH_obs(M_t, T_ref) - E_ref) | Tests multiplicative vs additive assumption. |
| Ablation of ours | Random prompt set (50 random HumanEval prompts instead of curated hallucination-inducing prompts) | Tests dependency on prompt selection. |

### Why this setup validates the claim

This setup tests the central claim that a multiplicative decomposition with a single reference technique accurately predicts hallucination reduction on new models. By comparing against direct transfer, we test whether the susceptibility parameter is necessary. Against additive decomposition, we test the functional form. The ablation with random prompts tests the importance of curated hallucination-inducing prompts for baseline evaluation. MAE directly measures predictive accuracy on the continuous reduction values, making it the appropriate metric to detect whether our predicted effects match the ground truth. If our method outperforms both baselines and the ablation underperforms, the claim is supported. Additionally, the validation step (ratio test with a second reference technique) provides an internal check of the multiplicative assumption, allowing us to flag cases where the method is not applicable (ratio outside [0.8, 1.2]), thereby avoiding overclaiming on violation cases.

### Expected outcome and causal chain

**vs. Direct Transfer** — On a case where the target model has a much higher baseline hallucination rate than the reference (e.g., CodeLlama vs GPT-4), direct transfer predicts the same absolute reduction for all techniques, but actual reductions are proportionally larger due to higher baseline. Our method scales effects by susceptibility S_t, capturing this variation. We expect our MAE to be significantly lower on high-discrepancy models, while both perform similarly on models with similar baselines.

**vs. Additive Decomposition** — On a case where the target model's baseline is an order of magnitude higher (e.g., 20% vs 2%), additive decomposition adds a constant bias, predicting similar absolute reductions irrespective of baseline. Our multiplicative method predicts proportionally larger reductions for techniques with higher efficacy. We expect additive decomposition to show systematic over/under prediction, leading to higher MAE, especially on models with disparate baseline rates.

### What would falsify this idea

If our method's MAE is not substantially lower than direct transfer on models where baseline hallucination rates differ by more than 5x, or if additive decomposition achieves lower MAE overall, then the multiplicative susceptibility assumption is invalid. Furthermore, if more than 20% of target models trigger the validation warning (ratio outside [0.8, 1.2]), the method's applicability is severely limited.

## References

1. Hallucination by Code Generation LLMs: Taxonomy, Benchmarks, Mitigation, and Challenges
2. LLM Hallucinations in Practical Code Generation: Phenomena, Mechanism, and Mitigation
3. We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs
4. Collu-Bench: A Benchmark for Predicting Language Model Hallucinations in Code
5. <inline-formula><tex-math notation="LaTeX">$\mathbf{A^{3}}$</tex-math><alternatives><mml:math display="inline"><mml:mrow><mml:msup><mml:mi mathvariant="bold">A</mml:mi><mml:mrow><mml:mn mathvariant="bold">3</mml:mn></mml:mrow></mml:msup></mml:mrow></mml:math><inline-graphic xlink:href="huang-ieq1-34
6. RLCoder: Reinforcement Learning for Repository-Level Code Completion
7. ClarifyGPT: A Framework for Enhancing LLM-Based Code Generation via Requirements Clarification
8. On Mitigating Code LLM Hallucinations with API Documentation
