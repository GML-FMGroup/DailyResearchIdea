# IV-DiagVal: Causal Validation of Synthetic Clinical Dialogues Using Instrumental Variable Regression

## Motivation

Existing synthetic dialogue benchmarks (e.g., LingxiDiagBench) assume generated dialogues are valid substitutes for real clinical interactions, but no validation framework ties synthetic diagnoses to actual patient outcomes. Without a causal validity check, deploying these models risks diagnostic errors. We leverage EMR longitudinal treatment outcomes as ground truth for a subset of cases, but naïve correlation is confounded by severity and comorbidities; instrumental variables from physician prescribing habits can isolate the causal effect of the LLM's diagnosis on downstream outcomes.

## Key Insight

Physician prescribing habits act as exogenous instruments that affect treatment assignment but are unrelated to unobserved confounders, enabling unbiased estimation of the causal impact of synthetic diagnoses on actual patient outcomes.

## Method

(A) **What it is**: IV-DiagVal is a validation framework that takes a set of synthetic dialogues (each associated with an EMR patient case) and outputs a diagnostic consistency score for each dialogue. Inputs: synthetic dialogue D_i, EMR baseline covariates X_i, prescribed treatment T_i, observed outcome Y_i (e.g., symptom improvement). Outputs: causal effect estimate β̂_IV of the LLM's recommended diagnosis on outcome.

(B) **How it works** (pseudocode):
```python
# IV-DiagVal Algorithm
Input: Synthetic dialogues D_i with recommended diagnosis R_i (from LLM), 
       EMR data: covariates X_i, prescribed treatment T_i, outcome Y_i, 
       instrument Z_i = physician's historical prescribing rate for each drug class.
Output: Diagnostic consistency score = β̂_IV (causal effect of R_i on Y_i)

Step 1: Align each synthetic dialogue D_i to its source EMR case.
Step 2: For each aligned pair, compute the LLM's diagnostic recommendation R_i (e.g., major depressive disorder).
Step 3: From EMR, extract instrument Z_i: for the prescribing physician, compute their past prescribing rate for the drug class that matches R_i (e.g., SSRI prescription rate for MDD). 
  Hyperparameter: use k=10 prior prescriptions to stabilize rate. Use logit transform.
Step 4: First-stage regression: T_i = α_0 + α_1 Z_i + α_2 X_i + ε_i. Obtain predicted treatment T̂_i.
  Condition: F-statistic > 10 for instrument strength.
Step 5: Second-stage regression: Y_i = β_0 + β_1 R_i + β_2 T̂_i + β_3 X_i + ν_i.
  β̂_IV is the coefficient on R_i.
Step 6: Diagnostic consistency score = β̂_IV. Positive and significant score indicates that LLM's diagnosis leads to appropriate treatment and improved outcome.
```

(C) **Why this design**: We chose instrumental variable regression over simple correlation because physician treatment decisions are confounded by patient severity and comorbidities, which also affect outcomes; IV isolates exogenous variation in treatment assignment via physician habits, yielding unbiased causal estimates (trade-off: IV requires a valid instrument that satisfies exclusion restriction, which may be violated if physician habits affect outcomes directly). We chose physician prescribing rates as instruments because they vary across physicians and are plausibly unrelated to patient outcomes beyond the treatment path (trade-off: weak instrument risk; we guard with F-statistic threshold). We use a two-stage least squares estimator for consistency and interpretability (trade-off: linear models may mis-specify nonlinear relationships; we accept this for parsimony and use robust standard errors). We include baseline covariates X_i to improve precision and adjust for remaining confounders (trade-off: potential collider bias if X_i are affected by treatment; we avoid variables measured after treatment).

(D) **Why it measures what we claim**: The computational quantity β̂_IV measures the causal effect of the LLM's recommended diagnosis on actual outcomes because the IV method exploits exogenous variation in treatment assignment via the instrument to deconfound the relationship; this assumption fails if the instrument Z_i affects the outcome through paths other than treatment (e.g., physician habits correlate with clinical skill), in which case β̂_IV reflects a mixture of treatment effect and direct instrument effect. The first-stage predicted treatment T̂_i captures the component of treatment influenced by physician prescribing habits, which is assumed independent of unobserved confounders given covariates; this assumption fails if prescribing habits are themselves driven by patient characteristics not in X_i, in which case T̂_i remains confounded and β̂_IV is biased.

## Contribution

(1) A causal validation framework for synthetic clinical dialogues that uses physician prescribing habits as instrumental variables to estimate the effect of LLM diagnoses on real-world outcomes. (2) A diagnostic consistency score derived from IV regression that discerns whether synthetic dialogues align with appropriate care pathways, providing a substitute for costly prospective trials. (3) A design principle: longitudinal EMR outcomes and exogenous provider-level variation can serve as ground truth for synthetic data validation, enabling safe deployment of clinical LLMs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|----------|
| Dataset | Synthetic dialogues paired with real EMR cases | Realistic with known ground truth |
| Primary metric | Diagnostic consistency score (β̂_IV) | Causal effect on outcomes |
| Baseline 1 | Naive correlation (β_OLS) | Confounded by patient severity |
| Baseline 2 | Static benchmark (accuracy) | Lacks outcome grounding |
| Baseline 3 | Multi-task reasoning (F1, coherence) | Indirect proxy for quality |
| Ablation-of-ours | IV-DiagVal without instrument (OLS) | Isolates instrument benefit |

### Why this setup validates the claim

This experimental design forms a falsifiable test of the central claim that the IV-based diagnostic consistency score provides an unbiased causal estimate of an LLM's diagnostic recommendation on patient outcomes. The dataset of synthetic dialogues paired with real EMR cases ensures we have both controlled inputs and realistic confounding. The primary metric directly measures the causal effect via IV, while the baselines test alternative evaluation approaches: naive correlation (which ignores confounding), static benchmark accuracy (which lacks outcome validation), and multi-task reasoning (which is an indirect proxy). The ablation (no instrument) isolates whether the instrument is essential for deconfounding. If IV fails to outperform naive correlation in settings with known strong confounding, the claim is falsified.

### Expected outcome and causal chain

**vs. Naive correlation (β_OLS)** — On a case where a severely ill patient receives aggressive treatment regardless of LLM diagnosis, naive correlation shows a positive association between diagnosis and outcome even if the diagnosis is wrong, because severity drives both. Our method instead instruments treatment via physician prescribing habits, isolating exogenous variation, so we expect a noticeable gap on high-severity subsets but parity on low-severity subsets.

**vs. Static benchmark (accuracy)** — On a case where the LLM correctly diagnoses a condition but the physician is reluctant to prescribe the recommended drug due to personal habit, static accuracy is high yet outcomes are poor. Our method detects low diagnostic consistency because the instrument captures the misalignment between diagnosis and treatment assignment. We expect a large divergence on cases with strong physician habit variation.

**vs. Multi-task reasoning (F1, coherence)** — On a case where the LLM generates coherent and factual reasoning but recommends a flawed treatment plan (e.g., off-label drug not suited to patient), multi-task metrics remain high. Our method penalizes such plans because the predicted treatment from the instrument does not match the recommended diagnosis, lowering β̂_IV. We expect a gap on cases where reasoning quality does not translate to treatment adherence.

### What would falsify this idea

If IV-DiagVal yields similar β̂_IV to naive correlation across all patient subsets, or if the ablation (OLS) performs equally well in settings with known strong confounding, then the instrument is not effectively deconfounding, and the central claim is wrong.

## References

1. LingxiDiagBench: A Multi-Agent Framework for Benchmarking LLMs in Chinese Psychiatric Consultation and Diagnosis
2. MentraSuite: Post-Training Large Language Models for Mental Health Reasoning and Assessment
3. The Multi-Round Diagnostic RAG Framework for Emulating Clinical Reasoning
4. MDD-5k: A New Diagnostic Conversation Dataset for Mental Disorders Synthesized via Neuro-Symbolic LLM Agents
5. AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environments
6. NoteChat: A Dataset of Synthetic Patient-Physician Conversations Conditioned on Clinical Notes
7. HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs
8. IM-RAG: Multi-Round Retrieval-Augmented Generation Through Learning Inner Monologues
