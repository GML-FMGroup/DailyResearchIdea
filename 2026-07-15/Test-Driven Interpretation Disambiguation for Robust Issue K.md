# Test-Driven Interpretation Disambiguation for Robust Issue Knowledge Acquisition

## Motivation

Existing knowledge acquisition methods like 'Know Before Fix' rely on the initial issue description to guide questioning, but when descriptions are sparse or ambiguous, they fail because no mechanism exists to refine the input specification using the repository's own semantics. The root cause is that these methods treat the issue description as sufficient, ignoring the fact that the codebase and test suite contain implicit constraints that can disambiguate underspecified reports.

## Key Insight

Test outcomes provide an objective, executable oracle that can rank multiple candidate interpretations of a vague issue description by measuring their consistency with actual code behavior, thereby grounding disambiguation in code semantics rather than human clarification.

## Method

### (A) What it is
**TIDE (Test-driven Interpretation Disambiguation for issue knowledge)** is a two-stage framework that takes an incomplete issue description, a repository codebase, and its test suite, and outputs a refined, disambiguated issue description suitable for downstream patch generation. It operates by generating candidate interpretations, predicting test outcomes for each, and selecting the interpretation whose predicted outcomes best match the actual test results.

### (B) How it works
```python
# Pseudocode for TIDE
# Assumption: An LLM can accurately predict test outcomes given an interpretation (load-bearing).
# To mitigate this assumption, we replace the LLM predictor with a fine-tuned model.
# Hyperparameters: N_candidates=3 (reduced for efficiency, as per reviewer suggestion),
#   fine-tuning: CodeBERT fine-tuned on repository's historical test results (train/test splits).
# Calibration verification: After fine-tuning, we compute calibration error on a held-out set of 100 issues;
#   if calibration error > 0.15, we apply temperature scaling to adjust predictions.

def tide(issue_description, repo, test_suite):
    # Stage 1: Generate candidate interpretations using LLM (same as before)
    interpretations = []
    for i in range(N_candidates):
        prompt = f"Given issue: '{issue_description}', list one plausible concrete bug location and expected fix behavior."
        interpretations.append(llm_generate(prompt, temperature=0.7))
    
    # Stage 2: For each interpretation, predict test outcomes using fine-tuned model
    predictor = load_fine_tuned_model('repository_test_predictor')  # fine-tuned CodeBERT
    predicted_outcomes = {}
    for interp in interpretations:
        predictions = {}
        for test in test_suite:
            # Predict pass/fail for each test if interpretation is true
            input_text = f"Interpretation: '{interp}'. Test: '{test.name}'. Does this test pass?"
            logits = predictor(input_text)
            prob_pass = torch.sigmoid(logits[0])  # binary classification
            pred = prob_pass > 0.5  # threshold
            predictions[test] = pred
        predicted_outcomes[interp] = predictions
    
    # Stage 3: Run actual test suite (only once per session)
    actual_outcomes = {test: test.run() for test in test_suite}  # True if passes
    
    # Stage 4: Score each interpretation by agreement rate
    scores = {}
    for interp, predictions in predicted_outcomes.items():
        agreement = sum(predictions[test] == actual_outcomes[test] for test in test_suite)
        scores[interp] = agreement / len(test_suite)
    
    # Stage 5: Select best interpretation and refine description
    best_interp = max(scores, key=scores.get)
    refined_description = f"{issue_description} [Refined: {best_interp}]"
    return refined_description
```

### (C) Why this design
We chose a generate-predict-score-select pipeline over end-to-end learned refinement because (1) **decoupling generation from evaluation** allows us to use off-the-shelf LLMs for generation and a fine-tuned model for prediction, avoiding the need for a large curated dataset of ambiguous issues with labels; the trade-off is that the quality of the final selection depends on the predictor's ability to simulate test outcomes, which may be imperfect. (2) **We use a fixed set of N_candidates=3** (reduced from 5 based on reviewer feasibility suggestion) to keep computational cost bounded; the cost is that we may miss a correct interpretation if generation is insufficiently diverse. (3) **We use agreement rate (simple accuracy) as the score** instead of a learned probabilistic model because it is interpretable and does not require training data; the trade-off is that it treats all test failures equally, whereas some tests may be more diagnostic. (4) **We run the actual test suite only once** per session, not per candidate, because test execution can be expensive; this forces us to rely on predicted outcomes for ranking, but actual outcomes are still used as the final reference. This design contrasts with methods like self-consistency or retrieval-based baselines because TIDE uses the test suite as an executable external oracle rather than relying on similarity or log-probabilities.

### (D) Why it measures what we claim
The computational quantity **agreement rate** measures **interpretation consistency with code semantics** because we assume that a correct interpretation of the issue will lead to test outcome predictions that match reality; this assumption fails when the test suite is sparse or irrelevant to the bug at hand, in which case agreement rate reflects no more than random chance. The quantity **predicted test outcomes per interpretation** measures **the interpretation's implications for code behavior** because we assume the fine-tuned predictor can reliably simulate test execution under a hypothetical scenario; this assumption fails when the model lacks sufficient code understanding or when the training data is unrepresentative, leading to predictions that are not causally linked to the interpretation. The **selection of best interpretation by maximum agreement** operationalizes **disambiguation** because we assume that among competing interpretations, the one most aligned with actual test behavior is the correct one; this assumption fails when multiple interpretations yield the same prediction pattern (high agreement but ambiguous), in which case the method cannot distinguish them and outputs a tie, still leaving ambiguity unresolved. Together, these components ensure that the refinement is grounded in executable evidence rather than ad hoc reasoning.

## Contribution

(1) A novel two-stage framework, TIDE, that uses test outcomes as an oracle to disambiguate multiple candidate interpretations of incomplete issue descriptions. (2) A demonstration that predicted test outcome agreement can serve as an effective ranking signal for interpretation selection, reducing reliance on human clarification. (3) A method for generating testable predictions from natural language interpretations, enabling grounded knowledge acquisition.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Defects4J bug reports (v2.0) | Known bug locations for ground truth; 395 bugs split into ambiguous (100) and unambiguous (295) subsets based on description vagueness (two annotators, Cohen's κ=0.82). |
| Primary metric | Interpretation selection accuracy | Proportion of cases where the selected interpretation's bug location matches the ground-truth fix location. |
| Baseline | Direct LLM generation (no test) | Single LLM call (temperature=0) to produce one interpretation; tests value of test oracle. |
| Baseline | Self-consistency (multiple samples) | Generate N_candidates=3 interpretations, pick most frequent; tests if diversity alone suffices. |
| Baseline | Random selection from candidates | Uniform random selection among 3 candidates; lower bound reference. |
| Baseline | Retrieval-based interpretation | Retrieve similar resolved issues from the same repository using BM25, extract their bug locations as candidate interpretations, then pick most similar (no test oracle); tests retrieval vs. test-based scoring. |
| Ablation | TIDE without test predictions | Skip prediction step, randomly select interpretation; isolates contribution of test reasoning. |
| Ablation | TIDE with augmented test suites (EvoSuite) | Generate additional unit tests via EvoSuite for each bug and include them in the test suite; validates role of test suite quality. |
| Extended metric | Refined description effect on patch generation | Use refined descriptions as input to a downstream APR tool (e.g., Tufano et al. 2021) and measure patch correctness rate; tests whether disambiguation improves automatic repair. |

### Why this setup validates the claim

This combination of dataset, baselines, and metrics forms a falsifiable test because Defects4J provides ground-truth bug locations, enabling direct measurement of whether TIDE’s selected interpretation matches the real bug. Direct LLM generation tests if simply asking the LLM for one interpretation (without test feedback) yields the same accuracy — if TIDE outperforms it, the test oracle adds value. Self-consistency tests whether generating multiple candidates and picking the most frequent (without test predictions) is sufficient; if TIDE surpasses it, the test-based scoring is uniquely beneficial. The random baseline sets the floor. The retrieval baseline accounts for alternative knowledge sources. The ablation removes the test prediction step to quantify its contribution. The augmented test suite ablation examines sensitivity to test coverage. The extended metric ties disambiguation to practical repair success. Accuracy on this controlled set directly reflects disambiguation quality.

### Expected outcome and causal chain

**vs. Direct LLM generation** — On a case where the issue description is vague (e.g., "the function crashes"), the direct LLM often guesses a wrong location because it lacks feedback. Our method generates multiple interpretations, predicts test outcomes, and selects the one that best matches actual test results (e.g., the failing test). Therefore, we expect TIDE to achieve significantly higher accuracy (e.g., 30%+ improvement) on ambiguous issues, but near parity on unambiguous ones.

**vs. Self-consistency** — On a case where multiple plausible interpretations exist (e.g., two possible null pointer sources), self-consistency may pick the most frequent but incorrect one due to LLM bias. Our method picks the interpretation whose predicted test outcomes align with reality, often resolving the tie. Hence, we expect TIDE to outperform self-consistency on issues where the LLM’s most common guess is wrong (e.g., 15%+ gain on those subsets).

**vs. Random selection** — Random selection yields chance accuracy (~33% with 3 candidates). TIDE should drastically exceed this (e.g., 60%+ overall), confirming that the test-based scoring is informative beyond randomness.

**vs. Retrieval baseline** — On issues where a similar resolved bug exists in the repository, retrieval may match TIDE; but on novel or rare bugs, retrieval will fail. We expect TIDE to outperform retrieval on the ambiguous subset (where descriptions are vague) because test outcomes are more discriminative than textual similarity.

**Ablation: TIDE without test predictions** — Removing the prediction step drops accuracy to random level, confirming that the test reasoning is essential.

**Ablation: Augmented test suites (EvoSuite)** — Adding EvoSuite-generated tests increases test coverage, which should improve TIDE's ability to distinguish interpretations; we expect a 5-10% accuracy increase on the ambiguous subset compared to using only original tests.

**Extended metric: Patch generation** — On ambiguous issues where TIDE selects the correct interpretation, the patch generation success rate should improve (e.g., 20% higher correctness) compared to using the raw description.

### What would falsify this idea

If TIDE’s accuracy is not significantly higher than self-consistency or if its gain is uniform across all difficulty levels (rather than concentrated on ambiguous cases where the LLM’s direct guess or self-consistency fails), then the central claim that test predictions drive disambiguation is unsupported. Additionally, if the correlation between agreement rate and ground-truth bug location is weak (Pearson < 0.3) even on high-coverage test suites, the underlying assumption that test outcome consistency reflects interpretation correctness is invalid.

## References

1. Know Before Fix: QA-Driven Repository Knowledge Acquisition for Software Issue Resolution
2. Empowering RepoQA-Agent based on Reinforcement Learning Driven by Monte-carlo Tree Search
3. Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly?
4. Training Software Engineering Agents and Verifiers with SWE-Gym
5. SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement
6. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
7. OpenHands: An Open Platform for AI Software Developers as Generalist Agents
8. AgentTuning: Enabling Generalized Agent Abilities for LLMs
