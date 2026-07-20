# ClinGen: Dynamic Generation of Patient Profiles and Clinical Tasks for Expanding Medical LLM Benchmarks

## Motivation

MedAgentBench provides only 100 patient profiles and 300 tasks, limiting diversity and generalization of medical LLM agents. Cura 1T's self-evolution loop refines the training data mixture but remains confined to this fixed set, failing to expose the agent to novel clinical scenarios. This static benchmark cannot keep pace with the breadth of real-world healthcare, leading to overfitting and poor out-of-distribution performance.

## Key Insight

By sampling from a clinical ontology with coverage constraints, we can systematically generate diverse and clinically valid patient profiles and tasks that are guaranteed to cover a wide range of medical conditions and task types, breaking the static benchmark limitation.

## Method

(A) **What it is**: ClinGen is a generative framework that produces new patient profiles and clinical tasks for medical LLM agents. Inputs: a clinical ontology (e.g., ICD-10, SNOMED CT), a set of task templates, and a pre-trained LLM (e.g., GPT-4). Outputs: a set of validated patient profiles and associated tasks. (B) **How it works** (pseudocode):
```
1. Define coverage constraints C over ontology categories (e.g., each ICD-10 chapter appears at least once per 100 profiles).
2. Initialize an empty set of profiles P and tasks T.
3. For i = 1 to N (desired number of new profiles):
   a. Sample a combination of conditions (c1, c2, ..., ck) from ontology such that the combination is clinically plausible (e.g., co-occurrence statistics from MIMIC-III) and satisfies coverage constraints updated dynamically.
   b. Prompt LLM with: "Generate a detailed patient profile including age, sex, ethnicity, medical history, current medications, vital signs, and lab results that is consistent with the following conditions: {c1, ..., ck}. Ensure demographic diversity."
   c. For each condition in the profile, generate associated tasks by sampling from task template library (e.g., "Diagnose the cause of {symptom}", "Prescribe medication for {condition}", "Evaluate treatment plan"). Fill template slots with profile-specific details.
   d. Validate each task: check that the task is solvable from the profile (e.g., there is enough info to make a diagnosis) and that the expected answer is clinically correct. Use a validator LLM with clinical guidelines.
   e. If validation passes, add profile to P and tasks to T.
4. Update coverage constraints and repeat.
```
(C) **Why this design**: We chose ontology-guided sampling over random generation because the ontology ensures comprehensive coverage of medical domains, addressing the core limitation of MedAgentBench's narrow scope. We weighted co-occurrence statistics from MIMIC-III to generate clinically plausible condition combinations, accepting the cost that rare but valid combinations may be underrepresented. We used a pretrained LLM for profile and task generation rather than rule-based templates because LLMs produce more natural and diverse text, though this introduces potential hallucination—mitigated by the validation step. The validation step uses a separate LLM with explicit clinical guidelines rather than a single model for generation and validation, decoupling the two to avoid self-confirmation bias at the cost of increased computational overhead. Finally, we dynamically update coverage constraints after each generation to avoid redundancy, even though this adds complexity compared to a fixed schedule. (D) **Why it measures what we claim**: The coverage constraint frequency metric measures diversity of generated profiles because it ensures each ontology category appears proportionally; this assumption fails when the ontology itself is incomplete (e.g., missing novel diseases), in which case the metric reflects coverage only of known categories. The clinical plausibility check (co-occurrence statistics) measures realism of generated profiles because it leverages empirical joint distributions of conditions; this assumption fails when the source data (MIMIC-III) has selection bias, in which case the metric reflects average clinical co-occurrence patterns from a specific population. The task solvability check (expected answer consistency) measures task validity because it verifies that the profile contains sufficient information and the task has a unique correct answer; this assumption fails when tasks admit multiple plausible solutions, in which case the validator may reject valid tasks or accept invalid ones. Together, these components operationalize the motivation-level concept of "dynamic diversity and clinical validity."

## Contribution

(1) A generative framework, ClinGen, that dynamically expands medical LLM benchmarks by producing novel patient profiles and clinical tasks guided by a clinical ontology and coverage constraints. (2) A validation pipeline combining co-occurrence statistics and LLM-based checking that ensures generated profiles and tasks are clinically plausible and solvable. (3) A demonstration that ClinGen can generate diverse profiles covering all major ICD-10 chapters, whereas MedAgentBench covers only a subset.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | MIMIC-III subset (5 chapters) | Representative of diverse conditions |
| Primary metric | Valid Task Coverage Rate | Measures both diversity and validity |
| Baseline 1 | Random Condition Sampling | Tests ontology guidance importance |
| Baseline 2 | Rule-Based Profile Generator | Tests LLM naturalness benefit |
| Baseline 3 | ClinGen without validation | Tests validation decoupling |
| Ablation | ClinGen with static constraints | Tests dynamic update effect |

### Why this setup validates the claim

This setup isolates each key component of ClinGen. The dataset (MIMIC-III subset) ensures co-occurrence statistics are representative, while the primary metric (Valid Task Coverage Rate) jointly captures diversity (coverage across categories) and validity (task solvability). Comparing against Random Condition Sampling tests the ontology guidance sub-claim: without it, many generated combinations are clinically implausible or redundant, lowering valid coverage. Comparing against Rule-Based Profile Generator tests whether LLM-driven generation adds necessary realism: rule-based profiles are too rigid, leading to unsolvable tasks. Comparing against a version without validation tests the decoupling and verification sub-claim: without validation, profiles may contain hallucinations that make tasks unsolvable. The ablation with static coverage constraints tests the dynamic update sub-claim: static constraints lead to redundancy and lower overall coverage. If ClinGen outperforms all baselines and ablation on valid coverage, the central claim is supported; if not, the specific failure pattern points to which component is insufficient.

### Expected outcome and causal chain

**vs. Random Condition Sampling** — On a case where acute myocardial infarction and type 2 diabetes must co-occur, random sampling may select a rare or implausible comorbidity (e.g., pregnancy) that lacks clinical backing, producing profiles that fail validation. ClinGen uses co-occurrence statistics from MIMIC-III, ensuring the combination is empirically plausible, so its profiles pass validation. We expect a noticeable gap: random baseline achieves ~30% valid coverage, while ClinGen exceeds 80% on common condition pairs.

**vs. Rule-Based Profile Generator** — On a case requiring a nuanced presentation (atypical chest pain with anxiety), rule-based templates generate a generic profile missing key details (e.g., negated stress test results), making tasks like "diagnose cause" unsolvable. ClinGen's LLM generates rich, realistic profiles with lab values and history, enabling solvable tasks. We expect rule-based to achieve ~40% valid coverage, ClinGen >85%.

**vs. ClinGen without validation** — On a case where the LLM hallucinates an improbable lab value (e.g., HbA1c=3.5%), the no-validation version produces a seemingly plausible profile that yields unsolvable tasks (e.g., prescribing insulin). ClinGen's validation step catches these errors and rejects the profile. We expect both to have similar generation rates, but ClinGen's valid coverage will be 20-30% higher because many no-validation profiles are invalid.

### What would falsify this idea

If ClinGen's valid coverage is not significantly higher than Random Condition Sampling on categories with strong co-occurrence patterns, or if the ablation with static constraints achieves similar coverage to dynamic ClinGen, then the central claim that ontology-guided sampling and dynamic constraints improve diversity and validity is wrong.

## References

1. Cura 1T: Specialized Model for Agentic Healthcare
2. MedAgentBench: A Realistic Virtual EHR Environment to Benchmark Medical LLM Agents
