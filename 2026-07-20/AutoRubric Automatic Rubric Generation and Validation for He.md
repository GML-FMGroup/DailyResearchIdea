# AutoRubric: Automatic Rubric Generation and Validation for Healthcare LLM Evaluation

## Motivation

Existing healthcare LLM evaluations, such as MedAgentBench, rely on physician-created rubrics, which are expensive to produce and limit scalability. Cura 1T's self-evolution loop similarly depends on human gating for safety verification. This reliance persists because the field assumes that high-quality evaluation requires expert-crafted rubrics, leaving automated rubric generation unvalidated.

## Key Insight

A rubric generated from a task description can be validated through self-consistency across multiple generative trajectories, as the underlying task structure imposes invariance in the evaluation criteria.

## Method

### (A) What it is
AutoRubric is an LLM-based system that inputs a task description (e.g., "Diagnose the patient with chest pain") and outputs a structured rubric (list of criteria with scoring guidelines) along with a consistency score. It uses self-consistency across multiple generated drafts to validate the rubric, with a supervised calibration step that maps consistency to correctness probability.

### (B) How it works
1. **Sample K rubric drafts**: Given task description D, query an LLM (e.g., GPT-4) with a structured prompt: "Generate a rubric for evaluating an LLM's performance on the following task: {D}. The rubric should be a list of criteria, each with a name, description, and scoring scale (e.g., 1-5)." Use temperature T=0.7, K=5.
2. **Parse drafts**: Extract criteria sets C_i from each draft using regex-based parsing (e.g., extract items under "Criteria:" section).
3. **Compute pairwise agreement**: For each pair (i,j), compute Jaccard similarity on the sets of criterion names (after normalization: lowercase, remove punctuation). Average over all pairs to get A_avg.
4. **Calibrate and accept**: Use a calibration set of 10 tasks from the ECRR dataset (with human rubrics). For each calibration task, compute A_avg (over K=5 drafts) and correctness C (Jaccard similarity between the majority-vote draft's criteria and the human rubric criteria). Train a logistic regression model with input A_avg and binary target C > 0.7. On a new task, compute A_avg, predict correctness probability p, accept rubric if p > 0.8. If accepted, select the draft with the highest average similarity to others (or majority vote on criteria). If not, increase K to 10 and re-sample; if still predicted low, default to a generic rubric.
5. **Optional alignment check**: Use a separate LLM call with prompt: "Does the following rubric adequately cover the requirements of the task? Task: {D}. Rubric: {accepted rubric}. Answer Yes/No and reason." If No, fall back to the second-best draft.

### (C) Why this design
We chose self-consistency over a single-generation approach because it provides a natural validation signal without external supervision; the trade-off is computational cost from multiple samples (K=5 vs. 1). We chose Jaccard similarity on parsed criteria over embedding-based similarity because it is interpretable and enforces exact match of criterion names, accepting that synonyms (e.g., "accuracy" vs. "correctness") may reduce agreement and cause false rejections. We introduced a calibration step to mitigate the load-bearing assumption that high self-consistency (Jaccard similarity > 0.8) among multiple drafts implies the rubric is correct; this assumption fails when task ambiguity causes systematic bias. The logistic regression maps consistency to correctness using a small set of human-validated rubrics, providing a probabilistic acceptance criterion. We used temperature sampling (T=0.7) to explore diverse yet valid formulations, accepting that very high temperatures (e.g., T=1.2) produce incoherent outputs. Unlike self-consistency for final answer selection (Wang et al.), where consistency implies correctness, here consistency validates the rubric's structure as a stable artifact independent of generation noise; this is a different structural property because the rubric is an intermediate specification rather than a final answer.

### (D) Why it measures what we claim
The average pairwise agreement A_avg measures rubric consistency because it captures the invariance of evaluation criteria across different language realizations; this assumes that a correct rubric should be robust to phrasing variations. This assumption fails when the task description is ambiguous, leading to multiple valid but different rubrics (e.g., different dimensions of quality), in which case A_avg reflects ambiguity rather than correctness. The calibration step addresses this by learning a mapping from A_avg to correctness on a held-out set. The alignment check measures how well the rubric covers the task description's intent; this assumes that the LLM's judgment of alignment is reliable, which fails when the task description contains implicit requirements not captured by the rubric (e.g., "be efficient" without explicit time constraints), in which case the alignment check may overestimate coverage.

## Contribution

(1) A novel method for automatic rubric generation from task descriptions using LLMs with self-consistency validation, eliminating the need for physician-created rubrics. (2) The design principle that self-consistency across multiple generations can substitute for human expert validation in rubric creation, demonstrated on healthcare tasks. (3) An open-source implementation and a validation protocol using MedAgentBench tasks to assess rubric quality against physician-crafted gold standards.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | ECRR (10 tasks), Medical Summarization (10 tasks from MIMIC-III), Medical Safety (10 tasks from MedSafety) | Diverse healthcare tasks for broad evaluation |
| Primary metric | Spearman correlation on scoring | Measures rubric validity for evaluation |
| Baseline1 | Single-shot GPT-4 rubric generation | No self-consistency; baseline lower bound |
| Baseline2 | Ensemble of 3 rubrics (majority vote) | Simpler consistency without Jaccard threshold |
| Baseline3 | Embedding-similarity rubric merging | Tests if semantic matching beats Jaccard |
| Baseline4 | Keyword extraction (TF-IDF key phrases as criteria) | Non-LLM baseline to highlight LLM benefit |
| Ablation-of-ours | AutoRubric without alignment check | Isolates alignment module's contribution |
| Sensitivity | Vary τ from 0.5 to 1.0 on calibration tasks | Assess robustness of acceptance threshold |

### Why this setup validates the claim
This combination of dataset, baselines, metric, and sensitivity analysis forms a falsifiable test of AutoRubric's central claim: that self-consistency with calibration across drafts validates a rubric without full human oversight. The ECRR dataset provides ground truth rubrics for 10 clinical reasoning tasks; two additional task sets (summarization and safety) test generalizability. Spearman correlation between scores from generated and human rubrics directly tests practical utility for evaluation. Single-shot baseline checks whether self-consistency adds value beyond a single generation. Ensemble baseline tests if simple majority voting suffices. Embedding baseline tests if exact match on criterion names is preferable to semantic similarity. Keyword extraction baseline tests the unique benefit of LLM generation. Ablation tests the alignment check's contribution. Sensitivity analysis on the calibration threshold τ ensures the method is not brittle. If AutoRubric outperforms all baselines and ablation, the claim is supported; if not, the core assumptions are challenged.

### Expected outcome and causal chain

**vs. Single-shot GPT-4 rubric** — On ambiguous tasks (e.g., "evaluate diagnostic reasoning"), single-shot may miss key criteria due to interpretation bias. Our method samples multiple drafts and selects via calibrated consistency, capturing more facets. Expect a noticeable gap in Spearman correlation on ambiguous tasks but parity on clear-cut tasks.

**vs. Ensemble of 3 rubrics (majority vote)** — When majority drafts share a common but incomplete set (e.g., all omit "cost-effectiveness"), ensemble will be incomplete. Our pairwise agreement may select a higher-quality minority draft. Expect our method to outperform on tasks where majority draft is suboptimal.

**vs. Embedding-similarity rubric merging** — On tasks where synonyms cause low Jaccard but high embedding similarity (e.g., "accuracy" vs. "correctness"), embedding may merge distinct criteria incorrectly. Our exact match enforces precise criteria, avoiding over-merging. Expect higher precision and better correlation on tasks with fine-grained distinctions.

**vs. Keyword extraction baseline** — On tasks requiring semantic understanding (e.g., "assess patient safety risks"), keyword extraction may miss context-dependent criteria. Our LLM-based generation captures nuanced criteria. Expect a significant gap on complex tasks but smaller gap on simple fact-based tasks.

### What would falsify this idea
If AutoRubric's Spearman correlation is not significantly higher than the single-shot baseline on tasks with high LLM output variability, or if the calibration step never triggers rejection, then the claim that self-consistency with calibration provides validation is falsified.

## References

1. Cura 1T: Specialized Model for Agentic Healthcare
2. MedAgentBench: A Realistic Virtual EHR Environment to Benchmark Medical LLM Agents
