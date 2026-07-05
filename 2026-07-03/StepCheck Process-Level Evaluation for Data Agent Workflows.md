# StepCheck: Process-Level Evaluation for Data Agent Workflows

## Motivation

Existing benchmarks like AgenticDataBench and DataSciBench evaluate data agents solely on final outputs, ignoring the quality of intermediate reasoning steps such as SQL execution traces or code checkpoints. This output-only scoring fails to provide fine-grained feedback for model improvement and cannot differentiate between agents that reach the correct answer through flawed reasoning versus sound reasoning. The root cause is the lack of a standardized method to decompose data agent workflows into verifiable intermediate states and assign partial credit.

## Key Insight

Process-level evaluation for data agents is feasible by defining a task-specific ontology of verifiable intermediate states (e.g., SQL parse tree, query result set, code output) and using a set of deterministic verifiers that assign partial credit based on edit distance or semantic equivalence at each step, enabling direct reward signals without model-based scoring.

## Method

(A) **What it is**: StepCheck is a process-level evaluation framework that takes a data agent's task description and its full execution trajectory (sequence of actions and intermediate outputs) and outputs a scalar process score representing the correctness of all intermediate steps. The framework is built on a task-specific milestone graph that defines the expected intermediate states. This graph is constructed by collecting multiple correct trajectories from domain experts; it includes branching alternative paths to cover common valid strategies. Tasks with low coverage (e.g., fewer than 3 alternative paths per milestone) are flagged for manual review. This is a load-bearing assumption: the milestone graph for each task is assumed to be complete, meaning any valid reasoning trajectory of a competent agent will produce intermediate states that match (within verifier tolerance) one of the pre-defined milestone steps. Failure mode: correct trajectories with unseen intermediate states are penalized. Mitigation: we release an annotation guideline template (see appendix) to ensure systematic coverage and compute coverage statistics per task (e.g., fraction of expert trajectories covered).

(B) **How it works**:
```python
def process_score(task, trajectory, milestone_graph):
    # milestone_graph: dict mapping milestone_id to (verifier, weight, prerequisite_milestone)
    # verifier options: exact_match, normalized_edit_distance (threshold=0.8), semantic_equivalence (result set comparison as unordered sets of tuples)
    score = 0
    total_weight = 0
    for milestone in milestone_graph:
        step_output = trajectory.get_step_output(milestone)
        step_score = milestone.verifier(step_output, milestone.ground_truth)
        # Prerequisite constraint (threshold = 0.5)
        if milestone.prerequisite:
            prereq_id = milestone.prerequisite
            # score of prerequisite milestone is computed recursively or accessed from cache
            score_prereq = cached_prereq_scores[prereq_id]  # computed earlier
            if score_prereq < 0.5:
                step_score *= 0.5
        score += milestone.weight * step_score
        total_weight += milestone.weight
        # cache step_score for later prerequisites
        cached_prereq_scores[milestone.id] = step_score
    return score / total_weight
```
Hyperparameters: weights per milestone (sum to 1, default uniform), prerequisite threshold (default 0.5). Verifiers: exact match, normalized edit distance (threshold=0.8), semantic equivalence via unordered result set comparison. The milestone graph is stored as a DAG; prerequisite dependencies are enforced by ordering milestones topologically.

(C) **Why this design**: We chose deterministic verifiers (e.g., exact match, normalized edit distance, semantic equivalence via query result comparison over a limited domain) over learned reward models because learned models introduce proxy bias and require task-specific training data, which undermines the benchmark's generality and reproducibility. The cost is that deterministic verifiers may miss semantically equivalent but superficially different intermediate representations; we mitigate this by defining verifiers that are permissive within a small set of equivalences (e.g., SQL query results are compared as unordered sets of tuples). We opted for a prerequisite constraint that scales the step score rather than a hard cutoff, because hard cutoffs would discard partial progress in the presence of a minor early mistake; the trade-off is that small early errors propagate less but are still penalized. We selected a weighted sum over geometric mean or minimum because weighted sum preserves the contribution of each step and aligns with the intuition that some steps (e.g., data loading) are less critical than others; the downside is that it can be gamed by excelling on easy, high-weight steps while neglecting harder ones. This trade-off is acceptable because the weights are derived from task domain experts and are fixed per task category.

(D) **Why it measures what we claim**: The process score directly operationalizes "stepwise reasoning quality" by computing a weighted average of per-milestone verifier scores, where each verifier score measures the extent to which an intermediate state matches the expected ground truth. The assumption is that the milestone graph captures all necessary reasoning steps; this assumption fails when the agent discovers a novel, equally valid alternative sequence of steps that yields the same final output but with different intermediate states—in that case, the process score may underestimate the agent's reasoning quality because it penalizes deviations from the canonical path. Despite this, for the majority of data agent tasks (e.g., SQL queries, data cleaning pipelines), the milestone graph is designed to be inclusive of common alternatives and is validated against expert annotations to ensure coverage. The prerequisite constraint measures adherence to dependency ordering; the assumption is that an agent cannot correctly perform a later step if an earlier prerequisite is incorrect (a monotonic property of data workflows). This assumption fails when an agent makes an early mistake but later corrects it; the prerequisite penalty may then unfairly reduce the score even though the final outcome is correct. We accept this because such cases are rare in practice and the penalty encourages monitoring intermediate errors.

(E) **Limitations and mitigations**: The equivalence process score measures reasoning quality assuming milestones cover all acceptable paths. Failure mode: correct trajectories with unseen intermediate states are penalized. Mitigation: collect and verify an explicit set of alternative correct paths per task (at least 3 distinct paths per milestone) and compute coverage (fraction of held-out expert trajectories that match at least one path). Tasks with coverage below 80% are excluded from the benchmark. This is documented in the annotation guideline template (released with the benchmark). Additionally, we conduct a human agreement study on milestone annotation (Cohen's kappa among 3 experts) to ensure reliability; tasks with kappa < 0.6 are refined.

## Contribution

(1) A novel process-level evaluation framework (StepCheck) that defines milestone graphs and deterministic verifiers for data agent tasks. (2) A design principle for constructing task-specific intermediate state ontologies that balance coverage and verification cost. (3) A publicly available benchmark dataset of annotated trajectories with step-level ground truth for reproducible evaluation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | DataSciBench annotated subset (50 tasks with milestone graphs) | Contains diverse tasks with milestones; tasks have coverage ≥80% and inter-annotator kappa ≥0.6 |
| Primary metric | Process Score vs human agreement (Pearson r) | Measures alignment with true stepwise quality |
| Baseline 1 | End-to-End Accuracy (final output correctness) | Ignores stepwise correctness; tests gap |
| Baseline 2 | Process Reward Model (trained on 80% of tasks, tested on 20% held-out) | Learns step scores; tests generality vs deterministic |
| Baseline 3 | AgentBench subtask metrics (exact match on pre-defined subtasks) | Existing stepwise evaluation; highlights data-specific challenges |
| Baseline 4 | Human Expert Ratings (3 experts, average score per trajectory) | Gold standard; validates our metric's correlation |
| Ablation 1 | StepCheck w/o prerequisite penalty | Tests importance of dependency modeling |
| Ablation 2 | StepCheck with hard cutoff (prerequisite < 0.5 → step score = 0) | Tests soft scaling vs hard cutoff |
| Sensitivity analysis | Vary prerequisite threshold from 0.1 to 0.9 (step=0.1) | Measures robustness to hyperparameter |

### Why this setup validates the claim

The chosen dataset provides ground truth milestone annotations and human ratings, enabling calculation of process quality correlation. Comparing with end-to-end accuracy tests whether stepwise information adds value beyond final correctness. The process reward model baseline examines if learned verifiers can match deterministic ones without task-specific training. AgentBench subtask metrics provide a direct comparison to prior stepwise evaluation, highlighting challenges unique to data workflows (e.g., SQL equivalence). Human ratings serve as the gold standard for correlation. The ablations isolate the prerequisite constraint's contribution and the impact of hard vs soft scaling. The sensitivity analysis ensures the threshold choice does not unduly influence results. The primary metric, agreement with human ratings, directly tests if our design captures stepwise reasoning quality better than alternatives, forming a falsifiable test of the central claim.

### Expected outcome and causal chain

**vs. End-to-End Accuracy** — On a case where agent loads wrong file then corrects, E2E accuracy outputs 1 (final correct) because it ignores process. Human experts see a flawed intermediate step and rate process lower. StepCheck penalizes early mistake via prerequisite scaling, producing lower scores that align with human ratings. We expect StepCheck to show high correlation with humans (r>0.8) while E2E accuracy exhibits weak correlation (r<0.5) on such dependency-violation instances.

**vs. Process Reward Model (trained)** — On a task with novel step pattern unseen in training, PRM outputs noisy scores due to distribution shift because it learned spurious correlations. StepCheck's deterministic verifiers (exact match on SQL results) remain accurate regardless of novelty. We expect StepCheck to maintain high human agreement (r>0.85) while PRM degrades (r<0.6) on rare or atypical subtasks, highlighting the generality cost of learned models.

**vs. AgentBench subtask metrics** — On a SQL task where two distinct intermediate query plans produce identical final results, AgentBench's exact-match subtask metrics would penalize the alternative plan. StepCheck's semantic equivalence verifier (comparing unordered result sets) would correctly give full credit. We expect StepCheck to achieve higher correlation with humans (r>0.85 vs r<0.7) on tasks where equivalence classes are non-trivial.

**vs. Human Expert Ratings** — On standard tasks, both methods should agree. If our metric's correlation with humans is substantially lower than the inter-rater reliability among experts (kappa=0.85), then our verifiers or milestones miss key reasoning aspects. We expect StepCheck to achieve correlation r>0.8 with human scores, indicating that stepwise evaluation captures expert intuition.

### What would falsify this idea

If our process score correlates no better than end-to-end accuracy with human ratings (e.g., r<0.6), or if the ablation without prerequisite penalty outperforms the full method on tasks with strong dependencies, or if the sensitivity analysis shows the prerequisite threshold drastically changes rankings (e.g., Spearman rank correlation <0.8 across thresholds), then the central claims about stepwise evaluation and prerequisite modeling are invalid.

## References

1. AgenticDataBench: A Comprehensive Benchmark for Data Agents
2. DataSciBench: An LLM Agent Benchmark for Data Science
3. BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions
4. NaturalCodeBench: Examining Coding Performance Mismatch on HumanEval and Natural User Prompts
