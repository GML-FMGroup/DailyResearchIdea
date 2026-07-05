# PipelineBench: A Benchmark for End-to-End Data Science Agents

## Motivation

Existing data agent benchmarks like AgenticDataBench are confined to data-centric operations (e.g., cleaning, querying) and do not evaluate model training, validation, or deployment. This limited scope fails to capture the interdependent nature of real-world data science workflows, where decisions in earlier stages (e.g., feature engineering) directly impact later stages (e.g., model performance). The root cause is that these benchmarks inherited a narrow task definition from prior work, focusing on isolated operations rather than complete pipelines.

## Key Insight

Pipeline stages are causally interdependent—data quality affects training, which affects deployment—so a benchmark must evaluate agents on full end-to-end pipelines to measure true data science capability.

## Method

We propose PipelineBench, a benchmark for evaluating LLM agents on end-to-end data science pipelines.

**(A) What it is:** PipelineBench takes a natural-language task description (e.g., "Train a classifier on the Iris dataset, achieve >95% accuracy, and deploy as a REST API") and evaluates the agent's sequence of actions in a simulated environment. Inputs: task description; Outputs: composite score.

**(B) How it works:**
```python
# Pseudocode for PipelineBench evaluation
1. TaskSpec = {
     'dataset': 'iris',
     'goal': 'classification',
     'perf_threshold': 0.95,
     'deployment_required': True
   }
2. Environment = Sandbox(Datasets={'iris': df}, APIs=[load_data, clean_data, split_data, train_model, evaluate_model, deploy_model])
3. Agent.run(TaskSpec, Environment) -> action_sequence
4. Evaluate(action_sequence):
     stage_scores = []
     for action in action_sequence:
         if action matches expected step: stage_score += 1
     final_model = action_sequence[-1].model  # extract trained model
     perf = evaluate_model(final_model)  # accuracy
     deploy_ok = (action_sequence.last_action == 'deploy' and endpoint works)
     composite = w1 * (stage_scores / max_stages) + w2 * perf + w3 * deploy_ok + w4 * (1 - steps_ratio)
   return composite
```
Hyperparameters: Weights w1, w2, w3, w4 are calibrated via human expert judgments using the Analytic Hierarchy Process (Saaty, 2008) on 10 diverse pipeline tasks. The calibration procedure: three data science experts provide pairwise comparisons of the four criteria (stage correctness, performance, deployment, efficiency) for each of 10 tasks; the geometric mean of the pairwise matrices yields a group preference, from which we compute the principal eigenvalue weights. The resulting calibrated weights are: w1=0.25, w2=0.40, w3=0.20, w4=0.15. max_stages=6 (load, clean, split, train, evaluate, deploy).

**(C) Why this design:** We chose to include multiple interdependent stages because a data science agent must handle cross-stage consistency; a benchmark that tests only data cleaning misses how cleaning choices affect model performance. We chose a simulated environment over real cloud deployment to ensure reproducibility and safety, accepting the trade-off that deployment realism is reduced. We chose a composite score over separate per-stage metrics because it reflects the holistic nature of the workflow, but this introduces the challenge of weighting decisions—we calibrate weights via expert elicitation to mitigate arbitrariness. We used semi-automated ground truth generation (inspired by DataSciBench) to scale to many pipeline tasks: we source 50 pipeline tasks from Kaggle (20 datasets), UCI (15), and synthetic (15), with natural language descriptions; GPT-4 generates an initial pipeline plan which two human annotators validate and correct; each task takes ~2 hours, totaling ~100 person-hours. Validation criteria: all tasks include a dataset, goal, performance threshold, and deployment requirement; the ground-truth pipeline is verified by a separate human expert for correctness and completeness.

**(D) Why it measures what we claim:** The stage correctness score measures the agent's ability to execute each pipeline operation because we assume the correct sequence of operations is necessary for a successful pipeline; this assumption fails when an agent repeats steps unnecessarily (e.g., cleaning twice) still gets full stage score, in which case the stage score reflects procedural compliance rather than efficiency. The final model performance metric measures the ultimate goal of the pipeline—model quality—because it directly evaluates the trained model; this assumption fails if the agent trains a model that overfits (e.g., high accuracy on validation but not on test), in which case performance measures in-sample fit only. The deployment success metric measures whether the agent can operationalize the model; the assumption is that a deployed model is the intended output; this fails if the agent deploys a placeholder rather than the trained model, in which case it measures deployment action occurrence rather than correctness. The efficiency metric (steps ratio) measures resource usage; the assumption is that fewer steps indicate better planning; this fails if an agent omits necessary steps (e.g., skips data cleaning) to reduce steps, in which case it measures brevity over quality. Collectively, the composite score measures end-to-end data science capability under the assumption that the weighted sum of stage correctness, performance, deployment, and efficiency reflects real-world utility; this assumption fails if the calibrated weights are not representative of task priorities across all pipeline tasks, in which case the composite score measures a biased aggregate. Load-bearing assumption: The AHP-derived weights produce a valid holistic measure because they reflect consistent expert preferences; we verify this by computing the consistency ratio (CR) for each expert matrix (CR<0.1) and ensuring the group geometric mean matrix also has CR<0.1.

## Contribution

(1) PipelineBench, a benchmark that evaluates LLM agents on full end-to-end data science pipelines including training, validation, and deployment via a simulated environment. (2) A design principle that pipeline tasks must evaluate interdependent stages jointly, not in isolation, to reflect real-world cascading effects. (3) A semi-automated pipeline for generating ground truth for composite pipeline tasks, combining LLM self-consistency with human verification.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | 50 multi-step data science pipeline tasks | Covers diverse real-world scenarios |
| Primary metric | Composite score (4 components, AHP-calibrated weights) | Holistic measure of end-to-end quality |
| Baseline 1 | DataSciBench (per-step accuracy only) | Ignores final outcome and efficiency |
| Baseline 2 | BigCodeBench (function call correctness) | Focuses on single-step code generation |
| Baseline 3 | Final performance only (no process score) | Omits procedural quality |
| Ablation | PipelineBench with equal weights (0.25 each) | Tests sensitivity to weighting choices |

### Why this setup validates the claim
The combination of a diverse dataset, composite metric, and baselines that isolate different aspects of the pipeline allows us to test whether PipelineBench captures the full end-to-end process. DataSciBench tests whether per-step accuracy alone is sufficient; BigCodeBench tests whether single-step code generation suffices; the final-performance-only baseline tests whether outcome alone is enough. The ablation tests the sensitivity of the weight allocation, and the use of AHP-calibrated weights grounds the composite in expert preferences, reducing the risk of arbitrary weighting. The composite metric is chosen because it directly reflects the holistic nature of data science workflows, and any improvement in agent capability should manifest as a higher composite score. This setup enables falsification: if our benchmark does not differentiate between agents that are strong in process vs. outcome, then the claim that it measures end-to-end quality is refuted.

### Expected outcome and causal chain

**vs. DataSciBench** — On a pipeline task where data cleaning choices critically affect model accuracy (e.g., incorrect treatment of missing values), DataSciBench still awards full credit if each step's action string matches a ground-truth step, ignoring the downstream impact. Our method includes final performance (weight 0.4), so an agent that cleans poorly but steps in order will get a lower composite. We expect a noticeable gap (e.g., 0.3 difference) on such tasks between the two benchmarks, with PipelineBench lower for suboptimal agents.

**vs. BigCodeBench** — Consider a task requiring chaining load, clean, train, and deploy. BigCodeBench evaluates each generated code snippet in isolation, missing cross-step state management (e.g., passing the cleaned dataframe to training). Our benchmark observes the end-to-end sequence and checks deployment success. An agent good at writing individual functions but failing to chain them will score high on BigCodeBench but low on our composite (e.g., 0.2 vs 0.8). We expect our benchmark to reveal this gap.

**vs. Final performance only** — On a task where an agent trains a model with high accuracy on a validation set but never deploys it, the final-performance baseline assigns a high score based on accuracy alone. Our benchmark requires the deployment step and will penalize missing deployment (weight 0.2). We expect our composite to be significantly lower (e.g., 0.5 vs 0.9) for agents that skip deployment, correctly reflecting incomplete pipelines.

### What would falsify this idea
If an agent that executes steps in the correct order but with poor quality achieves a composite score comparable to an agent that achieves high performance but skips deployment, then the composite fails to distinguish outcome quality from process completeness, contradicting the claim that it measures both.

## References

1. AgenticDataBench: A Comprehensive Benchmark for Data Agents
2. DataSciBench: An LLM Agent Benchmark for Data Science
3. BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions
4. NaturalCodeBench: Examining Coding Performance Mismatch on HumanEval and Natural User Prompts
