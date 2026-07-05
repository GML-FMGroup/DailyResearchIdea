# WorkloadBench: Validating Data Agents' Assumptions about Scale, Heterogeneity, and Redundancy through Targeted Probes

## Motivation

Existing benchmarks like AgenticDataBench evaluate overall task performance but do not verify whether data agents' internal assumptions about workload characteristics (e.g., ability to scale, handle diverse data, manage redundancy) hold. This gap leads to models that may excel on aggregate metrics yet fail under specific workload conditions, as the root causes of failure remain unvalidated due to the absence of probes targeting these dimensions.

## Key Insight

The fundamental relationship between workload characteristics and agent failures is governed by three independent dimensions—scale, heterogeneity, redundancy—each causing distinct failure modes that can be isolated through controlled synthetic probes with known ground truth.

## Method

```python
# WorkloadBench construction
# Input: agent model, list of task specifications
# Output: validation scores for scale, heterogeneity, redundancy

def generate_scale_tasks(max_sizes=[1e3, 1e4, 1e5, 1e6]):
    tasks = []
    for size in max_sizes:
        data = generate_relational_data(n_rows=size, n_cols=10, seed=42)
        task = {
            "description": "Compute the average of column A",
            "data": data,
            "ground_truth": data['A'].mean(),
            "category": "scale"
        }
        tasks.append(task)
    return tasks

def generate_heterogeneity_tasks(type_counts=[2, 5, 10]):
    tasks = []
    for count in type_counts:
        datasets = [generate_random_format() for _ in range(count)]
        data = combine_datasets(datasets)  # different formats: CSV, JSON, Parquet, etc.
        task = {
            "description": "Merge all datasets on the 'id' column and compute total sum of 'value'",
            "data": data,
            "ground_truth": sum(ds['value'].sum() for ds in datasets),
            "category": "heterogeneity"
        }
        tasks.append(task)
    return tasks

def generate_redundancy_tasks(dup_ratios=[0.0, 0.2, 0.5, 0.8]):
    tasks = []
    for ratio in dup_ratios:
        base = generate_dataset()
        data = add_duplicates(base, ratio=ratio)
        task = {
            "description": "Remove duplicate rows and count unique entries",
            "data": data,
            "ground_truth": len(set(base)),
            "category": "redundancy"
        }
        tasks.append(task)
    return tasks

# Evaluation: for each task, agent receives data and description, outputs answer.
# Compute metrics: accuracy (binary correct/total), execution time, peak memory.
# For each dimension, plot accuracy vs. difficulty parameter.
```
(B) **How it works** — the pseudocode above generates three suites of tasks. During evaluation, each task is presented to the agent, and we record the agent's answer, execution time, and memory usage. We then compute the success rate as a function of the workload parameter (e.g., max_size for scale, type_counts for heterogeneity, dup_ratio for redundancy) and identify thresholds where performance drops significantly (e.g., a step function).

(C) **Why this design**: We chose synthetic data over real-world data because it allows precise control over workload parameters (scale, heterogeneity, redundancy) and yields uncontaminated ground truth, accepting the cost of reduced ecological validity. We designed three separate task suites rather than a single combined task because the three dimensions are orthogonal and each reveals a distinct failure mode (e.g., memory blow-up vs. format incompatibility vs. double-counting); the cost is that we do not test simultaneous interactions, but that can be added as a follow-up. We used multiple difficulty levels (e.g., max_sizes spanning four orders of magnitude) to pinpoint the precise boundary where assumptions break, accepting that some levels may be trivial for strong agents. We chose to measure both task accuracy and resource usage (time and memory) because an agent might produce correct answers with unrealistic resource consumption that violates scalability assumptions (e.g., O(n^2) algorithm on large data). This design is not a variant of AgenticDataBench, because we do not evaluate overall end-to-end task completion but instead isolate specific workload features and directly validate the agent's underlying capabilities.

(D) **Why it measures what we claim**: The computational quantity "accuracy as a function of max_size" measures the agent's scalability assumption because it tests whether performance remains invariant as data volume increases beyond the training distribution; this assumption fails when the agent uses a method with quadratic complexity, in which case the metric reflects empirical runtime blow-up rather than correctness. "Accuracy as a function of type_counts" measures heterogeneity tolerance because it tests whether the agent can integrate diverse data formats without explicit training; this assumption fails when the agent's schema inference relies on a fixed format, in which case the metric reflects format mismatch errors. "Accuracy as a function of dup_ratio" measures redundancy sensitivity because it tests whether the agent can handle redundant data without overcounting or confusion; this assumption fails when the agent's deduplication logic is naive, in which case the metric reflects double-counting errors. Additionally, execution time and memory usage metrics measure resource efficiency assumptions: an agent that maintains constant time regardless of max_size violates known algorithmic scaling, so the time metric captures the assumption that the agent uses a linear-time algorithm; this assumption fails when the agent uses a suboptimal method, in which case the metric reflects actual runtime growth.

## Contribution

(1) WorkloadBench, a benchmark with three suites of validation tasks (scale, heterogeneity, redundancy) that directly test data agents' assumptions about workload characteristics using controlled synthetic datasets. (2) An empirical finding that current data agents exhibit performance cliffs at specific thresholds (e.g., a sharp drop in accuracy when data size exceeds 10^5 rows or when more than 5 data formats are combined), revealing critical limitations not captured by existing benchmarks. (3) A methodology for constructing isolated workload probes that can be extended to additional dimensions or combined with real-world tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | WorkloadBench synthetic tasks | Controls three orthogonal factors |
| Primary metric | Accuracy vs. workload parameter | Captures performance change with difficulty |
| Baseline 1 | GPT-4 based data agent | State-of-the-art general-purpose agent |
| Baseline 2 | CodeLlama based data agent | Specialized code generation agent |
| Baseline 3 | Rule-based pandas script | Simple baseline without LLM reasoning |
| Ablation-of-ours | Fixed single difficulty level | Removes variation to test necessity of scaling |

### Why this setup validates the claim

By evaluating diverse agents on controlled synthetic tasks with multiple difficulty levels, we can isolate which specific assumption each agent violates (e.g., linear scaling, format tolerance, duplicate handling). The metric accuracy vs. parameter reveals failure thresholds linked to underlying mechanisms. Baselines span general LLMs, code-specialized models, and rule-based scripts, ensuring broad coverage of failure modes. The ablation (fixed difficulty) demonstrates that without parameter variation, critical boundary behaviors remain hidden, confirming our design's diagnostic power.

### Expected outcome and causal chain

**vs. GPT-4 agent** — On a scale task with 1e6 rows, GPT-4 agent attempts to load entire dataframe into memory due to lack of incremental processing, causing memory overflow. Our benchmark records high memory usage and accuracy drop at large sizes. Expected: strong accuracy for small sizes (1e3) but steep decline at 1e5 or 1e6, indicating memory-bound failure.

**vs. CodeLlama agent** — On a heterogeneity task with 10 different formats, CodeLlama agent may generate code assuming all inputs are CSV, causing parsing errors for JSON/Parquet. Our benchmark detects format-specific errors and low accuracy on mixed-format tasks. Expected: high accuracy on single-format tasks but significant drop as type count increases beyond its training coverage.

**vs. Rule-based pandas script** — On a redundancy task with 80% duplicates, a simple `drop_duplicates` works, but the script may fail on large data due to memory issues. Our benchmark captures both accuracy and resource usage. Expected: perfect accuracy on small redundancy tasks but memory failure on large scale tasks regardless of redundancy, showing rule-based methods lack adaptive resource management.

### What would falsify this idea

If all agents exhibit similar accuracy curves across difficulty levels (e.g., monotonic decay at same rate), the benchmark fails to discriminate failure modes. Alternatively, if the ablation (fixed difficulty) yields identical conclusions to the full benchmark, then varying difficulty is unnecessary, contradicting our claim.

## References

1. AgenticDataBench: A Comprehensive Benchmark for Data Agents
2. DataSciBench: An LLM Agent Benchmark for Data Science
3. BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions
4. NaturalCodeBench: Examining Coding Performance Mismatch on HumanEval and Natural User Prompts
