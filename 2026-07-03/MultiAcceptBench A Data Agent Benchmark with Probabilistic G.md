# MultiAcceptBench: A Data Agent Benchmark with Probabilistic Ground Truth for Real-World Ambiguity

## Motivation

Existing data agent benchmarks such as AgenticDataBench and DataSciBench assume tasks have unique correct answers, inheriting a deterministic evaluation paradigm from code generation. In real-world data analysis, multiple valid outputs exist due to ambiguous requirements, varying domain conventions, or subjective judgment. This discrepancy causes benchmarks to penalize agents that produce valid but non-canonical outputs, masking true competence and encouraging overfitting to a single interpretation.

## Key Insight

Data analysis tasks possess a natural structure of multiple acceptable outputs characterized by equivalence classes under domain-specific transformations, which can be captured by a probabilistic coverage metric over the space of valid solutions.

## Method

### (A) What it is
**MultiAcceptBench** is a benchmark construction framework that generates data analysis tasks with probabilistic ground truth. Inputs are raw datasets and open-ended natural language queries. Outputs are a set of acceptable answer templates, each weighted by a probability reflecting real-world prevalence. The framework replaces exact-match scoring with a coverage score measuring how many acceptable outputs the agent's answer belongs to.

### (B) How it works
```
Procedure BuildMultiAcceptBench(T={tasks from real-world analytics}, K=5 experts, M=3 LLM judges):
  For each task t in T:
    1. Query K human experts to provide diverse reference solutions.
    2. Use each expert solution as seed to prompt M LLMs (e.g., GPT-4, Claude) to generate alternative valid solutions via paraphrasing or different analytic choices.
    3. Cluster all generated solutions into equivalence classes based on output equality (after deterministic normalization: rounding to 2 decimal places, sorting columns alphabetically).
    4. Assign each cluster a probability proportional to its frequency among experts + LLM samples (smoothed with Dirichlet prior α=1).
    5. Define the ground truth as a probability distribution over clusters.
  Return evaluation protocol: For an agent's answer a, compute coverage = sum_{cluster c} P(c) * I(a ∈ c).
```
Hyperparameter: number of experts K=5 (balances diversity with annotation cost); number of LLM judges M=3 (covers stylistic variation).

**Load-bearing assumption:** The normalization (rounding to 2 decimal places, sorting columns alphabetically) maps all acceptable variants of the same analytic path to the same output, so that cluster membership captures semantic acceptability.

### (C) Why this design
We chose expert-LLM hybrid generation over purely automatic generation because experts provide ecologically valid initial solutions, while LLMs cheaply expand the space; the trade-off is that LLM-generated variants may drift from plausibility, requiring a clustering step to filter outliers. We chose probabilistic cluster weighting over uniform weighting because real ambiguity is not uniform—some solution families are more common in practice; the cost is sensitivity to the Dirichlet prior (α=1 is uninformative, but if domain knowledge suggests peakiness, α could be set lower). We chose output equality (after normalization) as equivalence criterion rather than semantic similarity because it is computationally cheap and objective; the downside is that syntactically different but semantically identical outputs (e.g., using different column names for the same derived feature) may be split into multiple clusters, but this is acceptable since coverage penalizes missing common surface forms. The clustering step does not rely on any learned model, avoiding anti-patterns of disentanglement or controller gating.

### (D) Why it measures what we claim
The coverage score computed from equivalence-class probabilities measures the degree to which an agent's output is acceptable among plausible real-world solutions. Specifically, the equivalence class membership I(a ∈ c) measures alignment with a common analytic path; the cluster probability P(c) measures the prevalence of that path in the true distribution of human and LLM solutions—this assumes that the sampling distribution (experts + LLMs) approximates the true decision space of domain practitioners. This assumption fails when domain practitioners systematically prefer solutions underrepresented in our sample (e.g., due to institutional bias), in which case coverage reflects sample coverage rather than true acceptability. The normalization step (rounding, column ordering) operationalizes the assumption that acceptable outputs are invariant under trivial transformations—this fails when normalization destroys semantic distinctions (e.g., rounding two distinct values to the same number), causing false positives in equivalence assignment. Despite these failure modes, the metric causally separates agents that consistently hit common solution families from those that produce rare or invalid outputs, addressing the motivation's need for ambiguity-aware evaluation.

**Explicit naming of equivalence assumption:** Cluster membership (after normalization) measures semantic acceptability under the assumption that normalization (rounding to 2 decimal places, sorting columns alphabetically) maps all acceptable variants of the same analytic path to the same output. **Failure mode:** Normalization may produce false positives (merging distinct meanings) or false negatives (splitting identical semantics) due to arbitrary thresholds. **Calibration/verification:** To verify cluster coherence, we randomly sample 20% of clusters and have human annotators judge whether all members are semantically equivalent; clusters with >10% disagreement are flagged and reassigned according to majority vote among annotators. This verification step does not alter the clustering mechanism but identifies and corrects likely misclusters, thereby improving the reliability of the coverage metric.

## Contribution

(1) A benchmark construction framework that models real-world ambiguity in data analysis tasks via probabilistic ground truth over equivalence classes of acceptable outputs. (2) An evaluation metric (coverage score) that replaces exact-match with a prevalence-weighted measure of acceptability. (3) A semi-automated pipeline combining expert annotation and LLM-based expansion to generate ground-truth distributions without requiring exhaustive manual specification.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | 100 analytics tasks from DataSciBench | Realistic, diverse, expert-annotated |
| Primary metric | Coverage over probabilistic clusters | Measures acceptability spectrum |
| Baseline | Exact Match Accuracy (EMA) | Standard strict metric |
| Baseline | Top-1 Cluster Match (T1CM) | Ignores cluster probability distribution |
| Ablation | Uniform cluster weighting (UWC) | Tests importance of probabilistic weights |

### Why this setup validates the claim
The central claim is that coverage over probabilistic clusters captures ambiguity better than exact match or top-cluster accuracy. By comparing EMA, we test whether strict matching misses acceptable variations. T1CM tests whether considering only the most probable solution is insufficient. The ablation UWC tests the weighting scheme. The dataset, with expert-annotated solution distributions, provides ground truth to quantify alignment. The metric coverage directly measures how well agent outputs match the true distribution. If coverage shows higher correlation with human judgment and better distinguishes agents that hit common solutions from those that produce rare or invalid ones, it validates the claim. This design isolates the key mechanism—probabilistic, multi-acceptance evaluation—and falsifies alternative explanations like semantic similarity or single-reference matching.

### Expected outcome and causal chain

**vs. Exact Match Accuracy (EMA)** — On a case where the query has multiple valid analytic approaches (e.g., different aggregation methods for a trend), EMA requires exact match to one predefined reference, failing for agents that produce a different acceptable output. Our method includes that output in a probable cluster, so coverage is high. We expect EMA to show low sensitivity (many false negatives) on ambiguous tasks, while coverage maintains high scores for diverse correct agents, producing a noticeable gap (e.g., 20% absolute) on such tasks but near parity on unambiguous ones.

**vs. Top-1 Cluster Match (T1CM)** — On a case where the most probable solution is not the only common one (e.g., two approaches with probabilities 0.45 and 0.40), T1CM only gives credit for hitting the top cluster. Our method gives partial credit for hitting the second cluster. We expect T1CM to underestimate quality of agents that output the second solution by about 0.4 on average for such balanced tasks, while coverage captures it, leading to a clear separation on tasks with no dominant cluster.

**vs. Uniform cluster weighting (UWC)** — On a case where expert solutions cluster into a high-probability group (0.8) and a low-probability group (0.05), UWC treats both equally, overvaluing rare solutions. Our method gives proper weight, so we expect UWC to reward agents that hit rare clusters, while coverage correctly downweights them. The gap should be largest when agents deliberately output uncommon but still acceptable answers, with coverage scores up to 0.75 lower than UWC for such outputs.

### What would falsify this idea
If coverage and EMA produce identical rankings across all tasks, or if coverage correlates negatively with human judgment of acceptability (e.g., agents with high coverage rated as poor), then the clustering or weighting scheme fails to capture true ambiguity.

## References

1. AgenticDataBench: A Comprehensive Benchmark for Data Agents
2. DataSciBench: An LLM Agent Benchmark for Data Science
3. BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions
4. NaturalCodeBench: Examining Coding Performance Mismatch on HumanEval and Natural User Prompts
