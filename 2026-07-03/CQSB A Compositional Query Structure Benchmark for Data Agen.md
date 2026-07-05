# CQSB: A Compositional Query Structure Benchmark for Data Agent Generalization

## Motivation

Existing data agent benchmarks like AgenticDataBench expand domain coverage but retain a fixed query-answering format, thus failing to measure an agent's ability to generalize to novel query structures. This is a critical gap because real-world data agents must handle ad-hoc queries that combine operations in unfamiliar ways. Without systematic evaluation of compositional generalization, progress on truly adaptable data agents remains obscured.

## Key Insight

The combinatorial space of data agent query structures can be factorized into a finite set of atomic operations and a grammar of composition rules, enabling principled generation of unseen query structures by varying which composition rules are allowed in training versus testing.

## Method

```pseudocode
# CQSB Benchmark Generation
Input:
  Ops = {SELECT, FILTER, JOIN, GROUP, SORT, VISUALIZE}  # atomic operations
  Datasets = {D1, D2, ...}  # from AgenticDataBench
  Grammar G = (N, T, P, S):
    N = {Query, Filter, Group, Sort, ...}
    T = Ops ∪ Datasets
    P = production rules (e.g., Query -> Filter Query | Group Sort | ...)
    S = Query
  SeenRules = subset of P (e.g., 70% of productions)
  NumTrain, NumTest = 500, 200
Output: TrainTasks, TestTasks

Procedure:
  for split, rules in [(train, SeenRules), (test, P - SeenRules)]:
    for i = 1..count(split):
      repeat:
        tree = sample_derivation(G, rules, max_depth=8)
        ops_seq = linearize(tree)  # e.g., depth-first order
        assign_params(ops_seq, Datasets)  # pick dataset, columns, constants
      until check_executable(ops_seq)  # pre/post conditions satisfied
      prompt = generate_prompt(ops_seq)  # natural language from template
      gt = execute_pipeline(ops_seq)  # ground truth output
      store(task = (prompt, gt))
```

**Why this design** (≥80 words): We chose grammar-based generation over manual curation because it enables systematic control over structural novelty and scalability to large task sets, accepting that generated tasks may lack the linguistic naturalness of human-written ones. We split grammar productions rather than randomly sampling tasks to isolate the effect of compositional structure on agent performance, though this may produce test tasks that are unrealistic in their exact composition pattern. We enforce executability via pre/post conditions (e.g., join requires a common key) to ensure tasks are solvable, at the cost of limiting the complexity of generated pipelines. We use a depth limit of 8 to keep tasks tractable, which may miss very deep compositions. These decisions prioritize controlled measurement of generalization over ecological validity.

**Why it measures what we claim** (≥60 words): The grammar derivation tree defines the query structure; using only unseen productions in the test split ensures that the task uses composition patterns absent from training, thereby directly measuring compositional generalization. The set of atomic operations and datasets is identical across splits, so performance differences are attributable to structural novelty rather than operation familiarity. However, this assumes that the grammar productions are a valid proxy for real query structure variety; this assumption fails when real queries involve operations not in Ops or compositions beyond the grammar's generative capacity, in which case the benchmark may overestimate generalization to truly out-of-distribution scenarios.

## Contribution

(1) A benchmark generation framework that systematically creates data agent tasks with controlled query structure novelty, using a grammar to define seen and unseen composition patterns. (2) The principle of isolating compositional generalization in data agent evaluation by varying only the production rules while keeping atomic operations and datasets fixed. (3) A set of 700 executable tasks (500 train, 200 test) built on AgenticDataBench's datasets and atomic operations, enabling reproducible evaluation of adaptability to unseen query types.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | CQSB | Generated with controlled compositional splits |
| Primary metric | Task success rate | Measures ability to handle novel compositions |
| Baseline 1 | AgenticDataBench | Lacks explicit compositional structure control |
| Baseline 2 | DataSciBench | Tasks not split by grammar productions |
| Baseline 3 | RandomSplit of CQSB | No structural isolation; tests confound |
| Ablation-of-ours | Without executability check | May produce unsolvable tasks; tests necessity |

### Why this setup validates the claim

This setup directly tests whether our grammar-based benchmark isolates compositional generalization from other task difficulty factors. By comparing to existing benchmarks (AgenticDataBench, DataSciBench) that lack structural control, we can attribute any performance differences to our split methodology. The random-split baseline (RandomSplit) controls for the possibility that simply having more diverse tasks drives the result. The ablation without executability checks ensures that task solvability is necessary for measuring true generalization. Using task success rate as the primary metric captures the central goal of the benchmark: to measure how well agents generalize to unseen compositions. Together, these comparisons form a falsifiable test: if our benchmark does not reveal a performance gap where novel compositions are harder, then the design fails to measure compositional generalization.

### Expected outcome and causal chain

**vs. AgenticDataBench** — On a case involving a filter-then-join composition unseen in training, an agent trained on AgenticDataBench's uncontrolled splits may succeed because similar joint patterns appear in training. The baseline lacks explicit composition novelty control, so it cannot isolate this factor. Our method ensures that such joint patterns are never seen in training, so the agent must combine operations compositionally. We expect a noticeable gap: agents perform well on our training tasks but drop significantly on test tasks that share no production rules with training, whereas on AgenticDataBench the gap is smaller or absent.

**vs. DataSciBench** — On a case requiring a group-by with a custom aggregation followed by sort, DataSciBench's tasks may contain similar sequences by chance, as splits are random. Our grammar split ensures that any test task uses a production rule never seen in training. The baseline's random assignment confounds composition novelty with other factors. Our method thus forces the agent to rely on compositional abstraction. We expect our test performance to be lower than baseline's test performance, especially on tasks with high production novelty, while training performance is comparable.

**vs. RandomSplit of CQSB** — On a case where the same filter-sort-join sequence appears in both training and test of RandomSplit, an agent may memorize the pattern, achieving high test accuracy. Our grammar split deliberately prevents such memorization by ensuring no production rule overlaps. The baseline thus measures performance inflated by instance overlap. Our method cleanly tests composition generalization. We expect a substantial drop in accuracy on our test set compared to RandomSplit's test set, while training accuracies are similar.

### What would falsify this idea

If performance on our test set is comparable to training performance, or if the performance gap between our benchmark and the random-split baseline is negligible, then the claim that our benchmark isolates compositional generalization is false.

## References

1. AgenticDataBench: A Comprehensive Benchmark for Data Agents
2. DataSciBench: An LLM Agent Benchmark for Data Science
3. BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions
4. NaturalCodeBench: Examining Coding Performance Mismatch on HumanEval and Natural User Prompts
