# MCDG-Verifier: Path-Agnostic Evaluation of Long-Horizon Agents via Minimal Causal Dependency Graphs

## Motivation

Existing long-horizon benchmarks like Long-Horizon-Terminal-Bench and SWE-EVO rely on reference solutions or fixed test suites, which cannot recognize valid alternative strategies that achieve the same goal through different subtask orderings. This structural reliance on a single correct path prevents the establishment of solution path multiplicity, limiting evaluation generality and fairness. The root cause is the assumption that a unique solution path exists, which fails for open-ended tasks where multiple valid strategies are possible.

## Key Insight

The minimal causal dependency graph (MCDG) extracted from diverse successful demonstrations identifies the essential subtask ordering constraints that all valid strategies must satisfy, providing an invariant ground truth that is independent of any specific solution path.

## Method

### (A) What it is
We propose MCDG-Verifier, a path-agnostic evaluation method that checks whether a candidate agent trajectory complies with a learned Minimal Causal Dependency Graph (MCDG). Input: a set of successful demonstrations (trajectories annotated with subtask start/end times) and a candidate trajectory. Output: a success verdict (binary) based on compliance score.

### (B) How it works (Pseudocode)
```pseudocode
function LearnMCDG(demonstrations):
    # Step 1: Segment each demonstration into subtasks using a task-specific parser (e.g., regex for log files).
    subtask_sequences = [parse(traj) for traj in demonstrations]
    
    # Step 2: For each pair of subtasks (A, B), test for causal necessity: B occurs after A in all demonstrations.
    # Use PC algorithm (Spirtes et al.) with conditional independence tests based on discrete-time occurrence.
    # Parameter: significance level alpha=0.05
    edges = []
    for A, B in ALL_SUBTASK_PAIRS:
        if A precedes B in ALL demonstrations:
            edges.append((A, B))
    
    # Step 3: Remove transitive edges to obtain minimal DAG.
    mcdg = transitive_reduction(digraph(edges))
    return mcdg

function VerifyTrajectory(candidate_traj, mcdg):
    # Parse candidate into subtask sequence
    seq = parse(candidate_traj)
    # Count satisfied dependencies
    satisfied = 0
    total = len(mcdg.edges)
    for (A, B) in mcdg.edges:
        if A occurs in seq and B occurs in seq and A precedes B:
            satisfied += 1
        elif A not in seq or B not in seq:
            # Missing subtask automatically violates dependency
            continue
    compliance = satisfied / total
    threshold = 0.9  # hyperparameter: accept if >=90% dependencies satisfied
    return compliance >= threshold

# Example usage:
mcdg = LearnMCDG(demonstrations)
verdict = VerifyTrajectory(candidate_demonstration, mcdg)
```

### (C) Why this design
We chose the PC algorithm over simple correlation or manual annotation because it provides a principled way to infer causal structure from observational data, which is essential since demonstrations are not interventional. The trade-off is computational cost: PC has O(n^2) conditional independence tests, but for typical benchmarks with <20 subtasks, this is acceptable. We set threshold 0.9 rather than 1.0 to tolerate minor ordering variations (e.g., parallel subtasks) while still requiring near-complete dependency satisfaction; the cost is that some truly invalid trajectories may pass if they exploit the slack. We use transitive reduction to keep the graph minimal, preventing over-penalization; the trade-off is that we may remove dependencies that are not strictly transitive but still important in practice (e.g., if A→B and B→C but A→C is not always needed). Finally, we require all demonstrations to contain the same subtask set; otherwise, we apply a procedure to align subtask labels, which may introduce ambiguity if demonstrations use different granularities.

### (D) Why it measures what we claim
The compliance score measures **causal necessity** because it quantifies how many of the invariant ordering constraints (edges in MCDG) the candidate trajectory respects; the assumption is that any valid strategy must respect the minimal causal ordering that holds across all successful demonstrations. This assumption fails when the demonstrations themselves are biased (e.g., all from one strategy), in which case the MCDG overfits to a particular strategy and the compliance score measures adherence to that strategy rather than universal necessity. The **total dependency count** measures the complexity of the task's causal structure; the assumption is that more dependencies indicate a more constrained task. This fails when dependencies are redundant or spurious due to limited data. The **threshold** measures the stringency of the evaluation; the assumption is that a single threshold works across tasks, but it may fail on tasks with highly varying autonomy levels. By explicitly naming these assumptions, we ensure the method does not claim universal validity without stating its boundary conditions.

## Contribution

(1) Introduces MCDG-Verifier, a path-agnostic evaluation framework for long-horizon agent tasks that checks compliance with a learned minimal causal dependency graph rather than a single reference solution. (2) Establishes that causal structure inference from diverse successful demonstrations provides a reliable, path-invariant ground truth for verification, as demonstrated on benchmarks like Long-Horizon-Terminal-Bench. (3) Provides a concrete methodology for constructing minimal causal dependency graphs from agent trajectories using the PC algorithm and transitive reduction, with explicit design trade-offs and assumption analysis.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Long-Horizon-Terminal-Bench | Dense reward provides ground truth. |
| Primary metric | Verification accuracy | Measures correctness of binary verdict. |
| Baseline 1 | Subtask set completeness | Checks all subtasks present. |
| Baseline 2 | Exact sequence matching | Requires fixed subtask order. |
| Ablation | MCDG-Verifier w/o transitive reduction | Tests necessity of minimal graph. |

### Why this setup validates the claim

The setup uses Long-Horizon-Terminal-Bench, which provides dense reward as ground truth for trajectory success, enabling accurate evaluation of verifier correctness. Comparing against two baselines—subtask set completeness (order-agnostic) and exact sequence matching (order-rigid)—isolates the effect of capturing causal necessity. Our method should outperform both on average, especially on trajectories that reorder subtasks while respecting constraints. The ablation (without transitive reduction) tests whether the minimal graph reduces false negatives without sacrificing true positives. Verification accuracy directly measures the central claim of correctly classifying valid/invalid trajectories, making the experiment a falsifiable test of causal necessity estimation.

### Expected outcome and causal chain

**vs. Subtask set completeness** — On a case where a candidate includes all subtasks but in a causally invalid order (e.g., opening a door before unlocking it when unlocking must precede opening), the baseline reports success because it only checks presence. Our method correctly rejects it because the dependency (unlock→open) is violated. Thus we expect a noticeable gap: our method will have higher specificity (correctly identify invalid trajectories) while maintaining similar sensitivity.

**vs. Exact sequence matching** — On a case where a candidate reorders subtasks in a way that respects all dependencies (e.g., cleaning before making bed when no dependency exists), the baseline requires the exact sequence from demonstrations and thus fails. Our method accepts it because all dependencies are satisfied. Hence we expect our method to have higher recall (correctly accept valid trajectories) while maintaining precision.

**vs. MCDG-Verifier w/o transitive reduction** — On a case where transitive reduction removes a spurious dependency (e.g., A→C always holds due to A→B→C), the ablation retains the edge A→C and may penalize a valid trajectory that satisfies A→B and B→C but not A→C directly (though it is implied). Our method, with transitive reduction, accepts such trajectories. Thus we expect our method to have higher recall on tasks with transitive chains, while precision remains similar.

### What would falsify this idea

If the gain in accuracy over baselines is uniform across all trajectory types rather than concentrated on trajectories that reorder subtasks while respecting dependencies, the claim that our method measures causal necessity would be unsupported.

## References

1. Long-Horizon-Terminal-Bench: Testing the Limits of Agents on Long-Horizon Terminal Tasks with Dense Reward-Based Grading
2. CUARewardBench: A Benchmark for Evaluating Reward Models on Computer-using Agent
3. SWE-EVO: Benchmarking Coding Agents in Long-Horizon Software Evolution Scenarios
4. Process Reward Models That Think
5. The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs
6. HCAST: Human-Calibrated Autonomy Software Tasks
7. RM-Bench: Benchmarking Reward Models of Language Models with Subtlety and Style
8. OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments
