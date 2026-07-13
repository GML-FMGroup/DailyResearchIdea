# Causal Subgoal Equivalence: A Faithfulness Metric for Agent Evaluation

## Motivation

Current benchmarks like UniClawBench use fixed checkpoints, penalizing agents that achieve the same outcome via different paths. The root cause is that evaluation assumes a linear step sequence, ignoring the causal structure of tasks. This structurally fails to credit diverse strategies that respect necessary prerequisites.

## Key Insight

The partial order of subgoals derived from task causality defines an invariant structure; any trajectory that respects this partial order is equally valid, so faithfulness to the task is equivalent to respecting subgoal dependencies, not exact steps.

## Method

### Causal Subgoal Equivalence Metric (CSEM)

**Assumption:** The causal subgoal graph G is correctly specified and complete, capturing all necessary prerequisite dependencies. If this assumption is violated, the metric may penalize valid trajectories.

We define CSEM as follows:

```python
def csem(task_causal_graph_G, trajectory_T, subgoal_detector_phi):
    # Step 1: Extract completed subgoals from T
    completed = set()
    for state in T:
        for subgoal in G.nodes:
            if phi(state, subgoal):
                completed.add(subgoal)
    # Step 2: Compute completion ratio (only required subgoals; optional subgoals are excluded)
    required_subgoals = [s for s in G.nodes if G.nodes[s].get('required', True)]
    C = len(completed & set(required_subgoals)) / len(required_subgoals) if required_subgoals else 1.0
    # Step 3: Count violations of prerequisite edges
    violations = 0
    for (a, b) in G.edges:  # a must be completed before b
        # Check if both a and b are completed
        if a in completed and b in completed:
            # Find first occurrence in T
            first_a = min(i for i, s in enumerate(T) if phi(s, a))
            first_b = min(i for i, s in enumerate(T) if phi(s, b))
            if first_b < first_a:  # b completed before a -> violation
                violations += 1
    # Step 4: Faithfulness score
    F = 1 - violations / len(G.edges) if len(G.edges) > 0 else 1
    # Step 5: Combine
    score = F * C
    return score
```

**Hyperparameters**: None; the metric is deterministic given G and phi.

**Subgoal detector phi**: For each subgoal, phi is implemented as a 2-layer MLP with hidden size 64, ReLU activation, trained on synthetic data to 95% accuracy. For simple subgoals (e.g., light_on), a rule-based detector using simulator state is used.

**Why this design**: We chose state abstraction via subgoal detectors over human-annotated checkpoints because it allows arbitrary agent paths; the trade-off is that we need reliable subgoal detectors, which may require extra engineering but avoids forcing a step sequence. We use a violation count rather than sequence edit distance because edit distance penalizes reordering even if all prerequisites are met; the trade-off is that violations only catch dependency-breaking orders, allowing full freedom otherwise. We combine completion ratio to ensure agents actually finish required subgoals; the trade-off is that it penalizes partial progress, which is acceptable because our metric focuses on structural faithfulness. Optional subgoals are designated in G via a 'required' flag and are excluded from the completion ratio.

**Why it measures what we claim**: The violation count operationalizes "causal faithfulness" because each prerequisite edge in G represents a necessary ordering — if an agent violates it, the trajectory is impossible in the true task; this assumption fails when the causal graph is incomplete or incorrectly specified (e.g., missing edges), in which case the metric may penalize valid trajectories. The completion ratio operationalizes "task completion" because a subgoal not completed means the task output is incomplete; this assumption fails when subgoals are not independent (e.g., some may be optional), in which case we require a specification of required vs optional subgoals. Required subgoals are defined as those without which the final goal cannot be achieved. If subgoals are independent and required, completion ratio equals the fraction of task completeness.

## Contribution

(1) A novel evaluation metric (CSEM) that measures agent faithfulness to task causal structure rather than fixed step sequences. (2) A design principle: state equivalence classes based on completed subgoals enable unbiased evaluation of diverse strategies. (3) An open-source tool for analyzing agent failures as either causal violations or incomplete subgoal coverage.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | MiniGrid (tasks: DoorKey, Unlock, MultiTask) with ground-truth causal subgoal graphs | Causal graph available; subgoals well-defined; tasks with varying independence structure |
| Primary metric | CSEM (our method) | Measures structural faithfulness and completion |
| Baseline 1 | Step completion rate (SCR) | Matches human-annotated step sequence; penalizes reordering |
| Baseline 2 | Success rate (SR) | Only final goal; ignores dependencies |
| Baseline 3 | Subgoal sequence edit distance (SED) | Edit distance on subgoal order; penalizes any reordering |
| Ablation-of-ours | CSEM without completion ratio (also tested with subgoal detector noise at 10% FP/FN) | Isolates faithfulness component and assesses robustness |

### Why this setup validates the claim

The combination of MiniGrid tasks with ground-truth causal subgoal graphs allows direct assessment of whether CSEM faithfully reflects prerequisite compliance and task completion. Comparing against SCR (which penalizes any reordering), SR (which ignores intermediate structure), and SED (which penalizes all swaps) isolates the effect of CSEM's selective attention to causal edges. The ablation tests the necessity of the completion component. Noise variations in phi (false positives/negatives at 10%) test robustness. The primary metric CSEM is the proposed measure; we expect it to diverge from baselines on tasks where agents achieve the goal but violate prerequisites or skip subgoals. This setup provides a falsifiable test: if CSEM correlates perfectly with SED, then the causal filtering adds no value.

### Expected outcome and causal chain

**vs. Step completion rate (SCR)** — On a task with independent subgoals (e.g., open fridge and close window, both required), an agent completing them in reverse order of the fixed script gets lower SCR but perfect CSEM (no prerequisite violations). We expect CSEM > SCR on such tasks, and similar on strictly ordered tasks.

**vs. Success rate (SR)** — On a case where an agent opens a door by brute force without unlocking (final goal achieved), SR gives full credit (1.0). CSEM detects missing subgoal via completion ratio, giving a lower score. We expect CSEM < SR on incomplete tasks, and equal on fully completed tasks.

**vs. Subgoal sequence edit distance (SED)** — On a task where an agent completes subgoals in a different but permissible order (respecting prerequisites), SED penalizes all swaps. CSEM ignores legal reorderings, producing a higher score. We expect CSEM > SED on tasks with independent subgoals, and similar on strictly ordered tasks.

**vs. Ablation (CSEM without completion ratio)** — On a case where an agent satisfies all prerequisites but misses a final subgoal (e.g., picks up key, unlocks door but does not open it), the ablation returns a perfect faithfulness score. Full CSEM penalizes the missing subgoal. We expect full CSEM < ablation on incomplete tasks, and equal on complete tasks. Under subgoal detector noise (10% FP/FN), we expect the ablation to be less sensitive than full CSEM because completion ratio amplifies detector errors.

### What would falsify this idea

If CSEM shows no systematic difference from SED across tasks with varying causal structure (correlation near 1), or if its performance on incomplete tasks is indistinguishable from its ablation, then the central claim that CSEM captures both completion and causal faithfulness would be invalidated. Additionally, if CSEM scores degrade catastrophically under 10% subgoal detector noise (e.g., >20% change in mean score), then reliance on phi undermines practical applicability.

## References

1. UniClawBench: A Universal Benchmark for Proactive Agents on Real-World Tasks
2. AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents
3. AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents
4. ASSISTGUI: Task-Oriented Desktop Graphical User Interface Automation
5. CogAgent: A Visual Language Model for GUI Agents
6. ToolTalk: Evaluating Tool-Usage in a Conversational Setting
7. Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding
8. AnyTOD: A Programmable Task-Oriented Dialog System
