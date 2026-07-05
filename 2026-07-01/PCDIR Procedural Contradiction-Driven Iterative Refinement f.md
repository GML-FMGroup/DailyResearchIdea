# PCDIR: Procedural Contradiction-Driven Iterative Refinement for Agent Self-Correction

## Motivation

Existing methods like AFTER (Zhang et al., 2024) rely on single-round external feedback to refine procedural memory, but such feedback is often incomplete or noisy, leading to brittle improvements. The root cause is that agents lack a mechanism to detect when their current behavior contradicts what they have learned, making iterative refinement without external supervision impossible. This limitation stems from treating procedural memory as a static repository rather than an active consistency oracle.

## Key Insight

Procedural memory encodes a set of action sequences that have been successful in previous contexts; a mismatch between the agent's current action and the memory's predicted action at a given state signals a likely mistake, providing a self-contained error signal for targeted refinement.

## Method

### (A) What it is
PCDIR (Procedural Contradiction-Driven Iterative Refinement) is a self-supervised framework that takes as input an agent's current procedural memory (a collection of skill-specific action sequences) and a task execution trace, and outputs an updated procedural memory refined to reduce contradictions detected during execution. The core operation is contradiction detection: at each state, the memory predicts an expected action; if the actual action diverges, a contradiction is flagged, and the memory is updated to resolve it. **Load-bearing assumption:** The expected action from procedural memory is assumed to be more reliable than the agent's actual action, but to avoid imprinting noise, updates are only performed when the actual action leads to a positive outcome (task success).

### (B) How it works
```python
# PCDIR procedure (online refinement after each task)
# Hyperparameters: min_contradiction_count=2, success_gate=True, skill_encoder_dim=256
Input: agent (with procedural memory M), task description T
Output: refined memory M'

for task T:
    execution_trace = []
    state = env.reset(T)
    while not done:
        # Step 1: predict expected action from M
        # Uses a 2-layer MLP skill decoder (hidden=256, ReLU) to predict expected action from state embedding
        expected_action = M.predict_action(state)  # returns action if confident, else None
        
        # Step 2: agent selects actual action (via LLM or policy)
        actual_action = agent.select_action(state, T)
        
        # Step 3: detect contradiction
        if expected_action is not None and actual_action != expected_action:
            contradiction = (state, expected_action, actual_action)
            execution_trace.append(contradiction)
        else:
            execution_trace.append((state, actual_action)) # normal step
        
        # step 4: execute action and get reward
        state, reward, done = env.step(actual_action)
    
    # Step 5: refine M based on contradictions, only if reward positive (success)
    if success_gate and reward <= 0:
        continue  # skip update on failure to avoid imprinting noise
    for each contradiction (s, a_expected, a_actual):
        # identify which skill produced the expected action via nearest-neighbor retrieval over skill centroids (L2 distance on state embedding from frozen encoder)
        skill = M.identify_skill(s)  # returns skill ID
        # update skill sequence: add branch (state-specific deviation) with count
        M.update_skill(skill, s, a_actual)  # stores (s, a_actual) in skill's branching tree; if branch exists, increment count
        # optionally, verify consistency with other skills by checking if two skills predict different actions for the same state
        if M.has_inconsistencies(skill):  # checks for conflicts with other skills
            M.resolve_global_conflict()  # weighted majority voting based on skill success counts; if tie, keep original
    
    M = M_refined
return M
```

### (C) Why this design
We chose contradiction detection over uncertainty-based gating (e.g., confidence thresholds) because the latter suffers from poor calibration (ANTI-PATTERN 3): LLM log-probs for action correctness are unreliable. Contradiction detection instead exploits the structural property that if memory has been learned from successful traces, its predictions are generally reliable; deviations likely indicate errors. The trade-off is that if memory itself is noisy, contradictions may be false positives, hurting refinement efficiency. We mitigate this by requiring multiple contradictions for the same state across episodes before updating (a hyperparameter: min_contradiction_count=2). We also chose to update memory by directly inserting the actual action as a new variant for the specific state, rather than overwriting the expected action, to preserve original knowledge. This design follows from the observation that skills often have multiple valid action branches; overwriting would lose correct alternatives. The cost is memory growth, acceptable given bounded task diversity. Finally, we use a contradiction buffer aggregated over episodes to avoid premature updates from rare stochastic actions; this trades off latency for robustness. These choices together ensure that updates are conservative yet corrective when patterns emerge.

### (D) Why it measures what we claim
The contradiction signal between `expected_action` (from memory) and `actual_action` measures *behavioral inconsistency* because we assume that memory encodes successful action sequences from prior experience; a deviation thus indicates a likely mistake in either memory or action. This assumption fails when memory is inaccurate (e.g., containing hallucinated steps from a single noisy trace), in which case contradictions measure *memory error* rather than agent error. The update that replaces the expected action with the actual action (when the actual leads to success) measures *memory correction* because we assume the actual action is more reliable in the current context, which holds only if the actual action is from a high-confidence source (e.g., human demonstration or multiple successful trials). When actual actions are also noisy (e.g., from LLM self-generation), the update measures *noise imprinting*. The `min_contradiction_count` hyperparameter operationalizes *pattern consistency*: requiring multiple occurrences screens out spurious deviations, under the assumption that repeated mismatches indicate systematic issues rather than random variation; this fails when systematic errors are rare (e.g., hard-to-reproduce bugs), in which case the metric reflects confirmation bias (ignoring isolated but critical missteps). The `resolve_global_conflict` step across skills measures *memory coherence* because we detect when two skills predict different actions for the same state; this assumes the state representation is sufficiently discriminative to assign unique skill contexts, which fails when states are ambiguous (e.g., overlapping tasks), leading the metric to measure *representation collision*.

**Calibration experiment:** On a held-out set of 100 episodes, compute precision and recall of contradiction as a predictor of actual task failure, where a contradiction is considered a true positive if the task eventually fails. We measure precision and recall across varying min_contradiction_count thresholds (1, 2, 3). This experiment validates that contradiction count measures error likelihood under the assumption that memory is mostly correct; if precision is low, the contradiction signal is noisy.

Implementation details and code will be released at [anonymous link].

## Contribution

(1) A novel framework, PCDIR, that enables agents to iteratively self-correct their procedural memory using contradictions between memory predictions and actual actions as an internal error signal. (2) The design principle that procedural memory can serve as a consistency oracle for self-supervised refinement, eliminating reliance on external feedback. (3) A taxonomy of contradiction types (state-mismatch, skill-conflict) and their differential impact on refinement effectiveness, derived from empirical analysis.

## Experiment

### Evaluation Setup
| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | ToolBench (20 tasks) and MetaWorld (10 tasks) | Diverse skill sequences; real-world robotic manipulation for generality |
| Primary metric | Task success rate | Practical performance measure |
| Additional metric | Contradiction-failure correlation (Pearson r) | Validates that contradictions measure genuine errors |
| Baseline 1 | No refinement | Tests if memory updates help |
| Baseline 2 | Uncertainty-gated refinement | Tests anti-pattern: confidence-based update |
| Baseline 3 | Overwrite-only (no branching) | Tests benefit of preserving variants |
| Baseline 4 | Self-consistency refinement (no memory) | Isolates contribution of memory-based contradiction detection |
| Ablation of ours | Single-contradiction update | Tests impact of min_contradiction_count |

### Why this setup validates the claim
This setup tests the central claim that contradiction-driven refinement corrects behavioral inconsistencies in procedural memory. The ToolBench dataset provides diverse tool manipulation tasks requiring multi-step sequences with clear success/failure outcomes, enabling measurement of practical benefit via task success rate. MetaWorld extends to continuous robotic control, testing generalizability. The contradiction-failure correlation metric directly links the contradiction signal to actual errors. The no-refinement baseline establishes default performance. Uncertainty-gated refinement tests whether contradiction detection is preferable over confidence-based updating, which suffers from calibration issues. Overwrite-only tests whether branching preserves knowledge across multiple valid strategies. Self-consistency (LLM self-check without memory) isolates the improvement from memory structure. The ablation—removing the minimum contradiction count—assesses robustness to noise. If our method outperforms all baselines and shows positive contradiction-failure correlation, it validates that contradiction detection with branching and thresholding effectively corrects real errors while avoiding spurious updates.

### Expected outcome and causal chain
**vs. No refinement** — On a case where the initial memory contains a wrong action ordering (e.g., "use hammer before screwdriver" but the correct order is reversed), the baseline repeats the error indefinitely because it never updates. Our method instead detects a contradiction when the actual action (correct) differs from memory’s expected action (wrong), and updates memory to reflect the correct order for that state. We expect a significant gap (>20% higher success) on tasks where initial memory has errors, but parity on tasks where memory is already correct.

**vs. Uncertainty-gated refinement** — On a case where the agent is uncertain but correct (e.g., a novel state where both possible actions lead to success, but the memory’s expected action is one of them), the baseline might update memory because low confidence triggers an update, even though the actual action matches the expected action (no contradiction). This can incorrectly replace a valid expected action with a different correct action, reducing consistency. Our method only updates on contradictions (expected ≠ actual) and only after success, so no update occurs here. We expect our method to maintain higher success (e.g., ~15% advantage) on tasks with ambiguous states where uncertainty and correctness are misaligned.

**vs. Overwrite-only (no branching)** — On a case where a state has two equally valid actions (e.g., using either a wrench or a plier works), overwrite would replace the memory’s expected action (e.g., wrench) with the actual action (e.g., plier) upon first contradiction, losing the knowledge that wrench is also valid. Our method adds a branch, preserving both. Over time, overwrite loses coverage, leading to failures on tasks requiring the overwritten action. We expect our method to outperform by 10-15% on tasks with multiple viable strategies.

**vs. Self-consistency refinement (no memory)** — On a case where the agent’s initial memory is correct but the LLM makes a stochastic error (e.g., hallucinates a tool), self-consistency may still produce a wrong action via majority voting if the error is rare. Our method directly compares to memory’s expected action, which is based on past successful traces, thus more robust. We expect a ~10% advantage on tasks where LLM self-consistency is misled by rare hallucinations.

**Ablation (single-contradiction update)** — On a case where a random noise action (e.g., accidental drop) causes a single contradiction, the ablation updates memory immediately, encoding the noise as a spurious branch. Our method’s minimum count filter (e.g., 2) ignores isolated deviations. We expect the ablation to show higher variance and lower average success (e.g., 5-10% worse) on tasks with stochastic elements.

**Correlation analysis** — We expect a positive Pearson r (e.g., r > 0.5) between contradiction count per state and eventual task failure, confirming that contradictions proxy errors. Failure mode: low correlation would indicate contradictions are noisy.

### What would falsify this idea
If our method does not outperform uncertainty-gated refinement on tasks with miscalibrated uncertainty, or if the performance gain is uniform across all subsets rather than concentrated where predicted failure modes (e.g., multiple valid actions, stochastic noise) occur, or if the contradiction-failure correlation is near zero, then the central claim that contradiction-driven refinement corrects true behavioral inconsistencies is false.

## References

1. Managing Procedural Memory in LLM Agents: Control, Adaptation, and Evaluation
2. From Exploration to Mastery: Enabling LLMs to Master Tools via Self-Driven Interactions
3. MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering
4. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
5. ML-Bench: Evaluating Large Language Models and Agents for Machine Learning Tasks on Repository-Level Code
6. CLOVA: A Closed-LOop Visual Assistant with Tool Usage and Update
7. Confucius: Iterative Tool Learning from Introspection Feedback by Easy-to-Difficult Curriculum
8. PAL: Program-aided Language Models
