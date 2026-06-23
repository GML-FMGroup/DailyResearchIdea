# Consistency-Driven Adaptive Verification for Multi-Agent Orchestration under Distribution Shift

## Motivation

Existing multi-agent orchestration frameworks, such as Verified Multi-Agent Orchestration (VMAO), rely on a static LLM verifier that assumes fixed reliability, while fuzzy evaluation modules in systems like Neural Orchestration assume static accuracy. Both fail under distribution shift because verifier errors remain undetected, leading to unverified reasoning. The root cause is the lack of a signal that verifiers are operating outside their training distribution, which can be provided by monitoring consistency of verification scores across subtasks.

## Key Insight

The DAG decomposition in orchestration naturally produces subtask verification scores that are consistent under in-distribution conditions, so distribution shift can be detected by measuring variance increase across these scores without requiring ground-truth labels.

## Method

### Consistency-Driven Adaptive Verification (CDAV)

**Assumption (explicit):** In-distribution, the LLM verifier's scores across subtasks of a decomposed query have low variance (i.e., consistency). Distribution shift increases variance beyond a calibrated threshold.

**(A) What it is**: Consistency-Driven Adaptive Verification (CDAV) extends a plan-execute-verify-replan orchestration framework (e.g., VMAO) by monitoring the variance of LLM-verifier scores across subtasks. When variance exceeds an adaptive threshold (calibrated via a held-out validation set), external verification (tool or human) is triggered and verifier weights are updated online. Input: DAG of subtask responses from execution. Output: final answer with adaptive verifier confidence.

**(B) How it works** (pseudocode):
```python
# Hyperparameters: α (learning rate, default 0.1), window_size (default 10), initial_threshold (default 0.5)
# State: verifier_weights (list of floats, one per subtask index type, initialized to 1.0)
# Calibration set: 100 in-distribution queries, each with 5-10 subtasks; compute per-query std of verifier scores, then compute μ_cal (mean of stds) and σ_cal (std of stds).
# Historical consistency buffer: buffer = deque(maxlen=window_size)

def cdav(subtask_responses, dag_structure, llm_verifier):
    # Step 1: Compute verification scores for each subtask
    scores = []
    for node in dag_structure.topological_order():
        response = subtask_responses[node]
        score = llm_verifier.evaluate(response, context=node.prompt)  # returns float in [0,1]
        scores.append(score)
    
    # Step 2: Measure consistency as standard deviation
    consistency = np.std(scores) if len(scores) > 1 else 0.0
    buffer.append(consistency)
    
    # Step 3: Adaptive threshold using z-score based on calibration set
    if len(buffer) >= 1:
        z_score = (consistency - μ_cal) / (σ_cal + 1e-8)
        triggered = z_score > 2.0  # margin factor of 2 standard deviations
    else:
        triggered = False
    
    # Step 4: Detect shift and trigger external verification
    if triggered:
        external_scores = []
        for node in dag_structure.topological_order():
            response = subtask_responses[node]
            # Query external verifier (e.g., ToolLLM API, human) -> returns score
            ext_score = external_verifier.evaluate(response, context=node.prompt)
            external_scores.append(ext_score)
        # Update verifier weights via gradient descent on MSE between internal and external
        for i, (int_score, ext_score) in enumerate(zip(scores, external_scores)):
            error = ext_score - int_score
            verifier_weights[i] = np.clip(verifier_weights[i] + α * error, 0.1, 5.0)
        # Use external scores as final verification (or blend)
        final_verdict = orchestrate_replan(external_scores)  # place for replanning
    else:
        final_verdict = orchestrate_replan(scores)
    return final_verdict
```

**(C) Why this design**: We chose standard deviation over entropy as a consistency metric because variance is directly interpretable as dispersion in continuous scores, and many LLM verifiers output continuous Likert scores; entropy would require discretization, losing sensitivity. We set a margin factor of 2.0 over the running mean to avoid false positives from natural noise; the trade-off is delayed detection of small shifts. We updated verifier weights per subtask type (e.g., question type) rather than per instance to avoid overfitting to noise; this assumes shared error patterns across tasks, which may not hold if subtasks are highly heterogeneous. We opted for a calibrated z-score using a held-out validation set instead of a simple moving average to ground the threshold in in-distribution statistics; the cost is requiring a representative calibration set. Finally, we used full external verification in triggered cases rather than selective sampling to ensure all subtasks are corrected; this increases cost but guarantees verifier calibration across the entire DAG.

**(D) Why it measures what we claim**: The standard deviation of subtask verification scores measures distribution shift under the assumption that, given an in-distribution query, a calibrated verifier should produce consistent scores across subtask responses because they are generated from independent yet similarly challenging subproblems; this assumption fails when subtasks have intrinsically varying difficulty (e.g., some require factual knowledge, others reasoning), in which case variance reflects inherent difficulty differences rather than distribution shift. **Equivalence:** high variance → distribution shift. **Assumption:** verifier's score consistency is stationary in-distribution. **Failure mode:** intrinsic difficulty differences cause variance. The adaptive threshold ensures that only variance exceeding a calibrated baseline (2σ above μ_cal) triggers external verification; this operationalizes the concept of 'consistency breach' under the assumption that in-distribution variance is stationary; if the environment changes quickly, the calibration set may become outdated, causing delayed detection. External verification scores provide a ground-truth proxy for updating verifier weights; this assumes the external verifier is unbiased and accurate, failing when it is also miscalibrated or expensive, in which case the update may introduce bias or be infeasible. The weight update reduces the gap between internal and external scores, thereby improving verifier calibration; this assumes that the error is systematic and can be corrected by a linear scaling, which may not hold for nonlinear biases.

**Resource estimate (API calls and time):** Each query incurs |subtasks| internal LLM verifier calls (typically 5-10). When triggered, an additional |subtasks| external verifier calls are made (e.g., human or tool API). For a 500-query dataset with 100 held out, we estimate 400 test queries × 10 internal calls = 4,000 internal API calls. Assuming a trigger rate of ~20%, that adds 400 × 0.2 × 10 = 800 external calls. Internal calls take ~1 second each, external calls (if human) take ~10 seconds each. Total time: ~1.2 hours (internal) + ~2.2 hours (external) ≈ 3.4 hours. Code repository: https://github.com/example/cdav

## Contribution

(1) A consistency-driven mechanism that detects distribution shift in multi-agent orchestration by monitoring variance of LLM verifier scores across DAG subtasks. (2) An online weight update procedure that recalibrates the verifier using external verification triggered by consistency breaches, enabling adaptive reliability without continuous human oversight. (3) A design principle: DAG decomposition naturally provides a consistency signal for verifier trustworthiness.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Expert-curated market research queries (500 queries, each requiring 5-10 subtasks, with 100 held out for calibration) | Multi-step, requires decomposition and verification |
| Primary metric | Final answer accuracy (1-5 scale) | Captures overall correctness and completeness |
| Secondary metric | Trigger rate (fraction of queries with high variance) | Measures detection sensitivity |
| Baseline 1 | Single-agent LLM | No decomposition or verification |
| Baseline 2 | VMAO (fixed verification) | Tests adaptive threshold benefit |
| Baseline 3 | Random agent selection | Tests orchestration vs. random selection |
| Baseline 4 | Non-adaptive threshold (fixed threshold = 1.0) | Isolates benefit of adaptive threshold |
| Ablation | CDAV without online weight update | Isolates effect of adaptive threshold only |

### Why this setup validates the claim

This combination of dataset, baselines, and metric forms a falsifiable test of the central claim that consistency-driven adaptive verification improves orchestration reliability. The market research queries require decomposing complex questions into subtasks with varying difficulty, where distribution shift can occur. To directly validate that variance correlates with shift, we include a synthetic distribution shift experiment: we take 100 in-distribution queries and artificially inject errors into subtask responses (e.g., replace a factual subtask with a random response). We then measure whether variance increases significantly (p<0.05) compared to unperturbed queries. This experiment uses the same verification pipeline and establishes that our detection method is sensitive to controlled shifts. Comparing against VMAO (which uses fixed verification) isolates the adaptive threshold effect; comparing against single-agent tests the necessity of decomposition and verification; comparing against random agent selection tests whether orchestration provides benefit beyond random assignment; the non-adaptive threshold baseline shows the value of adaptation. The ablation removes online weight updates, isolating the contribution of adaptive threshold detection. The accuracy metric directly reflects the quality of the final answer, which is the ultimate goal. If our method outperforms VMAO primarily on queries where subtask score variance exceeds the adaptive threshold, the claim that consistency detection drives improvement is supported.

### Expected outcome and causal chain

**vs. Single-agent LLM** — On a query with multiple interdependent subproblems (e.g., market trends and competitor analysis), the single-agent baseline produces a vague or incomplete answer because it lacks decomposition and verification, leading to missing details or hallucination. Our method decomposes the query, verifies each subtask, and adaptively triggers external verification when subtask scores are inconsistent, ensuring each subproblem is accurately addressed. Thus, we expect a large accuracy gap (e.g., 1.5-2 points on the 5-point scale) favoring CDAV, with the gap concentrated on queries requiring multi-step reasoning.

**vs. VMAO** — On a query where subtasks have inherent difficulty variation (e.g., one subtask requires factual recall, another requires reasoning), VMAO's fixed verifier scores produce high variance, mistakenly triggering costly external verification or missing actual distribution shift. Our adaptive threshold uses a running baseline of consistency, so natural variance from difficulty differences is tolerated, while actual shifts (e.g., sudden topic change) trigger correction. Consequently, we expect CDAV to show comparable or slightly better accuracy on routine queries but a noticeable advantage (e.g., 0.5-1 point) on queries with subtle distribution shifts, where VMAO either over-verifies (wasting resources) or under-verifies (reducing accuracy).

**vs. Random agent selection** — On a query requiring specific expertise (e.g., legal knowledge), random selection may assign the wrong agent to a subtask, leading to low-quality responses. Our orchestration uses verification scores to guide execution, but the method does not directly select agents; however, the DAG structure ensures appropriate subtask ordering. Random selection baseline ignores subtask dependencies, producing inconsistent answers. We expect CDAV to outperform random selection significantly (e.g., 1-2 point accuracy gap) across all queries, as orchestration provides structured decomposition.

**vs. Non-adaptive threshold** — On a query with moderate distribution shift, a fixed high threshold (1.0) may fail to detect the shift because the baseline variance is already high in-distribution; our adaptive threshold calibrated to the validation set can distinguish shift from inherent variance. We expect CDAV to show a higher trigger rate on truly shifted queries and lower false alarm rate on in-distribution queries, leading to better accuracy when external verification is applied judiciously.

### What would falsify this idea

If CDAV's accuracy gain over VMAO is uniform across all queries (rather than concentrated on those where subtask variance exceeds the calibrated threshold), then the central claim that consistency-driven detection is the key mechanism would be falsified. Specifically, if the improvement is equally large on queries with low variance, the benefit must come from another component (e.g., weight update) rather than adaptive thresholding.

## References

1. Verified Multi-Agent Orchestration: A Plan-Execute-Verify-Replan Framework for Complex Query Resolution
2. Tree of Thoughts: Deliberate Problem Solving with Large Language Models
3. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
4. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
5. Self-Instruct: Aligning Language Models with Self-Generated Instructions
6. Faithful Reasoning Using Large Language Models
7. Neural Orchestration for Multi-Agent Systems: A Deep Learning Framework for Optimal Agent Selection in Multi-Domain Task Environments
