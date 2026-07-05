# IterativeRefine: A Multi-Round Evaluation Protocol for Procedural Memory in LLM Agents

## Motivation

Existing evaluation protocols for procedural memory, such as the one in the AFTER benchmark, assess agent performance based on a single attempt per task, which underestimates an agent's capacity to improve through practice. The root cause is that these benchmarks treat each task as an isolated event, ignoring the structural property that procedural memory tasks naturally permit repeated practice. Consequently, they cannot distinguish between agents that succeed by luck and those that demonstrate consistent improvement.

## Key Insight

Self-generated feedback across multiple attempts provides a natural intrinsic signal that proxies for learning progress, enabling evaluation to disentangle stable skill acquisition from stochastic success.

## Method

(A) **What it is**: IterativeRefine is a multi-round evaluation protocol for procedural memory tasks. It takes an LLM agent and a task, and outputs a skill stability score and an aggregate performance metric over a consistent set of attempts. The protocol incorporates an external verification step to filter self-feedback, mitigating cases where self-generated feedback is unhelpful. **Assumption**: Self-generated feedback from previous attempts leads to meaningful improvement. To verify, we use a small critic model (e.g., fine-tuned T5 or a separate LLM with low compute) to assess feedback quality; if the critic assigns a confidence score below 0.7, the feedback is discarded and the agent uses a default prompt (e.g., "Analyze your previous output and identify correctable errors") instead.

(B) **How it works** (pseudocode):
```python
def iterative_refine(agent, task, max_rounds=10, threshold=0.9, critic=None):
    attempts = []
    for round in range(1, max_rounds+1):
        output = agent.execute(task)
        score = compute_task_score(output, task.ground_truth)  # e.g., accuracy
        if round == 1:
            feedback = ""  # initial attempt has no prior feedback
        else:
            raw_feedback = generate_self_feedback(agent, task, attempts[:-1])
            # External verification: critic scores feedback quality
            if critic is not None:
                confidence = critic.evaluate_feedback(raw_feedback)  # e.g., 0-1 score
                if confidence < 0.7:
                    feedback = "Analyze your previous output and identify correctable errors"  # default
                else:
                    feedback = raw_feedback
            else:
                feedback = raw_feedback
        agent.refine(feedback)  # agent updates its skill based on feedback
        attempts.append((output, score))
        if round >= 3:  # start consistency check after 3 rounds
            scores = [s for (_, s) in attempts[-3:]]
            if max(scores) - min(scores) < 0.1 * threshold:  # stability condition
                break
    stable_attempts = attempts[-3:]  # last 3 or until stability
    final_score = mean([s for (_, s) in stable_attempts])
    return final_score, stable_attempts
```

(C) **Why this design**: We chose to use a sliding window of the last 3 attempts for consistency over a single global threshold because it captures the recent trajectory without averaging in early poor attempts, accepting a small sample size that may miss long-term drift. We opted for agent-generated self-feedback rather than external verifier feedback because it preserves the autonomous nature of evaluation and avoids introducing a separate supervisory model, at the cost of potential hallucination or unproductive feedback. The stability condition uses relative difference (max-min) rather than variance because it is more interpretable and less sensitive to scale, but it may prematurely stop if scores plateau early due to randomness. These trade-offs prioritize autonomy and simplicity over statistical rigor, aligning with the goal of evaluating agents' intrinsic learning capacity.

(D) **Why it measures what we claim**: The score from the stable set of attempts measures the agent's stable skill level because the consistency condition ensures that performance has plateaued, indicating that further attempts are unlikely to yield improvement; this assumption fails if the agent's improvement is non-monotonic and the plateau is temporary, in which case the metric may reflect a local optimum rather than true mastery. The self-feedback generation measures the agent's ability to diagnose its own errors because it requires the agent to introspect on previous outputs; this assumption fails if the agent generates generic or uninformative feedback, in which case the metric primarily measures initial skill rather than improvement capacity. The number of rounds to stability measures learning speed because it quantifies how quickly the agent adapts; this assumption fails if the agent simply reproduces the same output without learning, in which case the rounds measure persistence rather than speed.

(E) **Assumptions and failure modes**: 
- **Self-feedback leads to improvement**: We assume the agent can generate corrective feedback that improves future attempts. Failure mode: Self-Refine (Madaan et al., 2023) shows self-feedback can be unhelpful or degrade performance. Mitigation: External critic filters low-quality feedback (confidence <0.7). 
- **Stability condition (low variance over 3 attempts) implies true plateau**: We assume that when scores stabilize, the agent has reached its maximum skill. Failure mode: Non-monotonic learning (e.g., forgetting) or random noise cause premature stopping. Mitigation: If the protocol stops before round 10 but scores are low (<0.5), we flag the task as potentially noisy and continue to max_rounds. The default threshold of 0.1*threshold (i.e., 0.09) is calibrated on a small dev set of 5 tasks to minimize false positives.

## Contribution

(1) A multi-round evaluation protocol, IterativeRefine, that captures iterative refinement through self-feedback and consistency checking. (2) The finding that the number of rounds to stability and the stable score together provide a more informative assessment of procedural memory than single-round metrics. (3) A demonstration that agents' performance can be underestimated by single-round evaluations, highlighting the need for dynamic evaluation protocols.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AFTER benchmark (20 tasks: tool use, database query, code debugging, etc.) | Diverse procedural tasks for generalizability |
| Primary metric | Skill stability score (mean of stable attempts) | Measures final stable performance after refinement |
| Baseline: Single-shot | No refinement; one execution per task | Measures initial skill only |
| Baseline: Fixed-round external feedback | 10 rounds with GPT-4 as external verifier providing corrective feedback | Tests if self-feedback is competitive with supervised alternative |
| Baseline: Fixed-round self-feedback no stability | 10 rounds with self-feedback, no stopping; no external critic | Tests importance of stability convergence and verification |
| Ablation-of-ours | IterativeRefine without stability check (always runs 10 rounds) | Removes stability check; always runs max rounds |

### Why this setup validates the claim
This experimental design tests the central claim that IterativeRefine autonomously measures an agent's stable skill level and learning capacity. The dataset includes tasks where improvement is possible (complex: e.g., debugging a multi-step error) and where it is not (trivial: e.g., printing a string), ensuring coverage. Single-shot baseline isolates initial skill, while external feedback baseline tests whether self-feedback (with verification) is as effective as a supervised alternative. The no-stability baseline tests whether the stability condition improves measurement accuracy by avoiding wasted rounds or premature stopping. The primary metric directly captures the stable skill level after refinement. We also analyze self-feedback quality by measuring the proportion of feedbacks that contain correct error identification (e.g., if the task is to fix a bug, we check if the feedback points to the actual bug line). The external critic's confidence is calibrated on a dev set of 5 tasks. If our method yields higher stable scores than single-shot on improvable tasks, comparable to external feedback but with full autonomy, and better than no-stability on plateau-prone tasks, then the claim is supported. Conversely, if the stability check harms scores on continuously improving tasks, the claim is weakened.

### Expected outcome and causal chain

**vs. Single-shot** — On a complex tool-use task where the agent initially makes a procedural error (e.g., wrong API call order), single-shot yields a low score (e.g., 0.2) because it has no chance to correct. Our method allows the agent to analyze its output via self-feedback (with verification), notice the error, and refine its approach; after a few rounds, performance stabilizes near 0.8. We expect a noticeable gap (e.g., 0.5+ difference) on such tasks, but parity on trivial tasks where initial performance is near-perfect.

**vs. Fixed-round external feedback** — On a task with ambiguous success criteria (e.g., generating a data analysis report), self-feedback may produce vague corrections, while external feedback gives precise guidance. The external baseline may then achieve slightly higher stable scores (e.g., 0.02–0.05 higher). However, on tasks with clear right/wrong outputs (e.g., code that compiles), self-feedback is as good, so scores are nearly identical. Thus we expect a small gap on ambiguous tasks but no significant difference overall, demonstrating autonomy without major sacrifice.

**vs. Fixed-round self-feedback no stability** — On a task where performance plateaus after 3 rounds (e.g., simple math), the no-stability baseline continues for all 10 rounds, including early unstable attempts in the final average, yielding a lower score (e.g., 0.75 vs ours 0.9 from stable rounds). Our method stops early and uses only stable attempts, giving a higher and more representative score. We expect a 0.1–0.2 improvement on plateau tasks, but on tasks where performance improves steadily, both may converge, so scores are similar.

### What would falsify this idea
If our method's stable score is consistently lower than the single-shot baseline on tasks where improvement is expected (e.g., complex tasks), then the refinement process is harmful, falsifying the claim. Alternatively, if the stability check systematically stops too early on tasks with gradual improvement, causing our score to be lower than the no-stability baseline, the stability condition is counterproductive. If the external verification reduces feedback quality (e.g., too many false positives), that would also weaken the approach.

## References

1. Managing Procedural Memory in LLM Agents: Control, Adaptation, and Evaluation
2. From Exploration to Mastery: Enabling LLMs to Master Tools via Self-Driven Interactions
3. MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering
4. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
5. ML-Bench: Evaluating Large Language Models and Agents for Machine Learning Tasks on Repository-Level Code
6. CLOVA: A Closed-LOop Visual Assistant with Tool Usage and Update
7. Confucius: Iterative Tool Learning from Introspection Feedback by Easy-to-Difficult Curriculum
8. PAL: Program-aided Language Models
