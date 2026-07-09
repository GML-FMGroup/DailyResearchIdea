# DEDUCER: Self-Deductive Verification for Teacher-Free Exploration in Reinforcement Learning from Feedback

## Motivation

Existing exploration methods in RL from feedback, such as TREK, rely on a high-quality external teacher model to generate verified candidate trajectories, creating a dependency that limits scalability and applicability when such an oracle is unavailable. The meta-level structural problem is that all prior approaches assume supervision from a stronger model during the exploration phase, failing to establish a teacher-free mechanism. TREK specifically requires an oracle to propose solutions, and even on-policy distillation methods like GKD assume access to teacher feedback, perpetuating the bottleneck.

## Key Insight

Requiring trajectories to satisfy a deductive closure property computed from self-annotated facts enforces logical consistency that is inherently correlated with correctness for objective tasks, enabling reliable self-verification without external supervision.

## Method

DEDUCER is a self-deductive verification mechanism for RL exploration. Input: student policy πθ, a problem set with objective correctness criteria (e.g., math, code). Output: a set of self-generated trajectories that satisfy a deductive closure property and are verified by final answer check, used for on-policy policy updates (e.g., PPO or GRPO).

**Load-bearing assumption**: For objective reasoning tasks, a trajectory that satisfies deductive closure (each step logically entailed by prior steps, problem, and self-annotated anchors) and whose final answer matches the ground truth (verified by a symbolic calculator) is correct overall. This assumption is checked by the final answer verification step.

**Pseudocode**:
```python
def deducer_training_step(πθ, batch_problems):
    # For each problem in batch:
    for problem in batch_problems:
        # Step 1: Generate trajectory and anchors
        trajectory, anchors = sample_from_πθ(problem)  # anchors: self-annotated facts
        # Step 2: Deductive closure check (using lightweight theorem prover, e.g., Metamath)
        valid = True
        for step, next_step in zip(trajectory[:-1], trajectory[1:]):
            if not is_logically_entailed(next_step, [step, problem] + anchors, threshold=0.9):
                valid = False
                break
        if valid and is_consistent(anchors, problem):
            # Step 2b: Final answer verification (using symbolic calculator, e.g., SymPy)
            predicted_answer = extract_final_answer(trajectory)
            if is_correct(predicted_answer, problem_answer):
                accepted_trajectories.append((problem, trajectory, anchors))
    # Step 3: Update policy using RL on accepted trajectories
    loss = compute_rl_loss(πθ, accepted_trajectories)
    update(πθ, loss)
```
Hyperparameters: entailment threshold=0.9 (binary), anchor extraction method: prompt to extract atomic facts, final answer verifier: SymPy symbolic calculator. All other parameters as in PPO (e.g., learning rate 1e-5, clip ε=0.2).

**(C) Why this design**: We chose self-generated anchors over external knowledge bases because it eliminates the need for an oracle, accepting the risk that noisy anchors may reduce acceptance rate; the deductive closure filter mitigates this by rejecting inconsistent trajectories. We chose a strict entailment check (binary) rather than a soft score to ensure a clear correctness signal, at the cost of rejecting many plausible but non-rigorous solutions. We chose to perform the filter on-policy during exploration instead of in a separate phase, simplifying the pipeline but increasing computational cost per step; this avoids the staged procedure of TREK. The design prioritizes autonomy over efficiency, as scalability of teacher-free learning is the primary goal. The final answer verification ensures that only logically consistent and factually correct chains are accepted, directly addressing the risk of self-generated anchors being incorrect.

**(D) Why it measures what we claim**: The computational quantity 'deductive closure satisfaction' measures the motivation-level concept 'trajectory correctness' because we assume that for objective tasks, a logically consistent chain-of-thought necessarily leads to the correct answer; **Assumption A**: logical consistency of chain-of-thought implies correct answer for objective tasks. This assumption fails when the problem requires inductive or analogical reasoning (**Failure mode F**), in which case closure may be satisfied by a wrong answer that is internally consistent. The quantity 'anchor consistency' measures 'factual grounding' because we assume that self-annotated facts that are logically consistent and entailed by the problem are likely correct; this assumption fails when the student generates plausible but incorrect facts, in which case the closure condition must trap inconsistency. The final answer verification adds an extra safeguard against such failures. Together, these checks operationalize a self-verification that avoids external supervision.

## Contribution

(1) DEDUCER, a self-deductive verification framework that eliminates the need for an external teacher in RL exploration for objective tasks. (2) An empirical design principle: deductive closure filtering with self-annotated anchors enables stable self-improvement without supervised candidate generation. (3) Analysis of the trade-offs between self-annotation noise and verification rigor, providing guidance for practical deployment.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | MATH (Level 3-5) | Objective multi-step reasoning |
| Primary metric | Solve rate (pass@1) | Measures trajectory correctness |
| Baseline 1 | TREK | External teacher + staged exploration |
| Baseline 2 | GAD | Black-box distillation with discriminator |
| Baseline 3 | DAPO | State-of-the-art on-policy RL |
| Ablation 1 | DEDUCER w/o closure check | Tests importance of deductive filter |
| Ablation 2 | DEDUCER with self-consistency (5 samples, majority vote) | Compares to self-consistency baseline |

### Why this setup validates the claim

This design forms a falsifiable test of DEDUCER's central claim: self-generated deductive verification can improve exploration without external supervision. Using the MATH dataset (hard multi-step problems) forces reliance on logical consistency rather than pattern matching. The primary metric (solve rate) directly captures trajectory correctness. TREK tests whether DEDUCER can match or exceed a method with external teacher guidance; GAD tests whether the deductive filter provides a stronger signal than learned discriminators; DAPO tests competitive performance against a well-tuned RL system. Ablation 1 isolates the contribution of the closure check, showing that any gains stem from the deductive mechanism, not just on-policy sampling. Ablation 2 contrasts deductive closure with self-consistency, demonstrating that logical entailment is more precise than majority voting. Before the main experiment, we conduct a correlation study on a subset of 200 MATH problems (Level 3) to verify that trajectories satisfying deductive closure more frequently arrive at correct answers than those that do not, providing empirical support for the load-bearing assumption. The final answer verification in DEDUCER ensures that accepted trajectories are both logically consistent and factually correct, addressing the risk of noisy anchors. We will release a reference implementation of the entailment verifier (a fine-tuned T5-NLI model) to lower the entry barrier.

### Expected outcome and causal chain

**vs. TREK** — On a case requiring a multi-step algebraic manipulation, TREK first distills a solution from a teacher, then explores to refine; if the teacher solution contains a subtle error, the student may propagate it and never recover because the staged exploration does not revisit the initial trajectory. Our method instead generates many trajectories, filters each with deductive closure, and only keeps those that are logically consistent and verified by final answer; even if all initial attempts have errors, the exploration continues until a consistent chain appears. We expect DEDUCER to achieve higher solve rates on problems where teacher solutions are noisy, but similar performance on clean, well-covered problems where TREK benefits from high-quality teacher data.

**vs. GAD** — On a case with a non-obvious but valid intermediate step, GAD's discriminator might incorrectly label it as low-quality because it relies on learned preferences that may not generalize to novel reasoning patterns, causing the student to avoid that path. Our method instead uses a strict logical entailment check, which accepts any step that is logically entailed by prior facts, regardless of its surface form. Thus, we expect DEDUCER to explore more diverse correct paths, yielding a higher solve rate on problems requiring creative leaps that are still logically sound, while GAD may stagnate.

**vs. DAPO** — On a case requiring precise anchor extraction (e.g., noting that a certain algebraic identity holds), DAPO's pure RL with rule-based reward might miss subtle factual errors because its reward signal is only based on final answer correctness, allowing internally inconsistent chains to receive high reward if they happen to guess the answer. Our method explicitly checks consistency of every step and verifies the final answer, so it rejects such trajectories. We expect DEDUCER to have a higher proportion of correct solutions that are also logically rigorous, leading to better generalization on out-of-distribution problems, while DAPO may overfit to spurious patterns.

### What would falsify this idea
If DEDUCER shows no significant improvement over Ablation 1 (no closure check) on hard MATH problems, or if its gains are uniform across all problem types rather than concentrated on multi-step tasks where internal consistency is critical, then the central claim that deductive closure drives learning is invalidated. Additionally, if Ablation 2 (self-consistency) matches or exceeds DEDUCER's performance, then deductive closure is not more effective than simple majority voting.

## References

1. TREK: Distill to Explore, Reinforce to Refine
2. Black-Box On-Policy Distillation of Large Language Models
3. DAPO: An Open-Source LLM Reinforcement Learning System at Scale
4. VinePPO: Refining Credit Assignment in RL Training of LLMs
5. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
6. Large Language Models Are Reasoning Teachers
7. GEAR: A GPU-Centric Experience Replay System for Large Reinforcement Learning Models
8. PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU
