# FARA: Automated Self-Evolution for Healthcare LLMs via Textual Gradient Reward Modeling

## Motivation

Cura 1T relies on human-in-the-loop verification for each evolution round, which limits scalability because human oversight cannot be sustained at large scale. The root cause is that current failure analysis requires human judgment to assess model outputs for correctness and safety, preventing automated closure of the training-evaluation cycle. This structural bottleneck persists in self-evolving medical LLMs, as exemplified by Cura 1T's human gating step.

## Key Insight

Textual gradients decompose model failures into explicit causal statements that a learned reward model can evaluate for correctness and safety independently of the model's generation quality, ensuring reliable automated gating.

## Method

**FARA (Failure-Analysis Reward Automation)**

(A) **What it is:** FARA is a self-evolution loop that replaces human verification with a reward model trained to evaluate textual failure gradients for safety and correctness. Input: base model (e.g., Meditron 7B), initial dataset (1000 medical QA pairs), strong LLM (GPT-4) for synthetic labels. Output: evolved model after multiple rounds.

**Assumption:** The reward model's scores accurately reflect safety and correctness because they are trained on labels from GPT-4; however, if GPT-4's judgments are biased or inconsistent, scores may misalign. We mitigate this by periodically calibrating with human annotations.

(B) **How it works (pseudocode):**

```python
def fara_self_evolution(model, base_data, strong_llm, num_rounds):
    # Initialize reward model: BioBERT encoder (hidden_dim=768) + two linear heads (safety, correctness)
    reward_model = RewardModel(encoder='dmis-lab/biobert-base-cased-v1.1', hidden_dim=768, num_heads=2)
    synthetic_data = []  # store (gradient, safety_score, correctness_score)
    # Seed reward model training: generate 1000 random trajectories
    for _ in range(1000):
        traj = generate_random_trajectory(model)  # sample task from held-out bank (100 tasks), success if output matches gold
        if not traj.success:
            gradient = generate_textual_gradient(model, traj)  # GPT-4 prompt: "Explain why this trajectory failed."
            safety, correctness = strong_llm.evaluate(gradient)  # GPT-4 with rubric: safety [0,1], correctness [0,1]
            synthetic_data.append((gradient, safety, correctness))
    # Train reward model: MSE loss, AdamW, lr=1e-5, batch_size=32, epochs=3, validation split 10%
    train(reward_model, synthetic_data, epochs=3, lr=1e-5, batch_size=32, optimizer='AdamW')
    # Calibrate reward model with 100 human-annotated gradients (Platt scaling per head, target FPR=0.05)
    calibration_set = collect_human_labels(synthetic_data[:100])  # clinical experts annotate safety (0/1) and correctness (0/1)
    reward_model = calibrate_platt(reward_model, calibration_set)

    for round in range(num_rounds):  # num_rounds=5 in full, 2 in pilot
        target_capability = plan_target(model, benchmark_failures)  # cluster with lowest success rate
        trajectories = generate_trajectories(model, tasks_for_capability(target_capability))  # 200 trajectories
        augmented_data = base_data.copy()
        for traj in trajectories:
            if traj.success:
                augmented_data.append((traj.input, traj.output))
            else:
                gradient = generate_textual_gradient(model, traj)
                safety_score, correctness_score = reward_model(gradient)  # scores in [0,1]
                if safety_score > 0.7 and correctness_score > 0.7:
                    corrected = correct_trajectory(traj)  # replace output with gold answer
                    augmented_data.append((traj.input, corrected.output))
                elif safety_score > 0.7 and correctness_score < 0.7:
                    pass  # threshold can be tuned; optionally include corrected trajectory
                # else: discard unsafe gradients
        model = train(model, augmented_data, epochs=1, lr=1e-5, batch_size=32, loss='cross-entropy')
        new_failures = evaluate(model, benchmark_suite)  # 1000 tasks, compute task success rate
        # Update reward model with new failures: add at most 200 new examples
        new_synthetic = []
        for f in new_failures[:200]:
            grad = generate_textual_gradient(model, f)
            s, c = strong_llm.evaluate(grad)
            new_synthetic.append((grad, s, c))
        # Experience replay: mix with last 500 seed examples
        train(reward_model, synthetic_data[-500:] + new_synthetic, epochs=1, lr=1e-5, batch_size=32)
        synthetic_data.extend(new_synthetic)
        # Recalibrate with 100 new human-annotated gradients from new_synthetic
        calib_new = collect_human_labels(new_synthetic[:100])
        reward_model = calibrate_platt(reward_model, calib_new)
    return model
```

(C) **Why this design:** We chose to use a textual gradient as an intermediate representation because it transforms an opaque trajectory failure into a human-interpretable causal chain that can be evaluated for specific dimensions (correctness, safety) independently. This contrasts with using raw trajectory log-probs, which conflate multiple failure types and are poorly calibrated (anti-pattern 3). We chose a separate reward model trained on synthetic labels from a strong LLM over using the same model's self-evaluation because self-evaluation suffers from overconfidence and self-confirmation bias; by distilling external judgment into a reward model, we obtain a stable, reusable evaluator. We decomposed evaluation into two heads (safety and correctness) rather than one composite score because safety failures require immediate rejection while correctness failures may be recoverable with further training, allowing the loop to treat them differently (e.g., safety-failing gradients are discarded, correctness-failing gradients may be used if above threshold). The trade-off is that training the reward model requires periodic queries to a strong LLM, incurring cost, but this cost is amortized over many evolution rounds as the reward model improves.

(D) **Why it measures what we claim:** The reward model's safety score measures the safety of the failure explanation because it is trained on labels derived from GPT-4 that explicitly checks for harmful content, under the assumption that GPT-4's safety judgment aligns with human clinical standards; this assumption fails when the LLM's understanding of medical safety is incomplete or out-of-date, in which case the safety score reflects the LLM's own biases rather than true safety. The correctness score measures the factual accuracy of the gradient's causal claims because the training labels are generated by requiring GPT-4 to verify the gradient's steps against a known correct solution; this assumes GPT-4 can accurately assess reasoning chains, but if the gradient introduces nuanced medical knowledge beyond GPT-4's expertise, the correctness score becomes a measure of surface-level plausibility rather than genuine correctness. By training on diverse gradients from each round, the reward model gradually aligns with the evolving failure patterns, ensuring its measurements remain relevant to the current model's weaknesses. Periodic human calibration (100 examples per round) mitigates drift and misalignment, anchoring scores to expert judgment.

## Contribution

(1) A novel framework FARA that replaces human-in-the-loop verification with an automated reward model evaluating textual gradients for correctness and safety, enabling fully closed-loop self-evolution without human oversight. (2) A training procedure that distills external LLM judgment into a compact reward model with dual heads, allowing low-cost repeated evaluation across evolution rounds. (3) An empirical design principle that textual gradients serve as a reliable intermediate representation for automated failure analysis in healthcare LLMs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Healthcare evaluation suite (consultation, reasoning, diagnosis, EHR) | Covers core clinical tasks |
| Primary metric | Task success rate | Direct measure of agent capability |
| Baseline 1 | GPT-4 | Strong frontier generalist LLM |
| Baseline 2 | Claude | Alternative top generalist |
| Baseline 3 | MedPaLM 2 | Healthcare-specialized baseline |
| Ablation | FARA w/o reward model | Isolates effect of learned reward |

### Why this setup validates the claim
This combination tests the central claim that FARA's learned reward model, trained on textual failure gradients, enables more effective self-evolution than direct strong LLM feedback or static baselines. GPT-4 and Claude probe whether FARA surpasses generalist frontier models that lack iterative failure-driven training. MedPaLM 2 tests against a domain-specialized model with curated data. The ablation (using only strong LLM for evaluation each round) isolates the benefit of distilling judgment into a reward model. Task success rate on a diverse clinical suite captures overall capability; but the key diagnostic is whether FARA gains disproportionately on tasks involving complex safety or correctness trade-offs, where gradient evaluation matters most. If FARA fails to outperform on those subsets, the mechanism is unsupported.

### Expected outcome and causal chain

**vs. GPT-4** — On a case where a model incorrectly suggests a contraindicated drug because it overgeneralizes from similar symptoms, GPT-4 as a baseline avoids evolving because it lacks a mechanism to analyze its own failures. Our method uses the reward model to flag the gradient as unsafe (low safety score) and discards or corrects the trajectory, then retrains on corrected examples. We expect a noticeable gap on cases involving nuanced safety failures (e.g., drug interactions), with FARA improving by ~15-20% on that subset while matching GPT-4 on routine tasks.

**vs. Claude** — On a case where a model misdiagnoses due to missing a key lab value in the reasoning chain, Claude has no self-improvement loop; it relies on its initial training. Our method's reward model marks the gradient as incorrect (low correctness score) and uses the corrected trajectory for further training. We expect FARA to show higher task success on cases requiring multi-step clinical reasoning, particularly where failure analysis distinguishes correctable reasoning errors from fundamental knowledge gaps, yielding a 10-15% absolute improvement on that subset.

**vs. MedPaLM 2** — On a case where a model incorrectly handles a rare disease because the training distribution is sparse, MedPaLM 2 is static and cannot adapt. Our method generates textual gradients from the failure, evaluates them, and includes high-safety, low-correctness examples as negative training signals, effectively learning from errors. We expect FARA to outperform MedPaLM 2 on rare disease diagnoses by 5-10%, reflecting targeted improvement from failure-driven evolution that the specialized baseline cannot achieve.

### What would falsify this idea
If FARA's gain over the ablation (FARA without reward model) is uniformly distributed across all tasks rather than concentrated on failures where safety or correctness gradients are critical, then the reward model is not driving the improvement and the central claim is wrong.

## References

1. Cura 1T: Specialized Model for Agentic Healthcare
2. MedAgentBench: A Realistic Virtual EHR Environment to Benchmark Medical LLM Agents
