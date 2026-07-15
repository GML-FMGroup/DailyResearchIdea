# IRECUS: A Multi-Turn Simulation Framework for Evaluating Interactive Robustness in Multimodal Models

## Motivation

Current static benchmarks like Blind-Spots-Bench evaluate multimodal models on single-turn questions, failing to capture coordination failures that arise in multi-turn conversational settings. These failures—such as inability to recover from ambiguous user inputs, maintain consistent context, or handle uncooperative behavior—require dynamic interaction. Without a simulation framework that systematically perturbs conversational parameters, these blind spots remain unmeasured, limiting our understanding of model robustness in deployed interactive agents.

## Key Insight

Systematically varying user model parameters across dimensions of cooperativeness and consistency isolates coordination failures that static benchmarks cannot detect, because the interaction imposes a dependency between turn-level decisions and long-term context tracking.

## Method

### (A) What it is
IRECUS (Interactive Robustness Evaluation via Controlled User Simulation) is a multi-turn evaluation framework that pairs a model under test (MUT) with a parameterized user model (UM) in goal-oriented dialogs. The UM's behavior is controlled by three parameters—cooperation probability (sampled from [0.0, 0.5, 1.0]), memory noise (sampled from [0.0, 0.5, 1.0]), and ambiguity level (sampled from [0.0, 0.5, 1.0])—which define configuration profiles. The output is a task success rate averaged over many episodes and configurations.

### (B) How it works
```python
# Pseudocode for IRECUS evaluation
# Config: dict with 'cooperation_prob', 'memory_noise', 'ambiguity_level'
# Hyperparameters: num_episodes=100, max_turns=10
def evaluate(MUT, config, tasks, num_episodes=100, max_turns=10):
    success_count = 0
    total = num_episodes * len(tasks)
    for task in tasks:
        for _ in range(num_episodes):
            state = initialize_state(task)
            um = UserModel(config)
            for turn in range(max_turns):
                mut_action = MUT.generate(state)
                um_action = um.respond(state, mut_action)
                update_state(state, mut_action, um_action)
                if goal_achieved(state, task):
                    success_count += 1
                    break
    return success_count / total
```

### (C) Why this design
We chose a parameterized user model over a fixed script because it allows systematic stress-testing along three orthogonal dimensions of interaction difficulty—user cooperativeness, memory consistency, and ambiguity—which are common in real human interactions but absent in static benchmarks. We opted for goal-oriented tasks (e.g., image description, question answering) over open-domain chitchat because goal success provides an objective, repeatable metric; the trade-off is that goal-oriented tasks may not capture rapport-building or social bonding, but these are less critical for robustness in task-driven assistants. We further selected discrete configuration profiles over continuous ranges to keep the evaluation interpretable and computationally feasible; the cost is potential missed edge cases, which we mitigate by including a medium setting. Finally, we average over multiple episodes and tasks to reduce variance caused by task difficulty or random seed, accepting a higher total compute cost for statistical reliability.

### (D) Why it measures what we claim
The cooperation_prob parameter operationalizes "user cooperativeness" in interaction robustness: a lower probability forces the MUT to ask clarifying questions or rephrase, testing its ability to handle under-specified requests. The assumption is that reduced cooperative responses simulate a user who is not helpful yet still responsive; this assumption fails when the user is entirely silent or non-sequitur, in which case the metric reflects the MUT's ability to detect silence rather than cooperative ambiguity. The memory_noise parameter measures context-tracking robustness: the MUT must recall user preferences over turns, and noise corrupts those preferences along a known dimension (e.g., forgetting a previously stated constraint). The assumption is that memory noise produces plausible but inconsistent statements; this fails when noise makes the user incoherent (e.g., contradicting in the same turn), in which case the metric reflects dialogue breakdown detection. The overall success rate measures "interactive robustness" as the ability to complete a task despite these perturbations; the assumption is that failures are primarily due to coordination breakdowns (e.g., missed context signals or ineffective clarification). This assumption fails when the task is trivially easy, causing ceiling effects; we mitigate this by selecting tasks that have shown difficulty in prior work (e.g., from Blind-Spots-Bench).

### (E) Load-bearing assumption and calibration
The central load-bearing assumption is that the three user model parameters independently and realistically perturb the interaction along the intended dimensions without introducing artifacts that trivialize the robustness challenge. To verify this, we calibrate the user model by collecting a small set of 512 human-human goal-oriented dialogs where annotators rate each utterance for cooperativeness (helpful vs. unhelpful), memory consistency (consistent vs. inconsistent with prior information), and ambiguity (clear vs. vague). We then tune the parameter values so that the simulated user model's output distribution matches the empirical distribution from human dialogs on these three dimensions. Additionally, we validate the operationalization of cooperation_prob by comparing performance under two constrained response modes: (i) when the user model's response type is restricted to clarifying-only versus (ii) non-sequitur. If the performance gap between high and low cooperation_prob persists under clarifying-only but disappears under non-sequitur, it confirms that the parameter indeed tests cooperative ambiguity rather than random noise.

## Contribution

(1) A novel evaluation framework, IRECUS, that generates multi-turn conversational simulations with a parameterized user model to assess interactive robustness of multimodal models, enabling controlled stress-testing along dimensions of cooperativeness, consistency, and ambiguity. (2) A concrete set of user model configurations and goal-oriented task templates that systematically vary these dimensions, providing a diagnostic tool to identify specific failure modes (e.g., poor context tracking vs. inability to elicit information). (3) An empirical finding that performance on static benchmarks (e.g., Blind-Spots-Bench) does not correlate with interactive robustness scores, highlighting the need for dynamic evaluation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Blind-Spots-Bench goal-oriented tasks | Cover diverse difficulty tasks from prior work. |
| Primary metric | Task success rate | Direct measure of goal completion under perturbations. |
| Baseline 1 | Static benchmark (no interaction) | Tests task difficulty without interaction effects. |
| Baseline 2 | Interactive with fixed user (coop=1, mem=0, amb=0) | Tests interaction in ideal cooperative scenario. |
| Ablation | IRECUS without memory noise | Isolates effect of memory noise parameter. |

### Why this setup validates the claim
This combination of dataset, baselines, and metric forms a falsifiable test of the central claim that IRECUS measures interactive robustness. The static baseline isolates task difficulty, showing where interaction is unnecessary. The fixed-user baseline controls for the ideal cooperative case, so any drop in success under IRECUS must stem from parameter variations. The ablation of memory noise directly tests whether that parameter drives observed differences. The metric, task success rate, is objective and directly tied to the goal-oriented nature of the evaluation. By comparing results across these conditions, we can attribute performance changes to specific dimensions of user behavior. If IRECUS reveals deficits that neither baseline captures, it validates that the framework probes interaction-specific robustness. The calibration step (using 512 human dialogs) ensures the parameters correspond to realistic perturbations, and the clarifying-only vs. non-sequitur validation confirms that cooperation_prob measures the intended construct.

### Expected outcome and causal chain

**vs. Static benchmark** — On a task where the user gives an ambiguous description like "the object on the left" without specifying which object, the static benchmark assumes a perfectly clear input and returns a high success rate. Our method instead requires the model to handle a user who may not clarify, generating a low cooperation probability response that forces the model to ask follow-up questions. We expect a noticeable gap on ambiguous tasks: static benchmark success ~80%, IRECUS success ~40%.

**vs. Fixed user model** — On a multi-turn task where the user previously stated a preference (e.g., "I like modern art") but later contradicts it due to memory noise, the fixed user model always remembers correctly, so the model easily succeeds. Our method injects memory noise, causing the user to produce inconsistent statements. The model must detect and resolve the contradiction (e.g., by confirming). We expect a significant drop in success on tasks requiring consistent context tracking: fixed user ~90%, IRECUS ~60%.

### What would falsify this idea
If IRECUS shows no difference from the fixed-user baseline on tasks where memory noise is applied, or if the performance gap on ambiguous tasks is uniform across all ambiguity levels rather than concentrated on the hardest cases, the central claim that IRECUS measures interactive robustness would be invalid. Additionally, if the calibration step reveals that simulated user behaviors do not match human distributions, or if the clarifying-only vs. non-sequitur validation shows no difference in the cooperation_prob effect, the operationalization would be disconfirmed.

## References

1. Blind-Spots-Bench: Evaluating Blind Spots in Multimodal Models
2. SIV-Bench: A Video Benchmark for Social Interaction Understanding and Reasoning
