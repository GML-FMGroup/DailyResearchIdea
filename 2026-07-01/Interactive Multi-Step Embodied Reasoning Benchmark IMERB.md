# Interactive Multi-Step Embodied Reasoning Benchmark (IMERB)

## Motivation

Existing VLA evaluations, such as the Act2Answer protocol from 'Does VLA Even Know the Basics?' and the Cosmos-Reason1 benchmarks, test single-step action-grounded knowledge or static reasoning chains without environmental feedback. This gap prevents assessment of VLAs' ability to adapt their plans based on action outcomes—a critical capability for real-world deployment. The root cause is that prior benchmarks do not require the model to update its reasoning after observing the effects of its own actions, thereby failing to isolate failures in credit assignment and adaptive behavior.

## Key Insight

Multi-step interaction with explicit environmental feedback forces the VLA to maintain and update a coherent internal state based on action consequences, revealing failures in adaptive reasoning that static or single-step evaluations cannot expose.

## Method

### (A) What it is

IMERB is a benchmark that presents a VLA (specifically, a 7B-parameter model based on LLaMA-2 with a learned action head outputting a 7-DOF end-effector pose [delta x,y,z, delta roll,pitch,yaw, gripper width]) with a sequence of dependency-linked subgoals in a simulated tabletop environment (PyBullet, 8 objects per scene sampled from a set of 20 shapes). Each action produces explicit textual feedback that the VLA must incorporate to decide the next action.

**Load-bearing assumption:** The VLA's internal state updated via prompt history is equivalent to its adaptive reasoning capacity for multi-step tasks.

**Calibration/verification:** We include control tasks where (1) textual feedback contradicts visual feedback (e.g., visual shows object moved but feedback says 'failure: collision'), and (2) an impossible action is required (e.g., subgoal to pick an invisible object). If the model still outputs plausible actions, then it is not truly adapting.

### (B) How it works

```python
# IMERB evaluation for one episode
def evaluate_episode(model, episode):
    state = episode.initial_state
    subgoals = episode.subgoals  # list of strings, e.g., "Pick up the red block"
    actions = []
    feedback = ""
    action_distributions = []
    for i, subgoal in enumerate(subgoals):
        obs = render(state)
        if i > 0:
            prompt = f"History: {''.join(actions[-3:]) if len(actions)>=3 else ''.join(actions)}. Observation: {obs}. Subgoal: {subgoal}. Previous action result: {feedback}."
        else:
            prompt = f"Observation: {obs}. Subgoal: {subgoal}."
        # Record action distribution before feedback incorporation (using internal logits)
        action_logits = model.get_action_logits(prompt)  # shape [7, num_bins_per_dim] if discrete, else [7] for continuous
        action_distributions.append(action_logits)
        action = model.generate_action(prompt)  # output: 7-DOF continuous action
        next_state, result = simulator.execute(state, action)
        success[i] = (result == "success")
        state = next_state
        feedback = result  # text string from set {"success", "failure: object slipped", "failure: wrong object", "failure: collision"}
        actions.append(action)
    # Compute adaptive reasoning metric: KL divergence between action distributions at step i and step i+1 (after feedback)
    kl_divs = [KL(action_distributions[i], action_distributions[i+1]) for i in range(len(subgoals)-1)]
    return all(success), np.mean(kl_divs)
```
**Hyperparameters:** episodes=50, steps per episode=4 (±1), feedback types={"success", "failure: object slipped", "failure: wrong object", "failure: collision"}. Simulator: PyBullet with 8 objects per scene, objects randomly placed, 50 episodes split into 40 seen dependency structures and 10 unseen dependency structures (novel graphs not in training). Action distribution bins: each continuous dimension discretized into 10 bins (for KL computation).

### (C) Why this design

(1) We use a fixed sequence of subgoals rather than an open-ended task to isolate the multi-step credit assignment component from exploration; this sacrifices generality but focuses evaluation on reasoning over dependencies. (2) We provide explicit textual feedback after each action (e.g., "the block is now at position X") rather than only visual changes, because VLAs may fail to detect subtle visual changes; the cost is that we test VLA's ability to integrate language-encoded feedback rather than purely visual reasoning. (3) We require the VLA to output an action at each step without an explicit "stop" token, ensuring that the model must produce a meaningful action for every subgoal; this prevents degenerate policies that only terminate early. (4) We compute subgoal success per step rather than only final success, enabling fine-grained diagnosis of where reasoning fails; this increases evaluation time but provides actionable insights. Each choice prioritizes clarity of credit assignment over ecological validity.

### (D) Why it measures what we claim

The per-step subgoal success rate measures the VLA's ability to adapt its reasoning based on environmental feedback because each action's outcome (success/failure) is known only via the next observation/feedback; the model must incorporate this information into its next reasoning step. The assumption is that the VLA's internal state updated via the prompt history is equivalent to its adaptive reasoning capacity; this assumption fails when the VLA relies on primitive reactive policies that ignore context (e.g., always repeating the same action), in which case subgoal success reflects habitual behavior rather than adaptive reasoning. The requirement to produce different actions for different subgoals in the same episode measures the model's ability to maintain a coherent plan; this assumption fails if the VLA memorizes fixed sequences across episodes, in which case success reflects rote imitation rather than online reasoning. The explicit feedback text measures the model's capacity to integrate linguistic outcome signals; this assumption fails if the model ignores the feedback string, in which case the metric captures only visual input processing. Additionally, we compute the KL divergence between action distributions before and after feedback as a direct measure of adaptive reasoning: a high KL indicates the model changed its policy in response to feedback, whereas low KL suggests feedback was ignored. We also measure internal state probe accuracy (linear probe on hidden states of the VLA to predict the correct subgoal step) to validate that the model's internal state is updated accordingly.

## Contribution

(1) A novel interactive benchmark comprising 50 multi-step embodied reasoning episodes with dependency-linked subgoals and explicit environmental feedback, each with 3-5 steps.
(2) A quantitative metric suite that decomposes success into per-step credit and recovery from negative feedback, enabling diagnostics of reasoning failures.
(3) An analysis protocol to attribute VLA failures to perception, planning, or adaptation stages based on error patterns across feedback types.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | IMERB (50 episodes: 40 seen dependency structures, 10 unseen; 8 objects, 4 steps per episode) | Enables test of generalization to novel dependency graphs.
| Primary metric | Subgoal success rate + Action distribution KL divergence (before vs. after feedback) | Per-step adaptive reasoning + direct policy change measure.
| Baseline 1 | No-feedback VLA (feedback string empty) | Ignores textual feedback, tests its necessity.
| Baseline 2 | Shuffled-subgoal VLA (random subgoal order) | Breaks dependencies; tests if order matters.
| Ablation 1 | Visual-only feedback (no text, only image change) | Replaces textual feedback with visual change.
| Ablation 2 | Contradictory feedback (text says failure but visual shows success) | Tests if model ignores contradictory feedback.
| Ablation 3 | Impossible action (subgoal requires picking invisible object) | Tests if model detects impossible actions.
| Additional metric | Internal state probe accuracy (linear probe on hidden states predicting correct subgoal step) | Directly validates internal state update assumption.

### Why this setup validates the claim

The combination of IMERB's dependency-linked subgoals, per-step textual feedback, and fine-grained subgoal success metric creates a targeted test of a VLA's ability to integrate outcome signals into adaptive reasoning. The no-feedback baseline isolates the contribution of textual feedback; if feedback is critical, this baseline should underperform. The shuffled-subgoal baseline tests whether the model truly reasons about dependencies versus memorizing sequences; if dependency matters, this baseline should fail on later subgoals. The visual-only ablation tests whether textual feedback provides unique information beyond visual changes. The contradictory feedback and impossible action ablations serve as sanity checks: if the model truly adapts, contradictory feedback should degrade performance, and impossible actions should cause the model to output a special 'no-op' action or low confidence. The KL divergence metric directly measures whether the model changes its action distribution after receiving feedback, providing a computational proxy for adaptive reasoning. Finally, the internal state probe accuracy tests the load-bearing assumption that the prompt history updates the model's internal state; high probe accuracy on seen dependencies and generalization to unseen dependencies would support the assumption. Together, these comparisons form a falsifiable test: if the main model outperforms all baselines and passes the sanity checks, it validates that IMERB captures adaptive reasoning; if not, the claim is unsupported.

### Expected outcome and causal chain

**vs. No-feedback VLA** — On a case where the VLA drops an object and receives failure feedback, the no-feedback baseline repeats the same action because it cannot adapt from textual outcomes. Our method uses the feedback to change strategy (e.g., adjust grip), so we expect a large gap (e.g., 20% higher success) on episodes with repeated failures but no gap on easy single-attempt tasks. Additionally, the KL divergence after a failure should be higher for our method than for no-feedback baseline.

**vs. Shuffled-subgoal VLA** — On a case where subgoal 2 depends on subgoal 1 (e.g., pick then place), the shuffled baseline often tries subgoal 2 first, causing early failure because the prerequisite object isn't held. Our method respects order, so we expect a large gap (e.g., 30% higher overall success) on dependency-heavy episodes but parity on independent subgoals.

**vs. Visual-only feedback** — On a case where feedback is purely visual (e.g., object changed color), our method should show only a minor gain (∼5%) because visual information is also available. However, on cases where feedback is non-visual (e.g., object weight inferred from feedback text), our method should significantly outperform (∼15%).

**Contradictory feedback control** — Our method's success rate should drop significantly when feedback contradicts visual input (e.g., by 10-20%) because it relies on feedback for adaptive reasoning. If not, the model is ignoring the feedback.

**Impossible action control** — Our method should output a low-confidence action or special 'no-op' token for impossible subgoals, resulting in a high KL divergence from the prior distribution. Baseline that ignores feedback would attempt an action.

**Internal state probe** — Probe accuracy should be >80% on seen dependencies and >70% on unseen dependencies, indicating that the model's hidden states encode the current subgoal step.

### What would falsify this idea

If our method's gain over the no-feedback baseline is uniform across all subgoal types (i.e., not concentrated on steps after a failure), or if the shuffled-subgoal baseline performs similarly to our method on dependent subgoals, then the claim that IMERB measures adaptive reasoning is falsified. Additionally, if the contradictory feedback control shows no performance drop, or if the internal state probe accuracy is low on seen dependencies, then the load-bearing assumption (prompt history equals adaptive reasoning) is unsupported.

## References

1. Does VLA Even Know the Basics? Measuring Commonsense and World Knowledge Retention in Vision-Language-Action Models
2. Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning
3. The Llama 3 Herd of Models
4. HybridFlow: A Flexible and Efficient RLHF Framework
5. Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling
6. Robotic Control via Embodied Chain-of-Thought Reasoning
7. NVLM: Open Frontier-Class Multimodal LLMs
8. BridgeData V2: A Dataset for Robot Learning at Scale
