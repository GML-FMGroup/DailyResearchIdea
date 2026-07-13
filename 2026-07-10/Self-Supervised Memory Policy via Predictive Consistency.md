# Self-Supervised Memory Policy via Predictive Consistency

## Motivation

Existing memory-augmented agents (e.g., Remember When It Matters) train the memory policy with task-specific supervision (SFT or GRPO on curated datasets), binding policy behavior to the training tasks and preventing zero-shot generalization to new domains. This structural assumption—that memory decisions must be shaped by external task rewards—forces re-collection of supervision for every new environment, limiting scalability.

## Key Insight

The predictive consistency of an agent's own observations—the degree to which storing a past observation reduces future observation prediction error—provides a domain-agnostic, self-supervised training signal that eliminates the need for task-specific rewards.

## Method

Self-Supervised Memory Policy (SSMP) is a framework that trains a memory controller to decide when to store and retrieve observations by maximizing the reduction in a learned predictor's loss on the agent's own trajectories. Input: trajectory of observations and actions. Output: store/retrieve decisions and memory content.

(B) How it works:
```pseudocode
# Offline training on collected trajectories
for each trajectory (obs_1..obs_T, actions a_1..a_T):
    initialize memory bank M = []
    for t = 1 to T:
        # Encode current observation
        h_t = encode(obs_t)
        # Memory controller decides to store
        store_prob = controller(h_t)   # 2-layer MLP, hidden=128, ReLU, output sigmoid
        store = sample(store_prob)
        if store:
            M.append(h_t)
        # Predict next observation with memory
        query = h_t
        if controller_retrieve(query) > 0.5:  # retrieve decision (fixed threshold)
            retrieved = nearest_neighbor(query, M) if M else None  # cosine similarity
        else:
            retrieved = None
        pred_obs_{t+1} = predictor(obs_t, a_t, retrieved)   # 2-layer MLP, hidden=256, ReLU, output dim = obs dim
        # Compute prediction loss
        L_t = MSE(pred_obs_{t+1}, obs_{t+1})
    end
    # After full trajectory, compute counterfactual utility
    # For each store decision, compute Δ = avg_loss_future_without_store - avg_loss_future_with_store
    # where avg_loss_future_without_store is computed by rerunning predictor without that stored observation, holding other stored observations fixed
    # Reward = clip(Δ, -1, 1)
    # Update controller via REINFORCE with reward = Δ (learning rate α=0.01)
    # Update predictor via supervised learning on (obs_t, a_t, retrieved) -> obs_{t+1} (MSE loss, learning rate 0.001)
end

# Optional calibration: fine-tune the predictor on a small calibration set of 512 task-specific examples from the target environment to align prediction loss with task reward.
```
Hyperparameters: predictor is a 2-layer MLP with hidden dim 256; controller is a 2-layer MLP with hidden dim 128; nearest neighbor uses cosine similarity; α=0.01 for REINFORCE learning rate; predictor learning rate 0.001; optimizer Adam.

(C) Why this design: We chose REINFORCE over differentiable approximations (e.g., Gumbel-Softmax) because it avoids biased gradients from relaxation while accepting higher variance; the clipped reward [-1,1] stabilizes training. We used a separate predictor instead of the agent's own model to isolate the effect of memory from action policy, avoiding interference. We opted for offline counterfactual reward computation over online rollouts to eliminate environment interaction during training, accepting that the counterfactual assumes the predictor would have been unchanged (ignoring potential retraining). We chose a simple nearest-neighbor retrieval rather than a learned retriever to keep the control loop interpretable and avoid coupling controller and retriever training. Finally, we thresholded retrieval at 0.5 rather than learning a separate policy, reducing complexity at the cost of not optimizing retrieval timing. The optional calibration step uses 512 examples to adjust the predictor so that prediction loss better correlates with task reward, mitigating misspecification.

(D) Why it measures what we claim: The computational quantity Δ (reduction in future prediction loss from storing an observation) measures **predictive gain** because it quantifies how much a stored observation helps the predictor anticipate future observations; this assumption fails when the predictor is misspecified (e.g., too small to utilize memory), in which case Δ may be zero or negative for useful memories. The controller's store probability is trained to maximize Δ, thereby operationalizing **self-supervised memory policy** because no external reward is involved; this assumption fails if the predictor is trained on data from a different distribution, causing Δ to reflect spurious correlations rather than true predictive gain. To address this, we optionally fine-tune the predictor on a small calibration set (512 examples) from the target task, ensuring Δ better aligns with task utility.

## Contribution

(1) Self-Supervised Memory Policy (SSMP), a framework that trains memory controllers using only the agent's own observation prediction loss as reward, eliminating task-specific supervision. (2) Empirical demonstration that SSMP-trained memory policies transfer zero-shot across domains without retraining, achieving competitive performance with task-specific baselines on long-horizon agent tasks. (3) Analysis of predictive gain as a general-purpose training signal for memory decisions, including failure cases where predictor capacity limits effectiveness.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Long-horizon agent tasks (Terminal-Bench) | Requires storage and retrieval |
| Primary metric | Task completion rate (pass@1) | Reflects agent utility |
| Baseline 1 | No memory (raw predictor) | Tests memory necessity |
| Baseline 2 | Passive store-all, retrieve-all | Tests control necessity |
| Baseline 3 | Surprise-based baseline (store when prediction error > threshold 0.5) | Tests benefit of learned REINFORCE over heuristics |
| Baseline 4 | Oracle store (store based on true task reward, using hindsight and ground-truth utility) | Upper bound to test how close SSMP gets to task-specific policy |
| Ablation 1 | Random store decisions | Tests REINFORCE training |
| Ablation 2 | Different predictor sizes (hidden=64, 128, 256) | Tests sensitivity to capacity |
| Additional metric | Δ statistics (mean, variance, correlation with task completion) | Validates proxy quality |
| Zero-shot transfer | Trained on Terminal-Bench, tested on BabyAI and MiniGrid without retraining | Tests generalization |
| Computational cost | 10k trajectories for training, ~2 GPU hours; counterfactual computation O(T^2) per trajectory, can be reduced with importance sampling or truncated window of 10 steps | Feasibility estimate |

### Why this setup validates the claim

This setup validates the claim that SSMP's learned memory policy improves agent performance by testing it against baselines that isolate key sub-claims. The no-memory baseline tests whether memory is needed at all; passive store-all tests whether selective storage is beneficial; the surprise-based baseline tests whether the learned policy outperforms a simple heuristic; the oracle baseline tests how close SSMP gets to task-specific training. The ablation with random store decisions isolates the effect of REINFORCE training. Using a long-horizon agent benchmark ensures that memory is relevant, and task completion rate as metric directly measures the practical utility of better prediction. Additionally, measuring Δ statistics helps verify that predictive gain correlates with task performance. The zero-shot transfer experiment tests the central claim of domain-agnostic training: if SSMP works across diverse environments without retraining, it confirms that the self-supervised signal is indeed task-independent. Finally, the computational cost estimates allow feasibility assessment. If SSMP outperforms these baselines, it confirms that learning when to store and retrieve based on predictive gain yields a tangible improvement, and the ablation shows that the training process is necessary for that improvement.

### Expected outcome and causal chain

**vs. No memory** — On a task requiring recalling a tool's location from earlier steps, the no-memory baseline cannot retrieve that information and fails. Our method stores the observation when it reduces future prediction loss, so later it retrieves the location and completes the task. We expect a noticeable gap (e.g., 20-30% higher task completion) on memory-dependent tasks, but parity on short tasks.

**vs. Passive store-all** — On a noisy task with irrelevant distractions, passive store-all stores everything, causing retrieval of noise that misleads the agent. Our method stores only observations that reduce the predictor's loss, filtering out noise. Thus, we expect SSMP to outperform on high-noise subsets by 15-25%, while performing similarly on low-noise tasks.

**vs. Surprise-based baseline** — The surprise baseline stores when prediction error is high, which may store novel but irrelevant observations. SSMP's learned policy should store observations that are both novel and predictive, leading to better retrieval utility. We expect SSMP to outperform by 10-20% on tasks where surprise and predictive gain diverge.

**vs. Oracle store** — SSMP should approach the oracle's performance, with a gap of less than 10% on average, demonstrating that predictive gain is a good proxy for task utility. If the gap is large (>15%), it indicates the need for calibration.

**Zero-shot transfer** — SSMP trained on Terminal-Bench should achieve higher task completion on BabyAI and MiniGrid than no-memory and passive store-all baselines, showing generalization. Expected improvement of 5-15% over baselines.

### What would falsify this idea

If SSMP performs no better than the random store ablation, or its advantage is uniform across all task types rather than concentrated on tasks with high memory demand or noise, then the central claim that learned storage timing based on predictive gain is beneficial would be falsified. Additionally, if the zero-shot transfer shows no improvement over no-memory, the claim of domain-agnostic generalization is invalidated.

## References

1. Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents
2. Augmenting Language Models with Long-Term Memory
3. Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks
4. How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models
5. LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning
6. Training Language Models with Memory Augmentation
7. YaRN: Efficient Context Window Extension of Large Language Models
8. MemGPT: Towards LLMs as Operating Systems
