# Geometric Abstention: Adaptive Stopping via Hidden-State Contraction in Tool-Using Agents

## Motivation

Existing abstention methods rely on static prompts or log-probability thresholds (e.g., AbstentionBench, Do LLMs Know When to NOT Answer) and treat abstention as a one-shot decision rather than a sequential process informed by the model's own knowledge accumulation. The structural limitation is that these methods do not adapt to task difficulty or internal confidence trajectory, leading to either premature or delayed abstention. Agentic Abstention shows that agents often abstain too late or never, highlighting the need for a principled, adaptive criterion that leverages internal state dynamics.

## Key Insight

The cosine distance between successive hidden-layer representations of a tool-using LLM monotonically decreases as it gathers relevant information, approaching a fixed point when it has sufficient knowledge to answer, providing a domain-independent signal for when to abstain without external verification.

## Method

### (A) What it is
**GAbstain** (Geometric Abstention) is a meta-learned abstention policy that, given the sequence of hidden-state representations from an LLM during tool-use, computes the cosine distance between consecutive representations and applies a task-adaptive threshold to decide whether to continue tool use or abstain (or produce the final answer).

**Load-bearing assumption (explicit):** We assume that the cosine distance between consecutive hidden states monotonically decreases as the model gathers relevant information, and that a low distance with low variance indicates sufficient knowledge for a correct answer. This assumption is validated via a calibration phase on a held-out set.

### (B) How it works
```python
# Input: list of hidden states H = [h_1, h_2, ..., h_T] from last layer (extract after each tool call, shape [d_model])
# Output: stop_time t* where abstention or final answer is triggered

# Hyperparameters:
#   lambda = 0.1 (interaction cost per step)
#   window_size = 3 for moving average
#   epsilon = 0.01 for variance threshold
#   d_model = 4096 (for Llama-2-7B)

# Calibration phase (on held-out set of 512 examples):
# Compute Pearson correlation between average d_t and answer correctness. If r < 0.3, learn a bias b via logistic regression on correctness ~ (mu_t, var_t) and adjust tau_t += b. Otherwise, proceed as is.

# Meta-training phase:
for each task in meta-training set (400 tasks):
    Run agent on task; record H and per-step reward R(t) = correct(t) - lambda * (t-1)
    Compute distances d_t = 1 - cos_sim(h_t, h_{t-1}) for t=2..T
    Compute moving average mu_t = mean(d_{max(1, t-window_size):t})
    Compute variance var_t = var(d_{max(1, t-window_size):t})
    # Meta-learn threshold predictor f_theta: 2-layer MLP with hidden size 64, GeLU activation, input=(mu_t, var_t), output=tau_t in [0,1]
    # Policy: stop at first t where d_t < tau_t and var_t < epsilon
    # Loss: negative expected reward under this stopping policy
    # Update theta via REINFORCE-style gradient with baseline (average reward per task)

# Meta-testing phase (new task):
H = []
for t=1..max_steps (max_steps=10):
    Execute tool call, get h_t from last layer (shape [d_model])
    H.append(h_t)
    if t > 1:
        d_t = 1 - cos_sim(h_t, h_{t-1})
        mu_t = mean(d_{max(1, t-window_size):t})
        var_t = var(d_{max(1, t-window_size):t})
        tau_t = f_theta(mu_t, var_t)
        # Apply calibration bias if learned: tau_t = clamp(tau_t + b, 0, 1)
        if d_t < tau_t and var_t < epsilon:
            stop_time = t
            break
# After stopping, generate final answer from last hidden state (or from LLM with full context)
```

**Compute requirements:** Meta-training on 400 tasks, 100 epochs, uses 1 A100 GPU (~2 hours).

### (C) Why this design
We chose cosine distance over Euclidean or dot product because it normalizes for vector magnitude, which can vary across tasks and models, ensuring the threshold is stable across different hidden-state norms. We meta-learn the threshold instead of using a fixed value because the contraction rate depends on task difficulty and tool availability; a fixed threshold would either stop too early (high abstention but low accuracy) or too late (high accuracy but many interactions). We incorporate variance of distances over a sliding window because a single small distance could be due to noise (e.g., a trivial tool call that doesn't change representation), so requiring both low mean and low variance reduces false positives. The trade-off for using a moving average is that it introduces a delay in detecting contraction, potentially missing the optimal stopping point; we accept this cost for robustness. We chose a small MLP as the threshold predictor (2-layer, hidden size 64) over a recurrent model to keep overhead minimal and avoid overfitting on meta-training tasks. Compared to prior work that uses simple confidence thresholds (e.g., Do LLMs Know When to NOT Answer), our method adapts the threshold per task and uses a more reliable signal (hidden-state dynamics) that is less prone to miscalibration.

### (D) Why it measures what we claim
The cosine distance between consecutive hidden states measures the **information gain** from the latest tool-use step because the hidden state is a compressed representation of all prior context and tool outputs, and a small change indicates that the new information is not altering the representation significantly. This assumption fails when the hidden state is artificially collapsed due to model regularization (e.g., layer normalization, residual connections) or when multiple tool calls provide contradictory information that cancels out in representation space, in which case the distance may be small even though the model is uncertain; we address this by also requiring low variance over a window (var_t < epsilon), so that low distance only triggers abstention when the representation is stable. Additionally, we explicitly validate this relationship via a calibration phase: on a held-out set of 512 examples, we compute the Pearson correlation between the average cosine distance (over the window) and the KL divergence of the output distribution from a uniform distribution (as a proxy for information gain). If the correlation is below 0.3, we learn a logistic regression bias to adjust the threshold, ensuring that the stopping criterion aligns with actual sufficiency. The threshold tau_t, predicted by f_theta from (mu_t, var_t), operationalizes the **task-specific tolerance** for stopping; meta-learning tau_t across tasks ensures that for each task, the stopping point trades off accuracy vs. interaction cost appropriately. Thus, the combination of cosine distance (information gain) and adaptive threshold (task-adaptive tolerance) directly implements the motivation-level concept of an internal confidence trajectory that determines the optimal abstention point.

## Contribution

(1) A novel abstention mechanism that uses geometric contraction of LLM hidden states during tool-use to dynamically determine the optimal moment to stop. (2) A meta-learning framework that adapts the abstention threshold across tasks, balancing accuracy and interaction cost without task-specific tuning. (3) An empirical finding (to be validated) that hidden-state contraction provides a more reliable signal for tool-use completion than log-probability or prompt-based methods.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | ToolEmu simulated tool-use scenarios (250 tasks) | Tests abstention in varied tool tasks |
| LLMs | Llama-2-7B and Mistral-7B | Tests generalization across architectures |
| Primary metric | Abstention F1 (balance precision/recall) | Captures correct abstention vs. unnecessary |
| Baseline 1 | No abstention agent | Always continues tool use |
| Baseline 2 | System prompt abstention | LLM with "say I don't know" prompt |
| Baseline 3 | Quit instruction agent | Explicit quit option in prompts |
| Baseline 4 | Log-prob trajectory (stop when max token log-prob > 0.9) | Isolates novelty of hidden-state signal |
| Ablation | GAbstain w/o meta-learning (fixed tau=0.5) | Tests necessity of adaptive threshold |
| Mechanistic analysis | Pearson correlation between cosine distance and KL divergence of output distribution (on held-out 100 tasks) | Validates information gain assumption |

### Why this setup validates the claim
This setup tests the central claim that meta-learning an adaptive threshold on hidden-state cosine distances enables optimal abstention in tool use. The dataset includes tasks where further tool calls provide diminishing returns or cause harm, allowing detection of both false positives (unnecessary abstention) and false negatives (missed abstention). The baselines isolate key sub-claims: No abstention tests the need for any abstention mechanism; System prompt tests whether static confidence suffices; Quit instruction tests fixed adaptation; Log-prob trajectory tests whether a simpler confidence signal can replace hidden-state dynamics. The ablation tests whether meta-learning adds value over a fixed threshold. The F1 metric directly measures the trade-off between precision and recall that the method aims to optimize, making it sensitive to the predicted effect. Additionally, we perform a mechanistic analysis on a held-out set of 100 tasks: for each task, we compute the Pearson correlation between the average cosine distance (over the window) and the KL divergence of the output distribution from the previous step's distribution, to quantitatively confirm that cosine distance tracks information gain.

### Expected outcome and causal chain

**vs. No abstention agent** — On a case where further tool calls are harmful (e.g., executing a dangerous command), the baseline will continue and cause harm because it never abstains. Our method detects low information gain (small cosine distance with low variance) and stops, preventing harm. We expect a large gap on unsafe/harmful tasks but parity on safe tasks.

**vs. System prompt abstention** — On a case where the LLM is overconfident (e.g., a plausible-looking but unachievable task), the baseline will answer incorrectly because it relies on miscalibrated confidence. Our method uses hidden-state contraction which is less prone to miscalibration, so it will abstain. We expect our method to have higher recall on unanswerable tasks while maintaining similar precision.

**vs. Quit instruction agent** — On a case where the LLM must decide when to stop amid ambiguous signals (e.g., multi-step task with confusing outputs), the baseline may quit too early due to conservative prompt or too late due to lack of per-step adaptation. Our method dynamically adapts threshold per task based on recent distance trends. We expect our method to achieve a better trade-off (higher F1) across tasks with varying optimal stop times.

**vs. Log-prob trajectory** — On tasks where the hidden-state contraction signal is more informative than token-level confidence (e.g., tasks requiring synthesized reasoning rather than direct lookup), the log-prob baseline may plateau early or misestimate certainty. Our method captures representation-level stability, leading to more accurate abstention timing. We expect our method to achieve higher F1 on tasks with complex reasoning chains.

**Ablation: GAbstain w/o meta-learning** — On tasks with varying difficulty, fixed threshold will either stop too early (high false abstention on easy tasks) or too late (low abstention on hard tasks). Our meta-learned threshold adapts, so we expect higher overall F1, especially on tasks with unusual representation dynamics (e.g., sudden shifts in information gain).

### What would falsify this idea
If our method's improvement over the ablation is uniform across all difficulty levels rather than concentrated on tasks where representation dynamics vary (e.g., similar gains on both easy and hard tasks), it would suggest the meta-learning is not exploiting task-specific patterns and the central claim is wrong. Additionally, if the mechanistic analysis shows a weak correlation (<0.3) between cosine distance and KL divergence across most tasks, the foundational assumption would be invalidated.

## References

1. Agentic Abstention: Do Agents Know When to Stop Instead of Act?
2. AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions
3. Check Yourself Before You Wreck Yourself: Selectively Quitting Improves LLM Agent Safety
4. Identifying the Risks of LM Agents with an LM-Emulated Sandbox
5. Do LLMs Know When to NOT Answer? Investigating Abstention Abilities of Large Language Models
6. The Art of Saying No: Contextual Noncompliance in Language Models
7. Know Your Limits: A Survey of Abstention in Large Language Models
8. When Not to Answer: Evaluating Prompts on GPT Models for Effective Abstention in Unanswerable Math Word Problems
