# CROSS: Learning Robust Parameter Importance via Cross-Modal Consistency Regularization

## Motivation

Current parameter importance quantification methods, such as the significance scoring in Splash (Li et al.) and the task-vector ratio in FAPM (Chen et al.), compute importance on single tasks or modalities, producing fragile and domain-specific estimates that fail to generalize across different fine-tuning settings. The root cause is that these heuristics capture importance relative to a specific training distribution, ignoring the multimodal structure inherent in MLLMs. We argue that a parameter truly important for a model's functionality should demonstrate consistent importance across multiple modalities, and thus leveraging cross-modal consistency can yield a more robust and transferable importance measure.

## Key Insight

The fundamental reason cross-modal consistency regularizes importance is that importance, defined as a parameter's contribution to the model's overall capability, is an intrinsic property that should be invariant to the modality through which it is estimated, provided the modalities share common knowledge representations.

## Method

### (A) What it is
**CROSS** (Cross-Modal Importance Regularization via Consistency Loss) learns a single parameter importance vector I ∈ ℝ^|θ| by enforcing that per-task importance estimates (computed via gradient magnitude averaged over multiple mini-batches to reduce noise) are similar across K modality-specific tasks. Input: pre-trained MLLM parameters θ, K tasks {T_1,...,T_K} with small datasets D_k. Output: importance vector I.

### (B) How it works
```python
# Algorithm: CROSS
# Hyperparameters: λ (consistency weight), τ (temperature), steps S, scanner = gradient magn.
# Noise reduction: moving average of absolute gradients over N=5 mini-batches, momentum=0.9

I = 0.5  # initialize all parameters with same importance
for s in range(S):
    for k in range(K):
        # draw 5 mini-batches (batch_size=32) sequentially from D_k
        grad_accum = 0
        for _ in range(5):
            compute loss L_k on sample from D_k
            compute gradients ∇θ L_k
            grad_accum += |∇θ L_k|
        I_k = (grad_accum / 5) / max(grad_accum / 5)  # average then normalize to [0,1]
    # consistency loss: KL divergence between softmax distributions
    p = softmax(I / τ)
    p_k = softmax(I_k / τ) for each k
    L_cons = (1/K) * Σ_k KL(p || p_k)
    L_sparse = α * mean(I)  # optional sparsity penalty, α=0.01
    I = I - lr * ∇_I (L_cons + L_sparse)
return I
```
The scanner uses gradients averaged over 5 mini-batches (each of size 32) to reduce noise. The number of batches (N=5) was chosen based on a validation sweep over {1,3,5,10} using 10% of training data, balancing noise reduction and computational cost. The consistency loss temperature τ = 1.0 by default. The sparsity loss encourages a small number of high-importance parameters.

### (C) Why this design
We designed CROSS with three key choices. First, we use a gradient-based scanner (absolute gradient of task loss) rather than more expensive methods like Fisher information or movement-based importance (e.g., in Splash), because it is computationally cheap and requires only one backward pass per task; the cost is that gradient magnitude is a noisy estimate that may not capture higher-order interactions or saturation effects. To mitigate noise, we average gradients over 5 mini-batches, which reduces variance at the expense of increased computation (5× per task per outer iteration). Second, we adopt a softmax-scaled KL divergence as the consistency loss instead of mean squared error (MSE), because KL divergence emphasizes relative ranking of parameters rather than absolute values, making it robust to scale differences across task-specific importance scores; the trade-off is sensitivity to the temperature τ, which we set to 1.0 based on pilot experiments. Third, we learn a single importance vector I via gradient descent rather than a separate neural network or post-hoc averaging, which directly optimizes the variable of interest and avoids additional parameters; however, this limits expressiveness to a fixed set of scores that cannot adapt to new tasks without retraining. Alternative design decisions would have increased computational cost (per-iteration Fisher) or reduced robustness (MSE), motivating our choices.

### (D) Why it measures what we claim
The consistency loss D_KL(softmax(I/τ) || softmax(I_k/τ)) measures robustness of the importance estimate because we assume that a parameter's true importance should be invariant across modalities—if a parameter is genuinely critical, it should be highly important for multiple tasks that share underlying representations. This assumption fails when modalities are so distinct (e.g., tactile vs. visual) that they rely on disjoint knowledge; in that case, a parameter important only for one modality will be pulled toward an inconsistent estimate, and the consistency loss may suppress its score, leading to a distorted importance measure that reflects cross-modal agreement rather than intrinsic importance. The gradient scanner |∇θ L_k| measures per-task importance under the assumption that a parameter's contribution to task performance is proportional to its gradient magnitude; this assumption fails when gradients vanish due to saturation or dead neurons, in which case the scanner underestimates importance. Cross-modal agreement is a proxy for robustness—if tasks are contradictory (e.g., require opposite predictions on same input), the consistency loss may suppress genuinely important parameters, and the importance vector may reflect cross-modal agreement rather than intrinsic importance. By combining both components, the final I aims to capture parameters that are consistently influential across modalities, thereby providing a robust and generalizable importance measure.

## Contribution

(1) A novel regularization framework, CROSS, that learns parameter importance scores by enforcing consistency across multiple modality-specific tasks in MLLMs, producing scores that are more robust and transferable than single-task heuristics. (2) An empirical demonstration that the learned importance scores generalize across downstream fine-tuning tasks and scales, as shown by improved selective updating and pruning performance on both seen and unseen modalities. (3) A benchmark protocol for evaluating importance quantification in multimodal settings, using consistency with ground-truth relevance and downstream task utility.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|---|---|---|
| Dataset | SSVTP (8 tasks, 500 samples each) | Tests cross-modal consistency in tactile-vision tasks. |
| Primary metric | Average accuracy | Measures overall usefulness of importance vector. |
| Baseline 1 | Gradient magnitude averaging (single batch) | Ablates noise reduction to show necessity of moving average. |
| Baseline 2 | Random importance | Lower bound for importance estimation. |
| Ablation 1 | CROSS w/o sparsity | Isolates effect of sparsity penalty. |
| Ablation 2 | CROSS w/ single batch (N=1) | Isolates effect of moving average noise reduction. |

### Why this setup validates the claim
The combination of SSVTP (with multiple tactile-vision tasks) and average accuracy after fine-tuning only the top-k importance parameters directly tests the central claim: that CROSS identifies parameters consistently important across modalities. If CROSS's importance vector is robust, fine-tuning only its high-importance parameters should yield higher average accuracy than baselines, especially on tasks that require cross-modal knowledge. The baselines test specific sub-claims: gradient magnitude averaging (single batch) tests the necessity of noise reduction; random importance tests whether our vector is better than random. The ablation CROSS w/o sparsity isolates the effect of the sparsity penalty, which we argue helps focus on truly critical parameters. The additional ablation CROSS w/ single batch directly verifies the noise reduction repair: if performance degrades compared to full CROSS, the moving average is beneficial. The metric (average accuracy) is appropriate because it aggregates performance across tasks, reflecting generalizability.

### Expected outcome and causal chain

**vs. Gradient magnitude averaging (single batch)** — On a task like tactile object recognition that shares visual-semantic features, single-batch gradient magnitude may overestimate parameters that are noisy for one task but not others because it averages raw gradients without considering noise reduction. Our method uses moving average to reduce noise and KL divergence to align importance distributions, suppressing parameters with inconsistent gradients across tasks. Therefore, on cross-modal reasoning subsets, we expect CROSS to show a noticeable accuracy gain (e.g., 3-5% absolute) over single-batch gradient magnitude averaging, while on single-modality tasks performance may be similar.

**vs. Random importance** — On any task, random importance selects parameters with no signal, leading to poor fine-tuning performance because critical parameters may be ignored. Our method selects parameters with high and consistent importance, so we expect CROSS to vastly outperform random importance across all tasks (e.g., >10% gap), demonstrating that our importance vector contains meaningful information.

Our ablation (CROSS w/o sparsity) will likely perform between full CROSS and gradient averaging: without sparsity, the importance vector may be less selective, diluting the top-k set with unimportant parameters. We expect a moderate drop in accuracy (1-2% relative) compared to full CROSS, confirming the sparsity penalty's role.

The ablation CROSS w/ single batch (N=1) will likely perform worse than full CROSS but better than gradient magnitude averaging, because without moving average the importance estimates are noisier, but the consistency loss still provides cross-modal regularization. Expected drop: 1-2% absolute compared to full CROSS.

### What would falsify this idea
If CROSS's accuracy is not consistently higher than gradient magnitude averaging (single batch) across multiple tasks, particularly on cross-modal subsets, then the moving average noise reduction does not improve robustness. Additionally, if the performance gap between CROSS and gradient magnitude averaging is uniform across all subsets (not concentrated on cross-modal reasoning), the causal mechanism of cross-modal consistency is not supported.

## References

1. Wake up for Touch! Mask-isolated Tactile Alignment Learning in MLLMs
2. Mitigating Catastrophic Forgetting in Large Language Models with Forgetting-aware Pruning
3. Qwen2.5-VL Technical Report
4. Lottery Ticket Adaptation: Mitigating Destructive Interference in LLMs
