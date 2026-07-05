# SoftMiT: Soft Mixture of Token-level Continuous Offsets for Fine-Grained Robotic Manipulation with VLAs

## Motivation

Existing vision-language-action models like Qwen-RobotManip and Seer use autoregressive token prediction heads that force actions into a discrete space via hard argmax, causing quantization errors that limit precision in tasks like millimeter-accurate grasping. This structural limitation discards the rich distribution information in softmax probabilities and prevents sub-token resolution, a problem acknowledged in the Qwen-RobotManip technical report.

## Key Insight

The softmax probabilities over action tokens encode a fine-grained preference distribution; by treating each token's probability as a mixing weight for a learned continuous offset, we can interpolate within the discrete grid without altering the autoregressive architecture.

## Method

**SoftMiT: Soft Mixture of Token-level Continuous Offsets for Fine-Grained Robotic Manipulation with VLAs**

**Assumption:** The softmax probabilities over action tokens represent a unimodal preference distribution within each bin, so that a weighted average of per-bin offsets yields a meaningful continuous action.

**(A) What it is:** SoftMiT is a lightweight correction head inserted after the LLM's final token projection. It takes logits (shape [D, V]) for D action dimensions and V vocabulary entries, and outputs a continuous action vector of dimension D.

**(B) How it works:**
```python
# Assume LLM output logits per dimension: logits[d] for d in range(D), shape [V]
# Learned offset matrix O of shape [D, V]
# Gating threshold tau = 0.9 (hyperparameter, tuned on validation set)
for d in range(D):
    probs = softmax(logits[d])  # [V]
    conf = max(probs)
    if conf > tau:
        # Unimodal high confidence: use softmax-weighted sum
        action[d] = sum(probs[v] * O[d][v] for v in range(V))
    else:
        # Multimodal or uncertain: use argmax token's offset
        argmax_v = argmax(probs)
        action[d] = O[d][argmax_v]
return action
```
Hyperparameters: V=256, D=7, tau=0.9. Offset matrix O initialized with Xavier uniform and trained end-to-end with the rest of the model.

**(C) Why this design:** We chose a per-dimension learned offset vector over a fixed bin mapping (e.g., uniform bins) because it preserves the LLM's output distribution while enabling continuous outputs; the trade-off is increased parameters (D*V=1792) but negligible relative to model size. We chose softmax weighting over Gumbel-Softmax to avoid sampling noise and maintain differentiability; the trade-off is that softmax may produce an action that is an average of modes when the distribution is multimodal. We chose to apply the offset after the LLM rather than modifying the embedding layer to avoid breaking pretrained language capabilities. The gating mechanism mitigates the multimodal failure case by falling back to argmax when confidence is low.

**(D) Why it measures what we claim:** The computational quantity `sum_v softmax(logits[d,v]) * O[d,v]` measures the continuous action estimate because the softmax probabilities encode the model's confidence over discrete grid cell centers, and the learned offsets O[d,v] capture the residual error between the cell center and the true action for that cell; this equivalence assumes that the optimal action for a given cell is a unimodal local correction within the cell, which fails when the action distribution is multimodal (e.g., two valid grasps), in which case the weighted average blurs modes and produces an unexecutable intermediate action—our gating mechanism handles this by switching to argmax.

**(E) Correlation measure:** To operationalize 'offset captures residual error', we compute for each token v the average residual error on the training set: r_v = mean over samples assigned to token v of (ground_truth_action - bin_center). We then compute Pearson correlation between r_v and O[d,v] across all v and d. The assumption is that this correlation is positive and statistically significant (p < 0.05) for all dimensions, indicating that learned offsets align with true residuals.

**Training details:** Learning rate 1e-4, batch size 64, optimizer AdamW, converge in 50k steps on RLBench (single A100 GPU). We provide open-source implementation with minimal dependencies (PyTorch, transformers, robosuite).

## Contribution

(1) SoftMiT, a method to convert discrete action tokens into continuous actions via softmax-weighted learned offsets, achieving sub-token precision without altering the autoregressive VLA architecture. (2) An empirical finding that discarding the softmax distribution in discrete-action VLMs causes a quantifiable precision loss that can be recovered with a lightweight correction head. (3) A design principle that per-dimension learned offsets effectively interpolate within the discrete action space, requiring only D*V additional parameters.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | RLBench (10 manipulation tasks) | Standard simulation benchmark with continuous actions |
| Primary metric | Success rate | Measures task completion, ultimate goal |
| Baseline 1 | VLA + argmax over bins | Represents naive discretization baseline |
| Baseline 2 | VLA + uniform bin centers | Typical softmax expectation baseline |
| Baseline 3 | Octo (pretrained VLA) | Strong generalist VLA baseline |
| Ablation-of-ours | SoftMiT with fixed offset = 0 | Ablates learned offset; tests importance of per-cell correction |

### Why this setup validates the claim
This experimental design creates a falsifiable test of SoftMiT's central claim: that learned per-bin offsets improve continuous action prediction by correcting residual errors. RLBench provides diverse continuous-action tasks where precision requirements vary, enabling detection of effects tied to correction granularity. The baselines isolate key sub-claims: VLA+argmax tests the role of coarse discretization, VLA+uniform tests the inability to shift from fixed bin centers, and Octo tests whether SoftMiT's lightweight correction can match or surpass a large pretrained model after task-specific fine-tuning. The ablation (fixed offset=0) directly tests whether the learned offsets themselves drive any improvement. Success rate is chosen because it reflects end-to-end task performance, collapsing both action accuracy and downstream effects; if SoftMiT truly reduces action errors, success rate should increase on tasks requiring fine-grained control while remaining similar on coarse tasks.

### Expected outcome and causal chain

**vs. VLA+argmax** — On a precise peg-in-hole task where the optimal continuous action falls near the edge of a discrete bin, VLA+argmax selects the bin center, producing a misaligned grasp that often fails. Our method instead learns an offset vector per bin that shifts the decoded action toward the true optimum within that bin, so we expect a noticeable success rate gap (e.g., ~20% higher) on precision-demanding tasks, but parity on coarse tasks like picking a large object.

**vs. VLA+uniform** — On a task where the true action is consistently offset from the uniform bin center (e.g., a calibrated twist motion), VLA+uniform averages across bins weighted by softmax but has no mechanism to correct the biased center. SoftMiT learns per-bin offsets via its O matrix, so on tasks with systematic misalignment between bin centers and true actions, we expect a clear success rate advantage (e.g., ~15% higher). However, on tasks where bin centers happen to align well, performance matches.

**vs. Octo** — On a novel task with distribution shift (e.g., different object textures), Octo's pretrained action decoder may produce systematic errors because its fixed bin centers are biased toward training data distribution. SoftMiT, being trained end-to-end on the target data, learns offsets that compensate for such bias. Expect SoftMiT to show higher success rate (e.g., 10–25% higher) on out-of-distribution tasks, while Octo may still excel on in-distribution tasks where its generalization is strong.

### What would falsify this idea
If the ablation (fixed offset) achieves similar success rates across all tasks as the full SoftMiT, then learned offsets are unnecessary and the central claim fails. Also, if SoftMiT's improvement is uniform across coarse and fine tasks rather than concentrated on precision-critical subsets, the hypothesized causal chain (offset correcting residual errors) would be unsupported.

## References

1. Qwen-RobotManip Technical Report: Alignment Unlocks Scale for Robotic Manipulation Foundation Models
2. Qwen3-VL Technical Report
3. Scalable Vision-Language-Action Model Pretraining for Robotic Manipulation with Real-Life Human Activity Videos
4. X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model
5. TimeMarker: A Versatile Video-LLM for Long and Short Video Understanding with Superior Temporal Localization Ability
6. MegaSaM: Accurate, Fast, and Robust Structure and Motion from Casual Dynamic Videos
7. PaliGemma 2: A Family of Versatile VLMs for Transfer
8. Predictive Inverse Dynamics Models are Scalable Learners for Robotic Manipulation
