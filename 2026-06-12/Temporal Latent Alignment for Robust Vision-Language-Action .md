# Temporal Latent Alignment for Robust Vision-Language-Action Training

## Motivation

Current VLA models, as evaluated in LIBERO-PRO, achieve over 90% accuracy on static benchmarks but collapse to 0% under perturbations because they process each visual frame independently without modeling temporal dependencies. This structural failure arises from the action prediction head treating each frame as a separate observation, allowing errors to compound under distribution shift. The root cause is the absence of a coupling between the temporal evolution of visual states and the fixed language goal during training.

## Key Insight

Temporal consistency is implicitly enforced when the action is computed as the difference between an evolving visual latent and a fixed goal latent, because any temporal mismatch directly increases action error, eliminating the need for an auxiliary loss or separate temporal module.

## Method

### (A) What it is
**Temporal Latent Alignment (TLA)** is an action prediction head for VLA models that computes actions as the difference between a latent representation of the current visual state (updated via a recurrent network) and a latent representation of the language goal. Input: a sequence of visual frames encoded per-frame by a visual encoder, and a fixed language goal latent. Output: an action vector at each step.

### (B) How it works
```python
# Pseudocode for TLA head forward pass
def tla_head(frame_features, language_goal):
    # frame_features: list of T frame-level feature vectors (dim D)
    # language_goal: fixed latent vector from language encoder (dim D)
    
    # Initialize hidden state for GRU
    H = 512  # hidden size hyperparameter
    hidden = torch.zeros(H)
    
    # Update hidden state over time using GRU
    gru_cell = GRUCell(input_size=D, hidden_size=H)
    for t in range(T):
        hidden = gru_cell(frame_features[t], hidden)  # visual latent at step t
    
    # Compute action latent as difference
    delta = hidden - language_goal  # shape (H,)
    
    # Project to action space via MLP
    action_mlp = nn.Sequential(
        nn.Linear(H, 256),
        nn.ReLU(),
        nn.Linear(256, action_dim)
    )
    action = action_mlp(delta)
    
    return action
```
During training, the action error (e.g., L2 loss for continuous actions or cross-entropy for discrete) is minimized. No additional temporal loss is used.

### (C) Why this design
We chose a **GRU** over a Transformer for the temporal encoder because manipulation tasks in LIBERO have short episodes (~20 steps), and the recurrent inductive bias suits local dependencies, avoiding the quadratic cost of self-attention; the cost is that long-range dependencies may be less faithfully captured, but we accept this as our target scenarios are short-horizon. We chose a **simple difference** between visual and goal latents rather than a learned comparator (e.g., cross-attention) because the difference directly encodes the distance from the goal, making action error sensitive to any temporal misalignment; the trade-off is that a linear difference may not capture complex relative dynamics, but we compensate with a two-layer MLP on top. We chose to **update the visual latent every step** using the recurrent unit rather than using a fixed exponential moving average (EMA) because the GRU learns to selectively incorporate relevant information, allowing the model to adapt its state update to visual content; the cost is additional parameters and training difficulties, but we mitigate with proper initialization and gradient clipping.

### (D) Why it measures what we claim
The quantity `delta = visual_latent - language_goal_latent` measures **temporal consistency** because it quantifies the discrepancy between the current visual state (evolved over time) and the fixed goal; any temporal inconsistency (e.g., ignoring a moved object) shifts `delta` away from the expected value, directly increasing action error. This assumption holds if the language goal latent is stationary and the visual latent captures all task-relevant dynamics; this assumption fails when the goal is ambiguous or the visual encoder misses crucial changes (e.g., occlusion), in which case `delta` may reflect unrelated differences. The GRU hidden state `hidden` measures **accumulated visual dynamics** because its gating mechanism retains and forgets information over time, driven by the action error gradient; this assumption holds if the GRU's recurrent weights learn to preserve temporally relevant features; it fails when gradients vanish early, causing `hidden` to only reflect the most recent frame. The action error loss measures **task performance** and indirectly enforces temporal consistency because any mismatch in the temporal progression increases error; this holds if the loss is differentiable and the optimizer effectively propagates gradients back to the visual encoder; it fails when action noise dominates, in which case temporal consistency is not reliably enforced.

## Contribution

(1) A novel action prediction head, Temporal Latent Alignment (TLA), that implicitly enforces temporal consistency in VLA models by computing actions as the difference between a temporally evolving visual latent and a fixed goal latent. (2) An empirical demonstration that training with TLA eliminates the catastrophic performance collapse under distribution shifts observed in LIBERO-PRO, establishing a new design principle for robust VLA training. (3) A new evaluation protocol that measures temporal consistency beyond single-frame accuracy, highlighting the model's robustness to perturbations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | LIBERO-PRO | Tests temporal misalignment robustness |
| Primary metric | Task success rate | Directly measures task completion |
| Baseline 1 | Per-frame action prediction | Lacks temporal consistency |
| Baseline 2 | Cross-attention temporal encoder | Learns complex comparator, not direct delta |
| Ablation | TLA w/o GRU (fixed visual latent) | Isolates recurrent update contribution |

### Why this setup validates the claim
This setup directly tests the central claim that the delta between an evolved visual latent (via GRU) and the fixed goal latent enforces temporal consistency, improving action prediction. LIBERO-PRO introduces perturbations (e.g., object moved mid-episode) that break naive temporal assumptions, creating a clear diagnostic for temporal alignment. The per-frame baseline tests whether any temporal modeling is necessary; its expected failure on perturbed tasks confirms that temporal dynamics matter. The cross-attention baseline tests whether a learned comparator suffices; if it fails relative to our explicit difference, it indicates that the simple delta is better at capturing misalignment. The ablation (fixed visual latent) isolates the GRU's role: without it, the model cannot adapt to changes, so performance on dynamic subsets should drop. The metric (success rate) directly captures task outcome, making the causal link between temporal consistency and success unambiguous. This combination of dataset, baselines, and metric forms a falsifiable test: if our method does not outperform on subsets where temporal misalignment occurs, the claim is invalidated.

### Expected outcome and causal chain

**vs. Per-frame action prediction** — On a task where an object is moved to a new location mid-episode (LIBERO-PRO perturbation), the per-frame baseline predicts actions based on the initial frame, resulting in reaching the wrong location and failure. Our method's GRU updates the visual latent each step, so after the move, the delta shifts, and the action redirects toward the new position. We expect a large performance gap on the perturbed subset (per-frame near 0%, ours >50%) but parity on static tasks.

**vs. Cross-attention temporal encoder** — On a task with multiple sequential changes (e.g., first move cup, then push plate), cross-attention may distribute attention to both past and present but lacks an explicit discrepancy signal; it might predict an average or wrong action. Our method's delta directly encodes distance from goal, making it sensitive to each change. We expect our method to outperform on tasks with many temporal transitions, with a noticeable gap (e.g., 20%+ difference) on high-chaos subsets.

**vs. TLA w/o GRU (fixed visual latent)** — On a task where the robot initially sees an object at location A, then the object moves to B, the ablation (fixed visual latent) never updates, so delta reflects the initial goal distance, leading to actions targeting A. Our full method updates the visual latent via GRU, so after the move, delta points to B. We expect the ablation to fail on any dynamic subset (e.g., success <10%), while ours succeeds >50%.

### What would falsify this idea
If our method shows no improvement over the per-frame baseline on perturbed subsets, or if the improvement is uniform across all subsets (static and dynamic), then the central claim that the delta mechanism specifically improves temporal consistency is wrong.

## References

1. LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization
2. InternSpatial: A Comprehensive Dataset for Spatial Reasoning in Vision-Language Models
3. SpatialRGPT: Grounded Spatial Reasoning in Vision Language Model
4. Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling
5. Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models
6. SpatialBot: Precise Spatial Understanding with Vision Language Models
7. ManipLLM: Embodied Multimodal Large Language Model for Object-Centric Robotic Manipulation
8. Ferret: Refer and Ground Anything Anywhere at Any Granularity
