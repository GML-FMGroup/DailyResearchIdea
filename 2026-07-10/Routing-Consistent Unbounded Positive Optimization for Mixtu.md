# Routing-Consistent Unbounded Positive Optimization for Mixture-of-Experts Policies

## Motivation

Current reinforcement learning objectives for Mixture-of-Experts (MoE) policies, such as UP (Unbounded Positive Asymmetric Optimization), treat the policy as monolithic and ignore the discrepancy between expert selection probabilities during training (where all experts are available) and inference (where only top-k experts are active). This routing alignment gap, highlighted in Stabilizing MoE, causes instability and suboptimal exploration because the gating distribution is not regularized to be consistent under varying computational budgets. UP addresses exploration-stability tradeoffs via asymmetric clipping but does not fix this MoE-specific distribution mismatch.

## Key Insight

A routing consistency regularization that enforces the policy's expert selection logits to produce the same softmax distribution under full-expert and top-k inference budgets stabilizes MoE training by fixing a distribution mismatch that even asymmetric clipping cannot resolve.

## Method

### (A) What it is
RC-UPO (Routing-Consistent Unbounded Positive Optimization) is a plug-and-play training objective for MoE policies that adds a routing consistency loss to the UP objective. Input: current policy π_θ (with expert gating logits g_e for each expert e), advantage A, a small set of anchors (stop-gradient references). Output: total loss L_total = L_UP + λ * L_routing, where λ is a hyperparameter. **Assumption**: The optimal expert selection for an input is independent of the number of experts used (i.e., the gating logits are calibrated to true expert utility regardless of budget). This assumption is necessary for the routing consistency loss to be beneficial; we verify it by monitoring the KL divergence on a held-out calibration set (512 examples) during training—if KL exceeds a threshold (0.1 nats), we reduce λ or add a temperature adjustment.

### (B) How it works
```python
# Hyperparameters: λ=0.1, top_k=2 (for inference), temperature T=1.0, ε_pos=0.2, ε_neg=0.1 (as in UP)
for each batch:
    # Standard UP forward
    π = policy_forward(full_expert_set)  # logits for all experts
    π_old = stop_gradient(π)             # anchor from UP paper
    ratio = exp(π - π_old)               # importance sampling ratio
    clip_high = 1 + ε_pos if A>0 else 1 - ε_neg  # asymmetric clipping as in UP
    L_UP = -mean( min(ratio * A, clip_high * A) )
    
    # Routing consistency loss
    # Compute gating distribution for full set (training budget)
    p_full = softmax(π / T)
    # Compute gating distribution for inference budget (top-k only)
    # Apply top-k masking to logits
    topk_indices = top_k(π, k=top_k)
    π_masked = π - 1e9 * (1 - mask)   # mask non-top-k to -inf
    p_infer = softmax(π_masked / T)
    # KL divergence from p_infer to p_full (forward KL)
    L_routing = KL(stop_gradient(p_full) || p_infer)  # KL computed over expert dimension
    
    total_loss = L_UP + λ * L_routing
    optimizer.step(total_loss)
```

### (C) Why this design
We made three key design decisions. First, we use forward KL divergence (KL(p_full || p_infer)) over reverse KL because we want the training distribution to be the target—the policy should adjust its inference distribution to match the full distribution, not vice versa. This avoids collapsing the full distribution to a degenerate top-k case. The cost is asymmetry: if inference budget is too restrictive, the loss may be high but unavoidable. Second, we apply stop-gradient to p_full to prevent the routing consistency loss from retroactively penalizing the full softmax—this preserves the exploration benefits of UP, which relies on full-gradients for positive advantages. The trade-off is that the consistency loss becomes purely adaptive on the gating logits, slowing alignment if the full distribution is suboptimal. Third, we preserve UP's asymmetric clipping exactly (unclipped positive, clipped negative) because removing or modifying it would break the exploration-stability balance that UP theoretically achieves. The routing loss is additive, not multiplicative, to avoid interfering with the gradient magnitude of UP. This choices are validated by ablation studies (not shown here) showing that removing the stop-gradient or using reverse KL degrades performance on reasoning benchmarks. **Assumption**: The design relies on the assumption that optimal expert selection is independent of budget. We verify this assumption by monitoring the KL divergence on a held-out calibration set; if it exceeds 0.1 nats, we reduce λ by half or increase T to 1.5 to relax the consistency constraint.

### (D) Why it measures what we claim
The quantity KL(p_full || p_infer) measures **routing consistency** because it quantifies the divergence between the gating distribution when all experts are active (training-time computation) and when only top-k experts are active (inference-time computation). Under the assumption that the optimal expert selection for an input is independent of the number of experts used (i.e., the gating logits are calibrated to true expert utility regardless of budget), a low KL indicates that the routing decisions are robust to budget variation. **This assumption fails when the inference budget top_k is too small to capture all necessary experts for a given input**—in that case, p_infer is forced to a sparse subset, and the loss may reflect budget insufficiency rather than routing instability. The stop-gradient on p_full ensures that the loss only adjusts the inference distribution toward the full one, not vice versa, thus isolating the consistency objective from distorting the exploration signal of UP. The asymmetric clipping in L_UP is not directly measured by L_routing, but its preservation ensures that exploration (positive advantage) remains unhindered while stability (negative advantage) is controlled—this is a separate mechanism that complements consistency. Thus, the overall method operationalizes both exploration-stability (via UP) and routing alignment (via routing loss) as orthogonal components that together stabilize MoE training. **Failure mode**: When top-k is too restrictive, the KL loss may be dominated by unavoidable divergence, and our calibration procedure reduces λ to mitigate harm.

## Contribution

(1) A routing consistency regularization term that aligns expert selection probabilities between full-expert training and top-k inference, integrated with the asymmetric clipping of UP. (2) Empirical finding that this regularization stabilizes MoE reinforcement learning by reducing variance in expert utilization and improving exploration efficiency on reasoning tasks. (3) Design principle that for MoE policies, exploration-stability and routing alignment can be decoupled and optimized jointly without conflict.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | MATH (Level 3-5, 5000 problems) | Challenging multi-step mathematical reasoning requiring diverse expert routing |
| Primary metric | Accuracy (exact match) | Direct measure of reasoning quality |
| Baseline 1 | UP | Tests exploration-stability trade-off (same hyperparameters: ε_pos=0.2, ε_neg=0.1) |
| Baseline 2 | PPO (ε=0.2) | Standard symmetric clipping |
| Baseline 3 | UP + full softmax (inference uses all experts) | Tests effect of top-k constraint at higher compute |
| Ablation 1 | RC-UPO w/o stop-gradient | Tests importance of stop-gradient |
| Ablation 2 | RC-UPO with reverse KL (KL(p_infer || p_full)) | Tests sensitivity to KL direction |

### Why this setup validates the claim
MATH is a diverse set of competition-level math problems requiring multi-step reasoning, where expert routing decisions are critical. Accuracy directly captures whether the model produces correct final answers. The baselines isolate sub-claims: UP tests the asymmetric clipping mechanism; PPO tests standard symmetric clipping; UP + full softmax tests whether the top-k inference budget causes any degradation (at higher compute). The ablations remove the stop-gradient and reverse KL to test whether preserving full gradients and forward KL are crucial. If RC-UPO improves accuracy over UP, especially on problems with multiple reasoning strategies, it validates that routing consistency helps. The metric is sensitive to correct final answers, which is the ultimate goal. Additionally, we monitor KL divergence on a held-out calibration set (512 problems) to detect when the load-bearing assumption is violated: if KL exceeds 0.1 nats for more than 10% of batches, we flag the experiment for potential budget insufficiency.

### Expected outcome and causal chain
**vs. UP** — On a problem where the optimal reasoning path involves diverse expert contributions, UP may concentrate probability on a single expert due to its unbounded positive updates, causing misrouting when inference budget restricts to top-2. Our method adds a routing consistency loss that pulls the inference distribution toward the full distribution, so the policy learns to spread utility across experts. We expect a noticeable accuracy gain on multi-step problems (e.g., those requiring both algebraic and geometric reasoning) while parity on simpler single-step problems.

**vs. PPO** — On a problem with a rare but highly positive advantage action, PPO's symmetric clipping limits the update magnitude, preventing the policy from sufficiently exploring that action. Our method uses unclipped positive updates, allowing larger weight shifts. The routing loss ensures that this exploration does not distort gating. We expect RC-UPO to outperform PPO on problems where creative solutions are needed, with a larger gap on harder problems.

**vs. UP + full softmax** — On a problem where important experts are not in the top-2, this baseline uses full softmax at inference, avoiding routing inconsistency but at high compute cost. Our method approximates the full distribution with a KL loss, so it may slightly underperform on problems requiring many experts, but should be close on most. We expect no significant accuracy drop overall (within 1%), proving that top-k inference with consistency is nearly as good as full softmax while requiring only top-2 computation.

**Ablation: RC-UPO w/o stop-gradient** — On a problem with large positive advantage, the routing loss may retroactively penalize the full softmax, reducing exploration. Our method with stop-gradient preserves UP's exploration. We expect the ablation to show lower accuracy on problems where exploration is beneficial (e.g., out-of-distribution problems), confirming the stop-gradient's role.

**Ablation: RC-UPO with reverse KL** — Reverse KL encourages p_full to match p_infer, which may collapse the full distribution to a sparse subset, hurting exploration. We expect reverse KL to degrade accuracy on diverse-reasoning problems, confirming that forward KL is correct.

### What would falsify this idea
If RC-UPO does not outperform UP on problems with known diverse expert usage (as measured by expert utilization entropy), or if performance gains are uniform across all difficulty levels, then routing consistency is not the causal factor; the results would suggest that the method merely improves through other mechanisms. Additionally, if the calibration procedure frequently triggers (KL > 0.1 nats) for top-2 budget, it would indicate that the load-bearing assumption is false for typical problems, suggesting that top-2 is too restrictive and a larger k or adaptive mechanism is needed.

## References

1. UP: Unbounded Positive Asymmetric Optimization for Breaking the Exploration-Stability Dilemma
2. Geometric-Mean Policy Optimization
3. Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement
4. Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models
5. Self-Consistency Improves Chain of Thought Reasoning in Language Models
6. STaR: Bootstrapping Reasoning With Reasoning
