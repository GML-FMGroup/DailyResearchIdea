# Recoverability-Aware Pruning via On-Policy Sensitivity

## Motivation

Existing pruning criteria for LLMs, such as layer similarity used in 'The Unreasonable Ineffectiveness of the Deeper Layers', select layers to prune based on static properties of the pre-trained model. However, these heuristics do not account for how well a pruned layer can recover during the subsequent on-policy distillation process, which is now standard for recovery. This structural mismatch means pruning decisions are suboptimal: layers that are similar may still be hard to recover, while dissimilar layers may recover easily under distillation. We need a pruning criterion that directly measures recoverability under the actual distillation process.

## Key Insight

The change in distillation loss after temporarily removing a layer during a short on-policy rollout is a causal measure of that layer's recoverability, because it isolates the effect of pruning on the student's ability to learn from the teacher without confounding factors from recovery steps.

## Method

### (A) What it is
**Recoverability-Aware Pruning via On-Policy Sensitivity (RAPOS)** takes a pretrained LLM (student S) and a frozen teacher T (identical to S initially), target depth k, rollout length L (calibrated), batch size B=64, and recovery steps per iteration R=50. It outputs a pruned model with k layers removed.

### (B) How it works
```pseudocode
Input: Pretrained model M (size N layers), teacher T (copy of M), target remaining layers N-k,
       batch size B=64, recovery steps per iteration R=50, short rollout set L_candidates = {64,128,256}
Student S ← M
PrunedLayers ← empty set
# Calibration: Validate short-rollout correlation
for L in L_candidates:
    compute R(l) = KL_div(S_with_identity[l](batch), T(batch)) on a validation set of 100 sequences
    record top-5 layers with smallest R(l)
if top-5 layers for L=64 and L=256 agree on at least 4 out of 5:
    L_used ← 64
else:
    L_used ← 256
while |S.layers| > N-k do
    for each layer l in S.layers do
        S' ← S with layer l replaced by identity (temporarily)
        batch ← random sample of B sequences from training data
        S'_outputs ← S'(batch)[:,:L_used]
        T_outputs ← T(batch)[:,:L_used]
        L_short ← KL_div(softmax(S'_logits/τ), softmax(T_logits/τ)) with τ=1.0, averaged over tokens and batch
        restore layer l to S
        R(l) ← L_short
    end for
    l* ← argmin_l R(l)
    PrunedLayers ← PrunedLayers ∪ {l*}
    S ← S with layer l* removed (permanently)
    for step in 1..R do
        batch ← sample B sequences
        rollout_len ← current effective length from ShortOPD repetition detector
        S_outputs ← S(batch)[:,:rollout_len]
        T_outputs ← T(batch)[:,:rollout_len]
        loss ← KL_div(softmax(S_logits/τ), softmax(T_logits/τ)) with τ=1.0
        update S parameters via SGD
    end for
end while
return S
```

### (C) Why this design
We chose short rollouts (L=64) over longer rollouts because they provide a fast, low-variance estimate of recoverability without requiring expensive generation; this efficiency allows iterative pruning, but risks missing long-range dependencies that affect recovery. We replaced pruned layers with identity mapping rather than zeroing or masking, because identity preserves the residual stream and avoids catastrophic activation shifts, giving a cleaner causal signal; however, this may underestimate the difficulty of recovering a fully removed layer, as the residual connection still passes information. We pruned iteratively (one layer per recovery phase) rather than all at once, because iterative pruning allows the student to adapt and ensures that each layer's recoverability is measured in the context of already-recovered layers; the cost is increased runtime proportional to the number of pruning rounds. Finally, we used the ShortOPD schedule for recovery steps (with repetition-based effective length) rather than fixed-length distillation, because it focuses on informative prefixes, but adds overhead of the repetition detector and assumes repetition is the main failure mode.

### (D) Why it measures what we claim
The computational quantity `R(l) = L_short` measures the recoverability of layer l because it quantifies the distillation loss contribution of that layer under the same on-policy process used for recovery, under the assumption that a lower loss after temporary removal indicates that the student can more easily learn to compensate. This assumption fails when the short rollout generation collapses (e.g., repetitive loops), in which case `L_short` reflects generation quality rather than intrinsic recoverability. The use of a short rollout assumes that short-sequence recoverability correlates with long-sequence recoverability; if the layer's recovery dynamics differ on long sequences (e.g., due to accumulating errors), the metric may identify layers that recover quickly but are brittle over longer contexts. To mitigate this risk, we perform a preliminary calibration step: before iterative pruning, we compare pruning decisions made with L=64 and L=256 on a small validation set (100 sequences). If the top-5 layer decisions agree on at least 4 out of 5, we use L=64; otherwise, we use L=256. Additionally, we note that the use of identity replacement assumes residual signal preservation. If identity overestimates compensation, R(l) may be too optimistic. The teacher-student KL divergence operationalizes the distillation objective: a small KL after layer removal means the pruned student already produces similar distributions to the teacher, implying that the removed layer is either redundant or easily replaced during recovery.

## Contribution

(1) A novel pruning criterion that directly measures a layer's recoverability under on-policy distillation by computing the distillation loss change after its temporary removal. (2) An iterative pruning-recovery algorithm that uses these recovery potentials to guide layer selection, ensuring pruning decisions are adaptive to the model's current state and the distillation process. (3) A framework for integrating on-policy feedback into the pruning loop, enabling closed-loop pruning that adapts to the recovery dynamics.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Wikitext-2 | Standard language modeling benchmark. |
| Primary metric | Perplexity | Direct measure of next-token prediction quality. |
| Baseline 1 | Random layer pruning | Tests if sensitivity metric outperforms chance. |
| Baseline 2 | One-shot pruning + full recovery | Tests if iterative pruning is beneficial. |
| Baseline 3 | Standard OPD without short-rollout | Tests if recoverability metric helps recovery. |
| Baseline 4 | Static sensitivity (gradient norms) | Isolates benefit of on-policy dynamic measurement. |
| Ablation | RAPOS without ShortOPD schedule | Isolates effect of dynamic rollout length. |

### Why this setup validates the claim
This experimental design creates a direct, falsifiable test of RAPOS's central claim: that iterative pruning guided by a short on-policy recoverability metric (R(l)) followed by ShortOPD-based recovery yields better final model performance than alternative pruning and recovery strategies. The combination of baseline comparisons targets each key novelty: random pruning isolates the value of the sensitivity metric; one-shot pruning isolates the benefit of iterative vs. simultaneous layer removal; static sensitivity (gradient norms) isolates the advantage of on-policy dynamic measurement over a static proxy; standard OPD (without the short-rollout metric) evaluates whether the sensitivity score adds value beyond a standard distillation-based recovery. The ablation (fixed-length recovery) tests the contribution of the ShortOPD schedule. Perplexity directly measures language modeling quality, which is the primary capability targeted by pruning and recovery. If RAPOS consistently achieves lower perplexity than these baselines, especially under high sparsity, the claim is supported; otherwise, the claimed mechanisms are not effective. Total compute for RAPOS is estimated at ~100 GPU-hours for 20% pruning on a 7B model (using NVIDIA A100), which is feasible and accounts for iterative pruning overhead.

### Expected outcome and causal chain

**vs. Random layer pruning** — On a case where a critical layer (e.g., one essential for long-range coherence) is removed, random pruning may select that layer, causing high perplexity after recovery because the student cannot fully compensate for a randomly chosen structural hit. Our method instead scores each layer by short-rollout recoverability, so it avoids removing such critical layers, prioritizing those whose function can be easily absorbed. We expect a noticeable gap in perplexity between RAPOS and random pruning, especially on long-context sequences (e.g., perplexity on segments of length >512) where the critical layer's role is most apparent.

**vs. One-shot pruning + full recovery** — On a case where the model must recover from removing multiple layers simultaneously (e.g., 25% of layers at once), one-shot pruning causes a large distribution shift and the student struggles to adapt because it must compensate for many missing functions at once. Our method instead prunes one layer per iteration, allowing the student to gradually adapt and recover after each removal. We expect RAPOS to achieve lower perplexity than one-shot pruning, particularly at high sparsity levels (e.g., >30% layers removed), where the cumulative advantage of iterative adaptation becomes large.

**vs. Standard OPD without short-rollout** — On a case where a layer appears redundant based on static similarity (e.g., high cosine similarity) but is actually needed for short-range dynamics, standard OPD (which may prune based on similarity) would remove it, causing recovery difficulty. Our method instead uses a dynamic, on-policy recoverability metric that captures how easily the student can compensate for that specific layer after actual generation. We expect RAPOS to outperform standard OPD across all sparsity levels, with the gap widening as more layers are pruned, as our metric targets truly recoverable layers.

**vs. Static sensitivity (gradient norms)** — On a case where gradient norms indicate a layer is important for convergence but the layer is actually easy to recover under distillation, static sensitivity may incorrectly retain it. Our on-policy metric directly measures recovery difficulty and thus may prune such layers more aggressively. We expect RAPOS to achieve lower perplexity than pruning by gradient norm sensitivity, especially when the recovery process can compensate for high-gradient layers.

### What would falsify this idea
If RAPOS does not outperform one-shot pruning with full recovery at high sparsity levels, or if random layer pruning achieves similar perplexity reduction across all sparsity levels, then the recoverability metric and iterative strategy are not providing the claimed benefit.

## References

1. ShortOPD: Recovering Pruned LLMs with Short-to-Long On-Policy Distillation
2. The Unreasonable Ineffectiveness of the Deeper Layers
3. LLM Pruning and Distillation in Practice: The Minitron Approach
4. Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning
5. The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction
6. LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models
7. Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt
8. Training Trajectories of Language Models Across Scales
