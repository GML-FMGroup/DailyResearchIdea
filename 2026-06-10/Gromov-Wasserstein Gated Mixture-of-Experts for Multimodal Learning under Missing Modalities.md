# Gromov-Wasserstein Gated Mixture-of-Experts for Multimodal Learning under Missing Modalities

## Motivation

Existing multimodal mixture-of-experts frameworks such as I2MoE and CAG-MoE assume all modalities are present at inference; their gating and expert routing mechanisms break when modalities are missing because they rely on fixed input sets or cross-attention over complete modality combinations. This structural limitation prevents deployment in real-world settings where sensor failures or occlusions are common, and addressing it requires a gating function that intrinsically handles variable-sized input sets.

## Key Insight

Gromov-Wasserstein distance compares two point clouds of different cardinalities by aligning their intra-distance matrices, making it a natural similarity measure for sets of available modality embeddings versus expert-specific prototype sets, thereby enabling robust routing under arbitrary missing modality patterns.

## Method

### (A) What it is
**GW-MoE** is a mixture-of-experts architecture where the gating network uses Gromov-Wasserstein (GW) distance between the set of available modality embeddings and each expert's learned prototype point set to compute routing weights. It outputs a weighted combination of expert outputs, handling missing modalities by simply omitting them from the input set.

### (B) How it works
```python
# GW-MoE Gating Mechanism
# Input: Set E = {e_i for i in available_modalities}, each e_i is a d-dim embedding
# Expert prototypes: each expert j has a fixed set P_j = {p_{j,1}, ..., p_{j,M}} where M = total modality types
# p_{j,m} is a learned d-dim vector (one per modality per expert, shared across instances? Actually per expert)

def gw_distance(X, Y, p=2):
    # Compute pairwise distance matrices D_X and D_Y (rows/cols indexed by points)
    # Solve GW optimization: min_T sum_{i,j,k,l} |D_X[i,k] - D_Y[j,l]|^p * T_{i,j} * T_{k,l}
    # subject to T being a soft coupling matrix (transport plan between points of X and Y)
    # Return optimal transport loss (scalar)

# For each expert j:
weights = []
for j in range(N_experts):
    d_j = gw_distance(E, P_j, p=2)
    weights.append(exp(-d_j / tau))  # tau = temperature, default 0.1
weights = softmax(weights)

# Final output: y = sum_j weights[j] * expert_j(E)  # expert_j processes the full set (missing modalities handled by zero-padding or mask)

# Training: minimize task loss + auxiliary loss that encourages expert specialization
# Expert prototypes P_j are learned jointly; we also simulate missing modalities at training time by randomly dropping modality sets.
```
Hyperparameters: temperature τ=0.1 (controls softmax sharpness), number of experts N_experts=4 per modality? Actually we use 4 experts total across modalities, each with its own prototype set covering all modality types.

### (C) Why this design
We chose **GW distance over KL divergence** for gating because KL requires distributions over the same space, whereas GW operates on sets of different cardinalities—critical for missing modalities. The trade-off is that GW is more computationally expensive (O(n³) in point count, but n is small, ≤5 modalities). We chose **learned prototype points per expert** rather than full Gaussian distributions to avoid sampling variance and keep training stable; this sacrifices the ability to model intra-modal variance, but in practice experts become specialized enough with deterministic prototypes. We opted for **softmax over all experts** instead of top-k selection to allow gradient flow to all experts, preventing dead experts; the downside is that poorly matched experts contribute small but non-zero weight, potentially diluting gradients. Finally, we **simulate missing modalities by random dropout during training** (each modality independently dropped with probability 0.2) to expose the gating to diverse missing patterns; this risks overfitting to the drop distribution, but we mitigate by varying the dropout probability per batch.

### (D) Why it measures what we claim
The computational quantity `d_j = gw_distance(E, P_j)` measures the **geometric dissimilarity** between the input modality set and expert j's prototype set. This operationalizes the concept of **expert affinity under missing modalities** because GW distance is invariant to the cardinality of the sets—it aligns points via their internal distances, so dropping a modality merely removes its entry from the distance matrix, and the remaining points still match to the closest prototypes. The **assumption** is that the expert's prototype set accurately represents the structural arrangement of modalities it expects; this assumption fails when the expert is specialized to a specific combination of modalities that are missing (e.g., expert for audio+video when only audio is present), in which case d_j reflects only the similarity of the available modalities to the closest subset of prototypes, potentially overestimating affinity. To mitigate, we incorporate a **contrastive auxiliary loss** during training that encourages prototypes of different experts to be well-separated and centered on the modalities they actually process.

## Contribution

(1) A novel gating mechanism for multimodal mixture-of-experts that uses Gromov-Wasserstein distance to compute routing weights from variable-sized sets of modality embeddings, enabling robust operation under arbitrary missing modality patterns. (2) A training framework that simulates missing modalities via random dropout and jointly learns expert-specific prototype point sets via an auxiliary contrastive loss, yielding specialized experts that maintain performance when modalities are missing. (3) Empirical validation (in concept, not experiment) that the proposed method outperforms fixed-input MoE baselines on a synthetic multimodal dataset with controlled missing rates.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CMU-MOSEI | Multimodal emotion benchmark with missing mods. |
| Primary metric | F1-score | Balanced measure for multi-class emotion. |
| Baselines | I2MoE | Interpretable MoE with interaction experts. |
| Baselines | CAG-MoE | Cross-attention gated MoE baseline. |
| Baselines | Vanilla MoE | Standard MoE with concatenation. |
| Ablation-of-ours | GW-MoE (no aux loss) | Tests importance of contrastive auxiliary. |

### Why this setup validates the claim
The experimental design tests the central claim that GW-MoE's Gromov-Wasserstein gating improves multimodal fusion under missing modalities by selecting experts based on geometric similarity rather than distributional alignment. CMU-MOSEI provides real-world missing modality patterns via incomplete sequences, making it a natural testbed. F1-score captures overall performance across emotion classes, penalizing both false positives and negatives. I2MoE and CAG-MoE represent state-of-the-art multimodal MoEs that lack explicit handling of missing modalities, while Vanilla MoE serves as a simple baseline. The ablation (GW-MoE without auxiliary loss) isolates the effect of the contrastive specialization loss, which is critical for prototype separation. Together, these baselines allow us to attribute any gains to the GW mechanism and auxiliary loss, and a clear performance gap on instances with missing modalities would validate the claim, while uniform gains would suggest alternative explanations.

### Expected outcome and causal chain

**vs. I2MoE** — On a case where audio modality is missing but video and text are present, I2MoE's interaction experts cannot align missing audio prototype, leading to degraded fusion weight for audio-related experts. Our method uses GW distance to match available modalities to expert prototypes, correctly routing to video+text experts despite missing audio. Thus, we expect a noticeable performance gap on samples with missing modalities, especially when a single modality is absent.

**vs. CAG-MoE** — On a case where all modalities are present but one is noisy, CAG-MoE's cross-attention amplifies noisy patterns due to pairwise attention. Our method's GW distance compares internal geometries, down-weighting experts whose prototypes are far from the noisy sample's structure. Therefore, we expect our method to be more robust to noise, with higher F1 on noisy subsets but similar performance on clean data.

**vs. Vanilla MoE** — On a case where two modalities are missing (e.g., only audio available), Vanilla MoE concatenates zero-padded embeddings, diluting signal. Our method omits missing modalities entirely in the GW distance computation, so the routing depends solely on the available audio's geometry relative to prototypes. We expect a large improvement on severely incomplete samples.

### What would falsify this idea
If GW-MoE shows similar improvement on full-modality samples as on missing-modality samples (i.e., no interaction effect with missingness), the claim that GW distance specifically aids missing modality handling is falsified.

## References

1. I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts
2. CAG-MoE: Multimodal Emotion Recognition with Cross-Attention Gated Mixture of Experts
3. What to align in multimodal contrastive learning?
4. Intrinsic User-Centric Interpretability through Global Mixture of Experts
5. MoMa: Efficient Early-Fusion Pre-training with Mixture of Modality-Aware Experts
6. The future of human-centric eXplainable Artificial Intelligence (XAI) is not post-hoc explanations
7. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
8. Multimodal Learning Without Labeled Multimodal Data: Guarantees and Applications
