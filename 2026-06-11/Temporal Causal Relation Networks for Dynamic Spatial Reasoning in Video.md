# Temporal Causal Relation Networks for Dynamic Spatial Reasoning in Video

## Motivation

Existing spatial reasoning benchmarks such as FloorplanQA evaluate only static scenes or simple temporal sequences, neglecting that real-world video involves dynamic interactions where object identities and spatial relations change non-trivially over time. This structural limitation prevents models from reasoning about causal dependencies between relation transitions, as they treat each frame independently or monotonically.

## Key Insight

By modeling the temporal evolution of spatial relations as a causal graph where each edge's state transition is conditioned on the history of object identities and graph structure, we can capture compositional dynamics while preserving permutation invariance through set-based representations.

## Method

### (A) What it is
Temporal Causal Relation Networks (TCRNet) predict a causal spatial relation graph for each video frame, maintaining object identity across time via a permutation-invariant memory module.

### (B) How it works
```python
Input: Video frames {I_1,...,I_T}
Output: Predicted spatial relation graph G_t = (V_t, E_t) for each t, and answer to queries.

1. Visual Encoder:
   - Use a DETR-style detector with object queries to extract a permutation-invariant set of object embeddings O_t = {o_i_t} for each frame.
   - Each o_i_t includes a feature vector and a positional embedding from the detected bounding box.

2. Permutation-Invariant Identity Module (PIIM):
   - Maintain a memory bank M of K slots with learned embeddings.
   - For each new frame t, compute a cost matrix C_{ij} = -cosine(o_i_t, M_j).
   - Apply Sinkhorn algorithm (τ=0.1, 20 iterations) to obtain a soft assignment matrix A (bijective).
   - Update M_j = α*M_j + (1-α)*Σ_i A_{ij}*o_i_t with α=0.9.
   - Output identity embeddings as the matched memory slot for each o_i_t.

3. Temporal Causal Graph Network (TCGN):
   - Construct a directed graph per frame where nodes are object identities (from PIIM) and edges encode spatial relations relative position and embedding differences.
   - Apply a relational graph convolution: node features are updated as h_i = σ(W1 h_i + Σ_j W2 [h_i, h_j, e_{ij}]).
   - Use a causal graph transformer with masked self-attention over past frames to predict next frame's graph: for each time t, the decoder outputs edge probabilities P(e_{ij}^t | previous graphs).
   - The decoder is a 2-layer MLP per edge, with shared parameters across time.

4. Training:
   - Supervised on synthetic videos with ground truth relation graphs (cross-entropy loss for edges).
   - Contrastive loss for identity: pull matched pairs together (from Sinkhorn) and push unmatched apart.

5. Inference:
   - For a video, run TCRNet to predict per-frame graphs. Answer spatial queries (e.g., 'Is object A on table? at time t) by thresholding edge probability.
```

### (C) Why this design
We chose a set-based visual encoder (DETR-style) over explicit object tracking because it naturally handles permutation invariance, accepting the cost of potential identity confusion after long occlusion; our PIIM mitigates this via Sinkhorn matching that enforces bijectivity. We opted for causal temporal masking in the transformer rather than bidirectional attention because the task requires forward prediction of relation changes, at the cost of ignoring future context that could disambiguate ambiguous transitions—but such future information would violate the causal assumption we want to learn. We selected graph transformers over standard RNNs because attention scales to variable object counts and captures long-range dependencies, though it increases memory use for long sequences; we accept this as synthetic videos have bounded length. The contrastive identity loss was chosen over reconstruction loss because it directly aligns with the goal of preserving identity, though it may be less stable with small batch sizes—we balance using memory bank soft updates (α=0.9).

### (D) Why it measures what we claim
The predicted edge probability in the output graph measures the model's understanding of spatial relations at each timestep, operationalizing 'spatial reasoning' under the assumption that causal dependency between relation transitions is correctly captured by the temporal decoder; this assumption fails when non-causal latent factors (e.g., lighting changes) alter visual features, in which case the model may still output correct relations if identity preservation remains robust. The identity preservation loss (contrastive between matched embeddings) measures the ability to maintain object identity over time, operationalizing 'identity preservation' under the assumption that object appearance evolves smoothly; this assumption fails when objects undergo radical appearance change (e.g., color change), in which case the metric reflects appearance similarity rather than true identity. The temporal causal masking ensures that edge predictions at time t depend only on past graphs, operationalizing 'causal reasoning' under the assumption that relation changes are Markovian; this assumption fails when long-range dependencies require memory beyond the transformer's context, in which case the metric reflects first-order approximations.

## Contribution

(1) A causal graph neural network architecture (TCRNet) for dynamic spatial reasoning that explicitly models temporal compositionality of relations via a temporal causal decoder. (2) A permutation-invariant identity preservation module (PIIM) that maintains object identity across frames using differentiable Sinkhorn matching and a memory bank. (3) A synthetic video dataset with ground truth spatial relation transitions, generated using ProcTHOR, for evaluating temporal spatial reasoning.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | CLEVRER | Ground-truth causal relation graphs per frame. |
| Primary metric | Relation graph F1 score | Measures spatial relation prediction accuracy. |
| Baseline 1 | Bi-directional Transformer | No causal masking, uses future context. |
| Baseline 2 | Object tracker + frame GCN | No temporal graph correlation. |
| Ablation of ours | w/o PIIM | No identity preservation module. |

### Why this setup validates the claim

This design tests the central claim that causal masking and identity preservation are essential for accurate spatial relation prediction over time. CLEVRER provides ground-truth causal graphs, enabling direct evaluation of edge predictions. The bidirectional baseline tests whether future context is necessary; if our method outperforms it, causal masking is validated. The tracker+GCN baseline tests whether per-frame static graphs suffice; outperforming it confirms the value of temporal graph modeling. The ablation without PIIM tests whether identity preservation is critical; a drop in performance would confirm its role. The F1 metric captures both precision and recall of relation edges, directly aligning with the goal of accurate spatial reasoning. Together, these components form a controlled falsification: if our method does not outperform both baselines and the ablation on CLEVRER, the central claim is unsupported.

### Expected outcome and causal chain

**vs. Bi-directional Transformer** — On a case where an object briefly occludes another and reappears with swapped identities, the bi-directional baseline uses future frames to see the swap and may incorrectly retroactively alter past relations; its mechanism lacks causal constraint, leading to temporally inconsistent graphs. Our method, with causal masking, predicts each frame only from the past, so it resolves identity via PIIM and maintains consistent relations; we expect a noticeable gap on frames after occlusion, but parity on static sequences.

**vs. Object tracker + frame GCN** — On a case where an object moves behind a table and its trajectory is temporarily lost, the tracker loses identity and the frame GCN treats it as a new object, producing a broken relation graph. Our method’s PIIM uses memory bank with Sinkhorn matching to maintain identity even under occlusion, and the causal decoder uses past relations to infer the object’s position; we expect a large gap on occlusion-heavy clips, but similar performance on simple sequences.

### What would falsify this idea

If our method’s performance on CLEVRER is not significantly higher than both baselines on the subsets with occlusions or identity swaps, then the causal masking and identity preservation mechanisms are not producing the expected benefit and the central claim is falsified.

## References

1. FloorplanQA: A Benchmark for Spatial Reasoning in LLMs using Structured Representations
2. Infinigen Indoors: Photorealistic Indoor Scenes using Procedural Generation
3. AnyHome: Open-Vocabulary Generation of Structured and Textured 3D Homes
4. Infinite Photorealistic Worlds Using Procedural Generation
5. Magic3D: High-Resolution Text-to-3D Content Creation
6. Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation
7. ProcTHOR: Large-Scale Embodied AI Using Procedural Generation
