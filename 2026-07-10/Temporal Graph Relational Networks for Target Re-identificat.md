# Temporal Graph Relational Networks for Target Re-identification in Multi-Target Video Segmentation

## Motivation

Current multi-target video segmentation methods like SAM-MT rely on sparse per-target memory that assumes consistent visibility, causing identity loss when a target is occluded. This is because they lack a mechanism to propagate identity through inter-target relationships during occlusion. Although EntitySAM introduces inter-object communication at the input level, it does not address temporal re-identification after extended disappearance. A structural mechanism is needed to leverage relational context across targets to recover identity when a target reappears.

## Key Insight

Relational graph edges between target queries capture invariant relative motion and appearance patterns that persist even when an individual target is temporarily unobserved, enabling identity propagation through the graph structure.

## Method

### Temporal Graph Relational Network (TGRNet)

(A) **What it is:** TGRNet is a graph-based module that takes per-target query embeddings from each frame and outputs updated queries with propagated identity information. Input: set of target queries at time `t` (Q_t) and historical queries (Q_{<t}). Output: updated queries Q'_t with re-identified targets.

(B) **How it works:**
```pseudocode
# Temporal Graph Relational Network (TGRNet)
# Hyperparameters: GAT layers L=2, hidden dim d=256, 
#   edge update steps K=3, graph matching threshold τ=0.1
# Training: Adam optimizer, lr=1e-4, batch size=8, 50 epochs, single V100 GPU (~20 hours)

For each frame t:
    # Build inter-target graph G_t with nodes = {q_i for each target i}
    # Edges: fully connected, edge features e_{ij} = MLP([q_i; q_j; f_i - f_j])
    #   where f_i are visual features (from encoder)
    
    # Temporal graph linking: connect node i at t to node j at t-1 if 
    #   cosine similarity(q_i_t, q_j_{t-1}) > τ (identity proposal)
    
    # Graph update for K steps:
    for k in 1..K:
        # Intra-frame message passing via GAT on G_t
        q_i' = GAT_layer(q_i, {q_j for j in neighbors})
        # Inter-frame message passing via cross-attention with history
        q_i'' = CrossAttn(q_i', {q_j_{t-1} for all j})
        # Update node features
        q_i ← LayerNorm(q_i' + q_i'')
    
    # Re-identification: after update, assign each updated query to a historical identity
    # using Hungarian algorithm on edge weights from temporal linking.
    # If no match (new target), create new identity.
    
    # Output updated queries for segmentation mask decoding.
```

(C) **Why this design:** We chose a graph-based relational network over independent per-target LSTMs because graphs naturally model pairwise interactions that are crucial for occlusion reasoning; independent memory cannot infer a reappearing target from the behavior of others. We chose GAT over simple convolution to learn attention weights between targets adaptively, trading off computational cost (O(N^2) for N targets) for expressiveness. We included inter-frame cross-attention to propagate identity over time without relying on a single temporal memory, accepting the cost of additional memory for historical queries. The thresholding for temporal linking avoids combinatorial explosion, but may miss true matches if appearance changes drastically; however, the graph update can correct misassignments through subsequent iterations.

(D) **Why it measures what we claim:** The intra-frame edge features `e_{ij} = MLP([q_i; q_j; f_i - f_j])` measure relational similarity: the concatenated query embeddings and feature difference capture joint appearance and motion cues that are invariant under occlusion (since an occluded target's own features are missing, but its relationship to visible targets remains). **Load-bearing assumption:** Relational graph edges between target queries capture invariant relative motion and appearance patterns that persist even when an individual target is temporarily unobserved, enabling identity propagation through the graph structure. *To verify this invariance, we conduct a controlled experiment on synthetic occlusions where target trajectories are generated with known relative motion patterns; we measure the alignment between edge features and ground-truth relational distances.* This assumption fails when all targets are simultaneously occluded in a correlated pattern (e.g., all go behind the same large object), in which case the graph loses all signal and the method reverts to random guess. The inter-frame cross-attention with history measures temporal identity consistency: high attention weight between a current query and a historical query indicates re-identification, because the attention mechanism learns to match queries that correspond to the same physical target under the assumption that query embeddings change slowly over time. This assumption fails when a target undergoes a drastic appearance change in a single frame (e.g., sudden illumination shift), causing the correct match to have low attention; in that case, the method may create a false new identity. The Hungarian matching produces a hard assignment that operationalizes the claim of re-identification: it enforces a one-to-one mapping between current queries and historical identities, assuming that the number of targets does not change. This assumption fails when a target splits or merges, which is not considered in our setting.

## Contribution

(1) A Temporal Graph Relational Network (TGRNet) that integrates intra-frame inter-target relations and inter-frame temporal linking to propagate identity during occlusion. (2) A design principle showing that relational context among targets can serve as an invariant signal for re-identification, outperforming per-target sparse memory. (3) The re-identification module is model-agnostic and can be plugged into any multi-target video segmentation framework that uses per-target queries.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|---|---|---|
| Primary dataset | YouTube-VOS 2019 validation | Multi-target, diverse occlusions. |
| Additional dataset | DAVIS 2017 validation | Long videos with frequent occlusions. |
| Primary metric | J&F mean (region & contour) | Standard for video object segmentation. |
| Secondary metric | IDF1 for identity consistency | Directly measures identity switches. |
| Baseline 1 | Independent per-target LSTM | Tests necessity of inter-target modeling. |
| Baseline 2 | Multi-target SAM2 (SAM-MT) | State-of-the-art single-target base. |
| Ablation | TGRNet w/o temporal cross-attention | Isolates benefit of temporal linking. |

### Why this setup validates the claim

This setup directly tests the central claim that graph-based modeling of inter-target relations improves re-identification and occlusion handling. YouTube-VOS provides natural occlusions and identity switches, while DAVIS 2017 contains long videos with frequent occlusions, testing generalization. Comparing against the per-target LSTM baseline isolates the value of inter-target interactions; if our method outperforms, it confirms that relational reasoning is beneficial. The SAM-MT baseline represents a strong independent-tracker approach; outperforming it shows that even a state-of-the-art per-target method fails under occlusion. The ablation removes temporal cross-attention, testing the necessity of explicit temporal identity propagation. J&F is the primary metric because it captures both segmentation accuracy and temporal consistency, directly reflecting successful re-identification. IDF1 provides a direct measure of identity switches, complementing J&F. To further validate the mechanism, we visualize edge attention weights during occlusion to confirm they capture relative motion patterns, e.g., by computing correlation between attention values and ground-truth relative velocities. A significant gain on occlusion-heavy sequences but parity on simple ones would confirm the mechanism.

### Expected outcome and causal chain

**vs. Independent per-target LSTM** — On a case where two targets cross and briefly occlude each other, the LSTM baseline extrapolates each target's motion independently, causing the occluded target's query to drift and lose identity upon reappearance. Our method uses the intra-frame graph to infer that the disappearing target's motion is linked to the occluding target's trajectory, updating queries accordingly; thus the target is correctly re-identified. We expect a noticeable gap on sequences with frequent mutual occlusions (e.g., J&F gap >5 points, IDF1 gap >10 points) but near-parity on sequences with no occlusions.

**vs. Multi-target SAM2 (SAM-MT)** — On a case where a target exits the frame and re-enters later after a long absence, SAM-MT treats each frame independently and may assign a new identity to the reappearing target, breaking segmentation consistency. Our method's temporal graph linking connects queries across time via cross-attention; the historical query for the missing target remains active and attends to reappearance cues, enabling correct re-identification. We expect a large J&F difference on sequences with temporary target disappearances (e.g., >10 points) but similar performance on continuous tracking.

### What would falsify this idea

If our method shows no gain over baselines on occlusion-heavy subsets, or if the gain is uniform across all scenes rather than concentrated on sequences with occlusions or reappearances, then the claim that graph-based relational reasoning specifically addresses these failure modes would be falsified.

## References

1. SAM-MT: Real-Time Interactive Multi-Target Video Segmentation
2. EntitySAM: Segment Everything in Video
3. Rethinking Query-Based Transformer for Continual Image Segmentation
