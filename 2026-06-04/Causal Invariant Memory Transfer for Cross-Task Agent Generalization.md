# Causal Invariant Memory Transfer for Cross-Task Agent Generalization

## Motivation

Existing memory generation methods, such as AgentRR and Agent Workflow Memory, rely on feature similarity to reuse trajectory patterns, limiting transfer to tasks sharing surface features. AgentRR, for instance, replays experiences only in tasks with similar interaction traces, failing when tasks differ in objects, goals, or environment dynamics. This structural limitation stems from treating trajectories as bags of actions rather than causal processes, preventing extraction of invariant causal structures that generalize across dissimilar tasks.

## Key Insight

Causal structures underlying successful agent behavior remain invariant across tasks despite surface differences, and enforcing invariance to causal-preserving augmentations via contrastive learning extracts these structures even when tasks share no observable features.

## Method

### Causal Invariant Memory Transfer (CIMT)

(A) **What it is**: We propose Causal Invariant Memory Transfer (CIMT), a method that extracts causal memory tips from agent trajectories by learning representations invariant to task-specific surface features. Input: a set of successful trajectories from multiple tasks; Output: a set of causal memory tips (subroutines) with preconditions for cross-task reuse.

(B) **How it works**:
```python
# Training Phase
Input: D = {τ_i} where τ_i = [(o1, a1, r1), ..., (oT, aT, rT)] from tasks T
Hyperparameters: temp τ=0.1, dim d=128, batch_size=64, lr=1e-4, epochs=100, K_initial=10

# Step 1: Causal-Preserving Augmentation
# For MTAB domain, augmentations ϕ include:
#   (a) Randomly rename object identifiers (e.g., 'key1' ↔ 'key2')
#   (b) Swap colors of objects (e.g., red/blue)
#   (c) Randomize irrelevant object positions (within bounds that do not affect reachability)
#   (d) Shuffle independent sub-steps (detected via temporal adjacency; actions with no causal dependency may be swapped)
def augment(trajectory):
    # apply random subset of (a)-(d) independently
    return τ'

views set V = {τ, τ'} for each τ

# Step 2: Contrastive Training
Initialize Enc: Transformer(d_model=128, n_layers=4, n_heads=8, ff_dim=512, dropout=0.1) → latent z ∈ ℝ^d
Optimizer: Adam, lr=1e-4
for epoch in range(100):
    for batch B of 64 trajectories (each with two views):
        encode all views → Z = [z_aug for each view]
        positive pairs: views from same trajectory; negative pairs: all other views in batch
        loss = -log(exp(sim(z, z+)/0.1) / sum(exp(sim(z, z_i)/0.1)))
        update Enc via Adam

# Assumption: The augmentations ϕ preserve all causal dependencies while altering surface features.
# Validation: On a held-out calibration set of 512 trajectories, we perform interventions:
#   - Causal intervention: remove a critical action; check if latent distance d(z_orig, z_interv) > threshold 0.5 (pairwise t-test, p<0.01)
#   - Surface intervention: rename objects; check if d(z_orig, z_surface) < 0.1
# Augmentations that fail these tests on >10% of cases are rejected and redesigned.

# Step 3: Memory Tip Extraction
After training, cluster all trajectory latents using K-means with K chosen by silhouette score (range 5–20).
For each cluster c:
    select the trajectory closest to centroid as representative τ_rep
    generate a natural-language tip using an LLM (e.g., GPT-4) with prompt: "Summarize the causal routine that made this trajectory successful: {τ_rep}"
    extract preconditions from the centroid via a learned decoder: 2-layer MLP (hidden=256, output=binary vector of state predicates) trained on the calibration set.

# Inference Phase (on new task, possibly unseen)
Input: initial steps τ_new (first k steps)
Encode τ_new to z_new.
Find nearest cluster centroid (cosine similarity) and retrieve the corresponding tip and precondition.
If precondition matches current state (via embedding similarity threshold >0.8), inject tip into agent context.
```

(C) **Why this design**: We chose contrastive learning with causal-preserving augmentations over explicit causal discovery (e.g., PC algorithm) because the latter requires symbolic state extraction and is brittle under sensor noise, whereas contrastive learning scales to high-dimensional observations and directly optimizes invariance. The trade-off is that the augmentations must be hand-designed per domain (e.g., renaming objects, shuffling independent sub-steps) to avoid introducing spurious invariances. We selected a Transformer encoder for trajectory representation over an RNN because Transformers capture long-range action dependencies critical for causal structure, at the cost of higher memory requirements. Clustering as an intermediate step before tip generation avoids the need to define causal patterns a priori; however, the number K must be set heuristically, potentially missing fine-grained routines. Using an LLM for tip generation produces human-readable output but may hallucinate; we limit this by providing the full trajectory as context.

(D) **Why it measures what we claim**: The contrastive loss's similarity between augmented views of the same trajectory measures **causal invariance** because the augmentations are designed to preserve causal structure while altering surface features; the assumption is that ϕ is a perfect causal-invariance filter. We validate this assumption via intervention tests on a calibration set (Step 2). The latent representation z after training measures **causal content** because it must remain consistent under causal-preserving perturbations; the assumption is that the contrastive objective's push/pull dynamics isolate the minimal sufficient statistics for predicting future states. This assumption fails when non-causal statistics (e.g., repeated action patterns) are also invariant to ϕ, in which case z captures spurious correlations. Finally, the centroid of a cluster measures **shared causal structure** across trajectories; the assumption is that trajectories with similar latents share the same underlying causal routine. This assumption fails when two distinct routines map to the same latent due to encoder capacity limits, leading to conflated tips.

## Contribution

(1) A novel Causal Invariant Memory Transfer (CIMT) framework that extracts cross-task generalizable memory tips by learning representations invariant to task-specific surface features via causal-preserving augmentations. (2) The design principle that defining augmentations which preserve causal structure is critical for unsupervised discovery of transferable agent behaviors, validated through contrastive learning. (3) A trajectory augmentation toolkit that renames objects, randomizes independent subroutines, and applies domain-specific invariances to facilitate causal invariance learning.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-Task Agent Benchmark (MTAB) + synthetic intervention dataset (SYNTH) | Tests cross-task causal invariance + verifies causal representation. |
| Primary metric | Task success rate on unseen tasks (MTAB); F1 score of causal structure recovery (SYNTH) | Measures transfer + causal faithfulness. |
| Baseline 1 | No memory (raw agent) | Shows baseline without tips. |
| Baseline 2 | Random tip retrieval | Tests relevance of tip selection. |
| Baseline 3 | Causal discovery (PC algorithm) | Contrast with explicit causal method. |
| Ablation | CIMT w/o contrastive loss | Tests effect of invariance learning. |
| Upper bound | Oracle (ground-truth causal tips) | Upper performance limit. |

### Why this setup validates the claim
This combination forms a falsifiable test of the central claim: that causal invariance via contrastive learning enables efficient cross-task transfer of memory tips. The MTAB dataset provides tasks sharing causal routines but differing in surface features (e.g., object colors, names). The primary metric directly measures transfer success on unseen tasks, which is the intended outcome. The three baselines test distinct sub-claims: (1) "No memory" checks whether any memory tips are beneficial; (2) "Random tip retrieval" tests whether the selection mechanism (nearest cluster) is crucial; (3) "Causal discovery" tests whether contrastive learning is more effective than explicit symbolic causal inference. The ablation removes the contrastive loss to verify that the learned invariance, not the architecture alone, drives gains. The synthetic SYNTH dataset with known ground-truth causal variables enables direct measurement of causal structure recovery (F1), validating that the latent representations indeed capture causal invariants. The metric must show improvements on tasks where causal invariants exist but surface features change, and parity on tasks with no such structure.

### Expected outcome and causal chain

**vs. No memory (raw agent)** — On a case where the agent must pick a key based on its shape (causal) but colors are random (surface variation), the raw agent treats each color as a new state, requiring re-learning. Our method extracts the invariant 'pick the square key' tip, so retrieval succeeds even with unseen colors. We expect a noticeable gap on color-altered tasks but parity on static tasks.

**vs. Random tip retrieval** — On a case where the correct causal tip is 'open door with blue key' but random retrieval might choose 'pull lever', the baseline fails to execute the right routine because it selects without state matching. Our method uses precondition matching, so it retrieves the correct tip when the current state satisfies preconditions. We expect a large gap on tasks requiring precise precondition matching, and a smaller gap where any tip works.

**vs. Causal discovery (PC algorithm)** — On a case with noisy sensors (e.g., partial observability), PC algorithm fails to recover the true causal graph because it requires exact conditional independence tests, producing false edges and wrong tips. Our method learns robust latent representations from raw observations, tolerating noise via contrastive invariance. We expect our method to maintain high performance under sensor noise while PC degrades sharply.

**vs. Ground-truth causal variables (SYNTH)** — On the synthetic dataset where true causal graph is known, we measure the F1 score of extracted tips against ground-truth causal subroutines. Our method should achieve high precision and recall (F1>0.8) because contrastive learning recovers the invariant causal structure. If F1 is low (<0.5), augmentations are not perfectly causal-preserving, invalidating our central assumption.

### What would falsify this idea
If our method shows uniform improvement over all baselines across all task subsets, including those without causal invariance, then the gains are not due to causal invariance but to some artifact (e.g., better representation). Alternatively, if the ablation (w/o contrastive) performs similarly to full CIMT, then the contrastive mechanism is not capturing causal structure, invalidating the central claim. Also, if on SYNTH the F1 score is below 0.5, the augmentations fail to preserve causal structure, directly falsifying our load-bearing assumption.

## References

1. Trajectory-Informed Memory Generation for Self-Improving Agent Systems
2. Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models
3. A-MEM: Agentic Memory for LLM Agents
4. Get Experience from Practice: LLM Agents with Record & Replay
5. StreamBench: Towards Benchmarking Continuous Improvement of Language Agents
6. TextGrad: Automatic "Differentiation" via Text
7. Agent Workflow Memory
8. AIOS: LLM Agent Operating System
