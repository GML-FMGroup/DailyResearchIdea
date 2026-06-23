# PSR-Embed: Parallel Subspace Refinement for Constant-Cost Text Embeddings

## Motivation

Iterative refinement methods like GIRCSE improve embedding quality but incur linear cost in the number of refinement steps due to sequential autoregressive generation. This structural limitation arises because each refinement step depends on the previous step's output, preventing parallelization. We propose a parallel subspace refinement approach that eliminates this dependence by decomposing the embedding into orthogonal subspaces refined independently.

## Key Insight

Decomposing the embedding vector into orthogonal subspaces and applying a fixed learned refinement function to each subspace independently makes the refinement cost constant per input, independent of the number of refinement dimensions, because all subspace refinements are computed in parallel with no inter-subspace dependencies.

## Method

### (A) What it is
PSR-Embed (Parallel Subspace Refinement) is a method that takes an initial embedding $h_0$ from a base LLM (dimension $D$) and produces a refined embedding $h_{\text{ref}}$ by projecting $h_0$ onto $K$ orthogonal subspaces, refining each subspace independently via learned functions, and combining them. The cost per input is $O(K \cdot d^2)$ where $d = D/K$, independent of the number of refinement "steps" in the sequential sense.

### (B) How it works
```python
# PSR-Embed: Training and Inference
# Hyperparameters: K subspaces (e.g., K=4), each of dimension d = D//K (assume D divisible, e.g., D=768, d=192)
# Learned: orthogonal projection matrices P_k (D x d) for k=1..K, refinement MLPs f_k with 2 layers, hidden size H=512, GeLU activation
# Training hyperparameters: batch size 256, temperature τ=0.07, λ=0.1 for orthogonality regularizer, AdamW optimizer, lr=1e-4, 100k steps

# Training step:
for each batch:
    h0 = base_LLM(input)  # D-dim, e.g., from BERT-base mean pooling
    h0_pos = base_LLM(positive_input)  # positive pair
    for k in 1..K:
        x_k = P_k.T @ h0               # project to subspace k (d-dim)
        r_k = f_k(x_k)                 # refine (d-dim)
        h_k = P_k @ r_k                # lift back to D-dim
    h_ref = sum_{k=1}^K h_k            # combine
    # contrastive loss with in-batch negatives (InfoNCE)
    sim = cosine_similarity(h_ref, h0_pos) * τ
    loss = -log(exp(sim) / sum(exp(sim with negatives)))
    # orthogonality regularizer
    orth_loss = sum_{k != l} ||P_k.T @ P_l||_F^2
    total_loss = loss + λ * orth_loss
    update parameters via AdamW

# Inference:
    same as training without loss computation
```
### (C) Why this design
We chose orthogonal projection matrices over learned projection heads (e.g., linear layers without orthogonality constraint) because orthogonality ensures that each subspace captures non-redundant information, preventing the same feature from being refined multiple times—a redundancy that would waste capacity and inflate cost linearly with $K$ without quality gain. We also chose $K$ to be a small constant (e.g., 4 or 8) rather than scaling with $D$, accepting that very large $K$ may lead to subspaces too narrow to express meaningful refinement; this trade-off keeps per-subspace dimension $d$ large enough ($d\geq 64$ in practice) for the MLP to learn effective transformations. We chose MLPs over linear transformations for $f_k$ because linear refinements are limited to scaling and rotating the subspace, while a two-layer MLP with ReLU can learn non-linear corrections; the cost is added parameters and compute per subspace, but since $K$ is constant and $d$ modest (e.g., 128), this remains cheap. Finally, we use a contrastive loss with in-batch negatives, following the standard practice from GIRCSE and GTE, because it aligns with the goal of producing discriminative embeddings; alternative losses like triplet or MSE would require explicit negative mining and may not scale to diverse tasks.

### (D) Why it measures what we claim
**Load-bearing assumption**: The sum of independently refined orthogonal subspace embeddings produces an embedding quality equal to a sequential refinement that processes all dimensions jointly. This assumption relies on the task-relevant features being separable into orthogonal subspaces; if cross-subspace interactions are required, independent refinement may lose performance. To verify this assumption, we measure **subspace decorrelation** via the orthogonality regularizer $||P_k^T P_l||_F^2$—when minimized to zero, subspaces are orthogonal, indicating reduced redundancy. However, orthogonality does not guarantee task-relevant independence; tasks requiring cross-subspace information (e.g., compositional reasoning) may suffer. As a diagnostic, we compute **cross-subspace mutual information** (squared canonical correlation) on a validation set: values near zero confirm that subspaces capture distinct information, while high values indicate overlap. This metric measures 
subspace disentanglement" because it quantifies statistical dependence; its failure mode occurs when the task requires joint processing of features across subspaces, in which case low mutual information would reflect artificial separation rather than beneficial independence. The total number of parallel MLP forward passes ($K$) measures **inference cost independence** because wall-clock time equals the max over $K$ of one MLP forward pass, independent of refinement depth in a sequential scheme.

## Contribution

(1) A parallel subspace refinement framework for text embeddings that achieves constant inference cost with respect to refinement steps, replacing sequential autoregressive generation with parallel independent subspace operations.
(2) A training objective combining contrastive loss with an orthogonality regularizer that encourages subspace disentanglement, enabling effective refinement without cross-subspace communication.
(3) Empirical demonstration that PSR-Embed matches or exceeds the quality of sequential refinement methods like GIRCSE on MTEB benchmarks while reducing inference latency by an order of magnitude.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | MTEB benchmark | Standard for embedding quality evaluation |
| Primary metric | Average MTEB score | Captures overall discriminative performance |
| Baseline 1 | Direct LLM embedding (mean pool) | Tests necessity of any refinement |
| Baseline 2 | GIRCSE (sequential refinement) | Tests parallel vs sequential refinement |
| Baseline 3 | NV-Embed (deterministic refinement) | Tests subspace-specific vs global refinement |
| Ablation-of-ours | PSR-Embed w/o orthogonality | Tests role of subspace disentanglement |

### Why this setup validates the claim
This combination of dataset, baselines, and metric forms a falsifiable test of the central claim—that PSR-Embed achieves cost-independent refinement via orthogonal subspaces. The Direct LLM baseline establishes whether refinement is needed at all; GIRCSE tests whether parallel refinement can match sequential refinement without the cost scaling; NV-Embed tests whether subspace-specific refinement beats a global deterministic transformation. The ablation (removing orthogonality) isolates the role of disentanglement. The single metric (average MTEB score) across diverse tasks directly measures embedding quality, so any improvement over baselines confirms the method's effectiveness. If our method fails to outperform on the expected subsets, the reasoning chain is invalidated.

### Expected outcome and causal chain

**vs. Direct LLM embedding** — On a case where the input is ambiguous (e.g., "Apple" without context), the direct LLM embedding produces a generic, averaged vector due to mean pooling over all token representations, failing to disambiguate. Our method projects into orthogonal subspaces, each refined independently; one subspace might emphasize fruit-related features, another company-related, so the refined embedding separates the senses. We expect a noticeable gap on polysemous queries (e.g., 5-10% higher MTEB score on tasks like classification of ambiguous terms) but parity on unambiguous ones.

**vs. GIRCSE** — On a case requiring complex reasoning (e.g., a multi-hop question like "What is the capital of the country where the Eiffel Tower is located?"), GIRCSE refines sequentially, each step building on previous, but cost grows linearly with steps (e.g., 3 steps = 3x cost). Our method refines all K subspaces in parallel, each with a learned MLP, achieving comparable transformation in constant time. We expect similar MTEB scores (within 1-2%) on such reasoning tasks, but with inference latency independent of refinement depth, unlike GIRCSE.

**vs. NV-Embed** — On a case with contradictory attributes (e.g., "small elephant"), NV-Embed's single global refinement may average out the contradiction because it treats all dimensions jointly. Our orthogonal subspaces can separate the "size" dimension from the "species" dimension, refining each independently to preserve the contradiction. We expect our method to outperform on tasks requiring fine-grained attribute discrimination (e.g., emotion classification with nuanced labels) by 3-5% on MTEB, especially on clustering and classification subsets.

**vs. PSR-Embed w/o orthogonality** — Removing the orthogonality regularizer allows subspaces to overlap, leading to redundant refinement. On a simple task (e.g., sentiment analysis), both may work, but on a complex task (e.g., 50-class classification), the overlapping version overfits due to correlated features. We expect the orthogonal version to outperform by 2-4% on high-dimensional tasks (e.g., MTEB's clustering tasks), while being comparable on low-dimensional tasks.

### What would falsify this idea
If PSR-Embed's average MTEB score is no better than Direct LLM embedding, then our refinement degrades embeddings, disproving the benefit of any refinement. If the improvement over GIRCSE is limited to only simple tasks (e.g., retrieval but not classification), then parallel refinement may miss sequential interactions needed for complex reasoning, falsifying the claim of general cost-independent refinement.

## References

1. Let LLMs Speak Embedding Languages: Generative Text Embeddings via Iterative Contrastive Refinement
2. NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models
3. Mistral 7B
4. InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining
5. Towards General Text Embeddings with Multi-stage Contrastive Learning
