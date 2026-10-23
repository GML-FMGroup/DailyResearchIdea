# Token-Ablation Certified Chunk Memory: Enforcing Per-Token Sufficiency in Shared Parametric Memory

## Motivation

Both chunk-level parametric memory (Deep Fusion Memory) and circuit-localized editing (MCircKE, CaKE) assume a coarse unit — a shared chunk vector or a sparse circuit — is a sufficient statistic for fine-grained downstream behavior, yet neither enforces per-token identifiability: the fused memory can be correct at the chunk level while being wrong for individual tokens. This structural gap recurs across paper-trees because the coarse unit is a differentiable object that is never subjected to an interventional, per-token test. Deep Fusion Memory in particular shares one predicted memory vector across all tokens in a chunk, with no mechanism to detect or correct over-sharing when token-level predictions depend on only a subset of memory coordinates.

## Key Insight

A chunk memory is a valid sufficient statistic for a token exactly when every memory coordinate that causally changes that token's prediction under an interventional ablation is retained and every coordinate that does not is discarded — turning sufficiency from an untested assumption into a per-token, enforcement-ready certificate.

## Method

**What it is.** TACCM (Token-Ablation Certified Chunk Memory) is a training and inference procedure for chunk-shared parametric memory that augments the Deep Fusion Memory predictor with a per-token interventional certificate: for each token position $t$ in a chunk, it computes a binary load-bearing mask $m_t \in \{0,1\}^d$ over the $d$ memory coordinates by counterfactual coordinate ablation, and supervises the memory predictor so that the fused memory restricted to $m_t$ is a minimal sufficient statistic for predicting token $t$ given its local context.

**How it works.**

```pseudocode
# Inputs: frozen backbone f, memory predictor g_phi (2-layer MLP, hidden=256, GeLU), fusion cross-attn F_psi
# Chunk c = (x_1..x_L); preceding context h; retrieved chunk embedding e_c (supervision target)
# Hyperparams: ablation threshold tau (default 0.05 nats), sparsity weight lambda (0.01), K=1 ablation pass
#              redundancy probe: pairwise correlation of delta_k across tokens per chunk, reported
#              context-only baseline: p_ctx = f(x_t | h_t) (no memory)

for each training chunk c:
    M = g_phi(h)                         # chunk memory vector, dim d
    for t in 1..L:
        p_full = f(x_t | F_psi(h_t, M))  # full-memory logits
        p_ctx  = f(x_t | h_t)            # context-only logits (no memory)
        for k in 1..d:
            M_abl = M with coord k zeroed  # interventional ablation
            p_abl = f(x_t | F_psi(h_t, M_abl))
            delta_k = KL(p_full || p_abl)  # per-coordinate causal effect
        m_t = 1[delta_k > tau for k in 1..d]   # load-bearing mask
        # certificate: M restricted to m_t must match full-memory prediction
        p_cert = f(x_t | F_psi(h_t, M * m_t))
        L_cert += max(0, CE(p_cert, x_t) - CE(p_ctx, x_t)) + CE(p_full, x_t)
        L_sparse += lambda * ||m_t||_1 / d     # penalize over-sharing
    L_pred = ||M - e_c||^2                     # original chunk supervision
    L = L_pred + L_cert + L_sparse
    update phi, psi (backbone frozen)

# Inference: no retrieval; g_phi predicts M; fusion uses full M (mask computed only for audit)
```

The certificate is computed once per token per chunk (K=1 ablation pass over coordinates, vectorized). The sparsity penalty enforces minimality; the certificate cross-entropy enforces sufficiency. At inference the mask is not applied — it is logged as an identifiability certificate, and a diagnostic reports the fraction of coordinates that are load-bearing per token, flagging over-sharing when this fraction is near 1 for all tokens in a chunk.

**Why this design.** We chose interventional coordinate ablation over observational attribution (e.g., gradient saliency or attention weights) because only an intervention answers the counterfactual question 'would this token's prediction change if this memory coordinate were absent', which is the operational meaning of sufficiency; the cost is $d$ forward passes per token per chunk, mitigated by vectorizing ablations across coordinates and using a single ablation pass (K=1) rather than a full leave-one-out sweep. We chose to supervise the predictor with the certificate loss rather than post-hoc pruning the memory because post-hoc pruning cannot change what the predictor generates for future chunks, accepting the cost that the predictor must learn to concentrate load-bearing information into few coordinates. We chose a hard binary mask with a KL threshold $\tau$ rather than a soft attention gate because a hard mask yields an auditable certificate and a well-defined minimality penalty, accepting the cost that the mask is non-differentiable and must be trained through the certificate loss on the masked memory (straight-through is not needed because the certificate loss acts on $M \cdot m_t$ with $m_t$ detached). We chose to keep the original chunk-embedding supervision $L_{pred}$ alongside the certificate rather than replacing it, because the chunk embedding anchors the memory to retrieved semantics while the certificate enforces per-token sufficiency — this is the specific coupling that Deep Fusion Memory lacks and that a simple retrieval baseline (RAG over chunk embeddings) cannot produce, since retrieval never intervenes on a shared memory vector.

**Why it measures what we claim.** The per-coordinate KL divergence $\delta_k$ measures the causal necessity of memory coordinate $k$ for token $t$ because zeroing coordinate $k$ is an intervention on the fused memory, and the KL between full-memory and ablated-memory predictive distributions is the interventional effect of that coordinate; this assumption fails when coordinates are redundant (two coordinates carry the same information, so ablating either alone yields near-zero KL), in which case $\delta_k$ reflects redundancy rather than non-necessity and the mask under-selects. The certificate cross-entropy $CE(p_{cert}, x_t)$ measures sufficiency of the masked memory because it evaluates the token prediction using only the load-bearing coordinates; this assumption fails when the token's correct prediction is achievable through the local context alone without any memory, in which case the certificate is satisfied trivially and does not certify memory sufficiency — we detect this by requiring the certificate loss to be evaluated against a context-only baseline and flagging tokens whose context-only prediction already matches. The sparsity penalty $\|m_t\|_1/d$ measures minimality because it counts the fraction of coordinates retained; this assumption fails when the threshold $\tau$ is set too low, in which case noise coordinates are retained and the penalty reflects threshold calibration rather than true over-sharing — we set $\tau$ by a held-out calibration split so that the retained fraction is stable across chunks.

## Contribution

(1) A per-token interventional certificate for chunk-shared parametric memory that converts the sufficiency assumption of Deep Fusion Memory into an enforceable training signal via counterfactual coordinate ablation. (2) A design principle — coupling chunk-embedding supervision with a per-token minimality penalty — that yields memory predictors whose fused vectors are token-auditable, with a diagnostic that flags over-sharing at the chunk level. (3) An ablation-based identifiability protocol that distinguishes causal necessity from redundancy and from context-only triviality, applicable to any differentiable coarse memory unit.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-hop QA with held-out chunk retrieval (MQuAKE-style) | Requires composing facts across chunks, exposing memory sufficiency |
| Primary metric | Accuracy on memory-critical subset vs context-only subset | Separates genuine memory use from trivial context prediction |
| Baseline | Deep Fusion Memory (chunk parametric memory) | Tests whether per-token certificate adds over chunk-level supervision |
| Baseline | RAG over chunk embeddings | Tests whether shared-vector intervention beats retrieval |
| Baseline | Post-hoc memory pruning of Deep Fusion | Tests whether certificate-time supervision beats pruning |
| Ablation-of-ours | TACCM without certificate loss (L_pred + sparsity only) | Isolates the certificate as the causal ingredient |
| Ablation-of-ours | TACCM with gradient-saliency mask instead of interventional ablation | Tests whether intervention does work observational attribution cannot |
| Ablation-of-ours | TACCM without context-only subtraction in L_cert | Tests whether context-insufficiency term is necessary |

### Why this setup validates the claim

The central claim is that per-token interventional certificates make a chunk-shared memory vector a minimal sufficient statistic per token, and that this yields better multi-hop composition than chunk-level supervision alone. The dataset must contain questions whose answer requires combining facts stored in different chunks but whose local context is insufficient — otherwise the context-only baseline trivially satisfies the certificate and the claim is untestable. Splitting the primary metric into memory-critical versus context-only subsets is the falsifiable core: the certificate is supposed to help precisely where memory is load-bearing, so any gain must concentrate there. Deep Fusion Memory tests whether token-level certification adds over chunk-level targets; RAG tests whether intervening on a shared vector does something retrieval cannot; post-hoc pruning tests whether certificate-time supervision beats post-hoc surgery. The no-certificate ablation isolates the certificate as the causal ingredient rather than the sparsity penalty or the chunk loss. The gradient-saliency ablation tests whether the interventional ablation is doing work that observational attribution cannot, directly addressing the redundancy concern. The no-context-subtraction ablation tests whether the context-insufficiency term is necessary to prevent trivial certificate satisfaction.

### Expected outcome and causal chain

**vs. Deep Fusion Memory** — On a multi-hop question where the bridge fact and answer fact live in separate chunks, Deep Fusion produces a memory vector dominated by the chunk's average semantics, so the token predicting the bridge entity is driven mostly by local context and the composition fails. Our method instead concentrates the load-bearing coordinates for that token via the certificate, because the masked-memory cross-entropy forces sufficiency for exactly that prediction, so we expect a clear accuracy gap on memory-critical multi-hop items with parity on single-hop items.

**vs. RAG over chunk embeddings** — On a case where the retrieved chunk is semantically adjacent but the needed fact is a small span, RAG injects the whole chunk and the model's attention is diluted across irrelevant spans, producing a plausible but wrong continuation. Our method instead routes the prediction through a small certified coordinate set for that token, because the per-token mask isolates the causally necessary memory, so we expect higher accuracy on span-localized facts and a smaller, roughly uniform load-bearing fraction reported by the diagnostic.

**vs. Post-hoc memory pruning of Deep Fusion** — On a case where a future chunk's prediction depends on coordinates that were never load-bearing during training, post-hoc pruning removes them and the predictor cannot recover, producing confidently wrong predictions on later chunks. Our method instead trains the predictor under the certificate so load-bearing information is concentrated during learning, because the sparsity penalty and certificate loss co-shape g_phi, so we expect pruning to degrade Deep Fusion sharply while our method's accuracy stays close to its unpruned level.

**vs. TACCM with gradient-saliency mask** — On a token where two memory coordinates redundantly encode the same information, gradient saliency assigns high importance to both (since both have nonzero gradient), while single-coordinate ablation yields near-zero KL for each, so the saliency mask over-selects and the ablation mask under-selects. Our method uses the ablation mask, so we expect the saliency variant to produce larger masks (higher load-bearing fraction) and lower accuracy on memory-critical items, confirming that the interventional test is not equivalent to observational attribution.

**vs. TACCM without context-only subtraction** — On a token whose correct prediction is achievable from local context alone, the certificate loss without subtraction is trivially satisfied and provides no gradient signal to concentrate memory. Our method subtracts the context-only cross-entropy, so we expect the no-subtraction variant to have a higher load-bearing fraction (over-sharing) and worse accuracy on memory-critical items, confirming that the context-insufficiency term is necessary.

### What would falsify this idea

If the accuracy gain over Deep Fusion Memory is uniform across memory-critical and context-only subsets rather than concentrated on memory-critical items, or if the certificate loss ablation matches the full method, then the certificate is not measuring sufficiency and the central claim fails. A second falsifier: if the load-bearing fraction is near 1 for all tokens across chunks (the over-sharing flag fires uniformly), the mask is not identifying minimal sufficient statistics and the method reduces to unconstrained chunk memory.

## References

1. Deep Fusion Memory: A Chunk-Level Parametric Memory for Large Language Models
2. Addressing the Reasoning Gap: Mechanistic Circuit-Based Knowledge Editing in Large Language Models
3. CaKE: Circuit-aware Editing Enables Generalizable Knowledge Learners
4. AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models
5. Uncovering Overfitting in Large Language Model Editing
6. PMET: Precise Model Editing in a Transformer
