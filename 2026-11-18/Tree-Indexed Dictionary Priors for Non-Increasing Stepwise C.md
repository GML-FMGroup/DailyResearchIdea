# Tree-Indexed Dictionary Priors for Non-Increasing Stepwise Code Lengths in Parallel rANS Image Compression

## Motivation

FMT establishes bit-exact parallel rANS decoding in a fixed number of sequential steps by storing an image as a coarse-to-fine tree, but it never shows that the inherited dictionary-style prior from learned entropy models (e.g., the Dictionary-based Cross Attention Entropy model of arxiv:2504.00496) remains a valid, non-increasing code-length assignment when the tree step granularity is coarsened or refined. The root cause is structural: the dictionary prior is queried over the full latent grid via cross-attention, so at any tree step a dictionary entry can be attended by a node and its descendants simultaneously, breaking the ancestor-only conditioning that a valid stepwise probability chain requires. This leaves FMT's parallelism as a single-GPU empirical property rather than a structural guarantee on resolution- and hardware-invariant decode-step bounds.

## Key Insight

Constraining the learnable dictionary to be indexed by coarse-to-fine tree depth, so that each cross-attention query at step t can only attend to dictionary entries whose depth index is < t, makes the per-step conditional a proper refinement of its parent's conditional, which forces the realized code length at every node to be non-increasing under tree refinement without any retraining of the base entropy model.

## Method

**What it is.** We introduce TD-FMT (Tree-Depth-indexed Dictionary FMT), a lossless image codec that takes a coarse-to-fine multiscale tree of an image and a learnable dictionary D = {d_k}_{k=0}^{K-1} partitioned by depth index k, and outputs per-step rANS bit counts that are non-increasing when the tree is refined. Inputs: image x, tree T with depth D (D_max = 6), dictionary size K = 64, cross-attention heads H = 4. Outputs: bit-exact rANS stream and a per-step realized bit count vector b_0..b_{D-1}.

**Load-bearing assumption (stated explicitly).** The method assumes that restricting cross-attention queries at tree step t to dictionary entries with depth_index < t makes each step's conditional a true refinement of its parent's conditional, so that realized rANS code lengths b_t are non-increasing under tree refinement without retraining. This assumption holds only if (i) the entropy head outputs normalized probabilities consumed verbatim by the rANS coder (no renormalization), (ii) the tree is a true refinement (each node's value distribution is a marginal of its parent's), and (iii) the depth_index assignments are fixed and do not drift during training. We enforce (i) by applying a softmax over the entropy head logits and feeding the resulting probabilities directly to the rANS coder with no additional renormalization; we enforce (ii) by constructing the tree via a fixed coarse-to-fine partition (e.g., quadtree) so that each parent's value set is exactly the union of its children's value sets; we enforce (iii) by freezing depth_index(d_k) = k // (K / D_max) as a fixed function of tree depth, never updated by gradients. Under these three conditions, the ancestor mask ensures that the conditional at step t is a proper restriction of the conditional at step t-1, and the realized code length b_t = -log2 p_t(values_t) is non-increasing by construction of the probability chain. If any condition fails, the monotonicity claim is not guaranteed and the realized b_t may increase; we report the fraction of nodes where b_t increases as a diagnostic.

**How it works.**

```pseudocode
# Phase 1: build depth-indexed dictionary (frozen assignment)
D = {d_k : k = 0..K-1}                         # learnable, K = 64 entries
assign depth_index(d_k) = k // (K / D_max)      # D_max = 6, fixed, not learned

# Phase 2: per-step ancestor-restricted cross-attention
for t in 0..D-1:                                # t = tree step (coarse to fine)
    Q_t = W_q * h_t                             # query from step-t latent h_t, W_q: 128x128
    A_t = {d_k : depth_index(d_k) < t}          # ancestor-only dictionary subset
    attn_t = softmax(Q_t * K_A^T / sqrt(128)) * V_A  # K_A, V_A from A_t only, 4 heads
    logits_t = entropy_head(h_t, attn_t)        # 2-layer MLP, hidden=256, GeLU
    p_t = softmax(logits_t)                     # normalized probabilities, no renormalization
    # Phase 3: rANS encode step t with state carried from t-1
    rANS_state_t = rANS_encode(p_t, values_t, rANS_state_{t-1})
    b_t = bit_length(rANS_state_t) - bit_length(rANS_state_{t-1})

# Phase 4: refinement check (diagnostic only, not a training loss)
# We do NOT train with a monotonicity penalty; monotonicity is enforced by the
# normalized-probability and fixed-tree construction above.
# We report the fraction of nodes where b_t > b_{t-1} on a held-out set.
```

**Why this design.** We chose depth-indexed dictionary partitioning over a single shared dictionary because a shared dictionary lets a step-t query attend to entries that a step-(t+1) query also attends to, which couples sibling and descendant conditionals and destroys the refinement property; the cost is that early steps see a smaller dictionary and must rely on the no-network rule for the coarsest level. We chose ancestor-only masking (depth_index(d_k) < t) over full-grid cross-attention because the latter is what arxiv:2504.00496 uses and it violates the probability-chain requirement; the cost is reduced dictionary expressivity at deep steps, which we compensate by increasing K to 64. We chose a fixed depth_index assignment over a learned one because a learned index can drift to a coarser bin during training, which would break the ancestor mask; the cost is that the partition is not adapted to data, which we accept to preserve the structural guarantee. We chose normalized softmax probabilities consumed verbatim by the rANS coder over unnormalized logits because only normalized probabilities make b_t equal -log2 p_t exactly; the cost is that the entropy head cannot use temperature scaling or other logit transformations that would require renormalization. A domain expert would not describe this as a variant of the Dictionary-based Cross Attention Entropy model because that model queries the full latent grid and has no notion of tree depth or ancestor restriction, whereas TD-FMT's core operation is the depth-indexed ancestor mask that makes the stepwise factorization a valid chain.

**Why it measures what we claim.** The realized bit count b_t measures the per-step code length under the ancestor-restricted conditional p_t because the rANS state length difference equals -log2 p_t(values_t) exactly when the rANS implementation is bit-exact and the conditional is normalized; this assumption fails when the entropy head outputs unnormalized logits that are renormalized by the coder, in which case b_t reflects the renormalized distribution rather than the learned p_t. We enforce normalization by applying softmax and feeding probabilities directly to rANS, so this failure mode is excluded by construction. The monotonicity of b_t measures refinement validity because a valid probability chain requires H(X_t | X_{<t}) to be non-increasing in t under the tree's refinement order; this assumption fails when the tree is not a true refinement (a node's value distribution is not a marginal of its parent's), in which case the penalty reflects tree-construction error rather than conditional validity. We enforce true refinement by using a fixed quadtree partition where each parent's value set is the union of its children's value sets. The depth-index mask depth_index(d_k) < t measures ancestor-only conditioning because dictionary entries are partitioned so that no entry at depth >= t can be attended at step t; this assumption fails when a dictionary entry's learned depth_index drifts during training to a coarser bin, in which case the mask reflects the drifted index rather than the intended ancestor set. We freeze depth_index as a fixed function of k, so drift is excluded by construction.

## Contribution

['A tree-depth-indexed dictionary prior with ancestor-restricted cross-attention that makes the per-step conditional of a coarse-to-fine parallel rANS codec a valid refinement of its parent, together with a realized-bit-count monotonicity penalty that operates on actual rANS state lengths rather than surrogate cross-entropy.', 'An empirical characterization of how realized rANS bit counts track the surrogate conditional across tree depths and dictionary sizes, identifying the depth at which ancestor restriction becomes binding and quantifying the bit-count gap between surrogate and realized lengths.', "A bit-exact codec family (TD-FMT) spanning no-network to large-network variants that preserves FMT's fixed-sequential-step decode while adding depth-indexed dictionary conditioning."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Kodak (24 images, 768×512) + high-resolution CLIC-style images (2K–4K) | Standard lossless benchmarks with varied tree depths |
| Primary metric | Realized bits per pixel (bpp) | Directly measures emitted rANS code length |
| Secondary metric | Fraction of nodes where b_t > b_{t-1} (monotonicity violation rate) | Directly tests the refinement claim |
| Secondary metric | Decode-step count (sequential steps) | Tests fixed-step parallelism claim |
| Baseline | FMT (no-network and large-network variants) | Same tree/rANS substrate, no depth-indexed dictionary |
| Baseline | Dictionary-based Cross Attention Entropy model (full-grid attention) | Tests whether ancestor masking beats full-grid attention |
| Baseline | P-LLM / classical HTJ2K | Tests against strong lossless codecs on size |
| Ablation-of-ours | TD-FMT with shared (non-depth-indexed) dictionary | Isolates the depth-partition effect |
| Ablation-of-ours | TD-FMT with learned (non-frozen) depth_index | Tests whether drift breaks monotonicity |

### Why this setup validates the claim

The central claim is that depth-indexed ancestor-restricted dictionary attention, combined with normalized probabilities consumed verbatim by the rANS coder and a fixed true-refinement tree, yields a valid coarse-to-fine probability chain whose per-step code lengths do not grow as the tree refines, and that this beats both the FMT substrate without our dictionary and the full-grid dictionary entropy model. FMT is the right baseline because it shares our tree and rANS machinery, so any gap isolates the dictionary-and-mask contribution rather than confounding it with codec architecture. The Dictionary-based Cross Attention Entropy model tests the opposite design choice — full-grid attention — so a win there directly supports the ancestor-mask sub-claim. The shared-dictionary ablation is essential: if TD-FMT matches it, the depth partitioning is doing nothing and the refinement argument collapses. The learned-depth-index ablation tests whether freezing the index is necessary for monotonicity. Realized bpp is the correct primary metric because our mechanism claims to control the actual emitted rANS code length, not a surrogate cross-entropy; a surrogate metric could show improvement while the codec emits more bits, which is exactly the failure our normalization and fixed-tree construction are designed to prevent. The monotonicity violation rate directly measures the refinement claim: under our construction it should be zero (or near-zero due to numerical precision), while the full-grid baseline should show non-zero violations.

### Expected outcome and causal chain

**vs. FMT (same tree/rANS, no depth-indexed dictionary)** — On a high-resolution image where the coarse tree levels carry most structure, plain FMT's per-step conditionals are learned without any shared prior across images, so early steps emit near-uniform codes and only deep steps compress well, producing a bpp curve that drops steeply rather than flatly. Our method lets early steps attend to coarse dictionary entries summarizing training-set structure, so coarse steps already emit short codes, and the ancestor mask keeps deeper steps consistent with them. We expect a noticeable bpp gap concentrated at shallow depths and near-parity at the deepest levels. The monotonicity violation rate should be zero for both methods (since both use normalized probabilities and a true-refinement tree), so this baseline tests the dictionary's compression benefit, not the monotonicity claim.

**vs. Dictionary-based Cross Attention Entropy model (full-grid attention)** — On an image where sibling subtrees have correlated structure, full-grid attention lets a step-t query attend to entries also used by step-(t+1), coupling the two conditionals so the realized b_t can increase at refinement, breaking the probability chain. Our ancestor-only mask forbids this by construction, so b_t is non-increasing. We expect our realized-bit vector to be monotone (violation rate 0) while the baseline's shows occasional increases (violation rate > 0), with the bpp gap largest on images with strong sibling correlation.

**vs. P-LLM / HTJ2K (strong lossless codecs)** — On a large image where decode latency matters, P-LLM and network-heavy codecs spend seconds decoding because their conditionals are sequential and dense. Our fixed-step tree with carried rANS state decodes in a depth-bounded number of steps (D_max = 6 sequential steps regardless of image size), so we expect competitive or better bpp at a fraction of the sequential depth, with the gap in decode steps widening as image size grows while bpp stays close.

**vs. TD-FMT with shared (non-depth-indexed) dictionary** — If the shared-dictionary ablation matches TD-FMT's bpp and monotonicity profile, the depth-indexing claim is wrong. We expect TD-FMT to show lower bpp at shallow depths and zero monotonicity violations, while the shared-dictionary variant may show violations because a step-t query can attend to entries also used by step-(t+1).

**vs. TD-FMT with learned (non-frozen) depth_index** — If the learned-depth-index variant shows non-zero monotonicity violations, this confirms that freezing the index is necessary for the structural guarantee. We expect the learned variant to drift and produce occasional b_t increases, while the frozen variant maintains zero violations.

### What would falsify this idea

If the shared-dictionary ablation matches TD-FMT's bpp and monotonicity profile, the depth-indexing claim is wrong. If our realized b_t still increases at refinement steps (violation rate > 0) despite normalized probabilities and a fixed true-refinement tree, or if the bpp gain over FMT is uniform across depths rather than concentrated at shallow steps, the ancestor-mask mechanism is not producing the predicted effect.

## References

1. FMT: Lossless Image Compression in a Fixed Number of Parallel Steps
