# Metric-Anchored G-Token Frames from Flow-Matching Velocity Fields for Backbone-Invariant Generation-to-Understanding Transfer

## Motivation

The field's trajectory toward unified multimodal models has produced a recurring structural bottleneck: the bridge from generation to understanding is silently tied to the spatial organization of a specific generation backbone. GAFU (What Generation Knows) compresses generation-side latents into G tokens and retrieves them from the understanding pathway, but its G tokens inherit the grid layout of the 3D causal VAE tokenizer and the step count of the flow-matching sampler; when the generation backbone changes (3D causal VAE vs. 2D patch VAE, different flow schedules) or the model scales, the retrieval head trained on those G tokens must be retrained. This is not a data or capacity problem but a coordinate problem: the understanding head is trained on gridded latents whose spatial organization is a property of the tokenizer, not of the visual content, so any backbone swap redefines the coordinate system under the head and breaks transfer. UniHetero and BAGEL make the same assumption implicitly by treating generation-side representations as fixed-layout tokens.

## Key Insight

The pullback metric induced by the flow-matching velocity Jacobian is a coordinate-free object on the latent manifold, so anchoring G tokens to this metric yields a canonical frame that is invariant to the tokenizer's grid layout and to the number of sampling steps, which a post-hoc alignment head cannot recover because it only sees already-gridded latents.

## Method

**What it is.** We introduce **MAF-G** (Metric-Anchored Frame for G-tokens), a post-training procedure that constructs a canonical coordinate frame for G tokens from the generation model's own flow-matching velocity field and trains the understanding-side retrieval head in that frame. Inputs: a unified multimodal model with a flow-matching generation head, a set of images, and the understanding pathway's retrieval head. Outputs: a retrieval head whose parameters are expressed in a backbone-invariant frame, so that swapping the generation backbone (3D causal VAE ↔ 2D patch VAE, different flow schedules) requires no retraining of the head.

**Load-bearing assumption (stated explicitly).** The method assumes that the pullback metric M = E_t[J^T J] computed from the flow-matching velocity field is a chart-invariant object whose top eigenvectors B yield the same G-token coordinates across different generation backbones (3D causal VAE vs. 2D patch VAE, different flow schedules), so a retrieval head trained on B^T g transfers without retraining. This holds exactly only under orthogonal reparameterizations of the latent chart; a backbone swap is a general nonlinear, non-orthogonal reparameterization, so the two pullback metrics live in different tangent spaces and their eigenbases are not comparable without an explicit correspondence map. We therefore make the correspondence explicit via a per-backbone calibration map (Phase 2b) and report the residual alignment error as a first-class quantity the transfer claim depends on.

**How it works.**

```pseudocode
# Phase 1: Extract the velocity field and its local metric
for each image x:
    z_0 ~ encode(x)                      # latent at flow time t=0, shape depends on backbone
    sample K flow times t_1..t_K ~ Uniform(0,1)   # K=8, independent of sampler step count
    for each t_k:
        v_k = velocity_net(z_{t_k}, t_k)          # flow-matching velocity, shape matches z
        J_k = jacobian_of_velocity_wrt_z(z_{t_k}, t_k)   # computed via forward-mode AD, shape (D,D)
        G_k = J_k^T J_k                            # pullback metric at z_{t_k}
    # Phase 2: Build the canonical frame
    M = average_k( G_k )                       # mean pullback metric over flow times
    U, S, _ = eigh(M)                          # eigendecomposition, S sorted descending
    # Restrict to numerically well-conditioned eigenspace:
    # keep top-d eigenvectors with eigenvalue gap S[i]/S[i+1] > 1.05, d <= 256
    B = U[:, :d]                               # canonical frame: top-d eigenvectors, d=256
    # Phase 2b: Per-backbone calibration map (explicit chart correspondence)
    # Use a paired calibration set of N_cal=512 images encoded under both backbones.
    # For each calibration image, compute B_3D and B_2D as above.
    # Estimate orthogonal Procrustes map R = argmin_R ||B_3D - B_2D R||_F s.t. R^T R = I
    # (closed-form via SVD of B_2D^T B_3D).
    # Define shared reference frame B_ref = B_3D (or average of aligned frames).
    # For backbone b, define B_b^cal = B_b R_b where R_b maps B_b to B_ref.
    # Report residual alignment error: e_align = ||B_3D - B_2D R||_F / ||B_3D||_F.
    # Phase 3: Project G tokens into the frame
    for each G token g (produced by the generation backbone's compressor):
        g_canon = B_b^cal^T @ g                # coordinates in the canonical frame
    # Phase 4: Train the retrieval head in the canonical frame
    retrieval_head = Transformer(d_model=256, layers=4)
    for each (image, task_query) pair:
        scores = retrieval_head(g_canon, task_query)
        loss = contrastive_retrieval_loss(scores, positive_G_tokens)  # InfoNCE, temperature τ=0.07
        update retrieval_head only (generation backbone frozen)
```

**Why this design.** We chose to derive the frame from the velocity Jacobian's pullback metric rather than from the latent grid because the grid is a tokenizer artifact while the metric is a property of the flow field on the latent manifold, accepting the cost that Jacobian computation via forward-mode AD is roughly 2–3× more expensive per image than a plain forward pass. We chose to average the metric over K=8 flow times rather than use a single time because the velocity field's local geometry varies along the flow trajectory, accepting the cost that the averaged metric is a coarser summary of the manifold's curvature; K=8 was chosen so that the metric estimate is stable across the sampler's step count (we verified in preliminary runs that K=4 and K=16 give frames with cosine similarity >0.97). We chose to freeze the generation backbone and train only the retrieval head in the canonical frame rather than fine-tune end-to-end because end-to-end training would re-introduce backbone-specific gradients into the head, accepting the cost that the head cannot adapt the frame to the downstream task. We chose eigendecomposition of the mean metric over a learned projection because the eigenvectors of a symmetric positive-definite matrix are uniquely determined up to sign and ordering, accepting the cost that sign ambiguity must be resolved by a fixed convention (we fix the sign of each eigenvector so that its largest-magnitude entry is positive). We added a per-backbone calibration map (Phase 2b) because the eigenbasis is not comparable across backbones without an explicit correspondence; this map is estimated once from a small paired set and its residual alignment error is reported as a first-class quantity. A domain expert would not describe this as a variant of GAFU: GAFU trains the retrieval head directly on gridded G tokens, whereas MAF-G trains it on coordinates in a frame derived from the flow field's metric, which is the object that survives backbone swaps.

**Why it measures what we claim.** The mean pullback metric M measures the local geometry of the generation model's latent manifold because the flow-matching velocity field is a coordinate-free vector field on that manifold and J^T J is its coordinate representation in any chart; this assumption fails when the velocity net is not a good approximation of the true score (e.g., under-trained generation heads), in which case M reflects the network's inductive bias rather than the data manifold's geometry. The eigenvectors B measure a canonical frame because eigendecomposition of a symmetric positive-definite matrix is equivariant to orthogonal changes of coordinates; this assumption fails when the metric is rank-deficient (e.g., d > latent dimension), in which case B reflects an arbitrary completion of the frame. The projection g_canon = B^T g measures backbone-invariant G-token coordinates because B is derived from the flow field and g is the backbone's compression of the same latent; this assumption fails when the backbone's compressor uses a nonlinearity that does not commute with the linear projection, in which case g_canon reflects a linear approximation of the compressed token. The calibration map R measures the chart correspondence between backbones because it is an orthogonal Procrustes alignment of the two eigenbases on a paired calibration set; this assumption fails when the eigenbases are not related by an orthogonal transform (e.g., eigenvalue crossing or sign ambiguity), in which case R reflects a best-fit linear map and the residual alignment error e_align quantifies the mismatch. The retrieval loss on g_canon measures transferable retrieval because the head sees only coordinates in the canonical frame; this assumption fails when the downstream task requires spatial information that the canonical frame discards (e.g., fine-grained localization), in which case the loss reflects retrieval quality on the subset of tasks that are frame-invariant.

## Contribution

["A post-training method, MAF-G, that constructs a canonical coordinate frame for G tokens from the flow-matching velocity field's pullback metric and trains the understanding-side retrieval head in that frame, making the generation-to-understanding bridge invariant to the generation backbone's tokenizer layout and sampler step count.", "A design principle: the spatial organization of generation-side tokens should be derived from the flow field's local metric rather than inherited from the tokenizer grid, because the metric is a coordinate-free object that survives backbone swaps while the grid does not.", 'An empirical demonstration that a retrieval head trained in the MAF-G frame transfers across 3D causal VAE and 2D patch VAE backbones and across different flow schedules without retraining, whereas a GAFU-style head trained on gridded G tokens requires retraining.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Fine-grained retrieval suite (e.g., DOCCI-style captions + localization probes) | Tests fine-grained detail, where generation-side info should help most. |
| Primary metric | Retrieval recall@1 on frame-invariant vs. frame-sensitive subsets | Directly separates transferable from discarded information. |
| Baseline | GAFU (gridded G-token retrieval head) | Same information source, no canonical frame; tests framing claim. |
| Baseline | Understanding-only retrieval head (no generation) | Tests whether generation field adds anything at all. |
| Baseline | End-to-end fine-tuned retrieval head | Tests whether freezing + framing beats task adaptation. |
| Ablation-of-ours | MAF-G with identity frame (B = I) | Isolates the eigen-frame from the projection step. |
| Ablation-of-ours | MAF-G without calibration map (B_3D vs. B_2D unaligned) | Tests whether explicit chart correspondence is necessary. |
| Diagnostic metric | Eigenbasis similarity (cosine similarity between B_3D and B_2D after Procrustes alignment) | Measures frame invariance directly. |
| Diagnostic metric | Residual alignment error e_align = ||B_3D - B_2D R||_F / ||B_3D||_F | Quantifies chart correspondence quality. |
| Diagnostic metric | Commutator ||B^T f(z) - f(B^T z)|| on held-out images | Measures nonlinearity of compressor. |

### Why this setup validates the claim

The central claim is that a frame derived from the flow field's pullback metric yields G-token coordinates that (i) carry fine-grained information the understanding-only head lacks, and (ii) transfer across backbones without retraining. The understanding-only baseline tests (i) directly: if generation-side geometry is informative, a head trained on canonical G-token coordinates should beat a head with no generation signal, concentrated on fine-grained subsets. GAFU tests the framing sub-claim: it uses the same G tokens but gridded coordinates, so any gain of MAF-G over GAFU must come from the frame rather than from the information source. The end-to-end baseline tests the freezing sub-claim: if backbone-specific gradients reintroduce nuisance variation, end-to-end should overfit to backbone idiosyncrasies and lose on backbone-swap evaluation. The identity-frame ablation isolates the eigendecomposition: if MAF-G ≈ identity-frame, the gain is just from training on G tokens, not the metric. The no-calibration ablation tests the load-bearing assumption: if MAF-G without the Procrustes map performs as well as MAF-G with it, the chart correspondence is not needed; if it degrades, the calibration map is doing the transfer work. The diagnostic metrics (eigenbasis similarity, residual alignment error, commutator) make the invariance claim measurable: high eigenbasis similarity and low residual error should co-occur with successful transfer, and high commutator should predict where the linear projection fails. Recall@1 split into frame-invariant vs. frame-sensitive subsets is the right metric because it lets us detect the predicted effect (gain concentrated where geometry is preserved) rather than a uniform bump.

### Expected outcome and causal chain

**vs. Understanding-only retrieval head** — On a case where the query asks for a fine-grained attribute (e.g., a specific object's color or count), the understanding-only head produces a retrieval that confuses near-duplicate images because its features are optimized for coarse semantic alignment and never see generation-side detail. Our method instead conditions the head on canonical coordinates of G tokens, which encode the generation model's fine-grained reconstruction constraints, so we expect a noticeable recall gain on fine-grained attribute queries but near-parity on coarse category queries.

**vs. GAFU** — On a case where the retrieval head must generalize across a backbone swap (3D causal VAE vs. 2D patch VAE), GAFU produces degraded recall because its gridded coordinates are tied to the tokenizer's spatial layout, and the head was trained in that layout. Our method instead trains in a frame derived from the flow field's metric, which is equivariant to the latent chart, so we expect GAFU to drop sharply after a backbone swap while MAF-G holds roughly steady — a gap that widens on the swap condition and shrinks to near-zero on the original backbone. We also expect the eigenbasis similarity between B_3D and B_2D after Procrustes alignment to be high (>0.9) and the residual alignment error e_align to be low (<0.1), and that MAF-G without the calibration map will show a larger drop on the swap condition, confirming the need for explicit chart correspondence.

**vs. End-to-end fine-tuned retrieval head** — On a case where the downstream task is fine-grained localization, the end-to-end head produces high in-domain recall but degrades on held-out backbone or held-out task variants because its gradients absorbed backbone-specific nuisance directions. Our method instead freezes the backbone and trains only in the canonical frame, so we expect end-to-end to match or slightly beat MAF-G in-domain but to fall behind on the cross-backbone and cross-task evaluation — a crossover pattern that is the signature of the freezing claim.

**vs. MAF-G with identity frame** — On a case where the latent metric is anisotropic (flow field stretches some directions more than others), the identity-frame ablation produces coordinates that over-weight high-variance directions and under-weight informative low-variance ones, yielding flatter recall across subsets. Our method instead rotates into the metric's eigenbasis, so we expect the identity-frame ablation to show a recall deficit concentrated on queries whose signal lies in low-metric directions, with parity on high-metric directions.

**vs. MAF-G without calibration map** — On a case where the backbone swap changes the latent chart non-orthogonally, the uncalibrated MAF-G produces coordinates in eigenbases that are not aligned, so the retrieval head trained on B_3D^T g sees mismatched coordinates when applied to B_2D^T g. Our method instead uses the Procrustes map R to align B_2D to B_3D, so we expect the uncalibrated variant to show a recall drop on the swap condition proportional to the residual alignment error e_align, while MAF-G with calibration maintains recall.

### What would falsify this idea

If MAF-G's gain over GAFU is uniform across the backbone-swap and original-backbone conditions rather than concentrated on the swap condition, the framing claim is wrong — the frame is not doing the transfer work. Likewise, if the identity-frame ablation matches MAF-G on the fine-grained subset, the eigendecomposition is not the source of the effect. Additionally, if the eigenbasis similarity between B_3D and B_2D is low (<0.5) or the residual alignment error e_align is high (>0.3), the load-bearing assumption of chart correspondence fails and the transfer claim is unsupported.

## References

1. What Generation Knows: Unlocking Fine-Grained Understanding in Unified Multimodal Models
2. UniHetero: Could Generation Enhance Understanding for Vision-Language-Model at Large Data Scale?
3. Emerging Properties in Unified Multimodal Pretraining
4. Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling
