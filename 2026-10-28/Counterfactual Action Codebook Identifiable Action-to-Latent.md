# Counterfactual Action Codebook: Identifiable Action-to-Latent Grounding for Caption-Free Video Generation

## Motivation

Across the interactive video generation literature, action semantics are injected through a domain-specific symbolic channel: Yan's foundational framework converts game controls into hierarchical autoregressive captions before conditioning the video diffusion model, and StreamEval's evaluation protocol assumes actions arrive as scripted multimodal request templates. This mediation prevents any agent-independent, caption-free grounding of action to the generative latent space, because the caption channel is both domain-specific and lossy — the mapping from action to latent is never identified, only the mapping from action to text to latent. The structural problem recurs at the forest level: no branch establishes an action-to-latent map whose entries are identifiable from the generative process itself rather than from text annotations.

## Key Insight

A shared action-latent codebook is identifiable up to permutation when its entries are defined as the minimal latent deltas whose counterfactual intervention on a frozen video diffusion model produces a consistent, action-specific change in the generated trajectory — a condition that text-mediated conditioning cannot satisfy because captions are not interventions on the latent space.

## Method

```pseudocode
# CACHE: Counterfactual Action Codebook Estimation
# Inputs: frozen video diffusion model G (latent-space), unlabeled video clips V,
#         action labels A (discrete, no text), encoder E (3D-VAE), decoder D
# Output: codebook C = {c_1..c_K} in latent space, action->entry assignment pi

Phase 1: Latent trajectory extraction (no captions)
  for each clip v in V:
    z_0 = E(v[0])                      # initial latent
    for t in 1..T:
      z_t = E(v[t])                    # encode each frame independently
    store trajectory Z = (z_0, ..., z_T) and action label a(v)

Phase 2: Counterfactual latent intervention
  for each (Z, a) pair:
    # find the minimal latent delta that reproduces the observed transition
    delta_obs = mean_{t} (z_{t+1} - G(z_t, 0))   # residual under null action
    # counterfactual: replace delta with each candidate codebook entry and measure
    # the induced change in the *next* latent under G
    for each candidate c in C (init: random unit vectors):
      z'_t+1 = G(z_t, c)
      score(c | Z, a) = -|| z'_t+1 - z_{t+1} ||^2   # reconstruction under intervention

Phase 3: Codebook assignment and update (EM-like, K=64 entries, 200 steps)
  repeat:
    # E-step: assign each observed action to the entry with best counterfactual score
    pi(a) = argmax_c  E_{Z~data(a)} [ score(c | Z, a) ]
    # M-step: update each entry as the average minimal delta for its assigned actions
    for k in 1..K:
      c_k = mean_{Z: pi(a(Z))=k} ( mean_t (z_{t+1} - G(z_t, 0)) )
    # identifiability regularizer: entries must be linearly independent
    C = C @ inv(C^T C + 0.01 I)   # orthogonalize (Gram-Schmidt)
  until assignments stable or 200 steps

Phase 4: Frozen codebook at inference
  Given new initial latent z_0 and action a:
    c = C[pi(a)]
    for t in 1..T:
      z_{t+1} = G(z_t, c)          # direct latent conditioning, no text
    return D(z_0..z_T)
```

**Why this design.** We chose to define codebook entries as *counterfactual latent deltas* rather than as learned embeddings or text-derived vectors because the latter two are not interventions on the generative process and therefore cannot be identified from data; the cost is that the codebook must be estimated per frozen model and does not transfer zero-shot across architectures. We chose an EM-style alternating assignment/update over end-to-end gradient descent because the assignment step is a discrete argmax that gradient descent would relax into a soft mixture, destroying the one-entry-per-action structure that makes the codebook identifiable; the cost is slower convergence (200 steps vs. 50 gradient steps) and sensitivity to initialization, which we mitigate with random unit-vector init and orthogonalization. We chose a linear-independence regularizer on the codebook over a sparsity prior because linear independence is the structural condition that guarantees distinct actions map to distinct latent directions — a sparsity prior would not prevent two actions from collapsing to the same direction; the cost is that K is bounded by the latent dimensionality (64 entries for a 512-dim latent). We chose to freeze the diffusion model and only learn the codebook over a learned caption-to-latent adapter (as in Yan's hierarchical autoregressive captioning) because the latter reintroduces a symbolic channel and therefore cannot establish caption-free grounding; the cost is that the codebook cannot correct errors in the frozen model's dynamics, only in the action conditioning. A domain expert would not describe this as a variant of retrieval-based action selection (e.g., Toolformer-style nearest-neighbor over a library) because the codebook entries are *estimated by intervening on the generative model* — the assignment is defined by the counterfactual reconstruction score, not by descriptor similarity, and the orthogonalization step enforces a structural identifiability condition that retrieval cannot provide.

**Why this measures what we claim.** The counterfactual reconstruction score `score(c | Z, a) = -||G(z_t, c) - z_{t+1}||^2` measures *causal necessity of the codebook entry for the observed action* because it compares the generated next latent under an intervention (replacing the action with c) against the factual next latent; this assumption — that the frozen model G is a faithful simulator of the transition dynamics — fails when G has systematic bias (e.g., blurry predictions for fast motion), in which case the score reflects G's reconstruction error rather than action-specific causal effect. The assignment `pi(a) = argmax_c E[score]` measures *action-to-entry grounding* because it selects the entry whose intervention best reproduces the action's effect across all clips; this assumption — that each action has a consistent latent effect across contexts — fails when the same action produces different latent deltas in different scenes (e.g., 'jump' in water vs. on land), in which case the assignment reflects the dominant context rather than the action semantics. The orthogonalization `C = C @ inv(C^T C + 0.01 I)` measures *identifiability of the codebook* because linear independence ensures distinct actions map to distinct latent directions; this assumption — that the latent space is approximately linear in the action effect — fails when the model's dynamics are strongly nonlinear, in which case orthogonal entries may still produce overlapping effects and the codebook overestimates identifiability.

**(D) Assumptions and failure modes.** The method relies on three assumptions that we state explicitly and test empirically. (1) *Frozen-simulator faithfulness*: the counterfactual reconstruction score measures causal necessity of an entry only if G is an unbiased simulator of transition dynamics; this fails when G has systematic bias (e.g., blur on fast motion), in which case the score reflects G's reconstruction error rather than action-specific effect. (2) *Context-invariant action effect*: the argmax assignment measures action-to-entry grounding only if each action has a consistent latent effect across contexts; this fails for context-dependent actions ('jump' in water vs. land), in which case the assignment reflects the dominant context. (3) *Approximate linearity of action effects*: orthogonalization measures identifiability only if latent action effects are approximately linear; this fails under strongly nonlinear dynamics, in which case orthogonal entries may produce overlapping effects and the codebook overestimates identifiability. We do not assume these hold; we report diagnostics for each (see experiment).

## Contribution

(1) A counterfactual action codebook estimation procedure that learns a shared action-to-latent map by intervening on a frozen video diffusion model, requiring no text captions or scripted request templates. (2) A design principle: codebook entries defined as counterfactual latent deltas with an orthogonalization regularizer yield identifiable action-to-latent grounding, whereas text-mediated conditioning and retrieval-based action selection do not. (3) A caption-free inference interface for interactive video generation that maps discrete actions directly to latent-space conditioning vectors.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Action-labeled latent video clips (no captions) | Tests caption-free grounding directly |
| Primary metric | Counterfactual action-reconstruction accuracy | Directly measures claimed causal necessity |
| Secondary metric | Held-out action-classification-from-generated-trajectory accuracy | Independent check not optimized by method |
| Baseline | Text-conditioned diffusion (caption channel) | Tests whether symbolic channel is needed |
| Baseline | Retrieval nearest-neighbor action selection | Tests intervention vs. descriptor similarity |
| Baseline | End-to-end soft-mixture codebook | Tests discrete assignment vs. gradient relaxation |
| Baseline | Learned-embedding codebook with same loss | Tests intervention-defined entries vs. reparameterization |
| Ablation-of-ours | CACHE without orthogonalization | Tests identifiability regularizer's role |
| Ablation-of-ours | CACHE with k-means-on-deltas initialization | Tests sensitivity to random init |
| Diagnostic | Score vs. ground-truth action-effect magnitude correlation | Tests frozen-simulator faithfulness assumption |
| Diagnostic | Permutation-aligned cosine similarity across model seeds | Tests model-specificity of grounding |

### Why this setup validates the claim

The central claim is that codebook entries defined as counterfactual latent deltas, estimated by intervening on a frozen generator, yield identifiable action-to-latent grounding that text-conditioned and retrieval baselines cannot achieve. The text-conditioned baseline tests the sub-claim that a symbolic channel is unnecessary and harmful for grounding: if caption conditioning matches CACHE, the caption-free motivation collapses. The retrieval baseline tests the sub-claim that intervention-defined entries beat descriptor-similarity entries: if nearest-neighbor over a latent library matches CACHE, the counterfactual score adds nothing. The soft-mixture baseline tests the sub-claim that discrete argmax assignment is essential: if gradient-relaxed assignment matches, the EM structure is unjustified. The learned-embedding baseline tests whether intervention-defined entries are merely a reparameterization of a learned embedding trained with the same counterfactual reconstruction loss; if the two codebooks converge to the same directions (high cosine similarity), the intervention definition adds no structure beyond the loss. The no-orthogonalization ablation isolates the identifiability claim: without it, distinct actions should collapse onto overlapping directions, degrading the metric specifically on actions with similar latent effects. The k-means-on-deltas initialization tests whether the EM depends on lucky random unit-vector init; if it converges to the same assignments, the init sensitivity is low. The held-out action-classification metric breaks the circularity between the optimized counterfactual reconstruction score and the primary metric: a classifier trained on generated trajectories to predict the action label must generalize to held-out clips, so a gain cannot be attributed to overfitting the score. The score-vs-effect-magnitude correlation diagnostic tests the frozen-simulator faithfulness assumption: if the score correlates with ground-truth action-effect magnitude (measured as the norm of the observed latent delta), the score reflects action-specific effect rather than G's bias. The cross-seed cosine similarity diagnostic tests whether the grounding is model-specific or a property of the action. Counterfactual action-reconstruction accuracy is the right primary metric because it is exactly the quantity the method optimizes — the reconstruction of the factual next latent under a substituted codebook entry — so a gain on it cannot be attributed to a proxy; the secondary metric provides an independent check. If the causal chain is correct, gains should concentrate on action subsets where the baselines' mechanisms fail, not be uniform.

### Expected outcome and causal chain

**vs. Text-conditioned diffusion (caption channel)** — On a case where two actions share near-identical captions (e.g. "spin" vs. "rotate"), the baseline produces the same latent trajectory for both because its conditioning collapses through the symbolic bottleneck. Our method instead assigns distinct codebook entries because the counterfactual score separates them by their measured latent delta, so we expect a noticeable accuracy gap on caption-ambiguous action pairs but near-parity on caption-distinct actions.

**vs. Retrieval nearest-neighbor action selection** — On a case where the same action label appears in visually dissimilar scenes, the baseline retrieves a descriptor-similar but dynamically wrong entry because it matches on context similarity rather than intervention effect. Our method instead selects the entry whose intervention best reproduces the factual next latent across clips, so we expect a gap concentrated on context-heterogeneous action subsets and parity on context-homogeneous ones.

**vs. End-to-end soft-mixture codebook** — On a case where two actions have small but consistent latent deltas, the baseline produces a blended trajectory because gradient relaxation smears the discrete assignment across entries. Our method instead commits to one entry per action via argmax, preserving separation, so we expect the baseline to underperform specifically on fine-grained action distinctions while matching on coarse ones.

**vs. Learned-embedding codebook with same loss** — On a case where the counterfactual reconstruction loss is minimized by a learned embedding, the baseline may find a solution that fits the score but does not correspond to an intervention-defined delta. Our method constrains entries to be minimal latent deltas, so we expect the two codebooks to diverge (low cosine similarity) and CACHE to outperform on the held-out action-classification metric, especially for actions with small deltas.

**vs. CACHE without orthogonalization** — On a case where two actions have similar latent effects, the ablated model collapses them onto overlapping directions because without orthogonalization the entries are not forced to be linearly independent. Our method enforces linear independence, so we expect the ablation to underperform specifically on action pairs with high cosine similarity in their observed deltas.

**vs. CACHE with k-means-on-deltas initialization** — On a case where random initialization leads to a poor local optimum, the k-means-initialized variant should converge to the same or better assignments. We expect near-identical primary metric (difference < 1%) and assignment stability (adjusted Rand index > 0.9) across 5 random seeds, indicating low sensitivity to init.

### What would falsify this idea

If CACHE's gain over the text-conditioned and retrieval baselines is uniform across action subsets rather than concentrated on caption-ambiguous and context-heterogeneous subsets where those baselines' mechanisms fail, the causal story is wrong. Equally, if removing orthogonalization leaves the primary metric unchanged, the identifiability claim is unsupported; if the score-vs-effect-magnitude correlation is near zero, the frozen-simulator faithfulness assumption fails; and if the learned-embedding baseline converges to the same codebook directions, the intervention definition adds no structure.

## References

1. Yan: Foundational Interactive Video Generation
2. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models
3. StreamEval: Evaluating Interactive Streaming Video Generation Models
4. TC-Bench: Benchmarking Temporal Compositionality in Text-to-Video and Image-to-Video Generation
5. Video-Bench: Human-Aligned Video Generation Benchmark
6. WorldScore: A Unified Evaluation Benchmark for World Generation
7. T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation
8. Davidsonian Scene Graph: Improving Reliability in Fine-grained Evaluation for Text-to-Image Generation
