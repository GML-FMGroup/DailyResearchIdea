# Granularity-Routed Cross-Branch Alignment for Unified Multimodal Understanding and Generation

## Motivation

Unified multimodal models such as UniSpark share one transformer backbone across understanding and generation, yet enforce no constraint that the object-, region-, and scene-level representations used for comprehension correspond to the same granularities the generation branch must synthesize; SEED-X makes the same architectural bet with a shared tokenizer and objective, and likewise leaves cross-branch granularity correspondence unconstrained. The structural root cause is that the two branches are optimized by disjoint losses (autoregressive text prediction vs. flow matching) over feature spaces that are free to drift apart at every block, so the understanding branch's fine-grained object tokens and the generation branch's synthesis tokens can encode incompatible partitions of the same image without any training signal detecting it. This is a field-level bottleneck: every architecture-level unification claim (UniSpark, SEED-X) is unfalsifiable because no mechanism ties comprehension granularity to synthesis granularity.

## Key Insight

Because object-, region-, and scene-level tokens partition the same visual scene into a nested hierarchy, matching understanding-branch features to generation-branch features at each granularity level and routing each level to a dedicated expert creates a nested-consistency invariant that a single shared loss cannot express and that retrieval-style or single-scale alignment cannot enforce.

## Method

**What it is.** GRACE (Granularity-Routed Alignment of Cross-branch Embeddings) is a training-time auxiliary mechanism added to a UniSpark-style hybrid-attention backbone. Inputs: for each training image, the per-block hidden states of the understanding branch and the generation branch, plus a granularity label for each visual token (object / region / scene) derived from segmentation and detection annotations. Outputs: a cross-branch granularity alignment loss and a routing assignment that specializes MoE experts by granularity level.

**Load-bearing assumption (stated explicitly).** The alignment loss treats attention-pooled hidden states within a fixed semantic granularity level as a sufficient statistic for that level's content, so that a similarity between the understanding-branch and generation-branch pooled summaries at that level is equivalent to cross-branch agreement on that level's content. This assumption has three failure modes that the method must guard against: (i) the two branches are optimized by different objectives (autoregressive text prediction vs. flow matching) and may encode identical content in mutually rotated or permuted bases, so a raw cosine similarity is sensitive to invertible linear transforms and does not measure shared information (Kornblith et al., 2019); (ii) the generation branch's hidden states at block l are conditioned on a flow-matching timestep that varies per training step, so the pooled generation summary is a function of the noise schedule and the alignment signal can be driven to zero by collapsing the generation branch's dependence on the noise level rather than by aligning content (Lipman et al., 2023); (iii) mislabeled instance masks make the object-level alignment reflect annotation noise rather than branch disagreement. The method below specifies how each failure mode is handled.

**How it works.**

```pseudocode
# Per training step, for each transformer block l in 1..L:
#   H_u[l] : understanding-branch tokens (clean image context)
#   H_g[l] : generation-branch tokens (noisy visual targets)
#   t      : flow-matching timestep for this step, t in [0,1]

1. Granularity assignment (precomputed, frozen):
   for each visual token t in image I:
       g(t) = OBJECT  if t falls in a detected instance mask
              REGION  if t falls in a segmenter region but no instance
              SCENE   otherwise
   # Labels are precomputed once and released as a frozen artifact;
   # label coverage and ambiguity rates are reported up front.

2. Cross-branch projection (per block, shared across granularities):
   P_u = W_u[l] @ H_u[l]      # W_u[l]: d -> d_proj, d_proj = 256
   P_g = W_g[l] @ H_g[l]      # separate projection, no weight sharing

3. Granularity-pooled matching, basis-invariant and timestep-conditioned:
   for level k in {OBJECT, REGION, SCENE}:
       A_u[k] = attention-pool(P_u, mask=g==k)   # pooled summary per level
       # Generation summary is pooled within timestep bins to remove the
       # noise-schedule confound: bin t into B=10 equal-width bins, pool
       # A_g[k, b] over tokens in bin b, then average over bins.
       A_g[k] = mean_{b=1..B} attention-pool(P_g, mask=(g==k) & (t in bin b))
       # Basis-invariant alignment: a learned linear map M_k: d_proj -> d_proj
       # per granularity level, trained jointly, so the loss measures shared
       # information rather than shared basis.
       L_align[k] = 1 - cosine(A_u[k], M_k @ A_g[k])

4. Routing constraint on MoE experts:
   for each token, router logits r(t) over E=16 experts
   L_route = sum_t  -log softmax(r(t))[e*(g(t))]
   where e*(k) is a fixed expert index assigned to granularity level k
   (OBJECT->experts {0..5}, REGION->{6..10}, SCENE->{11..15})

5. Total loss:
   L = L_understand + L_flowmatch + lambda_align * mean_l mean_k L_align[k]
       + lambda_route * L_route
   lambda_align = 0.05, lambda_route = 0.01 (warmup over 5k steps)
```

The alignment is applied at every block (not only the final layer) so that granularity correspondence is maintained throughout the residual stream. The routing loss uses a fixed, non-learned expert-to-granularity map rather than a learned router so that the constraint is a hard structural prior, not a soft preference. The per-level learned linear map M_k is the concrete repair for the basis/rotation failure mode: it lets the loss measure shared information rather than shared basis. The timestep-bin pooling is the concrete repair for the noise-schedule confound: it prevents the alignment signal from being satisfied by suppressing the generation branch's dependence on the noise level.

**Sufficiency probe (precondition for the loss to be meaningful).** Before trusting the alignment loss, we verify the sufficient-statistic assumption directly: for each granularity level k, we train a linear readout from A_u[k] and from A_g[k] to the same granularity label (object/region/scene) on a held-out set of 512 examples, and require both readouts to be predictive (probe accuracy above a chance baseline). If probe accuracy is low, the pooled summaries are not sufficient statistics for level content, and the alignment loss is measuring noise; we report this as a precondition and do not interpret the alignment loss in that case.

**Why this design.** We chose per-block alignment over final-layer-only alignment because the residual stream is where UniSpark's hybrid attention branches mix, so drift accumulates block-by-block and a final-layer loss would let intermediate blocks diverge unconstrained; the cost is an L-fold increase in alignment compute, which we accept because d_proj=256 keeps the projection cheap relative to the 9B backbone. We chose attention-pooling over mean-pooling within a granularity level because object masks vary in token count and mean-pooling would let a single large object dominate the level summary; the cost is one extra attention head per level that must be trained. We chose a fixed expert-to-granularity routing map over a learned router because a learned router can collapse to uniform assignment and thereby satisfy the alignment loss trivially, whereas the fixed map makes granularity specialization a hard constraint; the cost is reduced flexibility in expert utilization, which we mitigate by allowing 5-6 experts per level. We chose a learned per-level linear map and timestep-bin pooling over raw cosine because the two branches are free to learn rotated bases and the generation branch is conditioned on a varying noise level, so raw cosine would be confounded by basis and timestep; the cost is one d_proj x d_proj matrix per level and a binning hyperparameter B=10, which we accept because it makes the alignment loss measure shared content rather than shared basis or noise dependence. A domain expert would not describe this as a variant of SEED-X's shared-tokenizer unification, because SEED-X unifies the input surface (one tokenizer) while GRACE constrains the internal feature geometry across branches at multiple semantic scales — a property SEED-X's single shared objective does not produce and cannot check.

**Why it measures what we claim.** The per-level alignment L_align[k] measures cross-branch granularity consistency because the assumption is that attention-pooled features within a fixed semantic granularity level are a sufficient statistic for that level's content, so a basis-invariant similarity between the two branches' pooled summaries is equivalent to agreement on that level; this assumption fails when a detected object mask is wrong (mislabeled instance), in which case L_align[OBJECT] reflects annotation noise rather than branch disagreement, and it also fails when the learned linear map M_k absorbs a basis rotation that does not correspond to shared content, in which case L_align[k] reflects the map's capacity rather than branch agreement. The routing loss L_route measures granularity specialization of experts because the assumption is that a fixed expert-to-level map combined with a cross-entropy penalty forces tokens of that level to share an expert subspace, so low L_route is equivalent to specialization; this assumption fails when a token's granularity label is ambiguous at boundaries, in which case L_route reflects label noise rather than routing quality. The per-block placement measures drift accumulation because the assumption is that block-wise feature divergence is monotone in depth when unconstrained, so penalizing at each block is equivalent to bounding cumulative drift; this assumption fails when a block is frozen or near-identity, in which case the per-block term contributes no signal and the metric reflects only active blocks. The timestep-bin pooling measures content agreement rather than noise-schedule agreement because the assumption is that within a timestep bin the generation summary is approximately independent of the noise level, so averaging over bins removes the confound; this assumption fails when the noise schedule varies sharply within a bin, in which case L_align[k] still reflects residual timestep dependence.

## Contribution

(1) A cross-branch multi-granularity alignment mechanism (GRACE) with per-block object/region/scene matching and a fixed granularity-to-expert routing constraint, applicable to any dual-branch unified multimodal backbone. (2) The design principle that granularity specialization must be a hard structural prior (fixed expert map) rather than a learned soft preference, because learned routers admit a collapse solution that satisfies alignment losses trivially. (3) A granularity-labeled alignment protocol that reuses existing detection and segmentation annotations, requiring no new data collection.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | UniSpark interleaved + editing suite, granularity-labeled (frozen artifact; label coverage and ambiguity rates reported up front) | Provides OBJECT/REGION/SCENE labels and both branch targets |
| Primary metric | Granularity-stratified cross-branch consistency vs. task accuracy | Detects the predicted subset-specific effect |
| Baseline | UniSpark hybrid-attention backbone (no GRACE) | Tests whether auxiliary alignment helps at all |
| Baseline | Final-layer-only alignment (no per-block, no routing) | Isolates the per-block and routing sub-claims |
| Baseline | Learned-router MoE (no fixed granularity map) | Tests whether the hard structural prior matters |
| Ablation-of-ours | GRACE with mean-pooling instead of attention-pooling | Tests the pooling sub-claim |
| Ablation-of-ours | GRACE with raw cosine (no learned linear map M_k, no timestep-bin pooling) | Tests the basis-invariance and timestep-conditioning sub-claims |
| Diagnostic | Linear probe from A_u[k] and A_g[k] to granularity label on 512 held-out examples | Verifies the sufficient-statistic precondition |
| Diagnostic | Label-noise sensitivity sweep: inject 5/10/20% mask corruption | Bounds the annotation-noise failure mode |

### Why this setup validates the claim

The central claim is that constraining cross-branch feature geometry at multiple semantic granularities, throughout the residual stream and with a hard expert-routing prior, produces granularity-consistent branches that a single surface-level unification cannot. The plain UniSpark baseline tests whether any auxiliary alignment is needed; if GRACE shows no gain over it, the whole premise is dead. The final-layer-only baseline tests the per-block sub-claim specifically: it holds the alignment objective fixed but removes the depth-wise constraint, so any gap between it and full GRACE is attributable to drift accumulation rather than the loss itself. The learned-router baseline tests the hard-prior sub-claim by letting the router satisfy alignment softly, isolating whether the fixed map is what forces specialization. The mean-pooling ablation tests whether attention-pooling is doing real work or is decorative. The raw-cosine ablation tests whether the learned linear map and timestep-bin pooling are doing real work or whether the alignment signal is confounded by basis rotation and noise schedule. The linear probe is a precondition check: if the pooled summaries are not predictive of granularity content, the alignment loss is measuring noise and no downstream comparison is interpretable. The label-noise sweep bounds how much of the object-level alignment signal is annotation noise. The metric is granularity-stratified consistency, because the claim predicts gains concentrated on the granularity levels where branches actually diverge, not uniform improvement; a uniform gain would be evidence for a generic regularization effect rather than the proposed mechanism.

### Expected outcome and causal chain

**vs. UniSpark hybrid-attention backbone** — On an interleaved instance where the understanding branch sees a clean scene and the generation branch must reconstruct a small object inside a larger region, the unconstrained baseline lets the two branches drift so the generated object inherits region-level statistics and loses instance detail, because nothing ties branch features at the object scale. GRACE applies per-block, per-level basis-invariant alignment so the object-pooled summaries are pulled together, so we expect a visible gap on object-level consistency and on editing fidelity for small instances, with near-parity on scene-level coherence where the baseline already agrees.

**vs. Final-layer-only alignment** — On a deep hybrid-attention case where intermediate blocks progressively rotate the understanding and generation subspaces apart, the final-layer-only baseline can still match at the last block while earlier blocks have already diverged, so its alignment loss looks low but intermediate representations disagree and downstream generation is brittle. GRACE penalizes each block, bounding cumulative drift, so we expect lower block-wise divergence variance across depth and a measurable gap on tasks requiring mid-stack feature reuse, with convergence on shallow or frozen blocks where the term is inert.

**vs. Learned-router MoE** — On a batch dominated by scene-level tokens, the learned router collapses toward a near-uniform expert assignment that trivially satisfies the soft alignment objective, so experts never specialize and object tokens share subspaces with scene tokens. GRACE's fixed map forces object tokens into experts 0-5 and scene tokens into 11-15, so we expect sharper expert-usage entropy separation by granularity and a gap on object-heavy subsets, with parity on homogeneous scene-only batches where specialization does not matter.

**vs. GRACE with raw cosine (no learned map, no timestep-bin pooling)** — On a training run where the two branches learn rotated bases of the same object content, the raw-cosine ablation shows low alignment loss even though the branches encode different content in aligned bases, and the timestep confound lets the generation branch suppress its noise dependence to satisfy the loss. GRACE with the learned linear map and timestep-bin pooling measures shared content rather than shared basis or noise dependence, so we expect the raw-cosine ablation to show lower probe accuracy and higher downstream editing error on object-heavy subsets, with parity on scene-level coherence where basis rotation is small.

**vs. Linear-probe precondition** — If the linear probe from A_u[k] and A_g[k] to granularity label is at chance, the pooled summaries are not sufficient statistics for level content and the alignment loss is measuring noise; in that case we report the alignment loss as uninterpretable and do not claim a gain from it. If the probe is predictive, the alignment loss is interpretable and the downstream comparisons above are meaningful.

**vs. Label-noise sweep (5/10/20% mask corruption)** — As mask corruption increases, L_align[OBJECT] reflects annotation noise rather than branch disagreement, so we expect object-level consistency to degrade monotonically with corruption rate while scene-level consistency is unaffected; this bounds the annotation-noise failure mode and tells us how much of the object-level gain is robust to label noise.

### What would falsify this idea

If GRACE's gain over the plain backbone is uniform across all granularity strata rather than concentrated on the levels predicted to diverge, the multi-scale alignment claim is wrong and the effect is just generic regularization. If the learned-router baseline matches GRACE on object-heavy subsets, the hard structural prior is not what produces specialization. If the linear probe is at chance, the sufficient-statistic assumption fails and the alignment loss is measuring noise rather than content.

## References

1. UniSpark: Exploring Hybrid Attention for Multimodal Understanding and Generation
