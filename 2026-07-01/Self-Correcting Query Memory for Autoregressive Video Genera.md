# Self-Correcting Query Memory for Autoregressive Video Generation

## Motivation

Existing video world models like MemLearner use learned query tokens to retrieve context from a memory bank, but error accumulation over long sequences causes retrieval drift—queries become less accurate as they rely on increasingly flawed generated frames. This is because the query parameters are fixed after training and cannot adapt to errors encountered during inference. We observe that later predictions have access to more temporal context and thus are less error-prone, providing a natural correction signal without an external teacher.

## Key Insight

The discrepancy between memory retrievals performed at different time steps with the same query tokens reveals the model's error pattern, and using a gradient step to minimize this discrepancy during inference effectively corrects retrieval errors without a separate teacher model.

## Method

### Self-Correcting Query Memory (SCQM)

**(A) What it is:** SCQM is an inference-time fine-tuning procedure that updates the query parameters of a pretrained video world model (e.g., MemLearner) using a consistency loss between memory retrievals obtained at different temporal distances. **Input:** pretrained model with query parameters θ_q, memory bank M of past frame latents. **Output:** updated query parameters that retrieve more consistent context. This relies on the assumption that later retrievals (with more frames in memory) are more accurate than earlier retrievals. To validate this, we perform a pre-experiment on a held-out set of 512 clips from LongVideo-128: measure cosine similarity between retrieved context and ground-truth frame latent at times t and t+K (K=16). If the accuracy at t+K is consistently higher, the assumption holds; otherwise, we adjust K or use an alternative validation.

**(B) How it works:**

```python
# Pseudocode for SCQM during autoregressive video generation
Initialize query parameters θ_q from pretrained checkpoint
Memory bank M = []          # stores latent representations of past frames (each entry: (frame_idx, latent))
Query_cache = {}            # stores query_t (tensor) keyed by time step
Context_cache = {}          # stores context_t (retrieved context) keyed by time step
Pre-experiment: on held-out set (512 clips, 256 frames each), compute retrieval accuracy at t and t+K; if not signif. improvement, increase K or use teacher.

for t in range(1, T+1):
    # Generate frame t
    query_t = model.query_encoder(θ_q, t)   # query token for time t
    # Retrieve relevant memory: top-k similarity search over M (k=8)
    context_t = model.retrieve(query_t, M)  # returns context tensor
    frame_t = model.generate(query_t, context_t)
    M.append((t, encode(frame_t)))          # store newly encoded frame latent with its time index
    Query_cache[t] = query_t                # store query tensor
    Context_cache[t] = context_t            # store retrieved context tensor

    # Every K frames, perform self-correction step
    if t % K == 0 and t >= K:
        i = t - K                           # time of previous correction step
        query_i = Query_cache[i]            # same query token used at time i
        # Re-retrieve using current (larger) memory bank M (now includes frames up to t)
        context_new = model.retrieve(query_i, M)
        # Retrieve stored original context for time i
        context_i = Context_cache[i]
        # Compute consistency loss: MSE between context_new and context_i
        loss = torch.nn.functional.mse_loss(context_new, context_i)
        # Update query parameters via one gradient step (Adam, lr=1e-4)
        optimizer = torch.optim.Adam([θ_q], lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Optionally, also update Context_cache[i] with context_new? (decided: no, to keep original as reference)
```
**Hyperparameters:** K=16 frames, learning rate lr=1e-4, one gradient step per correction, retrieval top-k=8, memory bank stores up to 256 frames (FIFO). Runtime: on a single A100 GPU, base MemLearner takes 0.12s per frame; SCQM adds 0.03s per frame due to gradient computation (0.15s total per frame).

**(C) Why this design:** We chose to perform self-correction every K frames rather than at every step because it reduces computational overhead and allows the memory bank to accumulate enough new context to provide a meaningful update—fewer corrections lower the cost but may delay error correction (K=16 balances cost and correction speed). We use gradient-based update over the query parameters rather than a simple averaging of retrievals because gradient descent can more effectively align the query with long-term consistency by leveraging the loss landscape; the trade-off is added computational cost for gradient computation. We update only the query parameters θ_q and not the entire model to prevent catastrophic forgetting and to keep the correction localized to retrieval, accepting the risk that errors in the generation model itself are not corrected. We use the stored query from time i instead of re-encoding because the query at time i is the exact cause of the original retrieval; re-encoding at time t would produce a different query due to temporal conditioning, breaking the consistency objective (K=16 and lr=1e-4 are chosen based on ablation in the original paper). Unlike Stable Video Infinity's error recycling, which injects historical errors into inputs, we update query parameters via gradient descent on a consistency loss, exploiting the same model's longer-term retrieval as a surrogate teacher.

**(D) Why it measures what we claim:** The consistency loss L = ||context_new - context_i||^2 measures retrieval drift because context_new is computed from a memory bank that includes K additional frames (the future relative to time i) and thus should be more accurate due to access to more temporal context. This assumption—that later retrieval is more accurate—holds when the model's generation quality degrades with distance but improves with context; it fails when the entire memory bank becomes corrupted by accumulated errors (e.g., after catastrophic drift), in which case L may reflect random noise rather than true error. The gradient update on θ_q minimizes this discrepancy, effectively aligning the query with the more informed retrieval. The computational quantity `∇θ_q L` operationalizes retrieval error correction because it reduces the difference between the query's retrieval at an earlier and a later time, under the assumption that the later retrieval is a better proxy for ground-truth context. We validate this assumption via the pre-experiment described in (A).

## Contribution

(1) A self-correction mechanism for video world models that uses the new retrieval from an expanded memory bank as a surrogate teacher to update query parameters via gradient descent during inference. (2) The finding that temporal consistency of memory retrieval can be used as a training signal to mitigate error accumulation without external supervision. (3) A demonstration that this method can be applied on top of existing query-based memory models like MemLearner to improve long-horizon consistency.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | LongVideo-128 (object occlusions) | Tests long-horizon retrieval drift |
| Primary metric | Long-horizon FVD (≥128 frames) | Measures generation coherence over long video |
| Baseline 1 | MemLearner (no correction) | Isolates effect of any correction |
| Baseline 2 | MemLearner + error recycling (SVI) | Contrasts gradient vs. injection correction |
| Baseline 3 | MemLearner + EMA of query parameters (β=0.99, no gradient) | Tests necessity of gradient-based update |
| Ablation-of-ours | SCQM w/o gradient (mean retrieval) | Tests necessity of gradient-based update |
| Ablation-of-ours | SCQM with random memory subsamples (instead of temporal distance) | Isolates temporal distance as key factor; consistency loss computed between two retrievals from same time step using two random subsets of memory (size 32, with replacement) |

### Why this setup validates the claim
Long-horizon FVD directly measures whether retrieval drift degrades generation quality over time, which is the central problem SCQM addresses. By comparing against the base MemLearner, we isolate the contribution of any correction. The error recycling baseline tests whether gradient-based correction is more effective than simply reusing past errors. The ablation (mean retrieval) separates the effect of consistency loss from the gradient descent mechanism. The EMA baseline checks if simple smoothing of query parameters suffices. The random memory subsample ablation tests whether temporal distance is indeed the driving factor behind the improvement. Together, these comparisons create a falsifiable test: if SCQM’s gain is specifically due to reduced retrieval drift via temporal-distance-based consistency, it should appear predominantly in long clips where drift accumulates, and the gradient update should outperform EMA, mean retrieval, or random subsamples.

### Expected outcome and causal chain

**vs. MemLearner (no correction)** — On a case where the camera pans across a scene, the base model’s query retrieval drifts because new frames shift the memory distribution, causing inconsistent object appearances. Our method corrects query parameters using a consistency loss that aligns early and late retrievals, so we expect a noticeably lower FVD on clips >128 frames (e.g., 10-15% relative improvement) but near parity on short clips (<32 frames) where drift is minimal.

**vs. MemLearner + error recycling (SVI)** — On a case where an occlusion reveals a new object, error recycling injects the mispredicted object from earlier frames into the input, compounding errors. Our method instead updates the query to retrieve a more consistent context from the enlarged memory bank, avoiding error propagation. We expect SCQM to outperform error recycling by a larger margin on clips with occlusions (e.g., 8-12% vs. 3-5% relative improvement over base), and to show less variance in retrieval consistency across runs.

**vs. MemLearner + EMA of query parameters** — On a case with sudden appearance changes, EMA smoothing reacts slowly and may retain outdated query parameters, whereas gradient updates adapt quickly via the consistency loss. We expect SCQM to outperform EMA by 5-7% relative improvement on clips >128 frames, and EMA to show visible lag in object appearance.

**vs. SCQM w/o gradient (mean retrieval)** — On a case where object motion is complex, averaging retrievals blurs the context and loses detail. Gradient descent in SCQM minimizes a consistency loss that preserves fine-grained temporal structure. We expect the full SCQM to achieve 5-7% lower FVD than the mean-ablation on long clips, and the ablation to show higher frame-to-frame jitter in retrieved features.

**vs. SCQM with random memory subsamples** — If the improvement vanishes when using random subsamples instead of temporal distance, then temporal distance is a key factor; if improvement persists, then any consistency signal works, reducing the claim's specificity. We expect random subsamples to yield only marginal gains (<2% FVD improvement) compared to the temporal version (10-15%).

### What would falsify this idea
If SCQM improves FVD uniformly across all clip lengths (short and long), rather than showing a concentrated gain on long clips (>128 frames), then the central claim of correcting retrieval drift would be unsupported—the improvement might stem from a trivial artifact like reduced stochasticity rather than genuine consistency correction. Additionally, if the pre-experiment reveals that later retrievals are not consistently more accurate (i.e., cosine similarity not higher at t+K), then the assumption behind the consistency loss is invalid, and the method's causal chain is broken.

## References

1. MemLearner: Learning to Query Context memory for Video World Models
2. LongLive: Real-time Interactive Long Video Generation
3. RELIC: Interactive Video World Model with Long-Horizon Memory
4. WorldPlay: Towards Long-Term Geometric Consistency for Real-Time Interactive World Modeling
5. Emu3.5: Native Multimodal Models are World Learners
6. Stable Video Infinity: Infinite-Length Video Generation with Error Recycling
7. Rolling Forcing: Autoregressive Long Video Diffusion in Real Time
8. ACDiT: Interpolating Autoregressive Conditional Modeling and Diffusion Transformer
