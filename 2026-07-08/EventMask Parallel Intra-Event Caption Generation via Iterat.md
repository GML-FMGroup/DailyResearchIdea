# EventMask: Parallel Intra-Event Caption Generation via Iterative Masked Language Modeling

## Motivation

The frontier in dense video captioning parallelization (e.g., Parallelized Autoregressive Decoding) retains sequential decoding within events because it assumes strong intra-event token dependencies require causal ordering. This assumption creates a sequential bottleneck that scales with event length, preventing full parallelization and limiting inference speed gains.

## Key Insight

Intra-event token dependencies can be resolved via iterative refinement because event captions exhibit a 'phrase structure' where local clusters of tokens are conditionally independent given event visual features and the evolving phrase context, allowing parallel prediction with multiple refinement passes.

## Method

### (A) What it is
EventMask is a non-autoregressive decoder that generates all tokens within an event in parallel via iterative masked prediction, conditioned on event-level visual features from a latent planner.

### (B) How it works
```python
def event_mask_decoder(event_feature, L, vocab_size, num_iterations=5, mask_schedule='cosine', T=1.0):
    # Initialize with fully masked tokens
    tokens = [MASK] * L
    # Transformer decoder with cross-attention to event_feature
    model = EventTransformer(vocab_size)
    # Learn temperature scaling parameter on validation set (512 examples, L-BFGS) to calibrate confidence
    # T is applied to logits before softmax
    for t in range(num_iterations):
        logits = model(tokens, event_feature)  # shape (L, vocab_size)
        scaled_logits = logits / T
        if t == num_iterations - 1:
            tokens = argmax(scaled_logits, dim=-1)  # final greedy decode
        else:
            probs = softmax(scaled_logits, dim=-1)
            confidence = max(probs, dim=-1)  # max probability per position
            # Cosine mask ratio: from 1 down to 0
            mask_ratio = cos( (t / num_iterations) * (pi/2) )
            k = max(1, int(L * mask_ratio))
            mask_indices = argsort(confidence)[:k]  # lowest confidence
            tokens[mask_indices] = [MASK] * k
    return tokens
```
The decoder is applied independently per event decoded from a latent planner; final caption concatenates all event outputs.

### (C) Why this design
We chose iterative refinement over one-shot non-autoregressive generation because intra-event token dependencies (e.g., verb→object) are non-trivial; a single pass cannot resolve them without causal masking. Iterative refinement allows the model to condition on its own predictions, building consistent structure across multiple updates. We chose confidence-based masking (instead of random or learned masking) because it focuses refinement on uncertain positions, mimicking the sequential process where later tokens depend on earlier confident ones. This avoids wasting iterations on already-correct tokens. We selected a transformer decoder with cross-attention (instead of an encoder-decoder) to share the same backbone as the parallelization scheme in prior work, easing integration and reducing parameters. The trade-off of confidence-based masking is that early confident predictions may lock in errors, but with a cosine schedule (starting with high mask ratio) the model has many chances to correct. The cosine schedule balances exploration (high masking early) and convergence (low masking late), inspired by Mask-Predict but adapted to per-event length. We did not use a separate controller or router; the entire process is a single forward pass per iteration, avoiding the anti-pattern of glued subsystems. To address potential miscalibration of confidence, we apply temperature scaling to logits before softmax, learning a scalar T on a held-out validation set (512 examples) using log-likelihood optimization (Guo et al., 2017). This ensures that confidence estimates are better calibrated and that low-confidence masking effectively targets erroneous tokens.

### (D) Why it measures what we claim
The quantity `logits` from the event-conditioned transformer measures the conditional probability of each token given event features and the current partial sequence; this operationalizes the assumption that intra-event dependencies can be captured by iteratively conditioning on a subset of tokens. The `confidence` (max softmax probability after temperature scaling) measures the model's calibrated certainty about each token; we assume that higher confidence correlates with correctness, enabling selective masking. This assumption fails when the model is overconfident on wrong predictions, in which case confidence reflects miscalibration rather than correctness, potentially causing error propagation. We mitigate this by temperature scaling, which adjusts confidence to better align with accuracy. The iterative process where we mask low-confidence tokens and predict again measures the model's ability to refine predictions; we assume that with access to more context (as other tokens are fixed), the model can correct errors. This assumption fails when dependencies are inherently global and cannot be resolved by local refinement—e.g., if a verb requires a distant object that is also masked—in which case the process may fail to converge. The cosine mask schedule measures the rate at which we reduce uncertain positions; we assume that refinement difficulty decreases as more tokens become confident, which is true if dependencies are local; otherwise, convergence may stall.

## Contribution

(1) EventMask, a non-autoregressive decoder that replaces intra-event sequential decoding with iterative masked language modeling, achieving parallel generation within each event. (2) The empirical finding that confidence-based iterative refinement can effectively model intra-event dependencies, enabling linear reduction in inference time proportional to event length without sacrificing caption quality.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | ActivityNet Captions | Standard benchmark for dense video captioning. |
| Primary metric | SODA_c | Measures caption quality and temporal alignment. |
| Baseline 1 | Vid2Seq (autoregressive) | State-of-the-art autoregressive dense captioner. |
| Baseline 2 | PDVC (non-autoregressive one-shot) | Representative one-shot non-autoregressive method. |
| Ablation-of-ours | EventMask with random masking | Isolates effect of confidence-based masking. |
| Calibration validation | Reliability diagrams on validation set | Verify confidence–correctness assumption after scaling. |

### Why this setup validates the claim
Comparing EventMask to Vid2Seq tests whether iterative refinement without autoregressive decoding can match or exceed sequential quality, isolating the benefit of causal masking removal. The contrast with PDVC directly examines if iterative refinement (our core addition) improves over one-shot generation for capturing intra-event dependencies. The ablation with random masking isolates the confidence-based selection, testing whether focusing on uncertain tokens is key. SODA_c is chosen because it penalizes both caption quality and temporal misalignment, which are exactly the dimensions our method aims to improve: better consistency from refinement and accurate grounding from event-level features. Additionally, we validate the load-bearing assumption about confidence–correctness correlation via reliability diagrams on the held-out validation set (512 examples) after temperature scaling; if calibration is poor, the confidence-based masking strategy may be invalid. We also report computational cost: EventMask training requires approximately 48 GPU hours on 4 NVIDIA V100 GPUs (comparable to Vid2Seq), with inference time per event of ~5 ms (vs. ~15 ms for autoregressive decoding), demonstrating practical efficiency.

### Expected outcome and causal chain

**vs. Vid2Seq (autoregressive)** — On a complex event like "A person pours oil into a pan and then fries vegetables", autoregressive decoding (Vid2Seq) may suffer from error accumulation: if the verb "pours" is mispredicted early, subsequent tokens like "oil" and "pan" become unlikely, cascading errors. EventMask instead refines all tokens in parallel: initial low-confidence tokens (e.g., "pours") get masked and re-predicted using increasingly confident context from other positions. This bidirectional conditioning avoids error propagation. We expect EventMask to produce more consistent captions on events with multiple interdependent tokens, achieving comparable or higher SODA_c on such cases, and similar performance on simple events.

**vs. PDVC (non-autoregressive one-shot)** — On an event where tokens have strong local dependencies (e.g., "slicing a carrot" where "slicing" and "carrot" must agree), PDVC predicts all tokens independently from the event feature, often producing mismatched verb-object pairs like "cutting a carrot" or "slicing a knife". EventMask iteratively refines: after initial guesses, low-confidence tokens (e.g., the verb) are masked and re-predicted using the now-fixed object (if confident), resolving the dependency. On such instances, we expect EventMask to yield a noticeable gap in SODA_c (e.g., 5-10% relative improvement) compared to PDVC, while on loosely structured events (e.g., listing attributes) differences shrink.

**vs. EventMask with random masking** — On an event where early predictions are all low-confidence but one is correct (e.g., a rare verb), confidence-based masking likely retains the correct token because its confidence is higher, while random masking might mask it and force a correction. However, if the model is miscalibrated (high confidence on wrong token), confidence-based masking may lock the error. We expect the confidence-based version to outperform random masking on average, but on miscalibrated subsets the gap may vanish or reverse. The ablation will show whether the assumed correlation between confidence and correctness holds. Additionally, reliability diagrams on the validation set will provide direct evidence: a diagonal plot indicates good calibration; deviations (e.g., overconfidence curve above diagonal) would suggest that confidence-based masking may be unreliable for certain event types.

### What would falsify this idea
If EventMask's advantage over PDVC is uniform across all event types—i.e., no concentration on events with strong intra-event dependencies—then the refinement mechanism is not addressing the core dependency issue as claimed. Alternatively, if random masking matches confidence-based masking overall, or if reliability diagrams reveal poor calibration even after temperature scaling (e.g., calibration error > 0.1), then our hypothesis about focusing on uncertain positions is false.

## References

1. Parallelized Autoregressive Decoding for Omni-Modal Dense Video Captioning
2. Qwen3-VL Technical Report
3. Factorized Learning for Temporally Grounded Video-Language Models
4. TimeMarker: A Versatile Video-LLM for Long and Short Video Understanding with Superior Temporal Localization Ability
5. Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces
6. TRACE: Temporal Grounding Video LLM via Causal Event Modeling
7. Grounded-VideoLLM: Sharpening Fine-grained Temporal Grounding in Video Large Language Models
8. TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding
