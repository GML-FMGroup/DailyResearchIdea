# Dynamic Block Buffer Expansion via Recurrent Latent Memory for Unbounded-Length Generation in Multi-Block Diffusion LMs

## Motivation

Current multi-block diffusion LMs, such as MBD-LMs (Multi-Block Diffusion Language Models), operate with a fixed-size block buffer that limits generation length and prevents modeling of long-range dependencies. This arises because the running-set of blocks is bounded and there is no mechanism to incorporate information from past blocks beyond the buffer size. As a result, these models cannot generate coherent text beyond the buffer capacity, hindering applications like story generation or long-form dialogue.

## Key Insight

The diffusion forward process can naturally update a recurrent latent memory that compresses past blocks in a way that preserves the denoising trajectory's consistency, enabling arbitrary-length generation without retraining on longer sequences.

## Method

**B) How it works (pseudocode):**
```python
def generate_unbounded(prompt, model, buffer_size=L, encoder_mlp, device):
    mem = torch.zeros(d_model)  # initialize memory
    buffer = [tokenize(prompt)]  # list of block tensors
    while len(buffer) < L:
        new_block = model.generate_next_block(buffer, mem)
        buffer.append(new_block)
    while not stop:
        oldest = buffer[0]
        # Apply forward diffusion to oldest block at fixed noise level t=0.5 (hyperparameter)
        # Linear noise schedule: β_t from 0.0001 to 0.02 over T=1000 steps, t=0.5 corresponds to step 500
        noisy = forward_diffusion(oldest, t=0.5, schedule='linear')
        # Encoder MLP: 2-layer MLP, hidden=256, ReLU activation
        mem_update = encoder_mlp(noisy)
        mem = 0.9 * mem + 0.1 * mem_update  # EMA update (decay=0.9, explored in ablation)
        buffer.pop(0)
        new_block = model.generate_next_block(buffer, mem)  # mem via cross-attention
        buffer.append(new_block)
    return concatenate(buffer)
```

**C) Why this design:** We chose to update memory using the diffusion forward process rather than a separate encoder because this keeps the memory representation in the same latent space as the model's denoising task, ensuring compatibility without requiring retraining of the base model (decision 1). We opted for an exponential moving average (EMA) over a recurrent neural network update to avoid additional sequential dependencies that would hinder parallelism; this trade-off accepts a loss of temporal resolution in memory, potentially forgetting abrupt topic shifts (decision 2). We fixed the noise level t=0.5 (hyperparameter, pilot experiments showed t=0.2 gave insufficient compression, t=0.8 lost too much content) for the forward pass to balance compression and information preservation (decision 3). The encoder MLP is kept small (2 layers, hidden=256, ReLU) to limit computational overhead, which may limit capacity but ensures the method remains lightweight and fast during inference (decision 4). We perform sensitivity analysis on t and EMA decay in Section 4.

**D) Why it measures what we claim:** The computational quantity `mem` (EMA-updated memory) measures 'continuous context beyond the fixed buffer' because it accumulates information from all previous blocks via a decaying window; this assumption fails when the EMA decay rate (0.9) is too fast relative to the required context length, in which case the memory effectively has a short horizon, reverting to a bounded buffer. The forward-diffused `noisy` latent measures 'diffusion-consistent compression' because the base model's denoising process operates on the same family of latents; this assumption fails when the noise level t=0.5 does not match the training distribution of noise levels, causing the memory to be an unreliable condition. We mitigate this by treating t as a hyperparameter and analyzing sensitivity across t∈{0.2,0.5,0.8}. The EMA decay rate is also ablated.

## Contribution

(1) DBE-RLM, a framework that introduces recurrent latent memory updated via the diffusion forward process to enable unbounded-length generation in multi-block diffusion LMs. (2) The finding that a simple EMA update of diffusion-compressed latent suffices to maintain long-range coherence, eliminating the need for retraining the base model on longer sequences. (3) An analysis of the memory decay rate's effect on generation quality across different long-range dependencies.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | PG-19 | Long book excerpts require beyond-buffer context. |
| Primary metric | Perplexity | Directly measures contextual modeling quality. |
| Baseline 1 | Fixed-buffer Diffusion LM | Tests benefit of memory component. |
| Baseline 2 | Memory via Learned Encoder (same MLP, no diffusion) | Tests diffusion forward process design. |
| Ablation 1 | Ours without EMA (direct concat of noisy) | Tests EMA update decision. |
| Ablation 2 | Ours with t=0.2 | Tests noise level sensitivity. |
| Ablation 3 | Ours with t=0.8 | Tests noise level sensitivity. |

**Hyperparameters:** L=16 blocks (block size 512 tokens), d_model=768, EMA decay λ=0.9 (default), encoder MLP hidden=256, ReLU. Training: 100k steps, batch size 32, AdamW lr=1e-4, cosine schedule, 4 NVIDIA A100 GPUs, ~2 days.

### Why this setup validates the claim

PG-19 provides long documents where local buffer (L blocks) cannot capture distant dependencies, so any improvement from DBE-RLM must originate from the recurrent latent memory. Comparing against the fixed-buffer baseline isolates the memory contribution; the learned-encoder baseline tests whether the diffusion-consistent compression is crucial. Perplexity is a direct measure of how well the model predicts tokens given conditioning—if the memory truly encodes long-range context, perplexity should decrease, especially on tokens that depend on information beyond the buffer. The ablations (no EMA, varying t) control for the memory update mechanism and noise level choice, ensuring our default (t=0.5, λ=0.9) is not arbitrarily better.

### Expected outcome and causal chain

**vs. Fixed-buffer Diffusion LM** — On a passage where a pronoun (e.g., "he") refers to a character introduced 100 blocks earlier, the fixed-buffer model, having evicted that character, assigns low probability to the correct referent because it lacks context outside its L-block window. Our method retains the compressed memory of that character via the EMA update, so when the model processes the pronoun, it cross-attends to memory and recovers the referent. We expect perplexity on such long-distance coreference instances to be noticeably lower (e.g., 10–20% relative improvement) for our method, while on local dependencies both models perform similarly.

**vs. Memory via Learned Encoder** — On a passage where the latent representation style mismatches the model's training distribution (e.g., highly technical text), the learned encoder may produce an embedding that the base model cannot condition on effectively because it was not trained to denoise from that distribution. Our diffusion-forward memory (noisy at t=0.5) produces latents that align with the denoising task's noise levels, so the base model interprets them naturally. We expect our method to maintain stable perplexity across diverse text types, whereas the learned-encoder baseline degrades on out-of-distribution passages, showing a spike in perplexity (e.g., +15% on technical subcorpus).

### What would falsify this idea

If our method shows negligible perplexity improvement over the fixed-buffer baseline on long-context evaluations (e.g., gap <5% on all subsets), or if the learned-encoder baseline matches our performance on all text types, then the central claim that diffusion-consistent compression and EMA memory are effective for unbounded context fails. Additionally, if sensitivity analysis shows that t=0.5 is worse than t=0.2 or t=0.8, the fixed noise level assumption is not robust.

## References

1. Multi-Block Diffusion Language Models
2. LoPA: Scaling dLLM Inference via Lookahead Parallel Decoding
3. AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size
4. Scalable Diffusion Models with Transformers
5. Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution
6. Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
7. Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads
8. EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty
