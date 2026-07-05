# Self-Consistent Diffusion Speculative Decoding

## Motivation

Current speculative decoding methods (e.g., SpecDiff-2, DiffuSpec) rely on a separate autoregressive verifier or fine-tuned diffusion drafter, incurring additional training cost and architectural complexity. These approaches assume that draft generation and verification must be decoupled. We observe that diffusion models naturally satisfy a self-consistency property: adding a small amount of noise to a valid sample and then denoising recovers the original with high fidelity. This property can serve as a tuning-free verifier, eliminating the need for separate models or fine-tuning. For instance, DiffuSpec uses a separate AR verifier, while BlockPilot requires trained policy networks. By leveraging self-consistency, we unify draft and verification into a single diffusion model without additional components.

## Key Insight

The denoising reconstruction error of a diffusion model under controlled noise is a valid, tuning-free measure of drafting quality because a well-formed draft lies on the model's high-density manifold, making it robust to small perturbations.

## Method

### (A) What it is
Self-Consistent Diffusion Speculative Decoding (SCDS) uses a single diffusion language model to generate draft sequences in a non-autoregressive block and verify them via self-consistency: adding controlled noise and checking if the denoised output matches the original draft. The draft is accepted if a sufficient fraction of tokens are recovered, otherwise fallback to autoregressive generation occurs. The central assumption is that a well-formed draft lies on the model's high-density manifold, making it robust to small structured perturbations; we use span masking as the perturbation to test global coherence.

### (B) How it works
```pseudocode
Input: Target AR model M_AR, diffusion model M_diff (Transformer backbone: 12 layers, 12 heads, hidden size 768, trained on OpenWebText), noise parameters (span masking with span length L~Uniform{1,…,B}), draft block size B=5, acceptance threshold τ (calibrated on 512 sequences from WikiText-103 to maximize acceptance rate while keeping perplexity within 0.5% of AR), calibration set of 512 sequences.
1. Using M_diff, generate a draft block D of length B in parallel via DDIM reverse diffusion with 50 steps (instead of 1000 training steps).
2. Apply span masking: select a contiguous span of length L sampled uniformly from {1,…,B}, replace tokens in that span with a special [MASK] token, yielding D_masked.
3. Denoise D_masked using M_diff (same reverse process) to obtain D_denoised.
4. Compute self-consistency score: s = (number of masked positions where D_denoised == D) / L.
5. If s >= τ, accept full block D and append to generated sequence.
6. Else, reject D, and use M_AR to generate the next token (autoregressive) and continue.
7. Repeat until sequence is finished.
```

### (C) Why this design
We chose span masking over uniform random token noise because it tests global coherence: a locally plausible but globally incoherent draft might survive uniform noise but fail to recover a contiguous span. The threshold τ is calibrated on a small set to balance speed and quality. We chose fallback to AR rather than regenerating the draft because regeneration is costly and may not guarantee consistency; AR fallback ensures correctness at the cost of slower generation. Unlike DiffuSpec which relies on an external AR verifier, our verifier is intrinsic to the draft model, making the pipeline self-contained and tuning-free.

### (D) Why it measures what we claim
The self-consistency score s measures drafting quality because a well-formed draft should be robust to masking a contiguous span: the model should be able to infill the missing tokens consistently. This assumes that the diffusion model's learned distribution assigns high probability to coherent continuations. This assumption fails when the draft is globally coherent but the span masking is too easy (e.g., short span) or too hard (e.g., long span), causing the score to not reflect quality. The calibration on a small dataset mitigates this to some extent.

## Contribution

(1) We introduce Self-Consistent Diffusion Speculative Decoding (SCDS), a tuning-free method that uses the self-consistency property of diffusion models as a verifier for draft sequences, eliminating the need for separate verification models or fine-tuning. (2) We identify that denoising reconstruction error under controlled noise provides a valid, lightweight measure of drafting quality, enabling a unified draft-verify process. (3) We provide a design principle: leveraging inherent properties of generative models for verification, which can be extended to other generative modeling families (e.g., VAEs, flow models).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | WikiText-103 | Standard generation benchmark. |
| Primary metric | Tokens per second | Direct speed measure. |
| Baseline 1 | Autoregressive greedy | Slow but high quality. |
| Baseline 2 | Standard speculative decoding (separate AR verifier) | Separate verifier overhead. |
| Baseline 3 | DiffuSpec | Learned verifier cost. |
| Ablation-of-ours | Vary τ threshold | Tests self-consistency sensitivity. |
| Ablation: Learned verifier | Replace self-consistency with a small transformer (2 layers, 4 heads, hidden 256) trained on synthetic draft-quality pairs | Isolates benefit of tuning-free operation. |
| Validation of proxy | Measure Pearson correlation between s and perplexity/BLEU on 1000 held-out sequences from WikiText-103 | Directly tests assumption that s indicates quality. |
| Computational budget | GPU hours: 500 hours on 1 A100 for diffusion model training; draft generation: 0.5ms per token; self-consistency: 0.1ms per draft block | Demonstrates feasibility. |

### Why this setup validates the claim
This combination tests the core claim that self-consistency verification can replace an external verifier while maintaining speed and quality. The autoregressive baseline provides the lower speed bound; standard speculative decoding tests the benefit of using a separate verifier; DiffuSpec tests a learned verifier. The ablation on τ isolates the influence of the threshold. The learned verifier ablation isolates the benefit of tuning-free operation. The validation of proxy experiment directly measures whether s correlates with quality metrics, testing the central assumption. The primary metric (tokens per second) directly measures the target improvement, and the dataset allows quality assessment via perplexity. If our method achieves comparable speed to baselines with external verifiers but with less overhead or better quality, the claim is supported. Conversely, if speed is no better than AR, or if quality degrades significantly, or if s shows no correlation with quality, the claim is falsified.

### Expected outcome and causal chain
**vs. Autoregressive greedy** — On a long sequence, AR generates token-by-token causing linear latency. Our method drafts B tokens in parallel, so we expect significant speedup (e.g., 2-3x) while perplexity stays nearly identical because self-consistency rejects poor drafts and falls back.

**vs. Standard speculative decoding** — On a case where the draft model often disagrees with AR, standard spec decoding uses a separate AR verifier which adds overhead per verification. Our method uses lightweight self-consistency (no extra model). We expect higher acceptance rate (self-consistency is less discriminative) but faster verification, leading to comparable or better speed with slightly lower quality (if inferior drafts accepted). Overall speed gain is predicted.

**vs. DiffuSpec** — On out-of-distribution inputs, DiffuSpec's learned verifier may fail due to distribution shift. Our tuning-free self-consistency is intrinsic to the draft model, so we expect robust speedup and quality on such inputs, whereas DiffuSpec degrades.

**vs. Learned verifier ablation** — The learned verifier will likely have higher acceptance rate (better discrimination) but slower verification. We expect self-consistency to achieve similar speed with less training cost, but possibly lower quality at high thresholds.

### What would falsify this idea
If our speedup is no better than AR, or if quality degrades significantly on in-distribution data, or if the ablation shows threshold variation has no effect on speed-quality tradeoff, or if the correlation between s and quality metrics is below 0.3, then the self-consistency verification is not effective and the central claim is wrong.

## References

1. BlockPilot: Instance-Adaptive Policy Learning for Diffusion-based Speculative Decoding
2. BLOCK DIFFUSION: INTERPOLATING BETWEEN AU-TOREGRESSIVE AND DIFFUSION LANGUAGE MODELS
3. Fast-dLLM v2: Efficient Block-Diffusion LLM
4. SpecDiff-2: Scaling Diffusion Drafter Alignment For Faster Speculative Decoding
5. DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding
6. Simple Guidance Mechanisms for Discrete Diffusion Models
7. Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling
8. AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation
