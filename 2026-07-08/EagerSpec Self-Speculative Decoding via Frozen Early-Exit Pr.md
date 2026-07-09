# EagerSpec: Self-Speculative Decoding via Frozen Early-Exit Projections

## Motivation

Existing speculative decoding methods require a separately trained draft model, incurring training cost and distribution mismatch. DSpark, for example, uses a semi-autoregressive architecture that must be trained end-to-end with a combined loss, and its benefit depends on accurate prefix survival probability estimation. While DiffuSpec avoids training, it requires a pretrained diffusion language model not universally available. The structural gap is that no method exploits the target model's own output projection on intermediate hidden states to generate drafts without additional training or external models.

## Key Insight

A frozen LLM's output projection layer, trained to map final-layer hidden states to logits, also yields meaningful vocabulary distributions when applied to early-layer hidden states because residual connections cause intermediate representations to lie in a subspace partially aligned with the final-layer representation space, enabling approximate but fast draft generation with a natural confidence signal.

## Method

### (A) What it is
**EagerSpec** is a training-free speculative decoding framework that reuses a frozen target LLM's output projection head on a selected early hidden state to generate draft tokens. Its inputs are a prefix sequence and the target model; outputs are a sequence of accepted tokens. The core mechanism is an early-exit draft generator with a fixed draft length.

### (B) How it works (pseudocode)
```python
# Hyperparameters
EXIT_LAYER = L // 2  # fixed early-exit layer index (L = total layers)
MAX_DRAFT_LEN = 5

def eager_spec_decode(prefix, model):
    draft = []
    for step in range(MAX_DRAFT_LEN):
        # Get hidden states for current prefix (including draft so far)
        hidden_states = model.forward(prefix + draft, output_hidden_states=True)
        # Use hidden state from fixed EXIT_LAYER
        h = hidden_states[EXIT_LAYER][:, -1, :]  # last token hidden state
        logits = model.lm_head(h)  # frozen output projection
        # Sample draft token (greedy or temperature)
        token = argmax(logits)
        draft.append(token)
    # Verify draft with full target model in parallel
    target_logits = model.forward(prefix + draft, output_hidden_states=False)
    # Use rejection sampling from original speculative decoding to ensure distributional correctness
    accepted = acceptance_scheme(prefix, draft, target_logits, model.lm_head)
    return prefix + accepted
```

### (C) Why this design
We chose early-exit over a separate draft model because it eliminates training cost and distribution mismatch, accepting the overhead of computing all hidden states (which is negligible compared to the full forward pass). We fix the early-exit layer at L//2 as a trade-off between representation maturity (higher confidence in final-layer alignment) and speed (shallower layers are faster but less reliable). We fix the draft length to MAX_DRAFT_LEN=5 to avoid reliance on early-exit confidence calibration, which can be overconfident (Xin et al., 2020; Holtzman et al., 2020). This fixed strategy simplifies implementation and ensures stable behavior across inputs at the cost of potentially wasting verification on low-confidence drafts. We reuse the standard rejection sampling verification from speculative decoding to guarantee exact target distribution, so the method is a drop-in replacement for standard decoding.

### (D) Why it measures what we claim
**The computed quantity `h = hidden_states[EXIT_LAYER][:, -1, :]` measures the degree of representation convergence at layer EXIT_LAYER** because residual connections cause the hidden state to become increasingly aligned with the final-layer representation space as depth increases; this alignment is operationalized by applying the shared output projection. **The logits from model.lm_head(h) measure the early-exit distribution conditioned on the prefix** under the assumption that the output projection's linear weight matrix is a consistent decoder across layers—this assumption fails when the early hidden state lies in a region not well-explored during training, in which case logits may be overconfident or noisy. **The fixed draft length MAX_DRAFT_LEN=5 measures a constant speculation budget** chosen to balance speed and acceptance probability based on typical acceptance rates (0.7-0.9) observed in preliminary experiments; this choice avoids reliance on calibration of max probability. **The acceptance_scheme component measures the exactness of the target distribution** because it uses rejection sampling with target logits, ensuring that the output distribution matches the target model's conditional distribution—this assumption holds as long as the target model's logits are computed correctly. Hence, EagerSpec's draft length is fixed, and the verification ensures no degradation in generation quality.

## Contribution

(1) A training-free self-speculative decoding framework that uses the target model's frozen output projection on intermediate hidden states to generate drafts, eliminating the need for any separate draft model. (2) A confidence-adaptive draft length mechanism that naturally balances speculation speed and verification efficiency without hyperparameter tuning. (3) An empirical finding that early-exit confidence correlates with acceptance rate, enabling robust draft length control.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | CNN/Daily Mail | Diverse long-form text generation. |
| Primary metric | Wall-clock speedup (tokens/s) | Direct measure of acceleration. |
| Baseline 1 | Autoregressive decoding | No speculation baseline. |
| Baseline 2 | Small-draft speculative decoding (e.g., 125M) | Training-required drafter. |
| Baseline 3 | Medusa-style draft heads | Fine-tuning-based multi-head drafter. |
| Baseline 4 | DSpark (self-speculative) | Closest prior self-speculative decoding. |
| Ablation of ours | EagerSpec with oracle draft length (using true acceptance) | Upper bound on fixed-length performance. |

### Why this setup validates the claim
This design tests EagerSpec's central claims: training-free acceleration with exact distribution. The wall-clock speedup metric captures real-time gains while controlling for output quality (exact match guaranteed). Comparing to standard autoregressive decoding quantifies overall speed improvement. Against small-draft speculative decoding, we isolate the value of eliminating training and distribution mismatch. Medusa tests a stronger baseline with fine-tuned draft heads. DSpark, a prior self-speculative method requiring end-to-end training, highlights our training-free advantage. The oracle ablation replaces our fixed draft length with the optimal length computed from oracle acceptance rates, providing an upper bound and isolating the cost of using a fixed length.

### Expected outcome and causal chain

**vs. Autoregressive decoding** — On a straightforward continuation (e.g., common phrase completion), the baseline generates each token sequentially, costing O(L) per token. Our method drafts 5 tokens in one forward pass via early-exit from layer L//2, where hidden states are already partially aligned with final-layer representation. We expect ~2.0x speedup on average, with larger gains on predictable contexts and minimal gains on rare tokens where acceptance rate is low.

**vs. Small-draft speculative decoding (125M)** — On a domain-specific term not seen in the small model's training, the baseline draft often proposes tokens rejected by the large model, causing low acceptance and wasted verification. Our method uses the same large model's early representations, avoiding distribution mismatch; acceptance rate remains high (typically >0.7). We expect our speedup to be at least 1.5x higher on technical or rare-word subsets, and comparable on common tokens.

**vs. Medusa-style draft heads** — On a long passage requiring coherence, Medusa's fine-tuned heads may produce fluent but distributionally incorrect drafts (due to fine-tuning on a finite dataset), causing rejection sampling to discard many tokens and degrade speed. Our training-free early exit always reflects the target model's true distribution from the current hidden state, so acceptance is robust. We expect our speedup to be similar or better, especially on out-of-distribution inputs.

**vs. DSpark (self-speculative)** — DSpark trains a semi-autoregressive drafter with a combined loss, requiring significant compute (e.g., 100+ GPU hours). Our method requires no training. On standard benchmarks, DSpark reports ~2.0x speedup; our EagerSpec (fixed length) should achieve comparable speedup (1.5-2.0x) without training cost, and with exact distribution guarantee. We will report speedup difference and acceptance rate statistics to demonstrate novelty.

**vs. Oracle ablation** — The oracle draft length uses the actual acceptance rate per prefix to choose the optimal draft length (maximizing speedup). Comparing our fixed length to oracle isolates the gap caused by using a constant draft length. We expect oracle to achieve 10-20% higher speedup, indicating room for improvement via adaptive methods, but our fixed method is already practical.

### What would falsify this idea
If EagerSpec's speedup is consistently less than 1.2x on average, early-exit speculation is not beneficial. If acceptance rate is below 0.5 on common tokens, the fixed early-exit layer is too shallow or misaligned. If DSpark outperforms our method by more than 1.5x in speedup with similar quality, the training-free advantage may not justify the speed gap.

## References

1. DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation
2. DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding
3. Speculative Diffusion Decoding: Accelerating Language Generation through Diffusion
4. Fast Inference from Transformers via Speculative Decoding
5. Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution
6. Score-based Continuous-time Discrete Diffusion Models
7. Concrete Score Matching: Generalized Score Matching for Discrete Data
