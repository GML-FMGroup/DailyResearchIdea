# Entropy-Guided Adaptive Block Size Selection for Speculative Decoding

## Motivation

Existing adaptive block size policies for speculative decoding, such as BlockPilot, rely on access to the target model's hidden states (e.g., prefilling representations) to predict the optimal block size per sample. However, in many deployment scenarios, the target model is a black-box API that only exposes output logits, making hidden-state-based policies inapplicable. We identify this as a convergent gap: the field assumes target model internals are necessary for instance-adaptive block size selection, limiting the applicability of adaptive speculative decoding in black-box settings. For example, BlockPilot's policy requires extracting representations from the target's prefilling stage, which is unavailable when the target is a proprietary model.

## Key Insight

Because the draft model is trained to approximate the target, the entropy of its output distribution inversely correlates with the expected acceptance probability, allowing block size to be set via a closed-form throughput formula without any hidden state from the target.

## Method

### [Frozen — do not change]
Title: Entropy-Guided Adaptive Block Size Selection for Speculative Decoding
Motivation: ...
Key insight: ...

### [Current method]
(A) **What it is**: We propose EADS (Entropy-Adaptive Draft Size), a policy that selects the block size *B* for speculative decoding based solely on the draft model's output distribution entropy *H* and the compute-speed ratio *ρ* = (cost of one target forward pass) / (cost of one draft forward pass). The policy is derived from an analytical throughput model that optimizes expected throughput as a function of *B*, given an acceptance rate that is a monotone decreasing function of *H*.

(B) **How it works**:

```
Input: draft model D, target model T (black-box logits only), sample x with prefix.
Output: block size B for the next draft block.

1. Run draft model D on prefix to obtain output distribution p over next token.
2. Compute entropy H = -∑ p_i log p_i (empirical estimate using top-10 probabilities).
3. Estimate acceptance rate α = f(H), where f is a monotone decreasing function (e.g., α = 1 / (1 + exp(γ(H - H0))) with learned parameters γ, H0 from calibration data.
   - **Calibration**: γ and H0 are calibrated on a held-out validation set of 512 examples using maximum likelihood estimation on observed acceptance rates. This mapping assumes that draft entropy inversely correlates with acceptance rate and is stable across distribution shifts.
4. Compute optimal block size B* using closed-form throughput formula:
   B* = floor( 1 / (1 - α) ) if ρ < 1, else compute via solving d/dB [ (B * α^B) / (B * τ_d + τ_t) ] = 0,
     where τ_d = cost per draft token, τ_t = cost per target token (including verification). Hyperparameters: γ (steepness of sigmoid), H0 (entropy threshold) – calibrated as above.
5. Return B*.
```

(C) **Why this design**: We chose a sigmoid function to map entropy to acceptance rate because it provides a smooth, bounded mapping that aligns with the observed correlation: at low entropy, α is high and near 1; at high entropy, α drops to a baseline. This is a stronger choice than a linear mapping because it saturates, preventing extreme block size values when entropy is very low or high. We also opted to estimate α from a simple calibration set rather than learning a complex neural network, accepting the trade-off that the mapping may not capture higher-order interactions (e.g., entropy variance across tokens) but gaining simplicity, interpretability, and minimal computational overhead. The closed-form optimal B is derived from an idealized throughput model that assumes independent acceptance of each token in the block; while this assumption may fail in practice (e.g., early rejections affecting later token probabilities), it yields a tractable formula and the errors are corrected by the rejection mechanism of speculative decoding itself. Finally, we use a single scalar entropy computed from the draft's next-token distribution, ignoring the full distribution across multiple tokens; this design is motivated by the requirement of minimal computation, accepting that a more sophisticated measure (e.g., average entropy over the block) might improve accuracy but would require multiple draft passes.

(D) **Why it measures what we claim**: The computational quantity α (acceptance rate estimated from entropy) measures the draft-target agreement because we assume that the draft model's confidence (low entropy) implies high likelihood of matching the target's next-token choice; this assumption holds when the draft is well-trained on the target's distribution and the target's output logits are smoothly varying. This assumption fails when the draft is poorly calibrated or when the target has sharp mode preferences that the draft overconfidently assigns low entropy to a different token; in such cases, α may overestimate the true acceptance probability, leading to suboptimal block size. The closed-form B* formula measures the optimal throughput given α, because we assume tokens within a block are accepted independently; this assumption fails when rejections are correlated or when the draft's quality degrades over the block length, in which case B* may be larger than truly optimal. The calibration of f via sigmoid parameters γ, H0 ensures that the entropy-to-acceptance mapping is empirically grounded on a validation set; this mapping measures the correlation observed in practice, but it fails to capture distribution shift between calibration and deployment settings, potentially causing miscalibration. **Load-bearing assumption**: The entropy of the draft model's output distribution is a reliable and stable predictor of the acceptance rate, and this relationship can be accurately captured by a fixed sigmoid function calibrated on a small validation set. To verify this assumption, we provide calibration curves (entropy vs. actual acceptance rate) on held-out data and analyze rejection correlation within blocks. If the assumption is violated (e.g., due to miscalibration or distribution shift), the policy is expected to underperform; in deployment, an online calibration mechanism could be used to update the mapping using observed rejections.

## Contribution

(1) A novel block size selection policy for speculative decoding that uses only the draft model's output distribution entropy, eliminating the need for target model hidden states and enabling adaptive block sizes in black-box settings.
(2) An analytical throughput model that derives a closed-form formula for optimal block size as a function of entropy and compute-speed ratio, providing a principled alternative to learned policies.
(3) Empirical demonstration that entropy-based block size selection matches or exceeds the efficiency of hidden-state-based policies on several language generation tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | WikiText-2 | Standard language modeling benchmark |
| Primary metric | Throughput (tokens/sec) | Direct measure of decoding speed |
| Baseline 1 | Fixed block size B=4 | Common default in speculative decoding |
| Baseline 2 | Fixed block size B=8 | Fixed alternative block size |
| Baseline 3 | Confidence-adaptive: block size proportional to draft model confidence (max softmax prob) with linear scaling | No-hidden-state baseline using draft confidence |
| Ablation-of-ours | EADS with linear mapping (α = 1 - γ*H) | Ablates sigmoid for entropy-to-α mapping |
| Additional analysis | Calibration curves and rejection correlation within blocks | Validate entropy-to-acceptance mapping and independence assumption |

### Why this setup validates the claim

This experimental design directly tests the central claim that entropy-adaptive block size (EADS) improves throughput over fixed block sizes. WikiText-2 provides varied entropy levels across tokens, enabling us to observe the effect of adaptation. Comparing against two fixed block sizes (B=4 and B=8) isolates the benefit of dynamic selection: if EADS outperforms both, the adaptation is effective. The ablation (linear mapping) tests whether the sigmoid mapping is crucial; if linear performs similarly, the sigmoid choice is unnecessary. Adding a confidence-adaptive baseline (Baseline 3) allows us to distinguish our entropy-based method from a simpler confidence-based approach, highlighting the distinctiveness of our approach. Throughput as the primary metric captures the ultimate goal of faster generation. A falsifiable pattern: EADS should outperform fixed blocks specifically on sequences with intermediate entropy, where fixed blocks are suboptimal, but not necessarily on very low or very high entropy sequences where fixed blocks may already be optimal. Additionally, calibration curves (plotting true acceptance rate vs. entropy bin) will be reported to directly validate the mapping assumption. Rejection correlation within blocks will be measured using Pearson correlation coefficient of rejection indicators across adjacent token positions; a high positive correlation would violate the independence assumption and potentially degrade performance.

### Expected outcome and causal chain

**vs. Fixed block size B=4** — On a case where the draft entropy is low (e.g., repetitive text like "the the the"), the baseline B=4 wastes compute because it always uses 4 draft tokens even though a longer block would be accepted (since acceptance rate is high). Our method instead selects a larger B (e.g., 8) because low entropy implies high α, increasing throughput. Thus we expect a noticeable throughput gap on low-entropy sequences, but parity on high-entropy sequences where both choose small B.

**vs. Fixed block size B=8** — On a case where the draft entropy is high (e.g., ambiguous next token), the baseline B=8 suffers from many rejections and wasted compute. Our method selects a smaller B (e.g., 2) because high entropy implies low α, avoiding costly rejections. We expect a throughput improvement on high-entropy sequences, but similar performance on low-entropy sequences where both choose large B.

**vs. Confidence-adaptive baseline (Baseline 3)** — Confidence (max softmax prob) and entropy are correlated but not identical; entropy accounts for the entire distribution while confidence only considers the maximum. We expect EADS to outperform on cases where the draft is overconfident (high confidence but high entropy in tails) because the sigmoid mapping provides a more nuanced scaling. On low-entropy, high-confidence sequences, both should perform similarly.

### What would falsify this idea

If EADS shows no throughput gain over fixed block sizes on any entropy range (e.g., performance is equal across all subsets), the entropy-to-acceptance mapping fails to capture real acceptance dynamics, invalidating the core assumption that entropy predicts acceptance rate. Additionally, if calibration curves show no monotonic relationship or if rejection correlation is high (e.g., >0.3) and the independence assumption is critically violated, the closed-form formula may yield suboptimal B*, leading to worse performance than fixed baselines.

## References

1. BlockPilot: Instance-Adaptive Policy Learning for Diffusion-based Speculative Decoding
2. BLOCK DIFFUSION: INTERPOLATING BETWEEN AU-TOREGRESSIVE AND DIFFUSION LANGUAGE MODELS
3. Fast-dLLM v2: Efficient Block-Diffusion LLM
4. SpecDiff-2: Scaling Diffusion Drafter Alignment For Faster Speculative Decoding
5. DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding
6. Simple Guidance Mechanisms for Discrete Diffusion Models
7. Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling
8. AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation
