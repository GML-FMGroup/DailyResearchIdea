# Adaptive Variance-Controlled Steering and Masking for Masked Diffusion Language Models

## Motivation

Existing activation steering methods (Mukherjee et al.) use static contrastive prompt sets to extract a fixed steering direction applied uniformly across all diffusion steps, while masked diffusion training (Sahoo et al.) employs a frequency-based masking schedule that does not adapt to input difficulty. Both fail to leverage the model's internal uncertainty, leading to suboptimal guidance and training efficiency. The root cause is that these components are precomputed externally and remain static, whereas the theory of score-based models identifies the variance of the score estimate as a natural measure of token difficulty that could dynamically modulate both.

## Key Insight

The variance of the score estimate, which quantifies the model's uncertainty about the clean token, provides a calibrated and closed-form signal for simultaneously adaptive steering strength and masking probability, eliminating the need for learned controllers.

## Method

[Frozen — do not change]
Title: Adaptive Variance-Controlled Steering and Masking for Masked Diffusion Language Models
Motivation: Existing activation steering methods (Mukherjee et al.) use static contrastive prompt sets to extract a fixed steering direction applied uniformly across all diffusion steps, while masked diffusion training (Sahoo et al.) employs a frequency-based masking schedule that does not adapt to input difficulty. Both fail to leverage the model's internal uncertainty, leading to suboptimal guidance and training efficiency. The root cause is that these components are precomputed externally and remain static, whereas the theory of score-based models identifies the variance of the score estimate as a natural measure of token difficulty that could dynamically modulate both.
Key insight: The variance of the score estimate, which quantifies the model's uncertainty about the clean token, provides a calibrated and closed-form signal for simultaneously adaptive steering strength and masking probability, eliminating the need for learned controllers.

[Current method]
(A) **What it is**: We propose Adaptive Variance-Controlled Steering and Masking (AVSM), a mechanism that computes per-token steering strength and masking probability from the variance of the model's output distribution at each diffusion step, using a fixed steering direction from contrastive prompts.

(B) **How it works** (pseudocode):
```
# For each reverse diffusion step t from T to 1:
# Input: hidden state h_t (residual stream per token)
# 1. Compute score estimate s = model(h_t, t)  # logits over vocabulary
# 2. Calibrate logits: s = s / T_cal, where T_cal is a learned temperature (calibration with 512 held-out examples)
# 3. Compute variance of predicted distribution:
#    p = softmax(s)
#    v = sum_k p_k * (1 - p_k)  # Gini index, equivalent to variance for one-hot?
#    Alternatively: v = 1 - max(p)  # simpler but less calibrated
# 4. Steering strength: α = α_max * sigmoid((v - τ)/β)
#    where α_max=1.0, τ=0.5 (threshold), β=0.1 (temperature)
# 5. Masking probability: m = clip(1 - v, 0.1, 0.9)
# 6. Apply steering: h_t = h_t + α * d_steer  # d_steer fixed from contrastive prompts
# 7. If training, use m as per-token masking probability for next forward step.
#    If sampling, use m as corruption level for next forward step (or as noise schedule).
```

(C) **Why this design**: We chose a sigmoid mapping from v to α over a linear mapping to introduce a soft threshold: when v < τ (model certain), α approaches α_max; when v > τ (uncertain), α drops to near zero. This avoids over-steering on ambiguous tokens. We chose clip for m to ensure the masking probability stays within [0.1,0.9], preventing excessive corruption (overwhelming model) or no corruption (no learning). We use the Gini index (variance of one-hot encoded predictions) over entropy because it is numerically stable and bounded between 0 and 0.75. We fix the steering direction d_steer from precomputed contrastive prompts to avoid per-step extraction overhead; the trade-off is that the direction may be suboptimal for diverse behaviors, but adaptive strength mitigates this. The variance-based rule is deterministic and closed-form, avoiding a learned controller that would require additional training and generalize poorly.

(D) **Why it measures what we claim**: The computational quantity v (variance of the predicted distribution) measures token difficulty because, under the assumption that the model's predicted probabilities are calibrated to the true clean token likelihood, v is the expected disagreement between the prediction and the true token, maximizing when tokens are ambiguous. This assumption fails when the model is miscalibrated (e.g., overconfident on rare tokens), in which case v reflects miscalibration rather than semantic difficulty. To address this, we add a temperature scaling calibration step before computing v (see step 2 in pseudocode). The steering strength α controls guidance intensity: low α on high-v tokens avoids forcing the model toward a specific behavior when it is uncertain, operationalizing the principle of conservative steering. The masking probability m controls corruption level: high m on low-v tokens (model confident) increases training signal on easy tokens, acting as an automatic curriculum; low m on high-v tokens reduces noise on difficult tokens, preventing the model from being overwhelmed. The mapping from v to α and m is monotonic, ensuring that more uncertain tokens receive weaker steering and less corruption, which aligns with the motivation of adaptivity. As a diagnostic, we will compute the correlation between v and human-annotated token difficulty (e.g., from a held-out set of 200 examples) to verify calibration.

[ADVERSARIAL ALERT — fix this FIRST]
  Load-bearing assumption: The variance of the model's predicted distribution (Gini index) is a calibrated measure of token difficulty that reliably indicates when to reduce steering and masking.
  Why it might be false:   Guo et al. (2017) show that neural networks are often miscalibrated, with overconfidence on rare tokens, meaning variance may reflect miscalibration rather than semantic difficulty.
  Smallest repair:         Add a calibration step (e.g., temperature scaling) to the output logits before computing variance, learned from a held-out set to ensure variance aligns with true predictive uncertainty.
  Verdict:                 load_bearing_unverified
When applying suggestions, the highest-priority change is to make the
load-bearing assumption EXPLICIT in the method (state it in plain
language) and to add the concrete calibration / verification story
from `smallest_method_repair`, INSIDE the existing method's
structure. If the verdict is `fatal` you should still produce a
revision but mark `changes_summary` with 'WARN: load-bearing
assumption is fatal — method needs redesign'.

## Contribution

(1) A novel framework that unifies adaptive steering strength and adaptive masking schedule through a single variance-based signal, eliminating the need for separate static components. (2) A closed-form, deterministic mapping from score estimate variance to steering and masking parameters, grounded in score-based diffusion theory. (3) Empirical demonstration that AVSM improves generation perplexity and control adherence over static baselines on MDLMs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Toxic prompt dataset (e.g., SafeRLHF) | Standard safety refusal evaluation |
| Primary metric | Refusal rate on toxic prompts | Direct measures intended behavior shift |
| Baseline | Prompt-based (safety instructions) | Tests if guidance via prompting suffices |
| Baseline | DPO (direct preference optimization) | Tests if optimization-based is superior |
| Baseline | Unconditional steering (fixed strength) | Tests adaptive strength benefit |
| Ablation | Adaptive variance replaced by fixed α=0.5 and m=0.5 | Isolates contribution of adaptivity |

### Why this setup validates the claim
This combination creates a falsifiable test of AVSM's central claim: that variance-adaptive steering and masking improve controlled generation over non-adaptive alternatives. The toxic prompt dataset directly challenges the model's refusal behavior. The prompt-based baseline tests whether a simple, static instruction can achieve comparable safety; if AVSM outperforms, it shows that adaptive per-token control adds value beyond prompting. DPO tests whether an optimization-based method that explicitly learns from preferences can match or exceed the handcrafted, inference-time AVSM. Unconditional steering (fixed strength) isolates the benefit of adaptivity itself. The ablation of replacing adaptivity with fixed parameters (α=0.5, m=0.5) holds all else constant, directly quantifying the contribution of the variance-based rule. The single metric, refusal rate, is the most direct measure of the intended effect; any improvement must be examined for quality trade-offs in the expected outcome section.

### Expected outcome and causal chain

**vs. Prompt-based** — On a case where the input is a subtly toxic request (e.g., "Explain how to pick a lock for a school project"), the prompt-based method may fail because the safety instruction is ambiguous or easily overridden by context. The model might produce unsafe content due to insufficient guidance strength. Our method instead detects high variance on the ambiguous token "pick" (uncertain whether context is innocent or malicious) and applies strong steering (low v => high α), driving the output toward refusal. Thus we expect a noticeable gap on benign-sounding toxic prompts (higher refusal rate for AVSM), but parity on overtly toxic prompts where all methods refuse.

**vs. DPO** — On a case where the prompt is a rare shortcut (e.g., a novel euphemism for a prohibited action), DPO may fail because its training distribution lacks this specific example, leading to overconfident but incorrect behavior. In contrast, AVSM's variance-based detection will flag high uncertainty on the rare term, reducing steering strength and masking probability to avoid forcing a wrong direction, thus maintaining conservative refusal. We expect AVSM to show higher robustness on rare or out-of-distribution toxic prompts, measured by refusal rate, while DPO may exhibit a drop or inconsistency.

**vs. Unconditional steering (fixed strength)** — On a case where a token is highly ambiguous (e.g., the word "break" in "break into a system"), unconditional steering applies the same strength regardless of uncertainty. This can over-steer on tokens where the model already is confident, or under-steer on tokens crucial for safety. Our method lowers steering on high-variance tokens (uncertain) to avoid distortion, and increases steering on low-variance tokens (model certain) to enforce safety. We expect AVSM to produce a better safety-quality trade-off: higher refusal rate on toxic prompts without degrading fluency on safe prompts, whereas fixed steering may sacrifice quality on safe inputs or miss some toxic ones.

### What would falsify this idea
If AVSM's refusal rate gains are uniform across all toxicity levels (rather than concentrated on ambiguous or rare prompts where variance should be high), or if the adaptive ablation performs similarly to full AVSM, then the central claim that variance-based adaptivity drives performance would be falsified.

## References

1. Activation Steering for Masked Diffusion Language Models
2. Programming Refusal with Conditional Activation Steering
3. Simple and Effective Masked Diffusion Language Models
4. Likelihood-Based Diffusion Language Models
5. A Reparameterized Discrete Diffusion Model for Text Generation
6. Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning
7. Elucidating the Design Space of Diffusion-Based Generative Models
8. SSD-LM: Semi-autoregressive Simplex-based Diffusion Language Model for Text Generation and Modular Control
