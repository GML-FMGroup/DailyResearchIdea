# Contrastive Gradient Alignment for Joint Autoregressive-Diffusion Language Model Training

## Motivation

Existing joint training of autoregressive (AR) and diffusion objectives in a single model (e.g., Nemotron-Labs-Diffusion) leads to gradient interference, degrading performance in both modes. The structural problem is that the shared parameters receive conflicting gradient signals: the AR objective drives left-to-right prediction while the diffusion objective drives iterative denoising, causing optimization instability that prevents either mode from reaching its full potential.

## Key Insight

Gradient conflict is minimized when, for a given token, the hidden representations produced by the AR forward pass and the diffusion denoising pass at the same noise level are made consistent via contrastive learning, because aligned representations cause the two objectives' gradients to point toward a shared optimal point in parameter space.

## Method

## (A) What it is
**Contrastive Gradient Alignment (CGA)** is a training-time regularization method for joint AR-diffusion models. Input: a batch of sequences and a noise schedule. Output: a contrastive loss that is added to the standard AR+diffusion loss to align hidden representations.

## (B) How it works
**Load-bearing assumption:** Aligning hidden representations of AR and diffusion modes at matched noise levels reduces gradient conflict between the two objectives.
```python
# Hyperparameters: noise levels T=50, contrastive temperature tau=0.1, lambda_contrast=0.5
for each training step:
    sample batch of sequences x (batch_size=32)
    # AR forward pass with causal mask
    h_ar = model(x, mode='AR')  # hidden states at layer L=8 (middle layer of 12-layer transformer)
    # Sample noise level t ~ Uniform({1,...,T})
    t = random.randint(1, T)
    # Corrupt x to noise level t (mask random tokens, proportion = t/T)
    x_noisy = corrupt(x, t)
    # Diffusion denoising pass (bidirectional mask, from step t to 0)
    h_diff = model(x_noisy, mode='diffusion', noise_level=t)  # hidden states at layer L
    # For each token position i, define positive pair: (h_ar[i], h_diff[i])
    # Negative pairs: (h_ar[i], h_diff[j]) for j != i in the same batch+position set
    # Compute contrastive loss (InfoNCE)
    loss_contrast = -log( exp(sim(h_ar[i], h_diff[i])/tau) / sum_{j} exp(sim(h_ar[i], h_diff[j]/tau) ) )
    # sim is cosine similarity
    # Total loss = loss_ar + loss_diff + lambda_contrast * loss_contrast
```

## (C) Why this design
We chose **contrastive loss over mean-squared error (MSE)** because MSE would force exact identity of representations even when the two modes naturally diverge (e.g., diffusion denoising sees future context while AR does not), risking loss of mode-specific capabilities; contrastive loss only encourages similarity relative to other tokens, preserving flexibility. We selected **token-level alignment at a single noise level per step** (rather than all levels) to keep computational cost linear in sequence length (costs ~2X forward pass) and to provide stochastic gradient diversity that improves generalization. **Sampling t uniformly** rather than focusing on high-noise levels avoids the trivial solution where diffusion representations are nearly random and alignment becomes meaningless. We **applied alignment at a middle layer (layer 8 of 12)** not the final output layer because final-layer logits are already optimized by the separate losses and forcing them together would directly interfere; middle layers encode abstract representations that can be shared without harming task-specific heads. The trade-off is that contrastive alignment adds a hyperparameter lambda_contrast that must be tuned; if lambda_contrast is too large, mode-specific accuracy drops; if too small, conflict persists. We accept this overhead in exchange for a principled reduction of gradient interference.

## (D) Why it measures what we claim
**Contrastive loss at matched noise levels** measures **representation consistency** between AR and diffusion modes for the same token, under the assumption that when two objectives produce consistent hidden states, their parameter gradients are also consistent—i.e., gradient conflict (measured by cosine similarity of gradients) is reduced. This assumption fails when the alignment is applied at a layer where the two modes already agree by coincidence (e.g., early layers that learn generic features); in that case, the loss reflects agreement that does not affect gradient conflict. To mitigate this, we select layer L=8 empirically by monitoring gradient conflict on a validation set (e.g., we choose L where naive joint training shows highest gradient angular variance). The **positive pair** (same token, same noise level) operationalizes the **desired invariance** across modes; the **negative pairs** (different tokens) operationalize the **discrimination** that prevents collapse to a trivial solution where all representations become identical. The **temperature tau=0.1** controls the hardness of negative mining: smaller tau enforces stronger separation between different tokens, which is necessary because without it the model could align all tokens to a common point, losing token-specific information. We assume that token-specific information is preserved in the absolute differences among representations, not just their pairwise relations—this assumption is standard in contrastive learning and holds as long as the embedding space is not degenerate (e.g., all representations collapse to a single point).

## Contribution

(1) A contrastive loss mechanism (CGA) that aligns AR and diffusion hidden representations at matched noise levels during joint training, directly addressing gradient interference. (2) A design principle that gradient conflict can be reduced by enforcing representation consistency without sacrificing mode-specific capabilities, validated by improved performance on both AR and diffusion objectives. (3) A practical training recipe specifying layer selection, noise sampling, and contrastive temperature that can be integrated into existing joint AR-diffusion frameworks (e.g., Nemotron-Labs-Diffusion).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | OpenWebText subset (100K docs) | Diverse, common LM benchmark |
| Primary metric | Validation perplexity | Directly measures language modeling quality |
| Auxiliary metric | Gradient cosine similarity (between AR and diffusion gradients) | Checks if alignment reduces conflict |
| Baseline: no alignment | Joint AR+diffusion w/o contrastive | Tests if contrastive helps |
| Baseline: AR only | Autoregressive only | Upper bound for AR performance |
| Baseline: Diffusion only | Diffusion only | Upper bound for diffusion performance |
| Baseline: gradient surgery (PCGrad) | Joint training with PCGrad gradient projection | Compares with explicit conflict reduction |
| Ablation: MSE alignment | CGA with MSE instead | Tests contrastive vs. MSE |
| Ablation: t sampling distribution | CGA with t sampled from uniform vs. low-noise-centric | Tests sensitivity to noise level |
| Compute | 4x A100 80GB, 48 hours | Feasible for 125M parameter model |

### Why this setup validates the claim

This setup forms a falsifiable test of the central claim that contrastive alignment reduces gradient conflict. Validation perplexity on a diverse corpus directly reflects whether conflicting gradients impair training—if conflict is reduced, perplexity should decrease. The joint training without alignment baseline isolates the effect of the contrastive term: if CGA outperforms it, the conflict reduction is beneficial. The AR-only and diffusion-only baselines represent ideal single-mode performances; if CGA approaches them, it shows the method preserves mode-specific capabilities. The MSE ablation tests whether the contrastive mechanism, which allows flexible alignment, is superior to a rigid one. The PCGrad baseline compares with an explicit conflict-reduction method. The auxiliary gradient cosine similarity metric directly measures whether contrastive alignment indeed increases gradient agreement, providing causal evidence. Together, these comparisons reveal whether the predicted gradient conflict reduction translates into measurable improvement, and whether the design choices are essential.

### Expected outcome and causal chain

**vs. Joint training without alignment** — On a case where a sentence contains a syntactic ambiguity resolvable only by future context (e.g., "The horse raced past the barn fell"), the two modes produce opposing gradients: AR wants to predict the next word without future info, while diffusion uses future context. The baseline cancels these gradients, leading to slow convergence and high perplexity. Our method aligns representations via contrastive loss, reducing conflict. We expect a noticeable perplexity gap (e.g., 5–15% lower on such long-range dependency examples) but parity on simple short sentences where gradients naturally agree.

**vs. AR only** — On a case requiring bidirectional context (e.g., a gap-filling task like "He ___ the ball"), the AR model cannot access future tokens, so it predicts a generic verb (e.g., "threw") with moderate probability, yielding higher perplexity. Our joint model leverages the diffusion mode to condition on both preceding and following tokens, predicting the correct verb (e.g., "caught") with higher probability. We expect our method to outperform AR on such bidirectional tasks, but still be worse than diffusion-only on pure bidirectional tasks. The gap between our method and AR should be larger on bidirectional examples compared to left-to-right examples, indicating that conflict reduction preserves AR capabilities.

**vs. Diffusion only** — On a case requiring strict left-to-right generation (e.g., a story continuation prompt), the diffusion model generates all tokens simultaneously, often producing less coherent narrative flow because it lacks causal structure. Our method uses the AR mode for such tasks, maintaining causal coherence. We expect our method to outperform diffusion-only on standard next-token prediction benchmarks (left-to-right), but be on par or slightly worse on tasks like text infilling where bidirectional context is key. The pattern confirms that our method does not sacrifice AR strength.

**vs. PCGrad** — PCGrad explicitly projects gradients to reduce conflict, but may overly smooth gradient updates, hurting mode-specific performance. Our contrastive alignment is a softer regularization. We expect CGA to achieve lower perplexity than PCGrad on both AR and diffusion specific tasks, because CGA encourages representation sharing without forcing gradient orthogonality. On examples where gradients severely conflict, PCGrad might initially show faster conflict reduction, but we hypothesize CGA leads to better final perplexity.

**Ablation: MSE alignment** — On a case where the two modes require different abstract representations (e.g., AR encodes position-specific information while diffusion encodes bidirectional context), MSE forces exact identity, causing the model to compromise and lose mode-specific accuracy. Our contrastive loss allows representations to differ while being similar relative to others. We expect CGA to achieve lower perplexity than MSE alignment, especially on diverse examples where mode-specific features are important.

**Ablation: t sampling distribution** — Uniform sampling of noise level t is expected to be more robust than low-noise-centric sampling, because low-noise levels produce similar representations across modes, reducing the contrastive signal. We expect uniform t to yield lower perplexity and higher gradient cosine similarity.

### What would falsify this idea

If the perplexity of CGA is not significantly lower than joint training without alignment, or if the improvement is spread uniformly across all sequence lengths rather than concentrated on longer sequences where gradient conflict is predicted to be highest, then the central claim that contrastive alignment reduces conflict is false. Additionally, if the MSE ablation matches or beats CGA, the design rationale for contrastive over MSE is invalid. If gradient cosine similarity does not increase under CGA relative to no alignment, the assumed mechanism is unsupported.

## References

1. Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding
2. SDAR: A Synergistic Diffusion-AutoRegression Paradigm for Scalable Sequence Generation
3. LLaDA2.0: Scaling Up Diffusion Language Models to 100B
4. TiDAR: Think in Diffusion, Talk in Autoregression
5. Scaling Diffusion Language Models via Adaptation from Autoregressive Models
6. Efficient Training of Language Models to Fill in the Middle
