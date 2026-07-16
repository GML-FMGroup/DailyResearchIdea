# Self-Supervised Proxy Evaluation for Autonomous Guidance Generation in LLM Post-Training

## Motivation

Existing methods like the PUST framework (Proxy Exploration and Reusable Guidance) rely on an external reward model to guide proxy exploration, creating a dependency that limits autonomy and scalability. The quality of the guidance signal is bounded by the reward model's accuracy, and the proxy cannot independently assess its own outputs, hindering self-sufficient exploration. We need a mechanism where the proxy self-evaluates without external rewards, enabling truly autonomous guidance generation.

## Key Insight

The proxy can derive a reliable internal reward by measuring the consistency of its own outputs across multiple stochastic passes, because high-quality outputs (e.g., correct reasoning) tend to be more consistent across different sampling conditions than low-quality ones, providing a self-supervised signal that correlates with true quality.

## Method

**Self-Supervised Proxy Evaluation for Guidance (SSPEG)**

(A) **What it is**: SSPEG is a training algorithm that enables a proxy language model to autonomously generate guidance signals without an external reward, by using self-consistency as an intrinsic reward. It takes the proxy model as input and outputs a learned evaluation function and update signals that are transferred to a primary LLM.

(B) **How it works**:

```pseudocode
# SSPEG Algorithm
# Input: proxy model π_θ (smaller LLM), primary model π_φ (large LLM), unlabeled prompts D
# Hyperparameters: K = 5, λ = 0.5, η = 1e-5
# Calibration set: 512 labeled examples from the same distribution as D

# Preliminary calibration (run once before main loop):
# Compute Pearson correlation between mean pairwise agreement and ground-truth correctness on calibration set.
# If correlation > 0.5, we proceed with confidence; otherwise, we flag potential failure.
# Determine threshold τ as the median agreement among calibration set outputs (or use a fixed threshold if correlation is low).

for each batch of prompts x in D:
    # Step 1: Generate candidate outputs with multiple passes
    responses = [sample from π_θ(x) with temperature=1.0 for _ in range(K)]
    
    # Step 2: Compute consistency score for each response
    # For each response r_i, compute its pairwise agreement with other responses
    # using ROUGE-L as similarity metric 
    scores = []
    for i in range(K):
        agreement = mean([rougel(r_i, r_j) for j != i])   # rougel ∈ [0,1]
        scores.append(agreement)
    
    # Step 3: Assign binary labels based on consistency threshold
    # High-consistency: scores above threshold τ (tuned on calibration set)
    # Low-consistency: scores below τ
    high_cons = [r_i for i where scores[i] > τ]
    low_cons = [r_i for i where scores[i] <= τ]
    
    # Step 4: Self-supervised update of proxy via contrastive learning
    # Maximize likelihood of high-consistency responses, minimize of low-consistency
    loss = - (1/|high_cons|) * sum(log π_θ(r_i|x) for r_i in high_cons) \
           + (1/|low_cons|) * sum(log π_θ(r_i|x) for r_i in low_cons)
    θ = θ - η * ∇_θ loss
    
    # Step 5: Extract update signal from proxy
    # After proxy update, compute the change in proxy's policy or output distribution
    # e.g., logit difference between initial and updated proxy on a held-out prompt
    # We store the direction Δ = (θ_updated - θ_initial) as the guidance signal
    
    # Step 6: Transfer signal to primary model
    # Use gradient alignment: update φ to align with Δ via cosine similarity loss
    # e.g., minimize -cos(∇_φ L(π_φ(x)), Δ) where L is task loss
    # For simplicity, use KL divergence: minimize KL(π_φ || π_θ_updated)
    perform one step of KL divergence minimization between π_φ and π_θ_updated on batch x

return updated primary model π_φ
```

(C) **Why this design**: We chose a contrastive self-supervised update over ranking-based methods (e.g., DPO) because we need to avoid any external preference labels, and contrastive learning directly amplifies the signal from consistent outputs while suppressing inconsistent ones. We used pairwise agreement (e.g., ROUGE-L) as the consistency metric instead of output probability entropy because agreement captures semantic similarity better than token-level probabilities, which can be dominated by surface form. The trade-off is that computing pairwise agreement requires K(K-1)/2 comparisons per prompt, increasing computational cost, but it yields a more reliable intrinsic reward. We chose to update the proxy via contrastive loss rather than through reinforcement learning (e.g., PPO) because we want to avoid the complexity of a value function and the instability of policy gradient; the contrastive loss provides a simple, stable alternative. However, this design assumes that the median split provides a sufficiently clean separation between high- and low-quality outputs, which may not hold if the consistency distribution is uniform; in that case, we risk noisy labels. We calibrated the threshold τ on a held-out set of 512 labeled examples to ensure that high-consistency outputs indeed correspond to higher quality. We opted for a tuned threshold over a median split because it is more robust when the consistency–quality correlation is weak. Finally, we transfer signals via KL divergence minimization rather than gradient alignment to avoid requiring task-specific loss functions on the primary model, making the method general. The cost is that KL divergence may be less direct but it is simpler and widely used.

(D) **Why it measures what we claim**: The computational quantity `scores[i] = mean([rougel(r_i, r_j) for j != i])` measures **output consistency** because it quantifies the level of agreement among multiple stochastic passes; the assumption is that correct or high-quality outputs are more reproducible across different sampling trajectories. This assumption fails when the task admits multiple equally valid but semantically different correct answers (e.g., creative writing), in which case consistency reflects answer diversity rather than quality. The contrastive loss `loss = -mean(log π_θ(high_cons|x)) + mean(log π_θ(low_cons|x))` operationalizes **autonomous guidance** by increasing the proxy's likelihood of self-identified consistent outputs and decreasing it for inconsistent ones; the implicit assumption is that the proxy's own consistency evaluation is a valid proxy for external reward. This assumption fails when the proxy's sampling is not diverse enough (e.g., low temperature), causing all outputs to be similar regardless of quality, in which case the loss becomes uninformative. The signal transfer via KL divergence `minimize KL(π_φ || π_θ_updated)` measures the **transfer of self-evaluated guidance** because it aligns the primary model's distribution with the proxy's learned distribution that has been self-improved; the assumption is that the proxy's self-optimized distribution is a better target than its initial distribution. This assumption fails when the proxy is too weak (e.g., small model) and its self-improvement is negligible or misguided, in which case the primary model may regress.  
**Assumption**: High-quality outputs (e.g., correct reasoning) tend to be more consistent across different sampling conditions than low-quality ones. **Failure mode**: For weak or biased models, consistent outputs may be systematically wrong (e.g., identical errors), breaking the correlation. **Mitigation**: We compute the Pearson correlation between consistency scores and ground-truth correctness on a calibration set of 512 labeled examples; if correlation > 0.5, we proceed, otherwise we flag potential failure and adjust the threshold τ.

## Contribution

(1) A self-supervised proxy evaluation framework (SSPEG) that eliminates the need for an external reward model during guidance generation, using output consistency as an intrinsic reward. (2) The insight that consistency across multiple stochastic passes serves as a valid and general self-supervised signal for proxy learning, enabling autonomous exploration. (3) A demonstration of how self-evaluated proxy signals can be transferred to fine-tune a primary LLM via KL divergence minimization.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | GSM8K | Math reasoning with verifiable answers. |
| Primary metric | Accuracy | Direct measure of reasoning correctness. |
| Baseline 1 | PPO with gold reward | Standard RL with external reward. |
| Baseline 2 | SFT on 1000 gold | Supervised learning with limited labels. |
| Baseline 3 | DPO on self-consistency | Preferences from proxy without update. |
| Ablation of ours | SSPEG w/o contrastive | No proxy update, only initial consistency. |
| Resource estimate | 4×V100 (32GB), 1 week | Main experiment compute requirement. |

### Why this setup validates the claim

This combination of dataset, baselines, and metric enables a direct test of SSPEG's central claim: that self-consistency can serve as an autonomous reward for guidance. GSM8K provides clear correctness labels for gold reward baselines and objective accuracy measurement. PPO with gold reward tests whether external reward is necessary; if SSPEG approaches PPO performance, it demonstrates that intrinsic consistency can substitute. SFT on 1000 gold tests the label efficiency claim: SSPEG should outperform SFT because it leverages unsupervised consistency signals rather than requiring many labeled examples. DPO on self-consistency preferences (without proxy update) isolates the effect of the contrastive learning step; if SSPEG outperforms this baseline, it shows that updating the proxy via contrastive loss adds value beyond simply using consistency as preferences. The ablation (SSPEG w/o contrastive) tests the necessity of the proxy update step itself. Accuracy directly captures reasoning quality improvement, and the baseline comparisons isolate the contributions of each component. The calibration set of 512 labeled examples ensures that the consistency-quality correlation is verified before the main experiment, grounding the core assumption. If SSPEG fails to outperform these baselines on GSM8K, the claim is undermined.

### Expected outcome and causal chain

**vs. PPO with gold reward** — On a math problem where the primary model is uncertain, PPO uses a gold reward signal (correct answer) to reinforce the correct reasoning path. SSPEG, lacking gold labels, relies on self-consistency calibrated on a held-out set: it generates multiple answers and treats those that agree as high-consistency. This works because correct answers tend to be more reproducible than incorrect ones, a correlation we verify empirically on the calibration set. We expect SSPEG to achieve accuracy close to PPO (within 2-3%) on problems where correct answers are consistent across samples, but the gap may widen on problems where the correct answer is rare due to model bias.

**vs. SFT on 1000 gold** — On a problem outside the 1000 gold examples, SFT has no signal, so it may overfit to the limited training set. SSPEG, using unsupervised consistency on the problem itself, can adapt. Thus, we expect SSPEG to significantly outperform SFT (by >5% accuracy) on held-out problems, demonstrating that self-supervised guidance generalizes better than limited supervision.

**vs. DPO on self-consistency** — On a problem where proxy's consistency scores are noisy (e.g., multiple plausible answers), DPO treats the top-consistency response as preferred and pushes others away, which may force the model toward a single answer even when multiple are valid. SSPEG's contrastive loss updates the proxy to become more confident on consistent outputs, but the KL transfer to the primary model is softer. We expect SSPEG to perform similarly to DPO on unambiguous problems but better (by 1-2%) on ambiguous ones, because it avoids the collapse induced by rigid preference ranking.

### What would falsify this idea

If SSPEG performs no better than using the initial proxy's consistency without contrastive update (the ablation) across all subsets, then the contrastive update step is ineffective, falsifying the claim that self-supervised proxy evaluation improves guidance. Additionally, if the calibration set correlation between consistency and correctness is low (≤0.5) and yet SSPEG still works, the underlying assumption is not required, or if SSPEG fails when the correlation is high, the assumption is insufficient.

## References

1. Proxy Exploration and Reusable Guidance: A Modular LLM Post-Training Paradigm via Proxy-Guided Update Signals
2. Incentivizing Strong Reasoning from Weak Supervision
3. Weak-to-Strong Generalization beyond Accuracy: a Pilot Study in Safety, Toxicity, and Legal Reasoning
4. LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models
5. Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision
6. Discovering Language Model Behaviors with Model-Written Evaluations
7. Constitutional AI: Harmlessness from AI Feedback
8. Measuring Progress on Scalable Oversight for Large Language Models
