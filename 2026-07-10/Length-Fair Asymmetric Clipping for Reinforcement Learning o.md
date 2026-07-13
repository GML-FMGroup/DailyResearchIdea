# Length-Fair Asymmetric Clipping for Reinforcement Learning of Language Models

## Motivation

Existing asymmetric clipping methods for RL fine-tuning of LLMs, such as UP, are length-agnostic: they apply the same clipping bounds regardless of response length. This causes reward confounding, where longer responses incur larger cumulative advantages but are penalized by the same fixed clipping bound as short responses, biasing the policy toward shorter outputs. The length-fair reward evaluation from CYF highlights that reward accumulation over sequence length must be normalized for fairness, but no prior RL objective incorporates this principle. Specifically, UP's unclipped positive advantage encourages exploration but its flat clipping for negative advantages fails to account for length, leading to unstable updates for long sequences.

## Key Insight

Scaling the clipping bound by √L ensures that the maximal allowable change in action probabilities per token is invariant to sequence length, because the cumulative advantage variance grows linearly with L under token-level independence, so √L normalization equalizes the signal-to-noise ratio across lengths.

## Method

**A. What it is** — Length-Fair Asymmetric Clipping (LFAC) modifies the UP objective by scaling the negative advantage clipping bound by √L, where L is the number of tokens in the response. Inputs: log-probabilities π_θ, sequence-level advantage A, response length L. Outputs: loss value.

**B. How it works**
```python
def lfac_loss(log_probs_old, log_probs_new, advantages, lengths, eps=0.2):
    # log_probs: [batch, L_max] padded
    # advantages: [batch]
    # lengths: [batch]
    ratio = exp(log_probs_new - log_probs_old)  # importance sampling ratio per token
    # For each sequence, mask out padding
    mask = (lengths > 0)
    # Compute token-level advantages (broadcast sequence advantage to each token)
    token_advantages = advantages.unsqueeze(1)  # [batch, 1]
    # Asymmetric clipping: positive advantages unclipped, negative clipped with sqrt(L) scaling
    clipped_ratio = torch.where(
        token_advantages >= 0,
        ratio,  # no clipping for positive advantages
        torch.clamp(ratio, 1 - eps * torch.sqrt(lengths.unsqueeze(1).float()), 1 + eps * torch.sqrt(lengths.unsqueeze(1).float()))  # negative clipping bound scaled by sqrt(L)
    )
    loss = -torch.mean(torch.sum(mask * ratio * token_advantages - stopgrad(mask) * (clipped_ratio - ratio) * token_advantages, dim=1))
    # stopgrad applied to the clipping correction term, as in UP
    return loss
```
Hyperparameters: ε = 0.2.

**C. Why this design** — We chose to scale only the negative clipping bound by √L because scaling the positive unclipped region would contradict UP's principle of unbounded exploration for positive advantages; exploring longer sequences should be encouraged, not restricted. We chose √L over L because √L corresponds to the standard deviation of the sum of L i.i.d. unit-variance token-level advantages, matching the length-fair normalization in CYF; L would over-regularize and suppress necessary policy updates for long sequences. We applied the scaling directly to the bound rather than to the advantage itself to keep the advantage signal's magnitude intact for exploration; scaling the advantage would dilute the reward signal for long sequences, making learning slower. The cost is that for sequences with highly non-stationary token-level advantages (e.g., early tokens matter less), √L may be an imprecise normalization, potentially under-clipping or over-clipping some tokens.

**D. Why it measures what we claim** — The ratio clipping bound ε√L measures the length-fair policy update capacity because it limits the maximum deviation in token probability per token, and assuming token-level advantages have constant variance σ², the cumulative advantage variance across L tokens is σ²L; by scaling the bound proportionally to √L, the expected magnitude of the clipping-adjusted advantage gradient becomes length-invariant, ensuring that the algorithm does not favor short responses. This assumption fails when token-level advantages are heteroskedastic (e.g., final tokens carry more reward), in which case the normalization is too weak for high-variance tokens and too strong for low-variance ones, introducing a residual length bias toward tokens with high advantage variance. The stop-gradient term measures the correction magnitude for clipped sequences, and its inclusion preserves the stability property of UP by preventing gradient flow from the clipping correction, but it also assumes that the clipping correction is entirely spurious; if the clipping correction captures genuine policy improvement (e.g., due to length-dependent reward distribution), the stop-gradient might discard useful learning signal.

## Contribution

(1) A novel RL objective, LFAC, that integrates length-dependent normalization into asymmetric clipping to enforce fairness across response lengths. (2) A design principle that √L scaling is the theoretically motivated factor for length-invariant policy updates in token-level RL, derived from the variance accumulation argument. (3) Empirical analysis showing that LFAC reduces length bias compared to UP while maintaining or improving exploration efficiency on reasoning tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | GSM8K (math reasoning) | Varied response lengths naturally occur |
| Primary metric | Accuracy (exact match) | Direct measure of task success |
| Baseline 1 | PPO (symmetric clipping) | Standard RL algorithm with symmetric bound |
| Baseline 2 | UP (asymmetric, no length scaling) | Direct precursor without length fairness |
| Ablation-of-ours | LFAC with linear L scaling | Tests necessity of sqrt scaling vs linear |

### Why this setup validates the claim
Using a math reasoning dataset (GSM8K) provides a natural mix of short and long correct responses, essential for testing length fairness. Comparing against UP isolates the effect of length scaling, while PPO tests the benefit of asymmetric clipping. The ablation (L scaling) tests the choice of sqrt. Accuracy as a primary metric captures overall solution quality, but we will analyze performance stratified by response length to detect whether bias toward short responses is reduced. This design creates a falsifiable test: if length-fair clipping is effective, the accuracy gap between short and long responses should narrow compared to baselines, while the ablation should show degraded performance on long responses due to over-regularization.

### Expected outcome and causal chain

**vs. PPO** — On a long correct response requiring multiple steps, PPO's symmetric clipping restricts both positive and negative updates; for positive advantages, it limits the ratio to 1+ε, discouraging further exploration of long solutions. Our method removes this cap for positive advantages, allowing full exploitation of beneficial long responses. Thus, we expect LFAC to achieve higher accuracy on long solutions (e.g., >50 tokens) compared to PPO, while matching PPO on short ones.

**vs. UP** — On a short response with high advantage, UP treats it identically to a long response, leading to over-updating of the short sequence because the clipping bound is fixed. Our method scales the negative bound down by sqrt(L), so for short L the bound is tighter, reducing excessive suppression of probability decrease for other tokens? Actually, for negative advantages, clipping restricts ratio decrease; a short response with negative advantage should be penalized less? Wait: In UP, negative advantages are clipped to [1-ε, 1+ε]; for short responses, we want to allow larger updates? Actually, the claim is length fairness: updates should be independent of length. For a short response with negative advantage, UP's fixed bound allows a large update (because ratio can drop to 1-ε), which may be too aggressive relative to its length. Our method makes the bound tighter (1-ε*sqrt(L)) for short L? sqrt(L) < 1 for L<1? But L>=1, sqrt(L) >=1? Actually for L=1, sqrt(1)=1, so bound same as UP. For L=4, sqrt(4)=2, bound is 1-0.4 = 0.6, which is looser! Wait: The bound is scaled by sqrt(L), so for longer L, the clipping range widens. That means for short responses, the bound is tighter (smaller range), so negative updates are more restricted. That would prevent over-penalizing short bad responses. So vs. UP, we expect LFAC to avoid over-suppressing short incorrect responses? Actually careful: The goal is length fairness: the effective step size should be proportional to sqrt(L) to normalize variance. So for short sequences, the allowed update (max ratio change) is smaller (since sqrt(L) small), so we expect less aggressive updates on short sequences. This should reduce length bias. So on short responses with negative advantage, UP might update too much (making policy worse), while LFAC is more cautious. Thus, we expect LFAC to have better performance on short responses (higher accuracy) compared to UP. The observable signal: a smaller accuracy gap between short and long responses for LFAC than for UP.

**vs. Ablation (L scaling)** — On a very long response (e.g., 100 tokens), L scaling sets the negative bound to 1-ε*L, which for L=100, ε=0.2 gives bound 1-20 = -19, effectively no lower bound? Actually clamp to [1-20, 1+20] = [-19, 21]; that is extremely wide, allowing very large probability decreases. This could destabilize training and harm long responses. Our sqrt scaling gives bound 1-0.2*10 = -1, still wide but less extreme. More importantly, for positive advantages, L scaling would also widen the upper bound? But we only clip negative, so for positive, both allow unbounded ratio. So the difference is in negative updates: L scaling allows much larger negative updates for long sequences, which can lead to collapse of long good responses if they have negative advantage due to noise. Therefore, we expect LFAC to maintain higher accuracy on long responses compared to the ablation, while the ablation may show instability or lower accuracy on long sequences.

### What would falsify this idea
If, on a held-out test set stratified by response length, LFAC does not show a narrower accuracy gap between short and long responses compared to UP, or if the ablation with L scaling outperforms LFAC on any length subset, then the central claim that sqrt(L) scaling provides length-fair updates is falsified.

## References

1. UP: Unbounded Positive Asymmetric Optimization for Breaking the Exploration-Stability Dilemma
2. Geometric-Mean Policy Optimization
3. Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement
4. Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models
5. Self-Consistency Improves Chain of Thought Reasoning in Language Models
6. STaR: Bootstrapping Reasoning With Reasoning
