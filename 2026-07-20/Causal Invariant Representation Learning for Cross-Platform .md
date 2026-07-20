# Causal Invariant Representation Learning for Cross-Platform Code Review Automation

## Motivation

Existing code review automation models are trained predominantly on GitHub data and fail to generalize to enterprise platforms such as Gerrit or GitLab due to differences in UI constraints and workflows. For instance, the study 'From Human-Centric to Agentic Code Review' analyzes over a million GitHub pull requests but explicitly notes its limitation to GitHub projects, which may not represent all software development contexts. The root cause is that models learn platform-specific spurious correlations rather than the underlying causal structure of review decisions, which is invariant across platforms.

## Key Insight

The causal graph of review outcomes (approve or request changes) is structurally identical across platforms after conditioning on a small set of universal latent variables (e.g., code complexity, reviewer expertise), enabling learned representations that transfer via invariance of the conditional outcome distribution.

## Method

**CIRL-CR: Causal Invariant Representation Learning for Code Review**

(A) **What it is:** A framework that learns platform-invariant representations of review interaction sequences by aligning conditional outcome distributions across platforms via a maximum mean discrepancy (MMD) penalty. Input: sequences of review interactions (comments, changes, approvals) from multiple platforms. Output: a representation that generalizes to unseen platforms.

(B) **How it works (pseudocode):**
```python
# CIRL-CR
Input: 
  - D_train = {(x_i, y_i, p_i)} where x_i is interaction sequence, y_i ∈ {approve, request_changes}, p_i is platform label
Hyperparameters:
  - λ (MMD weight) = 1.0
  - latent_dim = 10
  - kernel: RBF with σ = 1.0
  - batch_size = 64
  - learning_rate = 1e-3 (Adam optimizer)
  - num_epochs = 100
  - encoder: 2-layer MLP, hidden=256, ReLU activation, output μ_z and log σ_z (both dim=10)
  - decoder: 2-layer MLP, hidden=256, ReLU activation, output reconstruction x̂
  - predictor f_ψ: 2-layer MLP, hidden=128 → 1, sigmoid activation
  - VAE prior: standard Gaussian N(0, I)

# Encoder: q_φ(z|x) → Gaussian parameters μ_z, σ_z
# Decoder: p_θ(x|z) → reconstruction
# Predictor: f_ψ(z) → probability of approval

for each epoch:
  for each batch:
    1. Sample z ~ q_φ(z|x) for each (x, p) using reparameterization.
    2. L_VAE = -ELBO(x, z) = reconstruction loss (MSE) + KL divergence (weighted by β=1.0).
    3. L_pred = cross_entropy(y, f_ψ(z)).
    4. For each pair of platforms (p_a, p_b) in batch:
        - Compute conditional outcome distributions: g_a(y) = f_ψ(z_a), g_b(y) = f_ψ(z_b).
        - Compute MMD^2 = E[k(g_a,g_a)] + E[k(g_b,g_b)] - 2E[k(g_a,g_b)] where k is RBF kernel with σ=1.0.
    5. L = L_VAE + L_pred + λ * MMD.
    6. Update φ, θ, ψ jointly.
```

(C) **Why this design:** We chose a VAE-based encoder over a deterministic one because the latent variables (code complexity, reviewer expertise) are inherently unobserved and require uncertainty quantification; the trade-off is that VAE training can be unstable and requires careful tuning of the KL weight. We chose MMD over adversarial domain discrimination because MMD provides a differentiable closed-form alignment of distributions without requiring a separate discriminator network, which avoids min-max optimization issues; the cost is that MMD may be less effective for high-dimensional distributions. We chose to align conditional outcome distributions (p(y|z)) rather than marginal latent distributions (p(z)) because causal invariance requires that the effect of z on y is stable across platforms, not that the marginal distributions of z are identical; this design is more computationally expensive as it requires sampling multiple outcomes per batch, but it directly targets the causal condition. Unlike domain-adversarial methods (Ganin et al., 2016) that align feature distributions without considering causal structure, our approach directly exploits the known invariance of the outcome mechanism.

(D) **Why it measures what we claim:** The MMD penalty on conditional outcome distributions measures causal invariance because it ensures that the probability of approval given the latent variables z is the same across platforms; this assumes that z captures the true causal parents of y and that the causal mechanism p(y|z) is invariant. This assumption fails when z does not include all confounders that vary across platforms (e.g., organizational culture), in which case the MMD penalty may still align distributions by learning spurious features that correlate with both z and platform. The VAE reconstruction loss measures the informativeness of z about the interaction sequence x; it operationalizes the requirement that z must compress the interactions sufficiently to predict outcomes, assuming the decoder can reconstruct x from z. This assumption fails when the decoder is too powerful, allowing z to be uninformative, in which case reconstruction loss reflects decoder capacity rather than latent sufficiency. **Explicit load-bearing assumption:** The latent variables z learned by the VAE encoder capture all causal parents of review outcome y, and the conditional distribution p(y|z) is invariant across different code review platforms. **Failure mode:** If unobserved platform-specific policies (e.g., mandatory two-reviewer rule on Gerrit) affect y independently of z, then p(y|z) is not invariant, and MMD alignment may produce biased representations. To verify this assumption, we conduct a controlled synthetic experiment (see experiment section) where the true causal graph is known and measure whether MMD recovers the invariant predictor.

## Contribution

(1) A novel framework CIRL-CR that learns platform-invariant representations of code review interactions by aligning conditional outcome distributions via MMD, leveraging structural causal invariance. (2) A design principle: aligning conditional outcome distributions rather than marginal latent distributions is necessary for cross-platform generalization in causal settings. (3) A synthetic benchmark for evaluating cross-platform code review models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | Multi-platform code review corpus (GitHub: 10k, Gerrit: 5k, GitLab: 5k) | Covers diverse review interaction patterns across platforms |
| Primary metric | Cross-platform macro F1 (average per-platform F1, macro across platforms) | Tests generalization to unseen platforms |
| Baseline 1 | Domain-adversarial network (DANN) with same encoder as CIRL-CR + domain classifier (2-layer MLP, hidden=256, gradient reversal) | Aligns features adversarially, no causal structure |
| Baseline 2 | VAE + classifier (no MMD) – same VAE as CIRL-CR, predictor directly from z | Ablates invariance penalty |
| Baseline 3 | Logistic regression on TF-IDF bag-of-words (unigrams + bigrams, max features=5000) | Simple non-neural baseline |
| Ablation | CIRL-CR without MMD (identical to Baseline 2) | Isolates contribution of MMD alignment (note: same as Baseline 2) |

### Why this setup validates the claim
This combination of dataset, baselines, and metric directly tests the central claim that aligning conditional outcome distributions across platforms improves generalization to unseen platforms. The multi-platform dataset ensures that domain shifts (e.g., different comment styles, approval norms) exist. The primary metric, cross-platform macro F1, evaluates performance on held-out platforms, directly measuring invariance. Comparing against DANN tests whether causal invariance (aligning p(y|z)) outperforms domain-adversarial alignment of p(z) when the true causal mechanism is invariant. Comparison with VAE without MMD tests the specific contribution of the MMD penalty. The ablation further isolates the effect of MMD. Logistic regression provides a simple reference. Additionally, we design a controlled synthetic experiment where the true causal graph is known: we simulate review interactions with latent variables (code complexity, reviewer expertise) that causally determine approval, plus a platform-specific confounder (e.g., base approval rate) that affects both interaction patterns and outcomes. We vary the confounder strength across three synthetic platforms and measure whether CIRL-CR recovers the invariant causal predictor more accurately than baselines. This synthetic experiment validates that MMD alignment captures causal invariance under known conditions. If CIRL-CR outperforms DANN on platform pairs where the outcome mechanism is stable but feature distributions differ, the causal assumption is validated. A falsifying outcome would be if CIRL-CR performs no better than the ablation or DANN on all subsets, indicating the MMD penalty does not capture causal invariance.

### Expected outcome and causal chain

**vs. Domain-adversarial network (DANN)** — On a case where platform-specific norms affect reviewer behavior (e.g., GitHub favors quick approvals while Gerrit requires more iterations), DANN aligns latent features across platforms but ignores that the approval mechanism may differ; it may learn spurious correlations that fail on a third platform. Our method instead aligns the conditional outcome distributions directly, forcing the predictor to rely on features that cause approval consistently. We expect a noticeable gap (e.g., 5-10% macro F1) on platforms with different base rates or interaction styles, with parity on similar platforms. In the synthetic experiment, we expect CIRL-CR to recover the true causal predictor with near-zero error on the invariant part, while DANN will exhibit bias proportional to confounder strength.

**vs. VAE + classifier (no MMD)** — On a case where a subtle confounder (e.g., project language) correlates with platform, VAE+classifier may learn to predict approval using that confounder rather than review content, causing failure when the confounder shifts. Our method penalizes differences in p(y|z) across platforms, preventing reliance on such confounders. We expect a clear advantage (e.g., 5-8% macro F1) on platform pairs with correlated confounders, and similar performance when no confounders exist. In the synthetic experiment, VAE+classifier will perform well within each platform but poorly on held-out platforms when the confounder distribution changes.

**vs. Logistic regression on bag-of-words** — On a case where interaction sequences involve complex temporal patterns (e.g., multiple comment rounds before approval), logistic regression misses temporal structure, leading to poor performance even on the source platform. Our method captures sequences via VAE, so we expect a large absolute gap (e.g., 15-20% macro F1) on all platforms, especially those with rich interaction patterns.

### What would falsify this idea
If CIRL-CR’s performance on held-out platforms is not consistently higher than the VAE+classifier ablation (e.g., the gap is less than 2%), or if the improvement is uniform across all subsets rather than concentrated on platform pairs where causal invariance is explicitly needed (e.g., greater improvement when domain shift is severe), then the central claim that MMD alignment captures causal invariance is falsified. Additionally, if in the synthetic experiment CIRL-CR fails to recover the true invariant predictor (e.g., its predictor weights on non-causal features are more than 0.1), the load-bearing assumption is violated.

## References

1. From Human-Centric to Agentic Code Review: The Impact of Different Generations of Generative AI Technology on Review Quality
2. Developer-LLM Conversations: An Empirical Study of Interactions and Generated Code Quality
3. On the Use of Agentic Coding: An Empirical Study of Pull Requests on GitHub
4. A Performance Study of LLM-Generated Code on Leetcode
5. On the Taxonomy of Developers’ Discussion Topics with ChatGPT
6. WildChat: 1M ChatGPT Interaction Logs in the Wild
7. LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset
