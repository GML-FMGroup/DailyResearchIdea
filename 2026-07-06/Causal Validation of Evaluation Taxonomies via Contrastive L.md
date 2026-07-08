# Causal Validation of Evaluation Taxonomies via Contrastive Latent Intervention

## Motivation

Existing taxonomies for evaluating research ideas, such as the two-axis taxonomy used in 'Measuring the Gap Between Human and LLM Research Ideas', may miss important dimensions, leading to biased assessments. The root cause is that taxonomies are derived from observed data and may be overfit to salient patterns, ignoring latent dimensions that causally influence idea quality. Current validation methods rely on static data and cannot guarantee coverage of all relevant evaluation aspects.

## Key Insight

If a taxonomy is complete, then a generative model conditioned on its dimensions should leave no residual variance that systematically affects evaluation; any systematic residual variance indicates a missing causal dimension.

## Method

```python
def CLITV(ideas, taxonomy_labels, eval_scores, n_known, n_unknown=5, lambda_contrast=1.0):
    # Step 1: Train CVAE with encoder q(z_known, z_unknown | idea) and decoder p(idea | z_known, z_unknown)
    # Encoder: two separate 2-layer MLPs (hidden=256, ReLU) outputting mean and log_var for each latent group (z_known: dim=n_known, z_unknown: dim=n_unknown)
    # Decoder: 2-layer MLP (hidden=256, ReLU) outputting parameters of Gaussian over idea embeddings (e.g., 768-d from Sentence-BERT)
    # Contrastive loss: For pairs with same taxonomy_labels, enforce z_known similar via MSE (weight lambda_contrast=1.0); no constraint on z_unknown.
    # Training: 100 epochs, batch size 64, Adam lr=1e-3.
    # Step 2: Sample a set of ideas. For each idea, obtain z_known and z_unknown from encoder (use mean, not sample).
    # Step 3: Intervention: fix z_known, set z_unknown to its mean over all ideas (precomputed from training set).
    # Step 4: Decode interventional ideas (using decoder mean). Compute evaluation scores using a fixed evaluator (e.g., human or LLM).
    # Step 5: Compare original and interventional scores via paired t-test and Cohen's d.
    # Step 6: If p < 0.05 and |d| > 0.2, flag missing dimension.
    # Post-training validation: On a synthetic calibration set (512 examples) with ground truth missing dimensions, compute mutual information between each true missing dim and each z_unknown component. If average MI < 0.5 bits, warn of insufficient disentanglement.
    return p_value, effect_size
```

(B) **How it works**: (see pseudocode above)

(C) **Why this design**: We chose a conditional VAE over a plain autoencoder because the structured latent space with known vs. unknown factors allows explicit separation for intervention, accepting that training such a model may be unstable with low data. We used contrastive training to force known latents to capture shared taxonomy information, at the cost that if the taxonomy is weak, unknown latents may also capture shared information, reducing detection power. We opted for a fixed reference intervention (mean of unknown latent) instead of random sampling to isolate the causal effect from stochasticity, though this reduces the ability to detect non-linear effects. We used a simple paired t-test rather than a more complex causal model (e.g., double ML) to maintain interpretability, assuming no hidden confounding; this assumption may be violated if evaluation confounders correlate with unknown latents. Hyperparameters: n_unknown=5, latent dims 256, batch 64, epochs 100, lr 1e-3.

(D) **Why it measures what we claim**: The `interventional evaluation difference` measures the causal necessity of the unknown latent for evaluation because we hold the known taxonomy dimensions fixed and only vary the unknown latent; this assumes that the known dimensions are correctly identified and that the decoder faithfully generates ideas from the latents. This assumption fails when the known dimensions are not causally sufficient (i.e., there are confounders affecting both known and unknown latents), in which case the intervention difference may be biased. The `paired t-test p-value` measures the statistical significance of the effect, assuming normality and independence of evaluation scores; this assumption fails when scores are correlated across ideas (e.g., due to shared evaluator bias), inflating false positives. The `effect size (Cohen's d)` measures the magnitude of the missing dimension's influence, assuming a linear scale; this assumption fails when the effect is non-linear or discontinuous, in which case the effect size may underestimate the true causal impact.

## Contribution

(1) A causal validation framework (CLITV) that uses contrastive latent intervention to identify missing evaluation dimensions in research idea taxonomies. (2) Empirical evidence that artificial gaps in taxonomy dimensions can be detected via residual latent variation, demonstrating the method's sensitivity. (3) A protocol for discovering novel evaluation criteria from latent interventions, enabling iterative taxonomy refinement.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | Synthetic research ideas with ground truth missing dims | Controlled test of detection |
| Dataset | Real-world research ideas from IDEA Dataset (expert-annotated missing dimensions) | Demonstrates practical significance |
| Primary metric | F1 score for missing dimension detection | Measures accuracy in flagging |
| Baseline 1 | Known-dimension regression | Tests if known dims alone suffice |
| Baseline 2 | Random intervention on known dims | Controls for spurious effects |
| Ablation of ours | CVAE without contrastive loss | Isolates contrastive loss benefit |

### Why this setup validates the claim
We use both synthetic data with known ground truth missing dimensions and real-world data with expert annotations to provide a clean causal test and demonstrate practical utility. If our method correctly identifies missing dimensions more often than baselines (high F1), it confirms that separating known and unknown latent factors via contrastive CVAE and intervening on the unknown latent effectively reveals causal gaps. The baselines test alternative explanations: known-dim regression checks if evaluation is fully predictable from observed taxonomy; random intervention checks if any perturbation would produce similar effects. Our ablation tests whether contrastive loss is critical for aligning known latents. Rejecting these alternatives while our method succeeds would provide strong evidence for the causal necessity claim.

### Expected outcome and causal chain

**vs. Known-dimension regression** — On a case where a missing dimension strongly influences evaluation, known-dim regression will fail to capture that variance, producing a high residual but no way to pinpoint the cause. Our method, by explicitly modeling unknown latents and intervening, can isolate the effect and detect significance. So we expect a large gap in F1 on instances with true missing dimensions, but parity on instances without missing dimensions.

**vs. Random intervention** — On a case where known dimensions are confounded with missing dimensions, random intervention on known dims might produce spurious effects. Our method holds known dims fixed and only varies the unknown latent, which isolates the causal effect. Thus we expect our method to have higher specificity (fewer false positives) on confounded data.

**vs. our ablation** — Removing contrastive loss allows known latents to also capture irrelevant variation, weakening the intervention's ability to detect missing dimensions. So our full method should show higher F1 especially when known dimensions are highly correlated with unknown ones.

On real-world data, we expect our method to detect known missing dimensions with high F1, while baselines may fail due to confounding or incomplete observed taxonomy.

### What would falsify this idea
If our method's detection performance is not significantly better than known-dim regression on instances where missing dimensions are known to exist (per ground truth), then the central claim that the intervention isolates causal necessity is false. Alternatively, if the ablation performs equally well, then contrastive loss is not necessary.

## References

1. Measuring the Gap Between Human and LLM Research Ideas
2. IdeaBench: Benchmarking Large Language Models for Research Idea Generation
3. A Comprehensive Analysis of Large Language Model Outputs: Similarity, Diversity, and Bias
4. The Impact of Large Language Models on Scientific Discovery: a Preliminary Study using GPT-4
5. Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers
