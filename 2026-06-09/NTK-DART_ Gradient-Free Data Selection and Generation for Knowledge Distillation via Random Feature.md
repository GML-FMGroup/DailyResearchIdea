# NTK-DART: Gradient-Free Data Selection and Generation for Knowledge Distillation via Random Feature Approximations of the Neural Tangent Kernel

## Motivation

Both data generation (e.g., DiffLM) and data selection (e.g., TAGCOS) for knowledge distillation of large language models are computationally prohibitive because they require full-gradient computations or complex generative models. TAGCOS, for instance, computes gradients for the entire dataset to cluster samples, which is infeasible for billion-parameter LLMs. This structural bottleneck stems from the reliance on exact gradient information for data curation, which cannot scale.

## Key Insight

Random feature approximations of the Neural Tangent Kernel allow us to compute gradient alignment between teacher and student using only forward passes, because the NTK inner product can be approximated by dot products of random features derived from the model's output, eliminating the need for backpropagation through the full model.

## Method

### (A) What it is
NTK-DART is a gradient-free method for selecting and generating a distillation dataset. It takes a teacher LLM, a student LLM, and an unlabeled pool, and outputs a curated dataset (selected and synthetic examples) that maximizes teacher-student alignment as measured by the NTK inner product approximated via random Fourier features (RFF).

### (B) How it works
```python
def ntk_dart(teacher, student, unlabeled_pool, budget, M=2048, sigma=1.0, lr=0.01, steps=10, vocab_embeddings=None):
    # Phase 1: Define random feature functions using RFF
    # w_i ~ N(0, sigma^2 * I), b_i ~ Uniform(0, 2pi) for i=1..M
    def phi(model, x):
        # model(x) returns logits (or penultimate layer)
        logits = model(x)
        features = []
        for i in range(M):
            proj = w_i @ logits.flatten() + b_i
            features.append(sqrt(2/M) * cos(proj))
        return tensor(features)

    # Phase 2: Select top-k data points by alignment score
    scores = []
    for x in unlabeled_pool:
        phi_t = phi(teacher, x)
        phi_s = phi(student, x)
        alignment = (phi_t @ phi_s) / (norm(phi_t) * norm(phi_s))
        scores.append(alignment)
    selected_indices = argsort(scores)[-budget_selection:]
    selected_data = [unlabeled_pool[i] for i in selected_indices]

    # Phase 3: Generate synthetic data by optimizing in input space
    # Input x is tokenized and embeddings are continuous; after optimization, project to nearest token via cosine distance
    synthetic_data = []
    for x in selected_data:
        x_adv = x.clone().requires_grad_(True)  # continuous embeddings
        for _ in range(steps):
            phi_t = phi(teacher, x_adv)
            phi_s = phi(student, x_adv)
            loss = - (phi_t @ phi_s)  # maximize alignment
            grad = autograd.grad(loss, x_adv)[0]
            x_adv = x_adv - lr * grad
        # Project each token to nearest embedding in vocab_embeddings
        x_projected = []
        for token_emb in x_adv:
            distances = torch.cdist(token_emb.unsqueeze(0), vocab_embeddings)
            nearest_idx = distances.argmin()
            x_projected.append(vocab_embeddings[nearest_idx])
        synthetic_data.append(torch.stack(x_projected).detach())
    
    # Phase 4: Combine and reweight by alignment (optional)
    distilled = selected_data + synthetic_data
    return distilled
```

### (C) Why this design
We chose random Fourier features (RFF) over exact NTK computation because RFF requires only forward passes and a fixed number of random projections (M), avoiding the O(d^3) cost of exact NTK while maintaining unbiased kernel approximation (error decays as 1/√M). This trade-off accepts approximation error but scales to billion-parameter models where full Jacobians are infeasible. We selected cosine features with random frequencies and biases instead of deterministic features (e.g., Nyström) because they provide unbiased kernel estimates with smaller memory footprint. We opted to generate synthetic data by gradient descent on the input to maximize alignment rather than training a generative model like DiffLM, because it avoids auxiliary model overhead and directly targets the alignment objective, though it may produce unnatural inputs if steps are too many or learning rate is too high. We prioritized selection before generation because high-alignment seeds constrain the search space, reducing both risk of divergence and compute cost. Finally, we used a fixed budget split (e.g., 70% selected, 30% generated) rather than dynamic allocation to avoid an extra hyperparameter, though this may reduce efficiency when the two phases have unequal returns.

**Load-bearing assumption:** Random Fourier features of model outputs approximate the Neural Tangent Kernel inner product between teacher and student gradients. This is not mathematically exact (NTK is defined on parameter gradients), so we empirically verify its effectiveness through calibration experiments.

### (D) Why it measures what we claim
**Alignment score** `phi_t @ phi_s / (||phi_t|| ||phi_s||)` measures gradient alignment under the assumption that the RFF kernel converges to the true NTK. This assumption fails when M is small, leading to biased kernel that misranks examples. To diagnose, we vary M = {256, 512, 1024, 2048} and compare alignment rankings with exact NTK on a small model (GPT-2); if correlation drops below 0.8 for M < 1024, we warn. **Gradient of `-phi_t @ phi_s` w.r.t. input x** measures the direction to maximize this alignment, equivalent to moving x toward regions where teacher and student features coincide; this assumes that the feature functions are differentiable and that the input space is continuous (we treat token embeddings as continuous). **Cosine normalization** measures the angle between feature vectors, isolating alignment from magnitude effects; this assumes that feature magnitude does not carry alignment information; if feature magnitude correlates with data difficulty, the normalized score may be insensitive to it.

## Contribution

(1) NTK-DART, a gradient-free data selection and generation method for knowledge distillation that uses random Fourier features to approximate the NTK alignment score, enabling scalable distillation for large LLMs. (2) A demonstration that alignment measured by RFF is a effective proxy for full-gradient alignment, allowing efficient data curation without backpropagation through the full model. (3) An analysis of the trade-off between the number of random features and estimation accuracy, providing practical guidance for choosing M.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Alpaca 52K | Common instruction tuning dataset |
| Primary metric | MMLU accuracy | Measures generalization to diverse tasks |
| Baseline 1 | Random sampling | Default naive selection baseline |
| Baseline 2 | K-center greedy | Maximizes diversity in subset |
| Baseline 3 | Perplexity-based | Selects high-difficulty examples |
| Ablation 1 | Selection-only (no generation) | Tests necessity of synthetic generation |
| Ablation 2 | Exact gradient alignment (small model) | Quantifies RFF approximation error on GPT-2 |
| Ablation 3 | Vary M (256,512,1024,2048) | Tests sensitivity to kernel convergence |
| Scale-up | LLaMA-7B teacher & student | Demonstrates practical scaling advantage |

### Why this setup validates the claim

The combination of Alpaca 52K as a representative instruction-tuning dataset and MMLU accuracy as a rigorous multi-task benchmark provides a strong test of whether our gradient-free alignment-based dataset distillation improves knowledge transfer beyond simple diversity or difficulty heuristics. Random sampling establishes a lower bound, while K-center greedy and perplexity-based selection represent two principled but alignment-agnostic strategies—diversity and difficulty, respectively. The ablation (selection-only) isolates the role of synthetic generation. Crucially, Ablation 2 directly tests the load-bearing assumption by comparing RFF-based alignment to exact gradient-based alignment (computed via full Jacobians) on a small GPT-2 model (117M params) on a 10% subset of Alpaca; we report the Spearman rank correlation between the two alignment scores. Ablation 3 varies the number of random features M to empirically verify convergence behavior. The scale-up experiment uses a 7B teacher and student (e.g., LLaMA-7B) to demonstrate the gradient-free advantage, as exact gradient methods would be infeasible. MMLU accuracy is chosen because it reflects broad knowledge retention, and if alignment truly enhances distillation, our method should yield higher accuracy than baselines that ignore teacher-student gradient similarity.

### Expected outcome and causal chain

**vs. Random sampling** — On a case where random selection includes many examples on which teacher and student disagree (e.g., ambiguous math problems), the baseline produces poor student generalization because it learns conflicting signals. Our method instead selects only high-alignment examples from the pool, so we expect a noticeable accuracy gap (e.g., +5% on MMLU) as the student better captures teacher knowledge.

**vs. K-center greedy** — On a case where diversity-driven selection picks examples from disparate clusters but with low teacher-student alignment per cluster, the baseline may force the student to memorize conflicting patterns. Our method prioritizes alignment over diversity, thus we expect similar performance on easy clusters but a clear improvement on clusters where teacher-student divergence is high (e.g., +3-4% overall).

**vs. Perplexity-based** — On a case where high-perplexity examples (e.g., long reasoning chains) cause teacher and student to produce very different logits, the baseline amplifies those differences, harming student calibration. Our method avoids such examples by filtering on alignment, so we expect our method to have better accuracy on hard, high-perplexity subsets (e.g., +2-3% on complex reasoning tasks).

**Ablation 2 (Exact gradient alignment on small model)** — We expect a high Spearman correlation (≥0.8) between RFF-based and exact gradient-based alignment scores when M ≥ 1024, confirming that RFF is a faithful proxy. If correlation is lower, we will increase M or revise the kernel approximation.

**Ablation 3 (Vary M)** — We expect MMLU accuracy to increase with M and saturate around M=1024, indicating kernel convergence. If accuracy plateaus earlier or drops, the approximation may be insufficient.

### What would falsify this idea

If NTK-DART fails to outperform random sampling on MMLU despite having higher alignment scores on the selected subset, then the central claim that alignment-based selection improves KD is wrong—the alignment score may be a poor proxy for downstream transfer. Alternatively, if our method only improves on subsets where the baseline already does well, but fails on the hard cases where alignment divergence is predicted, then the mechanism is not causal.

## References

1. Knowledge distillation and dataset distillation of large language models: emerging trends, challenges, and future directions
2. TAGCOS: Task-agnostic Gradient Clustered Coreset Selection for Instruction Tuning Data
3. A Comprehensive Survey of Small Language Models in the Era of Large Language Models: Techniques, Enhancements, Applications, Collaboration with LLMs, and Trustworthiness
4. From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning
5. The Internal State of an LLM Knows When its Lying
6. MCC-KD: Multi-CoT Consistent Knowledge Distillation
7. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
8. Self-Knowledge Guided Retrieval Augmentation for Large Language Models
