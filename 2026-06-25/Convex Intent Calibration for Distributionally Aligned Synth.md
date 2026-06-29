# Convex Intent Calibration for Distributionally Aligned Synthetic Data Generation

## Motivation

Existing synthetic data generation methods, such as Autodata, meta-optimize a data scientist agent to produce high-quality data, but they assume synthetic data alone can substitute for real user data in training. This assumption fails because LLM-generated synthetic data often exhibits distributional biases that do not reflect real user intents. While Autodata uses a validation set for meta-objective, it does not explicitly align the synthetic data distribution with the real user intent distribution, leading to downstream models that perform well on synthetic validation sets but poorly on real user inputs. We need a method that leverages a minimal set of real user interactions to correct this distributional misalignment without sacrificing scalability.

## Key Insight

A convex calibration of the LLM's output logits, learned from a small set of real user interactions, suffices to correct the distributional bias because the real intents lie on a low-dimensional manifold that can be captured by a linear correction in logit space.

## Method

We propose the Intent Calibration Module (ICM), a lightweight convex module that revises an LLM's output logits using a minimal set of real user interactions.

(A) **What it is**: ICM takes as input a base LLM (for synthetic data generation) and a small set of real user queries `Q_real` with associated target outputs or intent distributions. It outputs a corrected logit function that is applied during synthetic data generation to align with real user intents.

(B) **How it works**:
```python
# Input: Base LLM, set of real user queries Q_real with target outputs Y_real (e.g., next-token distributions or response texts)
# Hyperparameters:  regularization coefficient λ=0.1, embedding dimension d=4096
# Step 1: For each query q in Q_real, compute base LLM logits l_base(q) over vocabulary V.
# Step 2: Compute query embeddings phi(q) using a pretrained encoder (e.g., last hidden layer of the LLM).
# Step 3: Learn linear correction matrix W (size |V| x d) by minimizing convex objective:
#   min_W Σ_{q∈Q_real} KL( softmax(l_base(q) + W·phi(q)) || target_distribution(y|q) ) + λ*||W||_F^2
#   Solve via gradient descent (e.g., Adam for 50 steps) – convex in W due to linearity.
# Step 4: During synthetic data generation, for any new query q', sample from softmax(l_base(q') + W·phi(q')).
# Output: Calibrated LLM for synthetic data generation.
```

(C) **Why this design**: We chose a linear correction in logit space over full fine-tuning because it is lightweight and avoids overfitting to the small real dataset; the convex optimization guarantees convergence with few samples. Using KL divergence as the objective directly measures distributional alignment, and L2 regularization prevents the correction from being too large. We use query embeddings as features to capture context without requiring additional annotation. The trade-off is that linearity may struggle with highly nonlinear biases, but the convexity ensures robustness. We considered a nonlinear neural network but rejected it due to sample inefficiency and risk of overfitting with few interactions. The optimization is decoupled from the base LLM, so it does not require backpropagation through the model, reducing compute.

(D) **Why it measures what we claim**: The term `W·phi(q)` measures the deviation of the base LLM's distribution from the real intent distribution in a low-dimensional subspace defined by the query embedding. The assumption is that the real intent distribution is linearly separable from the base distribution in this subspace; this assumption fails when the bias is highly nonlinear or when phi(q) does not capture relevant variation (e.g., rare intents). In such cases, the correction may only partially align, and the synthetic data may still exhibit residual bias. The KL divergence term measures the alignment of the corrected distribution to the real intents, under the assumption that the target distribution is known; this assumption fails when the real interactions are sparse or noisy, in which case KL may reflect overfitting to noise rather than true intent alignment.

## Contribution

(1) A lightweight convex intent calibration module that adjusts LLM output logits using a minimal set of real user interactions, enabling distributionally aligned synthetic data generation without fine-tuning. (2) Empirical demonstration that as few as 100 real interactions suffice to correct distributional bias in synthetic data, as measured by downstream performance on held-out real user queries. (3) An analysis of the trade-off between the number of real interactions and the quality of alignment, showing diminishing returns beyond a few hundred samples.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | Multi-task synthetic data generation tasks | From Autodata benchmark covering diverse intents |
| Primary metric | KL divergence on held-out real queries | Directly measures distributional alignment |
| Baseline 1 | Standard Self-Instruct | No calibration, shows base LLM misalignment |
| Baseline 2 | Full fine-tuning on small real set | Overfits to few examples, tests sample efficiency |
| Baseline 3 | Constant bias correction | Context-independent logit shift, tests embedding need |
| Ablation | ICM without L2 regularization | Isolates regularization effect on overfitting |

### Why this setup validates the claim
This design forms a falsifiable test by comparing ICM against baselines that isolate each component of the claimed mechanism. Self-Instruct measures the base misalignment that ICM must correct. Full fine-tuning shows the cost of overfitting to small data, which ICM avoids via convexity. Constant bias correction tests whether context-dependent query embeddings are necessary. The ablation tests whether L2 regularization is critical for robustness. KL divergence on held-out real queries directly measures distributional alignment, the central claim. The multi-task dataset ensures diversity and avoids task-specific artifacts. If ICM outperforms all baselines on the metric, it confirms that the linear correction in embedding space generalizes and avoids overfitting, validating the lightweight design.

### Expected outcome and causal chain

**vs. Standard Self-Instruct** — On a case where real user intents diverge from the base LLM's default distribution (e.g., legal questions requiring precise terminology), Self-Instruct generates synthetic data with off-target responses because it amplifies the base LLM's biases. ICM instead uses real interactions to learn a correction W·phi(q) that shifts logits toward the target distribution on similar queries, so we expect a noticeable reduction in KL divergence (e.g., 30-50% lower) on queries similar to the real set, with minimal improvement on unrelated queries.

**vs. Full fine-tuning on small real set** — On a case where the real set contains only a few examples of a rare intent (e.g., mathematical proof generation), full fine-tuning overadjusts to those examples, causing catastrophic forgetting of other intents and high variance. ICM adds a low-rank linear correction that does not modify the base LLM's parameters, preserving its general knowledge while aligning only the relevant directions. We expect ICM to show significantly lower KL divergence on rare-intent queries (e.g., 2x improvement) and comparable performance on common intents, avoiding the sharp degradation seen in fine-tuning.

**vs. Constant bias correction** — On a case where the needed correction varies with query context (e.g., the same token "loss" means different things in legal vs. math contexts), a constant bias shift applies the same adjustment to all queries, failing to resolve context-specific misalignment. ICM uses query embeddings phi(q) to produce context-dependent corrections, enabling per-query adaptation. We expect ICM to outperform constant bias by a larger margin on context-sensitive subsets (e.g., 20-40% lower KL) while performing similarly on homogeneous subsets where context matters less.

### What would falsify this idea
If ICM shows no improvement over constant bias correction on context-dependent subsets, or if the L2 ablation performs equally well (indicating no overfitting concern), then the claim that query embeddings and regularization are essential would be falsified. Additionally, if ICM performs worse than Self-Instruct on held-out real queries unrelated to the real set, it would indicate that the correction harms general generation quality, contradicting the lightweight claim.

## References

1. Autodata: An agentic data scientist to create high quality synthetic data
2. GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning
