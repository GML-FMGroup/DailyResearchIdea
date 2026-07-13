# MetaCoRe: Meta-Learning for Compositional Reasoning in Video Generation

## Motivation

OpenCoF's dataset of 17K samples and 11 task families fails to cover diverse reasoning scenarios, leading to poor compositional generalization for unseen reasoning chains. The root cause is that supervised fine-tuning on a fixed set of task families does not force the model to learn reusable reasoning primitives; it overfits to the specific compositions seen during training.

## Key Insight

Compositional generalization in reasoning emerges when the model is trained to rapidly adapt to tasks drawn from a distribution that systematically varies the combination of reasoning primitives, causing the model to learn a shared latent representation of those primitives.

## Method

### MetaCoRe: Meta-Learning for Compositional Reasoning in Video Generation

**Load-bearing assumption (explicit):** The meta-learning procedure learns a shared representation of reasoning primitives (encoded as token embeddings) that can be recomposed for novel tasks, because the inner-loop updates force the model to quickly adapt to new compositions by reusing and adjusting these primitives. If this assumption fails (e.g., due to feature reuse without representation learning), the method will not improve compositional generalization.

(A) **What it is**: MetaCoRe is a meta-learning framework that trains a video generation model (Wan2.2 backbone with reasoning tokens) to compose reasoning tokens from a shared bank of primitives. Input: a meta-training set of tasks from merged benchmarks. Output: a model that can quickly adapt to a new compositional reasoning task with few gradient steps.

(B) **How it works**: The core mechanism is a bi-level optimization loop with a sparsity penalty to enforce disentanglement of reasoning primitives.

```
# MetaCoRe Training
Given: base model M with parameters θ (Wan2.2 backbone + reasoning tokens)
  - Reasoning tokens: 16 learnable embeddings, each dimension d_model=3840 (matching Wan2.2 token embedding size)
  - Shared primitive bank: a set of K=64 learnable embedding vectors of dimension d_model, from which reasoning tokens are initialized and updated via meta-learning
Task distribution p(T) from merged benchmarks (RULER-Bench, V-ReasonBench, MMGR): uniformly sample a benchmark, then uniformly sample a task family from that benchmark, then sample a specific task instance (support size=8, query size=16)
For each meta-iteration (total 10K iterations on 4 A100 80GB GPUs, ~72 hours):
  Sample batch of 16 tasks T_i ~ p(T)
  For each task T_i:
    Sample support set S_i and query set Q_i
    Compute adapted parameters θ'_i = θ - α ∇_θ L(M_θ, S_i)  # inner loop, α=0.01, 2 gradient steps
    Compute meta-loss L_meta = Σ_i L(M_{θ'_i}, Q_i)  # cross-entropy loss on generated video tokens
    # Disentanglement regularization: sparsity penalty on attention weights from video tokens to reasoning tokens
    Compute L_disent = λ * Σ_i ||attn_i||_1, where attn_i are attention weights from video tokens to reasoning tokens in the adapted model, λ=0.01
  Update θ = θ - β ∇_θ (L_meta + L_disent)  # outer loop, β=0.001, Adam optimizer
# At test time, given new task with few examples (support set of 8), adapt θ via inner loop (2 steps, α=0.01).
```

(C) **Why this design**: We chose meta-learning over naive fine-tuning on merged benchmarks because the meta-objective explicitly optimizes for fast adaptation to unseen task compositions, rather than minimizing average error on a static set. This trades off longer training time (due to inner loops) for improved compositional generalization. We used a first-order meta-learning approach (Reptile-like) instead of MAML to avoid expensive second-order gradients, accepting a slight drop in adaptation speed for computational feasibility. We designed the task distribution p(T) to cover diverse reasoning skill combinations (e.g., temporal ordering, spatial reasoning, causal inference) by merging three complementary benchmarks, ensuring that the model encounters novel compositions during meta-training. This design sacrifices benchmark-specific fine-tuning performance for broad generalization. We chose to insert reasoning tokens as in OpenCoF but treat their embeddings as part of the shared primitive bank, updated via meta-learning—this couples the token learning with the composition objective, unlike OpenCoF's static token addition. The sparsity penalty encourages each video token to attend to only a few reasoning primitives, promoting disentanglement and preventing the primitive bank from collapsing into monolithic representations.

(D) **Why it measures what we claim**: The inner-loop adaptation loss L(M_θ, S_i) measures the model's ability to quickly incorporate new task-specific patterns from few examples, which we assume is equivalent to composing learned primitives in novel ways. This assumption fails when the support set contains spurious correlations that are not indicative of the underlying composition, leading to overfitting rather than generalization. In that case, the loss reflects memorization of the support set's peculiarities rather than compositional skill. The meta-loss L_meta aggregated over tasks measures the average adaptability across task compositions, which operationalizes the robustness of the learned primitive representation; this assumes that the task distribution p(T) is representative of the target compositional space, and fails when test compositions involve primitives outside p(T). The shared primitive bank (learned reasoning token embeddings) operationalizes the concept of reusable reasoning modules because the same token embedding is used across multiple tasks; this assumes that reasoning primitives are linearly separable in the embedding space, which fails when primitives are highly context-dependent (e.g., negation in different reasoning chains), in which case the bank collapses to monolithic representations. The sparsity penalty L_disent directly enforces that only a few primitives are activated per video token, encouraging disentanglement and addressing the potential failure mode of feature reuse.

## Contribution

(1) MetaCoRe, a meta-learning framework that trains video generation models to achieve compositional generalization in reasoning by optimizing for rapid adaptation across diverse task compositions. (2) A unified meta-training set from three benchmarks (RULER-Bench, V-ReasonBench, MMGR) systematically varying reasoning primitives, demonstrating that meta-learning over this distribution improves zero-shot compositional reasoning. (3) Empirical analysis showing that the learned reasoning token embeddings form a compositional structure that enables recombining primitives unseen during training.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Combined test tasks from RULER-Bench (5 families), V-ReasonBench (8 families), MMGR (6 families) – 400 held-out tasks, each with support of 8 and query of 16 examples | Held-out compositional reasoning tasks with diverse primitive combinations |
| Primary metric | Accuracy on held-out compositional reasoning tasks (average over 10 seeds, judged by trained evaluator model) | Direct measure of compositional generalization |
| Baseline1 | Wan2.2-I2V-A14B | No reasoning tokens, no meta-learning |
| Baseline2 | Wan-CoF (OpenCoF) | Reasoning tokens, no meta-learning |
| Ablation-of-ours | MetaCoRe w/o meta-objective (joint fine-tuning on all tasks, no inner loop) | Isolates meta-objective contribution |

### Why this setup validates the claim

This experimental design forms a falsifiable test of the central claim that meta-learning across diverse reasoning tasks improves compositional generalization in video generation. The dataset includes tasks with novel combinations of primitives (e.g., temporal + spatial + causal) not seen during training, so performance reflects the ability to recombine learned reasoning modules. The primary metric directly measures reasoning accuracy on these unseen compositions. Comparing against Wan2.2 tests whether reasoning tokens are necessary; against Wan-CoF tests whether static tokens suffice; against the ablation (joint fine-tuning) tests whether the meta-objective itself drives fast adaptation. If our method outperforms all baselines particularly on rare compositions (e.g., >20% gap on novel triples), the claim is supported. The ablation pinpoints the meta-objective's role, and the metric ensures we measure generalization, not memorization. The sparsity penalty in MetaCoRe additionally tests whether explicit disentanglement helps; we will compare with a version without the penalty in ablation if needed.

### Expected outcome and causal chain

**vs. Wan2.2-I2V-A14B** — On a case requiring novel composition of temporal and spatial reasoning (e.g., "object moves to A, then B, only if C is present"), Wan2.2 lacks reasoning tokens to represent such structure, so it generates plausible but incorrect videos because it treats frames independently. Our method uses learned reasoning tokens that can be quickly adapted via inner loop to capture the new composition, so we expect a large gap (e.g., >30% accuracy difference) on such novel compositions, but parity on simple single-skill tasks.

**vs. Wan-CoF (OpenCoF)** — On a case requiring recombination of known primitives in a new order (e.g., causal then temporal, but reversed), Wan-CoF uses static tokens that may not adapt to the new sequence, causing misordering. Our method's meta-learning adjusts token embeddings to the new composition via few gradient steps, so we expect a moderate gap (e.g., 15–20% improvement) on tasks where the composition order is novel, but similar performance on familiar orders.

**vs. MetaCoRe w/o meta-objective (joint fine-tuning)** — On a task that is compositionally distant from training (e.g., involves three primitives never seen together), joint fine-tuning treats all tasks uniformly and overfits to common combinations, so it fails on rare ones. Our meta-objective explicitly prepares for unseen combinations, so we expect a significant gap (e.g., >25%) on rare compositions, but comparable on frequent ones.

### What would falsify this idea

If our method shows uniform improvement across all task types (including simple ones) rather than concentrated gains on novel compositions, or if the ablation performs similarly to MetaCoRe on novel compositions, then the meta-objective is not responsible for the improvement, contradicting the central claim. Additionally, if the sparsity penalty does not lead to better disentanglement (measured by lower mutual information between primitive embeddings), the assumption about representation learning may be false.

## References

1. OpenCoF: Learning to Reason Through Video Generation
