# Q-Adapt: Quantization-Aware Continued Pre-training for Low-Resource Code Completion on Consumer Hardware

## Motivation

Existing approaches for adapting code LLMs to low-resource languages, such as continued pre-training (Teaching LLMs a Low-Resource Language), produce large models requiring expensive GPUs for inference. Post-training quantization (Evaluating Quantized Large Language Models) reduces model size but disrupts language-specific features learned during adaptation, causing a sharp accuracy drop. The root cause is that naive quantization treats the adapted model as fixed, ignoring that language-specific representations may be sensitive to low-precision arithmetic.

## Key Insight

Simulating quantization errors during continued pre-training forces the model to learn features that are invariant to low-precision arithmetic, seamlessly compressing the model without post-hoc accuracy loss.

## Method

(A) **What it is**: Q-Adapt, a method that performs quantization-aware continued pre-training of a base code LLM on low-resource language data. Its inputs are a base LLM (e.g., CodeLlama-7B) and a low-resource language corpus (e.g., Pharo code). Its output is a compact model with weights quantized to 4-bit precision that achieves high code completion accuracy on that language.

(B) **How it works**: Q-Adapt operates in three phases plus a calibration step:
1. **Initialization**: Start from a base code LLM (full-precision).
2. **Quantization-Aware Forward Pass**: For each batch of low-resource code, apply simulated weight quantization using a straight-through estimator (STE). The quantizer maps each weight to its nearest point in a 4-bit uniform grid with step size s = (max_weight - min_weight) / 15, computed per layer per batch. Additionally, a small Gaussian noise (std=0.01) is added to activations to mimic quantization effects. The forward pass computes the language modeling loss.
3. **Regularized Update**: The training objective is `L = L_lm + λ * L_q`, where `L_q` is a regularization term penalizing the distance between weights and their quantized values (L2 norm). The coefficient `λ` starts at 0.1 and linearly increases to 1.0 over training steps (10,000 steps). The backward pass updates full-precision weights via STE (gradients pass through quantizer unchanged). After training, the final quantized weights are exported.
4. **Calibration Step (every 1000 steps)**: We verify the load-bearing assumption that STE provides sufficiently accurate gradients. We compare STE gradients to finite-difference estimates on a held-out batch of 64 samples. If the cosine similarity falls below 0.9, we reduce the learning rate by factor 0.5 for the next 1000 steps. This ensures gradient quality remains adequate.

**Hyperparameters**: target bits=4, learning_rate=5e-5, batch_size=8, λ_start=0.1, λ_end=1.0, total_steps=10000, noise_std=0.01.

(C) **Why this design**: We chose simulated quantization over post-training quantization because it allows the model to adapt its representations to be robust to quantization, preventing the accuracy drop observed in related work. We quantize only weights (not activations) to reduce computational overhead; this trade-off accepts that activation quantization may further improve compression but adds complexity. We adopted a dynamic regularization schedule (increasing λ) over a fixed one to first learn language features in a high-precision space before enforcing grid adherence, mimicking a curriculum that stabilizes training compared to aggressive regularization from the start. Using STE instead of full quantization-aware backpropagation is computationally efficient, though it introduces gradient bias; we explicitly calibrate its quality during training. We chose 4-bit based on prior evidence that 4-bit offers the best trade-off for code generation on consumer hardware.

(D) **Why it measures what we claim**: The simulated weight quantization (`quantizer(weights)`) measures the robustness of language-specific representations to low-precision arithmetic because it applies the same rounding errors as deployment; this assumption fails when weight distributions are highly non-uniform, in which case the fixed grid may overestimate actual error. The regularization term `L_q` measures the model's readiness for quantization because it penalizes weights far from grid points; this assumes weight distribution is uniform over grid intervals, which fails for heavy-tailed distributions, in which case regularization may overly constrain and degrade language learning. The combined loss `L = L_lm + λL_q` measures the ability to perform code completion under quantization constraints because it directly optimizes the target task with simulated deployment conditions; however, it ignores hardware-specific effects like arithmetic precision differences, so real performance may vary. We include an ablation using non-uniform quantization grids to test robustness to non-uniform distributions.

## Contribution

(1) Q-Adapt, a quantization-aware continued pre-training framework that jointly adapts a code LLM to a low-resource language and optimizes its weights for low-precision deployment, producing a compact model suitable for consumer hardware. (2) Empirical finding that incorporating quantization noise during language adaptation preserves code completion accuracy (e.g., on Pharo) while reducing model size by 4× compared to full-precision models, avoiding the accuracy degradation of post-training quantization. (3) A benchmark for evaluating efficiency-accuracy trade-offs on low-resource code completion, enabling fair comparison across adaptation and compression methods.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Pharo code corpus, Julia code corpus, R code corpus | Three low-resource languages for diversity |
| Primary metric | Exact match accuracy, inference latency (ms/token) on NVIDIA RTX 3090 | Measures correctness and cost |
| Baseline 1 | Base LLM (zero-shot) | No domain adaptation baseline |
| Baseline 2 | Full-precision fine-tuned | Upper bound without quantization |
| Baseline 3 | Post-training quantization (PTQ) | Standard compression baseline |
| Ablation 1 | Q-Adapt without QAT regularization | Isolates effect of quantization-aware training |
| Ablation 2 | Q-Adapt with non-uniform quantization grid (k-means clustering per layer) | Tests assumption of uniform grid intervals |
| Verification | STE gradient quality check (cosine similarity) | Ensures estimator bias is controlled |

### Why this setup validates the claim
This experimental design tests whether quantization-aware continued pre-training (Q-Adapt) can maintain high code completion accuracy under 4-bit weight quantization compared to uncompressed or naively compressed alternatives. The Pharo, Julia, and R datasets isolate the low-resource scenario across diverse syntax families. Base LLM measures the zero-shot gap, full-precision fine-tuned establishes an upper bound, and PTQ shows the accuracy drop without adaptation. The ablations gauge the necessity of the regularizer and the robustness to non-uniform weight distributions. Exact match accuracy directly reflects code correctness, which is the ultimate goal. Inference latency on consumer hardware (RTX 3090) validates the practical benefit. If Q-Adapt outperforms PTQ and approaches full-precision, the claim is supported; if not, the central hypothesis is falsified.

### Expected outcome and causal chain

**vs. Base LLM (zero-shot)** — On a case where the base LLM has never seen Pharo syntax (e.g., method definition with block closures), the base model produces syntactically invalid completions because its training distribution lacks such patterns. Our method instead generates valid completions because it fine-tunes on Pharo data, learning language-specific syntax and idioms. Therefore, we expect a large accuracy gap (e.g., >30% absolute difference) on all Pharo benchmarks, with the base model near zero. Similar gaps expected for Julia and R.

**vs. Full-precision fine-tuned** — On a simple completion like assigning a variable, the full-precision model outputs the exact token because it has no precision constraints. Our method may deviate occasionally due to 4-bit grid approximation, but the quantization-aware training adapts weights to fit the grid, so errors are minimal. We expect performance close to full-precision (within 1-2% accuracy) on most cases, with slight degradation where weight distributions are non-uniform.

**vs. Post-training quantization (PTQ)** — On a case requiring precise numerical reasoning (e.g., loop counter initialization), PTQ suffers accuracy loss because it quantizes fixed weight distributions without adaptation. Our method instead reshapes weights to align with the 4-bit grid during training, preserving representational fidelity. Thus, we expect Q-Adapt to outperform PTQ on complex tasks (e.g., 5-10% higher accuracy), while both may perform similarly on simple patterns.

### What would falsify this idea
If Q-Adapt does not outperform PTQ by a meaningful margin (e.g., <2% absolute difference) across all difficulty levels and languages, or if the gain is concentrated only in simple completions where PTQ already works well, then the central claim that quantization-aware training preserves accuracy for low-resource languages is falsified. Additionally, if the STE calibration step consistently triggers learning rate reductions without improvement, the gradient bias may be too severe.

## References

1. Teaching LLMs a Low-Resource Language: Enhancing Code Completion in Pharo
2. Mellum: Production-Grade in-IDE Contextual Code Completion with Multi-File Project Understanding
3. Enhancing Code Generation for Low-Resource Languages: No Silver Bullet
4. Evaluating Quantized Large Language Models for Code Generation on Low-Resource Language Benchmarks
5. Qwen2.5-Coder Technical Report
6. Context Composing for Full Line Code Completion
7. Full Line Code Completion: Bringing AI to Desktop
8. Multi-line AI-Assisted Code Authoring
