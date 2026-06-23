# Coupled Noise Diffusion for Semantically Coherent Parallel Region Generation

## Motivation

Current parallel region diffusion models like PerceptionDLM generate region descriptions independently, assuming conditional independence per region given the image. This leads to semantic inconsistencies, such as contradictory or unaligned descriptions across regions, because the denoising process lacks a mechanism to enforce cross-region coherence. The root cause is the independent per-region noise process, which prevents the model from learning global dependencies during training.

## Key Insight

A shared low-rank noise component couples the region generation processes, making the denoising task require joint reasoning across regions, which inherently induces cross-region semantic coherence.

## Method

**Load-bearing assumption**: A shared low-rank noise component can capture the essential cross-region semantic dependencies such that denoising jointly enforces consistency.

### Coupled Noise Diffusion (CND)

**Input**: image features I, region queries Q_i for i=1..R, diffusion step t
**Output**: clean token sequences for each region

**Hyperparameters**: d_shared = 64, alpha_t from cosine schedule (1000 steps), denoising network = 6-layer transformer, hidden=512, 8 heads, cross-attention to I and cross-region attention between regions.

**Training hyperparameters**: learning rate 1e-4, linear warmup 2000 steps, batch size 32, trained for 100k steps on 2 GPUs.

1. Sample shared low-rank noise Z ~ N(0, I) of dimension d_shared
2. For each region i, project Z to region-specific noise: N_i_shared = W_i @ Z, where W_i is learnable (d_shared x seq_len_i)
3. Sample per-region independent noise epsilon_i ~ N(0, I) (seq_len_i)
4. Combine: N_i = alpha_t * N_i_shared + sqrt(1 - alpha_t^2) * epsilon_i
5. Get noisy token embeddings: x_i_t = sqrt(alpha_bar_t) * x_i_0 + sqrt(1 - alpha_bar_t) * N_i
6. Concatenate each region's noisy embeddings with shared noise: input_i = concat([x_i_t, Z])  # shape: [seq_len_i, d_model+d_shared]
7. Denoising network (transformer with cross-attention to I and cross-region self-attention between regions; each region's token sequence attends to all other regions' token sequences via a shared cross-attention layer) predicts noise for each region: pred_N_i = Network(input_1..input_R, I)
8. Loss: L = sum_i MSE(pred_N_i, N_i)

### Why this design

We chose a low-rank shared noise structure over a full coupling (e.g., identical noise for all regions) because low-rank provides a manageable degree of correlation while allowing region-specific variations: the low-rank component enforces global alignment, while the independent noise handles local details. We used a learnable projection matrix per region to adapt the shared noise to different region lengths and semantic contexts, accepting the cost of additional parameters. We concatenated the shared noise representation to each region's input rather than adding it directly, because the denoising network can then condition on the shared noise explicitly, which encourages joint reasoning; the trade-off is increased input dimension. We set the rank d_shared to 64, which is small relative to typical token embeddings (e.g., 768), to ensure the shared noise does not dominate the signal; this is based on the intuition that cross-region consistency can be captured by a low-dimensional subspace. The denoising network uses cross-region attention to allow tokens from different regions to interact, complementing the noise coupling. Training uses standard diffusion hyperparameters (cosine schedule, 1000 steps, lr 1e-4, batch 32).

### Why it measures what we claim

The shared low-rank noise component Z measures cross-region semantic coupling because the denoising network must use Z to predict all regions' noise; this forces the network to learn dependencies that remove the correlated part, which is equivalent to enforcing consistency if the low-rank assumption holds. **Assumption A**: semantic dependencies across regions lie in a low-rank subspace. **Failure mode F**: when dependencies are high-rank or non-factorizable, the shared component captures only partial consistency, and the rest is learned via independent noise, potentially still leaving inconsistencies. The independent noise epsilon_i measures per-region noise that cannot be predicted from other regions; by training the network to also predict epsilon_i, we ensure that the network learns to separate shared and independent components, but this is only accurate if the noise decomposition aligns with the true semantic dependencies. The concatenation of shared noise Z to each region's input measures the extent to which the network can exploit global context; if Z is informative, the network's predictions should converge faster, providing a diagnostic for coherence.

**Ablation**: We will vary d_shared in {16, 32, 64, 128} to study the effect of the shared noise rank.

## Contribution

(1) We introduce a coupled noise diffusion mechanism for parallel region generation that injects a shared low-rank noise component across all regions, enabling cross-region semantic coherence. (2) We establish that coupling the noise process forces the denoising network to learn joint region representations, replacing the conditional independence assumption of previous parallel diffusion models like PerceptionDLM. (3) We provide an analysis of how the rank of the shared noise affects the trade-off between coherence and generation diversity.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ParaDLC-Bench (3,000 images, 2-4 regions each) | Multi-region images with varied dependencies |
| Primary metric | CIDEr | Measures caption relevance and diversity |
| Baseline 1 | Autoregressive MLLM (LLaVA-1.5 7B, finetuned) | Standard sequential generation baseline |
| Baseline 2 | Sequential diffusion MLLM (same architecture, runs regions sequentially) | Processes regions sequentially |
| Baseline 3 | Parallel diffusion (independent noise, no shared component) | No cross-region coordination |
| Ablation of ours | CND without independent noise (epsilon_i=0) | Tests necessity of independent noise |

### Why this setup validates the claim

The central claim is that Coupled Noise Diffusion (CND) improves cross-region consistency in parallel region captioning by enforcing joint denoising through a shared low-rank noise component. ParaDLC-Bench directly tests this with multi-region images requiring coherent descriptions. The autoregressive and sequential diffusion baselines isolate the benefit of parallelism and diffusion, while the independent noise baseline isolates the effect of coupling. The ablation tests the independent noise component's role. CIDEr captures overall caption quality; if cross-region consistency improves, CIDEr should increase on multi-region images where coherence matters. This combination forms a falsifiable test: if CND does not outperform the independent noise baseline on multi-region scenes with interacting objects, the claim fails.

### Expected outcome and causal chain

**vs. Autoregressive MLLM** — On a multi-region image of a child feeding a dog, the autoregressive model generates captions in order (e.g., region A then B). Without global context, it might caption region A as "a child" and region B as "a dog" but fail to capture the interaction, leading to inconsistent or incomplete descriptions like missing the feeding action. Our method, with shared noise across both regions, forces the denoising network to jointly predict all regions, so it learns to align the captions: both regions will mention the feeding action consistently. We expect a noticeable gap on images with strong inter-region interactions (e.g., +5-10 CIDEr) but parity on single-object images.

**vs. Sequential diffusion MLLM** — On a scene with three objects (e.g., cat, ball, sofa), sequential diffusion generates regions one by one, conditioning on previous outputs. Error propagation: if the first region misidentifies the cat as a dog, the second region might adapt incorrectly. Our method processes all regions in parallel with coupled noise, so the denoising network sees all noisy regions simultaneously and can correct such inconsistencies. We expect lower variance and higher CIDEr on complex scenes, with a gain of ~3-5 CIDEr on multi-region subsets.

**vs. Parallel diffusion without coupling** — On an image of two identical cups on a table, independent noise may generate captions like "a blue cup" for region 1 and "a red cup" for region 2 due to lack of coordination. Our shared noise encourages consistency: both regions should describe the same color. We expect a cleanup of inconsistencies, yielding higher CIDEr on multi-region images with repeated objects (e.g., +5 CIDEr) but similar performance on single-region images.

### What would falsify this idea

If CND shows no improvement over independent noise on multi-region images with interacting objects, or if the improvement is uniform across all image types (including single-region), then the shared noise is not capturing the intended coupling and the central claim is invalid. Additionally, if varying d_shared shows no correlation with consistency metrics, the low-rank assumption may be flawed.

## References

1. PerceptionDLM: Parallel Region Perception with Multimodal Diffusion Language Models
2. SDAR: A Synergistic Diffusion-AutoRegression Paradigm for Scalable Sequence Generation
3. LLaDA2.0: Scaling Up Diffusion Language Models to 100B
4. Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning
5. Scaling Diffusion Language Models via Adaptation from Autoregressive Models
6. Code Llama: Open Foundation Models for Code
7. Auto-Regressive Next-Token Predictors are Universal Learners
8. Graph of Thoughts: Solving Elaborate Problems with Large Language Models
