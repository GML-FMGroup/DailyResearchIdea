# StreamCache: Parallel Cache Construction via Temporal Basis Decomposition for Sub-100ms Streaming Perception

## Motivation

Existing streaming perception systems (e.g., Wan-Streamer v0.2) rely on a single-GPU autoregressive Transformer to build the generation cache, creating a sequential bottleneck that limits latency to ~200ms and prevents sub-100ms interactive response. This structural dependency on sequential cache construction is an unresolved meta-gap across multiple approaches that assume a single-GPU Transformer is sufficient, but fail to parallelize the cache building step.

## Key Insight

The cache at each timestep depends primarily on a local temporal window, allowing it to be expressed as a linear combination of contributions from independently processed chunks via a learned temporal basis, enabling parallel computation and fast accumulation.

## Method

(A) **What it is**: StreamCache is a streaming encoder that replaces the sequential autoregressive cache building with parallel temporal basis decomposition. It takes a stream of input chunks (e.g., 4 video frames each) and outputs a compact cache vector per output timestep, enabling sub-100ms latency on a single GPU.

(B) **How it works**:
```python
# Hyperparameters: chunk_size=4, num_basis=64, cache_dim=512, kernel_size=10
# Learned parameters: chunk_encoder f_enc (MLP 128→256→64), basis_conv (transposed conv1d)
def stream_cache(x_stream):
    # x_stream: list of input frames
    # Step 1: Split into non-overlapping chunks
    chunks = [x_stream[i:i+chunk_size] for i in range(0, len(x_stream), chunk_size)]
    # Step 2: Encode each chunk independently in parallel
    chunk_coeffs = parallel_map(f_enc, chunks)  # (N, num_basis)
    # Step 3: Accumulate via causal transposed convolution
    # basis_conv maps (N, num_basis) to (T, cache_dim) with causal padding
    cache = causal_transposed_conv(chunk_coeffs, basis_conv)  # (T, cache_dim)
    return cache
```
The basis_conv weights are learned offline via distillation to minimize L2 loss against the sequential Transformer's cache on a training dataset.

(C) **Why this design**: We chose independent chunk encoding (1) over sequential processing because it enables parallel computation across chunks, drastically reducing latency; the trade-off is that cross-chunk dependencies are only captured in the later convolution and may lose fine-grained temporal interactions. We selected learned temporal basis via transposed convolution (2) rather than fixed bases (e.g., Fourier) because it can adapt to the specific data distribution and minimize distillation error, at the cost of requiring offline training with a teacher model. We enforce causal accumulation (3) instead of bidirectional to respect streaming constraints, which limits the receptive field but ensures no future information leaks into the current cache. Finally, we use a lightweight MLP as chunk encoder (4) instead of a Transformer to keep chunk processing fast, accepting that it may not model intra-chunk dependencies as effectively as a deeper architecture.

(D) **Why it measures what we claim**: The computational quantity `parallel_map(f_enc, chunks)` operationalizes *parallel cache building* because each chunk is processed independently, making the per-chunk latency constant regardless of sequence length, assuming chunks are independent. This assumption fails when the cache requires long-range cross-chunk dependencies that are not linearly decomposable, in which case the approximation may degrade cache fidelity. The `causal_transposed_conv` operation measures *linear combination of chunk contributions* because it aggregates chunk coefficients using learned temporal weights, assuming that cache is a linear function of chunk features. This assumption fails when the relationship is nonlinear or when the required temporal support exceeds the kernel size, in which case the convolution captures only local patterns and misses global structure. The offline distillation loss (L2 between StreamCache output and teacher sequential cache) ensures the decomposition *measures fidelity* to the original sequential cache, assuming the teacher cache is the ground truth. This assumption fails if the teacher itself is suboptimal, in which case the student inherits its flaws.

## Contribution

(1) The StreamCache architecture that replaces sequential autoregressive cache building with parallel temporal basis decomposition, enabling sub-100ms streaming perception on a single GPU. (2) An offline distillation framework that learns the temporal basis functions to closely approximate the sequential cache, minimizing fidelity loss while maintaining parallelism. (3) A design principle that linear decomposability of local cache dependencies can be exploited for latency reduction, with explicit trade-offs between parallelism and approximation accuracy.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | Kinetics-600 streaming subset | Large-scale video with temporal diversity |
| Primary metric | FVD (Fréchet Video Distance) | Measures distributional quality of generated video |
| Baseline 1 | Sequential Transformer | Gold standard for cache quality |
| Baseline 2 | Linear interpolation cache | Tests necessity of learned temporal structure |
| Baseline 3 | Fixed Fourier basis cache | Tests adaptability of learned bases |
| Ablation | Ours w/o distillation | Tests importance of teacher supervision |

### Why this setup validates the claim

The combination of dataset, baselines, and metric directly tests the central claim that StreamCache achieves near-sequential quality with parallel computation. Using Kinetics-600 ensures diverse temporal dynamics, making the failure modes (e.g., long-range dependencies) prominent. Comparing to the sequential Transformer (teacher) quantifies the fidelity loss from parallelism. The linear interpolation baseline tests whether simple averaging suffices; the Fourier baseline tests whether fixed temporal bases work. The ablation without distillation isolates the contribution of distillation to cache fidelity. FVD is chosen because it captures both frame-level and temporal statistics, relevant to cache quality.

### Expected outcome and causal chain

**vs. Sequential Transformer** — On a case with rapid motion across chunks (e.g., a dancer spinning), the sequential cache captures precise temporal alignment by attending to each frame. StreamCache approximates this via independent chunk encoding plus convolution, which may blur fast transitions due to limited temporal resolution. Thus we expect a small but noticeable FVD gap (e.g., 10-15% worse) on high-motion clips, but near-equal on static scenes.

**vs. Linear interpolation cache** — On a case with complex non-linear temporal patterns (e.g., a gradually opening flower), linear interpolation produces jerky transitions because it assumes constant velocity. StreamCache's learned basis can adapt to acceleration, yielding smoother transitions. We expect our method to clearly outperform linear interpolation on metrics, especially on clips with varying speed.

**vs. Fixed Fourier basis cache** — On a case with sharp temporal edges (e.g., a door slamming), Fourier bases ring due to Gibbs phenomenon, introducing artifacts. StreamCache's learned basis can avoid such artifacts by specializing to the data distribution. Thus we expect our method to have lower FVD on scenes with sudden changes, while performing similarly on smooth motion.

**vs. Ours w/o distillation** — On any case, the end-to-end trained version may converge to a local optimum with poorer cache fidelity. Distillation provides direct regression to the teacher cache, improving reconstruction. We expect the distilled version to have consistently better FVD, especially on challenging temporal patterns.

### What would falsify this idea

If our method shows no statistical difference from the sequential Transformer on high-motion subsets, it would contradict the predicted failure mode, suggesting the chunk encoding captures long-range dependencies adequately and the claim of a trade-off is wrong.

Alternatively, if the ablation without distillation outperforms the distilled version, it would indicate that teacher supervision is harmful.

## References

1. Wan-Streamer v0.2: Higher Resolution, Same Latency
2. Wan: Open and Advanced Large-Scale Video Generative Models
3. StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars
4. INFP: Audio-Driven Interactive Head Generation in Dyadic Conversations
5. Sonic: Shifting Focus to Global Audio Perception in Portrait Animation
6. Hallo3: Highly Dynamic and Realistic Portrait Image Animation with Video Diffusion Transformer
7. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models
8. Can Language Models Learn to Listen?
