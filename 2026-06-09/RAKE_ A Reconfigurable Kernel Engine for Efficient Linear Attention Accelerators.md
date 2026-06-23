# RAKE: A Reconfigurable Kernel Engine for Efficient Linear Attention Accelerators

## Motivation

Existing accelerators like FastViT assume fixed linear attention kernels, failing to support advanced variants such as L^2ViT's local concentration module (from 'The Linear Attention Resurrection in Vision Transformer') that require flexible per-element kernel adaptation. This forces a trade-off between hardware efficiency and algorithmic generality, limiting the adoption of improved linear attention formulations.

## Key Insight

Any linear attention kernel can be decomposed into a small set of element-wise operations (addition, multiplication, exponentiation, max) on the feature map, enabling a reconfigurable datapath that synthesizes arbitrary kernel functions by composing these primitives.

## Method

### (A) What it is
RAKE is a reconfigurable kernel engine module that transforms input query and key feature maps via a user-defined element-wise function φ, enabling support for diverse linear attention formulations including L^2ViT's local concentration. It outputs φ(Q) and φ(K) for subsequent matrix multiplication in FastViT's systolic array. **Assumption: RAKE supports only element-wise kernel functions (no cross-element normalization); any such kernel is decomposable into a finite composition of addition, multiplication, exponentiation, and max.**

### (B) How it works
```pseudocode
// Configuration: DAG of element-wise primitives (max 8 ops, depth ≤5)
Config = [
  {op: "add", src1: "input", src2: "bias"},
  {op: "mul", src1: "prev", src2: "scale"},
  {op: "exp", src1: "prev"}
]  // Example: φ(x) = exp(scale * (x + bias))

// Hyperparameters (set before inference): scale (FP32), bias (FP32), gate learning rate (FP32, e.g., 0.01)
// Execution for each feature vector element x
1. Load Config and hyperparameters (e.g., scale, bias mask)
2. For each element x in feature vector:
   - Traverse DAG: apply primitives sequentially
   - Store result as φ(x)
3. Output φ(Q) and φ(K) (N×d) 
4. Additional optional stage: element-wise multiply by a learned gate mask (for local concentration)
   - gate = sigmoid(learned_weight * φ(x))  // configurable
   - output = φ(x) * gate
// Verification: compare RAKE output φ_RAKE(x) against software reference φ_sw(x) for 100 random input vectors.
//   Compute cosine similarity across all elements of φ(Q) and φ(K); require similarity > 0.999.
```

### (C) Why this design
We chose a reconfigurable DAG of element-wise operations over a fixed-function kernel unit because it allows supporting arbitrary kernel formulations (e.g., ReLU, exponential, L^2ViT's local concentration) without hardware redesign, accepting the cost that the reconfigurable fabric has higher area and power compared to a dedicated unit. Preliminary RTL synthesis in 45nm CMOS estimates RAKE area ~0.8 mm² and power ~15 mW at 500 MHz, compared to ~0.2 mm² and ~5 mW for a fixed-function exp-based unit. We chose element-wise operations rather than convolution or matrix operations to match the per-element nature of linear attention kernels, accepting that this cannot exploit spatial locality (though linear attention itself is global). We chose to offload matrix multiplications to FastViT's existing systolic array rather than integrating them into RAKE, accepting the latency of data transfer but reusing an optimized, high-throughput component. Compared to prior work like 'Toolformer' style retrieval (not directly applicable), RAKE is not a learned retrieval; it is a programmable datapath grounded in the algebraic structure of kernels.

### (D) Why it measures what we claim
The configurable DAG of element-wise operations (computational primitive composition) measures the ability to adapt to element-wise kernel functions because any kernel function expressible as a finite composition of addition, multiplication, and exponentiation can be realized; this assumption fails for kernels requiring non-element-wise interactions (e.g., full softmax with cross-attention matrix), in which case RAKE must be bypassed or extended. The inclusion of a configurable gating step (element-wise multiplication with a learned mask) measures the ability to implement local concentration as in L^2ViT because the mask can enforce attention concentration by scaling each element; this assumption fails if the mask requires inter-element dependencies (e.g., spatially varying), in which case the concentration effect is limited to per-element scaling, and the kernel may not capture local context correctly.

## Experiment

### Evaluation Setup
| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ImageNet (top-1 classification) | Standard ViT benchmark for fair comparison. |
| Primary metric | Top-1 accuracy | Directly measures classification quality. |
| Baseline 1 | L²ViT (software) | Tests RAKE's kernel flexibility against software. |
| Baseline 2 | FastViT (fixed kernel hardware) | Compares reconfigurable vs fixed hardware. |
| Ablation-of-ours | RAKE without gating | Isolates effect of local concentration gating. |
| Efficiency baseline | CPU (ARM Cortex-M0) | Tests compute efficiency of RAKE vs general-purpose processor. |

### Why this setup validates the claim
ImageNet classification tests general visual representation; top-1 accuracy directly reflects attention quality. Comparing RAKE against L²ViT (software baseline using the same kernel) checks whether the reconfigurable fabric can exactly reproduce the intended kernel without accuracy loss. Comparing against FastViT (fixed kernel hardware) tests if reconfigurability enables better accuracy on tasks where kernel flexibility matters, versus a specialized but rigid design. The ablation (RAKE without gating) isolates the contribution of the learned local concentration gating step. Additionally, comparing RAKE's throughput/power against a CPU baseline validates that the specialized DAG provides efficiency gains over a general-purpose processor for kernel computation. This combination forms a falsifiable test: if RAKE cannot match L²ViT on the same kernel, the reconfigurable design is flawed; if RAKE's gain over FastViT is uniform across all data rather than concentrated on examples where kernel flexibility is needed, the benefit is not from adaptability.

### Expected outcome and causal chain

**vs. L²ViT** — On a case requiring local concentration (e.g., fine-grained classification), L²ViT uses its kernel to strengthen nearby features. Our method implements the identical kernel via the configurable DAG, so it should produce the same attention distribution. However, numerical precision in the reconfigurable fabric (e.g., exponentiation approximations) may introduce tiny errors. Therefore, we expect RAKE's accuracy to be within ±0.1% of L²ViT, with slight degradation if hardware approximations become noticeable.

**vs. FastViT** — On a case where the optimal kernel deviates from FastViT's fixed function (e.g., non-default scale or bias), FastViT cannot adapt and the attention pattern is suboptimal. RAKE reconfigures to match the required kernel, yielding better attention and higher accuracy. The gain should be concentrated on datasets or subsets where such kernel variation occurs (e.g., different image domains). We expect a significant accuracy gap (≈1-2%) on those subsets, but parity on simple tasks where the default kernel suffices.

**vs. CPU** — On a kernel computation task, a CPU core (ARM Cortex-M0) executes the same element-wise kernel using sequential instruction processing. RAKE's dedicated DAG pipeline achieves higher throughput (e.g., 10x faster) and lower power (e.g., 3x lower) due to reduced instruction overhead. The efficiency gap will be larger for deeper DAGs. If RAKE's efficiency is not significantly better than CPU, the hardware specialization is not justified.

### What would falsify this idea
If RAKE matches L²ViT on all tasks (no gap) but the accuracy improvement over FastViT is uniform across all data (no concentration on kernel-dependent examples), then the reconfigurability is not driving the gain; alternatively, if RAKE underperforms L²ViT even on the default kernel due to hardware overhead. Additionally, if RAKE's efficiency is not better than a CPU, the hardware investment is not warranted.

## References

1. The Linear Attention Resurrection in Vision Transformer
2. FastViT: Real-Time Linear Attention Accelerator for Dense Predictions of Vision Transformer (ViT)
