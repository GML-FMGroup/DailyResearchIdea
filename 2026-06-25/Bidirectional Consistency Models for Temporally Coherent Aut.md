# Bidirectional Consistency Models for Temporally Coherent Autoregressive Video Generation

## Motivation

Causal-rCM achieves fast autoregressive video generation via causal masking, but lacks bidirectional temporal coherence, leading to drift over long sequences. The root cause is the structural assumption of causality, which prevents the model from aggregating future context without quadratic-cost full attention. Existing methods like Causal-rCM cannot enforce global temporal consistency because their forward-ODE trajectory is unidirectional; we propose to retain causal efficiency while introducing bidirectional coherence through cycle-consistency between forward and backward consistency models.

## Key Insight

Cycle-consistency between a forward and backward consistency model derived from the same teacher ODE enforces bidirectional coherence without bidirectional attention, because the consistency models share the underlying ODE flow and the cycle loss penalizes deviations from the ideal invertible mapping, resulting in a temporally consistent latent manifold.

## Method

(A) **What it is**: Bidirectional Consistency Model (BidirCM) is a pair of continuous-time consistency models: a forward model \(f_\theta\) that maps from any time \(t\) to the initial frame \(x_0\), and a backward model \(b_\phi\) that maps from any time \(t\) to a future frame \(x_{T}\) (e.g., the last frame of a clip). They share a common video transformer encoder (12 blocks, 8 attention heads, embedding dim 512) with two separate output MLP heads (2-layer, hidden=256, GeLU activation). The models are trained jointly with a cycle-consistency loss that enforces invertibility of the two mappings, providing bidirectional coherence without full attention. Input: noisy video frames at arbitrary times; output: denoised frames at boundaries, plus cycle-consistent reconstructions.  

(B) **How it works**: Pseudocode for training BidirCM.
```python
# Teacher ODE: pre-trained velocity field v(x,t) (3D U-Net from flow matching, fixed)
# Forward cCM f_theta(x_t, t) predicts x_0; Backward cCM b_phi(x_t, t) predicts x_T
# Hyperparameters: lambda_cycle = 0.5, lambda_cm = 1.0, N_steps = 20, dt = 0.001
# Optimizer: AdamW (lr=1e-4, weight_decay=0.01), batch size 64, 200K iterations, cosine lr schedule with 5K warmup

for each video clip (x_0, x_1, ..., x_T) do:
    Sample t ~ Uniform(0, T)
    x_t = add_noise(x_0, t)  # from forward diffusion/flow
    
    # Standard CM loss (compare with ODE step)
    x_0_pred = f_theta(x_t, t)
    x_t_minus = ODE_step(x_t, t, t - dt)  # one step of teacher ODE
    x_0_target = f_theta(x_t_minus, t - dt).detach()
    L_forward = ||x_0_pred - x_0_target||^2
    
    x_T_pred = b_phi(x_t, t)
    x_T_target = b_phi(ODE_step(x_t, t, t - dt), t - dt).detach()
    L_backward = ||x_T_pred - x_T_target||^2
    
    # Cycle consistency: forward then backward
    x_fb = b_phi(x_0_pred, 0)   # backward from time 0 to time t along ODE? Uses b_phi as a time-conditional model
    # Since b_phi is trained to map any time to T, we evaluate it at input time 0: it outputs prediction of x_T given x_0_pred.
    # To approximate state at time t, we use linear interpolation: x_fb = (t/T)*x_0_pred + (1 - t/T)*b_phi(x_0_pred, 0)  # simpler: treat b_phi as predictor of x_T, then ODE step back to t is too expensive.
    # Instead, we use a shortcut: b_phi(x_0_pred, 0) is a reconstruction of x_T; we can also compute f_theta(b_phi(x_0_pred, 0), T) to get back to x_0. The cycle losses are:
    L_cycle = ||f_theta(b_phi(x_0_pred, 0), T) - x_0_pred|| + ||b_phi(f_theta(x_T_pred, T), 0) - x_T_pred||
    # This measures how well forward-backward composition returns to the same frame.
    
    # Combined loss
    L = lambda_cm * (L_forward + L_backward) + lambda_cycle * L_cycle
    Update theta, phi to minimize L
```
*Note*: In practice, the cycle loss uses the cCM models' ability to map between any two times (since they are trained on all times). We train the models to be consistent on all pairs via the cycle, which enforces that \(b_\phi(f_\theta(x_t,t),0) \approx x_t\) and \(f_\theta(b_\phi(x_t,t),T) \approx x_t\).

(C) **Why this design**: We chose a pair of single-boundary cCMs (mapping to 0 and T) rather than a full two-time flow map (like FMM) because the latter requires learning a network for all time pairs, which doubles parameter count and memory; using two networks with disjoint boundaries keeps the model sizes similar to a single cCM (each 12M parameters, total 24M). We chose cycle-consistency loss over a symmetric Kullback–Leibler divergence because cycle-consistency directly enforces invertibility on the data manifold, while KL-based bidirectional matching (e.g., from DAN) would require adversarial training or density ratio estimation, introducing instability. We chose to train the forward and backward models jointly rather than sequentially (as in causal-rCM's teacher-forcing then self-forcing) because joint training allows the cycle loss to couple their representations from the start, reducing the chance of mode collapse. The cost of joint training is slower convergence per iteration, but we mitigate it by sharing low-level features through a shared encoder (e.g., the same transformer trunk with two output heads). This trade-off accepts increased training complexity in exchange for better temporal consistency.

(D) **Why it measures what we claim**: The forward-backward cycle loss \(\|b_\phi(f_\theta(x_t,t),0) - x_t\|\) operationalizes **bidirectional coherence** because if the combination is identity for all \(x_t\) and \(t\), then the forward mapping is invertible with inverse given by the backward mapping; this ensures that information flows both forward and backward in time without loss. The **load-bearing assumption** is that the forward and backward consistency models can be jointly trained via cycle-consistency loss to enforce invertibility, such that \(b_\phi(f_\theta(x_t,t),0) \approx x_t\) for all \(t\), thereby providing bidirectional temporal coherence. This assumption fails when model capacity is insufficient or training data is scarce; in that case, the cycle loss measures reconstruction error rather than true invertibility. To calibrate, we evaluate invertibility error on a synthetic Gaussian dataset where ground-truth ODE flow is known (see Experiment). The consistency loss \(\|f_\theta(x_t,t) - f_\theta(\text{ODE-step}(x_t), t-\Delta t)\|\), which is standard, operationalizes **temporal consistency along the ODE trajectory** because it enforces that the cCM output changes gradually as we move along the ODE; the underlying assumption is that the ODE is sufficiently smooth between time steps. This assumption fails near sharp transitions (e.g., scene cuts), where the loss reflects discontinuity in the teacher instead of model quality. Together, these losses ensure that the learned representations are both temporally smooth and invertible, which directly addresses the motivation-level concept of bidirectional coherence.

## Contribution

(1) A novel bidirectional consistency model framework (BidirCM) that enforces temporal coherence in autoregressive video generation through cycle-consistency between forward and backward continuous-time consistency models, avoiding quadratic-cost full attention. (2) A training procedure that jointly optimizes forward and backward cCMs with a cycle-consistency loss, enabling efficient gradient computation via existing FlashAttention-2 kernels (no custom kernel for reverse). (3) Identification of the cycle-consistency assumption's failure modes (capacity bottlenecks, scene cuts) and their effects on the metric (reconstruction vs. invertibility).

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | UCF-101 | Diverse motions, standard benchmark |
| Primary metric | FVD | Holistic video quality and temporal coherence |
| Baseline 1 | Causal-rCM | Strong autoregressive consistency baseline |
| Baseline 2 | Single cCM (forward only) | Isolates backward model effect |
| Ablation | Ours w/o cycle loss | Tests cycle-consistency contribution |

### Why this setup validates the claim

UCF-101 offers varied motion types that challenge temporal coherence, making it ideal to test bidirectional invertibility. FVD captures both frame quality and temporal dynamics, so a gap relative to baselines directly reflects the claimed benefits. Causal-rCM represents the standard autoregressive approach; outperforming it shows that cycle-consistency yields better future-frame prediction. The single cCM baseline (forward-only) isolates the need for a backward model: if our method improves, it confirms that bidirectional mapping helps. The ablation without cycle loss tests whether the cycle-consistency term itself is responsible for gains. Together, these comparisons turn the central claim—that cycle-consistency enforces invertible temporal flow—into a falsifiable prediction that our method will achieve superior FVD, especially on temporally demanding sequences.

### Expected outcome and causal chain

**vs. Causal-rCM** — On a case with rapid motion (e.g., a spinning top), the causal-rCM's forward-only consistency model accumulates error over autoregressive steps because it cannot correct future predictions from later frames. Our BidirCM, through the backward model and cycle loss, enforces that the initial frame prediction is invertible, allowing future frames to inform early predictions via the cycle. We expect a noticeable FVD improvement on clips with high motion or occlusions (e.g., ~10% lower FVD) while parity on static scenes.

**vs. Single cCM (forward only)** — On a sequence with repeated patterns (e.g., a walking person crossing behind an obstacle), the forward-only model produces inconsistent motion (jerky reappearance) because it lacks backward constraints. Our method's backward model forces the last frame to be consistent with the initial frame through cycle loss, smoothing the trajectory. We expect our method to achieve lower FVD (e.g., ~15% better) on temporally challenging subsets, with overall FVD advantage.

**vs. Ours w/o cycle loss** — On a sequence requiring long-range coherence (e.g., a camera pan across a static scene), the cycle-consistency loss ensures that forward-then-backward mapping returns to the same point, preventing drift. Without it, the model shows flickering or gradual appearance change. We expect the full model to maintain temporal stability, with lower FVD on long clips (e.g., >32 frames) than the ablation, while similar on short clips.

### What would falsify this idea

If the full BidirCM does not outperform the single-direction cCM on any subset (particularly on temporally challenging examples), then the backward model and cycle loss provide no measurable bidirectional coherence. Alternatively, if the improvement from cycle loss is uniform across all clips rather than concentrated on cases requiring invertibility (e.g., rapidly moving objects), then the cycle loss may act merely as a regularizer rather than enabling true invertibility.

## References

1. Causal-rCM: A Unified Teacher-Forcing and Self-Forcing Open Recipe for Autoregressive Diffusion Distillation in Streaming Video Generation and Interactive World Models
2. Mean Flows for One-step Generative Modeling
3. pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation
4. Multistep Distillation of Diffusion Models via Moment Matching
5. Flow map matching with stochastic interpolants: A mathematical framework for consistency models
6. Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis
7. One-Step Diffusion Distillation through Score Implicit Matching
8. Consistency Models Made Easy
