# Camera-Decoupled World Model for Independent Camera Control in Autoregressive Video Generation

## Motivation

Existing autoregressive video world models, such as AlayaWorld, conflate camera motion with other user actions by treating both as a combined action input. This structural coupling prevents the model from generating videos with independent camera control, e.g., panning across a scene without affecting object states. The root cause is the lack of a dedicated representation for camera pose that can be disentangled from the action embedding.

## Key Insight

A dedicated positional encoding for camera parameters creates a representational bottleneck that forces the cross-attention mechanism to learn camera-specific features, thereby naturally decoupling camera motion from action in the latent space.

## Method

(A) **Camera-Decoupled World Model (CDWM)** extends autoregressive world models with a separate camera control branch. Inputs: video frames x_t, actions a_t, camera poses c_t (6D: rotation+translation). Outputs: predicted next frame x_{t+1} conditioned on independent camera and action signals. **Load-bearing assumption**: The cross-attention mechanism will learn to use camera position embeddings to condition on camera motion independently of action, even when camera and action are correlated in the training data.

(B) **How it works**:
```pseudocode
For each frame t:
  h_t = Encoder(x_{t-1})  # base world model encoder
  e_t = PositionalEncoding(c_t)  # sinusoidal encoding with frequencies [1e-3,1e4]
  g_t = ActionEncoder(a_t)  # MLP with 2 hidden layers (256)
  h'_t = CrossAttention(query=h_t, key=e_t, value=e_t)  # 2 attention heads, residual+LayerNorm
  h''_t = MLP([h'_t, g_t])  # concatenation, 2 hidden layers (512)
  x_t = Decoder(h''_t)  # same as base model decoder
  # Decoupling regularization: minimize mutual information between e_t and g_t
  L_dec = -I(e_t; g_t)  # implemented via gradient reversal layer (Ganin et al., 2016)
  L = L_pred + λ L_dec, with λ=0.1
```
Hyperparameters: positional encoding frequencies = 10,000; cross-attention heads=2; MLP hidden dim=512; regularization weight λ=0.1.

(C) **Why this design**: We chose a dedicated sinusoidal positional encoding for camera parameters (rather than concatenating raw camera values) because it provides a continuous, smooth representation that captures relative pose changes effectively, accepting the cost of increased embedding dimension (e.g., 256). We chose cross-attention over additive fusion to allow the model to selectively attend to camera-relevant features in the hidden state, which improves generalization to unseen trajectories at the expense of higher computational cost. We kept the action encoder separate to enforce modularity, ensuring that the camera branch does not interfere with action conditioning; this trade-off requires paired training data but avoids the risk of accidental coupling. The 2-head cross-attention balances expressiveness and efficiency; more heads might overfit to correlated camera-action patterns, while fewer might underfit. We added a mutual information minimization term to actively encourage decoupling, which is critical given the strong correlation between camera and action in many datasets.

(D) **Why it measures what we claim**: The computational quantity e_t (positional encoding of camera pose) measures decoupled camera influence because it is computed solely from camera parameters and processed independently before fusion. The assumption linking e_t to decoupled camera control is that the cross-attention mechanism will learn to map camera poses to spatial transformations in the video output without interference from the action input; this assumption fails when camera and action are correlated in the training data (e.g., agent always moves camera when moving), in which case e_t may reflect co-occurrence rather than true decoupling. The cross-attention output h'_t measures state conditioned on camera because it results from attending to camera embeddings; however, this assumes the attention weights focus on the camera branch and not on residual camera information already present in h_t from the previous frame. This assumption fails when the encoder h_t already captures camera motion, leading to redundant conditioning. To verify this, we conduct an intervention test: modify camera parameters independently and measure if frame changes correspond solely to camera changes.

## Contribution

(1) A novel camera-decoupled world model architecture that introduces a dedicated camera control branch with sinusoidal positional encoding and cross-attention fusion, enabling independent camera manipulation in autoregressive video generation. (2) Empirical finding that adding such a branch significantly improves the model's ability to generate videos with arbitrary camera motions while preserving content consistency, as measured by FVD and temporal consistency metrics. (3) A new evaluation protocol for assessing camera disentanglement in video world models, based on controlling camera parameters while fixing actions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | DMLab | Large diverse videos with known camera poses |
| Primary metric | FVD | Measures video realism and motion |
| Baseline 1 | VideoSSM | Strong autoregressive baseline |
| Baseline 2 | Fast AR Video Diffusion | SOTA diffusion baseline |
| Ablation-of-ours | CDWM without cross-attention and without regularization | Tests necessity of camera attention and decoupling loss |
| Intervention test | Modify camera poses randomly while keeping actions fixed, measure frame changes | Directly quantify decoupling: if camera change causes proportional frame change, decoupling is achieved |

### Why this setup validates the claim

We evaluate on DMLab because it provides ground-truth camera poses, enabling direct test of decoupled camera control. FVD is chosen as it captures both frame quality and temporal coherence, essential for world models. Comparing against VideoSSM (autoregressive without explicit camera control) isolates the benefit of our camera branch; against Fast AR Video Diffusion, we test parity with non-autoregressive generation. The ablation removes cross-attention and the regularization loss to verify their roles. The intervention test directly probes decoupling: we corrupt camera poses during inference and measure the change in generated frames. If frame changes correlate with camera pose changes, the model is genuinely decoupled. This combination forms a falsifiable test: if CDWM excels specifically on camera-heavy sequences and passes the intervention test, the claim is supported; otherwise, the decoupling assumption fails.

### Expected outcome and causal chain

**vs. VideoSSM** — On a case where camera rotates rapidly, VideoSSM (which models camera implicitly via actions) produces blurred frames because its recurrent state cannot disentangle camera motion from agent actions. Our method, with cross-attention to explicit camera embeddings and decoupling regularization, preserves sharpness by conditioning frame prediction on independent camera signals. We expect a noticeable FVD gap (e.g., 20+ points) on high-motion camera clips, but parity on static camera clips.

**vs. Fast AR Video Diffusion** — On a case requiring precise zoom trajectory, the diffusion baseline generates inconsistent zoom because its noising process destroys camera continuity, while our autoregressive decoder uses camera pose at each step to maintain consistent viewpoint. Thus, we expect CDWM to achieve lower FVD (e.g., 10+ points) on clips with non-trivial camera paths, with similar quality on simple camera settings.

**Intervention test** — When we randomly perturb camera poses while keeping actions fixed, our model should produce frames that reflect the new viewpoint, whereas baselines will show inconsistent or unchanged frames. We will quantify this via a change correlation metric: the cosine similarity between camera change and frame change vectors.

### What would falsify this idea

If CDWM fails to outperform VideoSSM on high-camera-motion subsets, or if the ablation without cross-attention and regularization matches the full model on such subsets, then the decoupling claim is invalid because the camera branch is not providing a measurable benefit. Additionally, if the intervention test shows that frame changes do not correlate with camera pose perturbations, decoupling is not achieved.

## References

1. AlayaWorld: Long-Horizon and Playable Video World Generation
2. VideoSSM: Autoregressive Long Video Generation with Hybrid State-Space Memory
3. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models
4. Autoregressive Video Generation without Vector Quantization
