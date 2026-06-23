# Language-Conditioned Dual Flow Matching with Interleaved FiLM Gating for Joint Video-Action Generation

## Motivation

Existing joint video-action models like DiT4DiT couple video and action Diffusion Transformers via fixed hidden-state extraction at a single timestep, which lacks language-adaptivity. This structural limitation causes poor generalization to out-of-distribution tasks because the coupling cannot dynamically prioritize visual features relevant to novel language instructions. We address this by embedding FiLM gating layers between the video and action DiTs, enabling the model to selectively route features based on language embeddings.

## Key Insight

Multiplicative FiLM gating creates a language-adaptive coupling that allows the action stream to attend to task-relevant visual dynamics while suppressing irrelevant ones, a property that fixed hidden-state coupling cannot achieve.

## Method

**A) What it is:** We propose **LaFiLM-DiT** (Language-Conditioned FiLM Dual Diffusion Transformer), a dual flow-matching architecture that interleaves FiLM gating layers between video and action DiT blocks. It takes language instruction embeddings, a starting image, and noise, and outputs a joint video sequence and action sequence. **B) How it works:** 

```pseudocode
# LaFiLM-DiT forward pass at training time
Input: language L, initial image I0, noise for video ε_v, noise for action ε_a
Initialize: video latent z_v = encode(I0) + ε_v, action latent z_a = ε_a
For t in 1..T (timesteps from noise to data):
  # Video DiT block
  h_v = VideoDiT_block(z_v, t, L)  # conditional on language via cross-attention
  # FiLM gating: modulate video features with language
  γ, β = FiLM_net(L)               # learnable affine parameters per channel (FiLM_net: 2-layer transformer encoder with hidden dim 256, 4 attention heads)
  h_v_film = γ * h_v + β           # language-adaptive scaling and shifting
  # Action DiT block with FiLM-modulated video features
  h_a = ActionDiT_block(z_a, t, L, h_v_film)  # cross-attend to video features
  # Predict velocity (flow matching)
  v_v_pred = VideoDiT_output(h_v_film)
  v_a_pred = ActionDiT_output(h_a)
  # Update latents via flow matching ODE step (Euler)
  z_v = z_v + (v_v_target - v_v_pred) / (T - t)  # simplified; actual flow-matching loss
  z_a = z_a + (v_a_target - v_a_pred) / (T - t)
Output: video = decode(z_v), actions = decode(z_a)

Hyperparameters: T=256 diffusion steps, FiLM_net: 2-layer transformer encoder with hidden dim 256, 4 attention heads, VideoDiT/ActionDiT: DiT-B/2 with 12 blocks, FiLM layers inserted after every 4th VideoDiT block.
```

**C) Why this design:** We chose FiLM gating over simple concatenation of language features because FiLM provides multiplicative interactions that can dynamically amplify or suppress entire feature channels, which is critical for selectively routing information based on instruction semantics. We interleave FiLM layers every 4th block (not every block) to balance computational cost and modulation granularity, accepting that some blocks operate without language modulation, which prevents overfitting to training instructions. We use a separate FiLM network (instead of reusing the video DiT's cross-attention outputs) to decouple the gating signal from the video dynamics, ensuring that the modulation remains task-relevant and does not drift due to visual noise. These trade-offs collectively ensure that language conditioning is both potent and stable across diverse tasks. **Load-bearing assumption:** A separate network (originally a 2-layer MLP, now a small transformer encoder) that transforms language embeddings can generate FiLM parameters that effectively modulate video features for action generation, even for compositional instructions. To calibrate, we verify that the transformer-encoded FiLM parameters yield at least 10 distinct modulation patterns across 100 held-out instructions, ensuring nonlinear composition capture. **D) Why it measures what we claim:** The computational quantity `h_v_film = γ * h_v + β` measures **language-adaptive feature modulation** because the scaling γ and shift β are directly computed from language embeddings L; this assumes that a transformation (linear with MLP, or nonlinear with transformer) of the language embedding is sufficient to encode task-relevant visual priorities (e.g., which object to attend). This assumption fails when language instructions require nonlinear composition (e.g., 'pick the red block after moving the blue one'), in which case the metric reflects a simpler correlation rather than true compositional ability. However, by using a transformer-based FiLM network, we mitigate this failure to some extent, as the transformer can capture pairwise interactions. The cross-attention from the action DiT block to `h_v_film` measures **task-relevant visual grounding** because it aggregates FiLM-modulated video features; this assumes that the action stream can extract the relevant spatial-temporal cues from the modulated representation, but fails if the modulation introduces visual artifacts that mislead the attention.

## Contribution

(1) A novel architecture, LaFiLM-DiT, that interleaves FiLM gating layers between video and action DiT blocks for language-adaptive joint video-action generation. (2) A design principle that FiLM-based multiplicative gating, when interleaved at sparse intervals, provides better out-of-distribution generalization than fixed hidden-state coupling. (3) An empirical demonstration that language-conditioned feature modulation improves robustness to novel task instructions without increasing model size.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | LIBERO-10 | 10 tasks with language instructions and videos. |
| Primary metric | Task success rate | Direct measure of policy quality. |
| Baseline 1 | DiT4DiT | Joint video-action DiT without FiLM. |
| Baseline 2 | VideoVLA | Video generation + action prediction. |
| Ablation-of-ours | LaFiLM-DiT w/o FiLM | Replace FiLM with cross-attention only. |

### Why this setup validates the claim

This setup tests the central claim that FiLM gating from a separate network improves language-conditioned video-action generation by providing stable, task-relevant multiplicative modulation. LIBERO-10 offers diverse tasks requiring fine-grained language grounding, making it ideal to detect modulation benefits. Comparing against DiT4DiT (same architecture without FiLM) isolates the contribution of FiLM. VideoVLA represents a different family (two-stage video then action), testing whether joint modeling with FiLM is advantageous. The ablation (no FiLM) further separates the effect of FiLM from other design elements. Success rate directly reflects action quality, which measures whether modulated video features lead to better task execution.

### Expected outcome and causal chain

**vs. DiT4DiT** — On a case where two tasks have similar visual scenes but differ only in language (e.g., "pick the red cup" vs "slide the red cup"), DiT4DiT may confuse them because its additive language conditioning can be overwhelmed by visual salience. Our method uses FiLM to amplify task-critical features (e.g., motion affordances) and suppress irrelevant ones, producing more distinctive video representations. We expect a noticeable gap on such pairs with high visual overlap but opposite actions, but parity on tasks with unambiguous commands. Additionally, we will analyze the learned FiLM scaling factors (γ) per task to verify that they indeed suppress irrelevant visual features; we expect that for each task, channels corresponding to task-irrelevant objects have γ values significantly less than 1 (p<0.05 via t-test).

**vs. VideoVLA** — On a case requiring compositional reasoning like "move the blue block after pushing the green one", VideoVLA's separated video generation and action prediction may propagate errors because the video stream does not dynamically inform action selection. Our joint DiT with cross-attention from action to FiLM-modulated video ensures action-relevant temporal features directly guide action. We expect our method to achieve higher success on multi-step sequential tasks, while VideoVLA may excel on single-step tasks.

**vs. Ablation (no FiLM)** — The ablation uses only cross-attention from language to video DiT blocks, which can lead to overfitting on training instructions and weaker modulation. Our FiLM method should show improved zero-shot transfer to unseen language variants and more robust performance across environments with visual variations. We expect a consistent advantage on instruction pairs requiring selective feature emphasis.

### What would falsify this idea

If our gain over the ablation (no FiLM) is uniform across all instruction types rather than concentrated on tasks requiring precise feature modulation (e.g., visuomotor conflicts), or if DiT4DiT outperforms us on tasks with high visual diversity but simple language, then the central claim that FiLM provides selective and beneficial modulation is falsified.

## References

1. DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control
2. World Simulation with Video Foundation Models for Physical AI
3. VideoVLA: Video Generators Can Be Generalizable Robot Manipulators
4. π*0.6: a VLA That Learns From Experience
5. Diffusion Policy Policy Optimization
6. π0: A Vision-Language-Action Flow Model for General Robot Control
7. Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations
8. Open-Sora: Democratizing Efficient Video Production for All
