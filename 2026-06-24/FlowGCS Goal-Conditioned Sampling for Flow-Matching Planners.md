# FlowGCS: Goal-Conditioned Sampling for Flow-Matching Planners via Sparse Binary Feedback

## Motivation

FlowR2A demonstrates that reward-conditioned flow matching enables controllable multimodal planning, but its reliance on dense per-timestep reward labels is infeasible in real-world deployment where only sparse trajectory-level success/failure feedback is available. The structural limitation is that existing flow-matching planners assume a dense reward signal at each step, which cannot be replaced by simple goal conditioning because the flow model's invertible backbone is not exploited to learn a latent goal space from sparse feedback. This work addresses the forest-level bottleneck of supervision sparsity while retaining controllability and diversity.

## Key Insight

The invertible structure of rectified flow models establishes a bijection between trajectories and latent goal embeddings, enabling sparse binary success feedback to shape the latent goal space via cyclic consistency without per-step reward labels.

## Method

(A) **What it is** – FlowGCS is a flow-matching planner that learns a latent goal space from sparse trajectory-level binary success feedback. It comprises a goal encoder that maps a trajectory to a latent goal embedding, and a flow decoder that generates trajectories conditioned on the goal and initial state. Both are trained via cyclic consistency and contrastive learning, eliminating the need for dense per-step rewards.

(B) **How it works**
```python
# Hyperparameters: latent dim d=64, flow steps T=128, contrastive margin m=0.5, λ_cyc=0.1, batch size B=64, learning rate=1e-4.
# Encoder: 2-layer MLP, hidden=256, GeLU activation, output dim 64.
# Flow decoder: 4-layer Transformer, d_model=128, nhead=4, feedforward=512, with sinusoidal positional encoding for timestep.
# Training loop for each trajectory-label pair (τ=[s_1,...,s_N], y∈{0,1})
1. Encode trajectory: g = Encoder(τ)  # MLP to 64-dim embedding
2. Sample random other trajectories τ_neg from batch; compute negative embeddings g_neg
3. Contrastive loss for successful (y=1): L_contr = max(0, ||g - g_anchor||² - ||g - g_neg||² + m) where g_anchor is mean embedding of successful trajectories in batch
4. For flow decoder: sample noise ε ~ N(0,I) and timestep t ~ Uniform(0,1)
5. Decode: compute velocity v_θ(z_t, t, g, s_0) where z_t = t*τ + (1-t)*ε, and s_0 is initial state (first position and heading). The transformer outputs velocity field conditioned on goal g and initial state s_0 via cross-attention.
6. Flow matching loss: L_flow = E[||v_θ(z_t, t, g, s_0) - (τ - ε)||²]
7. Cyclic consistency: sample z_0 = random goal g' (from prior N(0,I)) or from another trajectory's goal, then decode to trajectory τ' = solve ODE with v_θ (using Euler steps, 128 steps), then encode g'' = Encoder(τ'), minimize ||g' - g''||²
8. Total loss: L_total = L_contr + L_flow + λ_cyc * L_cyc

# Inference:
- For a desired behavior, sample goal g from prior or interpolate between known successful goals
- Generate trajectory by running reverse ODE from noise with v_θ conditioned on g and current state (initial state s_0 taken from current observation)
```

(C) **Why this design**
We chose a contrastive objective to structure the latent goal space over a regression-based goal prediction because contrastive learning inherently separates successful from unsuccessful trajectories while clustering similar successful behaviors, enabling diversity—accepting that contrastive margins require tuning but avoiding mode collapse from L2 regression. We use cyclic consistency (encoding-decoding-encoding) instead of direct reconstruction loss on the trajectory because the latter would require the decoder to perfectly memorize each trajectory, which hurts generalization to novel start states; cyclic consistency forces the latent goal to be robust to trajectory variations. We condition the flow decoder on both the goal and the initial state rather than on the goal alone because the initial state provides the anchor for the probability path—otherwise the ODE must learn to translate noise to different starting positions, increasing optimization difficulty. The trade-off is that initial-state conditioning ties the decoder to a specific starting point, limiting direct reuse across different initial configurations, but this is acceptable in planning where the initial state is always available.

(D) **Why it measures what we claim**
The contrastive loss L_contr measures goal-space separation between successful and failure trajectories because it uses the assumption that success/failure labels are correlated with semantic driving intent; this assumption fails when a trajectory fails for reasons unrelated to the goal (e.g., an unavoidable accident beyond the planner's control), in which case the loss pushes those trajectories away from the success cluster even though they shared the same intent, potentially creating a spurious cluster. To validate this, we propose a human annotation study on a random subset of 500 nuScenes trajectories: annotators will label the intended behavior (e.g., "go straight", "turn left", "yield") for each trajectory, and we will measure the alignment between human intent clusters and the learned goal embeddings via adjusted Rand index. The flow-matching loss L_flow measures the accuracy of the generative mapping from goal and noise to trajectories under the assumed probability path; the assumption that the velocity field v_θ is well-approximated by a transformer holds for continuous trajectories but fails under discontinuous or heavily multimodal dynamics, where the conditional expectation of velocity becomes ill-defined, and the loss then reflects a poor fit that degrades generation quality. The cyclic consistency loss L_cyc measures the invertibility of the flow decoder with respect to the goal encoder; it assumes that the encoder-decoder pair forms a nearly bijective mapping, which holds only when the trajectory distribution is unimodal per goal—if multiple distinct successful trajectories share the same goal (multimodality), L_cyc penalizes the encoder for not compressing them to a single point, causing the encoder to split goals arbitrarily, leading to degraded controllability. This unimodality assumption is explicitly stated and will be tested by measuring the variance of encoded goals within each ground-truth goal region.

## Contribution

(1) A novel flow-matching planning framework, FlowGCS, that learns a latent goal space from sparse binary success feedback using contrastive learning and cyclic consistency, eliminating the need for per-timestep reward labels. (2) A demonstration that the invertible property of rectified flow models can be exploited to enable implicit goal conditioning, providing a principled alternative to dense reward conditioning for controllable trajectory generation. (3) A design principle for connecting sparse supervision to generative planners via bijective encoding, applicable beyond driving to other sequential decision making domains.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | nuScenes mini-train (4000 scenes) and validation (600 scenes) | Standard real-world driving benchmark with 20s trajectories. |
| Primary metric | Goal-conditioned success rate: fraction of trajectories where final position within 2m of goal, no collisions, no off-road | Measures goal adherence under diversity and safety. |
| Baseline 1 | FlowR2A (default config: T=128, d=64, regression goal predictor) | Flow matching without contrastive goal space. |
| Baseline 2 | DenseTNT (15 anchors, default settings) | Anchor-based multimodal planner. |
| Baseline 3 | VAE + sparse reward: encoder maps trajectory to latent, decoder reconstructs, trained with ELBO and sparse reward weighting (β-VAE with β=1) | Tests if invertible flow is essential beyond latent variable models. |
| Ablation of ours | FlowGCS w/o contrastive loss (only L_flow + L_cyc) | Tests importance of structured goal space. |

### Why this setup validates the claim

This setup directly tests the central claim that structuring the latent goal space via contrastive learning and cyclic consistency improves goal-conditioned planning under sparse binary feedback. The nuScenes dataset provides realistic multimodal driving scenarios where success/failure labels are trajectory-level. FlowR2A is the closest prior work; comparing against it isolates the benefit of the contrastive goal space over a standard flow matching without such structure. DenseTNT represents a strong anchor-based baseline that does not learn a latent goal embedding, testing whether the learned representation offers advantages over heuristic anchor sets. The VAE+sparse reward baseline directly evaluates whether the invertible flow structure is essential—if it matches FlowGCS, then the invertibility is not key; if it underperforms, the flow model's bijective property is crucial. The ablation (removing contrastive loss) pinpoints whether observed gains stem from the contrastive objective or other components. The primary metric, goal-conditioned success rate, directly measures the method’s ability to generate trajectories that achieve the intended goal, which is the core purpose of the proposed planner. Success rate is a robust single-number summary that captures both safety and goal attainment, and it is commonly reported in planning literature. The combination of dataset, baselines, and metric thus provides a clear falsifiable test: if our method outperforms all baselines and the ablation on this metric, the claim is supported; if not, it is challenged.

### Expected outcome and causal chain

**vs. FlowR2A** — On a case where the agent must navigate a complex intersection with multiple feasible but distinct successful trajectories, FlowR2A’s flow decoder trained with regression-based goal prediction tends to average modes, producing a trajectory that is ambiguous or unsafe. Our method, by contrast, uses contrastive learning to separate distinct successful behaviors in goal space and cyclic consistency to ensure each goal maps back to a distinct trajectory, so the decoder can generate sharp multimodal trajectories for different goals. We expect a noticeable gap (e.g., 10–15% higher success rate) on scenarios with high multimodality, but parity on near-deterministic paths.

**vs. DenseTNT** — In a scenario where the ego vehicle must yield to multiple pedestrians with unpredictable motion, DenseTNT relies on fixed anchors that may not capture the exact future path, leading to either overly cautious or aggressive plans. Our method learns a continuous latent goal space from feedback, allowing flexible adaptation to the specific situation; the flow decoder generates smooth trajectories conditioned on the learned goal. We expect our method to achieve 8–12% higher success rate in such dynamic scenarios, while DenseTNT may perform competitively in static environments.

**vs. VAE + sparse reward** — In a scenario where multiple distinct successful trajectories share the same goal (e.g., reaching a specific lane offset after a lane change), the VAE's stochastic encoder can assign the same goal embedding to different trajectories, but the decoder may produce noisy or inconsistent outputs because it lacks the invertible flow structure that ties a single embedding to a unique trajectory. Our method's cyclic consistency enforces a locally bijective mapping, leading to more precise and diverse generation. We expect FlowGCS to achieve 5–8% higher success rate than the VAE baseline, especially in tasks requiring fine-grained control.

**vs. Ablation (w/o contrastive loss)** — When the goal space is learned without contrastive loss (only flow matching and cyclic consistency), the encoder tends to map all successful trajectories to a similar embedding (mode collapse), reducing the ability to control diverse behaviors. As a result, the ablation produces trajectories that are less goal-consistent upon conditioning. We expect a drop of 5–7% in goal-conditioned success rate relative to full FlowGCS, particularly noticeable when the evaluation requires generating a specific goal among several possible intents.

### What would falsify this idea

If the full FlowGCS method does not outperform the ablation (no contrastive loss) on the primary metric by a statistically meaningful margin (p<0.05 via bootstrap), then the central claim about the necessity of contrastive learning for goal space structuring would be falsified. Alternatively, if the gain over FlowR2A is uniform across all scenario types rather than concentrated in highly multimodal cases, the hypothesized causal chain would be refuted. Additionally, if the VAE baseline matches FlowGCS performance, the claim that invertible flow is essential would be falsified.

## References

1. FlowR2A: Learning Reward-to-Action Distribution for Multimodal Driving Planning
2. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
3. Pseudo-Simulation for Autonomous Driving
4. Diffusion Guidance Is a Controllable Policy Improvement Operator
5. Boosting Latent Diffusion with Flow Matching
6. CogVLM: Visual Expert for Pretrained Language Models
7. DriveArena: A Closed-Loop Generative Simulation Platform for Autonomous Driving
8. STORM: Spatio-Temporal Reconstruction Model for Large-Scale Outdoor Scenes
