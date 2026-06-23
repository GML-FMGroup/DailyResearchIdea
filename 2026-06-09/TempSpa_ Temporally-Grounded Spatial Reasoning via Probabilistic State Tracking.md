# TempSpa: Temporally-Grounded Spatial Reasoning via Probabilistic State Tracking

## Motivation

Existing spatial reasoning models treat scenes as static snapshots, failing under dynamic occlusions and layout changes. For instance, CAPTURe evaluates static occluded counting, but real-world embodied agents must track objects over time as they move and occlude each other. This temporal blindspot prevents robust reasoning in interactive environments.

## Key Insight

Spatial relations and object states are deterministically coupled through shared latent dynamics, so jointly inferring them as a posterior over a single temporal generative model ensures consistency across time and occlusion.

## Method

### (A) What it is
TempSpa is a differentiable particle filter that maintains a set of particles representing the joint state of all objects (positions, velocities, occlusion masks) and their pairwise spatial relations. It takes as input a stream of egocentric images and outputs a temporally consistent spatial relation graph with object occupancy estimates.

### (B) How it works
```
# TempSpa: Temporal Spatial Reasoning via Particle Filter (Rao-Blackwellized)
# Input: stream of frames I_1..I_T, initial object set O_0
# Output: per-timestep spatial relation graph R_t and state estimates S_t

# Architecture specifics:
# - neural_transition: 2-layer GRU with hidden size 256, input=[state, action], output=delta
# - neural_relation_predictor: 3-layer MLP [256,128,64] with ReLU, output softmax over 5 relations
# - likelihood network: ResNet-18 pretrained on ImageNet, global avg pool + linear layer to scalar log-likelihood
# - Training: Adam lr=1e-4, batch=32, 100 epochs on SOD (10k sequences of 20 frames, 256x256)
# - Loss: negative log-likelihood + L2 smoothness (0.001)

1. Initialize:
   For each particle k = 1..K (K=1000, chosen to mitigate curse of dimensionality; effective sample size (ESS) monitored):
       Sample initial continuous state s_cont,k^0 = [pos_i, vel_i] for each object i (by Kalman prior)
       Sample initial discrete occlusion mask m_k^0 = [occ_mask_i] for each object i (by Bernoulli(0.5))
       Combined state s_k^0 = (s_cont,k^0, m_k^0)
       Initial relation graph r_k^0 = empty (or prior)

2. For each time step t = 1..T:
   # Prediction step (transition model for continuous part via Kalman, discrete via learned transition)
   For each particle k:
       # Rao-Blackwellized: continuous part updated with Kalman filter equations (linear Gaussian dynamics)
       s_cont,k^t = Kalman_predict(s_cont,k^{t-1}, action=None) + Gaussian noise (std=0.1)
       # Discrete part updated via neural transition MLP
       m_k^t = neural_transition_discrete(m_k^{t-1}, s_cont,k^t)  # outputs logits for each mask bit
       # Relation predictor
       r_k^t = neural_relation_predictor(s_cont,k^t, m_k^t)  # MLP

   # Update step (observation model)
       weight_k^t = p(I_t | s_cont,k^t, m_k^t) via ResNet-18 likelihood network (outputs log-likelihood)
   Normalize weights: w_k^t = weight_k^t / sum_j weight_j^t
   Monitor effective sample size: ESS = 1 / sum(w_k^t^2). If ESS < 100, perform resampling even if not scheduled.

   # Resample particles (systematic resampling) if ESS < threshold or scheduled every step
       New particles drawn with probabilities proportional to w_k^t
       # For Rao-Blackwellized, resample only discrete part, continuous part remains as Kalman mixtures

   # Aggregate relation graph
       R_t = weighted_mean(r_k^t, weights=w_k^t)
   # Aggregate state estimates
       S_t = weighted_mean(s_cont,k^t, weights=w_k^t)  # marginalizing over masks

3. Return sequence {R_t, S_t} for t=1..T
```

### (C) Why this design
We chose a Rao-Blackwellized particle filter over a plain particle filter to mitigate the curse of dimensionality: the continuous part (positions, velocities) is handled by exact Kalman updates, reducing the effective dimension sampled by particles to just the discrete occlusion masks (3 objects → 3 bits). The observation model is non-linear (ResNet-18 on pixels), and the discrete masks create multi-modality, justifying the particle filter. We chose a learned transition model for discrete masks because occlusion dynamics are task-specific. We infer spatial relations from the combined state (continuous + discrete) to ensure temporal consistency. A critical assumption is that K=1000 particles suffice to cover the discrete state space (2^3 = 8 possibilities, easily covered) and that the Kalman filter correctly handles continuous dynamics; we verify this by monitoring effective sample size on a validation set (if ESS < 10% of K, we increase K or revert to plain particle filter with more particles). Hyperparameters (K=1000, noise std=0.1) balance accuracy and runtime (single RTX 3090, training ~2 days).

### (D) Why it measures what we claim
The particle weight p(I_t | s_k^t, r_k^t) measures *temporal consistency* because a state that correctly explains occlusion patterns across frames will have consistently high likelihood; this assumes the likelihood network (ResNet-18) captures all relevant image evidence, which fails when the network ignores subtle shape cues (under occlusion, weight may plateau). Additionally, the assumption that K=1000 particles with Rao-Blackwellization maintain a diverse posterior is critical; if the discrete state space grows (more objects), ESS may drop, and the weight reflects only the surviving particles. The aggregated relation graph R_t measures *spatial reasoning accuracy* by averaging over particles, reflecting the most probable relation given all evidence up to time t; this assumes the particle posterior covers plausible states, which holds due to Rao-Blackwellization. The transition model enforces *motion smoothness*; this assumes Markovian dynamics, which fails under sudden camera cuts, causing prediction drift.

## Contribution

(1) A temporally-grounded spatial reasoning framework (TempSpa) that jointly tracks object states and infers spatial relations via a differentiable particle filter. (2) The identification that modeling temporal coherence through probabilistic state tracking resolves dynamic occlusion failures in static spatial reasoning benchmarks. (3) A new evaluation protocol for dynamic spatial reasoning using video sequences with ground-truth object trajectories and occlusion masks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Synthetic Occlusion Dynamics (SOD) | Controlled occlusion and motion patterns; 10k sequences, 20 frames each, 256x256. |
| Primary metric | Relation F1 score | Measures multi-class spatial relation accuracy (5 classes). |
| Baseline | Frame-wise ResNet-18 (same likelihood network, no temporal) | Tests need for temporal modeling. |
| Baseline | Kalman filter with learned observation (unimodal Gaussian) | Tests need for multi-modal posterior. |
| Ablation-of-ours | TempSpa without resampling (i.e., importance sampling) | Tests importance of particle diversity. |

### Why this setup validates the claim
The dataset SOD includes sequences with varying occlusion durations (2–15 frames) and non-linear motion (sudden direction changes), providing a controlled environment to test spatial reasoning under temporal ambiguity. The frame-wise baseline isolates the benefit of temporal consistency; the Kalman filter baseline tests the necessity of multi-modal state representations; the ablation without resampling tests the necessity of particle diversity. The F1 score on spatial relations directly quantifies the accuracy of the output graph, which is the core claim. If our method outperforms these baselines specifically on subsets where their assumed mechanisms fail (e.g., long occlusion for frame-wise, bimodal motion for Kalman), then the claim that temporal consistency and multi-modality are key is validated. We additionally monitor effective sample size (ESS) to ensure the particle filter remains diverse; if ESS drops below 100, we flag potential degeneracy.

### Expected outcome and causal chain

**vs. Frame-wise ResNet-18** — On a case where an object is fully occluded for 8 frames, the ResNet, lacking temporal memory, will output a random relation when it reappears because each frame is processed independently. Our method maintains particles that propagate state through occlusion using the learned transition model, so upon re-emergence the relation is correctly inferred from the consistent particle set. We expect a large gap (e.g., >20% F1) on sequences with long occlusions (≥5 frames) and parity on fully visible sequences.

**vs. Kalman filter** — On a scenario where an object suddenly changes direction (e.g., bouncing), the Kalman filter’s unimodal Gaussian posterior cannot represent the two possible velocities, causing the estimate to drift and relation to flip. Our Rao-Blackwellized particle filter maintains multiple hypotheses for the discrete occlusion mask, and the continuous part (handled by Kalman) can switch modes via the discrete state changes. We expect the Kalman filter to suffer a steep drop on turning sequences (e.g., 15% lower F1) while our method remains stable.

**Ablation: TempSpa without resampling** — On a long sequence (50 steps) with random motion, particles without resampling degenerate as weights concentrate, losing diversity and failing to track abrupt changes. Our method with resampling (ESS monitoring ensures diversity) prevents mode collapse. We expect the ablation’s F1 to decline over time (e.g., 10% lower after 30 steps) while full TempSpa stays high.

### What would falsify this idea
If our method shows no significant improvement over the frame-wise baseline on occluded subsets, or if the ablation without resampling performs equally well, then the claimed mechanisms of temporal consistency and multi-modal posterior are not critical for spatial reasoning. If the effective sample size consistently drops below 100 despite K=1000, the Rao-Blackwellization assumption fails and the method needs redesign.

## References

1. CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting
2. InternSpatial: A Comprehensive Dataset for Spatial Reasoning in Vision-Language Models
3. Are We on the Right Way for Evaluating Large Vision-Language Models?
4. Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models
5. MMMU: A Massive Multi-Discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI
6. RoboLayout: Differentiable 3D Scene Generation for Embodied Agents
7. LayoutVLM: Differentiable Optimization of 3D Layout via Vision-Language Models
8. LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning
