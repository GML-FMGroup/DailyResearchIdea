# Latent Particle Dynamics with Conservation Constraints for Unsupervised Video Dynamics Learning

## Motivation

Existing methods for learning physical dynamics from video, such as PhysisForcing, rely on external structured supervision (point trajectories or frozen encoders) to enforce physical consistency. This limits generalization to diverse scenarios and requires costly annotations. The root cause is that these approaches use heuristic losses that do not exploit intrinsic physical invariants. We address this by directly encoding conservation laws (momentum, mass) into a latent particle representation, enabling unsupervised learning of accurate dynamics from raw video.

## Key Insight

By enforcing exact momentum and mass conservation in a differentiable latent particle simulation, the model's forward pass is itself physically consistent, eliminating the need for external supervision because any deviation from observed frames must be due to perception error or unmodeled forces.

## Method

### (A) What it is
Conservation Particle Dynamics (CPD) is an unsupervised framework that learns physical dynamics from raw video by embedding frames into a latent particle representation and simulating their evolution using a differentiable physics engine with built-in conservation laws. Input: raw video frames. Output: predicted future frames and recovered physical parameters (mass, velocity) of particles.

Load-bearing assumption: A fixed set of N=100 particles with constant masses per particle can accurately represent the physical state of arbitrary rigid and articulated objects in a scene for simulating dynamics with conservation laws. To verify this assumption, we conduct a calibration experiment on a held-out subset of 512 frames from R-Bench, measuring the reconstruction MSE and particle overlap ratio. If the assumption holds, reconstruction error should be low; if violated (e.g., due to deformable objects or object splits), we flag the scene as requiring a dynamic particle allocation mechanism.

### (B) How it works
```pseudocode
# Training loop for CPD
for each video clip {I_0, I_1, ..., I_T}:
  # Encode frame I_t into particle states
  # Encoder E: image -> set of N particles (position, velocity, mass)
  for t in 0..T-1:
    {p_t^i, v_t^i, m^i} = E(I_t)   # m^i shared over time
    # Differentiable physics step (conservation enforced)
    dt = 0.1
    for each particle i:
      v_next^i = v_t^i + (F_ext_i / m^i) * dt   # F_ext_i = 0 by default
    # Collision detection and elastic collision
    for each pair (i,j):
      if norm(p_t^i - p_t^j) < r_i + r_j:  # r_i = 0.05 (fixed radius per particle)
        # Elastic collision: conserve momentum and kinetic energy
        v1f, v2f = elastic_collision(v_next^i, v_next^j, m^i, m^j)
        v_next^i, v_next^j = v1f, v2f
    # Update positions
    p_{t+1}^i = p_t^i + v_next^i * dt
  # Render predicted frame I'_t+1 from particles
  I'_t+1 = render({p_{t+1}^i, m^i})   # differentiable soft rasterizer
  # Loss computation
  L = MSE(I'_t+1, I_t+1) + 0.01 * sum(m^i - 1)^2   # mass regularization
  # Backprop through encoder, renderer, and physics step
  update E, masses, renderer parameters
```
Hyperparameters: N=100 particles, dt=0.1, learning rate=1e-4, perception loss weight=1.0, mass regularization weight=0.01. Encoder E: 4-layer CNN with 64 channels, followed by a transformer head that outputs particle states. Renderer: differentiable soft rasterizer using a Gaussian splatting kernel with sigma=0.02.

### (C) Why this design
We chose a latent particle representation over a dense grid or scene graph because particles can capture discrete object interactions and are naturally suited for enforcing momentum conservation per object. We use a differentiable elastic collision model rather than learned collision handling because conservation laws are analytically known, avoiding the need for annotated collision data. We opted for a shared mass per particle across time instead of per-frame mass estimation because mass is an intrinsic object property that should be constant; this reduces parameters and enforces temporal consistency. The trade-off of using a finite particle count is that very deformable objects may not be well represented; we accept that our method targets rigid and articulated objects where particle approximation is valid. In contrast to PhysisForcing, we do not require point trajectories or a frozen encoder: the conservation constraints themselves provide the physical grounding.

### (D) Why it measures what we claim
The differentiable physics step with elastic collision enforces momentum conservation exactly because the collision update is derived from Newton's laws; this means that the predicted dynamics are physically consistent by construction. The reconstruction loss measures the discrepancy between simulated and observed frames; when conservation is enforced, the only way to reduce this loss is for the particle encoder to extract physically meaningful trajectories from the video. The computational quantity `v_next^i` from elastic collision measures momentum conservation because we assume the collision model correctly captures the underlying physics; this assumption fails when forces are non-conservative (e.g., friction), in which case the model either learns to compensate via external forces or the loss remains high. The mass regularization `sum(m^i - 1)^2` measures mass consistency because we assume mass is constant over time; this assumption fails if objects split or merge, in which case the model's fixed-particle topology breaks. By embedding conservation directly in the simulation, we avoid the proxy problem of external supervision: rather than relying on a separate encoder to measure physical plausibility, the model's forward pass is itself physically plausible, so any deviation from observation must be due to perception error or unmodeled forces.

## Contribution

(1) A framework for unsupervised learning of physical dynamics from raw video by enforcing conservation laws in a latent particle representation. (2) A differentiable physics engine that exactly conserves momentum and mass, integrated with a neural renderer for end-to-end training without external supervision. (3) Demonstration that conservation constraints provide a strong inductive bias, enabling accurate dynamics prediction without trajectory or pose annotations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Initial validation dataset | Synthetic bouncing balls (10k videos, 10 frames) | Ensures correct dynamics before complex benchmarks |
| Primary dataset | R-Bench | Tests physical dynamics in manipulation |
| Primary metric | Momentum Conservation Violation (MCV) | Directly measures momentum conservation |
| Baseline 1 | PhysisForcing | Enforces physics via trajectory supervision |
| Baseline 2 | Cosmos-Predict1 | General video model without physics priors |
| Baseline 3 | Neural Physics Engine (NPE) | Particle-based with object-centric supervision |
| Ablation | CPD w/o elastic collision | Removes collision conservation constraint |

### Why this setup validates the claim
We first validate CPD on a synthetic bouncing balls dataset (10k videos, 10 frames each) to ensure correct dynamics before testing on R-Bench. The combination of R-Bench (which contains rigid and articulated objects with frequent collisions) with MCV as a metric directly tests the central claim that built-in conservation laws produce physically consistent predictions. PhysisForcing uses external trajectory labels to enforce physics, so if CPD matches or surpasses it without such supervision, it demonstrates that conservation alone suffices. Cosmos-Predict1 lacks any physics priors, so outperforming it shows the value of explicit conservation. Neural Physics Engine (NPE) requires object-centric supervision; surpassing it shows that unsupervised conservation can achieve similar physical consistency. The ablation (removing elastic collision) isolates the effect of the collision model. MCV specifically quantifies momentum violation, so any improvement over baselines is attributable to conservation enforcement. This setup falsifiably tests whether endogenous physical grounding via differentiable simulation outperforms learned or supervised alternatives.

### Expected outcome and causal chain

**vs. PhysisForcing** — On a collision-heavy scene (e.g., two blocks colliding) where no trajectory labels exist for that interaction, PhysisForcing may predict post-collision velocities that violate momentum conservation because it relies on supervised trajectory pairs; if the training distribution lacks this exact collision, its learned dynamics can be unrealistic. Our method enforces elastic collision analytically via Newton's laws, so it conserves momentum by construction. We expect CPD’s MCV to be near zero on such frames, while PhysisForcing’s MCV spikes (e.g., 0.05 vs 0.15), demonstrating that explicit conservation yields more physically consistent predictions without supervision.

**vs. Cosmos-Predict1** — On a long-horizon prediction of a bouncing ball, Cosmos-Predict1, as a general video foundation model, may produce visually plausible but physically implausible trajectories (e.g., energy drift or unnatural damping) because it lacks any physical constraints. Our method simulates particle dynamics with conservation laws, so momentum and energy remain constant over time. We expect CPD’s MCV to remain low and stable across frames (e.g., <0.01), while Cosmos-Predict1’s MCV increases with horizon length (e.g., 0.05 at frame 20), showing that built-in conservation prevents cumulative physical errors.

**vs. Neural Physics Engine (NPE)** — On a scene with multiple objects, NPE may require object-centric annotations to separate particles per object; without such supervision, it might incorrectly aggregate particles. Our method uses unsupervised particle encoding with conservation, which forces particles to represent distinct objects via motion segmentation. We expect CPD to have similar or lower MCV than NPE even without its supervised object separation (e.g., 0.02 vs 0.03), demonstrating that conservation alone can guide particle grouping.

**vs. CPD w/o elastic collision** — On a case where two objects approach each other, the ablation (which replaces elastic collision with a learned module) may fail to correctly resolve the collision if such events are rare in the training data; it might predict overlapping particles or unrealistic scattering. Our full method uses the known elastic collision formula, guaranteeing correct momentum transfer. We expect the full CPD to have near-zero MCV on collision frames (e.g., 0.005), while the ablation shows a significant increase (e.g., 0.1), confirming that analytical collision enforcement is critical for physical consistency.

### What would falsify this idea
If the full CPD method does not achieve significantly lower MCV than both baselines on collision-dense subsets of R-Bench, or if the ablation performs similarly to the full method, then the central claim that built-in conservation laws are key to physical consistency is falsified. Additionally, if CPD fails on the deformable subset of R-Bench (with MCV > 0.1), then the fixed-particle assumption is violated, indicating a need for an adaptive particle mechanism.

## References

1. PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation
2. World Simulation with Video Foundation Models for Physical AI
