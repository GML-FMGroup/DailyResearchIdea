# Time-Reversible Latent Dynamics for Physically Plausible Video Generation

## Motivation

Current video world models, such as DreamGen, generate visually coherent motion but often violate physical laws like momentum conservation and rigid-body constraints, producing unnatural artifacts. The root cause is that these models treat time asymmetrically and lack an inductive bias for the time-reversible structure inherent in Newtonian dynamics. Without enforcing such structure, the learned latent dynamics can drift, leading to non-physical behaviors such as spontaneous acceleration or unrealistic collisions.

## Key Insight

Time-reversal symmetry provides a structured inductive bias that forces the latent dynamics to approximate Hamiltonian flow, which inherently conserves energy and momentum without requiring explicit physics simulation or differentiable physics engines.

## Method

**TRS-World** is a training framework that enforces time-reversal symmetry (TRS) in the latent dynamics of a video world model via cycle-consistency constraints between forward and backward latent trajectories. It takes as input a pretrained latent video diffusion model (e.g., DreamGen's world model) with an encoder-decoder and a latent dynamics network f, and outputs a symmetrized version of f that approximates invertible Hamiltonian dynamics.

**(B) How it works (pseudocode)**
```python
# Hyperparameters: lambda_cycle = 0.1 (cycle loss weight), T = 16 (sequence length)
# Let M be the base world model with encoder E, decoder D, latent dynamics f (residual net: z_{t+1} = z_t + phi(z_t))
# Let g be a small inverse dynamics network (MLP) that predicts z_t from z_{t+1}

for each video clip of T frames:
    # Encode to latents
    z_1..T = E(video_frames)  # shape: [T, d_latent]
    
    # Forward trajectory
    zhat_2..T = [f(z_t) for t in 1..T-1]
    
    # Backward trajectory (inverse prediction)
    zhat_back_1..T-1 = [g(z_{t+1}) for t in 1..T-1]
    
    # Cycle consistency: forward-backward composition
    cycle_fwd = [f(g(z_{t+1})) for t in 1..T-1]   # approximate identity on z_{t+1}
    cycle_bwd = [g(f(z_t)) for t in 1..T-1]       # approximate identity on z_t
    
    # Losses
    L_fwd = MSE(zhat_2..T, z_2..T)
    L_cycle = MSE(cycle_fwd, z_2..T) + MSE(cycle_bwd, z_1..T-1)
    L_total = L_fwd + lambda_cycle * L_cycle
    
    # Update f and g jointly via gradient descent (Adam, lr=1e-4)
```

**(C) Why this design**
We chose a cycle-consistency loss over alternative approaches (e.g., explicit Hamiltonian neural networks or Lyapunov-based constraints) for three reasons. First, cycle consistency is architecture-agnostic: it can be applied to any differentiable latent dynamics f without requiring the model to be invertible by design, avoiding the computational cost of normalizing flows or reversible networks. The trade-off is that cycle consistency only enforces approximate reversibility, which may allow small violations to accumulate over long sequences. Second, we separate the inverse dynamics g from f rather than tying weights, because tied inversion (e.g., using f^{-1} approximated by minimizing a residual) would require iterative optimization during training, increasing computational load. The cost of a separate g is that it introduces additional parameters, but we mitigate this by keeping g small (2-layer MLP with 256 hidden units). Third, we apply the cycle loss to all pairs (z_t, z_{t+1}) rather than only at the trajectory endpoints, because local reversibility is necessary for the global conservation properties we target. The trade-off is increased gradient variance due to multiple synthetic steps, but we counteract this with a low cycle loss weight (0.1) to stabilize training. Hyperparameters like lambda_cycle are chosen via a small grid search on a validation set, balancing forward prediction accuracy and reversibility.

**(D) Why it measures what we claim**
The cycle loss L_cycle measures approximate reversibility of the latent dynamics: if f and g are near-perfect inverses, then z_t can be recovered after a forward-backward pass, implying that the dynamics are close to symplectic and thus conserve a pseudo-energy. This works under the assumption that the latent space is a Euclidean subspace where MSE reflects physically meaningful distances (e.g., positions and velocities); this assumption fails when the latent representation is highly non-linear or aggregates multiple physical quantities into a single dimension, in which case the cycle loss may measure only statistical consistency rather than physical reversibility. The forward loss L_fwd ensures that the model still accurately predicts next latents; together, they constrain the flow to be both accurate and reversible. Specifically, the inverse dynamics network g operationalizes the concept of 'time-reversed transition': if we denote the true inverse as f^{-1}, then g is trained to approximate it. The cycle-consistency ensures that f and g are nearly inverses of each other, i.e., f ∘ g ≈ I and g ∘ f ≈ I. When the latent space captures the state of a physical system (position and momentum), this constraint implies that the dynamics preserve the symplectic form, leading to energy conservation. However, if the latent space is not aligned with the true physical state (e.g., it includes appearance features that are not time-symmetric), the cycle loss may merely encourage visual smoothness rather than physical reversibility.

## Contribution

(1) A novel training framework, TRS-World, that enforces time-reversal symmetry in latent dynamics of video world models via cycle-consistency, without requiring explicit physics simulation. (2) Empirical demonstration that videos generated with TRS-World exhibit significantly fewer physical violations (e.g., momentum drift, unnatural object motion) compared to the base DreamGen model and existing motion-enhancing methods like VideoJAM. (3) Introduction of a physics violation metric based on temporal consistency of latent momentum estimates, enabling quantitative evaluation of physical plausibility in generated video.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Physion (physics simulation videos) | Evaluates physical consistency with ground truth. |
| Primary metric | Forward prediction MSE at long horizons | Measures effects of reversibility on accumulation error. |
| Baseline 1 | DreamGen base world model (without TRS) | Baseline without symmetry constraints. |
| Baseline 2 | VideoJAM (motion-aware) | Strong motion prior baseline. |
| Baseline 3 | Hamiltonian Neural Network (explicit symplectic) | Explicit energy conservation baseline. |
| Ablation-of-ours | Ours without cycle loss (only forward loss) | Isolates benefit of cycle consistency. |

### Why this setup validates the claim

Physion provides video sequences with known physical dynamics and ground-truth states, allowing precise measurement of long-term prediction error. The primary metric—forward prediction MSE at long horizons—directly tests the central claim that cycle consistency reduces error accumulation due to approximate reversibility. The baselines cover a spectrum of approaches: DreamGen represents unconstrained latent dynamics, VideoJAM enforces motion priors without reversibility, and Hamiltonian Neural Networks explicitly enforce symplectic structure. The ablation removes the cycle loss, isolating its contribution. Together, this design forms a falsifiable test: if cycle consistency truly improves reversibility, long-horizon errors should decrease compared to DreamGen and the ablation, while possibly approaching HNN performance on conservative dynamics.

### Expected outcome and causal chain

**vs. DreamGen** — On a video of a pendulum swinging for many periods, DreamGen's forward dynamics accumulate phase drift because its latent transitions lack invertibility, causing growing MSE. Our method's cycle-consistency constraint forces f and g to be approximate inverses, preserving phase relationships over long rollouts. We expect a noticeable gap (e.g., 30-50% lower MSE after 50 frames) on such oscillatory motions, with parity on short-term predictions (first 10 frames).

**vs. VideoJAM** — On a scene where an object briefly occludes then reappears, VideoJAM's motion-prior may incorrectly smooth the trajectory through the occlusion because it relies on appearance-motion joint embeddings. Our method, by enforcing latent reversibility, can recover the exact state after occlusion because the forward-backward consistency ensures the dynamics are deterministic. We expect our method to have lower error around occlusion events (e.g., 20-40% lower MSE during reappearance).

**vs. Hamiltonian Neural Network** — On a simulation with friction (non-conservative force), HNN's explicit symplectic constraint will incorrectly conserve energy, leading to unrealistic trajectories (e.g., pendulum never slows). Our method's soft cycle constraint allows approximate reversibility without strict energy conservation, so it can learn dissipative dynamics. We expect our method to outperform HNN on such sequences (e.g., 50% lower MSE over the whole rollout).

### What would falsify this idea

If our method's long-horizon MSE is not significantly lower than DreamGen's (especially on conservative dynamics), or if the ablation (without cycle loss) performs equally well, then the cycle-consistency constraint does not improve reversibility, falsifying the central claim.

## References

1. DreamGen: Unlocking Generalization in Robot Learning through Video World Models
2. VideoJAM: Joint Appearance-Motion Representations for Enhanced Motion Generation in Video Models
3. Still-Moving: Customized Video Generation without Customized Video Data
4. MagicEdit: High-Fidelity and Temporally Coherent Video Editing
5. Dream Video: Composing Your Dream Videos with Customized Subject and Motion
6. IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models
7. VideoBooth: Diffusion-based Video Generation with Image Prompts
8. Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning
