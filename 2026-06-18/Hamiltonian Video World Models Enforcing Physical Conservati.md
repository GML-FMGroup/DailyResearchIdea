# Hamiltonian Video World Models: Enforcing Physical Conservation Laws in Generative Video Latent Dynamics

## Motivation

DreamGen's video generation backbone (DreamGen, 2024) relies purely on learned temporal transformers that do not enforce physical consistency, leading to action recovery errors from hallucinated dynamics. The structural problem is that all four branches of the DreamGen family assume a small set of real demonstrations can ground a video world model without any physics modeling, causing failures in generalization to unseen physical interactions.

## Key Insight

Symplectic integration of latent dynamics via Hamiltonian Neural ODE guarantees exact conservation of energy and momentum, preventing hallucinated trajectories that violate fundamental physics.

## Method

(A) **What it is**: We propose HamiltonVideo, a video world model that replaces the temporal transformer in DreamGen's video generation backbone with a Hamiltonian Neural ODE (HNN) parameterized by a neural network that predicts the Hamiltonian of the latent state. Inputs are latent codes from a video encoder; outputs are future latent codes that evolve under symplectic dynamics. The encoder is trained with a symplectic regularization to ensure the latent space can be interpreted as canonical position and momentum coordinates (phase space). Specifically, the encoder's final layer is a symplectic linear layer (as in Jin et al., 2020) that enforces the Jacobian of the encoder to be symplectic, enabling the latent state to be split into q and p components. The HNN then models the Hamiltonian dynamics in this learned phase space.

(B) **How it works**: Pseudocode:
```python
# HamiltonVideo: Latent dynamics with Hamiltonian Neural ODE
# Input: initial latent state z0 ∈ R^d (from video encoder), time steps t0..T
# Parameters: neural network H_θ (Hamiltonian), encoder E_φ, decoder D_ψ
# Output: predicted latent states z_t for t=1..T

def hamiltonian_ode_step(z, dt, H_θ):
    # Symplectic integration (leapfrog)
    q, p = split_state(z)  # assume latent space splits into position and momentum
    # Leapfrog step:
    p_half = p - (dt/2) * ∂H/∂q(q, p, H_θ)
    q_next = q + dt * ∂H/∂p(q, p_half, H_θ)
    p_next = p_half - (dt/2) * ∂H/∂q(q_next, p_half, H_θ)
    z_next = combine(q_next, p_next)
    return z_next

def forward(z0, times, H_θ):
    z = z0
    traj = [z0]
    for t in range(1, len(times)):
        dt = times[t] - times[t-1]
        z = hamiltonian_ode_step(z, dt, H_θ)
        traj.append(z)
    return traj

# Training: Video frames x_t are encoded to latents z_t_gt = E_φ(x_t)
# The HNN predicts z_t from z_0 via Hamiltonian dynamics
# Loss: MSE between predicted latents and ground-truth latents + regularization on Hamiltonian
# Optionally, decoder D_ψ reconstructs frames for pixel-level loss
```
Hyperparameters: latent dimension d=64, symplectic integration step size dt=0.1, Hamiltonian network H_θ (3-layer MLP with 256 hidden units, tanh activation). For the symplectic autoencoder, we use a reconstruction loss (MSE) and a symplectic regularization loss that penalizes deviation from symplectic Jacobian (L_symp = ||J^T Ω J - Ω||_F^2, where J is the encoder Jacobian). The encoder backbone is a ResNet-18 followed by a symplectic linear layer (weights constrained to be symplectic via QR decomposition). Decoder D_ψ is a symmetric architecture (ResNet-based). Training is performed with Adam optimizer (lr=1e-4), batch size 32, for 100 epochs.

(C) **Why this design**: We chose HNN over standard neural ODE because HNN enforces conservation of energy by construction via the symplectic gradient structure, eliminating the need to learn conservation as a soft constraint. We chose leapfrog integration over Runge-Kutta because leapfrog is symplectic and reversible, ensuring long-term stability in latent dynamics. We split the latent state into position and momentum pairs; this is a design assumption that the latent space has a natural Hamiltonian structure. The trade-off is that we cannot model non-conservative forces (e.g., friction) without extension, but for the target manipulation tasks where energy and momentum are approximately conserved, this is acceptable. We also accept that the latent space must be 2n-dimensional and interpretable as phase space, which may require careful autoencoder training. Compared to prior work like DreamGen's temporal transformer, our approach does not rely on learned attention to capture dynamics but instead imposes a hard physical prior.

(D) **Why it measures what we claim**: The Hamiltonian H_θ is learned to approximate the true energy of the physical system; its gradient ∂H/∂q and ∂H/∂p determine the update. The symplectic integration ensures that the predicted latent trajectory conserves the learned Hamiltonian exactly (up to numerical precision). This means that the dynamics are energy-conserving by construction, which operationalizes the concept of 'physical consistency' because conservation laws are a necessary condition for physically plausible dynamics. The assumption that the latent space can be partitioned into canonical position and momentum coordinates is strong; when this assumption fails (e.g., the latent encoding mixes position and momentum nonlinearly), the Hamiltonian structure may not capture true physics, and the method reduces to a constrained neural ODE that enforces a spurious conservation. In that case, the metric reflects conservation of a learned quantity that may not correspond to true energy. To validate the equivalence, we compute the Pearson correlation between the learned Hamiltonian H_θ(z) and ground-truth total energy from the simulator on a held-out set of 512 trajectories. If correlation > 0.9, we interpret that the learned Hamiltonian approximates true energy. Additionally, we monitor the failure mode by comparing energy conservation error versus ground-truth energy deviation: if conservation error is low but ground-truth energy deviation is high, the prior has failed to capture true physics.

## Contribution

(1) We introduce HamiltonVideo, a video world model that enforces Hamiltonian dynamics in the latent space, guaranteeing conservation of energy and momentum without explicit physics simulation. (2) We show that replacing the temporal transformer with a Hamiltonian Neural ODE in DreamGen's pipeline improves action recovery accuracy on tasks involving ballistic motion and collisions, as verified by lower action reconstruction error and higher policy success on DreamGen Bench. (3) We provide a benchmark methodology for evaluating physical consistency of video world models using conserved quantities.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Physics simulation videos (pendulum, bouncing ball, spring-mass, rigid body collisions) | Ground truth physics available for validation |
| Primary metric | Energy conservation error (absolute difference between initial and final learned Hamiltonian) | Directly measures physical consistency |
| Baseline 1 | DreamGen temporal transformer | State-of-the-art without physics prior |
| Baseline 2 | Standard Neural ODE | Shows benefit of Hamiltonian constraint |
| Baseline 3 | Neural ODE with explicit energy regularization (L2 penalty on energy drift) | Isolates novelty of symplectic structure vs. soft constraint |
| Ablation | HamiltonVideo with RK4 solver | Isolates effect of symplectic integration |

### Why this setup validates the claim

The dataset provides ground-truth conserved energy from physics simulators, enabling direct measurement of how well each model preserves energy over long horizons (100 time steps). The primary metric (energy conservation error) is the most direct test of the central claim that Hamiltonian dynamics enforce physical consistency. DreamGen transformer represents a strong non-physics baseline; if our method performs noticeably better on energy conservation, it confirms the prior helps. Standard Neural ODE tests whether the Hamiltonian structure (symplectic gradients) matters beyond just learning an ODE; if ours outperforms it, the explicit symplectic geometry is responsible. Adding a third baseline (Neural ODE with energy regularization) isolates the novelty of the symplectic prior compared to a soft energy constraint. The ablation (Runge-Kutta instead of leapfrog) isolates the contribution of symplectic integration: if leapfrog improves stability and conservation, the ablation will show degraded performance. Together, these components form a falsifiable test: if the Hamiltonian prior is beneficial, we should see a clear advantage on energy conservation, especially in long sequences where non-physical drifts accumulate. We also compute the correlation between learned Hamiltonian and ground-truth energy on a calibration set of 512 trajectories to verify the equivalence.

### Expected outcome and causal chain

**vs. DreamGen temporal transformer** — On a video of a pendulum swinging, DreamGen may produce visually plausible frames but gradually drift in period or amplitude because its attention-based dynamics lack energy conservation. Our method, using symplectic integration, exactly conserves the learned Hamiltonian, so the pendulum's amplitude and period stay constant over many cycles. We expect HamiltonVideo to have energy conservation error near zero (e.g., <1% deviation) while DreamGen's error grows with trajectory length (e.g., >10% after 100 steps).

**vs. Standard Neural ODE** — On a bouncing ball dataset with elastic collisions, a standard Neural ODE learns to approximate dynamics but may slowly leak energy due to non-symplectic integration, causing the ball's bounce height to decay. Our Hamiltonian network with leapfrog integrator preserves total energy exactly, so the bounce height remains unchanged. We expect HamiltonVideo's mean energy deviation to be orders of magnitude lower (e.g., 0.01 vs 0.5) and to remain constant over time, whereas the Neural ODE's deviation increases with step count.

**vs. Neural ODE with Energy Regularization** — On a spring-mass system, the regularized Neural ODE may achieve moderate energy conservation but will still experience drift because regularization is a soft penalty, not a hard constraint. Our method's symplectic integration enforces conservation exactly, yielding nearly zero error over long horizons. We expect HamiltonVideo's energy conservation error to be at least an order of magnitude smaller than the regularized Neural ODE, especially for long sequences (e.g., 200 steps).

### What would falsify this idea

If energy conservation error for HamiltonVideo is not significantly lower than DreamGen, Neural ODE, and regularized Neural ODE on the physics dataset, or if the gap is uniform across all sequences (rather than larger on long trajectories), then the symplectic prior does not effectively enforce physical consistency and the central claim is unsupported.

## References

1. DreamGen: Unlocking Generalization in Robot Learning through Video World Models
2. STIV: Scalable Text and Image Conditioned Video Generation
3. How Far is Video Generation from World Model: A Physical Law Perspective
4. Benchmarks for Physical Reasoning AI
5. SEINE: Short-to-Long Video Diffusion Model for Generative Transition and Prediction
