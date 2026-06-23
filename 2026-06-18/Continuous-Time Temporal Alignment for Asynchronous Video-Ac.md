# Continuous-Time Temporal Alignment for Asynchronous Video-Action Learning in Dual Diffusion Transformers

## Motivation

DiT4DiT achieves state-of-the-art video-action modeling but requires perfectly synchronized video-action pairs, which are rare in real-world robot data collection. This assumption limits the use of larger, naturally asynchronous datasets. For instance, in VideoVLA and π0, data collection often involves varying frame rates and delayed action recordings, causing temporal misalignment that degrades performance.

## Key Insight

Modeling the latent video and action spaces as continuous-time dynamical systems via neural ODEs enables exact temporal interpolation, making the learned representations invariant to arbitrary timing offsets and missing action observations.

## Method

(A) **What it is**: Continuous-Time Temporal Alignment Layer (CTAL) is a learnable module inserted between the video DiT and action DiT in the dual DiT architecture. It takes as input the denoising features from the video DiT at arbitrary timestamps and outputs temporally aligned features for the action DiT at the exact action timestamps. (B) **How it works**: Pseudocode:
```python
class CTAL(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.ode_func = NeuralODE(hidden_dim=256)  # continuous-time dynamics
        self.readout = nn.Linear(latent_dim, latent_dim)

    def forward(self, video_features, video_times, action_times):
        # video_features: dict mapping each video timestamp to its denoising feature
        # video_times: list of timestamps for video frames
        # action_times: list of timestamps for action predictions (may be misaligned)
        
        # Step 1: Build continuous trajectory via neural ODE
        # Sort video features by time
        sorted_times, sorted_features = zip(*sorted(zip(video_times, video_features)))
        # Integrate ODE from earliest to latest video time to obtain latent state at each video time
        latent_traj = self.ode_func.odeint(sorted_features[0], sorted_times)[0]  # shape (T_v, D)
        
        # Step 2: Interpolate to action timestamps using ODE solver
        action_latents = []
        for t_a in action_times:
            # find closest video time interval
            idx = bisect_left(sorted_times, t_a)
            if idx == 0:
                # extrapolate backward using ODE
                latent = self.ode_func.odeint(latent_traj[0], [t_a])[0]
            elif idx == len(sorted_times):
                # extrapolate forward
                latent = self.ode_func.odeint(latent_traj[-1], [t_a])[0]
            else:
                t_prev = sorted_times[idx-1]
                t_next = sorted_times[idx]
                # integrate from t_prev to t_a (or from t_a to t_next and combine)
                latent_prev = latent_traj[idx-1]
                latent_mid = self.ode_func.odeint(latent_prev, [t_a])[0]
                action_latents.append(latent_mid)
        
        # Step 3: Readout to same dimension as action DiT input
        aligned_features = torch.stack([self.readout(l) for l in action_latents])
        return aligned_features
```
Hyperparameters: ODE hidden_dim=256, solver='dopri5', rtol=1e-3, atol=1e-6. (C) **Why this design**: We chose a neural ODE over a simple linear interpolation (e.g., bilinear) because linear interpolation assumes constant velocity dynamics, which fails for complex robot motions with acceleration changes; the ODE learns nonlinear dynamics, providing more accurate latent interpolation at the cost of higher computational overhead. We chose to integrate from the nearest past video frame rather than averaging multiple frames to avoid blurring temporal details; this design assumes that dynamics are locally deterministic, which may not hold for stochastic transitions but is acceptable for short time gaps. We used a separate readout layer to map ODE latents to the action DiT's input space, decoupling representation learning from alignment; this adds parameters but prevents the ODE from having to match the action DiT's specific feature requirements, enabling modular training. (D) **Why it measures what we claim**: The neural ODE's continuous-time latent trajectory measure 'temporal consistency' because the ODE enforces that latent states evolve smoothly according to learned dynamics; this equivalence assumes that the motion is well-approximated by a first-order ODE, which fails when sudden discontinuities occur (e.g., object grasp release), in which case the ODE will produce unrealistic intermediate states. The 'aligned_features' obtained by ODE integration at action timestamps measure 'temporal alignment' because they represent the video latent at the exact moment of action, assuming the ODE generalizes to arbitrary query times; this assumption fails for times far outside the observed video span, where extrapolation may drift. The readout layer's linear mapping measures 'feature compatibility' because it projects the ODE latents to the input space of the action DiT, assuming that a linear transformation suffices; this fails if the action DiT expects features with a specific nonlinear structure, in which case the readout can be replaced with an MLP.

## Contribution

(1) A novel continuous-time temporal alignment layer (CTAL) that relaxes the synchronization requirement in dual diffusion transformers for video-action modeling. (2) Empirical demonstration that CTAL enables effective training on asynchronous video-action pairs, improving data efficiency by leveraging naturally misaligned datasets. (3) A synthetic benchmark for evaluating temporal alignment methods in robotic manipulation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | CALVIN | Continuous control, varied tasks |
| Primary metric | Task success rate | Directly measures task performance |
| Baseline 1 | DiT4DiT | No temporal alignment layer |
| Baseline 2 | Cosmos-Predict1 | Direct video generation, no action alignment |
| Ablation-of-ours | CTAL w/ Linear Interpolation | Constant velocity assumption |

### Why this setup validates the claim
This setup isolates the effect of the Continuous-Time Temporal Alignment Layer (CTAL). Comparing against DiT4DiT (same architecture without alignment) tests whether CTAL's temporal alignment improves action prediction when video and action timestamps mismatch. Comparing against Cosmos-Predict1 (a video-only model) tests whether action-aligned video features are necessary for precise control. The ablation (linear interpolation) tests whether the neural ODE's nonlinear dynamics are superior to a constant-velocity baseline. The primary metric, task success rate, is the ultimate measure of control quality; if CTAL provides better temporal alignment, it should directly translate to higher success rates, especially on tasks with dynamic motions. A falsification pattern would be if CTAL does not outperform the ablation on tasks with non-constant velocity, or if gains are uniform across all task subsets.

### Expected outcome and causal chain

**vs. DiT4DiT** — On a case where action timestamps are offset from video timestamps (e.g., robot moving fast), DiT4DiT misaligns video features, leading to jerky or delayed actions. Our method uses ODE to interpolate to exact action times, producing smooth latent states. We expect a noticeable gap (~15% higher success rate) on dynamic tasks (e.g., reaching), but parity on static tasks.

**vs. Cosmos-Predict1** — On a case requiring precise timing (e.g., grasping a falling object), Cosmos-Predict1 generates video without aligning to action times, causing timing errors. Our method aligns video features to the exact action timestamps via ODE, so we expect ~20% higher success on timing-critical tasks.

**vs. CTAL w/ Linear Interpolation** — On a case with acceleration changes (e.g., smooth deceleration during grasp), linear interpolation assumes constant velocity, causing overshoot. Our ODE learns nonlinear dynamics, providing more accurate latent states. We expect ~10% higher success on tasks with non-constant velocity (e.g., pick-and-place).

### What would falsify this idea
If CTAL does not outperform linear interpolation on tasks with non-constant velocity (i.e., the ODE's nonlinear modeling is unnecessary), or if improvement is uniform across all task subsets rather than concentrated on dynamic or timing-critical tasks, then the central claim—that neural ODE-based temporal alignment is critical—is falsified.

## References

1. DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control
2. World Simulation with Video Foundation Models for Physical AI
3. VideoVLA: Video Generators Can Be Generalizable Robot Manipulators
4. π*0.6: a VLA That Learns From Experience
5. Diffusion Policy Policy Optimization
6. π0: A Vision-Language-Action Flow Model for General Robot Control
7. Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations
8. Open-Sora: Democratizing Efficient Video Production for All
