# Inference-Time Optimization for Embodiment-Aware Foresight in Vision-Language-Action Models

## Motivation

InternVLA-A1.5 relies on a frozen video generation model to supervise latent foresight tokens, but this supervision only enforces visual plausibility, not robot-specific dynamics or affordances. As a result, the foresight tokens predict futures that may be physically unrealizable for the robot's morphology, limiting their utility for action planning. This limitation stems from the frozen video model being pretrained on general video data, which does not cover the specific dynamics and affordances of a particular robot embodiment.

## Key Insight

The latent foresight token can be adapted to robot-specific physics without fine-tuning the large model by simultaneously enforcing consistency with a frozen video prior and a lightweight learned dynamics model through inference-time gradient optimization, because the gradient from the differentiable constraints forces the tokens onto the intersection of two manifolds: visual plausibility and physical realizability.

## Method

**(A) What it is:** FAPCO (Foresight Adaptation with Physics-Consistent Optimization) is an inference-time procedure that takes initial latent foresight tokens z_0 from a VLM policy and optimizes them to be consistent with both a frozen video generation model F (from InternVLA-A1.5) and a lightweight learned dynamics model D, without modifying the pretrained weights of either model. D is a 3-layer MLP with hidden size 256 and ReLU activations, trained on ≤100 robot episodes using a mean-squared error loss over next-state prediction. We explicitly assume that D provides accurate and differentiable gradients for the optimization; to verify this, we measure D's prediction error on a held-out validation set (10% of training data) and abort optimization if the mean error exceeds a threshold of 5% of state range.

**(B) How it works:**
```python
# Input: initial latent foresight tokens z_0 (64-dim), observation o_t (includes robot state s_t, camera image),
#        frozen video generation model F (provides log-likelihood evaluation via a 3D UNet decoder with 4 layers),
#        learned dynamics model D (3-layer MLP, hidden=256, ReLU, output dim = state dim),
#        differentiable physics simulator P (MuJoCo differentiable, optional; if unavailable, use finite differences with step size 0.01),
#        action decoder A (2-layer MLP, hidden=128, output H=10 actions of dim 7 each),
#        hyperparameters: λ_phys=0.1, λ_smooth=0.01, learning rate η=0.001 (Adam optimizer), steps K=50.

def optimize_foresight(z_0, o_t, F, D, A, P, λ_phys, λ_smooth, η, K):
    z = z_0.clone().detach().requires_grad_(True)
    for step in range(K):
        # 1. Video prior loss: negative log-likelihood of z under F
        #    F's log-likelihood is computed by decoding z into video frames and comparing with a Gaussian prior.
        L_prior = -F.log_likelihood(z)  # assumes well-calibrated likelihood on robot motions
        
        # 2. Decode action sequence
        actions = A(z)  # shape (H=10, action_dim=7)
        
        # 3. Physics consistency loss using learned dynamics model or differentiable simulator
        #    If P is available, simulate forward using P; otherwise use D.
        state = o_t.state  # 9-dim robot joint angles
        if P is not None:
            # Differentiable physics simulation
            states_pred = []
            for t in range(H):
                state = P.step(state, actions[t])  # differentiable step
                states_pred.append(state)
        else:
            # Use learned dynamics D
            states_pred = [state]
            for t in range(H):
                next_state = D(states_pred[-1], actions[t])
                states_pred.append(next_state)
        # Physics penalty: collision and joint limit
        L_phys = sum(collision_penalty(s) + joint_limit_penalty(s) for s in states_pred)
        # Smoothness loss
        L_smooth = sum((actions[t+1] - actions[t])**2 for t in range(H-1))
        
        # Total loss
        L = L_prior + λ_phys * L_phys + λ_smooth * L_smooth
        
        # Gradient step on z
        z = z - η * grad(L, z)
    return z
```

**(C) Why this design:** We chose inference-time optimization over fine-tuning the frozen model or fine-tuning the VLM (which would risk catastrophic forgetting of general knowledge) because the optimization uses a small dynamics model that is cheap to train on ≤100 robot episodes, while preserving the expressive power of the large pretrained video model. We opted for a gradient-based optimizer (Adam) rather than evolutionary or random-shooting methods (which are sample-inefficient) because the differentiable physics simulator provides smooth gradients, but this requires that P and D are differentiable, which we accept as a constraint that may not hold for all simulators. We used a simple negative log-likelihood for the video prior instead of a more complex contrastive loss because the frozen model already outputs a likelihood-based score, though this assumes the model is well-calibrated, which may not hold for out-of-distribution robot motions. The trade-off is computational cost: 50 gradient steps at inference adds ~200ms on an A100 GPU, but we prioritize accuracy over latency.

**(D) Why it measures what we claim:** L_prior measures visual plausibility because the frozen model F was trained on a large corpus of natural videos; this assumption fails when the robot's interaction dynamics are drastically different from natural video motion (e.g., precise manipulation), in which case L_prior may favor visually plausible but physically impossible futures. L_phys measures robot-specific physical realizability because the dynamics model D is trained on actual robot episodes; this assumption fails when D's training data is insufficient to cover all scenarios, in which case L_phys may incorrectly penalize physically valid trajectories. The action decoder A measures the mapping from latent tokens to executable actions; this measure assumes that the original VLM's action decoder is sufficient to produce plausible actions from the latent supervision, but if the initial z_0 is far from the manifold of feasible actions, the optimization may find a poor local minimum. The smoothness term L_smooth measures temporal coherence of actions; this assumes that robot motions are continuous, which holds for joint velocities but may penalize necessary sudden movements. The collision penalty in L_phys measures physical safety; this relies on the accuracy of the differentiable simulator's collision detection, which fails when the simulator's geometry does not match the real environment. We validate the calibration of F's log-likelihood by computing the expected calibration error (ECE) on a holdout set of 512 robot trajectories; if ECE > 0.1, we flag potential miscalibration. We also measure D's mean prediction error and abort optimization if it exceeds 5% of state range.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | SimplerEnv Google Robot | Standardized robot manipulation benchmark. |
| Primary metric | Task success rate | Directly measures action executability. |
| Additional metrics | Expected calibration error (ECE) of F's log-likelihood on robot data; Dynamics model D prediction error (MSE) on test episodes | Validate assumptions about prior and dynamics. |
| Baseline | Vanilla VLM (no optimization) | Baseline without inference-time refinement. |
| Baseline | Random shooting optimization (500 samples, top-1 action) | Non-differentiable approach, sample-inefficient. |
| Baseline | Test-time fine-tuning of VLM on one-step video reconstruction loss (TTT-VLM) | Gradient-based adaptation without physics loss. |
| Ablation | FAPCO without physics loss (λ_phys=0) | Isolates contribution of physics consistency. |
| Ablation | FAPCO with non-differentiable simulator (finite differences, step size 0.01) | Tests sensitivity to differentiability. |

### Why this setup validates the claim
This setup tests the central claim that inference-time optimization with a cheap physics model improves physical plausibility without sacrificing video prior guidance. Vanilla VLM tests the gain from any optimization; random shooting tests whether gradient-based optimization is beneficial; TTT-VLM tests whether a fully adaptive method (fine-tuning) would outperform our lightweight approach; the ablations test the necessity of the physics loss and the differentiability assumption. Task success rate directly captures whether optimized actions are executable and achieve goals, which is the ultimate test. Additional metrics (ECE, dynamics error) provide diagnostic verification of the assumptions underlying L_prior and L_phys. The chosen dataset provides diverse manipulation tasks with realistic physics, enabling generalization assessment.

### Expected outcome and causal chain

**vs. Vanilla VLM** — On a task requiring precise contact (e.g., pushing a cup), the vanilla VLM may produce physically implausible actions (e.g., floating or penetrating the cup) because it lacks physics awareness. Our method instead enforces physical constraints through the dynamics model, so we expect a noticeable success rate gain (10–20% improvement) on contact-rich tasks, with parity on simple reach tasks.

**vs. Random shooting optimization** — On a task with long-horizon dynamics (e.g., stacking blocks), random shooting requires many samples to find plausible actions due to the high-dimensional action space, leading to high variance and poor success. Our gradient-based optimization leverages the differentiable physics model to efficiently update latent tokens, so we expect higher and more stable success rates (15–25% gain) with lower variance across seeds.

**vs. Test-time fine-tuning of VLM (TTT-VLM)** — On a task requiring fast adaptation (e.g., novel object pushing), TTT-VLM may overfit to the single observation and forget prior knowledge, leading to degenerate actions. Our method avoids fine-tuning the large model, preserving prior knowledge while adapting only the latent tokens, so we expect comparable or better success (5–10% gain) with lower computational cost.

**vs. FAPCO without physics loss** — On a task requiring dynamic stability (e.g., swinging a pendulum), the video prior alone may favor visually smooth but unstable motions, causing failure. Adding the physics loss penalizes unstable states, so our full method should achieve higher success (20–30% gain) specifically on tasks where physics accuracy matters.

**vs. FAPCO with non-differentiable simulator** — Using finite differences may be noisier and require more steps (K=100); we expect slightly lower success (3–5% drop) but similar trends, validating the approach's robustness.

### What would falsify this idea
If the full FAPCO method shows no success rate advantage over Vanilla VLM on contact-rich tasks, or if the gain is uniform across all tasks (indicating no physics-specific benefit), the central claim that physics optimization is effective would be falsified. Additionally, if the expected calibration error of F is large (>0.2) or the dynamics model error exceeds 10% on test data, the assumptions underlying the losses are violated, weakening the rationale.

## References

1. InternVLA-A1.5: Unifying Understanding, Latent Foresight, and Action for Compositional Generalization
2. InternVLA-M1: A Spatially Guided Vision-Language-Action Framework for Generalist Robot Policy
3. UniUGP: Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving
4. Motus: A Unified Latent Action World Model
5. X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model
6. DrivingGPT: Unifying Driving World Modeling and Planning with Multi-Modal Autoregressive Transformers
7. DrivingWorld: Constructing World Model for Autonomous Driving via Video GPT
8. Flow Matching Guide and Code
