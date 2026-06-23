# Causal World Models from Sparse Observations via Action-Based Intervention Identification

## Motivation

Existing world models for robotics, such as Causal World Modeling and DiT4DiT, assume access to perfect observations (ground-truth states or high-fidelity video) for training and inference. This reliance on ideal observation quality limits real-world applicability where sensors are noisy, observations are sparse, or ground truth is unavailable. We address this structural limitation by leveraging the robot's own actions as known interventions to identify causal structure in latent state dynamics, enabling robust model learning from low-quality observations.

## Key Insight

Robot actions, as known interventions on the environment, provide a causal identification signal that is independent of observation quality, allowing latent state dynamics to be learned without requiring perfect observations.

## Method

**Causal Intervention World Model (CIWM)** learns a latent causal dynamics model from sparse or low-quality observations by treating each robot action as a known intervention on a subset of latent variables. Inputs are observation sequences (downsampled to 1 fps with 50% random occlusion patches of size 32x32) and action sequences; outputs are latent state beliefs and a learned causal graph over latent factors.

**Assumption (explicit):** Each robot action dimension directly and exclusively intervenes on a known, disjoint subset of latent variables, pre-specified based on the robot's kinematic structure (e.g., each joint angle is affected by exactly one action dimension). To verify this mapping, we collect a calibration set of 100 transitions where we intervene with a single action dimension while keeping others fixed, and compute conditional independence test (partial correlation, p<0.05) between the action and each latent variable. If the p-value exceeds 0.05, we reassign the mapping by selecting the latent with highest correlation.

**How it works:**
```python
# Hyperparameters:
beta = 0.1  # KL weight
margin = 1.0  # contrastive margin
lambda_acyc = 1e-3  # acyclicity penalty
latent_dim = 16  # dimension of latent state z
hidden_dim = 256  # MLP hidden size
tau = 0.07  # temperature for contrastive loss (added specification)

# Phase 1: Latent state estimation
# Use a recurrent variational autoencoder (V-RNN) with Gaussian latent states
for t in range(1, T+1):
    # Encode observation o_t into latent distribution parameters
    mu_t, sigma_t = encoder(o_t, recurrent_state)  # encoder: 2-layer MLP, hidden=256, GeLU
    # Sample z_t from q(z_t | o_t, previous recurrent state)
    z_t = reparameterize(mu_t, sigma_t)
    # Dynamics prediction: p(z_t | z_{t-1}, a_{t-1}) = f(z_{t-1}, a_{t-1})
    predicted_mu_t, predicted_sigma_t = dynamics_net(z_{t-1}, a_{t-1})  # dynamics_net: 2-layer MLP, hidden=256, GeLU
    # ELBO loss: reconstruction + KL
    recon_loss = -log p(o_t | z_t)  # observation decoder: transposed conv, same as DreamerV2 decoder
    kl_loss = KL(q(z_t) || p(z_t | z_{t-1}, a_{t-1}))
    loss += recon_loss + beta * kl_loss

# Phase 2: Causal structure learning
# Assume fixed action-to-variable mapping M (from calibration)
# Learn a DAG G over latent variables (size = d) where edges represent causal influence among latents
# Parameterize dynamics as SEM: z_{t+1} = f_G(z_t, a_t) where f_G factorizes according to G
# G is parameterized as a weighted adjacency matrix, initialized with small random uniform values (0.1*U(-0.1,0.1))
# Use NOTEARS-style acyclicity constraint: h(G) = trace(exp(G)) - d = 0, with rho=1e-8 (added specification)

# Interventional contrastive loss (InfoNCE with temperature tau):
for each transition (z_t, a_t, z_{t+1}):
    # Under true action:
    pred_true = f_G(z_t, a_t)
    # Under shuffled action (sample from different timestep within same trajectory to preserve marginal distribution):
    a_shuffled = random_sample_from_trajectory(trajectory, exclude_idx=t)
    pred_shuffled = f_G(z_t, a_shuffled)
    # Contrastive loss: InfoNCE variant
    contrast_loss = -log( exp(-||z_{t+1} - pred_true||^2 / (2*tau^2)) / (exp(-||z_{t+1} - pred_true||^2 / (2*tau^2)) + exp(-||z_{t+1} - pred_shuffled||^2 / (2*tau^2))) )
# Optimize G and dynamics jointly with ELBO + contrast_loss + lambda_acyc * h(G) using AdamW (lr=1e-3, weight_decay=1e-5)
```

**Why this design:** We chose a variational recurrent encoder over a deterministic one because uncertainty from sparse observations requires capturing multiple possible latent states; we accept the cost of slower inference and potential posterior collapse, but mitigate with KL annealing (beta linearly increased from 0 to 0.1 over 10k steps). We chose to learn a causal graph via NOTEARS-style gradient-based acyclicity constraint rather than independent mechanism learning because it allows end-to-end optimization with the dynamics model, accepting the risk of non-convexity and sensitivity to the acyclicity penalty (we reinitialize G if h(G) does not decrease after 500 iterations). We chose an interventional contrastive loss (InfoNCE) over explicit counterfactual generation because generating counterfactual observations would require high-quality video or ground-truth, which we precisely aim to avoid; accepting that the contrastive signal may be weak if actions are highly correlated across timesteps (mitigated by shuffling within trajectory and using a temperature to control sharpness). The action-to-variable mapping assumption is verified via calibration; violation may lead to misspecified causal graph, but the contrastive loss provides robustness by penalizing incorrect assignments.

**Why it measures what we claim:** The contrastive loss (InfoNCE with shuffling) measures the causal necessity of the action for predicting the next latent state, specifically the quantity E[log p(z_{t+1}|z_t, a_t) - log p(z_{t+1}|z_t, a_shuffled)]. This operationalizes the concept of intervention identification because, under the assumption that the true action is an intervention on a subset of variables while shuffled actions are independent interventions (marginal distribution preserved), a correctly learned causal graph should assign higher likelihood to the true action's effect. The assumption that shuffled actions are independent of the current state fails when the trajectory has non-stationary dynamics (e.g., time-varying effects); in that case, the contrast may be attenuated, but the ELBO on observations still provides a global constraint. The acyclicity constraint h(G) measures the absence of cycles in the learned graph; it operationalizes the causal Markov assumption that latent variables follow a DAG; this fails when the true system has feedback loops (e.g., state re-entering through actions), in which case h(G) forces an approximate DAG that may miss recurrent dynamics, but we mitigate by including action nodes explicitly (actions are non-recurrent) and using a small penalty weight. The ELBO reconstruction loss measures observation fidelity, which is necessary but not sufficient for causal correctness; combined with CL, ensures the latent space is consistent with both sensory data and intervention effects.

## Contribution

(1) A causal world model framework (CIWM) that learns latent state dynamics from sparse or low-quality observations by treating robot actions as known interventions, enabling data-efficient training without perfect observations. (2) A novel interventional contrastive learning objective that identifies causal structure in latent dynamics without requiring ground-truth states or high-fidelity video. (3) An action-to-variable mapping assumption that bridges the gap between actions and latent factors, making causal identifiability feasible in robotic settings.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | MetaWorld sparse obs (downsampled 1 fps, 50% occlusions) | Sparse obs challenge causal inference |
| Primary metric | Task success rate (%) | Direct measure of control quality |
| Baseline 1 | DreamerV2 (standard world model, 4 latent dim) | No causal graph, pure dynamics model |
| Baseline 2 | Oracle causal graph (ground-truth kinematic graph) | Upper bound with perfect structure |
| Ablation | CIWM w/o contrastive loss (CL weight=0) | Isolates effect of intervention contrast |

### Why this setup validates the claim

This experimental design tests the central claim that learning a causal intervention world model improves data efficiency and robustness under sparse observations. The dataset, MetaWorld with sparse observations (downsampled to 1 fps from 10 fps and 50% random occlusion patches of size 32x32), directly challenges the model's ability to infer latent dynamics and causal structure from limited sensory data. Task success rate is chosen as the primary metric because it reflects the ultimate goal of reliable robot control, aggregating the effects of both latent state estimation and causal dynamics prediction. Comparing against a standard world model (DreamerV2) isolates the benefit of explicit causal graph learning, while the oracle causal graph baseline (using ground-truth kinematic structure) provides an upper bound to gauge how close CIWM approaches perfect structure. The ablation of removing the contrastive loss tests the necessity of interventional contrast for identifying causal dependencies. Together, these components form a falsifiable test: if CIWM outperforms the standard model and approaches the oracle on tasks requiring causal reasoning, the claim is supported; if not, the mechanism fails.

### Expected outcome and causal chain

**vs. DreamerV2** — On a task like "pick-and-place with occluded intermediate state", the standard world model (DreamerV2) lacks a causal graph and treats all latent dimensions as fully coupled, causing it to conflate action effects with spurious correlations (e.g., believing that moving the arm also changes object color). This leads to poor long-horizon predictions and low task success (~30%). Our CIWM instead learns a disentangled causal graph where each action intervention isolates specific latent factors (e.g., joint angles), enabling accurate dynamics even with gaps in observations. We expect a noticeable gap of ~40% on such causal-chain tasks, but parity on simple single-step tasks.

**vs. Oracle causal graph** — On a task with slight model misspecification (e.g., unmodeled friction), the oracle causal graph assumes perfect structure but cannot adapt; it may overfit to noise in sparse observations, yielding ~70% success. Our CIWM, by learning the graph jointly with dynamics via NOTEARS and contrastive loss, can discover simplified but robust causal dependencies, achieving ~65% success. We expect a small gap (≤10%), confirming that CIWM approximates the oracle well.

**vs. Ablation (w/o contrastive loss)** — On a task where actions are highly interdependent (e.g., sequential joint movements), CIWM without contrastive loss may learn a graph that ignores action effects on certain latents (e.g., treating one joint as independent), because the ELBO alone does not penalize misattribution. This causes failure in precise control (~40% success). Our full CIWM uses the contrastive loss to enforce that the true action yields higher likelihood than a shuffled action, correctly identifying which latents each action targets. We expect a ~25% gap on such tasks.

### What would falsify this idea
If the full CIWM performs no better than the standard world model across all task types, or if the ablation (w/o contrastive loss) matches the full model, then the central claim that causal intervention learning is beneficial under sparse observations is falsified.

## References

1. Causal World Modeling for Robot Control
2. VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference
3. OpenVLA: An Open-Source Vision-Language-Action Model
4. DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control
5. VideoVLA: Video Generators Can Be Generalizable Robot Manipulators
6. World Simulation with Video Foundation Models for Physical AI
7. mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs
8. CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation
