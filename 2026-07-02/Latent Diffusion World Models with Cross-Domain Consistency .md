# Latent Diffusion World Models with Cross-Domain Consistency for 3D Control

## Motivation

Existing latent diffusion world models, such as VALDI (Valdi: Value Diffusion World Models), have demonstrated effective control in 2D benchmarks but fail to generalize to complex 3D environments due to visual domain gaps. The root cause is that these models are trained on fixed visual appearances and do not learn dynamics that are invariant to rendering variations. This limitation is a structural property across multiple branches (e.g., DIAMOND also limited to 2D), and addressing it requires explicit domain generalization.

## Key Insight

By leveraging a simulator's ability to render the same underlying physical state with multiple visual appearances, we can enforce an invariance constraint on the latent dynamics that forces the model to abstract away visual details, enabling zero-shot transfer to unseen 3D environments.

## Method

We propose Cross-Domain Latent Diffusion (CDLD), a world model that learns appearance-invariant dynamics.

**Assumption:** The visual appearance variations (texture, lighting) do not alter the underlying physical state transition dynamics. To verify this assumption in practice, we use a calibration set of 1000 state-action pairs sampled from the simulator; for each pair, we render the next state under two different appearances and measure the change in physical state (e.g., position, velocity). If the maximum change across the calibration set exceeds a threshold of 0.01, we flag a violation and adjust the rendering parameters.

**(A) What it is**: CDLD is a latent diffusion dynamics model trained with an additional cross-domain consistency loss that aligns latent state predictions from different visual renderings of the same physical state. Input: past observations and actions. Output: future latent states.

**(B) How it works (pseudocode):**

```python
# Architecture: Encoder E: 4-layer CNN, filters=[32,64,128,256], kernel=3, stride=2, ReLU activation, latent dim=512.
# Diffusion model: conditioned on action embedding (2-layer MLP, hidden=256, ReLU), U-Net with 3 down/up blocks, 64 base channels.
# Training hyperparameters: batch size=64, learning rate=1e-4, Adam optimizer, 100k steps, single NVIDIA A100 GPU (~48 GPU hours).

for each physical state s in simulator:
    renderings = simulate_appearances(s, appearances=[texture1, lighting1, texture2, lighting2])
    z0_list = [E(img) for img in renderings]
    a = policy or random
    z1_pred_list = []
    for z0 in z0_list:
        noise = sample_noise()
        z1_pred = diffusion_denoise(z0, a, noise, step=1)
        z1_pred_list.append(z1_pred)
    next_renderings = simulate_appearances(s', appearances)
    z1_true_list = [E(next_img) for next_img in next_renderings]
    pred_loss = sum([MSE(z1_pred_list[i], z1_true_list[i]) for i in range(len(renderings))])
    consistency_loss = 0
    for i in range(len(renderings)):
        for j in range(i+1, len(renderings)):
            consistency_loss += MSE(z1_pred_list[i], z1_pred_list[j])
    loss = pred_loss + 0.1 * consistency_loss
    update model parameters
```

**(C) Why this design**: We chose a single-step denoising over multi-step sampling to maintain low latency for online MPC, as in VALDI, accepting that predictive multimodality may be reduced (trade-off). We used a pairwise MSE consistency loss over all renderings rather than a mean or contrastive loss because MSE directly enforces numerical similarity in latent space, which is simple and avoids additional hyperparameters; the cost is that it may be sensitive to outliers. We used a shared encoder for all appearances to force it to learn appearance-invariant features; alternatively, separate encoders for each appearance could be used, but that would increase parameters and not force invariance in the encoder. We chose to apply the consistency loss on the predicted next latents rather than on the current latents because we care about dynamics invariance; the encoder already processes raw images, so enforcing invariance on dynamics directly addresses the goal. The trade-off is that this requires multiple forward passes through the diffusion model, increasing training cost.

**(D) Why it measures what we claim**: The consistency loss component, computed as MSE between predicted latents from different appearances of the same physical action, measures the degree of cross-domain invariance in the dynamics because it quantifies the discrepancy in latent state predictions caused solely by visual appearance variation; this relies on the assumption that the simulator's rendering variations are independent of the true physical state transition. This assumption fails when the rendering variations also alter the physical dynamics (e.g., changing friction via texture), in which case the consistency loss might incorrectly penalize legitimate differences. The prediction loss (MSE to true next latents) measures the predictive accuracy of the model; when combined with consistency loss, it ensures that invariance does not come at the cost of accuracy. The latent encoder's shared parameters across all appearances operationalize the concept of cross-domain generalization by forcing a single representation mapping for diverse inputs; this assumes that the encoder can disentangle appearance from dynamics, which may fail if appearances cause systematic encoder errors (e.g., texture leads to different latent features that are artificially pushed together, harming accuracy). We verify the load-bearing assumption using a calibration set: 1000 state-action pairs, render under two appearances, measure maximum change in physical state; if >0.01, flag violation.

## Contribution

(1) A cross-domain consistency loss for latent diffusion world models that enforces invariance to visual appearance changes during dynamics prediction. (2) Demonstration that leveraging simulator-controlled rendering variations enables zero-shot transfer to unseen 3D environments without additional domain adaptation. (3) A training procedure that combines predictive and consistency losses without requiring action labels across appearances.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | MuJoCo Ant-v2 with randomized textures (20) & lighting (5) | Tests appearance-invariant dynamics in 3D |
| Primary metric | Prediction MSE on held-out appearances | Directly measures dynamics invariance |
| Secondary metric | Control reward (MPC horizon=10, pop=500) | Tests downstream task performance |
| Baseline 1 | VALDI (single appearance training) | No cross-domain handling |
| Baseline 2 | CDLD without consistency loss | Tests benefit of our loss |
| Ablation-of-ours | CDLD with separate encoders per appearance | Tests shared encoder necessity |
| Additional ablation | Environment with textures correlated with friction | Tests robustness to assumption violation |

### Why this setup validates the claim
This combination of a dataset with diverse appearances, a metric that measures prediction error on unseen looks, and baselines isolating each design choice forms a falsifiable test of our central claim: that cross-domain consistency loss in latent diffusion dynamics learns appearance-invariant predictions. The MuJoCo Ant-v2 environment allows controlled appearance variation while preserving physics (with verification via calibration set). Our primary metric directly quantifies invariance by comparing predictions across different renderings of the same state-action pair, capturing whether the model treats appearance as nuisance. VALDI baseline shows necessity of cross-domain handling; CDLD without consistency loss tests the specific contribution of our loss; and the separate-encoder ablation tests whether a shared encoder is critical for forcing invariance. The additional ablation with correlated textures tests the boundary of our assumption: when appearance changes alter physics (e.g., ice texture lowers friction), the consistency loss may harm accuracy, and our method's degradation measures robustness. If our method outperforms all baselines on the primary metric and maintains decent control performance, it validates the claim; if not, the claim is weakened.

### Expected outcome and causal chain

**vs. VALDI** — On a case where the robot drives on a track with unseen grass texture (instead of default gray), VALDI's latent predictions drift because its encoder and dynamics were never trained to ignore texture changes. Our method, having seen many textures during training, produces near-identical latents for the same physical state, so prediction error on unseen textures remains low. We expect a clear gap: VALDI's MSE on held-out appearances >50% higher than ours (e.g., VALDI MSE=0.45 vs. ours=0.20). Control reward: VALDI achieves 30% of optimal, ours 70%.

**vs. CDLD without consistency loss** — When the model encounters a new lighting condition (e.g., dusk), the prediction-only model overfits to the specific lightings seen during training and fails to generalize because nothing forces latent invariance. Our consistency loss directly penalizes discrepancies across renderings, so predictions for the same action under different lights align. We expect our method to show <20% increase in MSE from training to held-out appearances, while the ablation may show >40% increase.

**vs. CDLD with separate encoders** — On a case with extreme texture variation (e.g., checkered vs. asphalt road), separate encoders learn distinct latent spaces, making dynamics inconsistent across appearances even if each encoder is accurate. Our shared encoder forces a common latent space, so prediction invariance is higher. We expect our shared-encoder version to achieve 10-15% lower MSE on held-out appearances than the separate-encoder version.

**vs. Additional ablation (correlated textures)** — When texture changes also alter friction (e.g., icy texture reduces friction), the consistency loss penalizes legitimate differences in dynamics. Here, CDLD may degrade relative to the uncorrelated case. We expect MSE on held-out appearances to increase by 25-30% compared to the uncorrelated case, while baselines (which ignore appearance) may not show such degradation but have higher overall MSE. This quantifies the cost of the assumption violation.

### What would falsify this idea
If CDLD shows no consistent advantage over the no-consistency baseline on held-out appearances, or if the gain is uniform across all appearance conditions rather than concentrated on the most visually diverse ones, then the central claim that cross-domain consistency loss learns appearance-invariant dynamics is false. Additionally, if on the correlated-texture ablation CDLD's performance drops below the no-consistency baseline, this would indicate that the assumption violation is too severe for the method to handle.

## References

1. Valdi: Value Diffusion World Models
2. ReSim: Reliable World Simulation for Autonomous Driving
3. TD-MPC2: Scalable, Robust World Models for Continuous Control
4. Diffusion for World Modeling: Visual Details Matter in Atari
5. VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training
6. GEM: A Generalizable Ego-Vision Multimodal World Model for Fine-Grained Ego-Motion, Object Dynamics, and Scene Composition Control
7. Bench2Drive-R: Turning Real World Data into Reactive Closed-Loop Autonomous Driving Benchmark by Generative Model
8. Transformer-based World Models Are Happy With 100k Interactions
