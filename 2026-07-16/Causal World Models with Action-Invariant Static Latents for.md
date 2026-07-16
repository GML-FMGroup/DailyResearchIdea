# Causal World Models with Action-Invariant Static Latents for Domain-Generalizable Dynamics

## Motivation

Existing world models for robotics rely on large-scale pretrained video encoders that assume all dynamics are covered by the pretraining data, failing under open-domain shifts. For instance, GigaWorld-Policy-0.5 uses mixed AC-WM and WAM pretraining but still assumes the pretrained representations transfer without explicit domain adaptation. The root cause is the lack of a structural separation between action-affected and environment-static variables, forcing the model to learn domain-specific correlations that do not generalize. This limitation is a convergent gap across multiple approaches (GigaWorld, mimic-video, etc.) that all assume pretraining data covers necessary dynamics.

## Key Insight

Action effects on the world are causally invariant across domains because the physical effect of a robot action (e.g., pushing) on an object is governed by the same dynamics regardless of visual background; isolating the minimal latent variables causally influenced by actions yields a dynamics model that generalizes across domains without large-scale pretraining.

## Method

### Causal World Model (CWM) – Specified Version

(A) **What it is** — Causal World Model (CWM): a latent dynamics model that explicitly factorizes the observation into static latent variables (z_s) and dynamic latent variables (z_d), and learns a transition model over z_d that is invariant to z_s. Inputs: observation x, action a. Outputs: next observation prediction via decoder. Training enforces causal sufficiency: z_s is not affected by actions, and the transition of z_d is independent of z_s.

(B) **How it works** — The architecture and training procedure are described below using a conceptual pseudocode:

```python
# Encoder: 4-layer CNN, channels=[32,64,128,256], kernel=4, stride=2, padding=1, ReLU; outputs (z_s, z_d) each dimension=64
# Transition: 2-layer MLP, hidden=256, ReLU; input (z_d, a), predicts delta eta
# Decoder: 4-layer transposed CNN mirroring encoder, outputs x'
# Predictor: 2-layer MLP, hidden=128, ReLU; predicts z_s from a
# DomainClassifier: 2-layer MLP, hidden=128, ReLU; predicts domain index from transition features eta

# Causal sufficiency loss: ensure that z_s is not predictable from a
z_s_pred = Predictor(a)
causal_loss = MSE(z_s_pred, z_s)

# Invariance loss: ensure that transition dynamics produce same z_d' regardless of z_s.
domain_pred = DomainClassifier(eta)  # eta = transition(z_d, a)
inv_loss = CrossEntropy(domain_pred, uniform)  # maximize entropy (adversarial)

# Sparse causality constraint: add L1 penalty on the attention weights from a to z_s in the encoder.
# Encoder uses bilinear attention: weight = softmax(z_s^T W a), L1 on attention logits before softmax
sparsity_loss = L1(encoder.action_to_static_attention_logits)

Training:
for each batch (x, a, x') in multiple domains:
  z_s, z_d = Encoder(x)
  z_s' = z_s  # static across time
  with stop_gradient(z_s):
      eta = Transition(z_d, a)  # predicted delta
  z_d' = z_d + eta
  x_recon = Decoder(z_s', z_d')
  recon_loss = MSE(x_recon, x')
  
  total_loss = recon_loss + 0.1 * causal_loss + 0.01 * inv_loss + 0.001 * sparsity_loss
  update parameters
```

Hyperparameters: lambda1=0.1, lambda2=0.01, lambda3=0.001, z_s and z_d dimension=64, encoder uses 4-layer CNN (channels 32,64,128,256), transition uses 2-layer MLP (hidden=256, ReLU), predictor and domain classifier are 2-layer MLPs (hidden=128, ReLU).

**Load-bearing assumption and calibration:** We explicitly assume that the encoder can disentangle static and dynamic latents using the proposed losses, such that z_s is truly action-invariant and z_d captures all action-affected dynamics. To verify this assumption, after training we perform a calibration test on a held-out validation set: (i) Compute the coefficient of determination (R²) for the predictor mapping actions to z_s; if R² < 0.05, we consider causal sufficiency satisfied. (ii) Compute the domain classifier's accuracy on transition features; accuracy near chance (1/n_domains) indicates invariance. If either condition fails, we increase the weight of the corresponding loss or adjust model capacity.

(C) **Why this design** — We chose a static-dynamic factorization over a monolithic latent representation (as used in GigaWorld-Policy-0.5) because it forces the model to identify which variables are action-relevant, directly addressing the problem of domain-specific correlations. The causal sufficiency loss using a predictor from actions to static latents is preferred over a contrastive method (e.g., SimCLR) because it directly measures the dependence we want to remove, accepting the cost that the predictor may not capture all forms of dependence if the mapping is complex. The adversarial domain invariance loss on transition features is chosen over direct alignment of distributions (e.g., MMD) because it is more effective in high-dimensional spaces and does not require a kernel choice; the trade-off is training instability that we mitigate with gradient clipping. Finally, the L1 sparsity constraint on action-to-static attention weights is used instead of a mutual information upper bound because it is computationally cheaper and empirically reduces false positives; the cost is that it may not enforce strict invariance but only sparsity, which is acceptable because we also have the causal loss.

(D) **Why it measures what we claim** — The reconstruction loss measures the model's ability to predict next observations, which operationalizes the concept of dynamics modeling. The causal sufficiency loss (predicting z_s from a) measures the dependence between static latents and actions; the assumption is that the predictor has sufficient capacity to capture any such dependence; this assumption fails when the predictor is too weak, in which case high causal loss would still reflect a violation because the predictor would fail to capture the dependence but the true dependence would remain. The domain adversarial loss on transition features (eta) measures invariance of action effects across domains, under the assumption that the domain classifier achieves the entropy lower bound; failure mode: the domain classifier might be fooled by non-invariant features (e.g., due to insufficient capacity), so we separately monitor classifier accuracy (should be chance for invariant dynamics). The sparsity loss measures the action's direct influence on static latents; the assumption is that the encoder's attention weights correspond to true causal influence; failure mode: encoder may learn non-causal shortcuts, in which case high sparsity may suppress irrelevant connections but not guarantee invariance. To further verify causal sufficiency, we compute a diagnostic metric: the causal effect of actions on z_s using the change in z_s when actions are intervened (e.g., via randomized action perturbations); if the average effect is below 0.01, the assumption holds.

## Contribution

(1) A novel causal factorized world model that explicitly separates action-affected dynamic latents from environment-static latents using causal sufficiency constraints. (2) A training framework that enforces domain invariance of action effects via an adversarial loss on transition features, eliminating the need for large-scale pretraining for domain generalization. (3) Empirical verification on simulated robotics tasks with varying backgrounds demonstrating zero-shot transfer to unseen domains with dynamics accuracy comparable to models pretrained on large data.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Multi-domain Robomimic (varying backgrounds, object colors, table textures, 5 domains total) | Tests invariance across visual domains |
| Primary metric | Task success rate per domain (averaged over 100 trials per domain) | Directly measures task completion accuracy |
| Diagnostic metric | Causal Effect Regularization (CER): average KL divergence between z_s distributions conditioned on different actions | Verifies action does not influence z_s |
| Baselines | mimic-video (video-conditioned action) | Baseline without causal factorization |
| | GigaWorld-Policy-0.5 (monolithic world model) | Baseline without static/dynamic separation |
| Ablation-of-ours | CWM w/o domain adversarial loss | Tests necessity of invariance enforcement |

### Why this setup validates the claim

This experimental design tests the central claim that explicit separation of static and dynamic latents, enforced by causal sufficiency and invariance losses, enables domain generalization. A multi-domain benchmark (Robomimic with 5 domains: varying backgrounds, object colors, and table textures) ensures that static context shifts while action effects remain consistent. The primary metric, success rate across domains, directly reflects the accuracy of dynamics predictions under distribution shift. Baselines capture contrasting approaches: mimic-video relies on video representations that entangle static and dynamic factors; GigaWorld-Policy-0.5 uses monolithic latents that may learn spurious correlations. The ablation (CWM w/o domain adversarial loss) isolates the impact of the invariance constraint. If CWM outperforms both baselines and its ablation on out-of-domain tasks, it validates the claim that causal factorization is beneficial; if not, the claim is falsified. Additionally, we compute the diagnostic metric CER to verify that the causal sufficiency loss indeed removes action influence on z_s; if CER is low (e.g., <0.01), the assumption holds.

### Expected outcome and causal chain

**vs. mimic-video** — On a case where background color changes from red to blue but the action of pushing a cube is identical, mimic-video may incorrectly predict the next state because its video encoder confuses static background with dynamic state, leading to lower success on unseen backgrounds. Our method explicitly encodes background into static latent z_s, which is unused for transition, so the dynamic latent z_d remains invariant. We expect CWM to achieve >90% success on training and test backgrounds, while mimic-video drops to ~40% on test backgrounds.

**vs. GigaWorld-Policy-0.5** — On a case where a static table color is correlated with object friction during training, the monolithic model may learn that table color influences dynamics, causing failure when friction changes. Our method's causal sufficiency loss forces z_s (table color) to be independent of actions, and the transition operates solely on z_d (object state). Thus, table color changes do not affect predicted dynamics. We expect CWM to maintain >85% success on all friction conditions, while GigaWorld-Policy-0.5 degrades to ~50% on the unseen friction+color combination.

**Ablation: CWM w/o domain adversarial loss** — Without the invariance constraint, the transition model might inadvertently use z_s information (e.g., background texture) to predict dynamics due to residual correlation, causing failure when background changes. With the domain adversarial loss, the transition features become domain-agnostic. We expect full CWM to outperform its ablation by at least 15% on out-of-domain test sets, particularly those with large static feature shifts.

### What would falsify this idea

If CWM does not outperform the monolithic baseline on out-of-domain tasks, especially when static features vary across domains, the central claim of benefit from causal factorization is wrong. More specifically, if the ablation (w/o domain adversarial) performs comparably to full CWM, then the invariance loss is unnecessary, contradicting the idea that explicit invariance enforcement is critical. Additionally, if the diagnostic CER is high (>0.1), the causal sufficiency assumption fails and the method would require redesign.

## References

1. GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch
2. mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs
3. GigaWorld-0: World Models as Data Engine to Empower Embodied AI
4. π0: A Vision-Language-Action Flow Model for General Robot Control
5. Latent Action Pretraining from Videos
6. The Ingredients for Robotic Diffusion Transformers
7. Visual Instruction Tuning
8. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
