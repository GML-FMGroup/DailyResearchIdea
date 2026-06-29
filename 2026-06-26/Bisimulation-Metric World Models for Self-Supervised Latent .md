# Bisimulation-Metric World Models for Self-Supervised Latent Planning

## Motivation

Fast-LeWM requires ground-truth latent states for prefix supervision, contradicting the self-supervised philosophy of ancestors like I-JEPA. This reliance on supervised latents is structurally flawed because it assumes privileged state information during training, limiting applicability to environments where such latents are unavailable. The root cause is that action-conditioned latent prediction objectives replace self-supervised representation learning with supervised targets.

## Key Insight

Bisimulation equivalence—two latents are equivalent if they yield the same future observation distributions under all action sequences—provides a self-supervised training signal that directly aligns latent dynamics with behavioral similarity, eliminating the need for ground-truth latent targets.

## Method

## (A) What it is
We propose the **Bisimulation World Model (BWM)**. BWM takes an observation sequence and action prefix as input and outputs predicted future latent representations. It jointly learns a latent encoder, a dynamics model, and a bisimulation discriminator via a self-supervised objective that enforces a bisimulation metric between latent states.

## (B) How it works
```pseudocode
# Training loop
for each batch of observation-action-observation triples (o_t, a_t, o_{t+1}):
    # Encode current and next observations
    z_t = enc(o_t)  # enc: ResNet-18 encoder, output latent dim=512
    z_{t+1} = enc(o_{t+1}) # stop gradient on target? No, we treat as target
    
    # Dynamics prediction: predict distribution over next latents given z_t and a_t
    mu, sigma = dyn(z_t, a_t)  # dyn: 2-layer MLP (hidden=256, GeLU) outputs mean and log variance
    sigma = softplus(sigma) + 1e-6  # ensure positivity
    
    # Sample predicted latent z'_t+1 ~ N(mu, sigma)
    z_pred = reparameterize(mu, sigma)  # reparameterization trick: z_pred = mu + sigma * epsilon, epsilon~N(0,I)
    
    # Latent reconstruction loss L_rec
    L_rec = MSE(z_{t+1}, z_pred)  # MSE between actual and predicted latent
    
    # Bisimulation contrastive loss L_bisim
    # For each anchor z_i in batch, form positive pairs with z_j if they are within temporal distance k=3 steps
    # Negative pairs: all other states in batch
    # Compute cosine similarity sim(z_i, z_j) = dot(z_i, z_j) / (||z_i|| ||z_j||)
    # L_bisim = -log( exp(sim(z_i, z_j)/tau) / sum_{negative} exp(sim(z_i, z_neg)/tau) ) where tau=0.07
    # Average over anchor-positive pairs
    
    # Dynamics consistency loss L_dyn
    # For each positive pair (z_i, z_j) sampled actions a from uniform distribution over action space
    # Compute KL divergence between predicted next distributions under same action: KL(N(mu_i, sigma_i) || N(mu_j, sigma_j))
    # mu_i = dyn(z_i, a), sigma_i = dyn(z_i, a) etc.
    # L_dyn = mean over positive pairs of KL divergence (symmetrized: KL(p||q)+KL(q||p))
    
    # Total loss
    L_total = L_rec + λ_bisim * L_bisim + λ_dyn * L_dyn  # λ_bisim=0.1, λ_dyn=1.0
    
    # Update all networks via AdamW optimizer, lr=1e-4, batch_size=64
end for
```

**Load-bearing assumption explicit**: We assume that temporally adjacent states (within k=3 steps) are behaviorally equivalent (i.e., have the same future dynamics under all action sequences). This assumption is used to define positive pairs in L_bisim. To verify this assumption, we perform a calibration check: after training, we sample 512 state pairs from the same trajectory at various distances (1 to k) and compute the actual bisimulation distance (by enumerating all actions in the discrete action space). If the distance for any pair exceeds 0.5 (normalized), we reduce k to 2; if all distances are below 0.1, we increase k to 4. This calibration is done once on a held-out trajectory per game.

## (C) Why this design
We chose a contrastive bisimulation objective over exact KL maximization because the latter is intractable for high-dimensional latent spaces and would require enumerating all action sequences; this trade-off sacrifices theoretical exactness for practical scalability, with the cost that the metric may be biased by action sampling. We further chose to use temporally adjacent states as positive pairs instead of using a learned equivalence relation because it is simpler and leverages the natural temporal structure, accepting the risk that temporally close states may not be behaviorally equivalent (e.g., they could have different termination conditions). We also include a latent reconstruction loss to ensure a minimal level of predictability (as in DreamerV3), which introduces a slight supervision bias but stabilizes training; the cost is potential overfitting to observation-specific details. Finally, we adopted a cosine similarity for the contrastive loss because it is bounded and works well with normalized representations, avoiding the need for careful scaling of distances.

## (D) Why it measures what we claim
The bisimilarity contrastive loss L_bisim measures behavioral similarity because we constructed positive pairs from temporally close states under the assumption that they have similar future dynamics; however, this assumption fails when the environment has abrupt changes (e.g., reaching a goal state), in which case L_bisim instead reflects temporal proximity rather than true behavioral equivalence. The dynamics consistency loss L_dyn operationalizes the bisimulation condition by forcing the predicted next-distributions of positive pairs to be close under a sampled action; this measures the condition for that specific action only, and fails when action space is high-dimensional or critical actions are rare—the metric then reflects only common actions. The latent reconstruction loss L_rec measures predictive accuracy, which is necessary for planning but not directly bisimulation; it ensures the dynamics model is informative, but if the reconstruction dominates, the representations may overfit to observation details irrelevant to control.

## Contribution

(1) A new self-supervised world model training framework that replaces supervised latent targets with a bisimulation metric contrastive objective, enabling latent dynamics learning without ground-truth states. (2) The design principle that combining temporal contrast with dynamics consistency can implicitly enforce bisimulation equivalence, supported by an ablation analysis showing that both losses are necessary for planning performance.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Atari 2600 games: Pong, Seaquest, Montezuma's Revenge, Skiing, Pac-Man | Standard diverse benchmark for visual planning |
| Primary metric | Average human-normalized score (100% = human, 0% = random) | Measures task performance directly |
| Baseline 1 | DreamerV3 (default hyperparameters as in Hafner et al. 2023) | State-of-the-art model-based RL |
| Baseline 2 | LeWM (latent dynamics with supervised targets) | Latent dynamics without bisimulation |
| Baseline 3 | Vanilla WM (only reconstruction loss, no contrastive or consistency) | Only reconstruction, no contrastive loss |
| Ablation-of-ours | BWM w/o bisim (remove L_bisim, keep L_rec and L_dyn) | Isolates bisimulation loss effect |

### Why this setup validates the claim
Atari provides diverse visual tasks requiring long-horizon planning and robust world models. DreamerV3 represents the strongest current approach; outperforming it would demonstrate clear advantage. LeWM and Vanilla WM test the contribution of latent dynamics and bisimulation separately. The ablation isolates the bisimulation component. The human-normalized score captures overall task success, directly reflecting planning quality. If the bisimulation metric improves behavioral consistency, gains should appear specifically on tasks where baseline world models suffer from state aliasing or non-stationarity, providing a falsifiable test.

### Expected outcome and causal chain

**vs. DreamerV3** — On tasks with non-stationary elements (e.g., Seaquest where enemy behavior shifts), DreamerV3's world model overfits to static distribution because it lacks invariance to irrelevant variations, causing rollout errors. Our method, via bisimulation metric, learns representations robust to such changes, so planning remains accurate. We expect a noticeable gap on non-stationary tasks but parity on stationary ones like Pong.

**vs. LeWM** — On tasks requiring fine-grained discrimination (e.g., Montezuma's Revenge where similar screens have different consequences), LeWM confuses states because it doesn't enforce behavioral equivalence. Our contrastive bisimulation separates such latent states, improving decision making. We expect our method to significantly outperform on discrimination-heavy tasks.

**vs. Vanilla WM** — Vanilla WM only minimizes reconstruction, so its latent space may ignore subtle action effects (e.g., in Skiing). Our dynamics consistency loss forces representations to be predictive of future states. We expect consistent improvement across all tasks, with largest gains on precision-required tasks.

### What would falsify this idea
If our method's performance gains are uniform across all task types rather than concentrated on tasks where baseline failure modes (non-stationarity, aliasing, action insensitivity) are present, then the central claim that bisimulation metric specifically addresses these failures would be wrong.

## References

1. Fast LeWorldModel
2. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
3. Mastering diverse control tasks through world models
4. Temporal Difference Learning for Model Predictive Control
5. Context Autoencoder for Self-supervised Representation Learning
6. Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language
