# Causal Action Masking World Models: Learning Action Representations via Sparse Causal Masks in Latent Space

## Motivation

Existing world models for robotics require either explicit chain-of-thought reasoning (e.g., VLA-R1) or video reconstruction (e.g., SwiftVLA) to achieve interpretability, introducing computational overhead and reliance on language or pixel-level supervision. The root cause is that these methods treat actions as monolithic inputs to a dynamics model, failing to disentangle which latent factors an action causally affects. For instance, GigaWorld-Policy-0.5 uses action-conditioned world modeling but does not model causal structure, limiting interpretability. We address this by learning a causal directed acyclic graph (DAG) over latent state variables, where actions are represented as sparse intervention masks, enabling causally interpretable action representations without explicit reasoning or reconstruction.

## Key Insight

Actions causally affect only a subset of latent environment factors, and this intervention structure can be learned via temporal contrastive learning that enforces graph invariance across time, yielding interpretable masks without requiring explicit causal inference.

## Method

**Causal Action Masking World Model (CAM-WM)**

(A) **What it is**: We propose CAM-WM, which takes as input an observation $o_t$ and outputs a latent state $z_t$, learns a single time-invariant causal DAG adjacency matrix $G$ over latent dimensions (load-bearing assumption fix), and represents actions as sparse binary masks $\mathbf{m}_t$ that gate which latent variables are intervened upon. The model predicts the next latent state $z_{t+1}$ under the intervention defined by $\mathbf{m}_t$.

(B) **How it works** (pseudocode):
```python
# Hyperparameters
latent_dim = 64
temp_lambda = 0.1  # contrastive loss weight (default)
beta = 0.01        # DAG penalty weight (default)
gamma = 0.001      # sparsity weight (default)
d = latent_dim

# Learnable parameters
G_params = nn.Parameter(torch.randn(d, d))  # single time-invariant adjacency matrix
base = nn.Parameter(torch.zeros(d))          # learned baseline for intervention

for each transition (o_t, a_t, o_{t+1}) in buffer:
    # Encode observations to latent
    z_t = encoder(o_t)      # shape: [batch, d]
    z_{t+1} = encoder(o_{t+1})
    
    # Causal graph: time-invariant, zero diag, sigmoid for edge weights
    G = torch.sigmoid(G_params)  # shape: [d, d]
    G = G * (1 - torch.eye(d))   # zero out self-loops
    # Apply continuous acyclicity penalty: h(G) = tr(expm(G)) - d
    dag_penalty = torch.trace(torch.matrix_exp(G)) - d
    
    # Action mask: sparse binary vector m_t from action embedding
    # Use a 2-layer MLP (hidden=128, ReLU) to produce logits
    m_t_logits = action_mask_network(action_embedding(a_t))  # shape: [batch, d]
    m_t = torch.sigmoid(m_t_logits)  # continuous relaxation during training; binarize at test by threshold 0.5
    
    # Apply intervention: set intervened variables to a learned base value
    z_t_int = (1 - m_t) * z_t + m_t * base.unsqueeze(0)
    
    # Predict next latent using graph structure
    # For each latent dimension i, use its parents in G (G[j,i] > 0 means j->i)
    # Use a 2-layer MLP (hidden=256, GeLU) for transition
    parent_contrib = torch.matmul(z_t_int, G)  # [batch, d] via G^T? Actually we want parents: G^T * z_t_int gives weighted sum of parents for each i
    # Better: weight = G.permute(1,0) so row i gives incoming edges to i
    parent_contrib = torch.matmul(z_t_int, G.T)  # [batch, d]
    pred_mean, pred_logvar = transition_mlp(parent_contrib)  # [batch, d], [batch, d]
    
    # Temporal contrastive loss (InfoNCE style)
    # Positive log-likelihood
    log_prob_pos = -0.5 * ( ((z_{t+1} - pred_mean)**2) / torch.exp(pred_logvar) + pred_logvar + math.log(2*math.pi) ).sum(-1)
    # Negative: sample K different actions (from other samples in batch) and compute log_probs similarly
    # For simplicity, we use actions from other samples in batch; K = batch_size - 1
    # Compute negative log-probs for each sample's next latent under other actions' masks
    # (omitted for brevity, but standard within-batch negative sampling)
    loss_contrast = - log_prob_pos.mean() + temp_lambda * torch.logsumexp(negative_log_probs, dim=0).mean()
    
    total_loss = loss_contrast + beta * dag_penalty + gamma * m_t.abs().sum(dim=-1).mean()
    optimize(total_loss)
```

(C) **Why this design**: We chose to learn a single time-invariant latent causal graph (load-bearing assumption fix) over learned latent variables because latent variables can capture meaningful factors while being lower-dimensional, and a fixed graph avoids non-identifiability from cross-sectional data. We represent actions as sparse intervention masks rather than as continuous vectors that directly modify all latent dimensions, because sparsity enforces that actions only affect a few causal factors, aligning with the principle of independent causal mechanisms and improving interpretability. We use temporal contrastive loss with negative action samples instead of pure reconstruction because contrastive learning forces the model to distinguish between different action effects, thereby learning a more discriminative intervention representation. The trade-off is that the contrastive loss requires sampling negative actions, which adds computational overhead; however, it avoids the need for explicit reconstruction of observations, which would introduce pixel-level noise. We use a continuous acyclicity penalty (trace expm) to enforce DAG structure, accepting that it may not guarantee exact acyclicity but is differentiable and scalable. The L1 sparsity on action masks encourages few intervened variables, but may overly penalize necessary interventions if the environment requires multi-factor changes; we mitigate this by tuning gamma per task. This design contrasts with approaches like GigaWorld-Policy-0.5 that model actions as unconditional inputs, as we specifically isolate which latent factors are affected.

(D) **Why it measures what we claim**: The learned causal graph $G$ measures the structural causal dependencies among latent factors because we assume a time-invariant graph and that the transition model factorizes according to $G$; this assumption fails when the true causal graph changes over time or when latent variables are not independent, in which case $G$ captures only the average dependence structure. The action mask $\mathbf{m}_t$ measures which latent variables are causally affected by the action because we define intervention as setting those variables to a base value, making their prediction independent of their previous value; this assumption fails when actions affect continuous variables in a non-binary way (e.g., applying force), in which case the mask indicates only that an intervention occurred but not its magnitude. The temporal contrastive loss measures the discriminability of action effects because it maximizes the likelihood of the observed next latent under the true action mask relative to other actions; this assumption fails when actions have overlapping effects, in which case the loss may be dominated by the most distinguishing features. The sparsity penalty measures the minimal set of intervened variables because L1 regularization biases towards zero; this assumption fails when there are truly many variables affected, in which case sparsity forces a trade-off between accuracy and mask sparsity.

## Contribution

(1) A novel world model framework (CAM-WM) that learns a latent causal DAG and represents actions as sparse intervention masks, enabling causally interpretable action representations without requiring explicit reasoning or video reconstruction. (2) A temporal contrastive learning objective that jointly discovers the causal graph and learns intervention masks by maximizing the likelihood of observed transitions under the correct action intervention. (3) Design principle that causal structure can be learned in latent space for robotics by exploiting the invariance of graph structure across time and the sparsity of action effects.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | MetaWorld manipulation task suite (e.g., push, pick-place) | Contains diverse causal structures and sparse action effects |
| Additional verification | Synthetic dataset with known causal graph (e.g., 5-dim linear SCM) | Verify learned graph matches ground truth (edge accuracy) |
| Primary metric | Action discrimination accuracy | Directly tests intervention mask's ability to distinguish actions |
| Baseline 1 | Dreamer-V2 (no causal graph) | Tests necessity of causal graph and mask |
| Baseline 2 | CAM-WM w/o sparsity (dense mask) | Tests benefit of sparsity in action masks |
| Ablation | CAM-WM w/o DAG penalty (fully connected) | Tests whether DAG constraint is useful |

### Why this setup validates the claim
This combination tests the core claim that learning a causal DAG over latent variables and representing actions as sparse intervention masks improves world model accuracy. The MetaWorld dataset offers tasks with known ground-truth causal factors (e.g., object positions) and actions that affect only a subset, enabling falsification: if our method outperforms Dreamer-V2 (which has no causal structure), it demonstrates the benefit of causal modeling; the non-sparse baseline isolates whether sparsity is essential; the fully-connected ablation tests whether enforcing acyclicity matters. Action discrimination accuracy directly measures how well the mask-encoded interventions differentiate actions, which is the key design objective. The synthetic dataset with known causal graph provides a controlled setting to verify that the learned time-invariant adjacency matrix recovers the true DAG, addressing the fundamental identifiability concern.

### Expected outcome and causal chain
**vs. Dreamer-V2** — On a task where a single action (e.g., "push cube left") should affect only the cube's position, Dreamer-V2's dense action embedding changes all latent dimensions, causing spurious correlations and poor generalization to unseen object layouts. Our method learns a graph where the cube's position variable is the only child of the action, and the mask gates only that variable, preserving other factor dynamics. Thus we expect a clear gap in action discrimination accuracy (e.g., >10% improvement) on tasks with sparse causal effects, but parity on tasks where actions affect many factors.

**vs. CAM-WM w/o sparsity** — On a task where actions affect only 2 out of 64 latent dimensions, the dense mask version must learn to set most mask weights near zero, but suffers from gradient noise and overfitting to irrelevant variables. Sparse mask with L1 regularization directly enforces few non-zero entries, leading to more stable training and better separation of action effects. We expect a moderate gap (e.g., 5% improvement) on tasks with high sparsity, but no gap on tasks where actions affect many dimensions equally.

**vs. CAM-WM w/o DAG penalty** — On a task where the true causal graph is a DAG (acyclic), the fully connected version may overfit to spurious correlations and degrade generalization. Our acyclicity constraint forces a feedforward structure, which is appropriate for transitions with temporal ordering. We expect the DAG version to perform similarly on acyclic tasks but better on tasks where temporal causality is directed; however, if the true graph has cycles, the non-DAG version might slightly outperform, revealing the limitation. The ablation controls for this trade-off.

**Synthetic dataset verification**: We expect the learned $G$ to achieve >90% edge accuracy (F1) relative to the ground-truth causal graph, confirming that the time-invariant graph learning approach is feasible for small-scale controlled settings.

### What would falsify this idea
If CAM-WM's advantage over Dreamer-V2 is uniform across all tasks regardless of causal sparsity, or if the non-sparse version performs equally well, then the central claims about causal graph and mask sparsity are not supported. Additionally, if the learned graph on synthetic data fails to recover the ground-truth DAG (edge accuracy <50%), the underlying assumption of time-invariant graph identifiability is invalidated.

## References

1. GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch
2. mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs
3. GigaWorld-0: World Models as Data Engine to Empower Embodied AI
4. π0: A Vision-Language-Action Flow Model for General Robot Control
5. Latent Action Pretraining from Videos
6. The Ingredients for Robotic Diffusion Transformers
7. Visual Instruction Tuning
8. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
