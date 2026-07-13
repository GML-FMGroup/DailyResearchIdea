# VidARC: Learning Abstract Reasoning via Self-Supervised Video Generation from ARC Tasks

## Motivation

Existing video reasoning datasets like OpenCoF focus on physical or commonsense reasoning but lack abstract reasoning about procedural transformations. ARC tasks require understanding arbitrary rules from few examples, which current video generation models cannot handle because they lack a mechanism to capture invariant transformation patterns across different visual contexts. OpenCoF's reasoning tokens are trained with supervised learning on specific task families, limiting generalization to novel abstract rules.

## Key Insight

By converting ARC tasks into videos and enforcing reasoning tokens to be invariant across different visual instantiations of the same rule via contrastive learning, the model learns to disentangle the abstract rule from the visual content, enabling generalization to unseen rule applications.

## Method

(A) **What it is**: VidARC is a self-supervised framework that trains a video diffusion model (DiT: Diffusion Transformer with 12-layer transformer, hidden size 1024, 16 attention heads, patch size 2, 1.2B parameters) on videos derived from ARC tasks. For each video, a reasoning token is extracted via an encoder head (encoder E: 3D convolutional network Swin3D, 4 stages, outputting 256 tokens of dimension 1024; reasoning token head H: 2-layer MLP with hidden size 4096, GeLU, maps mean-pooled encoded features to token z of dimension 512) and forced to be invariant across videos sharing the same abstract rule through a contrastive loss (InfoNCE, temperature τ=0.07). The token also conditions video reconstruction, ensuring it captures task-sufficient information. To verify the token captures rule identity and prevent shortcut reliance on visual features, we include a small supervised rule-prediction head (linear classifier on z) trained on 10% of tasks (5 tasks) with cross-entropy loss (only used for monitoring). Additionally, each video is augmented with random color jitter (hue shift ±30°, saturation scaling [0.8,1.2]) and grid shifting (±1 cell) to break visual correlations.

(B) **How it works**:
```python
# Input: set of video clips from ARC tasks, each video V with frames f1..fT
# Assume we have multiple examples per task (same rule, different grids).
# Model: video diffusion model with encoder E, denoiser D, and reasoning token head H (outputs token z).
# Contrastive projection head C maps z to embedding e.

for each training step:
  # Sample a batch of videos from same rule (positive pair) and different rule (negative).
  V_i, V_j = positive pair (same rule)
  V_k = negative (different rule)

  # Encode videos to tokens
  z_i = H(E(V_i))
  z_j = H(E(V_j))
  z_k = H(E(V_k))

  # Contrastive loss (InfoNCE) with tau=0.07
  e_i = C(z_i), e_j = C(z_j), e_k = C(z_k)  # C: 2-layer MLP hidden 2048, ReLU, output 128, L2-normalized
  L_cont = -log( exp(sim(e_i,e_j)/tau) / (exp(sim(e_i,e_j)/tau) + exp(sim(e_i,e_k)/tau)) )

  # Video reconstruction loss (Diffusion)
  # Condition denoiser D on z_i (e.g., via cross-attention)
  L_vid = E_t,epsilon [ || epsilon - D( V_i_t, t, z_i ) ||^2 ]

  # Total loss
  L = L_vid + lambda * L_cont   # lambda = 0.1
```
Hyperparameters: lambda=0.1, tau=0.07, diffusion steps=1000. Training uses 8 A100 GPUs for 3 days (600 GPU-hours total). Code skeleton available at https://github.com/anon/vidarc.

(C) **Why this design**: We chose contrastive learning over supervised classification for reasoning tokens because ARC tasks have no predefined rule classes, and contrastive learning allows discovering rule equivalence classes without specifying them. The trade-off is that contrastive learning requires careful data pairing (same vs. different rule), which we obtain from multiple examples per task; this limits the variety of tasks but avoids manual annotation. We used a diffusion loss for video generation because it is compatible with large latent spaces and produces high-quality frames; the cost is increased compute relative to simpler L2 loss. We integrated the reasoning token as a conditioning vector in the denoising U-Net, which forces the token to capture global task structure; alternative approaches like adding token after encoding could be less integrated. The trade-off is that conditioning affects all generation steps, making the token more informative but potentially causing overfitting to specific task visual styles.

(D) **Why it measures what we claim**: The contrastive similarity sim(z_i, z_j) measures abstract rule equivalence because it is computed across videos with different visual content but same transformation rule; the assumption is that the only shared information across such videos is the abstract rule, so token similarity reflects rule identity. This assumption fails when videos from different rules accidentally share low-level visual features (e.g., same colors), in which case similarity may reflect visual similarity instead. To mitigate, we apply data augmentation (color and shape randomization) during video creation to break spurious correlations. The reconstruction loss L_vid measures the token's sufficiency to generate the video; high reconstruction quality indicates the token captures all needed information, including the rule. This assumes that the token is the sole conditioning; if the model also uses visual residuals from previous frames, the token may not fully encode the rule. We control by ablating the token during generation (i.e., removing conditioning) and checking if reconstruction degrades. Load-bearing assumption: Contrastive learning on pairs of videos from the same ARC task forces the reasoning token to encode the abstract rule invariant to visual appearance, enabling generalization to unseen rules. To verify this, we use a supervised rule-prediction head (linear classifier on z) trained on 10% of tasks (5 tasks) with cross-entropy loss. This head is only used for monitoring; if its accuracy on a held-out set of tasks is above chance ( >1/N_rules ), it confirms the token captures rule identity.

## Contribution

(1) Introduces VidARC, a self-supervised framework that converts ARC abstract reasoning tasks into video format and learns reasoning tokens via contrastive invariance across visual variations. (2) Demonstrates that reasoning tokens learned without supervision can capture abstract transformation rules, enabling zero-shot transfer to new rule applications. (3) Provides a new benchmark of video-based ARC tasks for evaluating abstract reasoning in generative models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | ARC-50 video set (50 tasks, 10 videos per task, augmented) + Synthetic Grids (100 tasks, varying grid sizes and colors) | Controlled + test for confounding features. |
| Primary metric | Rule retrieval accuracy (k-NN, k=5) and rule prediction accuracy (from supervised head) | Directly measures token’s rule discrimination. |
| Baseline 1 | Vanilla DiT (no token) | Tests if generation alone captures rules. |
| Baseline 2 | Supervised rule classifier (trained on token from our encoder with labels) | Upper bound with explicit supervision. |
| Ablation-of-ours | VidARC w/o contrastive loss (only L_vid) | Isolates contrastive effect. |

Compute: All experiments run on 8 A100 GPUs. Code skeleton: https://github.com/anon/vidarc.

### Why this setup validates the claim

We test the central claim that VidARC’s reasoning token captures abstract rules by comparing token similarity across videos from known rules. The ARC-50 dataset provides ground-truth rule labels, enabling direct computation of rule retrieval accuracy via nearest-neighbor token matching. Vanilla DiT (no token) controls for whether video generation alone induces rule abstraction; if it does, we would see retrieval performance similar to ours. The supervised classifier provides an upper bound using explicit labels, revealing the headroom. Ablating the contrastive loss isolates its contribution: if the token still works without it, the invariance is not crucial. The metric directly probes the token’s discriminative power for rules, a necessary condition for the proposed reasoning mechanism. Additionally, the Synthetic Grids dataset is designed with controlled visual confounds (e.g., same colors across different rules) to explicitly test whether the token resists sharing low-level features. The rule prediction accuracy from the supervised head (trained on 10% tasks) monitors whether the token encodes rule identity beyond chance.

### Expected outcome and causal chain

**vs. Vanilla DiT** — On a case where two different rules share low-level visual features (e.g., both involve red blocks on a blue background), Vanilla DiT generates plausible videos but its latent representation encodes pixel-level patterns rather than abstract rules, so retrieval accuracy drops to near chance (~10% for 10 rules). Our method, through contrastive learning, forces the token to ignore such surface details; token similarities reflect rule identity, yielding >80% accuracy on such ambiguous subsets. We expect a ~70% gap on visually confusable tasks, with parity on visually distinct ones.

**vs. Supervised rule classifier** — On a held-out rule not seen during training, the supervised classifier has no learned representation and outputs uniform random predictions (accuracy ~1/N). Our method’s token, learned via self-supervised invariance, can still group videos of the same held-out rule together because it relies on relational structure, not class labels. We expect our retrieval accuracy to remain above 50% (due to clustering) while the classifier’s accuracy is at chance. This shows that our token generalizes to unseen rules.

**Ablation (w/o contrastive loss)** — On a task where examples of the same rule have high visual diversity (e.g., different colors each time), the ablation token relies solely on reconstruction loss and learns to memorize specific visual patterns; thus, it fails to recognize that two visually different videos follow the same rule, leading to poor retrieval (<20%). Our full model, with contrastive pressure, learns invariance to such changes, achieving >75% accuracy. The gap is diagnostic: the ablation fails exactly where visual variation is high.

### What would falsify this idea
If VidARC’s rule retrieval accuracy is not significantly higher than Vanilla DiT on visually confusable tasks (e.g., Synthetic Grids), or if the ablation without contrastive loss performs equally well on high-variation tasks, then the claim that contrastive learning induces abstract rule tokens is falsified. Additionally, if the supervised rule-prediction head achieves near-chance accuracy on a held-out task subset, it indicates the token does not capture rule identity.

## References

1. OpenCoF: Learning to Reason Through Video Generation
