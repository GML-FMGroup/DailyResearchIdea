# Masked Skill Transformer: Self-Supervised Skill Extraction from Incomplete Trajectories via Step Reconstruction

## Motivation

OPID extracts hierarchical skills from complete on-policy trajectories but relies on reliable identification of critical timesteps and decomposable trajectory structure. This fails when trajectories are incomplete, suboptimal, or lack clear decomposability—common in real-world agentic data. The structural root cause is that existing methods treat trajectory quality and completeness as given, whereas a self-supervised reconstruction objective can learn robust skill representations invariant to these factors.

## Key Insight

A trajectory's skill-level pattern is a latent variable that can be predicted from any subset of its steps; training a transformer to reconstruct masked steps forces the pooled representation to capture this latent variable, yielding embeddings that are robust to incompleteness and suboptimality.

## Method

## (A) What it is
**Masked Skill Transformer (MST)** is a self-supervised framework that takes a trajectory of (state, action) steps, randomly masks a subset of steps, and trains a transformer encoder to reconstruct the masked tokens. The pooled representation of the entire sequence (e.g., CLS token or mean pooling) serves as a skill embedding. This design rests on the assumption that the mean pooled embedding from a transformer trained with masked reconstruction captures the trajectory-level skill representation. To strengthen this, we add a contrastive learning objective that explicitly forces the pooled embedding to be similar for trajectories sharing the same skill and dissimilar for different skills during training. The embedding can be used to condition downstream policies or to provide dense supervision for reinforcement learning.

## (B) How it works
```pseudocode
# Input: trajectory τ = [(s1,a1), (s2,a2), ..., (sT,aT)] (may have missing steps represented as [MASK])
# Hyperparameters: mask_ratio = 0.15, embed_dim = 512, num_heads = 8, num_layers = 6, temperature τ_contrast = 0.1, λ = 0.5

1. Tokenize each step: for each t, compute token x_t = LLM_embed(s_t) + LLM_embed(a_t) (or a learned projection)
2. Randomly select a subset M of positions (15% of T) to mask; replace x_t with [MASK] token for t in M
3. Add positional embeddings to all tokens
4. Pass sequence through a transformer encoder (Vaswani et al., 2017) to get hidden states h_t for each position
5. For masked positions, apply a linear+softmax head to predict the original token (or a discretized version)
6. Reconstruction loss L_recon = cross-entropy on masked positions only
7. For contrastive loss: for each trajectory in the batch, create a positive pair by taking two random segmentations (or two trajectories with the same goal if available). Compute pooled embedding z = mean(h_1, ..., h_T) for each trajectory. Apply InfoNCE loss L_contrast = -log(exp(sim(z_i, z_i+)/τ) / Σ exp(sim(z_i, z_j)/τ)) where sim is cosine similarity, τ=0.1, and the sum is over all pairs in the batch. Total loss = L_recon + λ * L_contrast, update transformer parameters.
8. Skill embedding: after training, compute z = mean(h_1, ..., h_T) or use [CLS] token if appended
9. Downstream: concatenate z with state representation for policy conditioning, or use as dense reward signal by measuring similarity to target skill embeddings
```

## (C) Why this design
We chose a **transformer encoder with masked reconstruction** (like BERT) over an **autoencoder with a bottleneck** because reconstruction from any subset of steps forces the model to rely on temporal structure rather than compressing the entire sequence. The trade-off is higher computational cost due to full attention, but it enables handling arbitrary masks. We chose **mean pooling** over [CLS] token to avoid learning position-specific biases, accepting that it may dilute step-level details. We chose **cross-entropy on discretized tokens** over continuous L2 loss because step tokens are naturally categorical (e.g., discrete action IDs and state representations); this introduces quantization error but aligns with the language model backbone commonly used in agentic tasks. The mask ratio of 15% balances learning difficulty and training efficiency—too low provides easy reconstruction, too high loses global context. We commit to a fixed mask probability rather than adaptive masking to keep training stable and avoid additional hyperparameter tuning. The contrastive loss explicitly enforces that pooled embeddings capture trajectory-level skill similarity, mitigating the known issue that BERT-style mean pooling can yield poor sentence embeddings (Reimers & Gurevych, 2019). The temperature of 0.1 and λ=0.5 were chosen based on prior contrastive learning works and small-scale validation on a held-out set.

## (D) Why it measures what we claim
**The reconstruction loss on masked tokens measures the model's ability to capture skill-level patterns** from observed steps, because the assumption is that the latent skill is a sufficient statistic for predicting any missing step given the observed context. This assumption fails when steps are independent of the skill (e.g., random noise), in which case the reconstruction loss measures only local correlations rather than skill content. **The mean pooled embedding z measures the trajectory-level skill** under the assumption that averaging over steps aggregates the skill-relevant information; this assumption fails when skills are localized to specific steps (e.g., a critical decision), in which case z may dilute that signal and reflect only average behavior. **The contrastive loss measures the model's ability to distinguish between different skills** under the assumption that trajectories with the same goal or subgoal share a latent skill; this assumption fails when positive pairs are poorly defined (e.g., two trajectories with same goal but very different step-level patterns), in which case the contrastive loss may enforce false similarity. **The masking procedure simulates incomplete trajectories** under the assumption that missing steps are structurally similar to masked steps; this assumption fails when missingness is systematic (e.g., always missing the last step), in which case the learned representation may overfit to the absence pattern rather than skill content.

## Contribution

(1) A self-supervised masked trajectory modeling framework (MST) that extracts robust skill embeddings from incomplete or low-quality on-policy trajectories without requiring critical step identification or decomposable structure. (2) Design principles for pretraining skill embeddings via step reconstruction on agentic trajectories, including tokenization, masking strategy, and pooling. (3) An operationalization of dense supervision from trajectory-level embeddings that can replace hierarchical skill routing in methods like OPID.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | WebShop | Interactive agentic language task |
| Primary metric | Success rate | Measures task completion directly |
| Baseline 1 | PPO without skills | Tests need for skill abstraction |
| Baseline 2 | OPID | Tests alternative skill distillation |
| Baseline 3 | C-Skill (continuous L2 loss) | Tests discrete vs continuous reconstruction |
| Baseline 4 | ContrastiveTraj (contrastive loss on augmented trajectories without reconstruction) | Isolates benefit of masked reconstruction over pure contrastive learning |
| Ablation of ours | MST without masking (autoencoder) | Tests importance of masked objective |

### Why this setup validates the claim

This setup forms a falsifiable test of MST's central claim: that masked reconstruction of trajectory steps induces skill embeddings that improve downstream policy learning. WebShop is chosen because it requires multi-step reasoning and action sequences, where skill abstraction matters. Comparing against PPO (no skills) isolates the benefit of any skill representation. OPID uses an alternative skill learning method (on-policy distillation) to test whether MST's self-supervised approach is competitive. C-Skill replaces our cross-entropy discrete reconstruction with continuous L2 loss, isolating the effect of discrete token prediction. ContrastiveTraj uses only contrastive learning on trajectory augmentations (e.g., drop random steps) without reconstruction, testing whether the reconstruction objective is key. The ablation (no masking) tests whether masking is critical or a simple autoencoder suffices. Success rate directly reflects policy quality, capturing both exploration and generalization improvements from learned skills.

### Expected outcome and causal chain

**vs. PPO without skills** — On a case where the agent must generalize a multi-step pattern (e.g., "buy item A, then item B"), PPO treats each step independently, leading to slow learning and failure to reuse strategies. Our MST extracts a skill embedding from successful trajectories, enabling the policy to condition on this global pattern. Thus we expect a noticeable gap (e.g., 15-20% higher success) on tasks requiring reuse of subroutines, but parity on single-step tasks.

**vs. OPID** — On a case where the skill is rare in the on-policy data (e.g., a novel combination of actions), OPID struggles because it distills only current policy's behavior. MST's self-supervised reconstruction from past trajectories captures skills without needing a separate distillation phase. We expect MST to outperform OPID on tasks requiring rare or out-of-distribution skills, with a moderate gap (e.g., 5-10% success) on average but larger on specific rare-skill subsets.

**vs. C-Skill (continuous L2 loss)** — On a case where discrete action tokens have no natural ordering (e.g., 'turn left' vs 'pick up'), continuous L2 loss forces an arbitrary metric space, blurring skill boundaries. MST's cross-entropy treats actions as categorical, preserving semantic differences. We expect MST to show clearer skill separation in embedding space, translating to better policy conditioning and a 5-10% success rate advantage on tasks with many distinct action types.

**vs. ContrastiveTraj** — On a case where the agent must infer fine-grained step-level patterns from incomplete trajectories, pure contrastive learning may over-emphasize global similarity and miss local details. MST's reconstruction objective forces the model to capture per-step predictive structure, leading to richer skill embeddings. We expect a 5-10% success rate advantage on tasks requiring precise step ordering (e.g., assembling a recipe).

### What would falsify this idea

If MST underperforms the no-masking ablation (autoencoder) on the primary metric, it would indicate that masking degrades skill learning rather than enhances it. Alternatively, if MST's gains over PPO are uniform across all task types rather than concentrated on tasks requiring long-term or compositional structure, the claim of capturing skill-level patterns would be unsupported. Furthermore, if the pooled embedding z does not cluster by ground-truth skill type (silhouette score < 0.2) or if the silhouette score drops significantly (>0.3) under systematic masking (e.g., always mask last step), the assumption that z captures skill information uniformly across steps is violated.

## References

1. OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning
2. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
3. Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning
4. Large Language Models Are Reasoning Teachers
5. The Llama 3 Herd of Models
6. Qwen2.5 Technical Report
7. Grounding by Trying: LLMs with Reinforcement Learning-Enhanced Retrieval
8. Training Language Models to Self-Correct via Reinforcement Learning
