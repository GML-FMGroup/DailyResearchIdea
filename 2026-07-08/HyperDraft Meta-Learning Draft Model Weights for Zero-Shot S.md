# HyperDraft: Meta-Learning Draft Model Weights for Zero-Shot Speculative Decoding

## Motivation

Existing speculative decoding methods like DSpark train a separate draft model for each target LLM, which is costly and hinders deployment across diverse architectures. The root cause is that draft training ignores the structural similarity among target models: they share a common language distribution and differ mainly in a low-dimensional 'style' representation. Prior work on hypernetworks has shown that weights for a task can be generated from a compact description of the task, but no method applies this to draft model generation for speculative decoding.

## Key Insight

The output distribution of a target model on a small probe set captures its 'style' — the low-dimensional variation that determines how a draft model should behave — and a hypernetwork trained on a family of targets can learn to map that distribution vector to effective draft weights.

## Method

### (A) What it is
HyperDraft is a meta-learned hypernetwork that takes as input a behavioral fingerprint of a target LLM and outputs the weights of a small transformer-based draft model (4 layers, 4 attention heads, 256 hidden dim, vocab 50k). The fingerprint is computed over an adaptive, diversity-driven probe set of 100 prompts. The hypernetwork is trained on a distribution of target models (e.g., GPT-2 variants of different sizes) to minimize the expected KL divergence between the draft model's output and the target's output on a held-out validation set. At test time, given an unseen target, its fingerprint is computed and fed to the hypernetwork to obtain draft weights directly — no gradient updates are needed.

### (B) How it works
```python
# Pseudocode for HyperDraft training
# Let H be hypernetwork (MLP with 2 hidden layers, 512 units, ReLU)
# Let D be draft model architecture (fixed, small transformer)
# Let T be set of target models in meta-training (e.g., GPT-2 small/medium/large/XL, OPT-125M/350M/1.3B, Pythia-160M/410M/1B)
# Let P_adap be adaptive probe set of 100 prompts (diversity-driven, selected via k-means on target embeddings)
# Let V be held-out validation set (different prompts, 10k examples from OpenWebText)

for each target_model in T:
    # compute fingerprint with adaptive selection
    # Step 1: Estimate target model's embedding space by running 1000 random prompts, collecting hidden states from last layer
    # Step 2: Cluster embeddings into 100 clusters via k-means, select closest prompt to each cluster center
    # Step 3: For each selected prompt, compute top-100 logits from target_model, flatten to 100*100=10k floats
    # Step 4: Compute additional statistics: average entropy of target outputs over probe set (scalar), agreement with reference model (GPT-2 small) measured as fraction of identical top-1 tokens (scalar)
    # Step 5: Concatenate: 10k (log-probs) + 1 (entropy) + 1 (agreement) = 10,002 floats
    fp = compute_adaptive_fingerprint(target_model)
    
    # generate draft weights
    draft_weights = H(fp)  # H outputs 4*256*256 (self-attn) + 4*256*256 (cross-attn?) + ... weights as flat vector
    
    # set draft model parameters
    D.set_weights(draft_weights)
    
    # compute loss on validation set
    total_kl = 0
    for batch in V (batch_size=16):
        target_logits = target_model(batch)
        draft_logits = D(batch)
        kl = KL(target_logits || draft_logits)
        total_kl += kl
    
    # update hypernetwork H to minimize total_kl (via SGD, lr=1e-4, 50k steps)
```
Hyperparameters: learning rate 1e-4, batch size 16, meta-training 50k steps. Hypernetwork training budget: 4 GPUs (NVIDIA A100), 2 weeks. Per-target adaptation cost: fingerprint computation ~1 minute on single GPU, draft weight generation <1 second.

### (C) Why this design
We chose a behavioral fingerprint over architectural descriptors (e.g., layer counts) because the key determinant of an effective draft is how the target assigns probabilities, not its internal structure; this accepts the cost that the fingerprint is larger (10k floats plus auxiliary stats) but ensures it captures model-specific behavior. We used an MLP hypernetwork rather than a transformer to keep inference fast and because the mapping from fingerprint to weights is assumed to be smooth and low-dimensional; this may limit capacity for complex mappings but avoids overfitting given limited meta-training targets. The training objective is KL divergence on a held-out validation set rather than acceptance rate directly, because KL is differentiable and correlates with acceptance; the trade-off is that minimizing KL does not guarantee the same acceptance rate as optimizing for it, but it is a more stable and computationally efficient surrogate. We augmented the fingerprint with entropy and agreement to capture distributional properties beyond top-k probabilities, assuming that two targets with similar top-k but different entropies may require different draft behaviors; the failure case is that entropy and agreement still do not capture all relevant aspects (e.g., long-range dependencies).

### (D) Why it measures what we claim
The fingerprint vector (top-100 log-probs over 100 diverse prompts plus entropy and agreement) measures the target model's 'style' under the assumption that two models with similar output distributions on the probe set will require similar draft model behaviors; this assumption fails when two targets have nearly identical probe outputs but differ in internal mechanisms (e.g., different architectures that happen to produce similar probabilities), in which case the fingerprint may be insufficient to capture all necessary draft characteristics. The KL divergence between draft and target distributions on the validation set measures the draft's fidelity to the target's distribution under the assumption that the validation set is representative of the deployment distribution; this assumption fails when the validation set does not cover the diverse patterns encountered during inference, in which case a low KL may still lead to poor acceptance on out-of-distribution prompts. Additionally, we perform PCA on collected fingerprints and the corresponding optimal draft weights (obtained via LoRA fine-tuning on a subset of targets) to verify that the mapping is low-rank and learnable; if the first few PCs explain high variance, it supports the assumption.

## Contribution

(1) A meta-learning framework, HyperDraft, that generates draft model weights for any target LLM from a compact behavioral fingerprint, eliminating per-target training. (2) The identification of top-k log-probabilities over a shared probe set as a sufficient and efficient fingerprint for draft model generation. (3) Empirical demonstration that a hypernetwork can zero-shot generalize to unseen target architectures (e.g., from GPT-2 to OPT models) with draft quality comparable to trained-from-scratch draft models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | OpenWebText (100k prompts) | Diverse natural language coverage |
| Primary metric | Token acceptance rate | Direct measure of draft quality |
| Additional metric | Total GPU-hours (training+adaptation) | Quantifies cost savings |
| Baseline 1 | Autoregressive decoding | Baseline for speed comparison |
| Baseline 2 | Fixed draft model (no adaptation) | Tests need for adaptation |
| Baseline 3 | Per-target fine-tuned draft (LoRA) | Tests efficiency vs. fine-tuning |
| Ablation 1 | HyperDraft with random fingerprint | Isolates fingerprint importance |
| Ablation 2 | HyperDraft without diversity selection | Tests adaptive selection importance |
| Analysis | PCA on fingerprint-draft mapping | Validates low-rank structure |

### Why this setup validates the claim
This combination tests the central claim that a behavioral fingerprint is sufficient to generate a draft model that adapts to any target LLM without gradient updates. The diverse OpenWebText dataset ensures the method is evaluated on varying linguistic patterns, testing generalization. Token acceptance rate directly measures draft quality—the key driver of speculative decoding speedup. The total GPU-hours metric quantifies the cost advantage over per-target fine-tuning. The baselines isolate distinct sub-claims: autoregressive decoding provides a speed baseline; the fixed draft model tests whether adaptation is necessary (if our method fails to outperform it, the fingerprint is useless); per-target fine-tuning (LoRA) tests whether our zero-shot adaptation is competitive with expensive gradient-based methods. The ablations (random fingerprint, no diversity selection) verify that the fingerprint contains useful information and that the adaptive selection is beneficial. The PCA analysis provides evidence that the mapping from fingerprint to draft weights is low-rank, grounding the method theoretically. If our method matches or exceeds the per-target fine-tuning while requiring orders of magnitude less compute, and significantly outperforms the fixed draft, the claim is supported. Conversely, if our method is not better than the fixed draft, or if the ablations perform similarly, the claim is falsified.

### Expected outcome and causal chain

**vs. Autoregressive decoding** — On a prompt where the target model exhibits high output variance (e.g., rare word sequences), autoregressive decoding processes one token per step, limiting wall-clock speed. Our method leverages the draft model to propose multiple tokens per step, verifying them in parallel. We expect HyperDraft to achieve a consistent speedup (2-4×) across all prompts, with larger gains on longer sequences due to higher speculative acceptance.

**vs. Fixed draft model (no adaptation)** — On a target model trained on a specialized domain (e.g., medical text) that differs from the fixed draft's pretraining data, the fixed draft produces low acceptance rates because its distribution does not match the target's. Our method adapts the draft weights via fingerprint (computed from target's log-probs on probe prompts), so the draft better approximates the target's distribution. We expect a significant gap: our acceptance rate remains high (e.g., >0.8) while the fixed draft's drops below 0.5 on such out-of-distribution targets.

**vs. Per-target fine-tuned draft (LoRA)** — On a large target model (e.g., 70B parameters), fine-tuning a LoRA adapter on the draft requires collecting target outputs and running gradient updates, taking hours (e.g., 100 GPU-hours for 10 models). Our method instead computes a fingerprint (seconds) and directly predicts draft weights (milliseconds), yielding comparable acceptance with zero training cost. We expect our method to achieve acceptance rates within 5% of the LoRA baseline while being orders of magnitude faster to deploy. Total cost for HyperDraft (training 4 GPUs for 2 weeks) is fixed, while per-target LoRA scales linearly with number of targets; for 20 targets, HyperDraft saves >80% total compute.

**Ablation: random fingerprint** — If the random fingerprint ablation yields similar performance to the full method (e.g., within 10% acceptance rate), then the fingerprint conveys no useful information. We expect a large gap (>20%) showing the fingerprint is effective.

**Ablation: no diversity selection** — Using a fixed probe set (e.g., 100 random prompts) leads to lower acceptance on diverse targets compared to adaptive selection. We expect a 5-10% improvement with diversity selection.

**PCA analysis** — We expect the first 10 PCs of the fingerprints to explain >80% variance, and the corresponding draft weights to lie in a low-rank subspace, supporting learnability.

### What would falsify this idea
If HyperDraft's acceptance rate is not consistently higher than the fixed draft baseline across diverse targets, or if the random fingerprint ablation yields similar performance to the full method, then the central claim—that the behavioral fingerprint captures sufficient information to generate a good draft model—would be falsified. Additionally, if the PCA analysis shows no low-rank structure, the grounding of the mapping is weak.

## References

1. DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation
2. DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding
3. Speculative Diffusion Decoding: Accelerating Language Generation through Diffusion
4. Fast Inference from Transformers via Speculative Decoding
5. Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution
6. Score-based Continuous-time Discrete Diffusion Models
7. Concrete Score Matching: Generalized Score Matching for Discrete Data
