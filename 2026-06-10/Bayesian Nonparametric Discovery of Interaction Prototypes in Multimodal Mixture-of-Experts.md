# Bayesian Nonparametric Discovery of Interaction Prototypes in Multimodal Mixture-of-Experts

## Motivation

Existing multimodal mixture-of-experts models, such as I2MoE, require the interaction types to be predefined (e.g., fusion, alignment), limiting their ability to discover novel patterns that emerge from data. This rigid assumption arises because the number and nature of experts are fixed a priori, and no mechanism exists for automatically inferring interaction categories that reflect genuine semantic fusion rather than modality-specific transformations. The field needs a principled approach to open-ended discovery of interaction prototypes, which is essential for adapting to diverse and evolving multimodal tasks.

## Key Insight

A Bayesian nonparametric prior over interaction prototypes, combined with a jointly learned invariant metric that is stable under modality-specific perturbations, ensures that discovered categories correspond to semantically distinct fusion patterns rather than arbitrary modality alignments.

## Method

### (A) What it is
**BNP-MoE** (Bayesian Nonparametric Mixture-of-Experts) is a multimodal architecture that automatically discovers interaction categories by placing a Dirichlet Process prior over a potentially infinite set of interaction prototypes. Each prototype has an associated expert network that processes the fused modality features, and a jointly learned invariant metric ensures that the prototypes capture semantic fusion patterns invariant to modality-specific transformations.

### (B) How it works
```pseudocode
Input: Multimodal pair (x_mod1, x_mod2), task labels y (weakly supervised)
Hyperparameters: α (DP concentration, default=1.0), β (invariance loss weight, default=0.1), L (number of local steps for CRP sampling, default=5), τ (Gumbel-Softmax temperature, initial τ=1.0 annealed to 0.5)

1. Encode each modality: h1 = Enc1(x_mod1), h2 = Enc2(x_mod2)  # Enc1, Enc2: pre-trained BERT (text) and ResNet-50 (video), fine-tuned
2. Concatenate: h = [h1, h2]
3. Maintain a set of K active prototypes {π_k} with embeddings e_k and expert networks f_k (each f_k: 2-layer MLP hidden=256, ReLU, output=512).
4. For each sample, sample assignment z ~ CRP(α):
   - Compute distance d_k = d_metric(h, e_k) for each active prototype. d_metric is a learned 2-layer MLP (hidden=256, ReLU, output=1).
   - Compute probability of a new prototype: p_new = α / (N + α) where N is total assignments seen.
   - Sample z from a categorical distribution over active + new, with weights inversely related to distances: w_k = exp(-d_k) for active, w_new = p_new. Use Gumbel-Softmax with temperature τ for differentiable sampling.
   If z == new: initialize e_new (random vector of dimension 512) and f_new (random init).
5. Compute prototype-specific features: s_k = f_k(h) for the assigned prototype(s) (if multiple, mixture).
6. Task output: y_hat = classifier(s_z)  # classifier: 2-layer MLP hidden=256, softmax output

Loss = TaskLoss(y_hat, y) + β * InvarianceLoss
   - TaskLoss: cross-entropy for classification.
   - InvarianceLoss = mean over positive pairs (h and h') from same semantic interaction (e.g., data augmentations) of KL divergence between assignment distributions p(z|h) and p(z|h'). Augmentations: for video, temporal crops of 0.5-2s segments; for text, back-translation via paraphrase model. We validate on a held-out set of 100 augmented pairs that augmentations preserve interaction semantics (human annotators label interaction types, keep augmentations with >80% agreement).

Training: Update encoders, metric d_metric, prototype embeddings, experts, and classifier via Adam optimizer (lr=1e-4, weight decay=1e-5). The CRP prior is amortized via a neural network that predicts assignment probabilities (or direct sampling with Gumbel-Softmax). Train for 50 epochs with batch size 64.
```

### (C) Why this design
We chose a Dirichlet Process (DP) prior over a fixed number of experts because the number of interaction types is unknown and can grow with data; this avoids the need to pre-specify K as in I2MoE, but incurs the computational cost of maintaining an unbounded set of prototypes and resampling assignments during training (expected active prototypes ~ 5 for α=1.0 and 10k samples). We opted for a learned invariant metric over a handcrafted distance (e.g., Euclidean) because modality-specific transformations (e.g., image cropping, text paraphrasing) can distort direct distances; by enforcing consistency under augmentations via KL divergence, the metric learns to focus on semantic fusion patterns, though this requires constructing valid positive pairs that preserve the underlying interaction without altering the fusion semantics. We use prototype-specific expert networks rather than a single fusion network because each interaction type may require a different processing of concatenated features; the trade-off is increased parameter count and risk of overfitting if data is scarce. The CRP sampling is approximated with a Gumbel-Softmax relaxation to allow end-to-end gradient propagation, accepting that the relaxation may bias assignments toward uniform distributions if temperature is not tuned.

### (D) Why it measures what we claim
Specifically, the invariant metric d_metric measures semantic fusion patterns because assignments are consistent under augmentations that preserve the interaction type; failure mode: augmentations that change the interaction type (e.g., cropping the object that is the subject of sentiment) would make the invariance loss harmful. The Dirichlet process prior operationalizes 'open-ended discovery of interaction categories' because the stick-breaking construction ensures that the posterior probability of creating a new prototype scales with the concentration parameter α, guaranteeing that the model can instantiate novel prototypes when existing ones poorly explain the data; this assumption fails when α is set too low, in which case the model may underfit by merging distinct interaction types into existing prototypes, and the metric instead reflects aggregated, less discriminative patterns. The invariant metric, enforced via KL-divergence consistency under modality-specific augmentations, measures 'semantic fusion patterns' because the consistency loss forces the assignment distribution to be invariant to transformations that do not change the underlying interaction; this assumption fails when augmentations inadvertently alter the interaction semantics (e.g., cropping an object that is essential for interaction), in which case the metric reflects invariance to irrelevant noise rather than to modality-specific artifacts, and prototypes may become insensitive to meaningful variations.

## Contribution

(1) A novel Bayesian nonparametric mixture-of-experts framework, BNP-MoE, that automatically discovers interaction prototypes via a Dirichlet Process prior, eliminating the need for predefined interaction categories. (2) A joint invariant metric learning objective that ensures prototypes capture semantic fusion patterns by enforcing assignment consistency under modality-specific perturbations, leading to more interpretable and generalizable interaction discovery. (3) An end-to-end training procedure that integrates DP sampling via Gumbel-Softmax relaxation with contrastive invariance loss.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Multimodal sentiment analysis (CMU-MOSI) | Diverse interaction types, open-ended |
| Primary metric | Binary classification accuracy | Measures overall task performance |
| Baseline 1 | I2MoE (fixed K=5) | Tests open-ended discovery vs fixed |
| Baseline 2 | Vanilla MoE (K=8) | Tests prototype-specific experts |
| Baseline 3 | CoMM (contrastive alignment) | Tests invariance metric vs cross-modal |
| Ablation-of-ours | BNP-MoE w/o invariance loss | Isolates effect of invariance loss |

### Why this setup validates the claim
This combination creates a falsifiable test of BNP-MoE's central claim: that it automatically discovers the correct number of interaction categories and learns invariant fusion patterns. CMU-MOSI contains multiple, unknown interaction types (e.g., contrastive, causal), so a model with fixed K (I2MoE) must guess the number, risking under/overfitting. Vanilla MoE lacks prototype-specific experts, failing to capture diverse fusion functions. CoMM aligns modalities globally, ignoring interaction-specific invariants. The ablation removes the invariance loss to test if learned metric is crucial. Accuracy directly reflects whether the discovered prototypes help the task; if BNP-MoE outperforms baselines, it confirms the DP prior and invariant metric are beneficial.

### Expected outcome and causal chain

**vs. I2MoE** — On a sample where the interaction type is novel (e.g., a rare sarcasm pattern), I2MoE forces it into one of its fixed K=5 prototypes, causing misalignment and lower confidence. Our method creates a new prototype via the DP prior, capturing the unique pattern precisely. We expect a noticeable accuracy gap on test samples from rare interaction types (e.g., >5% absolute improvement), but parity on common types.

**vs. Vanilla MoE** — On a sample where the interaction requires non-linear fusion (e.g., contradictory sentiments across modalities), Vanilla MoE uses a single expert for all, blurring the distinct fusion needed. BNP-MoE assigns a prototype-specific expert that learns to combine contradictory signals. We expect a clear gap on samples with complex interactions (e.g., >3% absolute improvement), while simple additive interactions show similar performance.

**vs. CoMM** — On an augmented pair where cropping an image changes its pose but not the interaction (e.g., person pointing left vs right), CoMM's global alignment may treat this as a negative pair, hurting representation. BNP-MoE's invariance loss enforces consistent assignment across augmentations, ignoring pose changes. We expect BNP-MoE to maintain higher accuracy on augmented test sets (e.g., >2% absolute improvement), especially when augmentation alters only irrelevant details.

### What would falsify this idea
If BNP-MoE's accuracy gain over I2MoE is uniform across all test subsets (e.g., not concentrated on rare interaction types), then the DP prior is not actually discovering new categories—it may be overfitting or the fixed K was already sufficient. Similarly, if the ablation performs as well as the full model, the invariance loss is unnecessary, contradicting the claim that learned invariant metric is crucial.

## References

1. I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts
2. What to align in multimodal contrastive learning?
3. Intrinsic User-Centric Interpretability through Global Mixture of Experts
4. MoMa: Efficient Early-Fusion Pre-training with Mixture of Modality-Aware Experts
5. The future of human-centric eXplainable Artificial Intelligence (XAI) is not post-hoc explanations
6. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
7. Multimodal Learning Without Labeled Multimodal Data: Guarantees and Applications
8. Factorized Contrastive Learning: Going Beyond Multi-view Redundancy
