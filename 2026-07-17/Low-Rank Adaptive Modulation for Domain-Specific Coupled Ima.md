# Low-Rank Adaptive Modulation for Domain-Specific Coupled Image-Text Generation

## Motivation

The training-free coupling in SC-CMJP (Self-Correcting Coupled Markov Jump Processes) cannot adapt to domain-specific cross-modal patterns because the base unimodal MDMs remain frozen. Fine-tuning the full model is expensive and risks overfitting, especially with limited paired data. This prevents the framework from exploiting targeted cross-modal correlations that vary across domains.

## Key Insight

Constraining lightweight adapters to a subspace orthogonal to the base model parameters ensures that domain-specific cross-modal modulation does not interfere with unimodal generation quality, enabling adaptation with minimal parameter changes.

## Method

We propose LOAM (Low-rank Orthogonal Adapter for Modulation), a lightweight module that modifies confidence scores in SC-CMJP's transition rates. 

(B) How it works:
```pseudocode
# For each modality m, given cross-modal attention features h from other modality:
  base_conf = base_mdm.confidence(token_i, m)   # frozen
  δ = W_down @ (W_up @ h + b)                   # low-rank adapter, r=16, d=512, GeLU activation
  adapted_conf = base_conf + δ
  λ = exp(adapted_conf * cross_conf)            # transition rate in SC-CMJP

# Training: minimize variational lower bound on joint log-likelihood over domain-specific paired data.
# Additional regularization: KL divergence between base and adapted model outputs on a held-out unimodal set (no cross-modal features), λ_kl = 0.1.
# During training, project gradients of adapter parameters onto nullspace of base weight matrix W_base using QR decomposition:
  grad_W_adapter = proj_nullspace(grad_W_adapter, W_base)
```
(C) Why this design: We chose low-rank adapters over full fine-tuning because they introduce only r*(2d+1) parameters per modality (49408 parameters for d=512, r=16), allowing adaptation with fewer than 1000 samples without overfitting. We chose to modulate confidence scores rather than directly modify transition rates because confidence is the natural interface between unimodal and cross-modal components in SC-CMJP, enabling the adapter to leverage the existing self-correcting mechanism. We chose orthogonal subspace constraints over simple L2 regularization because orthogonality provably prevents the adapter from altering the base model's output on inputs outside the adapter's training distribution—when the adapter is zero (no cross-modal features), the model reverts exactly to the base model. The trade-off for this guarantee is that the adapter has limited expressivity: it cannot rotate the confidence representation in directions that lie within the base weight space, but we accept this cost because the domain-specific patterns are expected to be low-dimensional and orthogonal to the base model's learned manifold.

(D) Why it measures what we claim: The adapter offset δ measures domain-specific cross-modal pattern exploitation because it directly adjusts the confidence scores based on cross-modal attention features, and the variational bound training ensures that only patterns improving joint likelihood are learned; this equivalence relies on the assumption that the base model's confidences are calibrated for unimodal quality, which fails when the base model was not trained on any cross-modal data, in which case δ may compensate for base model deficiencies rather than learn domain patterns. Additionally, the orthogonality preservation relies on the assumption that the linear nullspace projection fully prevents interference in the final outputs, which fails when the base model uses non-linear activations (e.g., ReLU), allowing residual interference through non-linear propagation. To mitigate this, we add a KL divergence regularization loss that explicitly minimizes interference on unimodal held-out data.

## Contribution

(1) A novel low-rank adapter architecture for modulating cross-modal confidence scores in coupled Markov jump processes, enabling domain adaptation with minimal parameters. (2) A training procedure with orthogonal subspace constraints that preserve unimodal generation quality during adapter fine-tuning. (3) Demonstration that the approach can adapt joint image-text generation to a new domain using as few as 500 paired samples without degrading base model performance.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Medical (MIMIC-CXR) and Artistic (best-artworks) | Two diverse domains to test adaptation with limited samples (~1000 each) |
| Primary metric | FJD (Fréchet Joint Distance) | Joint distribution distance |
| Additional metric | KL divergence on unimodal held-out set (512 examples) | Quantifies interference on non-linear base models |
| Baseline 1 | Standard parallel MDM | No self-correction or coupling |
| Baseline 2 | SC-CMJP (base only) | No domain adaptation |
| Baseline 3 | SC-CMJP with full fine-tune | Parameter-heavy adaptation |
| Ablation-of-ours | LOAM without orthogonal constraint (but with KL regularization) | Tests necessity of orthogonal projection separately |
| Ablation-of-ours | LOAM without KL regularization (but with orthogonality) | Tests efficacy of KL regularization separately |

### Why this setup validates the claim

This setup combines two diverse domain-specific datasets (medical and artistic, each with ~1000 paired samples) with baselines that isolate key mechanisms: standard MDM tests the value of self-correction, base SC-CMJP tests domain adaptation, and full fine-tune tests parameter efficiency. FJD measures joint distribution fidelity, directly reflecting cross-modal consistency. The additional KL divergence metric on unimodal held-out data quantifies interference on non-linear base models, testing the load-bearing assumption. The two ablations (no orthogonality, no KL regularization) individually assess the contributions of each constraint. If LOAM outperforms base SC-CMJP on domain-specific examples while matching it on general data, and full fine-tune degrades on general data due to overfitting, the claim that low-rank orthogonal adaptation efficiently captures domain patterns is supported. The ablation without orthogonality showing worse generalization would confirm orthogonality's role, and the ablation without KL regularization showing increased KL divergence would confirm the necessity of explicit regularization for non-linear cases.

### Expected outcome and causal chain

**vs. Standard parallel MDM** — On a case where text and image are semantically mismatched (e.g., "a cat sitting on a mat" but image shows no mat), the baseline generates incoherent results because it lacks cross-modal coupling during the reverse process. Our method instead uses self-correcting transitions modulated by the adapter to iteratively align modalities, so we expect a noticeable gap in FJD on mismatched examples (e.g., 15% lower FJD) while parity on well-matched ones.

**vs. SC-CMJP (base only)** — On a case where domain-specific patterns exist (e.g., medical X-ray and report), the base model trained on general data fails to capture specific correlations (e.g., mentions of "infiltrate" with specific opacity). Our adapter learns these from few (<1000) examples by modulating confidence, leading to higher joint likelihood. We expect a clear FJD improvement on domain-specific test subsets (e.g., 20% lower) but no significant difference on general subsets.

**vs. SC-CMJP with full fine-tune** — On the same limited-data domain, full fine-tune overfits to the 1000 examples, degrading performance on out-of-domain inputs (e.g., general descriptions) because all parameters shift. Our method with orthogonal constraint restricts changes to a low-rank subspace orthogonal to base weights, preserving base model behavior on unseen inputs. We expect the full fine-tune to have lower FJD on the domain-specific test set (due to more parameters) but higher FJD on a general test set (e.g., 30% worse), while our method maintains near-base performance on general data. Additionally, our method should achieve lower KL divergence on unimodal held-out data (e.g., <0.05) compared to full fine-tune (e.g., >0.5).

### What would falsify this idea
If LOAM's improvement over base SC-CMJP is uniform across all test examples (domain-specific and general) rather than concentrated on domain-specific subsets, or if the ablation without orthogonality matches LOAM's performance, or if the KL divergence on unimodal data is not significantly lower than full fine-tune, then the central claim that low-rank orthogonal adaptation efficiently captures domain patterns without harming generalization is false.

## References

1. Concurrent Image Understanding and Generation: Self-Correcting Coupled Markov Jump Processes
2. SDAR: A Synergistic Diffusion-AutoRegression Paradigm for Scalable Sequence Generation
3. MMaDA-Parallel: Multimodal Large Diffusion Language Models for Thinking-Aware Editing and Generation
4. Don't Settle Too Early: Self-Reflective Remasking for Diffusion Language Models
5. Scaling Diffusion Language Models via Adaptation from Autoregressive Models
6. Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning
7. Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data
8. Likelihood-Based Diffusion Language Models
