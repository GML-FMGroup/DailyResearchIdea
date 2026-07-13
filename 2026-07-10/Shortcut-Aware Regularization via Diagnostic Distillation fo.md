# Shortcut-Aware Regularization via Diagnostic Distillation for Robust Video Understanding

## Motivation

Video-Oasis reveals that state-of-the-art Video-LLMs rely on shortcuts (e.g., text-only cues, static frame biases) in existing benchmarks, and after filtering such shortcuts, their performance drops to near random. However, Video-Oasis only diagnoses the problem without proposing any training intervention. The root cause is that current training objectives inadvertently reward shortcut exploitation because benchmarks are contaminated with non-video-native samples. Therefore, we need a training regularization that actively penalizes reliance on identified shortcuts, bridging the gap between diagnosis and mitigation.

## Key Insight

The diagnostic shortcut scores from Video-Oasis provide a direct, per-sample measure of shortcut reliance, which can be used as a supervisory signal to regularize the model's confidence toward uniform distribution on shortcut-prone samples, forcing it to learn true video-temporal reasoning.

## Method

**Shortcut-Aware Regularization via Diagnostic Distillation (SAR-DD)**

(A) **What it is:** SAR-DD is a training-time regularization method that uses diagnostic shortcut scores (from Video-Oasis) to discourage a Video-LLM from relying on shortcuts. Its inputs are the model, training samples, and precomputed shortcut scores; output is a regularized loss that updates the model parameters.

(B) **How it works:**
```pseudocode
# Hyperparameters: lambda_reg=0.1 (regularization strength), T=2 (temperature for distillation)
# Load-bearing assumption: Shortcut scores from Video-Oasis reflect current model's shortcut reliance.
# Scores are recomputed every K=5 epochs using the current model checkpoint to adapt to shifts.
# Precomputation: For each training sample, s(x) is computed once per refresh and cached.
for epoch in 1..total_epochs:
    if epoch % 5 == 1:
        recompute_shortcut_scores(model, train_dataset)  # updates cache
    for each batch (x, y) in training set:
        # Step 1: Compute model logits f(x) and soft targets via temperature scaling
        p_model = softmax(f(x) / T)
        
        # Step 2: Retrieve shortcut score s(x) from cache (0 = no shortcut, 1 = pure shortcut)
        # s(x) is computed using Video-Oasis: average of text-only baseline accuracy and temporal shuffle drop
        
        # Step 3: Define target distribution q_target: uniform over classes if s(x) > threshold tau=0.5, else one-hot on true label
        if s(x) > tau:
            q_target = uniform_vector(num_classes)
        else:
            q_target = one_hot(y)
        
        # Step 4: Compute regularization loss as KL divergence between p_model and q_target
        L_reg = KL(p_model || q_target)
        
        # Step 5: Combine with standard cross-entropy L_ce
        L_total = L_ce + lambda_reg * L_reg
        
        # Step 6: Backpropagate and update
        optimizer.step(L_total)
```

(C) **Why this design:** We chose to use a hard threshold (τ=0.5) on shortcut scores rather than a continuous weighting because it creates a clear separation between shortcut-dominated and clean samples, avoiding ambiguous regularization that could confuse the model; the cost is that some borderline samples may be misclassified, but the diagnostic scores of Video-Oasis are already binarizable based on sufficiently large drop. We used KL divergence instead of L2 penalty because KL directly targets the model's predicted distribution, encouraging uniform confidence rather than just penalizing high logits, which is more aligned with reducing shortcut reliance. We set temperature T=2 for soft targets to avoid overconfident regularization, trading off sharper gradients for smoother learning dynamics. The shortcut score s(x) is computed externally via Video-Oasis rather than learned, because the diagnostic suite is already validated and avoids the computational overhead of an auxiliary shortcut detector, though it requires a separate inference pass on each training sample before training. To reduce overhead, scores are cached per sample and recomputed only every K=5 epochs.

(D) **Why it measures what we claim:** The quantity s(x) (shortcut score) measures “shortcut susceptibility” because the Video-Oasis diagnostic defines it as the performance drop when removing visual/temporal information; this drop reflects how much the model relies on non-native cues. The assumption is that a high drop indicates the sample is solved via shortcuts; this assumption fails when a sample genuinely requires both text and video but the text-only baseline still performs well (e.g., a question explicitly answered in subtitles), in which case s(x) overestimates shortcut use, and regularization might incorrectly weaken performance on such samples. The regularization term KL(p_model || q_target) measures “discouragement of shortcut confidence” because it penalizes divergence from a uniform distribution, which is the maximum-entropy state that avoids favoring any class; this assumes that uniform confidence is the correct response when shortcuts dominate, but in reality the model should still have some structure (e.g., random but not entirely uniform), so the regularization may cause slight underfitting on high-shortcut samples. The threshold τ operationalizes “shortcut severity” because it binarizes the continuous score; this assumes that samples above τ are reliably shortcut-dominated, but if τ is set too low, clean samples are penalized, and if too high, shortcut reliance is not curbed—we chose τ=0.5 based on the Video-Oasis finding that many benchmarks have >50% shortcut samples, but this may vary across datasets.

## Contribution

(1) SAR-DD, a training-time regularization framework that leverages per-sample diagnostic shortcut scores to explicitly penalize shortcut reliance in Video-LLMs. (2) A design principle: shortcut-aware regularization via distributional softening (uniform target) is more effective than confidence penalty or adversarial training for improving robustness on video-native challenges. (3) Demonstration that integrating diagnosis and mitigation closes the loop between benchmark auditing and model improvement.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Datasets | Video-Oasis video-native challenges, Charades, Something-Something | Focuses on genuine video understanding across diverse benchmarks. |
| Primary metric | Accuracy on video-native challenges | Directly measures video-native reasoning. |
| Baseline 1 | Text-only baseline | Isolates shortcut from textual priors. |
| Baseline 2 | Random guessing | Lower bound for non-shortcut performance. |
| Baseline 3 | Pretrained Video-LLM (e.g., Video-LLaVA) | State-of-the-art without our regularization. |
| Baseline 4 | LfF (Learning from Failure) | Standard debiasing method; differs from SAR-DD by using bias-amplified model, not diagnostic scores. |
| Baseline 5 | JTT (Just Train Twice) | Error-driven reweighting; differs from SAR-DD by requiring two training phases and using upweighting instead of uniform regularization. |
| Ablation | Standard cross-entropy fine-tuning (no SAR-DD) | Isolates effect of regularization. |

### Why this setup validates the claim
This setup validates the claim because the Video-Oasis video-native challenges are specifically designed to require genuine video understanding, filtering out samples solvable via textual or static shortcuts. Comparing against the text-only baseline and random guessing establishes whether any improvement stems from reduced shortcut reliance rather than general gains. The primary metric directly captures performance on tasks where shortcuts are minimized. The pretrained Video-LLM baseline shows current state-of-the-art, while the ablation (standard fine-tuning) isolates the contribution of the shortcut-aware regularization. If SAR-DD improves accuracy on these challenges beyond standard fine-tuning, it confirms that the regularization effectively discourages shortcut reliance. Conversely, if gains are absent or uniform across all samples, the claim that SAR-DD targets shortcuts would be falsified. Additionally, we will validate on a held-out set of manually verified video-native samples that lower confidence predictions (approaching uniform) correlate with higher accuracy, confirming the appropriateness of the uniform target. Comparison to LfF and JTT highlights uniqueness: SAR-DD leverages explicit diagnostic scores rather than learned biases or error patterns, enabling direct regularization on shortcut-prone samples.

### Expected outcome and causal chain

**vs. Text-only baseline** — On a sample where the answer is implicitly in subtitles (e.g., a question about dialogue), the text-only baseline answers correctly because it exploits the text shortcut. Our method assigns a high shortcut score (s > 0.5) due to text-only performance, regularizing towards uniform distribution, so the model avoids leveraging subtitles and may guess wrong. Consequently, we expect our accuracy to be lower on such samples but substantially higher on video-native samples where text is uninformative. The overall gap should show our method outperforming text-only on video-native challenges.

**vs. Random guessing** — Random guessing yields chance accuracy (e.g., 1/num_classes) on all samples. Our method learns genuine video features by ignoring shortcuts, so on video-native challenges it should significantly exceed random. The magnitude of improvement is expected to be large (e.g., 20-30% absolute gain), as the model must discover true spatiotemporal patterns.

**vs. Pretrained Video-LLM** — On a sample requiring temporal reasoning (e.g., determining if a person picks up an object before leaving), a pretrained model might rely on static frame content or text, ignoring order. Our method incorporates temporal shuffle drop into s(x); if shuffle drops performance, the sample is deemed shortcut-prone and regularized to uniform, forcing the model to attend to temporal cues. We expect our method to outperform the pretrained baseline specifically on temporally challenging subsets, with a noticeable gap (e.g., 5-10%) while matching on static-reliant samples.

**vs. LfF** — LfF trains a biased model and then upweights samples where the biased model makes errors. SAR-DD differs by using diagnostic scores that directly measure shortcut reliance rather than relying on a biased model's errors, which may miss shortcuts that benefit all models. We expect SAR-DD to achieve higher accuracy on video-native challenges because it targets shortcut-prone samples more precisely.

**vs. JTT** — JTT identifies error-prone samples via an initial model and upweights them. SAR-DD instead regularizes toward uniform confidence on samples identified as shortcut-dominated via diagnostic scores, which may be more effective than upweighting (which still allows high confidence on wrong predictions). We expect SAR-DD to outperform JTT on video-native challenges.

### What would falsify this idea
If our method’s accuracy gain is uniform across all samples (both high and low shortcut scores) rather than concentrated on samples with s > 0.5, then the regularization is not specifically targeting shortcuts, and the claim of shortcut-aware improvement is falsified. Similarly, if our method fails to outperform standard cross-entropy fine-tuning on the video-native challenges, the central premise is invalidated.

## References

1. Video-Oasis: Rethinking Evaluation of Video Understanding
