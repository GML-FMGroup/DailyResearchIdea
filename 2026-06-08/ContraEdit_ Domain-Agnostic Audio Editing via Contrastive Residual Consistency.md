# ContraEdit: Domain-Agnostic Audio Editing via Contrastive Residual Consistency

## Motivation

Existing audio editing models, including DirectAudioEdit, rely on pre-trained backbones and benchmarks that assume a static input distribution, causing systematic failures on out-of-distribution inputs such as novel recording conditions or background noise. This structural limitation arises because these models lack mechanisms to ensure the editing operation (the residual between source and edited audio) remains unaffected by nuisance factors. Contrastive learning can enforce invariance to such factors by pulling residuals from augmented versions of the same edit together, but prior work does not apply this to the edit residual itself.

## Key Insight

The editing residual should be invariant to distribution shifts in the source audio because it captures only the task-relevant change; contrastive learning on residuals across diverse augmentations forces the model to separate domain-invariant edit content from domain-specific audio characteristics.

## Method

### (A) What it is
ContraEdit is a fine-tuning method for pre-trained text-guided latent diffusion models (e.g., DirectAudioEdit's backbone) that adds a contrastive loss on the editing residual. Input: source audio, target text description, and a set of data augmentations. Output: an edited audio whose residual is consistent across distribution shifts. **Load-bearing assumption: The editing residual is invariant under distribution shifts in the source audio, i.e., the same edit yields the same residual regardless of domain-specific augmentations.**

### (B) How it works (pseudocode)
```python
# Precondition: Pre-trained diffusion model M that edits source audio s using target text t to produce ē = M(s, t)
# Augmentation set A = [add_noise (SNR=10dB), reverb (RT60=0.5s), bandpass (300-3400Hz), speed_change (0.9x, 1.1x), background_mix (SNR=5dB)] with random sampling
# Each augmentation a transforms source s to s_a

for each training step:
  1. Sample source s, target text t, and a random edit (e.g., add bird chirp) to create ground-truth edited audio e.
  2. Sample K=4 augmentations from A, producing s_1,...,s_K.
  3. For each augmented source s_i, compute edited audio ē_i = M(s_i, t) (with gradient flow through M).
  4. Compute residual r_i = ē_i - s_i.
  5. Compute contrastive loss on residuals:
       L_contra = - log( exp(sim(r_0, r_1)/τ) / (sum_{i=0}^{K} sum_{j≠i} exp(sim(r_i, r_j)/τ) ) )
     where r_0 is residual from non-augmented source (or use one augmented as anchor). τ = 0.1, sim is cosine similarity.
  6. Total loss: L = L_diffusion + λ * L_contra, with λ = 0.5.
  7. Fine-tune M via gradient descent (Adam, lr=1e-5, batch_size=8, 10 epochs on 4 GPUs).
```

### (C) Why this design
We chose to apply contrastive loss on the residual rather than on the edited audio directly because the residual is a more compact representation of the edit; direct contrast on outputs could encourage the model to ignore the edit content. We selected a set of diverse augmentations (e.g., additive noise, reverb, bandpass) to cover common distribution shifts (recording quality, environmental acoustics) without requiring task-specific knowledge. Using a single residual anchor with in-batch negatives is computationally efficient but assumes that residuals from different edits are naturally separated. We accepted the trade-off that this might not enforce invariance across entirely different edit types, but in-domain edits dominate. The hyperparameters λ=0.5 and τ=0.1 were chosen based on prior contrastive learning works; tuning on a held-out set can refine them.

### (D) Why it measures what we claim
The contrastive loss L_contra measures domain invariance of the edit residual because it pulls residuals from the same edit across augmentations together — the assumption is that the residual should be identical if the edit is domain-agnostic. This assumption fails when the augmentations do not capture the true distribution shift (e.g., a completely unseen microphone type), in which case L_contra may still be low but the residual varies in deployment. To verify this assumption, after fine-tuning we compute the standard deviation of residuals across a held-out calibration set of 200 samples with unseen distortions (e.g., microphone IRs from a different dataset). If the standard deviation exceeds 0.05 in amplitude, we flag the model as potentially violating domain invariance and may augment A further. The diffusion loss L_diffusion ensures that the edited audio remains high-quality and faithful to the target text; together, L_contra operationalizes the motivation-level concept of "domain-agnostic editing" by enforcing that the residual component is invariant under nuisance transformations.

## Contribution

(1) A contrastive learning framework for audio editing that enforces consistency of the edit residual across distribution shifts, enabling robust out-of-distribution editing. (2) Demonstrates that augmenting source audio and contrasting residuals improves edit fidelity on novel inputs compared to fine-tuning without contrastive loss. (3) Provides a set of audio augmentations tailored for distribution shift simulation in editing tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | MMAE (2,000 samples) | Diverse edits across domains. We reserve 200 samples for calibration (unseen distortions) and use the rest for evaluation. |
| Primary metric | FAD on augmented test set (additive noise, reverb, bandpass, speed change, background mix) | Measures quality and robustness; we report mean and std over 3 runs. |
| Baseline 1 | DirectAudioEdit (no contrastive) | Ablates contrastive loss effect. |
| Baseline 2 | DDPM inversion-based editing | Training-free standard baseline. |
| Baseline 3 | Prior contrastive output editing (contrastive loss on ē directly) | Tests residual vs output contrast. |
| Ablation-of-ours | ContraEdit without augmentation (K=0, only L_diffusion) | Tests need for augmentation diversity. |

### Why this setup validates the claim
This setup forms a falsifiable test of the central claim that contrastive loss on residuals enforces domain-agnostic editing. The MMAE dataset provides diverse edits and domains, ensuring the test covers real-world variability. FAD on an augmented test set (with unseen distortions not in the training augmentations) simultaneously assesses audio quality and invariance: if residuals are truly domain-agnostic, FAD should remain low despite augmentations. DirectAudioEdit baseline isolates the contrastive loss effect. DDPM inversion tests whether the method outperforms training-free approaches on domain shifts. Prior output contrast ablation tests the design choice of residual-level contrast. The augmentation ablation tests whether diversity of augmentations is critical. If our method surpasses all baselines on augmented FAD, the claim is supported; if not, the logic fails.

### Expected outcome and causal chain
**vs. DirectAudioEdit (no contrastive)** — On a case where the source audio has background noise (unseen during training), DirectAudioEdit produces an edit that is corrupted by the noise because its diffusion loss does not enforce invariance, so the residual changes with the noise. Our method instead constrains residuals to be similar across noise augmentations via contrastive loss, so it extracts the edit cleanly. We expect a noticeable FAD gap (e.g., 0.3 lower) on noisy test samples but parity on clean samples.

**vs. DDPM inversion-based editing** — On a case requiring precise edit localization (e.g., adding a bird chirp at a specific time), DDPM inversion struggles because its inversion process loses fine details and distorts the source, producing artifacts. Our method fine-tunes the diffusion model directly on the task, preserving source integrity via the diffusion loss while ensuring edit consistency via contrastive loss. We expect our method to achieve significantly higher FAD (e.g., 0.5 lower) on localization-heavy edits.

**vs. Prior contrastive output editing** — On a case where the edit is subtle (e.g., slightly increasing reverb), contrasting directly on the edited audio encourages the model to ignore the edit and output similar audio regardless, leading to under-editing. Our method contrasts residuals, which are zero for no edit and nonzero for edits, so it preserves edit strength while enforcing invariance. We expect our method to show higher edit fidelity (e.g., higher CLAP score) on subtle edits with similar FAD.

### What would falsify this idea
If the improvement from our method is uniform across all test subsets (e.g., clean and noisy both improve equally) rather than concentrated on cases with distribution shifts, then the central claim of domain-agnostic editing via residual contrast is unsupported; the gain may come from unrelated fine-tuning effects. Additionally, if the calibration set (unseen distortions) shows residual std > 0.05, the invariance assumption is violated.

## References

1. DirectAudioEdit: Inversion-Free Text-Guided Audio Editing via Diffusion Prediction Contrast
2. MMAE: A Massive Multitask Audio Editing Benchmark
3. Guiding Audio Editing with Audio Language Model
4. Qwen3-Omni Technical Report
5. Recomposer: Event-roll-guided generative audio editing
6. Sketch2Sound: Controllable Audio Generation via Time-Varying Signals and Sonic Imitations
7. Music ControlNet: Multiple Time-Varying Controls for Music Generation
8. PicoAudio: Enabling Precise Timestamp and Frequency Controllability of Audio Events in Text-to-audio Generation
