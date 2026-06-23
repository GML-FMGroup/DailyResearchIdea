# Adaptive Edit Modulation for Inversion-Free Audio Editing Across Large Semantic Gaps

## Motivation

Existing inversion-free audio editing methods such as DirectAudioEdit rely on contrastive prediction between source and target text embeddings to steer the diffusion process. However, this contrastive steering fails for large semantic edits (e.g., adding instruments, genre change) because the source-target embedding distance becomes too large, causing the steering signal to be diluted or misdirected. The field's trajectory has not addressed how to handle such large discrepancies without resorting to inversion or multi-stage optimization, creating a structural bottleneck for training-free editing.

## Key Insight

The discrepancy between source and target embeddings provides a natural signal to adaptively scale the diffusion step size and contrastive weight per layer, enabling the model to traverse dissimilar acoustic regions by stretching the effective edit magnitude where needed.

## Method

### Adaptive Edit Modulation for Inversion-Free Audio Editing Across Large Semantic Gaps

[Frozen — do not change]
Title: Adaptive Edit Modulation for Inversion-Free Audio Editing Across Large Semantic Gaps
Motivation: Existing inversion-free audio editing methods such as DirectAudioEdit rely on contrastive prediction between source and target text embeddings to steer the diffusion process. However, this contrastive steering fails for large semantic edits (e.g., adding instruments, genre change) because the source-target embedding distance becomes too large, causing the steering signal to be diluted or misdirected. The field's trajectory has not addressed how to handle such large discrepancies without resorting to inversion or multi-stage optimization, creating a structural bottleneck for training-free editing.
Key insight: The discrepancy between source and target embeddings provides a natural signal to adaptively scale the diffusion step size and contrastive weight per layer, enabling the model to traverse dissimilar acoustic regions by stretching the effective edit magnitude where needed.

[Current method]
(A) **What it is**: Adaptive Edit Modulation (AEM) is a training-free extension to DirectAudioEdit that dynamically adjusts the denoising step size and contrastive steering weight based on the cosine distance between source and target CLAP embeddings, but with a calibrated threshold to ensure reliability. Inputs: source audio, source text, target text. Output: edited audio preserving source structure while achieving target semantics.

(B) **How it works**:
```python
# Pre-trained components: diffusion model (e.g., AudioLDM), CLAP encoder
# Hyperparameters: alpha=0.5, beta=2.0, T=1000
# Calibration: threshold tau=0.3 learned from a calibration set of 512 diverse edit pairs from AudioCaps training set,
# such that only pairs with cosine distance >= tau are considered for modulation.

# Phase 1: Compute calibrated discrepancy
z_T = noise latent from source audio (via forward diffusion to step T)
cos_sim = cosine_similarity(E_source, E_target)  # in [-1,1]
d = max(0, cos_sim - tau) / (1 - tau)  # calibrated distance in [0,1], conditionally active only when cos_sim >= tau
mod_factor = 1 + alpha * tanh(beta * d)   # range [1, 1+alpha]

# Phase 2: Adaptive denoising loop
for t in reversed(range(T)):
    # Contrastive predictions (as in DirectAudioEdit)
    eps_source = model(z_t, t, source_text)
    eps_target = model(z_t, t, target_text)
    w = 0.5 * (1 + d)  # contrastive weight, linearly increases with d (range [0.5, 1])
    eps_edit = (1 - w) * eps_source + w * eps_target
    
    # Adaptive step size scaling
    step_size = (alpha_t - alpha_{t-1}) * mod_factor  # alpha_t is noise schedule
    z_{t-1} = z_t - step_size * eps_edit + sigma_t * noise

# Result: z_0 is the edited audio latent, decode with VAE
```

(C) **Why this design**: We chose a multiplicative modulation factor (1 + α·tanh(β·d)) instead of a linear scaling because tanh saturates for large d, preventing instability from overly aggressive steps; this accepts that very large discrepancies may not benefit from unbounded growth. We set the contrastive weight w = 0.5·(1+d) to smoothly shift reliance from source to target as discrepancy grows, rather than a hard threshold, because a hard cutoff would create discontinuities in the editing trajectory. We applied modulation to the step size (the effective per-step update) rather than to the noise schedule or model weights, because step size directly controls the extent of change and is easily invertible without additional training. The cost is that step size scaling may accumulate error over many steps, but we mitigate via the saturation in tanh and keep α modest (0.5). The calibration threshold τ=0.3 is set using a small validation set to ensure modulation only activates when the embedding distance reliably indicates a large semantic gap, avoiding over-editing on ambiguous pairs.

(D) **Why it measures what we claim**: The modulation factor (mod_factor) operationalizes **adaptive edit magnitude** because it expands the effective denoising step size proportionally to the calibrated discrepancy d, under the assumption that calibrated cosine distance is a monotonic proxy for the required semantic edit strength; this assumption fails when source and target are dissimilar in content but share similar CLAP embeddings (e.g., two different genres with similar mood), in which case mod_factor underestimates the needed step. The contrastive weight w operationalizes **source-target balance** because it increases target influence with d, assuming that the contrastive prediction (eps_edit) remains directionally correct even for large d; this assumption fails when the target prediction becomes dominated by prior bias rather than the edit direction, leading to content drift. The calibration threshold ensures that we only trust the distance measure when it exceeds a conservative value, reducing the risk of false positives from assumption violations. The combination of both ensures that for large d the model takes larger steps under stronger target guidance, directly addressing the structure of the motivation: contrastive steering alone is insufficient for large gaps, but adaptive scaling can compensate by making the edit more aggressive.

## Contribution

['(1) A novel training-free adaptive modulation mechanism that adjusts denoising step size and contrastive weight based on source-target embedding discrepancy, enabling large semantic edits (e.g., adding instruments, genre change) without inversion or fine-tuning.', '(2) An empirical principle: the embedding distance between source and target can serve as a reliable signal for controlling edit aggressiveness, challenging the assumption that contrastive steering suffices uniformly.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | AudioCaps & MusicCaps, partitioned into low-d (cosine distance < 0.3) and high-d (distance ≥ 0.3) subsets | Covers music and events; high-d tests large-gap edits. |
| Primary metric | CLAP Score | Measures semantic alignment with target. |
| Secondary metric | STFT Distance (L1 on log-mel spectrograms) | Measures content preservation vs. structural loss. |
| Baseline 1 | DirectAudioEdit | Contrastive steering baseline. |
| Baseline 2 | DDPM Inversion | Classic inversion-based editing. |
| Baseline 3 | AudioMorphix | Exemplar-based editing baseline. |
| Ablation | AEM w/o adaptive step | Tests step size modulation. |

### Why this setup validates the claim

This combination forms a falsifiable test because it isolates our central claim—that adaptive step size and contrastive weight enable larger edits without structure loss. Using both music and event sounds ensures the discrepancy range (d) is varied; CLAP Score directly measures if the edit achieves target semantics, while STFT Distance gauges whether the adaptive step size causes structural degradation. DirectAudioEdit tests whether our adaptive scaling improves upon vanilla contrastive steering, especially for high discrepancy. DDPM Inversion tests whether we outperform expensive inversion while being fully training-free. AudioMorphix tests against an exemplar-based approach that may preserve structure but lacks text control. The ablation isolates the step size modulation component; if it matches our full method, then the weight adaptation alone is responsible for gains. Partitioning into low-d and high-d subsets forces a specific predicted pattern: performance gains should concentrate on high-d samples where our method’s adaptive mechanisms are expected to help most.

### Expected outcome and causal chain

**vs. DirectAudioEdit** — On a case where the calibrated distance d is high (e.g., turning a symphony into a jazz beat), DirectAudioEdit (fixed w=0.5, fixed step size) produces distorted or content-drifting audio because its uniform contrastive weight cannot shift enough toward target, and step size is too small to achieve the large semantic gap. Our method instead increases both w and step size adaptively via d: w rises toward 1, promoting target guidance, and step size grows up to 1.5× nominal, allowing bolder moves. We expect a noticeable gap in CLAP Score on high-d samples (d>0.3) but near parity on low-d samples, confirming the adaptive mechanisms target the core failure mode. STFT Distance should remain comparable, indicating no extra structure loss.

**vs. DDPM Inversion** — On a case requiring subtle structural preservation (e.g., changing only the instrument while retaining melody), DDPM Inversion often produces artifacts or global timbre shift because inversion errors accumulate and the edit is applied in latent space without contrastive cues. Our method, being inversion-free and contrastive, maintains source structure better; the adaptive step size ensures that for small d, step size stays near 1, preventing over-editing. We expect our method to achieve higher CLAP Score while also achieving lower STFT Distance, indicating better structural retention despite the semantic change.

**vs. AudioMorphix** — On a case requiring text-guided semantic change with no reference audio (e.g., changing a recording from "rain" to "thunder"), AudioMorphix fails because it requires an exemplar reference, or if no exemplar is provided, it defaults to source preservation. Our method directly uses target text via CLAP, so it can perform novel edits. We expect our method to dominate on purely text-guided edits, as measured by CLAP Score, while AudioMorphix may occasionally outperform on exemplar-based tasks not tested. STFT Distance may favor AudioMorphix when a good exemplar is available, but our method should show competitive preservation.

### What would falsify this idea

If our method’s CLAP Score improvements over DirectAudioEdit are uniformly distributed across discrepancy levels (rather than concentrated on high-d samples), or if the ablation shows that adaptive step size provides no benefit, then the central claim that adaptive magnitude modulation is crucial for large-gap edits would be falsified.

## References

1. DirectAudioEdit: Inversion-Free Text-Guided Audio Editing via Diffusion Prediction Contrast
2. Guiding Audio Editing with Audio Language Model
3. AudioMorphix: Training-free audio editing with diffusion probabilistic models
4. SteerMusic: Enhanced Musical Consistency for Zero-shot Text-Guided and Personalized Music Editing
5. Flowedit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models
6. FlowAlign: Trajectory-Regularized, Inversion-Free Flow-based Image Editing
7. InstructME: An Instruction Guided Music Edit Framework with Latent Diffusion Models
8. Zero-Shot Unsupervised and Text-Based Audio Editing Using DDPM Inversion
