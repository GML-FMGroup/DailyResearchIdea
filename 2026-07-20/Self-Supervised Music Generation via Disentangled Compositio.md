# Self-Supervised Music Generation via Disentangled Compositional Latent Spaces and Cycle-Consistent Text Alignment

## Motivation

Existing text-to-music generation models, such as Qwen-Music, rely on large-scale paired text-music datasets, which are scarce and costly to collect. This reliance stems from the assumption that text and music must be directly aligned in a joint embedding space requiring extensive paired supervision. We identify that this assumption is shared across multiple works (e.g., Qwen-Music and MusicLM), creating a convergent gap that can be broken by leveraging self-supervised disentanglement of musical components (melody, harmony, rhythm) and cross-modal cycle-consistency.

## Key Insight

Musical components (melody, harmony, rhythm) can be disentangled in an unsupervised latent space because they have distinct temporal and spectral signatures that are statistically independent given the music signal, enabling alignment with text via a cycle-consistency loss without requiring large paired datasets.

## Method

**DCM-GEN: a framework that trains a music generation model by first learning a compositional latent representation of music (factored into melody, harmony, rhythm codes) using only unpaired music data, then aligning text descriptions to this compositional space via a cycle-consistency loss that leverages a handful of paired examples (e.g., 100) as anchors.**

**Pseudocode:**
```python
# Stage 1: Unsupervised Disentangled VAE for Music
encoder = ResNet-18-based encoder()  # outputs 512-dim hidden h
melody_head = Linear(512 -> 128) + GeLU
harmony_head = Linear(512 -> 128) + GeLU
rhythm_head = Linear(512 -> 128) + GeLU
decoders = {'melody': 4-layer ConvTranspose1d, 'harmony': 4-layer ConvTranspose1d, 'rhythm': 4-layer ConvTranspose1d}  # separate decoders with kernel sizes 3, 5, 7 respectively

for music_sample in unlabeled_music:  # batch size 32, sample rate 22050 Hz, 4-second clips
  h = encoder(music_sample)
  z_m, z_h, z_r = melody_head(h), harmony_head(h), rhythm_head(h)
  # Reconstruction with structured decoders
  recon_melody = decoders['melody'](z_m)
  recon_harmony = decoders['harmony'](z_h)
  recon_rhythm = decoders['rhythm'](z_r)
  # Aggregate by summing (or weighted sum) the three reconstructions
  recon = recon_melody + recon_harmony + recon_rhythm
  loss_recon = L1(music_sample, recon)
  # Mutual information minimization via CLUB estimator with gamma=0.1
  # CLUB uses a discriminator that predicts each latent from the others
  loss_mi = CLUB(z_m, z_h) + CLUB(z_m, z_r) + CLUB(z_h, z_r)
  total_loss = loss_recon + 0.1 * loss_mi
  update parameters (Adam, lr=1e-4)

# Stage 2: Cycle-Consistent Text Alignment
text_encoder = BERT-base (frozen) -> 768-dim embedding e
projector = Linear(768 -> 384) + LayerNorm
paired_set = [(text_i, music_i)]  # 100 pairs from MusicCaps
simple_decoder = Linear(384 -> 512) + ConvTranspose1d(512 -> 1)  # lightweight decoder to generate waveform

for epoch in range(50):
  for text, music in paired_set:
    # Forward through music VAE to get z_m, z_h, z_r
    h = encoder(music)
    z_m, z_h, z_r = melody_head(h), harmony_head(h), rhythm_head(h)
    z_music = concat([z_m, z_h, z_r])  # 384-dim
    # Project text to same space
    z_text = projector(text_encoder(text))  # 384-dim
    # Cycle-consistency: generate music from text, then text from generated music
    gen_music = simple_decoder(z_text)  # waveform
    # Re-encode generated music (detach gradient from gen_music? Yes, to avoid trivial solutions)
    with torch.no_grad():
      h_gen = encoder(gen_music)
      z_m_gen, z_h_gen, z_r_gen = melody_head(h_gen), harmony_head(h_gen), rhythm_head(h_gen)
      z_music_gen = concat([z_m_gen, z_h_gen, z_r_gen])
    # Cycle loss: text -> music -> text; we reconstruct text embedding
    # Use a simple text decoder (Linear(384 -> 768) + cosine loss)
    z_text_cycle = text_decoder(z_music_gen)  # text_decoder = Linear(384 -> 768)
    loss_cycle = cosine_distance(z_text, z_text_cycle) + cosine_distance(z_music, z_music_gen)
    update projector & simple_decoder & text_decoder (Adam, lr=1e-4)

Inference: input text -> z_text = projector(BERT(text)) -> generate music via simple_decoder(z_text) or auto-regressive decoder conditioned on z_music from VAE.
```

**Why this design:** We chose a structured VAE with separate heads for melody, harmony, and rhythm over a monolithic latent code because musical components exhibit distinct temporal and spectral structures (e.g., melody evolves over notes, harmony over chords, rhythm over beats); separating them allows each decoder to specialize, improving reconstruction and downstream alignment. This design incurs the cost of designing three separate decoder architectures, which increases engineering complexity but is justified by the improved disentanglement. We used mutual information minimization (via CLUB) rather than β-VAE alone because β-VAE can lead to posterior collapse; CLUB directly penalizes mutual information between latent variables, providing a stronger regularization for disentanglement. We prioritized cycle-consistency loss over direct contrastive alignment (e.g., CLIP-style) because cycle-consistency enforces a bijective mapping between text and music spaces, which is structurally necessary to ensure that text can faithfully reconstruct music and vice versa; contrastive losses only enforce relative similarity and may not capture fine-grained compositionality. We anchored the alignment with 100 paired examples instead of zero because a small anchor set prevents the cycle-consistency loss from collapsing to trivial solutions (e.g., mapping all texts to the same music) during early training; the cost is still far below the typical requirement of thousands of pairs. **Load-bearing assumption:** Melody, harmony, and rhythm are statistically independent given the music signal, allowing unsupervised mutual information minimization to disentangle them without harming reconstruction quality. To calibrate, we will verify on a held-out validation set of 500 unlabeled music samples that the mutual information between extracted features (using hand-crafted MFCC, chroma, and onset features) is low (<0.05 using a nonparametric estimator) before training, and after training we check that reconstruction quality does not degrade more than 5% compared to a β-VAE (β=1) without MI penalty.

**Why it measures what we claim:** The computational quantity `loss_cycle = cosine_distance(z_text, z_text_cycle)` measures **compositional faithfulness** of the text-to-music mapping because it forces that the text embedding derived from generated music matches the original text embedding; this assumes that the pretrained BERT and projector are sufficiently expressive to capture semantic meaning, and that the simple_decoder can generate music that preserves the compositional structure (melody, harmony, rhythm) encoded in `z_text`. This assumption fails when the generated music is semantically unrelated to the text or when the simple_decoder outputs low-quality audio that loses compositional content; in such cases, `loss_cycle` instead measures the degree of semantic drift during the cycle. The mutual information penalty `loss_mi` measures **disentanglement** because it minimizes the dependence between latent codes; this assumes that the true generative process of music has independent factors for melody, harmony, and rhythm. This assumption fails when musical components are inherently correlated (e.g., a sad melody often uses minor chords); then `loss_mi` forces independence at the cost of reconstruction quality, and instead measures the trade-off between disentanglement and fidelity. The reconstruction loss `loss_recon` measures **reconstruction fidelity** under the structured decoder assumption that the aggregate of melodic, harmonic, and rhythmic streams can reconstruct the original audio; this assumption fails if the decoders cannot capture non-separable global effects (e.g., timbre), in which case `loss_recon` measures only the sum of separately reconstructed components, possibly missing global coherence.

## Contribution

(1) A self-supervised framework (DCM-GEN) that learns disentangled compositional representations of music (melody, harmony, rhythm) without any supervision, using a structured VAE with mutual information minimization. (2) A cycle-consistency loss for aligning text descriptions to the compositional latent space using only a handful of paired examples (e.g., 100), demonstrating that large paired datasets are unnecessary for text-to-music generation. (3) A design principle that musical components can be factorized based on temporal and spectral signatures, enabling cross-modal alignment via cycle-consistency.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | FMA (100k unlabeled) + MusicCaps (100 paired) | Large unlabeled music, small paired anchor |
| Primary metric | CLAP score (on 250 test prompts from MusicCaps) | Measures text-music alignment |
| Baseline 1 | MusicLM (pretrained, fine-tuned on full MusicCaps) | Strong prior, uses full paired data |
| Baseline 2 | Vanilla VAE (single latent code 384-dim) + contrastive alignment (InfoNCE) | No disentanglement, no cycle loss |
| Ablation 1 | DCM-GEN w/o cycle-consistency (contrastive only, InfoNCE) | Isolates cycle-consistency effect |
| Ablation 2 | DCM-GEN w/o disentanglement (single latent code 384-dim, maintain cycle) | Isolates disentanglement effect |
| Additional metric | Attribute manipulation accuracy (melody, harmony, rhythm swap) | Direct test of compositional control |

### Why this setup validates the claim
This design creates a falsifiable test of the central claim—that cycle-consistent alignment on a compositional latent space enables high-quality text-to-music generation with only 100 paired examples. MusicLM, trained on full paired data, sets an upper bound for quality. The vanilla VAE baseline isolates the necessity of disentanglement: if our method’s advantage disappears without structured codes, then disentanglement is the key. The ablation of cycle-consistency directly tests whether the cycle loss adds value over simple contrastive alignment. The CLAP score is chosen because it measures semantic alignment between generated music and text, which directly reflects compositional faithfulness—a high score requires the model to capture multiple facets (melody, harmony, rhythm) from the text. This combination ensures that failure of any sub-claim (disentanglement, cycle-consistency, few-shot learning) will lead to a detectable drop in CLAP score on specific prompt types. The attribute manipulation accuracy metric further validates that the latent codes indeed control the intended musical attributes.

### Expected outcome and causal chain

**vs. MusicLM** — On a case like "a fast rock melody with slow harmonic progression", MusicLM (fine-tuned on full MusicCaps) may learn a global mapping and generate a generic rock song without precise compositional control because it lacks explicit disentanglement. Our method separately encodes melody and harmony, so the text prompt can independently guide each component. We expect our method to achieve comparable CLAP scores on generic prompts (e.g., "upbeat electronic dance music"), but >0.1 higher CLAP on compositional prompts that mention multiple factors. Additionally, on attribute manipulation accuracy, our method should score >80% vs. <50% for MusicLM (which cannot manipulate individual factors).

**vs. Vanilla VAE + contrastive alignment** — On a case like "a jazz improvisation with a walking bass line and syncopated drum pattern", the vanilla VAE’s single latent code cannot factor these components; the contrastive loss maps text to a global style, losing fine-grained compositionality. Our method’s disentangled latent spaces allow each factor to be aligned independently via cycle-consistency. We expect our method to show >0.15 higher CLAP on prompts that mention multiple musical dimensions (e.g., both rhythm and harmony), but parity on single-dimension prompts (e.g., "a sad piano melody"). The attribute manipulation accuracy will be near random for the vanilla VAE (25% for guessing one of four factors) while ours >80%.

**vs. Ablation 1 (DCM-GEN w/o cycle)** — Without cycle-consistency, the contrastive alignment may still capture some cross-modal correspondence but will fail to enforce a bijective mapping. On prompts that require fine-grained factor control (e.g., "change key from C major to G major but keep same rhythm"), the contrastive-only method will produce music that matches the style but not the exact compositional instructions. We expect our full method to have >0.1 higher CLAP on such prompts, and attribute manipulation accuracy will drop from >80% to <60%.

**vs. Ablation 2 (DCM-GEN w/o disentanglement)** — With a single latent code but cycle-consistency, the model still benefits from the cycle loss but cannot leverage factor-specific alignment. On prompts that explicitly name multiple musical components, the single-code version will struggle to decouple them. We expect our full method to have >0.1 higher CLAP on multi-factor prompts, and attribute manipulation accuracy will be near 25% (chance) vs. our >80%.

### What would falsify this idea
If our method fails to exceed baselines specifically on compositional prompts (e.g., those mentioning multiple musical factors) and instead shows uniform improvement or no improvement across all prompts, then the central claim that cycle-consistency on disentangled representations boosts compositional faithfulness is false. Additionally, if attribute manipulation accuracy is not significantly above 25% for the full method, then the latent codes do not control the intended factors.

## References

1. Qwen-Music Technical Report
2. Vevo2: A Unified and Controllable Framework for Speech and Singing Voice Generation
3. MuCodec: Ultra Low-Bitrate Music Codec for Music Generation
4. Zero-shot Voice Conversion with Diffusion Transformers
5. CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models
6. F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
7. BigCodec: Pushing the Limits of Low-Bitrate Neural Speech Codec
8. High Quality Audio Coding with Mdctnet
