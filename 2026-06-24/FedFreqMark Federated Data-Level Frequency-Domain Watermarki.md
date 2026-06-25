# FedFreqMark: Federated Data-Level Frequency-Domain Watermarking for Architecture-Invariant Generative Model Ownership

## Motivation

Existing watermarking methods for generative models, such as FedOT and Stable Signature, rely on specific architectural components like UNet or latent decoder, and fail to generalize to other model families (GANs, VAEs, Transformers). This structural dependence arises because these methods embed watermarks into model parameters or latent spaces that are architecture-specific. Consequently, a universal watermarking scheme that operates at the data level is needed to enable ownership verification and client tracing in heterogeneous federated learning environments.

## Key Insight

Frequency-domain patterns embedded in the training data distribution act as invariant fingerprints that are learned by any generative architecture, because they modify the underlying statistical structure that all generative models approximate, independent of their internal design.

## Method

**(A) What it is:** FedFreqMark is a data-level watermarking framework that embeds a set of frequency-domain patterns into the training images distributed to clients. The method takes as input a cover dataset, a set of client identifiers, and produces a watermarked dataset uniquely tied to each client (via a superimposed signature) while a global ownership pattern is common across all clients. During federated training, clients use the watermarked data to train their local generative models (any architecture: GAN, VAE, Transformer, UNet). Later, generated images are analyzed in the Fourier domain to extract both the global watermark for ownership verification and the client-specific signature for tracing.

**Load-bearing assumption:** We assume that any generative architecture, due to spectral bias, will learn the injected frequency patterns because they modify the data distribution statistics that all models approximate. This assumption must be verified across two fundamentally different architectures (e.g., CNN-based GAN and ViT-based generator) under the same federated training protocol.

**(B) How it works:**
```pseudocode
# FedFreqMark: Federated Data-Level Frequency Watermarking

# Parameters:
# - alpha: strength of global watermark (e.g., 0.1)
# - beta_i: strength of client-specific signature for client i (e.g., 0.05)
# - delta: threshold for detection (e.g., 0.7 normalized correlation)
# - k: number of frequency components to modify (e.g., 256)
# - architecture: generative model family (e.g., DCGAN for CNN-based, ViT-GAN for transformer-based)

# Server-side initialization:
1. Choose a base pattern P_G in frequency domain (e.g., random binary mask in middle frequencies).
2. For each client i, generate a unique signature pattern P_i (orthogonal to P_G and other P_j via Gram-Schmidt).
3. Combine: P_total_i = alpha * P_G + beta_i * P_i (global + client-specific).

# Data watermarking (per local dataset at client i):
4. Take each training image x in client i's dataset.
5. Compute its 2D discrete Fourier transform (DFT) F = fft2(x).
6. Select a set of frequency indices (e.g., k largest magnitude positions) from F.
7. Modify the complex amplitude at those indices: F'(u,v) = F(u,v) + P_total_i(u,v) * max(|F|) * epsilon, where epsilon is a small perturbation (e.g., 0.01).
8. Compute inverse DFT: x' = ifft2(F').
9. Clip pixel values to valid range [0,255].

# Federated training:
10. Each client i trains its local generative model on watermarked dataset D'_i for E local epochs (e.g., E=5).
11. Models are aggregated via FedAvg with C communication rounds (e.g., C=100).

# Verification:
12. Given a generated image y (from any client), compute its DFT.
13. Extract the watermark by correlation with P_G: score = |corr(fft2(y), P_G)|.
14. If score > delta, claim ownership.
15. For client tracing, correlate with each client-specific pattern P_i: trace_i = |corr(fft2(y), P_i)|; client with max trace_i > delta2 (e.g., 0.5) is identified.
```

**(C) Why this design:** We chose frequency-domain over spatial-domain embedding because frequency perturbations are less perceptible and survive nonlinear transformations like resizing and compression, at the cost of requiring DFT computation for every image. We used a fixed set of k frequency indices selected by largest magnitude to ensure robustness against common distortions, accepting that content-aware selection may correlate with semantic content and reduce security against collusion attacks. We employed orthogonal client signatures via Gram-Schmidt to minimize false positives in tracing; the alternative of random assignment would risk overlap and confound client identification, at the cost of increased computational overhead as client count grows. We set alpha > beta_i to prioritize global ownership detection over tracing; equal weighting would allow multiple clients to be mistakenly identified if signatures are partially correlated. The data-level approach avoids architecture-specific modifications, but watermark strength must be high enough to survive training epochs, which may slightly degrade generation quality.

**(D) Why it measures what we claim:** The global pattern correlation score (step 13) measures ownership verification because the pattern is embedded in all training data, so any generative model trained on that data learns to reproduce the pattern as a statistical artifact; this assumption fails when the model is trained on a mixed dataset without the watermark, in which case the correlation reflects random chance. The client-specific pattern correlation (step 15) measures client tracing because each client's unique signature is only present in their local data; this assumption fails if clients collude and average their datasets or if multiple clients share similar data distributions, in which case the signature correlation may indicate the dominant contributor rather than the exact client. The selection of k largest magnitude frequencies (step 6) measures robustness to image content because these positions are least affected by typical image processing; this assumption fails under adversarial attacks that specifically erase high-frequency content, in which case correlation drops to noise level. Correlation measures watermark presence; assumption: model does not erase or distort the pattern; failure modes: model overfits to the watermark (causing it to appear even in non-watermarked inputs) or aggregation suppresses the pattern (reducing detectability).

## Contribution

(1) A data-level watermarking framework for generative models that is invariant to model architecture, enabling ownership verification and client tracing in heterogeneous federated learning. (2) The design of orthogonal client-specific frequency signatures that allow individual tracing while a global pattern ensures ownership, with analytical bounds on false positive rates. (3) Empirical demonstration that the watermark survives training across GAN, VAE, Transformer, and UNet architectures, and is robust to common post-processing attacks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CIFAR-10 | Common benchmark, diverse classes. |
| Primary metric | Detection Accuracy | Directly measures ownership verification. |
| Baseline | Random guessing | Lower bound for detection. |
| Baseline | StegaStamp (spatial) | Spatial watermark baseline. |
| Baseline | DWT-DCT (freq) | Freq watermark baseline without orthogonal signatures. |
| Baseline | Wavelet-domain watermark (Haar DWT) | Compare frequency vs wavelet invariance. |
| Ablation | Ours w/o client sign. | Isolates client tracing component. |
| Architecture variation | DCGAN (CNN) and ViT-GAN (transformer) | Test load-bearing assumption across families. |

### Why this setup validates the claim

CIFAR-10 provides sufficient class diversity to test generalization of watermark patterns across different image content. Random guessing establishes the chance-level detection baseline. StegaStamp tests whether spatial-domain watermarks survive the federated learning process, while DWT-DCT tests frequency-domain watermarking without orthogonal client signatures. The wavelet baseline (Haar DWT with same embedding procedure) isolates the effect of frequency-domain vs. transform-domain invariance. The ablation (removing client-specific signatures) isolates the tracing component. Detection accuracy directly measures whether the global watermark is extractable after federated training and aggregation. To verify the load-bearing assumption, we test two different architectures: a CNN-based GAN (DCGAN) and a transformer-based ViT-GAN. To provide theoretical grounding via spectral bias, we will track the magnitude of the injected frequency components across training epochs and analyze their retention after FedAvg aggregation; we expect high-magnitude components to be preserved due to their large gradient influence. This combination creates a falsifiable test: if our method fails to outperform spatial or non-orthogonal frequency baselines under common distortions, the central claim of robustness is invalid. The ablation further tests whether client-specific orthogonal patterns provide tracing capability beyond a global-only design.

### Expected outcome and causal chain

**vs. Random guessing** — On any generated image, random guessing yields ~50% detection accuracy (chance). Our method embeds a specific frequency pattern with high correlation, so we expect near-perfect detection (>95%) on clean generated images because the pattern is consistently present across all training data.

**vs. StegaStamp** — On a generated image subjected to JPEG compression (quality=75), StegaStamp's spatial pattern suffers distortion because compression disrupts local pixel correlations. Our method uses frequency-domain embedding that spreads the pattern across multiple frequencies, making it more robust. We expect detection accuracy for StegaStamp to drop to ~70% while ours remains >90% under moderate compression.

**vs. DWT-DCT** — When clients collude by averaging their local datasets, DWT-DCT with random client patterns may cause cross-client interference, reducing client tracing accuracy. Our method employs Gram-Schmidt orthogonalization to ensure distinct signatures, so even after dataset averaging, each client's signature remains separable. We expect DWT-DCT tracing accuracy ~60% (due to pattern overlap) while ours >80%.

**vs. Wavelet-domain watermark** — Under Gaussian blur (sigma=1), wavelet-domain watermarks may be smoothed out because blur affects spatial details. Frequency-domain watermarks are spread across the spectrum and partially survive. We expect wavelet detection accuracy ~65% while ours >85%.

**Ablation: Ours w/o client sign.** — Without client-specific signatures, client tracing becomes impossible; global detection remains similar to the full method. We expect global detection accuracy to match the full method (~95%), but client tracing accuracy drops to random (20% for 5 clients). This confirms that client signatures are responsible for tracing capability.

### What would falsify this idea

If our method fails to achieve detection accuracy significantly above random on clean images for either DCGAN or ViT-GAN, or if the advantage over baselines disappears under common attacks like resizing or compression, then the frequency-domain robustness claim is false. Additionally, if client tracing accuracy with orthogonal patterns is not significantly higher than with random patterns under collusion, the orthogonal design fails.

## References

1. FedOT: Ownership Verification and Leakage Tracing via Watermarks for Federated LDMs
2. WMAdapter: Adding WaterMark Control to Latent Diffusion Models
3. The Stable Signature: Rooting Watermarks in Latent Diffusion Models
4. Flexible and Secure Watermarking for Latent Diffusion Model
5. IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models
6. CycleGANWM: A CycleGAN watermarking method for ownership verification
7. Versatile Diffusion: Text, Images and Variations All in One Diffusion Model
