# Continuous Domain Modulation for Subject-Driven Video Generation

## Motivation

DomainShuttle's Domain-MoT conditions on a binary domain type (in-domain vs cross-domain), which is too coarse for nuanced domain shifts where the reference image's domain varies continuously (e.g., partially styled, mixed lighting). This discretization structurally limits adaptation because a single binary flag cannot represent the fine-grained domain characteristics present in the reference image, leading to suboptimal subject fidelity and editability for intermediate domain scenarios.

## Key Insight

Domain variation is inherently continuous, and the reference image itself provides a continuous signal for domain modulation, enabling precise adaptation where binary discretization fails.

## Method

## ConDoMo (Continuous Domain Modulation)

### (A) What it is
ConDoMo replaces the binary domain type in Domain-MoT's AdaLN with a learned continuous domain embedding derived from the reference image via a lightweight encoder, enabling fine-grained adaptation across domain shifts.

### (B) How it works
```python
# ConDoMo Module
# Input: reference image I_ref, video latent features F (T x H x W x C), timestep t
# Output: modulated features F_mod

# 1. Encode domain embedding
e_domain = DomainEncoder(I_ref)  # e_domain ∈ R^d, d=128
# DomainEncoder: ResNet-18 (pretrained, frozen except last two residual blocks) -> global avg pool -> two linear layers: 512->256 (GELU) -> 128

# 2. For each transformer block at layer l:
#    - t_embed = sinusoidal_embedding(t)  # t_dim = 256
#    - condition = concat(t_embed, e_domain)  # shape (t_dim + d) = 384
#    - scale_l, shift_l = AdaLN_MLP(condition)  # 2-layer MLP: 384 -> 256 (GELU) -> 2*C
#    - LN_output = LayerNorm(F)
#    - F_mod = LN_output * (1 + scale_l) + shift_l
```

### (C) Why this design
We chose a lightweight encoder (ResNet-18) over a larger pretrained visual backbone because the domain signal is primarily in low-to-mid level features (texture, color, style) and a smaller encoder suffices, accepting the cost that it may not capture very high-level semantic domains (e.g., fine art style). We opted for a single shared domain embedding across all layers rather than per-layer embeddings to keep parameters minimal and avoid overfitting; the trade-off is that one embedding must balance adaptation across all layers, which could dilute layer-specific domain sensitivity. We used AdaLN (adaptive layer normalization) instead of cross-attention injection because AdaLN directly modulates the representation's scale and shift, offering a computationally cheap way to condition the entire feature map, though it may be less expressive than cross-attention for spatially varying domain effects. We freeze the early layers of the ResNet-18 to retain general visual features and only fine-tune the last two residual blocks and the MLP to prevent overfitting to the limited number of subjects in training, accepting that early layers may not adapt to novel domain cues not present in ImageNet.

### (D) Why it measures what we claim
We assume the domain encoder extracts only domain-relevant visual features (e.g., lighting, texture) and ignores subject identity. The domain embedding e_domain measures continuous domain similarity between reference and target video because it is learned via the video generation loss to predict AdaLN parameters that minimize reconstruction error; the assumption is that training on diverse subjects forces the encoder to discard identity by focusing on common domain features. This assumption fails when the reference image contains strong identity or background cues that dominate the embedding, in which case e_domain may reflect identity similarity instead, potentially causing the model to overfit to the reference subject's appearance rather than its domain. To verify this, we conduct a probe experiment (see experiment) that visualizes clusters of domain embeddings by domain attributes and measures correlation with an identity embedding. If correlation is high, the assumption is violated, indicating identity leakage.

## Contribution

(1) A continuous domain modulation mechanism (ConDoMo) that replaces binary domain type with learnable domain embeddings for subject-driven video generation. (2) A domain encoder design that extracts fine-grained domain features from reference images, enabling adaptation across continuous domain shifts. (3) A principle that continuous modulation outperforms binary discretization for nuanced domain adaptation, verified by ablation studies in our experiments.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | DA-VINCI dataset (diverse subjects and domains) or custom collected video dataset with 100 subjects in 10 domains (lighting, texture, color) | Tests domain generalization across continuous shifts |
| Primary metric | CLIP domain similarity (cosine between CLIP image embeddings of ref and generated) and Identity similarity (cosine between face embeddings from a pretrained face recognizer) | Domain alignment without identity bias; identity leakage detection |
| Baseline 1 | Domain-MoT with binary domain label | Binary baseline, no continuous adaptation |
| Baseline 2 | Cross-attention injection (domain embedding as key/value in cross-attention) | More expressive conditioning baseline |
| Baseline 3 | DreamBooth | Subject personalization without domain modulation |
| Ablation | ConDoMo with per-layer domain embeddings (separate MLP per block) | Tests effect of shared vs per-layer conditioning |
| Probe experiment | PCA visualization and k-means clustering (k=10) of e_domain over all reference images; compute mutual information between cluster labels and ground-truth domain attributes vs. subject identity | Validates that domain embedding captures domain variation |

### Why this setup validates the claim
The chosen dataset covers a wide range of continuous domain shifts (lighting, texture, color), enabling a rigorous test of the method's ability to adapt to fine-grained variations. The primary metric (CLIP domain similarity) isolates domain alignment from identity preservation, directly measuring the core claim that the embedding captures domain-relevant features. The additional identity metric quantifies leakage, testing the load-bearing assumption. The baselines target specific sub-claims: Domain-MoT tests binary vs. continuous conditioning; cross-attention tests whether AdaLN is sufficient or if more expressive conditioning is needed; DreamBooth tests the necessity of explicit domain modulation over pure subject personalization. The ablation of per-layer embeddings tests the trade-off between shared flexibility and layer-specific adaptation. The probe experiment directly visualizes and measures the disentanglement of domain and identity in the embedding space. This combination ensures that any observed advantage of ConDoMo can be attributed to its novel continuous domain embedding and design choices, forming a falsifiable test: if ConDoMo fails to outperform on subsets where the predicted failure modes of baselines are active, the claim is unsupported.

### Expected outcome and causal chain

**vs. Domain-MoT** — On a case where the reference image has a subtle lighting shift (e.g., warm sunset light), Domain-MoT assigns a single binary domain label, causing coarse adaptation that leads to frame flickering as lighting changes gradually. Our ConDoMo uses a continuous embedding that tracks the lighting nuance, modulating each frame's normalization smoothly. We expect ConDoMo to achieve >10% higher CLIP domain similarity on videos with gradual domain changes (e.g., 5-10% of the dataset with slow lighting transitions), while performing similarly on abrupt shifts where binary suffices.

**vs. Cross-attention injection** — On a case with complex texture domain (e.g., wooden surface with grain), cross-attention may attend to identity-related regions (e.g., facial features) instead of texture, causing identity leakage into the domain embedding. Our AdaLN scales and shifts globally, preventing per-location attention errors and preserving identity. We expect ConDoMo to show 5% better identity retention (measured by face similarity) while maintaining comparable domain alignment, especially on high-frequency texture domains.

**vs. DreamBooth** — On a case where the domain is novel (e.g., a cartoon style not seen in training), DreamBooth overfits to the reference subject's appearance, failing to transfer the domain to new poses or contexts. Our domain encoder leverages frozen early layers that capture general low-level features, enabling generalization to unseen styles. We expect ConDoMo to achieve 15% higher domain alignment on unseen domain subsets, while DreamBooth's domain alignment collapses.

### What would falsify this idea
If ConDoMo outperforms baselines uniformly across all domain shift severities (not just fine-grained), then the continuous embedding is not the cause of improvement—instead, the encoder leakage or other factors dominate. Specifically, if the gain on gradual shifts is not larger than on abrupt shifts, the central claim of fine-grained adaptation is unsupported. Additionally, if the probe experiment shows high mutual information between domain embeddings and identity (e.g., >0.5), the load-bearing assumption is violated, indicating that e_domain does not measure domain similarity in isolation.

## References

1. DomainShuttle: Freeform Open Domain Subject-driven Text-to-video Generation
2. VACE: All-in-One Video Creation and Editing
3. Phantom: Subject-Consistent Video Generation via Cross-Modal Alignment
4. HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning
5. DreamVideo-2: Zero-Shot Subject-Driven Video Customization with Precise Motion Control
6. UNIC-Adapter: Unified Image-Instruction Adapter with Multi-Modal Transformer for Image Generation
7. OminiControl: Minimal and Universal Control for Diffusion Transformer
8. LTX-Video: Realtime Video Latent Diffusion
