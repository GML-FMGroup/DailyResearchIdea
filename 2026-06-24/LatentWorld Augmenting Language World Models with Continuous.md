# LatentWorld: Augmenting Language World Models with Continuous Latent Embeddings for Non-Verbal State Prediction

## Motivation

Qwen-AgentWorld (2026) represents environment states solely in natural language, which fails to capture continuous numerical dynamics (e.g., robot joint angles) or high-dimensional visual changes (e.g., pixel-level object positions). The root cause is that the training objective—next-state prediction—remains in language space, so the model never learns to represent non-verbal information. This structural assumption limits world model fidelity to language-expressible changes only, as evidenced by the language-only transition prediction in Web Agents with World Models.

## Key Insight

A shared latent space that reconstructs both language and continuous modalities forces the model to encode information that is predictive of state dynamics but not easily verbalized, thereby bridging the expressiveness gap without discarding language's interpretability.

## Method

(A) **What it is**: LatentWorld is a world model that maps multimodal observations (textual state descriptions and raw sensory data like images or numerical vectors) into a joint latent embedding space via dual encoders, predicts the next latent state using a transformer-based transition predictor, and decodes back to both modalities. Inputs: current textual state description $T_t$, raw observation $O_t$ (e.g., image, vector), and action $a_t$. Outputs: predicted next textual description $\hat{T}_{t+1}$ and raw observation $\hat{O}_{t+1}$.

(B) **How it works** (pseudocode):
```python
# Encoders
def encode(t, o):
    h_text = text_encoder(t)  # RoBERTa
    h_obs = obs_encoder(o)    # ResNet for images, MLP for vectors
    h = linear(concat(h_text, h_obs), d=512)
    return h

# Transition predictor (transformer)
state_seq = [h_0, h_1, ..., h_t]
action_embeds = embed_actions([a_0, ..., a_t])
h_pred = transformer(state_seq, action_embeds)[-1]  # next latent

# Decoders
pred_text = text_decoder(h_pred)  # LLaMA head
pred_obs = obs_decoder(h_pred)    # diffusion decoder for images, MLP for vectors

# Training loop
for (T_t, O_t, a_t, T_{t+1}, O_{t+1}) in dataloader:
    h_t = encode(T_t, O_t)
    h_next = encode(T_{t+1}, O_{t+1})  # target
    h_pred = transition(h_t, a_t)
    L_text = cross_entropy(T_{t+1}, text_decoder(h_pred))
    L_obs = mse(O_{t+1}, obs_decoder(h_pred))  # for vectors; for images use perceptual loss
    L_kl = kl_div(h_pred, N(0, I))  # beta-VAE
    loss = L_text + 1.0 * L_obs + 0.1 * L_kl
    update params
```
Hyperparameters: latent dim=512, β=0.1, λ=1.0.

(C) **Why this design**: We chose a joint latent space over separate modality-specific representations because it forces the model to find a shared representation that is predictive of both text and raw observations, preventing the language decoder from ignoring visual information. We selected a variational bottleneck (β-VAE) rather than a deterministic mapping because it regularizes the latent space to be smooth and prevents overfitting to noise in continuous modalities; the cost is potential information loss, which we tune via β. We used a transformer for transition prediction due to its ability to handle variable action sequences and long-range dependencies, unlike RNNs which suffer from vanishing gradients; the trade-off is higher computational cost. We chose to decode both modalities rather than only the next latent because it provides two supervision signals, increasing model size but ensuring both aspects are learned. The raw observation decoder is expensive for high-dimensional data (e.g., diffusion), but for many domains numerical states are low-dimensional, making a lightweight MLP sufficient.

(D) **Why it measures what we claim**: The reconstruction loss on raw observations $L_\text{obs}$ measures the model's ability to predict non-verbal state changes because it directly evaluates fidelity in the continuous space; if the model cannot capture visual/numerical dynamics, $L_\text{obs}$ will be high. This assumes that the raw observation decoder is powerful enough to represent the dynamics—if the decoder is too weak (e.g., low-resolution output), low $L_\text{obs}$ may not indicate good dynamics capture. The KL divergence term $L_\text{kl}$ measures the model's adherence to a Gaussian prior, encouraging generalization but not directly measuring non-verbal prediction; it serves as regularization to prevent overfitting. The text reconstruction loss $L_\text{text}$ measures verbal prediction fidelity, but high $L_\text{text}$ can be achieved even if the latent space ignores visual information, as long as language covers it. Therefore, the combined loss ensures both modalities are captured. The transition predictor's next latent state $h_\text{pred}$ encodes information from both modalities, and its quality is measured by both decoder losses. The assumption that the latent space is the shared bottleneck fails if the model memorizes trajectories without generalization (overfitting), in which case losses are low but predictions are not actually capturing dynamics; we mitigate this with held-out validation and the variational regularizer.

## Contribution

(1) LatentWorld, a world model architecture that jointly predicts text and continuous sensory states by learning a shared latent representation via dual encoders and decoders. (2) The insight that augmenting language world models with a continuous latent space improves prediction of non-verbal state changes, validated on domains with visual or numerical dynamics. (3) An extension of the Qwen-AgentWorld training pipeline that incorporates multimodal data without requiring language-only trajectories.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Multi-modal MiniGrid | Both text and grid images available. |
| Primary metric | Normalized MSE on next observation | Directly measures non-verbal prediction. |
| Baseline 1 | Language-only World Model | Tests necessity of raw observation input. |
| Baseline 2 | Separate-latent World Model | Tests joint representation benefit. |
| Baseline 3 | Deterministic latent World Model | Tests regularity from variational bottleneck. |
| Ablation-of-ours | LatentWorld without KL (β=0) | Isolates regularization effect. |

### Why this setup validates the claim

The dataset provides multimodal observations (text state descriptions and grid images), enabling direct test of the joint representation's ability to capture both verbal and non-verbal dynamics. The primary metric (normalized MSE on next observation) directly evaluates the central claim: predicting raw, non-verbal states. Baselines isolate key design choices: language-only probes whether raw observation input is necessary; separate-latent tests whether a shared latent space forces multimodal learning; deterministic latent examines whether the variational bottleneck improves generalization. The ablation (β=0) confirms the regularization role. This combination forms a falsifiable test: if the joint latent space is not essential, separate-latent will match ours; if the bottleneck is unnecessary, deterministic will match ours on held-out data.

### Expected outcome and causal chain

**vs. Language-only World Model** — On a case where the task requires visual information (e.g., object color or position changes), the language-only baseline will fail to predict the next observation correctly because it only sees text, missing visual changes. Our method uses both modalities, so on such cases we expect 20-40% lower error. On purely language-defined transitions, both perform similarly.

**vs. Separate-latent World Model** — On a case where visual and textual information are complementary (e.g., text says "left" but image shows exact distance), the separate-latent model may ignore visual cues in text decoding because it has its own latent for text. Our joint latent forces alignment, so on these cases we expect 10-20% lower error in both modalities. On independent redundancy, performance similar.

**vs. Deterministic latent World Model** — On a novel trajectory unseen in training, the deterministic model overfits to training noise, producing poor generalization. Our variational bottleneck regularizes, so on held-out sequences we expect 15-25% lower error, while on training distribution both are good.

### What would falsify this idea

If our method does not show clear advantage over language-only baseline on visual-dependent subsets, or if the gain over separate-latent is uniform across all transitions rather than concentrated on multimodal-dependent ones, the central claim that joint representation captures non-verbal dynamics is falsified.

## References

1. Qwen-AgentWorld: Language World Models for General Agents
2. Web Agents with World Models: Learning and Leveraging Environment Dynamics in Web Navigation
3. WebArena: A Realistic Web Environment for Building Autonomous Agents
4. Mind2Web: Towards a Generalist Agent for the Web
5. Learning Interactive Real-World Simulators
6. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances
7. Don’t Generate, Discriminate: A Proposal for Grounding Language Models to Real-World Environments
8. MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge
