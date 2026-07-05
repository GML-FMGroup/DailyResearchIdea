# HashDiff: Parallel Generation via Diffusion over Hash Encoded Sequences

## Motivation

Existing hash-based language models like MultiHashFormer compress tokens into continuous signatures but still rely on sequential autoregressive decoding, retaining the quadratic self-attention bottleneck. The root cause is the autoregressive formulation that conditions each token on previous ones; this prevents parallel generation of the entire sequence. Replacing this with a process that generates the sequence in parallel requires a mechanism that models the joint distribution over all hash signatures without sequential dependencies.

## Key Insight

The hash encoding space is a continuous, low-dimensional manifold where the hash embedding is invertible, allowing a diffusion process to generate the entire sequence of hash signatures in parallel by learning to reverse a noising process on the full sequence latent representation.

## Method

(A) **What it is**: HashDiff is a continuous diffusion model that operates on the concatenated hash signature sequence of the entire output. It takes as input an initial context (e.g., from a prompt) and generates the hash signatures for all future tokens in parallel via reverse diffusion. The hash signatures are then mapped to tokens via a fixed hash-to-token mapping (multiple independent hash functions as in MultiHashFormer).

(B) **How it works**:
```python
# Training
for each sequence of tokens x_1..x_T in dataset:
    # Compute hash signatures for each token using K hash functions
    H = [hash_1(x_1), ..., hash_K(x_1), ..., hash_1(x_T), ..., hash_K(x_T)]  # shape: (T*K, D/K)
    # Flatten to a single vector of dimension D*T (e.g., D=32, T=256 => 8192 dim)
    H_flat = flatten(H)
    # Sample noise epsilon ~ N(0, I)
    # Sample a diffusion step t uniformly from 1..T_steps (T_steps=1000)
    # Compute noisy H: H_t = sqrt(alpha_t) * H_flat + sqrt(1-alpha_t) * epsilon
    # Predict denoised H via denoising network f (transformer with conditioning)
    loss = ||H_flat - f(H_t, t, context)||^2
    # Optimize

# Sampling (parallel generation)
H_T ~ N(0, I)  # initial noise (same dimension as H_flat)
for t in reversed(range(1, T_steps)):  # T_steps=100 for fast sampling
    # Predict denoised H and add noise
    H_{t-1} = sqrt(alpha_{t-1}) * f(H_t, t) + sqrt(1-alpha_{t-1}) * epsilon_t
epsilon_t ~ N(0, I)
# After loop, H_0 = f(H_1, 1) approximation
# Reshape to per-token hash signatures and decode each via nearest neighbor in hash space
```
Hyperparameters: dimension per token D=32, K=8 hash functions (each output D/K=4 dim), number of diffusion steps (sampling) T_s=100, noise schedule: cosine with linear beta schedule from 1e-4 to 0.02, denoising network: 2-layer Transformer with 4 heads, hidden dimension 512, feedforward dimension 2048, GeLU activation. Training: AdamW optimizer (lr=1e-4, β1=0.9, β2=0.999), batch size 32 sequences of length 256, trained for 100k steps. Model has 12M parameters; training takes 24 hours on a single RTX 3090 GPU; memory usage 8 GB.

(C) **Why this design**: We chose continuous diffusion over discrete diffusion because the hash signatures are continuous vectors, allowing direct application of score-based models with simple Gaussian noise and invertible mapping. We chose to concatenate all hash signatures into a single vector rather than modeling each token's hash independently because this captures dependencies across tokens at the sequence level. A trade-off is that the dimension scales linearly with sequence length L, but we mitigate this by using low-dimensional hash signatures (e.g., 32 dimensions per token) and a transformer that handles long contexts efficiently via attention. We adopted the cosine noise schedule over linear because it provides more uniform reconstruction difficulty across steps. We opted for a fixed hash mapping rather than learned embeddings to maintain the constant parameter footprint advantage from MultiHashFormer. The denoising network is a transformer rather than a U-Net because the hash sequence is not spatially structured like images; attention can capture arbitrary token dependencies. This design contrasts with prior work like autoregressive U-Nets that operate on raw bytes and lack a continuous latent space for diffusion.

(D) **Why it measures what we claim**: The denoising loss L2(H_flat, f(H_t)) measures the ability to reconstruct the original hash sequence from noise, which corresponds to learning the joint distribution over hash signatures. This is necessary for parallel generation because a perfect denoiser can generate the sequence in one pass from noise. Specifically, the mean squared error on the hash dimensions measures the fidelity of the generated hash to the true hash; this is valid under the assumption that the hash space is smooth and the mapping from hash to token is locally linear (invertible), so small errors in hash space translate to correct token predictions. **Assumption A**: Each hash function h_i is a random projection onto [0,1]^{D/K} such that in a small neighborhood of any point, the mapping from hash vector to token index is approximately linear and invertible. This is assumed to hold with high probability given K independent hashes. This assumption fails when hash collisions occur (multiple tokens mapping to the same hash region), in which case the error measure reflects ambiguity rather than token accuracy. To mitigate this, we use multiple independent hash functions to ensure unique combinatorial signatures, as in MultiHashFormer. The reverse diffusion process operationalizes parallel generation because it updates all token representations simultaneously at each step, bypassing the sequential causal mask.

## Contribution

(1) A novel framework, HashDiff, that replaces autoregressive decoding of hash signatures with continuous diffusion, enabling parallel generation of entire sequences. (2) The insight that the continuous, invertible hash encoding space is naturally suited for diffusion models, providing a theoretical link between hash-based compression and non-autoregressive generation. (3) A design principle: using multiple independent hash functions preserves token uniqueness under diffusion, ensuring the mapping from generated hash to token is approximately unique.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | WikiText-103 | Standard word-level language modeling benchmark |
| Primary metric | Perplexity | Direct measure of generation quality |
| Baseline 1 | MultiHashFormer | Autoregressive hash-based baseline |
| Baseline 2 | Autoregressive U-Net | Byte-level autoregressive baseline |
| Baseline 3 | GPT-2 Transformer | Subword autoregressive baseline |
| Ablation 1 | HashDiff with single hash | Tests necessity of multiple hash functions |
| Ablation 2 | HashDiff with learned embedding | Tests effect of fixed vs learned hash (embedding dim=32, same network) |

### Why this setup validates the claim

This setup tests the central claim that parallel generation via continuous diffusion on concatenated hash signatures can match or exceed autoregressive baselines in language modeling. WikiText-103 provides a standard, challenging corpus with long-range dependencies. Perplexity directly measures the model's ability to assign high likelihood to test data, which is the ultimate test of a generative model. By comparing against three distinct families—autoregressive hash, byte-level, and subword—we cover the main paradigms and isolate the unique benefits of hash-based diffusion. The ablation with a single hash function pinpoints whether multiple hashes are crucial for avoiding collisions and preserving token identity. The additional ablation with learned embeddings isolates the effect of fixed hash functions, testing whether the hash mapping itself or the diffusion process drives performance. Together, these choices form a falsifiable test: if HashDiff achieves competitive perplexity, especially on complex instances, the claim is supported; if it fails uniformly, the central idea is refuted.

### Expected outcome and causal chain

**vs. MultiHashFormer** — On a long-range dependency instance in WikiText-103 (e.g., coreference where antecedent is >50 tokens before the target), MultiHashFormer's autoregressive decoding processes tokens one at a time, and its single hash representation may lose information due to collisions, causing incoherent long-range structure. Our HashDiff models the entire hash sequence jointly via diffusion, capturing dependencies across all positions simultaneously. We expect HashDiff to achieve perplexity <90 on the full test set, at least 5 points lower than MultiHashFormer on long-range sentences (≥50 tokens apart), while performing similarly on short sequences.

**vs. Autoregressive U-Net** — On a character-level reasoning task (e.g., reversing a word letter by letter from the training set), Autoregressive U-Net's byte-level representation and causal mask require sequential processing, making it slow and prone to error accumulation due to dependence on previous predictions. Our HashDiff generates all tokens in parallel from noise, bypassing sequential dependence. We expect HashDiff to have higher fidelity (lower perplexity) by 3-5 points on such tasks, but similar perplexity on general text.

**vs. GPT-2 Transformer** — On an ambiguous word prediction (e.g., "bank" in finance vs. river from WikiText-103 test set), GPT-2's subword tokens may lose fine-grained token identity due to tokenization merging words, causing ambiguity. Our HashDiff's multiple fixed hash mappings ensure token-level uniqueness, reducing ambiguity. We expect HashDiff to have a smaller perplexity gap on ambiguous tokens (≤2 points difference), but overall perplexity may be slightly higher (within 2-3 points) due to the difficulty of continuous diffusion.

**Ablation results**: HashDiff with single hash is expected to have perplexity 10-15 points higher than the full model, confirming the necessity of multiple hashes. HashDiff with learned embedding is expected to perform similarly to the fixed-hash version (within 1-2 points), indicating that the fixed hash mapping is not a bottleneck.

### What would falsify this idea

If HashDiff's perplexity on WikiText-103 is higher than all baselines by >5 points and the gain is not concentrated on long-range or ambiguous tokens but uniformly worse, then the central claim that parallel generation via hash diffusion can match autoregressive models is false. Additionally, if the learned embedding ablation significantly outperforms the fixed-hash version (>3 points), it would suggest that the fixed hash mapping is a limiting factor, requiring redesign.

## References

1. MultiHashFormer: Hash-based Generative Language Models
2. From Bytes to Ideas: Language Modeling with Autoregressive U-Nets
3. Bolmo: Byteifying the Next Generation of Language Models
4. Byte Latent Transformer: Patches Scale Better Than Tokens
5. From Language Models over Tokens to Language Models over Characters
6. Exact Byte-Level Probabilities from Tokenized Language Models for FIM-Tasks and Model Ensembles
7. T-FREE: Subword Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings
8. Should you marginalize over possible tokenizations?
