# Continuous Latent Memory for Long-Form Music Generation

## Motivation

Purely discrete autoregressive models like Qwen-Music lose fine-grained continuous control and long-range coherence because the discrete tokenization discards acoustic nuances and the autoregressive decoding propagates local errors without a global structure representation. This leads to degradation in longer generations and inability to smoothly vary attributes like pitch or timbre.

## Key Insight

A gated recurrent latent state, updated at each step from the previous token and latent, acts as a compressed bottleneck that forces the model to disentangle global structure from local details, enabling coherent long-form generation while preserving continuous controllability.

## Method

### (A) What it is:
CLM (Continuous Latent Memory) augments an autoregressive transformer decoder with a dynamically updated continuous latent vector that serves as a recurrent state, conditioning each token prediction via cross-attention.

### (B) How it works:
```python
# Input: token embeddings E = [e_1,...,e_p] (from prompt), global condition c
# Initialization
z_0 = MLP_initial(c)  # dim=256, 1 hidden layer of 512 neurons, ReLU activation
memory_state = z_0
# For each step t from 1 to T:
for t in range(1, T+1):
    # Decoder forward pass with cross-attention to memory
    # Each decoder layer has a cross-attention sub-layer that uses memory_state as K,V
    h_t = TransformerDecoder(E[:t], memory=memory_state)  # h_t is last layer hidden of current token
    # Update memory via GRU
    memory_state = GRU(memory_state, h_t)  # dim=256, hidden_size=256
    # Predict next token logits from h_t and memory
    logits = Linear(h_t + memory_state)  # dim=vocab_size
    y_{t+1} = sample(softmax(logits))
    # Append new token embedding
    E.append(Embed(y_{t+1}) + pos(t+1))
```
Hyperparameters: GRU hidden size = 256, memory dim = 256, MLP_initial uses 1 hidden layer of 512. Training uses cross-entropy loss on next-token prediction. Compute budget: Training on MAESTRO (~1200 MIDI files, ~10M tokens) on a single NVIDIA A100 80GB takes approximately 72 GPU hours. Model size: ~200M parameters.

### (C) Why this design:
We chose a GRU for the latent update over a transformer because the recurrence forces a fixed-dimension bottleneck that compresses long-term information, accepting that it may fail to capture very complex polyphonic structures that a larger memory could hold. We inject the latent via cross-attention rather than concatenation to allow the decoder to selectively attend to different aspects of the latent per layer, at the cost of increased computational overhead. We keep the latent dimensionality low (256) to enforce compression and prevent the model from memorizing token sequences in the latent, which would undermine the intended decoupling. The initial latent from an MLP ensures consistency across starting conditions.

### (D) Why it measures what we claim:
The memory update GRU(h_t, z_{t-1}) produces z_t that is a sufficient statistic for global sequence structure up to time t, under the assumption that the GRU's gating mechanism can propagate relevant information indefinitely; this assumption fails when the GRU's hidden size is too small to memorize complex long-range dependencies, in which case z_t reflects only local context. The cross-attention mechanism `TransformerDecoder(E, memory=z_t)` operationalizes conditioning on global structure by allowing every token hidden state to query the latent; this measures 'global coherence' under the assumption that the latent encodes high-level patterns like key and tempo. The disentanglement claim relies on the latent being updated solely from previous token and previous latent, not from the full context; this prevents the latent from encoding token-level details, but the assumption fails when the token embedding carries global information (e.g., a chord token), in which case the latent may redundantly encode local patterns. Additionally, FAD on long sequences measures global coherence under the assumption A that local errors are not length-correlated. The failure mode F is that uniform FAD gains across all sequence lengths could come from better local modeling rather than global structure encoding. We probe this by comparing FAD gain on long vs. short sequences; if gain is uniform, the latent is not serving as a global state. To verify the latent actually encodes global structure, we train a linear probe on z_t to predict musical key and tempo, testing on sequences of varying lengths.

### (E) Verification of load-bearing assumption:
The assumption that a 256-dimensional GRU can reliably encode and propagate global musical structure over long sequences (>15 seconds) is explicitly tested via a calibration experiment: we generate sequences of lengths 5s, 10s, 15s, 20s, 30s and compute FAD and latent probe accuracy for each. If FAD gain plateaus or probe accuracy drops sharply after 15s, we flag the method's limitation. Additionally, we include a variant with GRU hidden size 1024 to compare capacity. This verification does not alter the core method but informs the user about the effective range of the memory.

## Contribution

(1) A novel architecture, Continuous Latent Memory, that combines a recurrent continuous state with cross-attention in a transformer decoder to improve long-range coherence in autoregressive music generation. (2) A design principle that low-dimensional latent bottlenecks forced by gated recurrence effectively decouple global structure from local token prediction, validated through controlled experiments on phrase-level repetitions. (3) An open-source implementation and pre-trained model checkpoints for the music domain.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | MAESTRO | Standard piano performance dataset. |
| Primary metric | Frechet Audio Distance (FAD) | Measures global audio distribution match. |
| Baseline 1 | Qwen-Music | Strong text-to-music baseline. |
| Baseline 2 | MuCodec | Codec-based generation with high compression. |
| Baseline 3 | Memorizing Transformer (Wu et al., 2022) | Transformer with external memory retrieval for long sequences. |
| Baseline 4 | Sliding window attention (window size 4096) | Efficient long-context transformer baseline. |
| Ablation-of-ours | No memory (standard transformer) | Tests contribution of latent memory. |
| Probing experiment | Linear probe on latent z_t | Classify musical key (major/minor) and tempo (discretized into 20 bins) from z_t at each step; measure accuracy vs. sequence length. |

### Why this setup validates the claim
MAESTRO provides complex, long-range musical structures that challenge any memory-bottleneck method. FAD captures global coherence by comparing distributional similarity of generated audio to real performances, directly testing whether our latent encodes high-level patterns. Qwen-Music, lacking explicit recurrent state, is expected to falter on long-term dependencies, while MuCodec's fixed-length codes cannot adapt to evolving context. Memorizing Transformer and sliding window attention represent state-of-the-art long-context methods; comparing against them shows whether our lightweight memory offers practical gains. The ablation of removing memory isolates the effect of the latent state, ensuring any FAD improvement stems from the proposed mechanism. The probing experiment verifies that the latent actually encodes global musical attributes (key, tempo) and that this encoding persists over long sequences. This combination forms a falsifiable test: if our method improves FAD specifically on tasks requiring global structure (e.g., longer sequences) and the probe accuracy remains high, the central claim holds; if gains are uniform across all lengths or probe accuracy decays, the memory may be merely additive, not truly compressing global information.

### Expected outcome and causal chain

**vs. Qwen-Music** — On a case where a 30-second piano piece has a repeating chorus with subtle variations, Qwen-Music might lose track of the motif after a few bars because its attention window is finite and lacks a persistent global state. Our method instead maintains a continuous latent vector that updates via GRU, preserving the motif's essence across the entire piece, so we expect a noticeable FAD improvement on long sequences (e.g., >15s) but near-parity on short clips (<5s) where memory is less critical.

**vs. MuCodec** — On a case with a gradual tempo change and key modulation, MuCodec's fixed-length codec may compress local chunks independently, failing to capture the smooth transition at the global level. Our method dynamically updates the latent via cross-attention, allowing each token to query the evolving global context, so we expect lower FAD (better global coherence) on pieces with structural arcs, while on static, repetitive passages both methods perform similarly.

**vs. Memorizing Transformer** — On a case with a long sequence (30s) requiring precise recall of a motif from the beginning, Memorizing Transformer can retrieve it from external memory, but its retrieval is discrete and may miss subtle variations. Our method's continuous latent smoothly updates, potentially capturing gradual changes better. We expect comparable FAD on clearly segmented pieces, but better FAD for music with continuous variation (e.g., rubato).

**vs. Sliding window attention** — Sliding window attention effectively models local dependencies but cannot capture long-range structure beyond the window. On a piece with a global development (e.g., sonata form), our method should outperform due to the persistent latent, while on short or highly local pieces, results are similar.

**vs. Ablation (no memory)** — On a long sequence with complex chord progressions, the standard transformer without memory may generate locally plausible but globally incoherent music, e.g., starting in one key and drifting. Our method with memory maintains the key via the latent state, so we expect a clear FAD gap favoring CLM on sequences longer than the average attention span, and minimal difference on very short sequences.

### What would falsify this idea
If our method shows uniform FAD improvement across all sequence lengths rather than concentrated gains on long sequences (where the failure mode of limited memory is predicted), then the latent is not serving its intended role as a compressed global state. Additionally, if the latent probe accuracy for key/tempo fails to remain above chance for sequences longer than 15s, the assumption that the GRU propagates global structure is violated.

## References

1. Qwen-Music Technical Report
2. Vevo2: A Unified and Controllable Framework for Speech and Singing Voice Generation
3. MuCodec: Ultra Low-Bitrate Music Codec for Music Generation
4. Zero-shot Voice Conversion with Diffusion Transformers
5. CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models
6. F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
7. BigCodec: Pushing the Limits of Low-Bitrate Neural Speech Codec
8. High Quality Audio Coding with Mdctnet
