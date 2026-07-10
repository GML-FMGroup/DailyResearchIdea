# Temporal Energy-Guided Adaptive Memory Access for Linear Recurrent Neural Networks

## Motivation

Existing linear RNNs with explicit memory, such as Sparse Delta Memory, assume a fixed sparsity pattern or window size for memory access, which is suboptimal when input dynamics vary. The temporal energy of hidden states—reflecting information rate—is not leveraged to dynamically adjust memory resources, causing either wasted compute on low-information segments or missing critical context in high-information bursts, limiting long-context modeling efficiency.

## Key Insight

The temporal energy of the hidden state, measured as the L2 norm of its change, provides a reliable signal to modulate memory access because it correlates with the instantaneous rate of new information to be stored or retrieved.

## Method

We propose **Temporal Energy Gated Memory (TEGM)** , which uses the instantaneous energy of the hidden state to adaptively configure memory access.

(A) **What it is**: TEGM is a memory module for linear RNNs that dynamically adjusts the number of active memory slots and the window size for attention over memory based on the temporal energy of the current hidden state. Input: hidden state \(h_t\), current memory matrix \(M_t\). Output: updated memory \(M_{t+1}\), output \(o_t\). The memory consists of a fixed-size array of key-value pairs with maximum capacity \(M = 128\). Keys and values are vectors of dimension \(d_{kv} = 64\) when the hidden state dimension \(d_{model} = 512\).

(B) **How it works** (pseudocode):
```python
def tegm_step(h_t, h_{t-1}, M_t, params):
    # 1. Compute temporal energy via learned network on local context
    context = stack([h_{t-2}, h_{t-1}, h_t])  # local window of 3 tokens
    energy_net = nn.Sequential(
        nn.Linear(3 * d_model, 64), nn.ReLU(),
        nn.Linear(64, 1)
    )
    energy = energy_net(context.flatten())  # scalar output
    gate = sigmoid(energy)  # gate in (0,1)
    
    # 2. Map energy to memory configuration
    num_slots = int(params.M * gate)  # adaptive number of slots (straight-through estimator)
    window_size = int(params.W * (1 - gate))  # adaptive window for local attention
    
    # 3. Sparse read: top-k attention over memory keys
    keys, vals = M_t.keys, M_t.vals
    attn_scores = h_t @ keys.T  # dot product attention
    top_k_indices = top_k(attn_scores, k=num_slots)  # uses Gumbel-top-k for gradients
    attn_weights = softmax(attn_scores[top_k_indices])
    read = attn_weights @ vals[top_k_indices]
    
    # 4. Adaptive write: new key-value pair via two linear projections
    new_key = nn.Linear(d_model, d_kv, bias=False)(h_t)  # key projection
    new_val = nn.Linear(d_model, d_kv, bias=False)(h_t)  # value projection
    # FIFO update within the adaptive window: replace oldest entry in active window
    M_{t+1} = fifo_update(M_t, new_key, new_val, window_size)
    
    # 5. Combine read with hidden state
    o_t = LayerNorm(h_t + read)  # skip connection + layer norm
    return o_t, M_{t+1}
```
Hyperparameters: \(M=128\) (maximum slots), \(W=32\) (maximum window size). The energy network has two hidden layers (first: 3*512=1536 → 64, ReLU; second: 64 → 1). All other parameters (key, value projections) are learned jointly with the backbone linear RNN.

(C) **Why this design**: We chose a sigmoid gate over a hard threshold because it provides differentiable control, accepting the cost that continuous values may not yield exact integer slot counts (we round during training with straight-through estimator). We chose a learned energy network over a fixed L2 norm to make the energy estimation robust to noise and slow drifts; the trade-off is added parameters (≈100K) and a small computational overhead. We chose top-k attention over softmax with full memory because it enforces sparsity and aligns with hardware-efficient sparse operations, but it requires a selection step that can be non-differentiable (we use Gumbel-top-k for gradients). These decisions prioritize adaptability and hardware efficiency over simplicity.

(D) **Why it measures what we claim**: The computational quantity `num_slots` measures `memory capacity assigned to current token` because it operationalizes the idea that high energy (large change) requires more memory to store new information; this assumption fails when energy is high due to noise rather than informative content, in which case the metric reflects noise amplitude rather than information. The quantity `window_size` measures `temporal locality of relevant context` because low energy (steady state) implies reliance on local patterns; this assumption fails when low-energy states still require long-range dependencies, in which case the metric incorrectly reduces window size. The energy output of the learned network measures `information rate` because we assume that large state changes (or their learned combinations) indicate new salient information; this assumption fails when the model's state dynamics do not align with input informativeness (e.g., gradual accumulation), in which case energy may underestimate information. To verify this, we include a synthetic calibration experiment where we inject known-information tokens into a sequence and measure the correlation between the learned energy and the injected information rate (see experiment section).

## Contribution

(1) A novel adaptive memory access mechanism for linear RNNs that dynamically adjusts sparsity and window size based on temporal energy of hidden states. (2) A principle demonstrating that temporal energy can serve as a lightweight proxy for information rate, enabling compute-quality trade-offs in long-context modeling. (3) An analysis of the conditions under which energy-based gating is effective versus when it fails.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Long-range Arena (LRA) | Tests long-context memory |
| Dataset | IMDB reviews with >2k tokens | Real-world long document classification |
| Synthetic dataset | Sequences with injected information bursts | Calibrate energy-information correlation |
| Primary metric | Accuracy on Pathfinder | Measures retrieval in noise |
| Primary metric | Accuracy on IMDB (>2k) | Practical long-context performance |
| Primary metric | Pearson correlation (energy vs. injected info rate) | Verifies energy measures information |
| Baseline | Gated DeltaNet | Dense memory baseline |
| Baseline | Mamba | State-of-the-art linear RNN |
| Baseline | Softmax Transformer | Full attention baseline |
| Baseline | Adaptive Computation Time (ACT) | Dynamic computation baseline |
| Baseline | Neural Turing Machine (NTM) with learnable addressing | Explicit memory baseline |
| Ablation-of-ours | TEGM w/o adaptive window | Isolates effect of energy gating |
| Ablation-of-ours | TEGM with fixed L2 norm (original gate) | Isolates effect of learned energy |

Hyperparameter search: We fix \(M=128\), \(W=32\) and only tune the energy network parameters (learning rate, hidden size) to reduce search space.

### Why this setup validates the claim

LRA's Pathfinder task requires retrieving a target pattern embedded in a long noisy sequence, which directly tests the claim that TEGM allocates memory slots based on temporal energy to capture informative transitions. Gated DeltaNet tests the contribution of adaptive sparse write versus dense storage; Mamba tests whether explicit gated memory outperforms implicit state compression; Transformer tests whether our method can match full attention's retrieval while being more efficient. ACT and NTM test whether energy-based gating provides unique advantages over other adaptive mechanisms. The synthetic dataset with known information bursts (random tokens injected at known timesteps) allows us to directly measure the correlation between the learned energy and the actual information rate; a high correlation (e.g., >0.7) would support the central assumption. IMDB (>2k tokens) tests generalization to realistic long-document tasks. The ablation with fixed L2 norm isolates the benefit of the learned energy network. Accuracy on Pathfinder and IMDB are the right metrics because they demand precise recall of sparse critical tokens, where our energy-based gating should yield a measurable advantage.

### Expected outcome and causal chain

**vs. Gated DeltaNet** — On a sequence with sudden high-information tokens after a long static period, Gated DeltaNet wastes memory slots on irrelevant patterns because it writes densely at every step. Our method detects the energy spike and allocates more slots, capturing the key events. We expect a 5-10% accuracy gain on sequences with variable information density.

**vs. Mamba** — On a task requiring cross-token binding over long distances, Mamba’s state compression loses detail because it updates via linear recurrence. Our method retains explicit slot memory, so on long-context retrieval we expect a 5-10% accuracy gain.

**vs. Softmax Transformer** — On a long sequence (e.g., 8k tokens), Transformer’s quadratic attention forgets early tokens due to windowing or memory limits. Our method adaptively expands window on high-energy transitions, improving recall on early critical tokens. We expect parity on short sequences but a 3-5% gap on sequences >2k tokens.

**vs. ACT** — ACT adjusts the number of computational steps per token, not memory allocation. Our method is complementary; we expect comparable or better efficiency on tasks with variable information density.

**vs. NTM** — NTM uses learnable addressing but does not incorporate temporal energy. NTM may be less efficient due to full memory writes; we expect TEGM to be faster and more accurate on long sequences.

**Synthetic experiment** — We expect Pearson correlation >0.7 between learned energy and injected information rate, validating that the network captures information content.

### What would falsify this idea

If improving memory allocation via energy gating yields uniform accuracy gains across all input segments rather than concentrated on high-delta transitions, or if the adaptive ablation performs as well as the full method, then the central claim that temporal energy drives memory configuration is false. Also, if the synthetic correlation is low (<0.5), the learned energy does not reliably measure information rate.

## References

1. Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity
2. Short window attention enables long-term memorization
3. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
4. Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling
5. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
6. Learning to (Learn at Test Time)
7. Meta-Learning Fast Weight Language Models
8. What learning algorithm is in-context learning? Investigations with linear models
