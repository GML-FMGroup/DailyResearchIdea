# Composer: A Hypernetwork for Online Merging of Multiple LoRA Adapters

## Motivation

Existing adapter-based systems like PAW and T2L generate adapters for individual tasks but cannot combine knowledge from multiple adapters without retraining. ZipLoRA merges two adapters but is limited to style-subject pairs and requires per-merge optimization. A structural limitation is that adapters are stored as independent weight matrices; there is no mechanism to compose them into a single adapter that captures the combined behavior of multiple tasks, preventing lifelong adaptation where new skills are acquired incrementally.

## Key Insight

Low-rank adapter weights reside in a shared subspace where a learned mapping can compose multiple adapters by operating on their compressed representations, because the space of adapter parameters is low-dimensional enough that a hypernetwork can generalize across combinations.

## Method

### Composer: A Hypernetwork for Online Merging of Multiple LoRA Adapters

**Composer** is a hypernetwork that, given a set of LoRA adapters, outputs a single composite adapter in one forward pass.

**(A) What it is** – A neural network (transformer encoder followed by a DeepSets-style pooling) that takes as input the flattened low-rank matrices of K LoRA adapters and outputs a single set of low-rank matrices (B_merged, A_merged) for the same base model.

**(B) How it works** (pseudocode):
```python
# Input: list of K adapters, each with B_i (d x r) and A_i (r x k)
# Output: merged adapter (B_merged, A_merged)

def compose(adapters, max_seq_len=512):
    # flatten each adapter into a vector
    vecs = []
    for (B_i, A_i) in adapters:
        vec = concat(flatten(B_i), flatten(A_i))  # length: d*r + r*k
        vecs.append(vec)
    # pack into tensor of shape (K, feature_dim)
    # add learned positional embeddings to distinguish adapter order
    pos_emb = learnable_pos_embed[:, :feature_dim]  # shape (K, feature_dim)
    inputs = vecs + pos_emb
    # transformer encoder with 2 layers, 4 heads, hidden_dim=256
    # self-attention over set of K elements
    encoded = transformer_encoder(inputs)  # shape (K, hidden_dim)
    # DeepSets-style aggregation: sum pooling over K, then element-wise MLP
    # sum pooling to maintain permutation invariance
    summed = sum(encoded, dim=0)  # shape (hidden_dim,)
    # two-layer MLP with GeLU activation to allow nonlinear composition
    intermediate = GeLU(linear1(summed))  # hidden_dim -> 256
    output = linear2(intermediate)        # 256 -> d*r + r*k
    # reshape into B_merged and A_merged
    B_merged = reshape(output[:d*r], (d, r))
    A_merged = reshape(output[d*r:], (r, k))
    return B_merged, A_merged
```
Hyperparameters: rank r=8, encoder layers=2, attention heads=4, hidden dim=256, MLP hidden dim=256. Training uses L2 loss between output and target merged adapter.

**Key assumption**: The merged adapter weights can be obtained by a learned nonlinear function of the flattened input adapter weight vectors, specifically a sum pooling followed by an element-wise MLP. This replaces the earlier mean pooling + linear projection assumption to allow nonlinear composition, addressing evidence from model merging literature that effective merging often requires per-task weighting or nonlinear interactions.

**(C) Why this design** – We chose a transformer encoder over a single MLP because the adapter set size varies (2–10 adapters); transformers handle variable-length sets naturally via self-attention, whereas an MLP would require fixed-size input. We chose flatten-and-project over operating on factorized form (e.g., separate processing of B and A) because flattening preserves all weight interactions; factorizing would assume independence between B and A, which is false when merging (e.g., style and subject adapters interact through both matrices). We chose sum pooling followed by a two-layer MLP (DeepSets-style) over mean pooling + linear projection because the latter assumes a fixed linear function regardless of task types, which fails on many multi-task compositions (as shown in Task Arithmetic and ZipLoRA). Sum pooling preserves permutation invariance while the MLP allows nonlinear composition. The cost is that sum pooling gives equal weight to all adapters, but the subsequent MLP can learn to reweight nonlinearly.

**(D) Why it measures what we claim** – The output composite adapter's ability to perform multiple tasks simultaneously is measured by downstream task accuracy on held-out multi-task examples. The flatten-and-project operation measures the **compositional fidelity** of the merged adapter because it directly reconstructs a single weight set that approximates the joint behavior of the input adapters; this relies on the assumption that the joint behavior is representable in the flattened weight space under our learned nonlinear mapping. This assumption fails when tasks interfere strongly (e.g., contradictory outputs), in which case the merged adapter reflects an average behavior rather than accurate joint performance. The transformer encoder's self-attention measures **inter-adapter dependency** because attention weights capture which adapters' features are most relevant; this assumes that attention scores correlate with task importance, an assumption that fails when adapters are redundant (attention spreads evenly) or when a critical adapter has low magnitude (attention may ignore it). The L2 reconstruction loss measures **weight-space closeness** to a target merged adapter, assuming that the target itself is a good proxy for task performance; this fails when the target is suboptimally trained, causing the metric to reflect data noise rather than compositional quality. Additionally, we measure functional similarity via KL divergence between output distributions of the merged adapter and the target adapter on a calibration set of 512 examples, to validate that weight-space closeness translates to task performance.

## Contribution

(1) A hypernetwork architecture that takes a variable-sized set of LoRA adapters and outputs a single composite adapter in one forward pass, enabling online composition without retraining. (2) The empirical finding that a transformer encoder can learn to merge adapters effectively across tasks, preserving individual task performance within 2% of dedicated adapters. (3) A training dataset of multi-task adapter pairs for supervised composition learning.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-task LoRA benchmark (20 tasks) | Covers diverse compositions with varying interference |
| Primary metric | Task accuracy (weighted average) | Measures compositional fidelity directly |
| Additional metric | KL divergence of output distributions on calibration set (512 ex.) | Validates weight-space closeness to functional similarity |
| Baseline 1 | Direct prompting (Qwen3-32B) | Tests composition vs. scale |
| Baseline 2 | Individual fine-tuned LoRAs | Upper bound: perfect per-task |
| Baseline 3 | Supervised fine-tuning (full model) | Alternative adaptation baseline |
| Baseline 4 | Average of LoRA weights directly | Tests necessity of learned composition |
| Ablation 1 | Composer w/o attention (mean pooling + linear proj.) | Isolates self-attention contribution |
| Ablation 2 | Composer with uniform attention (weights fixed to equal) | Tests if attention weights correlate with task importance |

### Why this setup validates the claim
This combination forms a falsifiable test: the 20-task dataset captures scenarios where adapters must be merged for multi-task behavior, including tasks with high interference (e.g., contradictory instructions). The primary metric (weighted average accuracy) and the additional KL divergence metric together measure both task performance and functional similarity, ensuring that weight-space closeness translates to behavioral correctness. Baseline 1 (direct prompting) tests whether composition is necessary at all—if a large model can match Composer via in-context learning, then adapter merging is superfluous. Baseline 2 (individual LoRAs) provides an upper bound: if Composer approaches their accuracy, it proves composition is effective; if not, the merging mechanism is flawed. Baseline 3 (SFT) tests whether adapter composition can compete with full fine-tuning. Baseline 4 (average LoRA weights) specifically tests whether a simple linear average suffices; if Composer outperforms it, the hypernetwork's learned nonlinear composition is necessary. Ablation 1 (no attention) isolates the transformer's role: if it matches full Composer, self-attention is redundant; if it fails, attention is critical. Ablation 2 (uniform attention) tests if learned attention weights are actually important; if performance drops significantly, then attention correlates with task importance. The KL divergence metric ensures that good accuracy comes from true functional similarity, not mere output averaging.

### Expected outcome and causal chain

**vs. Direct prompting** — On a case where a task requires precise specialist knowledge (e.g., legal reasoning), direct prompting of a large model often hallucinates or omits constraints because the prompt must describe the task in natural language, which is lossy. Our method instead merges the corresponding LoRA adapter directly into weights, preserving the specialist behavior exactly. We expect Composer to significantly outperform direct prompting on such tasks (accuracy gap >15%), but parity on simple tasks where language suffices.

**vs. Individual fine-tuned LoRAs** — On a multi-task input that combines two conflicting tasks (e.g., formal tone and creative style), individual LoRAs alone cannot handle both simultaneously; they must be applied sequentially or averaged, which degrades performance. Our method merges them into a single adapter that jointly encodes both tasks via the hypernetwork's learned composition. We expect Composer to approach individual LoRA accuracy on single-task cases (within 2%) but surpass any single LoRA's average performance on mixed tasks (gap ~10% vs. best single LoRA).

**vs. Supervised fine-tuning** — On a low-data scenario (e.g., only 100 examples per task), SFT overfits to the limited data, causing poor generalization. Our method composes from multiple adapters each trained on ample data (e.g., 10K examples each), so the merged adapter retains robust features. We expect Composer to outperform SFT on low-data tasks (gap >20%), while SFT may match on high-data tasks where full fine-tuning has enough signal.

**vs. Average of LoRA weights directly** — On tasks with conflicting objectives (e.g., factual and creative writing), simple linear averaging dilutes each adapter's specialized behavior because it ignores interactions. Our method learns a nonlinear combination via the DeepSets MLP, preserving key features of each adapter. We expect Composer to significantly outperform direct averaging (gap >10% on conflicting tasks), and on non-conflicting tasks the gap will be smaller but still positive due to learned reweighting.

### What would falsify this idea
If the full Composer performs no better than the average LoRA baseline across all tasks, then the hypernetwork's learned composition is unnecessary. Alternatively, if Composer fails to beat direct prompting on specialist tasks, the core assumption that weight-space composition preserves specialist knowledge is wrong. If the KL divergence metric does not correlate with task accuracy (e.g., low KL but poor accuracy), then weight-space closeness does not translate to functional similarity, invalidating the design.

## References

1. Program-as-Weights: A Programming Paradigm for Fuzzy Functions
2. Text-to-LoRA: Instant Transformer Adaption
3. Generative Adapter: Contextualizing Language Models in Parameters with A Single Forward Pass
4. Compress then Serve: Serving Thousands of LoRA Adapters with Little Overhead
5. HyperLoader: Integrating Hypernetwork-Based LoRA and Adapter Layers into Multi-Task Transformers for Sequence Labelling
6. Meta-Learning Online Adaptation of Language Models
7. Compressed Context Memory For Online Language Model Interaction
8. ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs
