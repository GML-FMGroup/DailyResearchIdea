# Compositional Monoid Tokenization for Modality-Agnostic Representation

## Motivation

Existing multimodal representation learning methods, such as those surveyed in 'From Fragmentation to Fusion', fail to establish compositional closure under modality permutation because they treat each modality's tokenization independently (e.g., Chronos's per-series tokenization), preventing associative, order-invariant composition. This structural limitation means that representations cannot be reliably merged across modalities without losing information or introducing order-dependence, as the survey's compositionality capability remains aspirational and unaddressed by current joint embedding approaches.

## Key Insight

Embedding every modality token into a shared commutative monoid via a functor makes composition associative and order-invariant, so any permutation of modality tokens yields the same merged representation.

## Method

**What it is**: We introduce **MonoTok**, a modality-agnostic tokenizer that maps any modality's discrete symbols into a common commutative monoid (a free commutative monoid over a finite alphabet A with |A| = 50257) via a shared functor F, and composes tokens using the monoid's associative, order-invariant merge operation (multiset union). Input: a sequence of tokens from any modality (time-series patches, text subwords, image patches). Output: a single monoid element R (a multiset over A) representing the composed sequence. **Load-bearing assumption (stated explicitly)**: that discarding all token order and collapsing each modality to a bag-of-symbols multiset is a lossless-enough representation for the downstream cross-modal retrieval/composition tasks we evaluate; this assumption is what makes order-invariance by construction possible, and it is the assumption the experiment is designed to test (via the ordered-vs-shuffled gap and the order-sensitive forecasting probe).

**How it works**:
```pseudocode
# Step 1: Tokenization per modality (modality-specific front-end)
# Time series: split into non-overlapping patches of length P=16, quantize each patch to a codebook index using k-means (K=256, trained on 100k patches from the training split, 20 iterations).
# Text: byte-pair encoding with vocabulary size V=50257.
# Images: split into 16x16 patches, quantize each to a codebook index using k-means (K=1024, trained on 100k patches from the training split, 20 iterations).
# Each modality yields a sequence of integer symbols: s_1, s_2, ..., s_n.

# Step 2: Functor F: Symbol -> Monoid element
# Define a commutative monoid M = (S, ⊕) where S is the set of multisets over a finite alphabet A (|A| = 50257).
# The monoid operation ⊕ is multiset union: for multisets m1, m2, m1 ⊕ m2 = m1 ∪ m2 (with multiplicity).
# The functor F maps each symbol s to the singleton multiset {s}.
# F is shared across all modalities because symbols are drawn from the same alphabet A; modality-specific codebooks are mapped to disjoint subsets of A to avoid collisions:
#   time-series codes -> A_ts = {0, ..., 255}
#   image codes      -> A_img = {256, ..., 1279}
#   text codes       -> A_txt = {1280, ..., 50256}
# (These ranges are fixed; the non-disjoint ablation in the experiment overrides them to overlap.)

# Step 3: Composition via monoid operation
# Given token sequence s_1, ..., s_n, the composed representation is:
# R = F(s_1) ⊕ F(s_2) ⊕ ... ⊕ F(s_n)
# Because ⊕ is associative and commutative, R is invariant to the order of tokens.

# Step 4: Downstream use
# R is a multiset of symbols; to obtain a fixed-dimensional vector, apply a permutation-invariant pooling:
#   v = normalize( sum_{a in A} count_R(a) * E[a] )
# where E[a] is a learned embedding of dimension d=256 for each symbol a in A, count_R(a) is the multiplicity of a in R, and normalize is L2 normalization.
# This vector v is then used as input to a 4-layer Transformer encoder (hidden size 256, 4 attention heads, feed-forward size 1024, GeLU activation) for downstream retrieval.
```

**Why this design**: We chose a commutative monoid over a learned neural composition operator because commutativity is a structural property that guarantees order-invariance by construction, whereas a learned operator would require training to approximate commutativity and could fail under distribution shift; the cost is that the representation loses all sequential information, which we accept because the goal is modality-agnostic composition, not sequence modeling. We chose a shared functor F over modality-specific functors because a single functor ensures that tokens from different modalities are embedded into the same algebraic structure, enabling cross-modal composition; the trade-off is that modality-specific nuances may be lost, but this is mitigated by modality-specific tokenization front-ends that map into disjoint alphabet subsets. We chose multiset union as the monoid operation over string concatenation because it is inherently commutative and associative, and its simplicity allows exact invariance checks; the cost is that the representation grows with sequence length, but we handle this via pooling. A domain expert would NOT describe MonoTok as a variant of Chronos because Chronos tokenizes each time series independently and uses a language model for forecasting, whereas MonoTok replaces the tokenization with a categorical functor into a monoid, fundamentally changing the composition rule to be closed and order-invariant.

**Why it measures what we claim**: The computational quantity R (the multiset resulting from monoid composition) measures **compositional closure under modality permutation** because the named assumption that multiset union is commutative and associative makes R invariant to token order; this assumption fails when the alphabet A contains symbols that are not disjoint across modalities (e.g., if the same integer code is used for different meanings in different modalities), in which case R reflects spurious cross-modal collisions rather than true composition. The shared functor F measures **modality-agnostic embedding** because the assumption that all modality symbols are drawn from a common alphabet A ensures a single algebraic structure; this fails when modality-specific codebooks overlap in semantics but not in index, in which case F maps semantically similar tokens to different monoid elements, breaking comparability. The pooling step measures **fixed-dimensional representation** because the assumption that sum-pooling over multiset elements preserves all information needed for downstream tasks holds only if the downstream task is invariant to token multiplicity; this fails when the task requires counting occurrences, in which case the pooled vector loses that information.

## Contribution

['A novel modality-agnostic tokenization framework, MonoTok, that embeds tokens from any modality into a shared commutative monoid via a functor, guaranteeing associative and order-invariant composition.', 'A design principle showing that compositional closure under modality permutation can be achieved by construction using algebraic structures, rather than learned, and that this property is orthogonal to modality-specific tokenization choices.', 'A reproducible implementation recipe for mapping modality-specific codebooks into disjoint alphabet subsets and composing via multiset union, enabling exact invariance checks.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Cross-modal retrieval triples with shuffled-order variants (time series + text + image; 50k train / 5k val / 5k test triples) | Directly probes order-invariance and cross-modal composition |
| Primary metric | Retrieval accuracy gap: ordered vs. shuffled inputs (top-1 accuracy difference) | Isolates the order-invariance claim from raw accuracy |
| Baseline | Modality-specific tokenizer + Transformer (same 4-layer encoder, per-modality tokenizers, positional encodings) | Tests whether shared monoid functor beats per-modality pipelines |
| Baseline | Learned commutative-operator approximator (2-layer MLP with symmetric weight sharing, trained on the same data) | Tests whether exact algebraic commutativity beats learned approximation |
| Baseline | Concatenation-based fusion tokenizer (concatenate per-modality token sequences, then Transformer) | Tests whether monoid composition beats naive sequence fusion |
| Ablation-of-ours | MonoTok with shared (non-disjoint) alphabet A (all modalities use codes 0–50256, overlapping) | Tests whether disjoint alphabet subsets are load-bearing |
| Order-sensitive probe | Time-series forecasting (next-patch prediction) on the same data, measuring MSE | Quantifies information loss from discarding order |

### Why this setup validates the claim

The central claim is that a shared functor into a commutative monoid yields compositional closure under modality permutation while remaining modality-agnostic. The dataset must therefore expose both order sensitivity and cross-modal composition, which is why we use retrieval triples with paired ordered and shuffled variants: the shuffled variant is the diagnostic probe. The primary metric — the accuracy gap between ordered and shuffled inputs — measures order-invariance directly rather than conflating it with raw representational quality. The modality-specific tokenizer baseline tests whether the shared functor is actually necessary; the learned-commutative-operator baseline tests whether exact algebraic commutativity buys anything over a trained approximation; the concatenation baseline tests whether the monoid merge beats naive fusion. The ablation with a non-disjoint alphabet tests whether the disjoint-subset design is what prevents spurious collisions. The order-sensitive forecasting probe quantifies the information loss from discarding order, making the load-bearing assumption explicit and testable. Together these form a falsifiable grid: each baseline isolates one sub-claim, and the metric is chosen precisely because it should collapse to near-zero gap for our method only if the algebraic argument holds.

### Expected outcome and causal chain

**vs. Modality-specific tokenizer + Transformer** — On a case where the same time-series patch sequence and its shuffled counterpart are paired with one text query, the baseline produces different embeddings for the two orders because its positional encoding and attention mechanism encode sequence order explicitly. Our method instead composes both sequences into the identical multiset R because multiset union is commutative and associative, so we expect a near-zero ordered-vs-shuffled retrieval gap for MonoTok versus a substantial gap for the baseline, with parity on the unshuffled subset.

**vs. Learned commutative-operator approximator** — On a case where the input distribution shifts (e.g., longer sequences or unseen symbol multiplicities), the baseline produces order-dependent outputs because its learned operator only approximates commutativity on the training distribution and drifts off-manifold. Our method instead remains exactly order-invariant by construction because the merge is multiset union, so we expect the baseline's ordered-vs-shuffled gap to grow with sequence length and distribution shift while MonoTok's gap stays flat near zero.

**vs. Concatenation-based fusion tokenizer** — On a case where two modalities contribute tokens in interleaved order, the baseline produces different fused representations depending on interleaving because concatenation preserves position, so cross-modal composition is order-sensitive and modality-order-biased. Our method instead collapses interleavings to the same multiset because the functor maps both modalities into one commutative monoid, so we expect MonoTok to show a clear advantage on interleaved-order retrieval pairs while matching the baseline on fixed-order pairs.

**vs. Order-sensitive forecasting probe** — On next-patch prediction, MonoTok's pooled multiset representation is expected to underperform the modality-specific tokenizer + Transformer baseline (higher MSE) because the pooling discards sequential order, which is required for forecasting. This quantifies the information loss from the load-bearing assumption and bounds where MonoTok is applicable.

### What would falsify this idea

If MonoTok's ordered-vs-shuffled retrieval gap is not near zero — or if its advantage over the learned-commutative-operator baseline is uniform across sequence lengths rather than concentrated where commutativity should matter — the central algebraic claim is wrong. Likewise, if the non-disjoint-alphabet ablation performs identically to the full method, the disjoint-subset design is not load-bearing and the modality-agnostic embedding claim is unsupported. Additionally, if MonoTok matches or exceeds the baseline on the order-sensitive forecasting probe, the claimed information loss would be falsified.

## References

1. From Fragmentation to Fusion: A Survey and Roadmap for Omnimodal Embeddings
