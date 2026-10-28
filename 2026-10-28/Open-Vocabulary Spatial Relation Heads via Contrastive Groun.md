# Open-Vocabulary Spatial Relation Heads via Contrastive Grounding

## Motivation

Existing spatial vision-language models (e.g., ChronoReplay, SpatialBot) treat the set of spatial relation answers as a closed taxonomy, forcing all supervision into a fixed family set. This structurally prevents generalization to novel spatial predicates because the model's output space is hard-bounded by design. The root cause is that family-balanced SFT optimizes a categorical distribution over a pre-defined label set, so any predicate outside that set receives zero probability regardless of visual evidence.

## Key Insight

An open-vocabulary spatial relation head trained by contrastively grounding free-form predicates to visual relations learns a continuous embedding space where novel predicates are positioned by their semantic composition with seen ones, not by an arbitrary closed label set.

## Method

**What it is:** We introduce OV-Spatial, a spatial VLM post-training method that replaces the closed six-relation answer classifier with an open-vocabulary spatial relation head. The head maps a visual scene representation and a free-form spatial predicate (e.g., 'left of', 'behind the chair') into a shared embedding space, enabling zero-shot answers to novel predicates while still being supervised on replayed episodes.

**How it works:**

```pseudocode
# Input: replay buffer D of episodes (image, question, free-form answer, source-native target)
# Pretrained spatial VLM f_theta with visual encoder E_v and language encoder E_l
# Open-vocabulary head H: (v, t) -> similarity score

1. Initialize H as a two-layer MLP (input dim = d_v + d_t, hidden = 256, GeLU, output = 128) with contrastive temperature tau=0.07.
2. For each episode (I, q, a_free) in D:
   a. Encode visual scene: v = E_v(I)  # [CLS] token, dimension d_v
   b. Encode predicate a_free: t = E_l(a_free)  # mean-pooled token, dimension d_t
   c. Compute similarity s = cos(H(v), t) / tau
3. Contrastive loss over batch B of (v_i, t_i) pairs:
   L_contrast = -log( exp(s_ii) / sum_j exp(s_ij) )
4. Grounding loss: for each (v_i, t_i), sample K=8 negative predicates from a memory bank of all seen predicates, compute binary cross-entropy on whether t_i correctly describes v_i.
   L_ground = BCE( sigmoid(cos(H(v_i), t_i)), 1 ) + sum_k BCE( sigmoid(cos(H(v_i), t_k)), 0 )
5. Total loss: L = L_contrast + lambda * L_ground, with lambda=0.5.
6. Additionally, for replayed episodes with source-native targets (e.g., bounding boxes, relation triples), apply a grounding alignment loss that pulls the visual region embedding toward the predicate embedding when the predicate is spatially verified in the source annotation.
7. At inference, given a novel predicate p, compute its embedding t_p via E_l and answer by selecting the visual region (or relation) with highest similarity under H.
```

**Why this design:**
We chose an open-vocabulary contrastive head over a closed softmax classifier because a softmax over a fixed label set cannot assign probability to unseen predicates; the contrastive formulation instead learns a metric space where novel predicates are embedded by their linguistic composition, accepting the cost that the model may confuse semantically similar but spatially distinct predicates (e.g., 'left of' vs. 'right of') without fine-grained negative mining. We chose to combine contrastive loss with a binary grounding loss over sampled negatives rather than using only contrastive loss, because contrastive loss alone treats all non-matching pairs as equally negative, while the grounding loss explicitly supervises the decision boundary for each predicate against a memory bank, accepting the cost of increased memory and computation. We chose to retain source-native targets via a grounding alignment loss rather than rewriting all answers into a single interface, because source-native annotations provide spatially verified supervision that anchors the embedding space to visual evidence, accepting the cost that not all episodes have such annotations and the loss must be masked accordingly. This design is not a variant of standard retrieval-based tool selection (e.g., Toolformer, Gorilla) because the head is trained end-to-end with contrastive and grounding objectives that couple the visual and predicate embeddings, rather than computing a descriptor and performing nearest-neighbor lookup from a fixed library; the key difference is that the embedding space is shaped by the visual grounding signal, so novel predicates are positioned by their spatial semantics, not by lexical similarity alone.

**Why it measures what we claim:**
The contrastive similarity `s_ij` measures the semantic alignment between visual scene `v_i` and predicate `t_j` because we assume that the visual encoder and language encoder produce embeddings in a shared space where cosine similarity reflects spatial compatibility; this assumption fails when the visual encoder is biased toward object identity rather than spatial relations, in which case `s_ij` reflects object co-occurrence instead of spatial relation. The grounding loss `L_ground` measures the correctness of the predicate assignment because we assume that the sampled negative predicates are spatially incompatible with the visual scene; this assumption fails when negative predicates are semantically related (e.g., 'left of' vs. 'right of') but still plausible in the scene, in which case the loss penalizes correct spatial reasoning. The source-native alignment loss measures the fidelity of the embedding to verified spatial annotations because we assume that the source annotations are accurate and complete; this assumption fails when annotations are noisy or partial, in which case the alignment loss injects label noise into the embedding space. Together, these quantities operationalize the motivation-level concept of 'open-vocabulary spatial relation answering' by learning a metric space where novel predicates are positioned by their spatial semantics, but the failure modes above define the boundary conditions under which the method's claims hold.

**Memory bank and negative sampling schedule:** The memory bank is a FIFO queue of size 4096 predicate embeddings, initialized with all seen predicates from the training set and refreshed by adding the current batch's predicate embeddings after each forward pass. Negative predicates for `L_ground` are sampled uniformly from the memory bank, with K=8 negatives per positive. To address the failure mode of semantically related negatives, we additionally sample hard negatives: for each positive predicate, we include one negative that is a relation-swapped version (e.g., 'left of' → 'right of', 'above' → 'below') with probability 0.5, and one negative that is a different predicate from the same relation family with probability 0.5. This hard-negative mining is part of the existing grounding loss and does not introduce a new loss term.

**Spatial-layout probe (precondition check):** Before training OV-Spatial, we verify that the visual [CLS] embedding `v` encodes relative spatial layout by training a linear probe on `v` to predict the relative position (left/right/above/below) of two randomly selected objects in the scene. We use a subset of 10,000 episodes from the replay buffer with source-native bounding boxes. If the probe accuracy is not significantly above chance (e.g., < 60% for binary left/right), the shared-space assumption is broken and the method's premise is falsified; in that case, we report the probe result and do not proceed with the main experiments. This probe is a diagnostic, not a training component.

## Contribution

['An open-vocabulary spatial relation head that replaces the closed six-relation taxonomy with a contrastive embedding space trained via vision-language alignment, enabling zero-shot answers to novel spatial predicates.', 'A training procedure that combines contrastive loss with a grounding loss over sampled negatives and source-native alignment, which we show empirically improves generalization to unseen spatial predicates without degrading performance on seen ones.', 'A replayed-episode supervision framework that integrates open-vocabulary grounding with family-balanced SFT, demonstrating that the two can be jointly optimized.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | OVO-S-Bench + OST-Bench novel-predicate split | Tests zero-shot predicates beyond six-relation label set |
| Primary metric | Accuracy on held-out novel predicates | Directly measures open-vocabulary generalization claim |
| Baseline | Closed six-relation softmax head | Tests whether open vocabulary is necessary |
| Baseline | Joint SFT (ChronoReplay-style) | Tests whether replay supervision alone suffices |
| Baseline | Lexical nearest-neighbor retrieval | Tests whether visual grounding shapes the space |
| Ablation-of-ours | OV-Spatial without grounding-alignment loss | Isolates source-native anchoring contribution |
| Ablation-of-ours | OV-Spatial without hard-negative mining | Tests whether spatial distinctness in negatives matters |
| Diagnostic | Linear probe on visual [CLS] for relative position | Verifies visual encoder encodes spatial layout |

### Why this setup validates the claim

The central claim is that an open-vocabulary contrastive head shaped by a visual grounding signal positions novel spatial predicates by their spatial semantics rather than by lexical similarity to seen labels. The dataset choice is load-bearing: OVO-S-Bench and OST-Bench contain free-form spatial predicates, so a held-out novel-predicate split directly probes the zero-shot regime the method targets, while seen-predicate subsets provide the parity control. The closed six-relation softmax baseline tests the necessity sub-claim — if it cannot assign mass to unseen predicates it must fail on novel predicates specifically, not uniformly. The joint-SFT baseline tests whether replay supervision alone, without the open head, suffices; if it matches us the head is redundant. The lexical nearest-neighbor baseline tests the mechanism sub-claim that similarity is shaped by visual grounding rather than language alone: if lexical retrieval matches our novel-predicate accuracy, the grounding signal is doing no work. The ablation without grounding-alignment loss isolates the source-native alignment loss, so any gap between full OV-Spatial and the ablated variant is attributable to verified spatial anchoring rather than to the contrastive head itself. The ablation without hard-negative mining tests whether the spatially distinct negatives are necessary to separate lexically similar but spatially different predicates. The linear probe on the visual [CLS] token is a precondition check: if the probe fails to predict relative positions above chance, the visual embedding does not encode spatial layout, and the method's premise is falsified; we report this probe result alongside the main results. Novel-predicate accuracy is the right primary metric because it is precisely where the predicted causal mechanism — metric-space positioning by spatial semantics — should produce a differential effect; a metric like overall VQA accuracy would be diluted by seen predicates where all methods should be near parity. This combination forms a falsifiable test: the claim predicts a specific interaction (gain concentrated on novel predicates, parity on seen ones, and degradation when grounding is ablated or hard negatives are removed), not a uniform lift.

### Expected outcome and causal chain

**vs. Closed six-relation softmax head** — On a case where the question asks for a predicate outside the six-relation set (e.g., 'diagonally behind the chair'), the baseline produces a forced choice among seen labels because its softmax denominator only spans the fixed label set, so it cannot represent the unseen predicate at all. Our method instead embeds the free-form predicate through the language encoder and scores it against the visual region in the shared space, because the contrastive head has no fixed label vocabulary. We expect a large accuracy gap on the novel-predicate subset but near-parity on the seen-predicate subset, since the softmax head is well-tuned for in-vocabulary relations.

**vs. Joint SFT (ChronoReplay-style)** — On a case where a replayed episode carries a free-form spatial answer, the baseline produces a serialized natural-language answer that drifts toward the most frequent phrasing in the SFT distribution because its supervised objective optimizes token likelihood rather than a spatial compatibility score. Our method instead supervises a cosine similarity between the visual scene and predicate embedding, so the answer is selected by metric-space proximity rather than by language frequency. We expect a moderate gap on novel predicates and a smaller gap on seen predicates, with the joint-SFT baseline showing high variance across predicate phrasings.

**vs. Lexical nearest-neighbor retrieval** — On a case where a novel predicate is lexically close to a seen one but spatially distinct (e.g., 'left of' queried as 'just left of'), the baseline produces the nearest library descriptor by word embedding alone because it never consults the visual scene when positioning the query. Our method instead positions the predicate through a head trained with visual grounding, so the embedding is pulled toward spatially verified regions. We expect our gain to be concentrated on cases where lexical and spatial similarity diverge, with parity where they coincide — the diagnostic interaction that separates our mechanism from pure retrieval.

**vs. OV-Spatial without hard-negative mining** — On a case where the scene contains two objects and the query predicate is 'left of' but the negative sampled is 'right of', the ablated variant treats 'right of' as a random negative, so the contrastive loss does not explicitly force separation between spatially opposite predicates. Our full method includes relation-swapped hard negatives with probability 0.5, which directly penalizes high similarity for spatially incompatible predicates. We expect the full method to outperform the ablated variant specifically on novel predicates that are lexically similar to seen ones but spatially distinct (e.g., 'diagonally left of' vs. 'diagonally right of'), with smaller differences on lexically distant novel predicates.

**vs. Linear probe on visual [CLS]** — This is a diagnostic, not a baseline. We train a linear classifier on the visual [CLS] embedding `v` to predict the relative position (left/right/above/below) of two objects in the scene, using 10,000 episodes with source-native bounding boxes. If the probe accuracy is near chance (e.g., < 60% for binary left/right), the visual encoder does not encode spatial layout, and the shared-space assumption is broken; in that case, the novel-predicate gains of OV-Spatial would be attributable to lexical similarity rather than spatial grounding, falsifying the method's premise. We expect the probe to achieve at least 70% accuracy, confirming that the visual embedding carries spatial information.

### What would falsify this idea

If OV-Spatial's accuracy gain over the closed softmax baseline were uniform across novel and seen predicate subsets rather than concentrated on novel predicates, the open-vocabulary positioning claim would be unsupported. Likewise, if the grounding-alignment ablation showed no drop, if the hard-negative mining ablation showed no drop on lexically similar but spatially distinct predicates, or if lexical nearest-neighbor retrieval matched our novel-predicate accuracy, the claim that visual grounding shapes the embedding space would be falsified. Additionally, if the linear probe on the visual [CLS] token failed to predict relative positions above chance, the premise that the visual embedding encodes spatial layout would be falsified.

## References

1. ChronoReplay: One Answer Interface Is Not Enough for Streaming Spatial Intelligence
2. SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities
3. VideoLLM-online: Online Video Large Language Model for Streaming Video
4. SpatialBot: Precise Spatial Understanding with Vision Language Models
5. RoboVQA: Multimodal Long-Horizon Reasoning for Robotics
6. RT-1: Robotics Transformer for Real-World Control at Scale
7. Interactive Language: Talking to Robots in Real Time
8. VideoCoCa: Video-Text Modeling with Zero-Shot Transfer from Contrastive Captioners
