# Compositional Closure via Invariant Augmentation for Audio-Visual Reasoning and Fusion

## Motivation

Existing methods like TinyLLaVA-Video-R1 rely on holistic reward signals that fail to provide step-level credit, and AVQACL++ lacks invariance under task shifts. Both depend on monolithic supervision that overlooks the compositional interactions between reasoning steps and cross-modal fusion operations. This structural absence of compositional closure leads to brittle performance when task composition changes, as neither approach enforces that the output remains consistent under reordering of independent operations.

## Key Insight

Compositional closure can be self-supervised by permuting commutative operations in training traces, creating a contrastive signal that forces the model to learn order-invariant representations.

## Method

### Compositional Closure via Invariant Augmentation (CCIA)

**What it is**: CCIA is a training framework that enforces compositional closure in multi-step video reasoning and cross-modal fusion by augmenting operation sequences and applying a self-supervised contrastive loss to ensure invariance under commutative rearrangements.

**How it works**:
1. **Learn commutativity classifier**: Instead of a fixed manual taxonomy, we train a binary commutativity classifier on a held-out validation set. For each operation type pair, we randomly swap their occurrences in teacher-forced traces and compute the fraction of swaps that leave the gold answer unchanged; if that fraction exceeds 0.9, we label the pair as commutative. Operation embeddings are extracted from the model's operation predictor (dimension 64) and concatenated as input to a 2-layer MLP (hidden=128, ReLU) that outputs a commutativity score. Only pairs with score > 0.5 are considered commutative for augmentation. This classifier is trained once before main training and frozen afterward. **Load-bearing assumption**: We assume the classifier accurately captures true commutativity; misclassifications may cause the contrastive loss to penalize valid order-dependent behavior, reducing accuracy.
2. During training, the model autoregressively generates an operation token sequence (trace), then produces an answer from the trace and visual features. A trace embedding is computed as the mean of operation token hidden states (dimension 512).
3. For each generated trace, create augmented traces by swapping pairs of commutative operations (identified via the learned classifier, score > 0.5). For each augmented trace, the model (with teacher-forced operation sequence) re-computes answer logits and trace embedding.
4. Apply a contrastive loss (InfoNCE, temperature τ=0.07) to maximize cosine similarity between original and augmented trace embeddings while minimizing similarity to other traces in the batch. Also, enforce consistency between answer logits via symmetric KL divergence (with a small epsilon 1e-8 to avoid log(0)).
5. Before training, we validate each commutative pair by randomly swapping them on a batch of 512 training traces and computing the average KL divergence between answer distributions. If the average KL > 0.1, we exclude that pair from augmentation. This ensures the model only enforces invariance where it is empirically justified.
6. Total loss: L_task + λ_contrast * L_contrast + λ_kl * L_kl, with λ_contrast=0.1, λ_kl=0.05. L_task is cross-entropy on final answer.

**Why this design**: We chose contrastive learning over direct regression because it naturally handles multiple augmentations without a fixed target and prevents collapse to a trivial constant. Using a pre-defined taxonomy for commutativity avoids learning spurious correlations, but it limits generalizability to new operations and requires manual effort; our learned classifier mitigates this by adapting to data while still providing reliable invariance enforcement. We enforce invariance on both trace embeddings and answer logits because intermediate representations are more sensitive to ordering perturbations, while final outputs capture task-level consistency; however, this doubles computational cost due to multiple forward passes per training step. Cosine similarity is scale-invariant and avoids mode collapse, but it may ignore small but important differences compared to Euclidean distance; we accept this to maintain stable gradients.

**Why it measures what we claim**: The contrastive loss on trace embeddings measures **representational invariance under operation reordering**, which operationalizes compositional closure by requiring the model to produce similar latent codes regardless of commutative operation order. This equivalence holds under the assumption that the commutativity classification is correct and that the trace embedding is a sufficient statistic for the final answer; this assumption fails when non-commutative operations are misclassified, in which case the loss penalizes legitimate order-dependent behavior, reducing accuracy. The KL divergence on answer distributions measures **output-level invariance**, directly reflecting whether the final answer is unchanged under permutation; it assumes the model's answer distribution is well-calibrated, but if the model is overconfident, the divergence may be small even when answers differ, yielding false positive invariance. Our pre-training validation (step 5) checks that swapping commutative pairs does not significantly change answer distributions: if the average KL exceeds 0.1, we exclude the pair, avoiding the failure mode of misclassified commutativity. Together, the two losses provide a robust proxy for compositional closure, with failure modes primarily arising from taxonomy errors or miscalibration.

## Contribution

(1) A training framework (CCIA) that enforces compositional closure through operation-level data augmentation and contrastive invariance losses, applicable to multi-step video reasoning and cross-modal fusion. (2) A taxonomy of commutative atomic operations for video reasoning and fusion, enabling principled augmentation without manual annotation of closure. (3) Introduction of a self-supervised invariant learning objective that does not require ground-truth compositional labels.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|----------|
| Dataset | Split-AVQA | Tests compositional reasoning in AVQA |
| Primary metric | Answer accuracy | Directly measures task performance |
| Baseline 1 | Standard AVQA model | No invariance enforcement |
| Baseline 2 | AVQA + random permutation | Augments but ignores commutativity |
| Baseline 3 | AVQA + direct regression | Alternative to contrastive loss |
| Ablation-of-ours | Ours w/o contrastive loss | Isolates effect of trace invariance |

### Why this setup validates the claim
This experimental design tests whether enforcing compositional closure via invariant augmentations actually improves multi-step AVQA reasoning. Split-AVQA contains questions requiring sequential operations (e.g., locate sound source then track object), making it ideal for evaluating order sensitivity. The standard baseline shows the default failure when commutative operations are reordered. The random permutation baseline tests whether arbitrary augmentations help, isolating the benefit of commutativity-aware changes. The direct regression baseline tests whether contrastive learning is necessary or a simpler alternative works. Our ablation removes the contrastive loss to isolate its contribution. Accuracy directly captures the final answer correctness, which is the ultimate test of invariance under commutative reordering. If our method works, it should yield higher accuracy specifically on questions where operation reordering is possible, while performing similarly on non-commutative ones. This provides a falsifiable prediction: improvement concentrated on commutative subsets.

### Expected outcome and causal chain

**vs. Standard AVQA model** — On a case where a question requires first locating a barking dog and then identifying its color, the standard model may fail if the operation order is swapped during inference because its latent representation is order-sensitive. Our method, by explicitly enforcing invariant trace embeddings under commutative swaps, would produce the same answer regardless of order. We expect a noticeable accuracy gap (e.g., 5-10% relative improvement) on questions with multiple commutative steps, but parity on single-step or strongly ordered questions.

**vs. AVQA + random permutation** — On the same example, random permutation might swap non-commutative operations (e.g., order-dependent fusion), degrading performance. Our method only swaps truly commutative operations, preserving correct order dependencies. Thus, we expect our method to outperform random augmentation on all subsets, with a larger margin on questions where commutativity is common (accuracy gap 3-7% relative).

**vs. AVQA + direct regression** — Direct regression (e.g., predicting a single embedding) may collapse to a trivial constant, losing discriminability between different operations. Our contrastive loss prevents collapse by pulling together embeddings of commutative variants while pushing apart unrelated traces. Thus, we expect our method to achieve higher accuracy on diverse operation sequences, with a gap of 4-8% relative, and better separation on validation trace embeddings.

### What would falsify this idea
If our method shows uniform accuracy gains across all subsets (including non-commutative ones) compared to the standard baseline, rather than a concentrated improvement on commutative-heavy questions, then the central claim that compositional closure is the driving factor would be invalid. Alternatively, if the random permutation baseline matches our performance, it would suggest commutativity knowledge is unnecessary.

## References

1. TinyLLaVA-Video-R1: Towards Smaller LMMs for Video Reasoning
2. LLaVA-Video: Video Instruction Tuning With Synthetic Data
3. Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding
4. Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models
5. EVA: Exploring the Limits of Masked Visual Representation Learning at Scale
6. Scaling Instruction-Finetuned Language Models
7. Expanding Language-Image Pretrained Models for General Video Recognition
8. AVQACL++: Toward a Robust Framework and Benchmark for Audio-Visual Question Answering Continual Learning
