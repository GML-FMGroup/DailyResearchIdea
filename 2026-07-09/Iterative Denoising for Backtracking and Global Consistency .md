# Iterative Denoising for Backtracking and Global Consistency in Structure Token Sequences

## Motivation

Existing autoregressive models for structure-property prediction, such as SciReasoner, generate structural tokens left-to-right without revision, preventing correction of early errors and enforcing only local consistency. This monotonic decoding accumulates errors and fails to enforce global geometric constraints, becoming the primary limitation as the field converges on discrete structural tokens. We address this bottleneck by reframing generation as a denoising process that naturally enables iterative refinement.

## Key Insight

By training a single transformer to reverse a Markov corruption process that progressively masks structural tokens, the model learns to infer missing tokens from global context, enabling iterative refinement that inherently allows backtracking and enforces global consistency without separate modules.

## Method

## (A) What it is
**Global Structure Denoising (GSD)** – a transformer-based model that, given a partially masked structural token sequence, predicts the original tokens. The model is trained via a denoising objective and used iteratively to refine generated sequences, enabling revision and global consistency.

## (B) How it works
```pseudocode
# Architecture: 12-layer transformer, hidden=768, 12 heads, max_seq=512, vocab=5000
# Training
for each sequence S of T tokens:
    # Autoregressive pre-training (optional)
    train autoregressive model on S (e.g., SciReasoner)  # 50k steps, batch=64, lr=1e-4, AdamW
    # Denoising fine-tuning  # 20k steps, batch=64, lr=1e-5, AdamW
    sample noise level t ~ Uniform(0,1)
    compute mask rate m = sigmoid(3*(t - 0.5))  # schedule: m in (0,1)
    # Structured corruption: mask contiguous segments
    create corrupted sequence X = S.copy()
    while mask_fraction < m:
        segment_length = geometric(p=0.3) + 1  # at least 1
        start = random(0, T - segment_length)
        mask tokens X[start:start+segment_length]
        update mask_fraction
    logits = transformer(X, position_embeddings)
    loss = cross_entropy(logits, S, ignore_non_masked)
    update weights

# Inference (iterative refinement)
input: primary sequence (or None)
if primary sequence is None:
    generate initial sequence auto-regressively
else:
    use given sequence
for iteration in 1..K (K=5):
    sample a contiguous segment of random length (geometric p=0.3, min=1, max=T/2)
    mask that segment
    transformer predicts new tokens for masked positions
    update sequence with predicted tokens
return final sequence
```
Hyperparameters: K=5, pre-training learning rate 1e-4, denoising learning rate 1e-5, batch size 64, AdamW with no weight decay, gradient clipping 1.0.

## (C) Why this design
We chose a denoising objective over pure autoregressive generation because it inherently requires the model to consider the full context (unmasked tokens) to predict missing ones, enforcing global consistency at training time. The trade-off is that denoising training is more computationally expensive than autoregressive due to multiple forward passes per sample (one per noise level). Second, we use structured contiguous masking during both training and inference to force the model to use long-range context, as uniform random masking can be solved using local n-gram patterns. Third, we include optional autoregressive pre-training to provide a good initial sequence, balancing between starting quality and training cost: a pure denoising start from all-mask would require many iterations, while a good autoregressive start reduces iterations needed. We also chose a sigmoid noise schedule to smoothly vary mask rates, avoiding extreme corruption levels that hurt training stability.

## (D) Why it measures what we claim
- The denoising loss (cross-entropy on masked tokens) measures **global consistency** when using structured contiguous masking, because the model must integrate information from distant unmasked tokens to fill a contiguous gap. This fails when the gap is short (local patterns suffice).
- The iterative refinement process measures **backtracking** because the number of token replacements that reduce the model's self-consistency loss (cross-entropy on the full sequence) during a refinement step quantifies error correction. The linking assumption is that every replacement lowering self-consistency corresponds to fixing a previously incorrect token. This assumption fails when the model replaces correct tokens with incorrect ones that happen to lower self-consistency (due to model miscalibration), inflating backtracking count without improving quality.
- The random contiguous masking policy ensures that the model must handle varying-length dependencies, operationalizing unbiased backtracking across all sequence positions.

## Contribution

(1) A novel formulation of structural token generation as a denoising process, enabling iterative refinement that naturally supports backtracking and global consistency without auxiliary modules. (2) A training procedure combining autoregressive pre-training with denoising fine-tuning that improves prediction accuracy over pure autoregressive models. (3) An inference protocol with uniform random masking that avoids calibration-based gating, making the method robust and simple.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Protein homo-oligomer symmetry, plus QM9 solubility and melting point | Tests diverse structure-property tasks |
| Primary metric | AUC-PR per symmetry class; RMSE for solubility/melting point | Reflects ranking and regression quality |
| Baseline: Autoregressive transformer | Same architecture, no denoising | Isolates benefit of denoising objective |
| Baseline: Graph Neural Network | GNN on structure (e.g., GVP) | Strong domain-specific baseline |
| Baseline: D3PM (discrete denoising diffusion) | Uniform transition, same transformer | Differentiates from diffusion models |
| Ablation: GSD w/o iterative refinement | Single-pass prediction | Tests need for refinement steps |
| Additional metric: Global consistency | Fraction of predicted pairwise distances >50 apart within 1Å | Direct measure of global structure accuracy |

### Why this setup validates the claim
This design forms a falsifiable test of GSD's core claims—global consistency and backtracking—by comparing against an autoregressive baseline that lacks both, a GNN that uses explicit structure, a D3PM baseline that shares denoising but uses different objective, and an ablation without iteration. The symmetric oligomer dataset (from Seq2Symm) requires understanding long-range contacts and global arrangement, making it ideal: poor local predictions can mislead symmetry class. QM9 tasks test broader applicability. AUC-PR captures per-class ranking, sensitive to confidence calibration. The global consistency metric directly quantifies structural accuracy. If GSD outperforms the autoregressive version, it confirms denoising enforces global context. If it beats the GNN and D3PM, it shows our specific corruption strategy is effective. If the ablation underperforms, it proves iterative refinement is necessary. Conversely, uniform gains or failure on specific subsets would refute the mechanism.

### Expected outcome and causal chain

**vs. Autoregressive transformer** — On a protein with distant interacting residues that determine symmetry (e.g., a homodimer with interface far in sequence), the autoregressive model, generating left-to-right, may produce a plausible local patch but miss the global arrangement because it cannot revise early tokens. It thus predicts the wrong symmetry class (e.g., C2 instead of D2). Our GSD, trained to denoise masked contiguous segments, learns to fill gaps using faraway context; during iterative refinement, it corrects the initial misprediction by masking and replacing a segment based on full sequence. We expect a noticeable gap in AUC-PR on sequences with long-range contacts (e.g., >100 residues apart) but parity on short-range cases.

**vs. Graph Neural Network** — On a symmetry type rarely seen in training (e.g., cyclic with odd order), the GNN, which relies on local geometric features from predicted 3D structure, may misclassify due to poor structure prediction or lack of template. Our GSD, operating directly on the token sequence, captures higher-order patterns from the denoising objective; it can infer symmetry from sequence motifs even without explicit geometry. We expect GSD to outperform on rare symmetry classes while matching GNN on common ones, yielding higher overall AUC-PR especially on long-tail classes.

**vs. D3PM** — D3PM uses a fixed uniform transition while our GSD uses a learned corruption schedule with contiguous masking. On long-range dependencies, our structured masking forces the model to attend globally, while D3PM may rely on local patterns from uniform noise. We expect GSD to yield better global consistency scores and higher AUC-PR on symmetry tasks.

**vs. GSD w/o iterative refinement** — On a sequence where the initial autoregressive generation places a token that is locally plausible but globally inconsistent (e.g., a residue that breaks the symmetry pattern), the single-pass model cannot backtrack, leading to an incorrect final prediction. The full GSD with refinement replaces that token after observing the rest of the sequence, correcting the error. We expect the full model to show better AUC-PR on cases where initial sequence has at least one wrong token, while the ablated version saturates quickly.

### What would falsify this idea
If GSD's gains over the autoregressive baseline are uniform across all sequence lengths and symmetry classes—not concentrated on long-range or rare cases—then the denoising objective is not providing global consistency as claimed. Alternatively, if the ablation with iterative refinement performs equally well, then backtracking is not the active mechanism. Also, if the global consistency metric shows no correlation with task performance, the claimed mechanism is unsupported.

## References

1. Accurate, Interdisciplinary and Transparent Structure-property Understanding with Deep Native Structural Reasoning
2. Training a Scientific Reasoning Model for Chemistry
3. Rapid and accurate prediction of protein homo-oligomer symmetry using Seq2Symm
4. RSGPT: a generative transformer model for retrosynthesis planning pre-trained on ten billion datapoints
5. Evolutionary-scale prediction of atomic level protein structure with a language model
6. Protein language models can capture protein quaternary state
7. O1 Replication Journey - Part 2: Surpassing O1-preview through Simple Distillation, Big Progress or Bitter Lesson?
8. Aviary: training language agents on challenging scientific tasks
