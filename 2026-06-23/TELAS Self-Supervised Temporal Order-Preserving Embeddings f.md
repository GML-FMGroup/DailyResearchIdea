# TELAS: Self-Supervised Temporal Order-Preserving Embeddings for Heterogeneous Lab Notebooks

## Motivation

Existing lab notebook information extraction methods, such as Notes2Skills, assume well-structured input with explicit markers (e.g., section headers), failing on heterogeneous or sparse entries. This reliance on consistent formatting prevents reliable extraction across diverse laboratory settings. A fundamental structural invariant—experimental steps are always described in chronological order—remains untapped as a supervisory signal.

## Key Insight

The chronological order of experimental steps is an invariant that holds across all lab notebooks irrespective of formatting variations, providing a natural supervisory signal for representation learning without requiring annotated data or consistent markers.

## Method

**TELAS: Self-Supervised Temporal Order-Preserving Embeddings**

(A) **What it is**: TELAS is a self-supervised framework that learns dense embeddings of text segments (step descriptions) from lab notebooks by enforcing that the embeddings preserve the chronological order of steps. Input: raw text segments from a notebook in sequence. Output: a vector embedding per segment such that earlier steps have lower scalar order scores (projected from embeddings) than later steps.

(B) **How it works** (pseudocode):
```
Input: Notebook N = [s_1, s_2, ..., s_T] (text segments in order)
Hyperparameters: temperature τ=0.1, order margin α=0.5, embedding dim d=128, MLP hidden=256, activation=GeLU, λ=0.1
Encoder: BERT-base (frozen for first 5 epochs, then finetune)
OrderProjector: 2-layer MLP (d -> 256, GeLU -> 1 scalar)

Preprocessing: For each notebook, split into step sequences using sentence boundaries or bullet points.

for each batch of notebooks:
    for each notebook N:
        E = Encoder(s_i) for i=1..T  # [T, d]
        o = OrderProjector(E)        # [T, 1] scalar scores
        # Ensure consecutive steps satisfy o_i + α < o_{i+1}
        order_loss = mean( max(0, o_i - o_{i+1} + α) ) over all adjacent pairs
        
        # Optional: encourage distinctness via contrastive loss between non-adjacent steps
        contrastive_loss = sum_{i≠j} -log( exp(sim(E_i,E_j)/τ) / sum_k exp(sim(E_i,E_k)/τ) )
        loss = order_loss + λ * contrastive_loss
    
    update encoder and projector
```
(C) **Why this design**: We chose a BERT-based encoder over a bag-of-words or BiLSTM because pretrained contextual embeddings capture semantic variation in step descriptions (e.g., "add 5mL" vs "pour 5ml") without assuming consistent vocabulary (trade-off: BERT's computational cost vs. simplicity). We use a 2-layer MLP order projector with margin loss instead of a linear layer because linear projections of BERT may only capture surface features (Tenney et al., 2019), while a small nonlinear MLP can better map to temporal order (trade-off: MLP adds parameters and risk of overfitting, mitigated by early stopping). The margin explicitly enforces a gap between steps, mimicking the temporal progression (trade-off: margin introduces hyperparameter α, but prevents score collapse where all steps get the same order score). We include a contrastive auxiliary loss to separate non-adjacent steps, which avoids trivial solutions where encoder outputs the same embedding for all steps (trade-off: λ must be tuned to balance order vs. distinctness; if too high, order signal is diluted).

(D) **Why it measures what we claim**: The order score o_i = MLP(E_i) (a scalar projection) measures temporal ordering because we assume that the MLP defines a direction in embedding space that aligns with the chronological axis; the margin loss enforces monotonic growth. Formally, margin loss minimized ↔ adjacent pairs ordered ↔ high POA, under the assumption that the scalar projection captures temporal direction. This assumption fails when two consecutive steps are semantically identical (e.g., "swirl gently" repeated) — then o_i may not increase, and the loss pushes embeddings apart artificially, causing the score to reflect forced separation rather than true temporal order. In such cases, the embedding still captures textual similarity but loses temporal signal for those steps.

## Contribution

(1) A self-supervised framework TELAS that learns order-preserving embeddings from raw lab notebook text by exploiting temporal order as a supervisory signal, requiring no manual annotation or predefined markers. (2) A demonstration that the learned embeddings enable robust step extraction and skill representation that is resilient to heterogeneous formatting, validated on multi-source lab notebooks. (3) A public benchmark of real-world lab notebooks with varying degrees of structure to facilitate further research.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | Lab notebook procedural text corpus, Recipe dataset (e.g., Recipe1M steps) | Realistic step sequences with temporal order; test generalizability |
| Primary metric | Pairwise ordering accuracy (POA) | Directly measures order preservation |
| Additional metric | Visualization of learned order score vs. step index (scatter plot for 50 random notebooks) | Verifies monotonicty assumption |
| Baseline 1 | BoW + linear order regressor | Tests need for semantic embeddings |
| Baseline 2 | BiLSTM + margin order loss | Tests need for pretrained context |
| Baseline 3 | TELAS with linear projector (ablation of nonlinearity) | Tests if nonlinear MLP is necessary |
| Ablation of ours | TELAS without contrastive loss | Isolates effect of distinctness loss |

### Why this setup validates the claim
This experimental design forms a falsifiable test of TELAS's central claim: that enforcing monotonic order scores via a margin loss on contextualized embeddings yields a proper temporal ordering signal. The lab notebook corpus provides ground-truth step order, making POA a direct measure. The recipe dataset tests generalizability to a different domain with similar structure. The BoW baseline tests whether simple bag-of-words can capture order—expected to fail because it ignores semantic variation. The BiLSTM baseline tests whether a non-pretrained sequential model can learn order from scratch—likely to struggle with sparse vocabularies. The linear projector ablation tests whether the nonlinear projection is necessary for capturing order. The ablation isolates the contrastive auxiliary loss's role. Visualization provides qualitative evidence of monotonic ordering. If TELAS outperforms all baselines and the ablated versions, and visualizations show clear monotonic trends, the causal chain is supported; if not, the claim is undermined.

### Expected outcome and causal chain

**vs. BoW + linear order regressor** — On a case where two steps differ in phrasing but share meaning (e.g., "add 5mL HCl" vs. "pour hydrochloric acid 5mL"), the BoW model produces a bag of overlapping but not identical words, leading to inconsistent order scores because it lacks semantic similarity. Our method captures the semantic equivalence via BERT embeddings, so the order scores remain consistent and monotonic. We expect a large gap in POA on steps with paraphrastic variation, but near parity on steps with identical vocabulary.

**vs. BiLSTM + margin order loss** — On a case where a step description contains rare or compound words (e.g., "centrifuge at 5000g for 10min"), the BiLSTM from scratch has no prior knowledge of these terms and thus produces noisy embeddings, causing order violations. Our BERT encoder leverages pretrained knowledge to map such phrases to stable order representations. We expect a noticeable improvement on steps with rare terminology, but smaller gains on common phrases.

**vs. TELAS with linear projector** — The linear projector may fail to capture complex mappings from BERT embeddings to order scores, especially when steps are conceptually similar but temporally distinct. The MLP provides flexibility to learn a nonlinear separation. We expect the MLP version to achieve higher POA on notebooks where steps have high semantic overlap (e.g., many "incubate" steps with varying durations).

### What would falsify this idea
If the improvement over the BiLSTM baseline is uniform across all subsets (rather than concentrated on rare-vocabulary or paraphrastic steps), the central claim that BERT's contextual embeddings drive the gain would be wrong, suggesting the margin loss itself suffices even without pretraining. Additionally, if the scatter plot of order scores vs. true step index shows non-monotonic or flat trends for a significant portion of notebooks, the assumption that the scalar projection captures temporal direction is false.

## References

1. Notes2Skills: From Lab Notebooks to Certainty-Aware Scientific Agent Skills
2. Structured information extraction from scientific text with large language models
3. ProtoCode: Leveraging large language models (LLMs) for automated generation of machine-readable PCR protocols from scientific publications.
4. Agent Workflow Memory
5. ChatGPT Chemistry Assistant for Text Mining and the Prediction of MOF Synthesis
6. Llama 2: Open Foundation and Fine-Tuned Chat Models
7. The Laboratory Automation Protocol (LAP) Format and Repository: a platform for enhancing workflow efficiency in synthetic biology
8. Evaluating the Usefulness of a Large Language Model as a Wholesome Tool for De Novo Polymerase Chain Reaction (PCR) Primer Design
