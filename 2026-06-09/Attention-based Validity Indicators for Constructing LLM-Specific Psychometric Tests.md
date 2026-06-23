# Attention-based Validity Indicators for Constructing LLM-Specific Psychometric Tests

## Motivation

Existing psychometric evaluations of LLMs, such as those in 'Do Psychometric Tests Work for Large Language Models?', rely on human-validated instruments without establishing whether items are semantically critical for the model. This leads to low ecological validity because items may not capture model-specific biases. A model-grounded method is needed to select items that the LLM actually attends to, ensuring tests are valid for the model.

## Key Insight

The degree to which an LLM's attention focuses on semantically critical parts of an item is a proxy for that item's construct validity for the model, because attention allocation reflects the model's internal representation of relevance.

## Method

(A) **What it is**: Attention-based Item Selection (AIS) is a method to select psychometric test items by computing an attention-based validity score for each candidate item, using the LLM's attention distributions over the item text. Input: a set of candidate items (e.g., from existing tests) and a definition of "semantically critical" words for the construct. Output: a subset of items with high validity scores.

(B) **How it works**:
```pseudocode
Input: candidate items I = {i_1,...,i_M}, each item is a text prompt;
       concept keywords K = {k_1,...,k_L} for the target construct;
       LLM with L layers and H attention heads (e.g., GPT-2 small: L=12, H=12);
       attention aggregation function f_att: mean over heads and layers 8-11 (last 4 layers) then average over query positions;
       threshold θ (default 0.5, tuned via grid search on calibration set of 200 items with human validity labels, searching over {0.3,0.5,0.7,0.9});
Output: valid items V ⊆ I.

For each item i_j:
  1. Compute attention maps from LLM when processing i_j:
     For each layer l and head h, get attention weights A_{l,h} [seq_len x seq_len].
  2. Aggregate attention over heads and layers:
     A_agg = f_att({A_{l,h}}) — a [seq_len, seq_len] matrix.
     Specifically, take mean over heads and layers 8-11, then average over query positions to get per-token attention received.
     Let a_t = mean over queries of A_agg[:, t] for each token t.
  3. Identify positions of critical tokens:
     Let P_critical = positions in i_j where token belongs to K (exact match, case-insensitive).
  4. Compute validity score s_j = mean_{t in P_critical} a_t.
  5. If s_j >= θ, add i_j to V.
Return V.
```
Hyperparameters: threshold θ (tuned via grid search on a calibration set of 200 items with human validity labels), attention aggregation f_att (mean over heads and layers 8-11 for a 12-layer model).

(C) **Why this design**: We chose mean attention over critical tokens as the validity score because it directly quantifies how much the model focuses on concept-relevant words versus distractors. We considered using variance of attention across tokens as an alternative, but that would penalize items that require attending to multiple critical tokens equally. We also considered using attention from the last layer only versus all layers; we choose deeper layers (8-11) because in GPT-2, deeper layers encode more semantic information, but we accept the cost of increased computation and potential noise from lower layers. We set a threshold θ instead of selecting top-k items because a threshold ensures that all selected items meet a minimum validity standard, though it may discard some items that are borderline. The trade-off is that top-k could include items with varying validity, whereas thresholding guarantees a floor. We define critical tokens based on a priori concept keywords rather than learning them from data, which is simpler but risks missing implicit concept tokens; we accept this because our aim is to bootstrap psychometric tests from existing construct definitions.

(D) **Why it measures what we claim**: The aggregated attention received by critical tokens (computational quantity s_j) measures item construct validity for the LLM because we assume that the LLM's attention to concept keywords reflects the extent to which the item activates the construct representation in the model; this assumption fails when the model attends to critical tokens for spurious reasons (e.g., token frequency or positional bias) or when the construct is not lexicalized in the item (e.g., abstract concepts without trigger words), in which case s_j may underestimate validity. The attention aggregation function f_att (mean over heads and layers 8-11) measures the average importance of tokens across high-level representation spaces, under the assumption that deeper layers contribute most to semantic relevance, which fails when validity is localized to specific heads (e.g., attention heads specialized for bias detection) or when shallow layers capture critical syntactic cues, in which case averaging could dilute the signal. The threshold θ operationalizes a minimal validity criterion, assuming that items above θ are sufficiently attended, but this assumption fails when the optimal threshold varies by construct or model size, requiring calibration. To verify the core assumption, we calibrate θ using a held-out set of 200 items with human validity labels; we select θ that maximizes Pearson correlation between AIS scores and human validity. Additionally, we perform a per-token analysis on 50 items comparing attention weights to human-annotated token relevance (binary labels); we compute the Pearson correlation between token-level attention and human relevance, expecting r>0.3 for valid items. This fine-grained analysis empirically supports the link between attention and construct validity.

## Contribution

(1) A novel attention-based validity score for item selection in LLM psychometric tests, grounded in model internals rather than human judgment. (2) An empirical finding that items selected via AIS produce more reliable and ecologically valid test scores compared to random item selection or human-validated items. (3) A validation dataset of attention-validated items for three constructs (sexism, racism, morality) and five LLMs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Psych-101 item bank (1,000 items) with human validity ratings (1-5 Likert) | Ground truth from human theory; 200 items held out for calibration, 800 for evaluation |
| Primary metric | Pearson correlation between AIS scores and human validity | Directly measures selection quality |
| Baseline 1 | Random item selection (average over 10 random seeds) | Lower bound for selection performance |
| Baseline 2 | Keyword frequency count (number of critical token occurrences, normalized by length) | Simple lexical baseline |
| Baseline 3 | Perplexity-based item selection (select items with perplexity below median) | Captures language difficulty |
| Baseline 4 | Probing classifier accuracy (train a linear probe on LLM hidden states to predict construct label, select items where probe accuracy is highest) | Tests whether attention is better than alternative intrinsic method |
| Ablation | AIS with only last-layer attention (layer 11 only) | Tests necessity of multi-layer aggregation |

### Why this setup validates the claim
This setup tests the central claim—that attention to critical tokens indicates item validity—by comparing against random selection (which ignores item content), keyword frequency (which lacks contextual importance), perplexity (which measures global difficulty), and probing classifier accuracy (which captures construct information from hidden states). The ablation isolates the contribution of multiple layers. The primary metric (correlation with human validity) directly quantifies how well AIS aligns with human-judged construct relevance. If AIS outperforms baselines and ablation, it supports that attention-based scores capture construct-specific validity beyond superficial cues. Conversely, if AIS fails to outperform keyword frequency or probing, the mechanism of attending to critical tokens may be no better than simpler signals. Additionally, a case study on items where human validity and AIS scores diverge (e.g., an item with high human validity but low attention due to non-lexicalized construct) will highlight real-world limitations and inform future improvements.

### Expected outcome and causal chain
**vs. Random** — On an item with many irrelevant words, random selection may include it accidentally. Our method assigns low attention to critical tokens if they are not attended to, so the score is low. We expect AIS-selected items (top 20% by score) to show significantly higher average human validity (mean difference >0.5) than random-select items (p<0.01, one-sided t-test).

**vs. Keyword frequency** — On an item where critical words are common but not semantically central (e.g., "man" in a gender bias item used generically), keyword frequency overestimates validity. Our method's attention captures contextual importance, so these items get lower scores. We expect AIS to have higher correlation with human validity (Δr >0.1) on items where keyword frequency is high but human validity is low (bottom quartile).

**vs. Perplexity** — On a complex item with clear construct words (e.g., a long legal text about discrimination), perplexity may incorrectly flag it as difficult, but our method focuses on critical tokens, giving a high validity score. We expect AIS to show significantly higher correlation on items with high perplexity (>median) but high human validity (top quartile).

**vs. Probing classifier** — On an item where construct information is distributed across tokens (e.g., sentiment requiring contextual composition), probing accuracy may be low because the probe assumes fixed representation, whereas attention dynamically highlights relevant tokens. We expect AIS to outperform probing on such items (Δr >0.05).

**vs. Last-layer only** — On an item requiring semantic integration (e.g., metaphor for bias), last-layer attention may miss nuanced patterns from intermediate layers. Our method aggregates layers 8-11, capturing both deep and intermediate attention. We expect higher correlation for full AIS (Δr >0.05) on abstract constructs (e.g., social desirability).

### What would falsify this idea
If the correlation of AIS with human validity is not significantly higher than random selection (p>0.05), or if the improvement over keyword frequency is uniform across all item types (indicating no special benefit from attention to critical tokens), then the claim that attention-based validity scores are effective is falsified. Additionally, if per-token analysis shows no correlation (r<0.1) between attention and human token relevance, the core assumption is unsupported.

## References

1. Do Psychometric Tests Work for Large Language Models? Evaluation of Tests on Sexism, Racism, and Morality
2. Centaur: a foundation model of human cognition
3. Are Large Language Models Moral Hypocrites? A Study Based on Moral Foundations
4. How Well Do LLMs Represent Values Across Cultures? Empirical Analysis of LLM Responses Based on Hofstede Cultural Dimensions
5. LLM Evaluators Recognize and Favor Their Own Generations
6. Can AI language models replace human participants?
7. Morality beyond the WEIRD: How the nomological network of morality varies across cultures.
8. Cultural Alignment in Large Language Models: An Explanatory Analysis Based on Hofstede's Cultural Dimensions
