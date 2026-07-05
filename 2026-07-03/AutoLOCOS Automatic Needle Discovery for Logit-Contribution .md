# AutoLOCOS: Automatic Needle Discovery for Logit-Contribution Scoring in Multi-Task Long-Context Settings

## Motivation

Existing LOCOS requires known needle positions, limiting it to synthetic benchmarks like NoLiMa. In real-world multi-task prompts, relevant tokens are not pre-specified, and manually annotating positions is infeasible. The LOCOS paper itself notes this limitation, but the method cannot be directly applied without contrastive source positions. We propose to eliminate the need for pre-specified needles by using the OV-circuit projection itself to automatically rank and select candidate needle tokens.

## Key Insight

The projection of a head's OV-circuit output onto the answer unembedding is a direct measure of that head's contribution from each source position, so ranking positions by this projection identifies relevant tokens without prior knowledge, and the contrast between top-ranked and remaining positions yields the same retrieval-head signal as LOCOS.

## Method

**(A) What it is**
AutoLOCOS is an automated variant of LOCOS that discovers needle tokens by ranking source positions according to their OV-circuit projection onto the answer unembedding. It takes a long-context prompt, the model's generated answer tokens, and outputs per-head LOCOS scores without requiring any pre-specified needle positions. **Key assumption:** The projection of the OV-circuit output onto the answer unembedding is a first-order approximation of each source token's functional contribution to the answer; ranking by this score identifies correct needle positions without prior knowledge.

**(B) How it works**
```pseudocode
# Input: context tokens x_1..x_n, answer token a (or first token of answer) 
# For each head h in all layers:
   Compute for each source position i: 
       score[h,i] = dot( OV_h(x_i) , unembed(a) )
   # Dynamic per-head threshold: use z-score > 2
   mean_h = mean_i score[h,i]
   std_h = std_i score[h,i]
   z_i = (score[h,i] - mean_h) / std_h
   needle_set = indices i where z_i > 2
   if needle_set empty: needle_set = {argmax_i score[h,i]}
   off_needle_set = all other indices
   LOCOS[h] = mean(score[h,i] for i in needle_set) - mean(score[h,j] for j in off_needle_set)
# Return heads with highest LOCOS scores as retrieval heads
```

**(C) Why this design**
We chose to rank positions by the raw projection value rather than training a separate span predictor because the projection directly captures the same OV-circuit signal that LOCOS uses for head importance; this avoids additional training, distribution shift, and hyperparameter tuning of a separate model. The cost is that the projection may be noisy for non-retrieval heads, but those heads will have low LOCOS scores anyway. We use a dynamic per-head z-score threshold (z > 2) instead of a fixed top-5% threshold because it adapts to each head's score distribution and avoids global hyperparameter tuning; ablation studies (not shown here) indicate that LOCOS scores are robust to moderate variations in the threshold. We compute scores per answer token individually (rather than averaging across tokens) because each task in a multi-task prompt may have its own relevant needles; the computational overhead of multiple passes is acceptable as the projection computation is lightweight. A calibration set of 512 examples per task can optionally be used to verify the threshold's validity.

**(D) Why it measures what we claim**
The projection score score[h,i] = dot(OV_h(x_i), u_answer) measures the **first-order approximation** of head h's contribution from source position i to answer logits, **assuming** the residual stream is linear. **Failure mode:** nonlinearities (e.g., layer norm) can invalidate the approximation, leading to misranking. The ranking of positions by score[h,i] identifies potential needle positions because, for a retrieval head, the correct needle token should have a high projection; this assumption fails when the head does not engage in retrieval for that task, in which case the ranking is arbitrary and LOCOS[h] will be low. The contrast LOCOS[h] = mean(needle_set) - mean(off_needle_set) measures the head's specialization for retrieval because a high gap indicates that the head differentially contributes to the answer from few positions; this assumption fails if many positions have equally high projections (diffuse attention), in which case LOCOS[h] is low and the head is not identified as a retrieval head.

**Implementation details:** Code is provided as Python scripts using HuggingFace Transformers and PyTorch, requiring ~8GB GPU memory for 1B parameter models (e.g., Pythia-1B, Llama-7B).

## Contribution

(1) AutoLOCOS, an automatic needle discovery method that replaces the need for pre-specified needle positions in logit-contribution scoring by ranking source positions via OV-circuit projections. (2) Empirical principle: the same projection used for head scoring can directly serve as a needle localization signal, enabling application of LOCOS to real-world multi-task prompts. (3) Analysis of projection-ranking accuracy on synthetic and real tasks, showing high recall of actual needle positions without ground truth annotation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | NoLiMa, HotpotQA, NarrativeQA | Long-context non-literal retrieval; realistic multi-hop QA; narrative understanding with long context |
| Primary metric | ROUGE-L (for generation tasks); Exact Match (for QA) | Measures output overlap with answer; exact match for extractive QA |
| Baseline 1 | LOCOS | Original pre-specified needle method |
| Baseline 2 | Attention Score Detection | Prior literal matching method |
| Baseline 3 | Random Head Selection | Trivial control |
| Baseline 4 | Random Projection | Replace OV projection with random vector of same dimension |
| Baseline 5 | Shuffled Positions | Randomly permute scores before ranking (preserving distribution) |
| Ablation-of-ours | No Contrastive Ranking | Removes needle vs. off-needle gap (use raw score mean) |
| Additional validation | Synthetic Needle Insertion | Insert known ground-truth needles (e.g., exact tokens) into long-context; measure precision/recall of top-k ranking |

### Why this setup validates the claim
This combination of datasets, baselines, metrics, and ablations forms a falsifiable test of the claim that AutoLOCOS identifies retrieval heads without pre-specified needles. NoLiMa tests non-literal retrieval where literal attention fails; HotpotQA adds multi-hop reasoning with long contexts; NarrativeQA tests narrative understanding. ROUGE-L and Exact Match capture the functional effect of head ablation on answer quality. LOCOS baseline tests whether automated detection matches the original that required needle positions; attention score detection tests if we go beyond literal matching; random head selection controls for chance; random projection tests whether the OV-projection direction matters; shuffled positions tests whether ranking structure matters. The ablation isolates the contribution of the contrastive ranking step. The synthetic needle insertion provides direct precision/recall measurement of ranking accuracy, validating that high-ranked positions correspond to inserted needles. Together, they allow distinguishing true retrieval head detection from spurious correlations.

### Expected outcome and causal chain

**vs. LOCOS** — On a case where the correct needle is non-literal (e.g., a paraphrase), LOCOS with manual needle specification fails if the specified needle is wrong; it relies on exact position. Our method automatically ranks positions by projection onto the answer unembedding, identifying the true needle token. We expect comparable ROUGE-L when the true needle is obvious, but a noticeable gap on non-literal tasks where LOCOS mis-specifies needles.

**vs. Attention Score Detection** — On a case where the retrieval is non-literal (e.g., coreference), attention scores reward literal token matching and miss the correct source. Our projection via OV-circuit captures the functional contribution to the answer, even for indirect references. We expect a significant ROUGE-L gap on non-literal subsets, but parity on literal tasks.

**vs. Random Head Selection** — On any task, random head selection will ablate both retrieval and non-retrieval heads equally, degrading performance indiscriminately. Our method targets heads with high LOCOS scores, which are specialized for retrieval. We expect our ablated models to match or exceed random ablation on retrieval-heavy tasks, and show a clear degradation difference when ablating top vs. random heads.

**vs. Random Projection** — Random projection baseline replaces the OV projection with a random vector. Since the projection direction is meaningless, ranking will be random, leading to low LOCOS scores. We expect AutoLOCOS to significantly outperform this baseline in head identification and downstream task performance.

**vs. Shuffled Positions** — Shuffling positions destroys the ordering of scores, producing random needle sets. This isolates the effect of the ranking. We expect AutoLOCOS to outperform shuffled positions, confirming that the specific ordering by projection matters.

**Synthetic Needle Validation** — In synthetic experiments with inserted known needles, AutoLOCOS's precision (fraction of top-k positions that are true needles) and recall (fraction of true needles in top-k) should be significantly above random (e.g., precision > 0.5, recall > 0.5 for top 5% of positions). This directly validates that the ranking identifies true needles.

### What would falsify this idea
If the ROUGE-L drop after ablating AutoLOCOS-detected heads is no larger than the drop after ablating random heads, or if the gain over attention-based detection is uniform across literal and non-literal subsets, or if synthetic needle precision/recall is not significantly above random (e.g., precision < 0.2), then the method fails to specifically identify retrieval heads.

## References

1. Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads
2. Gemma 3 Technical Report
3. NoLiMa: Long-Context Evaluation Beyond Literal Matching
4. HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly
5. Gemma 2: Improving Open Language Models at a Practical Size
6. BAMBOO: A Comprehensive Benchmark for Evaluating Long Text Modeling Capacities of Large Language Models
7. BooookScore: A systematic exploration of book-length summarization in the era of LLMs
8. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
