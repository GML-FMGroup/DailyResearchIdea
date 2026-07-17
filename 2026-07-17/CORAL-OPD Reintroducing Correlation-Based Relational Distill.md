# CORAL-OPD: Reintroducing Correlation-Based Relational Distillation in On-Policy Settings

## Motivation

On-policy distillation methods, as analyzed in Demystifying On-Policy Distillation (2026), drop correlation-based losses that capture inter-sample relationships, leading to loss of fine-grained structural information. The token-level guidance only learns pointwise mappings, missing the teacher's relational knowledge across diverse inputs. This structural information is critical for tasks requiring reasoning, and its omission is a root cause of the student's failure to generalize beyond surface patterns.

## Key Insight

The pairwise similarity structure in the teacher's embedding space forms a consistent invariant that can be transferred to the student via a correlation-based loss, even when individual teacher logits are noisy or biased by exploration.

## Method

## (A) What it is
CORAL-OPD augments on-policy distillation (OPD) with a correlation-based loss that aligns inter-sample similarities in a joint embedding space. Input: teacher T, student S, batch of prompts X. Output: combined loss L = L_opd + λ L_corr.

## (B) How it works
```pseudocode
# CORAL-OPD Training Step
Input: teacher T, student S, batch of prompts X = {x_i}
Output: combined loss L

# Calibration: Before training, compute reference similarity matrix R from teacher embeddings on a held-out set of 512 teacher-generated responses (fixed reference set).

# Phase 1: On-policy generation
for each prompt x_i:
    y_i ~ S(·|x_i)                     # student response
    y'_i ~ T(·|x_i)                    # teacher response (on same prompt)

# Phase 2: Compute embeddings (last hidden layer, mean pooling)
H_S = [S.encoder(x_i, y_i) for i]     # shape (N, d)
H_T = [T.encoder(x_i, y'_i) for i]    # shape (N, d)

# Phase 3: Correlation-based loss
S_S = cosine_similarity_matrix(H_S)    # N×N
S_T = cosine_similarity_matrix(H_T)    # N×N
L_corr = MSE(S_S, S_T)                 # hyperparameter λ (e.g., 0.1)

# Verification (periodic, e.g., every 100 steps): Compute L_corr using reference set R (teacher embeddings on same reference prompts) and compare; if deviation > 0.05, flag potential violation of assumption that on-policy embeddings are semantically meaningful.

# Phase 4: On-policy distillation loss (from OPD 2026)
L_opd = KL_divergence(teacher_logits, student_logits) on-policy with length regulation

# Phase 5: Combined loss
L = L_opd + λ * L_corr
```

**Load-bearing assumption** (explicit): The pairwise cosine similarity matrix of teacher embeddings computed on on-policy student-generated responses reliably captures the teacher's relational knowledge about the semantic structure of correct answers. This assumption is verified via periodic calibration against a reference set of teacher-generated outputs.

## (C) Why this design
We chose pairwise similarity alignment (MSE on cosine similarity matrices) over contrastive loss because it directly captures relative distances, preserving structural information without negative sample mining. Using the same on-policy inputs for teacher embeddings ensures distribution match, avoiding a key OPD pathology; the trade-off is that teacher embeddings may reflect exploration noise, but this is acceptable as the student needs to learn on its own distribution. We weight the correlation loss with λ to balance against token-level guidance; a high λ risks overshadowing fine-grained token signals, while a low λ renders it ineffective. We tune λ via grid search over {0.01, 0.05, 0.1, 0.2} on a validation set. We opted for MSE over KL for similarity matrices because MSE is symmetric and efficient, though it assumes equal importance across all pairs; this assumption fails when some pairs are outliers, but normalization mitigates this. We use cosine similarity rather than Euclidean distance due to scale invariance, but it ignores magnitude differences; this is deliberate to focus on angular structure. The hyperparameter λ is tuned on a validation set, typically between 0.01 and 0.2.

## (D) Why it measures what we claim
The pairwise similarity matrix S_T measures inter-sample relational knowledge because it encodes the teacher's assessment of semantic closeness between responses; this assumption fails when the teacher's embeddings are not semantically meaningful (e.g., due to poor pretraining), in which case S_T reflects shallow token overlap. The MSE loss between S_T and S_S measures the alignment of relational knowledge because it penalizes deviations in pairwise distances; this assumption fails when embedding spaces have different dimensional scales (we normalize to unit length to mitigate), in which case MSE may reflect scaling differences rather than structural mismatch. The on-policy generation ensures the correlation loss operates on the student's own distribution, directly preventing distribution mismatch—a core pathology identified by OPD (2026). Thus, L_corr operationalizes the goal of preserving fine-grained structural information that token-level OPD drops. We explicitly assume that cosine similarity in embedding space is a proxy for reasoning structure; this assumption is validated by calibration against a reference set of teacher-generated outputs, ensuring that S_T indeed captures semantic relationships relevant to reasoning.

## Contribution

(1) A novel correlation-based distillation loss, CORAL, that captures inter-sample relationships in on-policy distillation using a joint embedding space and pairwise similarity alignment. (2) A design principle showing that relational knowledge transfer via correlation loss complements token-level guidance, improving structural understanding and generalization in reasoning tasks. (3) Open-source implementation and hyperparameter configurations for reproducible experiments on math reasoning benchmarks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MATH (primary), GSM8K, ARC | Math reasoning requires structural knowledge; extra tasks test generality. |
| Primary metric | Accuracy (pass@1) | Direct measure of response correctness. |
| Baseline 1 | On-policy distillation (OPD) | Tests value of correlation loss. |
| Baseline 2 | Off-policy distillation | Tests importance of on-policy generation. |
| Ablation | CORAL-OPD with contrastive loss (InfoNCE with τ=0.07) | Tests MSE over cosine choice. |
| Hyperparameter tuning | λ via grid search {0.01, 0.05, 0.1, 0.2} on validation accuracy | Ensures reproducibility. |

### Why this setup validates the claim

The MATH dataset requires multi-step reasoning, where token-level distillation may miss structural relationships. Baseline OPD isolates the correlation loss contribution, testing if aligning inter-sample similarities adds value beyond token-level KL. Off-policy distillation tests whether on-policy generation is essential for preventing distribution mismatch. The contrastive loss ablation tests whether MSE on cosine similarity matrices is the optimal design for preserving relational structure. GSM8K and ARC extend the evaluation to diverse reasoning formats, ensuring the method's generality. Together, these comparisons form a falsifiable test: if our method outperforms OPD and off-policy, and the contrastive ablation underperforms our MSE version, it validates that correlation-based alignment on on-policy distributions improves distillation by capturing semantic structure that token-level signals drop.

### Expected outcome and causal chain

**vs. OPD** — On a case where the student generates a plausible but flawed reasoning chain (e.g., missing a step), OPD's token-level KL aligns tokens but misses structural errors because the teacher output on the same prompt may have a different reasoning path. Our method aligns similarity matrices, flagging that the student's flawed output is far from correct solutions in the embedding space. Thus on such cases, CORAL-OPD imposes a higher penalty, leading to more correct reasoning. We expect a noticeable gap (e.g., 5-10% accuracy) on multi-step problems, with parity on simple ones.

**vs. Off-policy** — On a case where the student's generation style differs from the teacher (e.g., terse vs. verbose), off-policy forces the student to mimic teacher responses, hurting fluency and possibly correctness. Our on-policy approach avoids this mismatch by learning on the student's own distribution. We expect off-policy to perform worse on problems requiring diverse solution formats (e.g., different phrasing), while our method maintains performance.

**vs. CORAL-OPD with contrastive loss** — On a case where the batch contains semantically similar but not identical responses (e.g., two different correct approaches), the contrastive loss may push them apart, losing structural similarity. MSE on cosine similarity matrices preserves relative distances, better capturing nuanced relationships. We expect the contrastive ablation to underperform on problems with multiple correct reasoning paths (e.g., ARC puzzles), with a 2-5% accuracy drop.

### What would falsify this idea
If the improvement from CORAL-OPD is uniform across problem difficulties rather than concentrated on multi-step reasoning tasks, or if the correlation loss provides no benefit over OPD, the claim that structural alignment captures causal reasoning knowledge is false. Additionally, if the method fails to generalize to GSM8K and ARC (i.e., no improvement over OPD), the claim that correlation loss captures general reasoning structure is weakened.

## References

1. Demystifying On-Policy Distillation: Roles, Pathologies, and Regulations
2. Solving Quantitative Reasoning Problems with Language Models
3. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
4. Llemma: An Open Language Model For Mathematics
5. The Stack: 3 TB of permissively licensed source code
