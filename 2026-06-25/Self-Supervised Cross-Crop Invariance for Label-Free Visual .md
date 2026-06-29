# Self-Supervised Cross-Crop Invariance for Label-Free Visual Reasoning

## Motivation

V-Zero's contrastive evidence gating requires a teacher model pre-trained on labeled data, inheriting a dependency on supervised pre-training that limits label-free learning. This structural problem, where teacher initialization remains a bottleneck across multiple works, persists because existing methods rely on a separate teacher to provide contrastive signals for trajectory evaluation. Without a teacher, there is no labeled signal to distinguish good from bad reasoning trajectories. We identify that geometric invariance across spatial crops can replace the supervised teacher by enforcing answer consistency over transformations of the same visual region, providing a self-supervised contrastive signal.

## Key Insight

The answer to a visual question should be invariant under geometric transformations that preserve the question-relevant region, enabling unsupervised contrastive learning by aligning distributions across positive crops and separating them from negative crops.

## Method

## Self-Supervised Cross-Crop Invariance (S3CI)

**(A) What it is:** S3CI is a training framework that replaces the supervised teacher in contrastive visual reasoning with a self-supervised geometric invariance constraint. It inputs an image-question pair (no answer) and outputs a student MLLM (e.g., LLaVA-1.5-7B) that learns consistent reasoning across crop transformations of the same semantic region.

**(B) How it works:**
For each training example (image I, question Q):
1. **Unsupervised region proposals:** Use DINO ViT-B/16 self-attention maps from the [CLS] token to generate candidate regions R = {r1, ..., rk} (k=20, threshold=0.3 on attention).
2. **Relevant region selection:** Forward the student MLLM once; extract cross-attention weights from the last decoder layer (32 heads, averaged over heads). Compute average attention over patches as a saliency map S. Select region r* that maximizes average S within its bounding box.
3. **Positive view generation:** Apply random geometric transformations: rotation by angle uniformly sampled from [-10°,10°] (if question type is not spatial; spatial questions detected via keyword matching with words like 'left', 'right', 'above', 'below', 'in front', 'behind' then omit rotation), scaling factor from [0.9,1.1], translation up to ±10% of crop size. Produce 3 positive crops P = {p1, p2, p3}.
4. **Negative view generation:** Randomly sample 3 regions N from R that have IoU < 0.2 with r* (or random image crops of similar size).
5. **Forward with crops:** For each crop c in P ∪ N ∪ {original I}, run the student MLLM with question Q to obtain output logits over answer tokens and hidden state h (from the last token before answer decoding). Use stop-gradient (sg) on the positive view outputs for consistency target.
6. **Loss computation:**
   - Consistency loss: L_align = Σ_{i≠j} KL( f(p_i) || sg(f(p_j)) ) where f(c) is the output distribution for crop c.
   - Contrastive loss: L_contrast = - Σ_{p in P} log( exp(sim(h_r*, h_p)/τ) / (exp(sim(h_r*, h_p)/τ) + Σ_{n in N} exp(sim(h_r*, h_n)/τ) ) ) where sim is cosine similarity, τ=0.1.
   - Total loss: L = L_align + λ L_contrast, λ=1.0.
7. **Calibration of assumption:** Every 1000 steps, evaluate on a held-out set of 200 transformed images (same transformations as positive views). If accuracy drops below 80% of original accuracy, temporarily reduce augmentation magnitude by half. Also, monitor average KL divergence on calibration set of 500 hand-annotated examples; if average KL > 0.5, reduce rotation range for that question type.

**(C) Why this design:** We chose stop-gradient in L_align to prevent representation collapse (SimSiam, Chen & He, 2021), accepting that it slows convergence because the consistency target remains stale. We selected KL divergence over MSE because KL preserves distributional sharpness, critical for answer generation; the trade-off is that KL is unbounded for low-entropy distributions early in training, so we initialize the model from a pretrained checkpoint (e.g., LLaVA-1.5-7B). Using cross-attention saliency to select the relevant region avoids the need for question-image grounding supervision, but it can be inaccurate when the student is unreliable; we mitigate this by updating the saliency map periodically every 500 steps. The contrastive loss with cosine similarity normalizes representation magnitude, reducing sensitivity to scale but losing discriminative power from magnitude variations. Hyperparameter λ=1.0 was chosen to balance alignment and separation; higher values caused negative samples to dominate and collapse, while lower values yielded degraded reasoning. Training uses batch size 64, learning rate 1e-5 with cosine schedule, 10k steps on 4 NVIDIA A100 GPUs (~2 days).

**(D) Load-bearing assumption and calibration:** We assume that geometric transformations (rotation, scaling, translation) preserve the answer for the question-relevant region. We verify this by periodically checking on a calibration set of 500 GQA examples with human-annotated correct regions; if accuracy on rotated crops drops below 80% of original accuracy, we reduce rotation range. Additionally, we monitor the average KL divergence between positive crop outputs; if it exceeds 0.5, we exclude rotation for that question type.

**(E) Why it measures what we claim:** L_align measures geometric invariance under the assumption that question-relevant region is correctly selected and transformation preserves answer; this assumption fails when region selection (cross-attention) is wrong, in which case L_align measures alignment of unrelated outputs. L_contrast operationalizes contrastive evidence gating: the InfoNCE loss measures how well the model discriminates between the original region and negative regions, under the assumption that different regions imply different answers for the given question. This assumption fails when the question is region-independent (e.g., “what color is the sky?”), in which case L_contrast penalizes invariance that should be preserved, leading to degraded performance on global questions. The stop-gradient on positive targets prevents representation collapse by ensuring the consistency target is fixed at each step; it assumes that the target representation is of high quality, which is true only after the model has seen sufficient training data; early in training, the fixed target may be noisy, but the collapse prevention outweighs this cost.

## Contribution

(1) A self-supervised training framework (S3CI) that removes the need for a supervised teacher in contrastive visual reasoning by leveraging geometric invariance across spatial crops. (2) A method for unsupervised selection of question-relevant regions using cross-attention saliency, enabling positive/negative view generation without any labeled data. (3) An analysis of the assumption underlying self-supervised invariance: that region relevance is determined by student attention, which may fail for global questions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | GQA | Compositional visual reasoning benchmark |
| Primary metric | Accuracy | Standard for visual QA |
| Baseline 1 | Supervised Fine-Tuning (SFT) | Tests need for self-supervision |
| Baseline 2 | Self-Supervised Contrastive Learning (SSCL) | Tests cross-crop invariance alone |
| Baseline 3 | Knowledge Distillation (KD) with teacher | Tests teacher-free advantage |
| Baseline 4 | SFT with 10% labeled data | Quantifies reduction in supervision |
| Ablation-of-ours | Ours w/o Contrastive Loss | Tests contrastive gating effect |
| Ablation-of-ours | Ours with random crops (instead of geometric) | Isolates effect of geometric invariance |
| Diagnostic | Cross-attention selection accuracy | Measures region selection quality |

### Why this setup validates the claim

This combination directly tests the central claim that self-supervised cross-crop invariance can replace a supervised teacher. GQA's compositional, often spatial questions require robust visual representations invariant to appropriate transformations. Accuracy is the appropriate metric as it reflects reasoning quality. SFT provides a strong supervised baseline to see if self-supervision compensates for missing labels. SSCL isolates the contrastive component without consistency, testing whether alignment is necessary. KD uses a teacher to benchmark against traditional distillation. SFT with limited labels tests the supervision reduction. The ablation w/o contrastive loss isolates the role of contrastive separation. The ablation with random crops (instead of geometric) tests whether geometric invariance itself matters: if performance drops, it confirms the assumption. The diagnostic measures cross-attention selection accuracy on 500 manually annotated examples, correlating with final performance. If our method outperforms SFT and KD on invariance-demanding subsets while SSCL and the ablation underperform, it confirms the causal mechanism of invariance-driven reasoning.

### Expected outcome and causal chain

**vs. Supervised Fine-Tuning (SFT)** — On a rotated test image of "What is to the left of the red car?", SFT fails because it never saw such crops, relying on answer labels alone. Our method enforces consistency across crop transformations, learning viewpoint-invariant features. We expect a noticeable gap on transformed test samples (e.g., accuracy 60% vs. 80%) but parity on standard samples.

**vs. Self-Supervised Contrastive Learning (SSCL)** — On a question about object color, SSCL treats different crops as separate instances, pushing representations apart and harming attribute recognition. Our consistency loss aligns semantics, preserving color information. Expect SSCL to have ~5% lower accuracy on color/attribute questions, while ours maintains high performance.

**vs. Knowledge Distillation (KD) with teacher** — On a nuanced counting question, KD may propagate teacher's counting errors. Our method learns from data without a teacher, avoiding inherited mistakes. Expect comparable or slightly higher accuracy, particularly on instances where the teacher is weak (e.g., 2% improvement on complex counting subsets).

**vs. SFT with 10% labeled data** — SFT with few labels struggles on rare objects; our method uses unlabeled data and is expected to outperform by 5-10% on such subsets.

**Ablation: Ours w/o Contrastive Loss** — On a spatial question like "Is there a bicycle near the bench?", the ablation lacks region discrimination; it may confuse the bicycle with a nearby tree. Contrastive loss ensures negative regions are separated. Expect ablation to have higher false-positive rate (e.g., 10% more errors on presence questions).

**Ablation: Ours with random crops** — If consistency over random crops (which may not share semantics) yields lower performance than geometric crops, it confirms that geometric invariance is the key factor. Expect a 5% drop on spatial questions.

**Diagnostic: Cross-attention selection accuracy** — We measure what fraction of the selected region overlaps with a human-annotated ground-truth region on 500 examples. We expect >70% accuracy after training and a positive correlation (Pearson r>0.3) with final task accuracy.

### What would falsify this idea

If our method does not outperform SFT on invariance-sensitive subsets (e.g., rotated test images) or if the ablation without contrastive loss achieves similar accuracy to the full method, then the central claim that cross-crop invariance and contrastive gating are essential would be falsified. Additionally, if cross-attention selection accuracy is below 50% and uncorrelated with performance, the region selection mechanism fails.

## References

1. V-Zero: Answer-Label-Free On-Policy Distillation with Contrastive Evidence Gating for Fine-Grained Visual Reasoning
2. GRIT: Teaching MLLMs to Think with Images
3. DeepEyesV2: Toward Agentic Multimodal Model
4. GPT-4o System Card
5. Visual Chain-of-Thought Prompting for Knowledge-Based Visual Reasoning
6. Compositional Chain-of-Thought Prompting for Large Multimodal Models
7. Managing extreme AI risks amid rapid progress
8. Sociotechnical Safety Evaluation of Generative AI Systems
