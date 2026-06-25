# Cross-Modal Grounding for Factual Atomic Claims in Multimodal Misinformation Detection

## Motivation

Existing multimodal misinformation detection methods (e.g., ReMMD) rely on LLM-generated atomic claims without verifying their factual grounding in input evidence, leading to hallucinated or unsupported claims. This structural flaw arises because no mechanism ensures that each claim is causally attributed to specific visual or textual evidence, allowing the LLM to produce plausible but unsubstantiated statements. Prior work like MMFakeBench and Can LLMs Improve assume LLM outputs are trustworthy, perpetuating this issue.

## Key Insight

The factual accuracy of LLM-generated claims can be guaranteed by enforcing a differentiable cross-modal grounding constraint that requires each claim's attention over input evidence to entail its semantic content, thereby eliminating unsupported generations.

## Method

(A) **What it is**: Grounded Multimodal Claim Verification (GMCV) is a training framework that takes a multimodal post (image I, text T) and outputs a set of atomic claims {c_i}, each coupled with an attention mask over input evidence that causally supports the claim. The model is trained to minimize a combined loss of claim correctness and grounding fidelity. (B) **How it works**: 
```python
# Hyperparameters: λ = 0.3, NLI_model = DeBERTa-v3-large
for (I, T, ground_truth_claims) in dataset:
    # Encode image and text
    img_feats = CLIP_vision_encoder(I)  # [num_patches, d]
    txt_feats = CLIP_text_encoder(T)    # [num_tokens, d]
    # LLM decoder generates claims autoregressively with cross-attention
    claims, cross_attn = LLM_decoder(img_feats, txt_feats)  # cross_attn: [num_claims, num_patches+num_tokens]
    for c_i, attn_i in zip(claims, cross_attn):
        # Grounding loss: attend to evidence that entails c_i
        # Sample top-k evidence tokens by attn_i (k=3)
        top_indices = attn_i.topk(k=3).indices
        evidence_set = [img_feats[idx] if idx < num_patches else txt_feats[idx-num_patches] for idx in top_indices]
        # Compute entailment score via NLI model (evidence as premise, c_i as hypothesis)
        entail_score = NLI_model(evidence_set, c_i)  # [0,1]
        L_ground_i = -log(mean(entail_score))  # encourage high entailment
    # Claim correctness loss (e.g., cross-entropy with ground truth claims)
    L_claim = cross_entropy(claims, ground_truth_claims)
    L_total = L_claim + λ * mean(L_ground_i)
    update LLM_decoder parameters via gradient descent
```
(C) **Why this design**: We chose a pretrained NLI model for entailment scoring over training a separate verifier because the NLI model provides a strong inductive bias about factual consistency without requiring additional annotations, accepting the cost that its errors can misguide grounding when the claim and evidence share spurious lexical overlap. We selected the top-k attention indices (k=3) instead of all attended tokens to focus on the most salient evidence and avoid dilution from irrelevant regions, sacrificing some recall for precision. We used a single LLM decoder that jointly attends to both modalities rather than separate encoders for images and text, because cross-modal attention allows the model to directly associate textual claims with visual features, accepting the computational overhead of full cross-attention. The grounding loss is applied per claim rather than globally to ensure each individual claim is supported, rather than allowing some claims to be unsupported as long as the overall claim set is plausible. (D) **Why it measures what we claim**: The entailment score from the NLI model measures factual consistency between claim and evidence under the assumption that the NLI model is approximately monotonic in evidence sufficiency; this assumption fails when the NLI model is fooled by spurious correlations (e.g., lexical overlap), in which case L_ground rewards claims that are lexically similar but not factually entailed. The cross-attention weights measure the model's allocation of evidence attention, under the assumption that attention reflects causal relevance; this fails when attention is dominated by image position or text frequency biases, in which case the grounding loss optimizes for attention that is not truly causal. The top-k sampling ensures that only the most strongly attended evidence is considered, under the assumption that the strongest attention corresponds to necessary evidence; this fails when the necessary evidence has low attention due to initialization or learning dynamics, leading to incomplete grounding. Despite these risks, the combination provides a practical gradient signal that pushes claims toward supportedness, and the assumption failures can be mitigated by periodic calibration using human-annotated grounding.

## Contribution

(1) GMCV, a training framework that introduces a differentiable cross-modal grounding constraint for LLM-generated atomic claims in multimodal misinformation detection, enforcing that each claim must be entailed by attended input evidence. (2) A design principle that factual accuracy at the intermediate level can be operationalized via a pretrained NLI model and attention supervision, without requiring expensive human annotations for grounding. (3) An analysis of the assumptions underlying the grounding loss (entailment monotonicity, attention causality) and their failure modes, providing a diagnostic for when the method may fail.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|---------|-----------|
| Dataset | MMFakeBench subset with claim annotations | Tests mixed-source multimodal claims |
| Primary metric | Claim entailment accuracy | Directly measures claim correctness |
| Baseline 1 | LLaVA (no grounding) | Assesses need for explicit grounding |
| Baseline 2 | Text-only NLI | Isolates visual grounding contribution |
| Ablation | GMCV w/o grounding loss | Quantifies grounding loss effect |

### Why this setup validates the claim

This combination tests the central claim—that explicit grounding via NLI entailment improves multimodal claim verification. MMFakeBench's mixed-source data (AI-generated, real, etc.) challenges both visual and textual reasoning, making it a stringent test. Comparing to LLaVA reveals whether explicit grounding outperforms implicit multimodal reasoning. Text-only NLI isolates the contribution of visual evidence, which is critical for verifying image-dependent claims. The ablation directly measures the impact of the grounding loss, controlling for architectural confounds. Claim entailment accuracy aligns with the method's training objective, providing a direct measure of the claimed benefit.

### Expected outcome and causal chain

**vs. LLaVA** — On a case where an image contains subtle manipulation (e.g., a doctored object) and text is deceptive, LLaVA may rely on coarse multimodal agreement and miss the inconsistency because it lacks explicit evidence localization. GMCV attends to manipulated regions via cross-attention and the grounding loss ensures high entailment with correct evidence, so we expect a noticeable accuracy gap on visually grounded claims but parity on text-only claims.

**vs. Text-only NLI** — On a case where the image provides decisive counterevidence (e.g., a photo showing a different scene than the text), text-only NLI incorrectly verifies based solely on text because it ignores visual modality. GMCV incorporates both modalities and the grounding loss enforces consistency with visual evidence, so we expect a large gap on visually inconsistent claims and similar performance on text-sufficient claims.

**vs. GMCV w/o grounding loss** — On a case where a claim requires precise visual detail (e.g., "the sign says 30 mph"), the ablation may attend to irrelevant regions and produce an incorrect claim because no loss penalizes poor grounding. GMCV with grounding loss forces attention to the sign and high entailment, so we expect a gap on fine-grained visual claims but comparable performance on coarse or text-only claims.

### What would falsify this idea

If GMCV with grounding loss shows no improvement over the ablation on claims requiring visual grounding, or if its gain is uniform across all claim types rather than concentrated on visually grounded claims, then the central claim is wrong—the grounding loss is not causally responsible for any observed improvement.

## References

1. ReMMD: Realistic Multilingual Multi-Image Agentic Verification for Multimodal Misinformation Detection
2. MMFakeBench: A Mixed-Source Multimodal Misinformation Detection Benchmark for LVLMs
3. Can LLMs Improve Multimodal Fact-Checking by Asking Relevant Questions?
4. MiRAGeNews: Multimodal Realistic AI-Generated News Detection
5. QACHECK: A Demonstration System for Question-Guided Multi-Hop Fact-Checking
6. Detecting and Grounding Multi-Modal Media Manipulation and Beyond
7. EVA-CLIP: Improved Training Techniques for CLIP at Scale
8. LAION-5B: An open large-scale dataset for training next generation image-text models
