# Uncertainty-Sampled Benchmark Augmentation via Directional Perturbation

## Motivation

Existing multimodal evaluation benchmarks like Blind-Spots-Bench rely on static, human-annotated test cases, limiting their scalability and ability to keep pace with model evolution. The small sample size (235) and lack of automation mean that blind spots revealed by the benchmark may become outdated as models improve. A self-supervised mechanism is needed that uses model uncertainty to dynamically generate new test cases without human effort, but prior work lacks a principled way to create answer-preserving variations that target specific failure modes.

## Key Insight

Ensemble disagreement across models defines a directional signal in perturbation space that identifies which transformations of a test case expose inconsistent model behaviors, enabling targeted generation of challenging yet answer-preserving examples.

## Method

(A) What it is:
Uncertainty-Sampled Benchmark Augmentation (USBA) takes a base set of multimodal test cases { (image_i, text_i, answer_i) } and a pool of perturbation operators (e.g., blur with σ∈[0.1,2.0], colour shift with Δ∈[−50,50] in HSV, synonym replacement via WordNet with threshold similarity≥0.8). It outputs an expanded benchmark containing perturbed versions of base cases for which an ensemble of models shows a statistically significant increase in uncertainty, ensuring new cases probe blind spots without human annotation. A load-bearing assumption: all perturbation operators preserve the ground-truth answer for every base test case. We add an automatic answer-preservation check: after applying a perturbation, we verify that a holdout model (e.g., CLIP ViT-B/32 zero-shot classifier) predicts the same answer; perturbed cases where the answer changes are discarded.

(B) How it works:
```pseudocode
Input: Base benchmark B = {(img, txt, ans)}_i=1..N, Models M = {m1..mK} with K=4 (e.g., ViLT, LXMERT, UNITER, CLIP), Perturbation operators P = {p : X -> X} (each with a continuous parameter θ)
Output: Expanded benchmark B' = B ∪ { (img'_i, txt'_i, ans_i) }

For each (img_i, txt_i, ans_i) in B:
    // 1. Compute base uncertainty
    For each model m_j:
        p_j = m_j.predict_proba(img_i, txt_i)  // distribution over answer choices (over vocabulary of 3129 answers)
    U_i = 1 - mean( PAIRWISE_KL(p_j, p_l) )   // higher = more disagreement (uncertainty)

    // 2. For each perturbation p_k, sample parameter θ uniformly from [θ_min, θ_max] (e.g., σ∼U(0.1,2.0) for blur; Δ∼U(−50,50) for colour shift)
    For p_k in P:
        img'_ik, txt'_ik = p_k(img_i, txt_i; θ)
        // Answer-preservation check: compute pred_holdout = m_holdout.predict(img'_ik, txt'_ik). If pred_holdout != ans_i, skip this perturbation.
        If m_holdout.predict(img'_ik, txt'_ik) != ans_i: continue
        For m_j: p'_jk = m_j.predict_proba(img'_ik, txt'_ik)
        U'_ik = 1 - mean( PAIRWISE_KL(p'_jk, p'_lk) )
        ΔU_ik = U'_ik - U_i

    // 3. Select perturbations that increase uncertainty significantly
    p* = argmax_{p_k} mean(ΔU_ik over M_θ samples; M_θ=5 uniform samples per p_k)  // directional search
    if ΔU_ik* > τ (threshold = 0.1 * std(U_i across all base cases)):
        Add (img'_ik*, txt'_ik*, ans_i) to B'

// Additional step: remove duplicates via feature matching (CLIP similarity >0.95)
```

(C) Why this design:
We chose ensemble disagreement as the uncertainty signal over single-model log-probs because log-probs are poorly calibrated for correctness; ensemble disagreement provides a cheap, model-agnostic proxy that does not rely on calibration. This accepts the cost of requiring multiple forward passes, which is acceptable for benchmark construction. We used a continuous perturbation parameter space and sampling multiple θ values instead of a fixed perturbation grid because fine-grained parameter search increases the chance of finding a transformation that truly triggers failure, at the cost of more compute. We avoided training a generative model (e.g., VAE or diffusion) because such models can introduce artifacts that change the answer in unverifiable ways; perturbation operators are predefined to be semantically neutral (e.g., brightness change does not alter the answer to 'what color is the car?'). This restricts the diversity of generated cases but guarantees answer preservation without human verification, which is critical for unsupervised expansion. The threshold τ is set adaptively per base case based on the standard deviation of U across the full set, not as a fixed global value, because different base cases naturally have different base uncertainty levels; this prevents bias toward overly easy or hard cases but introduces sensitivity to outlier statistics. The answer-preservation check via a holdout model (CLIP ViT-B/32) explicitly validates the load-bearing assumption; if the holdout's prediction changes, the perturbation is discarded, reducing the risk of conflating answer change with blind-spot exposure.

(D) Why it measures what we claim:
The quantity ΔU_ik measures *blind-spot exposure* for perturbation p_k because the increase in ensemble disagreement after perturbation indicates that the models' predictions become inconsistent, which is a direct symptom of operating outside their reliable regime; this assumption fails when the ensemble is already highly uncertain on the base case (U_i is high), in which case ΔU_ik becomes small regardless of actual failure, and instead reflects ceiling effects of the uncertainty metric. The perturbation selection by argmax over mean ΔU_ik measures *targeted failure induction* because we hypothesize that the perturbation that maximally increases disagreement is most likely to expose a systematic blind spot; this assumption fails when the perturbation changes the answer (despite our answer-preservation check), in which case the high ΔU_ik reflects answer change rather than blind-spot exposure, so we rely on the holdout model to catch such cases. The pairwise KL divergence used in U_i measures *model disagreement* because it quantifies the average distance between output distributions of different models; this assumes the models are diverse enough (different architectures/train data) to capture genuine blind spots rather than correlated errors, a failure mode that occurs if the ensemble is homogeneous (e.g., all open-weight variants), in which case disagreement reflects residual variance not blind-spot structure.

## Contribution

(1) A self-supervised algorithm (USBA) that dynamically expands a multimodal benchmark by selecting perturbation operators that maximize ensemble disagreement, without any human annotation. (2) The design principle that directional perturbation search over a small set of semantically neutral transformations can expose blind spots as effectively as complex generative approaches, validated by the monotonic increase in disagreement correlated with known failure modes. (3) An automatic mechanism for answer-preserving test case generation that does not require ground-truth labels for new cases.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| ---- | ------ | --------- |
| Dataset | VQAv2 val | Standard multimodal benchmark, 214k images. |
| Primary metric | Accuracy drop on augmented cases | Measures how challenging new cases are. |
| Secondary metric | Answer-preservation rate via CLIP (ViT-B/32) | Validates that perturbations preserve original answer; rate >95% expected. |
| Baseline | Random perturbation | Adds same count perturbed cases randomly. |
| Baseline | Original benchmark only | No augmentation baseline. |
| Ablation | Fixed-threshold USBA | Uses global τ instead of adaptive (set to median of adaptive thresholds). |
| Ablation | USBA without answer-preservation check | Quantifies impact of answer-change contamination. |

### Why this setup validates the claim
This combination tests whether uncertainty-guided augmentation discovers harder cases than random addition, whether augmentation itself exposes blind spots, and whether the adaptive threshold is crucial. The accuracy drop metric directly quantifies how challenging the augmented cases are for models. Comparing to random perturbation isolates the effect of uncertainty selection; comparing to the original benchmark isolates the effect of augmentation; and the fixed-threshold ablation isolates the effect of adaptivity. Additionally, the answer-preservation rate measures the validity of the load-bearing assumption: if the rate is high (>95%), then ΔU primarily reflects blind-spot exposure rather than answer change. The ablation without the check reveals how much contamination occurs. If USBA causes a significantly larger accuracy drop than both baselines, and the adaptive version outperforms the fixed one, and the answer-preservation rate is high, then the central claim—that uncertainty-guided perturbation selection effectively exposes blind spots while preserving answers—is supported.

### Expected outcome and causal chain

**vs. Random perturbation** — On a case where a subtle blur does not affect models, random perturbation may select it, yielding no accuracy change. Our method instead selects a color shift that increases ensemble disagreement because it exploits model-specific sensitivities, so we expect USBA's augmented subset to show a noticeably larger accuracy drop than the random subset on visually ambiguous cases, but similar accuracy on easy cases.

**vs. Original benchmark only** — On a case like an image with unusual lighting, models may agree on the original but disagree after a contrast perturbation that exposes a blind spot. Our method adds exactly such cases via uncertainty increase, while the original benchmark lacks them. Thus we expect model accuracy on USBA-added cases to be substantially lower than on original cases, with the gap concentrated on cases where base uncertainty was low.

**vs. Fixed-threshold USBA** — On a base case with already high uncertainty, a fixed threshold may include too many trivial perturbations, diluting the benchmark. Our adaptive threshold requires a larger relative increase, so it selects perturbations that cause a genuine failure pattern. Consequently, we expect adaptive USBA to yield a smaller but more effective set with a higher per-case accuracy drop compared to fixed USBA.

**vs. USBA without answer-preservation check** — On cases where a perturbation accidentally changes the answer, USBA without the check may include such instances, inflating ΔU and accuracy drop due to answer change, not blind-spot exposure. Thus we expect the version with the check to have a higher answer-preservation rate (target >95%) and a slightly lower but more reliable accuracy drop.

### What would falsify this idea
If the accuracy drop on USBA-augmented cases is not significantly larger than that on random perturbation, or if the adaptive threshold does not outperform the fixed one, or if the answer-preservation rate is below 95% (indicating frequent answer changes), then the claim that uncertainty-guided selection and adaptivity are essential and that the answer-preservation assumption is valid is falsified.

## References

1. Blind-Spots-Bench: Evaluating Blind Spots in Multimodal Models
2. SIV-Bench: A Video Benchmark for Social Interaction Understanding and Reasoning
