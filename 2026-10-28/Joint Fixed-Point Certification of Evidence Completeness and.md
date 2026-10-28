# Joint Fixed-Point Certification of Evidence Completeness and Hop Validity for Multi-Region Spatial Reasoning

## Motivation

Existing multi-region reasoning pipelines such as EvidenceConnect (Zooming without Zooming v2) and InSight-o3 rely on answer-level proxies—cross-VLM agreement and region ablation—to validate evidence chains, but these proxies conflate two structurally distinct properties: whether the selected crop set is complete and whether each intermediate reasoning hop is logically valid. Because a single answer-level signal changes under both a missing region and an invalid hop, the two properties are never separately identified, so a chain can pass validation while being incomplete or containing a plausible-but-unrelated hop. This meta-gap recurs across the frontier: HiDe, ZoomEye, and Divide-Conquer-Combine all optimize region selection but never certify the logical chain that connects regions.

## Key Insight

A scalar certificate defined as the joint fixed point of residual answer entropy (after conditioning on the crop set) and counterfactual hop consistency is zero iff the crop set is complete and every hop is valid, because the two perturbations that break completeness and hop validity move these quantities in orthogonal directions, making the pair identifiable from contrastive data.

## Method

**What it is.** We introduce **CERT-CHAIN**, a learned scalar certificate `C(q, R, H)` for a question `q`, a selected crop set `R = {r_1,...,r_k}`, and an intermediate reasoning chain `H = {h_1,...,h_m}` that links the crops. `CERT-CHAIN` is trained with a contrastive objective that pairs a complete-and-valid chain (label 0) against minimally perturbed chains that are either incomplete (one necessary region removed) or hop-invalid (one region swapped for a plausible but logically unrelated one). At inference, `CERT-CHAIN` scores candidate evidence chains and can be used to reject or re-query low-certificate chains before answering.

**Load-bearing assumption (stated explicitly).** The certificate is zero iff the chain is both complete and valid **only if** (i) the residual-entropy score `s_comp` reflects true region completeness rather than VLM confidence, and (ii) the counterfactual substitution used for `s_hop` changes only hop validity and leaves `s_comp` unchanged. We do not assume these hold a priori; we test them empirically (see step 5) and report the cross-Jacobian of `(s_comp, s_hop)` under each perturbation. If the cross-terms are non-negligible, we report the two scores as a 2-D certificate and do not claim scalar identifiability.

**How it works.**

```pseudocode
# Inputs: question q, crop set R, hop chain H = [(r_i -> r_j, claim_ij)]
# Hyperparameters: temperature tau=0.1, fixed-point damping lambda=0.5, entropy bins B=32

1. RESIDUAL ENTROPY (region-completeness score):
   a. For each crop r in R, obtain VLM answer distribution p(a | q, r) over the
      answer vocabulary A (top-K=20 logits, renormalized).
   b. Compute the joint answer distribution under the crop set via a
      product-of-experts with learned per-crop weights w_r (softmax over
      attention weights from q to r):
         p_R(a) = softmax( sum_{r in R} w_r * log p(a | q, r) )
   c. Residual entropy: E_R = H( p_R(a) ) / log|A|  (normalized to [0,1]).
   d. Region-completeness score: s_comp = 1 - E_R.

2. HOP-VALIDITY SCORE (counterfactual consistency):
   a. For each hop (r_i -> r_j, claim_ij), form a counterfactual crop set
      R' = R \ {r_i} U {r_i'}, where r_i' is a region with high visual
      similarity to r_i (CLIP cosine > 0.85) but low question relevance
      (attention weight < 0.1). If no candidate satisfies CLIP > 0.85,
      fall back to nearest-neighbor retrieval over the dataset's own crops
      (minimum CLIP cosine 0.70) and record the fallback rate.
   b. Recompute p_R'(a) as in step 1b.
   c. Hop-validity for this hop: v_ij = 1 - TV( p_R(a), p_R'(a) ), where TV is
      total variation distance. A valid hop should be robust to substituting
      an irrelevant region; an invalid hop (one that depends on the wrong
      region) will shift the answer distribution.
   d. Chain hop-validity: s_hop = min_{ij} v_ij  (bottleneck: one invalid hop
      invalidates the chain).

3. JOINT FIXED POINT:
   Initialize c_0 = 0.5.
   Repeat until |c_t - c_{t-1}| < 1e-4 or t > 20:
      c_t = lambda * c_{t-1} + (1 - lambda) * f( s_comp, s_hop, c_{t-1} )
   where f( s_comp, s_hop, c ) = sigmoid( (s_comp + s_hop - 1) / tau + c )
   (the fixed point couples the two scores through the certificate itself).
   Return certificate C = c_T.

4. CONTRASTIVE TRAINING:
   For each training question, construct:
     - Positive chain: complete crop set + valid hop chain (C should be 0).
     - Negative chain A (incomplete): remove one necessary region (C should be 1).
     - Negative chain B (invalid hop): substitute one region with a visually
       similar but question-irrelevant one (C should be 1).
   Loss: L = sum over pairs of max(0, margin + C_pos - C_neg) with margin=0.5,
   plus a calibration term: MSE( C_pos, 0 ) + MSE( C_neg, 1 ).
   Train the per-crop weights w_r and the fixed-point temperature tau jointly
   with the VLM frozen (only a small 2-layer MLP over [s_comp, s_hop, c] is
   learned, 10K parameters).

5. ASSUMPTION CHECKS (run once on a held-out calibration set of 512 examples):
   a. Calibration of s_comp: compute expected calibration error (ECE, 10 bins)
      of the VLM's answer confidence on incomplete crop sets (constructed by
      removing one necessary region from ground-truth complete sets). If ECE > 0.1,
      s_comp is treated as a calibration readout and we report the 2-D certificate
      (s_comp, s_hop) instead of the scalar C.
   b. Orthogonality check: for each perturbation type (incomplete, hop-invalid),
      compute the finite-difference cross-Jacobian entries
      J_comp,hop = (s_comp(perturbed) - s_comp(original)) / (s_hop(perturbed) - s_hop(original))
      and J_hop,comp analogously. Report mean and 95% CI. If |J| > 0.2, the axes are
      not orthogonal and we report the 2-D certificate.
```

**Why this design.** We chose a *joint fixed point* over a simple linear combination of `s_comp` and `s_hop` because the two scores are not independent—an incomplete crop set can artificially inflate hop-validity (fewer hops to check) and an invalid hop can mask residual entropy (the model becomes confidently wrong)—so a fixed point forces the certificate to be consistent under both perturbations simultaneously, accepting the cost that training requires solving a small fixed-point iteration per example (20 steps, negligible). We chose *counterfactual region substitution* over region ablation for hop-validity because ablation removes evidence entirely and thus changes both completeness and validity, whereas substitution with a visually similar but question-irrelevant region isolates the hop's logical dependence, accepting the cost that we need a similarity-thresholded substitute pool (CLIP > 0.85) which may be unavailable for rare regions (fallback: nearest-neighbor retrieval, minimum CLIP 0.70). We chose a *bottleneck min* over a mean for chain hop-validity because a single invalid hop should invalidate the whole chain, accepting the cost that a single noisy hop estimate can dominate—mitigated by the fixed-point smoothing. We chose a *contrastive objective with both perturbation types* over answer-level correctness because it explicitly separates the two properties, accepting the cost that we must generate paired negatives at training time (automatic via the perturbation operators). This differs from EvidenceConnect (Zooming without Zooming v2), which uses cross-VLM agreement and region ablation as a single answer-level filter and never identifies hop validity; a domain expert would not describe CERT-CHAIN as a variant of EvidenceConnect because EvidenceConnect never computes a counterfactual over hops or a fixed-point certificate—it discards questions that fail ablation, whereas CERT-CHAIN *scores* chains and is trained to be sensitive to two orthogonal perturbations.

**Why it measures what we claim.** `s_comp` measures *region completeness* because residual entropy after conditioning on the crop set is low only when the crop set contains all evidence needed to pin down the answer distribution; this assumption fails when the VLM is confidently wrong on an incomplete set (overconfidence), in which case `s_comp` reflects the VLM's calibration rather than true completeness. `s_hop` measures *hop validity* because total variation under substitution of a visually similar but question-irrelevant region is small only when the hop does not depend on the substituted region's identity; this assumption fails when the substitute region is accidentally relevant (similarity threshold too low), in which case `s_hop` reflects visual similarity rather than logical dependence. The fixed-point certificate `C` measures *joint sufficiency* because the sigmoid coupling forces `C` to be near zero only when both `s_comp` and `s_hop` are high and the fixed point is stable; this assumption fails when the two scores are both moderately high but for conflicting reasons (e.g., incomplete set with an invalid hop that cancels), in which case `C` reflects a compromise rather than a true certificate—mitigated by the bottleneck min in `s_hop` and the contrastive training that penalizes such cancellations. We explicitly test the calibration and orthogonality assumptions in step 5 and fall back to a 2-D certificate if they fail.

## Contribution

['A joint fixed-point certificate (CERT-CHAIN) that provably separates region completeness from hop validity by coupling residual answer entropy and counterfactual hop consistency, with a scalar output that is zero iff both properties hold.', 'A contrastive training scheme that generates minimally perturbed negative chains (one region removed vs. one region substituted) to force the certificate to be sensitive to each property independently, enabling identifiability from the same data.', 'An inference-time scoring procedure that can reject or re-query low-certificate evidence chains, applicable to any multi-region reasoning pipeline without retraining the base VLM.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | V*Bench + HRBench4K | High-res spatial questions needing multi-crop evidence chains |
| Primary metric | Chain-rejection AUROC | Directly measures certificate's ability to separate valid vs invalid chains |
| Baseline | Base VLM (no certificate) | Tests whether scoring chains adds signal beyond raw answering |
| Baseline | EvidenceConnect filter | Tests whether hop-validity adds signal beyond ablation filter |
| Baseline | Linear-combination certificate | Tests whether fixed-point coupling beats naive score fusion |
| Ablation-of-ours | CERT-CHAIN minus s_hop | Isolates whether hop-validity term drives the predicted gain |
| Ablation-of-ours | CERT-CHAIN with 2-D certificate (s_comp, s_hop) | Tests whether scalar fixed point is necessary vs reporting both scores |
| Diagnostic | Cross-Jacobian of (s_comp, s_hop) under each perturbation | Tests orthogonality assumption; if |J|>0.2, scalar certificate claim is dropped |
| Diagnostic | ECE of VLM confidence on incomplete crop sets (512 held-out examples) | Tests calibration assumption; if ECE>0.1, s_comp is treated as calibration readout |
| Diagnostic | Fraction of questions with valid substitute (CLIP>0.85 or fallback NN) | Bounds feasibility of hop-validity score |

### Why this setup validates the claim

The central claim is that CERT-CHAIN's certificate separates valid evidence chains from minimally perturbed invalid ones, and that this separation is what drives downstream answer accuracy on questions requiring multi-region spatial reasoning. V*Bench and HRBench4K are the right testbeds because they contain questions where the answer is only recoverable by chaining evidence across crops, so a certificate that fails to detect an incomplete or hop-invalid chain will visibly degrade accuracy on exactly those items. The base VLM baseline tests the sub-claim that raw answering confidence is insufficient — if CERT-CHAIN only matches it, the certificate adds nothing. EvidenceConnect tests the sub-claim that hop-validity is a distinct axis: it filters by cross-VLM agreement and ablation, so if CERT-CHAIN's gain over it is concentrated on hop-invalid cases, the hop term is doing real work. The linear-combination baseline tests the fixed-point sub-claim: if coupling is unnecessary, a simple weighted sum of s_comp and s_hop should match, so any gap isolates the value of the joint fixed point. The s_hop ablation is the sharpest test: removing the hop term should collapse performance specifically on hop-invalid negatives while leaving incomplete-negatives detection intact. Chain-rejection AUROC is the right primary metric because it directly measures the certificate's discriminative power over the perturbation pairs the method is trained on, independent of any downstream answering head, so a null result there falsifies the mechanism rather than a pipeline. The added diagnostics (cross-Jacobian, ECE on incomplete sets, substitute coverage) directly test the load-bearing assumptions stated in the method; if they fail, we report the 2-D certificate and do not claim scalar identifiability.

### Expected outcome and causal chain

**vs. Base VLM (no certificate)** — On a case where a question requires linking a small object in one crop to a spatial relation visible only in another, the base VLM produces a confident answer from the single most salient crop because it has no mechanism to check whether the crop set is complete. Our method instead computes residual entropy over the joint crop distribution and flags the chain when s_comp is low, so we expect a noticeable AUROC gap on the incomplete-chain subset and near-parity on single-crop questions where completeness is trivially satisfied. If the ECE diagnostic shows high miscalibration on incomplete sets, we expect this gap to shrink and will report the 2-D certificate instead.

**vs. EvidenceConnect filter** — On a case where the crop set is complete but one hop substitutes a visually similar distractor region for the correct one, EvidenceConnect accepts the chain because cross-VLM agreement and ablation both pass — the answer distribution is stable under removing a region that was never load-bearing. Our method instead detects the hop as invalid via counterfactual substitution and the bottleneck min propagates that to the certificate, so we expect a concentrated gain on hop-invalid negatives with parity on incomplete negatives, which is the diagnostic signature that hop-validity is a separate axis. If the cross-Jacobian diagnostic shows non-negligible coupling, we expect this gain to be attenuated and will report the 2-D certificate.

**vs. Linear-combination certificate** — On a case where an incomplete crop set inflates hop-validity (fewer hops, each trivially robust) while a subtly invalid hop masks residual entropy, the linear combination averages the two scores into a middling value and misranks the chain as acceptable. Our method's fixed point forces consistency under both perturbations simultaneously, so we expect a modest but consistent AUROC edge concentrated on the adversarial subset where the two scores disagree, rather than a uniform lift. If the fixed-point edge is absent, we will report the 2-D certificate as the primary output.

### What would falsify this idea

If CERT-CHAIN's gain over EvidenceConnect is uniform across incomplete-chain and hop-invalid subsets rather than concentrated on hop-invalid cases, the hop-validity mechanism is not doing the work we claim. Likewise, if the s_hop ablation leaves chain-rejection AUROC essentially unchanged, the counterfactual substitution signal is redundant and the central two-axis claim is wrong. Additionally, if the cross-Jacobian diagnostic shows |J|>0.2 or ECE on incomplete sets exceeds 0.1, the scalar fixed-point certificate is not identifiable and we will report the 2-D certificate instead.

## References

1. Zooming without Zooming v2: From Seeing Details to Connecting Evidence for Generalizable Multimodal Perception
2. HiDe: Rethinking The Zoom-IN method in High Resolution MLLMs via Hierarchical Decoupling
3. InSight-o3: Empowering Multimodal Foundation Models with Generalized Visual Search
4. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
5. ZoomEye: Enhancing Multimodal LLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration
6. Divide, Conquer and Combine: A Training-Free Framework for High-Resolution Image Perception in Multimodal Large Language Models
