# Self-Fixed-Point Calibration for Open-Ended Video Generation Judges

## Motivation

Existing video-generation benchmarks (e.g., StreamEval, TC-Bench, WorldScore) all define judge correctness relative to a pre-specified human-authored target set, so calibration is only ever a distance to a known reference. When the target is open-ended and unscripted — exactly the regime StreamEval explicitly fails to cover — calibration is undefined: a judge can rank rollouts consistently yet be arbitrarily miscalibrated, because there is no reference against which to identify the generating model. This is a structural meta-gap across the paper-trees: the judge's scoring function is never required to be identifiable under its own re-conditioning.

## Key Insight

Calibration can be redefined as an identifiability condition: a judge is calibrated iff the ranking it induces on model-generated rollouts is invariant when its target set is replaced by the judge's own high-confidence generations, which forces the judge's scoring function to be a fixed point of its own re-conditioning rather than a distance to a human-authored reference.

## Method

# Self-Fixed-Point Calibration (SFPC)

**Load-bearing assumption (stated explicitly).** SFPC assumes that the judge's scalar score s_{k,p} = J(r_{k,p}; T) is a sufficient statistic for generator identity: that argmax_k s*_{k,p} recovers k_true at well above 1/K chance. This is the classic identifiability assumption of scalar preference/reward models, and it can fail — a judge can converge to a stable fixed point (high Delta_rank, |T*|/|P|*K >= 0.10) while Acc_id sits at chance, because iterating against the judge's own top-quantile outputs selects the region where its score is least discriminative. The method below makes this assumption testable rather than hidden: Acc_id is reported at t=0 alongside the fixed point, the target set is stratified per generator so it cannot be dominated by whichever generator the judge already over-scores, and a per-generator conditional-likelihood variant of Acc_id is reported alongside the argmax variant.

```pseudocode
# Self-Fixed-Point Calibration (SFPC)
Input: judge J (MLLM with scoring head), set of generators G = {g_1,...,g_K}, prompt suite P (open-ended, no human references)
Output: calibration certificate C(J), per-generator identifiability score

Phase 1 — Rollout collection
  for each p in P, for each g_k in G:
      r_{k,p} ~ g_k(p)                      # K rollouts per prompt

Phase 2 — Self-target construction (fixed-point iteration, stratified)
  T_0 = empty
  repeat t = 1..T_max (T_max=5):
      for each p, for each k:
          s_{k,p} = J(r_{k,p}; T_{t-1})       # score rollout against current self-target
      # Stratified per-generator quantile: hold target-set size fixed across generators
      for each k:
          tau_{k,t} = quantile({s_{k,p} : p in P}, q=0.90)   # per-generator 90th percentile
          T_{k,t} = { r_{k,p} : s_{k,p} >= tau_{k,t} }        # judge's high-confidence rollouts from g_k
      T_t = union_k T_{k,t}                  # stratified self-target, |T_t| = 0.10 * K * |P|
  until |T_t Δ T_{t-1}| / |T_t| < eps (eps=0.05)
  T* = T_t                                    # self-consistency fixed point

Phase 3 — Re-conditioning invariance test
  for each p, for each k:
      s*_{k,p} = J(r_{k,p}; T*)              # re-score against fixed-point target
  R  = rank of generators by mean s* over P
  R0 = rank of generators by mean s  at t=0 (no self-target)
  Delta_rank = Kendall-tau(R, R0)            # invariance statistic

Phase 4 — Identifiability recovery
  # (a) argmax identity recovery (original)
  for each p:
      k_hat = argmax_k s*_{k,p}              # judge's inferred generator
  Acc_id_argmax = mean_p [ k_hat == k_true(p) ]     # chance = 1/K
  # (b) per-generator conditional-likelihood recovery (added specification)
  #     J outputs a per-generator score vector v_{k,p} = [J(r_{k,p}; T*; g_1), ..., J(r_{k,p}; T*; g_K)]
  #     (the judge is prompted once per candidate generator identity; v is the vector of those K scores)
  for each p:
      k_hat_ll = argmax_k v_{k,p}            # likelihood-ratio identity
  Acc_id_ll = mean_p [ k_hat_ll == k_true(p) ]      # chance = 1/K
  # (c) t=0 identity recovery (reported so the certificate cannot be passed by a judge whose identity signal was destroyed by iteration)
  for each p:
      k_hat_0 = argmax_k s_{k,p} at t=0
  Acc_id_t0 = mean_p [ k_hat_0 == k_true(p) ]

Calibration certificate: C(J) = (Delta_rank, Acc_id_argmax, Acc_id_ll, Acc_id_t0, |T*|/|P|*K)
Judge is calibrated iff Delta_rank >= 0.95 AND max(Acc_id_argmax, Acc_id_ll) >= 0.80 AND |T*|/|P|*K >= 0.10 AND Acc_id_t0 >= 0.80
```

**Why this design.** We chose a fixed-point iteration over a single-pass self-target because a judge that is miscalibrated will produce a target set that drifts with each re-conditioning; only a converged fixed point certifies that the judge's ranking is invariant to its own re-conditioning, accepting the cost that some judges may not converge within T_max=5 and are flagged as uncalibrated rather than silently scored. We chose a stratified per-generator quantile threshold (q=0.90 within each generator) over a pooled quantile because a pooled top-10% pool is dominated by whichever generator the judge already over-scores, which is the exact confound that makes Acc_id uninformative; the stratified target holds |T*| fixed at 0.10*K*|P| across generators, accepting the cost that the target set is no longer a single global quantile. We chose identity-recovery accuracy (Acc_id) over correlation with human ratings because human ratings are exactly the unavailable reference the method is designed to replace, accepting the cost that Acc_id requires at least K>=3 generators to have non-trivial chance level; we report both the argmax variant and a per-generator conditional-likelihood variant so that a scalar-score identifiability failure is visible rather than hidden. This differs from StreamEval, TC-Bench, and WorldScore, which all fix the target set a priori; a domain expert would not describe SFPC as a variant of those benchmarks because the target set is an output of the judge's own scoring function, not an input.

**Why it measures what we claim.** Delta_rank measures re-conditioning invariance because the assumption is that a calibrated judge's ranking is invariant when its target set is replaced by its own high-confidence generations; this assumption fails when the judge has a target-independent bias (e.g., a fixed preference for high-motion rollouts) that yields a stable fixed point with high Delta_rank but chance Acc_id, in which case Delta_rank reflects self-consistency rather than calibration. Acc_id measures generator identifiability because the assumption is that a calibrated judge's score is monotone in true generator quality so argmax recovers identity; this assumption fails when the judge discriminates generators by spurious cues (resolution, watermark) yielding high Acc_id without calibration, in which case Acc_id reflects cue detection rather than calibration. The stratified target set measures the judge's revealed preference because the assumption is that per-generator high-confidence rollouts are comparable across generators; this assumption fails when one generator's score distribution is shifted wholesale, in which case the stratified target still holds size fixed but the per-generator thresholds differ.

## Contribution

["A self-fixed-point calibration procedure (SFPC) that replaces the human-authored target set with the judge's own high-confidence generations, turning calibration from a distance-to-reference into an identifiability condition testable without new annotation.", 'A calibration certificate consisting of three operational statistics — rank-invariance under re-conditioning (Kendall-tau), generator-identity recovery accuracy, and fixed-point target-set coverage — that jointly certify whether a judge is calibrated for open-ended video generation.', 'An empirical finding that judges passing standard agreement-with-human benchmarks can fail SFPC, because their rankings are not invariant under their own re-conditioning, exposing a miscalibration invisible to fixed-target evaluation.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Open-ended prompt suite across K≥3 video generators (K=4: two open-source, two proprietary; 200 prompts; 50-prompt pilot subset for convergence check) | No human references; enables identity-recovery test; pilot checks convergence before scaling |
| Primary metric | Acc_id_argmax and Acc_id_ll (generator identity recovery accuracy, argmax and per-generator conditional-likelihood) | Directly measures whether judge scores are informative; two variants expose scalar-score identifiability failure |
| Secondary metric | Acc_id_t0 (identity recovery at t=0, before iteration) | Reported so the certificate cannot be passed by a judge whose identity signal was destroyed by iteration |
| Secondary metric | Delta_rank (Kendall-tau between fixed-point ranking and t=0 ranking) | Re-conditioning invariance statistic |
| Baseline | StreamEval-style fixed-target scoring | Tests whether self-targeting beats a priori targets |
| Baseline | WorldScore-style multi-dimension rubric | Tests whether a static rubric is re-conditioning invariant |
| Baseline | TC-Bench-style human-correlation validation | Tests whether human-free calibration is even needed |
| Ablation-of-ours | SFPC with T_max=1 (single-pass self-target) | Isolates the contribution of fixed-point convergence |
| Ablation-of-ours | SFPC with pooled (non-stratified) quantile | Isolates the contribution of per-generator stratification |
| Control | Resolution/watermark-normalized rollouts | Tests whether Acc_id survives when spurious cues are removed |
| Adversarial judge | Constructed judge with target-independent motion bias | Demonstrates the failure mode: stable fixed point, high Delta_rank, chance Acc_id |
| Oracle check | Trained classifier on raw video features (e.g., 3D-ResNet, 512-d, linear probe) | Sanity check that K generators are distinguishable, so Acc_id chance-level is interpretable |

### Why this setup validates the claim

The central claim is that a judge's ranking is only trustworthy if it is invariant to its own re-conditioning, and that this invariance can be certified without human references. The setup operationalizes that claim as a falsifiable test: if self-targeting genuinely certifies calibration, then the fixed-point judge should show high Acc_id (well above the 1/K chance level) and high Delta_rank, while the fixed-target baselines—StreamEval and WorldScore—should show lower invariance because their target sets are fixed a priori and cannot adapt to the judge's own scoring drift. TC-Bench-style human-correlation validation tests the complementary sub-claim that human labels are not the only route to a trustworthy certificate: if our human-free certificate agrees with the ranking that human-correlation validation would produce, the claim that references are unnecessary is supported. The ablation with T_max=1 isolates the fixed-point mechanism: if the gain is due to convergence rather than to self-targeting per se, the single-pass variant should sit between the fixed-target baselines and full SFPC. The stratified-vs-pooled ablation isolates the per-generator target construction: if the gain is due to stratification rather than to self-targeting, the pooled variant should show lower Acc_id. The resolution/watermark control tests whether Acc_id reflects calibration or spurious cue detection. The adversarial judge with target-independent motion bias is the direct test of the load-bearing assumption: it should converge to a stable fixed point with high Delta_rank but chance Acc_id, demonstrating that the certificate's two criteria are not independent evidence. Acc_id is the right primary metric because it detects the exact effect the theory predicts—a calibrated judge must be able to recover which generator produced a rollout from its score alone—and it is sensitive to the predicted failure mode (a miscalibrated judge collapses toward chance), making the test diagnostic rather than merely descriptive. Acc_id_t0 is reported alongside the fixed point so that a judge whose identity signal was destroyed by iteration cannot pass the certificate.

### Expected outcome and causal chain

**vs. StreamEval-style fixed-target scoring** — On a case where the judge's scoring head has a systematic bias toward high-motion rollouts, the fixed-target baseline produces a ranking that is stable only because its target set never moves, masking the bias as consistency. Our method instead re-scores rollouts against the judge's own high-confidence set and iterates until the target stops drifting, because the fixed point is defined by the judge's own scoring function rather than an external anchor. We expect a noticeable gap in Delta_rank and Acc_id on the high-motion-heavy subset, with near-parity on subsets where the bias is weak.

**vs. WorldScore-style multi-dimension rubric** — On a case where controllability and quality dimensions conflict (a rollout that is high-quality but camera-inconsistent), the static rubric produces a composite score that is not invariant to re-conditioning because the rubric weights are fixed and cannot reflect which dimension the judge actually uses. Our method instead lets the target set emerge from the judge's own top-quantile scores, so the composite is implicitly re-weighted by the judge's revealed preference. We expect a moderate Acc_id advantage concentrated on multi-dimension-conflict prompts, with parity on single-dimension prompts.

**vs. TC-Bench-style human-correlation validation** — On a case where human raters disagree (ambiguous temporal composition), the human-correlation baseline produces a noisy reference that cannot certify the judge because the reference itself is unstable. Our method instead certifies the judge against its own fixed point, which requires no human labels and is therefore stable under rater disagreement. We expect our certificate to agree with the human-correlation ranking on unambiguous subsets but to remain stable where the human reference is noisy—observable as a divergence in Delta_rank only on the ambiguous subset.

**vs. SFPC with T_max=1 (single-pass self-target)** — On a case where the judge's score drifts across re-conditionings, the single-pass variant produces a target set that has not converged, so its ranking is not invariant. Our method iterates until the target stops drifting. We expect the single-pass variant to sit between the fixed-target baselines and full SFPC on Delta_rank and Acc_id, with the gap concentrated on prompts where score drift is largest.

**vs. SFPC with pooled (non-stratified) quantile** — On a case where one generator is over-scored by the judge, the pooled top-10% target is dominated by that generator, so Acc_id is computed against a target that already encodes the judge's bias. Our method stratifies the quantile per generator, holding target-set size fixed across generators. We expect the pooled variant to show lower Acc_id_ll and higher variance across generators, with the gap concentrated on prompts where generator score distributions are shifted.

**vs. resolution/watermark control** — On a case where generators differ in resolution or watermark, a judge can achieve high Acc_id by detecting the cue rather than by calibration. The control normalizes rollouts to a common resolution and removes watermarks. We expect Acc_id to survive normalization if it reflects calibration, and to drop toward chance if it reflects cue detection.

**vs. adversarial judge with target-independent motion bias** — On a case where the judge has a fixed preference for high-motion rollouts independent of the target set, the judge converges to a stable fixed point with high Delta_rank but chance Acc_id. This is the direct demonstration of the load-bearing assumption's failure mode. We expect the adversarial judge to pass the Delta_rank and |T*| criteria but fail the Acc_id criteria, showing that the certificate's two criteria are not independent evidence.

### What would falsify this idea

If the fixed-point judge's Acc_id gain over the single-pass ablation is uniform across all prompt subsets rather than concentrated on the subsets where score drift is predicted to occur, the fixed-point mechanism is not doing the causal work claimed. Equivalently, if Delta_rank is high but Acc_id remains at chance, the certificate is measuring self-consistency rather than identifiability, and the central claim that self-re-conditioning certifies calibration is wrong.

## References

1. StreamEval: Evaluating Interactive Streaming Video Generation Models
2. TC-Bench: Benchmarking Temporal Compositionality in Text-to-Video and Image-to-Video Generation
3. WorldScore: A Unified Evaluation Benchmark for World Generation
4. Davidsonian Scene Graph: Improving Reliability in Fine-grained Evaluation for Text-to-Image Generation
5. FETV: A Benchmark for Fine-Grained Evaluation of Open-Domain Text-to-Video Generation
6. Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation
7. VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models
8. All You May Need for VQA are Image Captions
