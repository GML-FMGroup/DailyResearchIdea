# Round-Trip Executable Verification for Code-as-Video-Caption via Reconstruction-Certified Fidelity

## Motivation

Code-as-caption paradigms such as CodeCap-Bench and Code2Video treat code as a unidirectional generation target: a caption is produced and scored against annotations, but the pipeline never requires that the emitted program be re-executed to reconstruct the source video and compared back against it. This structural omission means a caption can score well on faithfulness rubrics while being non-executable or visually divergent from the original, and CodeCap-Bench itself concedes its benefits depend on MLLMs producing faithful code — an assumption the benchmark cannot certify because it has no closed loop. The field's trajectory toward executable captions therefore hits a new bottleneck: without a back-execution check, 'code as caption' is indistinguishable from 'code-like text as caption'.

## Key Insight

A caption is a faithful description of a video exactly when the program it induces, when executed, is a perceptual and structural fixed point of the original video — making round-trip reconstruction error a necessary condition that no unidirectional caption score can substitute for.

## Method

**What it is.** We introduce **RT-Cert**, a round-trip verification module that wraps the Code2Video generation pipeline. Its input is a source video $V$ and a caption-to-code generator $G$; its output is a certified caption $c$ together with a reconstruction-fidelity vector $r(V, \hat{V})$ and a pass/fail certificate, where $\hat{V}$ is the video obtained by executing the code induced by $c$.

**Load-bearing assumption (stated explicitly).** RT-Cert certifies a caption only relative to the renderer's expressivity ceiling: the certificate is meaningful exactly when the source video $V$ is itself reproducible by the primitive-based renderer, so that reconstruction error is dominated by caption error rather than by renderer expressivity. We therefore do not apply absolute pixel thresholds to raw $V$; instead we calibrate every fidelity axis against an oracle reconstruction $\hat{V}^{\text{oracle}}$ obtained by executing the annotated ground-truth primitives for that video (see step 0 below). The certificate then measures caption error *above* the renderer's achievable floor, not absolute pixel agreement.

**How it works.**

```pseudocode
INPUT: source video V (frames F_1..F_T), generator G, margin delta, budget K=3
OUTPUT: certified caption c*, fidelity vector r*, certificate in {PASS, FAIL}

0. # Renderer-relative calibration (per video, once)
   c_oracle <- ANNOTATED_PRIMITIVES(V)          # ground-truth code caption
   \hat{V}^oracle <- EXECUTE(c_oracle)           # sandboxed render, deterministic seed
   r_oracle <- FIDELITY_RAW(V, \hat{V}^oracle)   # raw per-axis errors (see below)
   # r_oracle is the renderer's expressivity floor for this video

1. c0 <- G(V, prompt="emit executable code caption")      # initial code caption
2. for k in 1..K:
3.     \hat{V}_k <- EXECUTE(c_{k-1})                       # sandboxed render, deterministic seed
4.     if EXECUTE_FAILS(\hat{V}_k):
5.         c_k <- G(V, feedback=STDERR(\hat{V}_k))        # repair on runtime error
6.         continue
7.     r_k <- FIDELITY(V, \hat{V}_k, r_oracle)            # see metric bundle below
8.     if r_k.pass(delta): return (c_{k-1}, r_k, PASS)
9.     c_k <- G(V, feedback=DIFF_REPORT(V, \hat{V}_k))   # repair on perceptual/structural gap
10. return (c_{K-1}, r_{K-1}, FAIL)

FIDELITY(V, \hat{V}, r_oracle) returns normalized per-axis errors and a pass flag:
  - perceptual:  e_lpips = LPIPS(F_t, \hat{F}_t) averaged over T frames
  - structural:  e_ssim  = 1 - SSIM(F_t, \hat{F}_t) averaged over T frames
  - layout:      e_iou   = 1 - IoU of connected-component bounding boxes per frame
  - temporal:    e_dtw   = DTW distance between per-frame centroid trajectories
  - normalized:  \tilde{e}_a = max(0, e_a - e_a^oracle) / (1 - e_a^oracle)  for a in {lpips, ssim, iou, dtw}
  - pass:        \tilde{e}_lpips < delta AND \tilde{e}_ssim < delta AND \tilde{e}_iou < delta AND \tilde{e}_dtw < delta
                 with delta = 0.10 (fixed on calibration split; see below)
```

The repair loop is the load-bearing component: the DIFF_REPORT localizes *which* frame, *which* primitive, and *which* trajectory segment diverged, and this localization is fed back as structured text to $G$. The margin $\delta$ is fixed per category on a held-out calibration split of CodeCap-Bench, not tuned per video; concretely, $\delta$ is chosen as the smallest value in $\{0.05, 0.10, 0.15, 0.20\}$ that yields $\ge 0.90$ precision of the PASS decision against human faithfulness labels on the calibration split (see calibration study in experiments).

**Why this design.** We chose an *execution-grounded repair loop* over a single-shot generate-and-score because the structural reason captions fail is that the generator has no access to the consequence of its own output — a one-shot scorer can only reject, never correct, accepting the cost that RT-Cert requires a sandboxed renderer and up to $K=3$ generation calls per video. We chose *localized DIFF_REPORT feedback* over scalar reward feedback because a scalar LPIPS value tells the generator nothing about which primitive's position or which keyframe's timing is wrong, accepting the cost of implementing per-primitive connected-component matching and trajectory DTW. We chose *fixed calibration-split thresholds* over per-video adaptive thresholds because adaptive thresholds let a bad caption pass by moving the goalposts, accepting the cost that genuinely ambiguous videos may be conservatively marked FAIL. We chose to certify rather than to train a reward model because a learned verifier would reintroduce the calibration problem that CodeCap-Bench's rubric scoring already suffers from; a domain expert would not describe RT-Cert as a variant of a learned verifier or of retrieval-based caption selection, because its decision surface is the *executed* reconstruction, a quantity no retrieval or scoring-only method ever computes. Finally, we chose *renderer-relative normalization against an oracle reconstruction* over absolute pixel thresholds because absolute thresholds conflate caption error with renderer expressivity (the analysis-by-synthesis gap); the cost is one extra oracle render per video, which is negligible relative to the $K=3$ generation calls.

**Why it measures what we claim.** Normalized LPIPS $\tilde{e}_\text{lpips}$ on reconstructed frames measures *perceptual fidelity of the caption above the renderer's floor* because the named assumption is that a faithful code caption, when executed, renders the same pixels as the source up to the oracle's own reconstruction error under the benchmark's deterministic rendering seed; this assumption fails when the source video contains non-procedural texture or compression artifacts that no primitive-based program can reproduce, in which case even the oracle has high raw LPIPS and the normalization absorbs that floor — but if the oracle itself cannot execute, the certificate is undefined and the video is excluded from certification. Normalized layout-IoU $\tilde{e}_\text{iou}$ over connected-component boxes measures *structural faithfulness of primitive placement* because the assumption is that each annotated primitive maps to one connected component under the benchmark's rendering; this assumption fails under heavy occlusion or overlapping primitives, in which case IoU reflects component-merging artifacts rather than misplacement. Normalized temporal DTW $\tilde{e}_\text{dtw}$ over centroid trajectories measures *temporal faithfulness of the caption's time-dependent functions* because the assumption is that primitive centroids are the sufficient statistic of motion in structured 2D content; this assumption fails for deformable or rotating primitives, in which case DTW reflects centroid drift rather than timing error. The repair loop's DIFF_REPORT measures *actionable divergence* because the assumption is that localized per-primitive feedback is causally responsible for the generator's correction; this fails when the generator ignores feedback, in which case the loop reflects generator compliance rather than verification quality — we therefore measure compliance rate (fraction of iterations where the targeted axis error decreases after feedback) and report it alongside PASS rate.

## Contribution

(1) RT-Cert, a round-trip executable verification module that converts code captions from unidirectional generation targets into reconstruction-certified artifacts, with a localized perceptual/structural/temporal fidelity bundle and a feedback-driven repair loop. (2) The design principle that caption faithfulness is a fixed-point property of executed reconstruction rather than a rubric score, operationalized through per-primitive DIFF_REPORT feedback that lets the generator correct rather than merely be rejected. (3) A reconstruction-fidelity certification layer for CodeCap-Bench that yields PASS/FAIL certificates and per-category calibration thresholds, enabling the benchmark to report executable-caption rates alongside its existing multi-level faithfulness scores.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CodeCap-Bench (full split; pilot on 100-video subset) | Videos with annotated primitives and deterministic rendering seeds |
| Primary metric | Certified PASS rate at fixed δ=0.10 (normalized) | Directly measures whether executed reconstruction matches source above renderer floor |
| Secondary metric | Compliance rate (fraction of feedback iterations reducing targeted axis error) | Tests whether localized feedback causally drives repair |
| Calibration metric | Precision/recall of PASS vs. human faithfulness labels on held-out split | Grounds δ rather than stipulating it |
| Baseline | Single-shot code caption (no repair) | Tests whether the repair loop, not generation, drives gains |
| Baseline | Natural-language caption + reconstruction | Tests whether code grounding beats prose description |
| Baseline | Scalar-reward rejection sampling (best-of-K by LPIPS) | Tests whether localization, not extra samples, is load-bearing |
| Baseline | Learned-verifier selection (binary classifier on LPIPS/SSIM/IoU/DTW features) | Tests whether certification beats a learned scorer |
| Baseline | Retrieval-based caption selection (nearest training caption by feature distance) | Tests whether execution beats retrieval |
| Ablation-of-ours | RT-Cert with scalar feedback (no DIFF_REPORT) | Isolates the localized-feedback claim from the loop itself |
| Ablation-of-ours | RT-Cert with absolute thresholds (no oracle normalization) | Isolates the renderer-relative calibration claim |
| Ablation-of-ours | RT-Cert with feedback-format variants (scalar, frame-index-only, full DIFF_REPORT) | Tests feedback-format sensitivity of compliance |

**Calibration study.** On a held-out calibration split of 200 videos (disjoint from test), two annotators label each caption as faithful/unfaithful to the source video (Cohen's κ reported). We sweep δ ∈ {0.05, 0.10, 0.15, 0.20} and select the smallest δ achieving ≥0.90 precision of the PASS decision against human labels; we report precision, recall, and F1 at the selected δ on the test split. This grounds the threshold rather than stipulating it.

**Compute budget.** Pilot: 100 videos × (1 oracle render + up to 3 generation calls + 3 renders) ≈ 100 × 7 ≈ 700 render/generation units; on a single A100 this is ≈ 2 GPU-hours for generation (assuming ~10 s per generation call) plus ≈ 0.5 GPU-hour for rendering. Full CodeCap-Bench (assume ~2,000 videos): ≈ 40 GPU-hours generation + ≈ 10 GPU-hours rendering, plus baseline runs (single-shot: 1 generation/video ≈ 6 GPU-hours; best-of-K: 3 generations/video ≈ 18 GPU-hours; learned verifier training: ≈ 2 GPU-hours on features). Total ≈ 80 GPU-hours.

**Non-deterministic renderer fallback.** If the renderer is non-deterministic (e.g., stochastic anti-aliasing), we render each execution with 3 fixed seeds and report mean ± std of each fidelity axis; a caption passes only if the pass condition holds for all 3 seeds. We report seed variance per axis on the pilot subset.

### Why this setup validates the claim

The central claim is that execution-grounded, localized repair converts captions that a one-shot generator cannot fix into certified reconstructions, and that this gain comes specifically from *localized* feedback rather than from extra compute or rejection. The dataset supplies the deterministic renderer and per-primitive annotations that make both reconstruction and localization well-defined, and the oracle-normalized fidelity vector is a fair arbiter because it subtracts the renderer's expressivity floor per video. The single-shot baseline tests the sub-claim that correction, not generation quality, is the bottleneck; the natural-language baseline tests the sub-claim that executable structure is what makes reconstruction faithful; the best-of-K scalar-reward baseline tests the sub-claim that localization beats more samples with a scalar signal; the learned-verifier and retrieval baselines test the sub-claim that certification is not reducible to a learned scorer or retrieval. The ablation with scalar feedback is the sharpest test: it holds the loop, budget, and thresholds fixed while removing only DIFF_REPORT, so any gap between it and full RT-Cert is attributable to localization; the absolute-threshold ablation isolates the contribution of oracle normalization. The compliance-rate measurement directly tests the assumption that localized feedback causally drives repair, and the feedback-format ablation tests sensitivity to feedback format. Certified PASS rate is the right primary metric because it is exactly the decision the method makes, and it can only rise if the executed reconstruction genuinely satisfies the normalized fidelity bundle — a scorer that rejects cannot inflate it, and adaptive thresholds are excluded by construction. The calibration study grounds δ against human labels, so the PASS decision is validated rather than stipulated.

### Expected outcome and causal chain

**vs. Single-shot code caption (no repair)** — On a case where the generator misplaces one primitive by a small offset and emits a slightly wrong timing function, the single-shot baseline produces a caption that fails normalized layout-IoU and temporal-DTW with no recourse, because its mechanism has no access to the consequence of its own output. Our method instead renders the caption, localizes the offending primitive and trajectory segment, and feeds that back, so we expect a large PASS-rate gap concentrated on instances whose initial caption fails exactly one fidelity axis, with near-parity on instances the one-shot generator already passes.

**vs. Natural-language caption + reconstruction** — On a case with several interacting primitives and explicit motion, the natural-language baseline produces a fluent description that a renderer cannot execute into matching pixels, because prose lacks the primitive identity and time-indexed parameters the renderer needs. Our method instead emits executable code and verifies it against the source frames, so we expect a pronounced gap on multi-primitive and motion-heavy subsets and only a modest gap on single-static-object videos, where prose suffices.

**vs. Scalar-reward rejection sampling (best-of-K by LPIPS)** — On a case where a caption is globally close but one primitive drifts over the last third of the video, the scalar baseline produces a rejected or accepted caption with no information about *where* the error is, because a scalar LPIPS value cannot localize frame, primitive, or trajectory segment. Our method instead converts the same signal into a structured DIFF_REPORT and repairs the specific segment, so we expect RT-Cert to pass a visibly larger fraction of near-miss instances at equal generation budget, while both methods are comparable on instances far from the threshold.

**vs. Learned-verifier selection** — On a case where the learned verifier has not seen the failure mode (e.g., a rare primitive type), the verifier's decision boundary is miscalibrated because it was trained on features from a different distribution. Our method instead executes the caption and compares to the oracle-normalized reconstruction, so we expect RT-Cert to maintain PASS-rate precision where the learned verifier degrades, with the gap concentrated on out-of-distribution primitive types.

**vs. Retrieval-based caption selection** — On a case where the nearest training caption is structurally similar but temporally misaligned, retrieval returns a caption that fails temporal-DTW because it cannot adapt to the specific video's timing. Our method instead repairs the caption against the executed reconstruction, so we expect a gap on temporally distinctive videos and near-parity on videos whose nearest neighbor is already faithful.

**vs. RT-Cert with scalar feedback (no DIFF_REPORT)** — On a case where the caption fails one axis by a small margin, the scalar-feedback ablation receives only a scalar LPIPS value and cannot target the specific primitive or segment, so its repair iterations are undirected. Our method instead provides localized DIFF_REPORT, so we expect the full RT-Cert to achieve higher PASS rate at equal K, with the gap concentrated on near-miss instances and on instances where the compliance rate is high (i.e., the generator can act on localized feedback).

**vs. RT-Cert with absolute thresholds (no oracle normalization)** — On a video whose oracle reconstruction already has high raw LPIPS due to non-procedural texture, the absolute-threshold variant marks even a faithful caption as FAIL because the renderer cannot reproduce the texture. Our method instead normalizes against the oracle floor, so we expect the absolute-threshold variant to have low PASS rate on high-expressivity-gap videos while full RT-Cert maintains high PASS rate, with the gap concentrated on videos where the oracle's raw LPIPS exceeds 0.15.

### What would falsify this idea

If RT-Cert's gain over the scalar-feedback ablation and over best-of-K were uniform across all subsets rather than concentrated on near-miss instances with a single localized failure, the claim that localized repair is the causal driver would be wrong. Likewise, if the natural-language baseline matched certified PASS rates on multi-primitive motion-heavy videos, the claim that executable structure is what grounds reconstruction would collapse; and if the compliance rate were near zero, the repair loop would reflect generator non-compliance rather than verification quality.

## References

1. CodeCap-Bench: Can Code Serve as a Good Video Caption?
