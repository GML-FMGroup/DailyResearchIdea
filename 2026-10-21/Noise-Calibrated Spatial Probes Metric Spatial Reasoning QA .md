# Noise-Calibrated Spatial Probes: Metric Spatial Reasoning QA with 3D-Grounded Answer Validation under Low-Light Sensor Noise

## Motivation

DarkQA (Idea Card 1) deliberately restricts its QA pairs to visual primitives and does not require metric distance estimation, route planning, or cognitive-map construction; VSI-Bench (Thinking in Space, Idea Card 3) targets exactly these spatial-reasoning tasks but only from clean video and does not vary sensor noise. As a result, the field cannot tell whether MLLM spatial-reasoning failures are perceptual (low-light noise corrupting depth cues) or cognitive (inability to build a metric cognitive map), because no existing benchmark crosses spatial-reasoning task type with controlled, physically-faithful sensor noise and 3D-ground-truth validation. The structural cause is that low-light benchmarks and spatial-reasoning benchmarks were built on disjoint image-formation assumptions—DarkQA's RAW-domain noise model never meets VSI-Bench's metric scene annotations—so the noise-vs-reasoning interaction is unobservable.

## Key Insight

Because the metric answer of a spatial-reasoning probe is a deterministic function of the 3D scene graph, the same scene rendered at multiple RAW-noise levels yields a fixed ground-truth answer; therefore the slope of accuracy versus injected noise isolates perceptual degradation from cognitive spatial-reasoning ability, a separation that neither DarkQA nor VSI-Bench can produce.

## Method

**What it is.** We introduce NCS-Space (Noise-Calibrated Spatial Probes), a benchmark generator and evaluation protocol that takes a 3D indoor scene (mesh + semantic instances + camera trajectory) and produces a paired set of spatial-reasoning QA items—metric distance estimation (MDE), route planning (RP), and cognitive-map construction (CMC)—each rendered at K=5 inverse-ISP/forward-ISP RAW-noise levels calibrated to DarkQA's physical noise model, with answers computed deterministically from the 3D scene graph and validated by a geometric solver. **Load-bearing assumption (stated explicitly):** the slope of accuracy versus injected RAW sensor noise (acc_1 - acc_5) isolates perceptual degradation from cognitive spatial-reasoning failure, i.e., noise level is a clean, monotone, single-factor manipulation whose effect on VLM accuracy is separable from the cognitive difficulty of the spatial task itself. This assumption is not assumed away: it is operationalized and tested via a perceptual-only control task and a contrast-only ablation (Phase 6), and the slope is reported as a residual above the perceptual-degradation baseline.

**How it works.**
```pseudocode
Input: 3D scene S = (mesh, instance seg, camera poses P), noise levels sigma_1..sigma_5 (RAW-domain, DarkQA-calibrated)
Output: QA set Q = {(q, a, sigma_k, task_type, scene_id)}, plus per-item noise-sensitivity label and perceptual budget

Phase 1 - Scene graph & answer oracle (task-agnostic):
  G <- build_scene_graph(S)            # nodes: object instances; edges: metric distances, traversability
  for each camera pose p in P:
    V_p <- visible_instances(G, p, fov=90deg)

Phase 2 - Probe generation (deterministic, scene-graph-grounded):
  MDE: sample ordered pair (o_i, o_j) from V_p with graph distance d_ij in [0.5m, 8m]
       answer a = round(d_ij, 0.1m); distractor set = {d_ij * r : r in {0.7,0.8,1.25,1.4}} shuffled
  RP:  sample start s, goal g from V_p; shortest path length L computed on traversability graph
       answer a = L (meters); distractors = {L * r : r in {0.8,1.2,1.5}} plus one path that violates a wall
  CMC: sample 5 landmarks from V_p; answer a = 5x5 pairwise distance matrix (quantized to 0.5m bins)
       distractors = matrices from 3 other scenes with matched room area
  Perceptual-only control (new, same scenes/poses):
       PRES: object-presence probe (is o_i visible? yes/no) and depth-ordering probe (is o_i closer than o_j? yes/no)
       answer a = ground-truth from G; no metric estimation required

Phase 3 - Noise rendering (DarkQA ISP pipeline):
  for sigma_k in sigma_1..sigma_5:
    I_k <- forward_ISP(inverse_ISP(render(S, p)) + RAW_noise(sigma_k))
    # sigma_1 = clean; sigma_5 = DarkQA's max level
  Contrast-only control (new):
    for sigma_k in sigma_1..sigma_5:
      I_k^contrast <- forward_ISP(inverse_ISP(render(S, p)) * alpha_k)  # alpha_k matched to mean luminance of I_k, zero sensor noise

Phase 4 - Answer validation (3D-ground-truth):
  for each (q, a) and rendered I_k:
    a_hat <- VLM(I_k, q)   # greedy decoding, temperature=0; 3 seeds reported for variance
    valid <- geometric_solver_check(a_hat, G, task_type)
    # MDE: |a_hat - d_ij| <= 0.3m; RP: path exists with length |a_hat - L|/L <= 0.15;
    # CMC: Frobenius norm of (a_hat - A) / ||A||_F <= 0.2

Phase 5 - Noise-sensitivity scoring:
  acc_k(task) <- mean(valid) at sigma_k
  slope(task) <- OLS slope of acc_k over sigma_k
  noise_robustness(task) <- acc_1 - acc_5   # drop attributable to perception
  perceptual_budget(task, sigma_k) <- fraction of items whose answer is recoverable from I_k by the geometric solver alone (oracle perception)
  residual_slope(task) <- slope(task) - slope(PRES)   # task slope above perceptual-degradation baseline
  contrast_slope(task) <- OLS slope of acc_k^contrast over sigma_k
  sensor_noise_share(task) <- (slope(task) - contrast_slope(task)) / slope(task)   # fraction attributable to sensor noise vs dynamic-range loss

Phase 6 - Calibration / verification (new, tests load-bearing assumption):
  monotonicity_check: Spearman rho between sigma_k and (1 - acc_k(PRES)); require rho > 0.8 for the noise manipulation to be considered monotone
  dissociation_test: compare residual_slope(MDE), residual_slope(RP), residual_slope(CMC); report pairwise bootstrap 95% CIs (1000 resamples)
  determinism_check: report std of acc_k across 3 greedy seeds; require std < 0.01
```

**Why this design.** We chose a deterministic scene-graph oracle over human annotation because metric answers (distances, path lengths, distance matrices) must be exactly reproducible across noise levels—human labels would inject variance that confounds the noise slope—accepting the cost that scenes must be synthetic or densely 3D-scanned. We chose DarkQA's inverse/forward ISP + RAW-noise pipeline over simple gamma/brightness augmentation because DarkQA already validated this synthesis against real paired low-light camera data; adopting it means our noise levels are physically interpretable and cross-comparable to DarkQA's primitive results, at the cost of restricting us to scenes where we can re-render RAW. We chose the multiple-noise-level slope (noise_robustness) over a single low-light condition because a single condition cannot separate perceptual degradation from cognitive spatial-reasoning failure—the slope is the quantity that isolates perception—accepting the cost of 5x rendering and inference. We chose three task types (MDE, RP, CMC) rather than one because VSI-Bench shows cognitive maps help distance but not all spatial tasks; including route planning and full cognitive-map construction tests whether that dissociation holds under noise, at the cost of a larger benchmark. We added a perceptual-only control task (PRES) and a contrast-only rendering condition because the load-bearing assumption—that the slope isolates perception—is otherwise untested; PRES provides a perceptual-degradation baseline so spatial-task slopes can be expressed as residuals, and the contrast-only condition quantifies how much of the slope is sensor noise versus dynamic-range loss. A domain expert would not describe NCS-Space as 'DarkQA + VSI-Bench' because the mechanism is a noise-calibrated paired-rendering protocol whose output is a per-task noise slope validated against a 3D geometric solver and normalized by a perceptual budget—neither DarkQA (no spatial tasks, no 3D oracle) nor VSI-Bench (no noise control, no paired rendering) produces this quantity.

**Why it measures what we claim.** The quantity `acc_k(task)` measures spatial-reasoning accuracy under noise level sigma_k because the answer oracle is computed from the 3D scene graph and the rendered image is the only source of information available to the VLM—the assumption is that the scene graph is the complete information needed to answer, which fails when an object is occluded from the camera pose but present in the graph, in which case acc_k reflects the VLM's inability to hallucinate invisible geometry rather than a spatial-reasoning failure (we flag and exclude such items). The quantity `slope(task)` measures perceptual noise sensitivity because noise level is the only variable changed across paired renders—the assumption is that the ISP pipeline's noise is the dominant corruption, which fails when low light also reduces contrast below the VLM's tokenizer threshold, in which case slope conflates sensor noise with dynamic-range loss (we report both RAW-noise and ISP-only ablations, and the contrast-only condition in Phase 3/5 quantifies the split via `sensor_noise_share`). The quantity `noise_robustness(task) = acc_1 - acc_5` measures the perception-attributable drop because clean and noisy renders share identical scene content and camera pose—the assumption is that the VLM's answer is deterministic given the image, which fails under sampling temperature > 0, in which case we use greedy decoding and report variance across 3 seeds (Phase 6 determinism_check). The quantity `residual_slope(task) = slope(task) - slope(PRES)` measures spatial-task-specific noise sensitivity above what perception alone permits because PRES uses the same scenes, poses, and noise levels but requires only object presence or depth ordering—the assumption is that PRES captures the perceptual-degradation baseline, which fails if PRES is trivially easy at all sigma (ceiling effect), in which case we report `perceptual_budget` as an alternative normalizer. The `geometric_solver_check` measures answer validity because it recomputes the metric from the scene graph independently of the VLM—the assumption is that the solver's tolerance bounds (0.3m, 15%, 0.2 Frobenius) match human spatial-reasoning tolerance, which fails for CMC where coarse binning may accept wrong topologies, in which case we additionally report exact-match on the binarized adjacency matrix.

## Contribution

["NCS-Space, a noise-calibrated spatial-probe generator that pairs 3D-scene-graph answer oracles with DarkQA's RAW-domain noise pipeline to produce MDE, RP, and CMC QA items at five controlled noise levels with deterministic ground-truth validation.", 'A noise-slope evaluation protocol that isolates perception-attributable degradation from cognitive spatial-reasoning failure by regressing per-task accuracy on injected noise level across paired renders.', 'An empirical dissociation finding: cognitive-map construction accuracy degrades more steeply with noise than metric distance estimation, indicating that the VSI-Bench cognitive-map benefit is itself noise-fragile.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | NCS-Space generated from 3D indoor scans; fixed subset of 200 scenes × 5 noise levels, precomputed and released as frozen renders | Paired renders isolate noise as sole variable; frozen set bounds cost |
| Primary metric | Per-task residual noise slope: slope(task) - slope(PRES), with bootstrap 95% CI (1000 resamples) | Residual above perceptual baseline isolates spatial-specific sensitivity |
| Secondary metric | sensor_noise_share(task) = (slope - contrast_slope)/slope | Quantifies sensor-noise vs dynamic-range attribution |
| Baseline | DarkQA-style VLMs without LLIE | Tests whether noise alone degrades spatial reasoning |
| Baseline | VSI-Bench-style MLLM QA (clean only) | Tests whether clean-scene spatial skill transfers under noise |
| Baseline | LLaVA-style instruction-tuned VLM | Tests general-capability ceiling on paired renders |
| Ablation-of-ours | Single-noise-level (sigma_5 only) scoring | Tests whether slope adds info beyond one degraded condition |
| Ablation-of-ours | Contrast-only rendering (zero sensor noise, matched mean luminance) | Tests how much slope is dynamic-range loss vs sensor noise |
| Control task | PRES (object-presence + depth-ordering) at each sigma | Provides perceptual-degradation baseline for residual slope |

### Why this setup validates the claim

The central claim is that per-task noise slopes, computed on paired renders where only RAW noise changes, isolate perception-attributable spatial-reasoning failure from cognitive failure. The design is falsifiable because each baseline and control probes a distinct sub-claim. DarkQA-style no-LLIE baselines test whether noise alone is sufficient to degrade spatial reasoning: if their slope is flat, our premise that noise corrupts spatial perception collapses. VSI-Bench-style clean-only QA tests whether clean-scene spatial competence predicts robustness: if clean accuracy and slope are uncorrelated, then our slope measures something orthogonal to spatial skill, which is exactly the dissociation we predict. The LLaVA-style baseline anchors the capability ceiling so we can attribute slope differences to noise handling rather than model scale. The single-noise-level ablation is the load-bearing control: if sigma_5-only accuracy ranks models identically to the slope, then the five-level paired rendering is unnecessary and the mechanism claim is redundant. The PRES control task and contrast-only ablation directly test the load-bearing assumption: PRES provides a perceptual-degradation baseline so spatial-task slopes are expressed as residuals, and the contrast-only condition quantifies how much of the slope is sensor noise versus dynamic-range loss. The monotonicity check (Spearman rho > 0.8 between sigma and 1 - acc(PRES)) verifies that the noise manipulation is monotone, and the determinism check (std < 0.01 across 3 greedy seeds) verifies that answers are deterministic given the image. Without these controls, the slope would confound noise severity with task-specific tokenizer/contrast thresholds; with them, the residual slope is the quantity that isolates perception-attributable failure.

### Expected outcome and causal chain

**vs. DarkQA-style VLMs without LLIE** — On a metric-distance item where two objects are 4.2m apart but the render at sigma_4 has RAW noise obscuring the floor-wall boundary, the baseline produces a distance drawn from the wrong depth prior because it has no noise-aware preprocessing and its depth tokens are corrupted by sensor noise. Our method instead renders the same scene at five calibrated noise levels, so the geometric solver flags the sigma_4 answer as invalid while the sigma_1 answer passes, letting us attribute the failure to perception rather than to a broken spatial prior. We expect a steep negative slope on MDE for the baseline but a markedly shallower slope for noise-robust VLMs, with the gap concentrated at sigma_3-sigma_5 and near-parity at sigma_1. Crucially, the PRES control task should show a shallower slope than MDE for the same baseline, so the residual slope (MDE minus PRES) remains negative, confirming that the MDE degradation exceeds what perception alone permits.

**vs. VSI-Bench-style MLLM QA (clean only)** — On a route-planning item where the shortest traversable path is 6.0m through a doorway, the clean-only baseline answers correctly at sigma_1 because it has strong scene-graph-like spatial priors from VSI-Bench training, but at sigma_4 the doorway edge is visually degraded and it proposes a path that violates the wall, because it never learned to re-anchor spatial edges under noise. Our method's paired rendering exposes exactly this: the same model shows high acc_1 but a steep RP slope, revealing that clean-scene spatial competence does not transfer to noisy perception. We expect a dissociation where clean accuracy is high but the RP slope is among the steepest, and where CMC slope is shallower than RP slope, mirroring VSI-Bench's cognitive-map advantage. The contrast-only ablation should show a shallower RP slope than the RAW-noise condition, with sensor_noise_share(RP) > 0.5, indicating that sensor noise rather than dynamic-range loss drives the RP degradation.

**vs. LLaVA-style instruction-tuned VLM** — On a cognitive-map item requiring a 5x5 landmark distance matrix, the baseline produces a matrix whose off-diagonal entries drift by more than one 0.5m bin at sigma_3 because its visual tokenizer loses fine metric cues under noise and it falls back to coarse room-layout heuristics. Our method's Frobenius-norm solver check catches this drift and the five-level slope quantifies it, whereas a single-condition evaluation would report only a binary pass/fail. We expect the LLaVA-style baseline to show a moderate MDE slope but a pronounced CMC slope, with the CMC gap widening at higher sigma, confirming that full cognitive-map construction is more noise-fragile than pairwise distance estimation. The perceptual_budget for CMC should remain high at sigma_3 (most items recoverable by the solver), so the CMC slope is not explained by perception alone.

### What would falsify this idea

If the per-task residual slopes (after subtracting the PRES baseline) are statistically indistinguishable across all baselines and our method, or if the single-noise-level ablation ranks models identically to the five-level slope, or if the monotonicity check fails (Spearman rho ≤ 0.8), then the paired-rendering protocol adds no diagnostic information and the central claim that slopes isolate perception-attributable failure is wrong. A uniform residual slope across MDE, RP, and CMC would also falsify the predicted task dissociation.

## References

1. DarkQA: Benchmarking Vision-Language Models on Visual-Primitive Question Answering in Low-Light Indoor Scenes
2. Improved Baselines with Visual Instruction Tuning
3. Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces
