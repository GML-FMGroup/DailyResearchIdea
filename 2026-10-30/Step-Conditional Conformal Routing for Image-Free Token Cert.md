# Step-Conditional Conformal Routing for Image-Free Token Certification in Diffusion VLMs

## Motivation

Existing efficient diffusion-VLM decoders substitute cheap step-local proxies for validated correctness signals — LOWN treats image-free/image-conditioned prediction agreement as an uncalibrated certificate, KLASS uses KL stability as a convergence proxy, and Elastic-Cache uses KV drift as a refresh trigger — but none establishes a calibration relationship between the proxy and ground-truth token accuracy. Consequently these gates cannot be tuned to a user-specified error rate, and their fixed thresholds do not transfer across denoising steps or tasks. The structural gap is the absence of a distribution-free, finite-sample-valid mapping from the routing statistic to the probability that a reused token is actually correct.

## Key Insight

Step-conditional Mondrian conformal prediction converts the routing gate's agreement statistic into a per-step threshold with a distribution-free finite-sample guarantee on the image-invocation error rate, because exchangeability holds within each (denoising-step, task) stratum and the image-conditioned prediction provides near-oracle correctness labels for free at calibration time.

## Method

**What it is.** Step-Conditional Conformal Routing (SCCR) is a calibration layer that wraps LOWN's agreement gate. Its inputs are the per-step agreement statistic between the image-free and image-conditioned predictions, a held-out calibration set of (agreement, ground-truth-correctness) pairs stratified by denoising step and task, a small truly-labeled audit set used to estimate the image-conditioned prediction's per-stratum error rate, and a user-specified error level α. Its output is a per-step, per-task threshold τ_{t,k} such that reusing the image-free prediction when agreement ≥ τ_{t,k} provably limits the image-invocation error rate to α in finite samples, after a conservative correction for label noise in the image-conditioned prediction.

**Load-bearing assumption (stated explicitly).** SCCR's finite-sample guarantee is stated for the observed label y_{t,i} = 1[image-conditioned prediction == ground-truth token]. It transfers to the latent true correctness of the image-free prediction only under the assumption that the image-conditioned prediction is a near-oracle correctness label — i.e., that its per-stratum error rate ε_{t,k} = P(y_{t,i}=0 | image-free prediction correct) is negligible. This assumption fails on dense grounding, hallucinated OCR, and small-object tasks where the image-conditioned prediction is itself wrong. SCCR therefore does not assume ε_{t,k}=0; it estimates ε_{t,k} on a small truly-labeled audit set and inflates the conformal quantile by ε_{t,k} (conservative noisy-label conformal), so the realized error rate on reused tokens is bounded by α + ε_{t,k} rather than α alone, and the bound reduces to α when ε_{t,k} is small.

**How it works.**

```pseudocode
# Phase 0: audit-set construction (small, truly labeled; used only to estimate label noise)
for each audit example (x, image, task k, ground-truth tokens):
    run LLaDA-V denoising with full image processing at every step
    at each step t and each masked position i:
        y_audit_{t,i} = 1[ image-conditioned prediction == ground-truth token ]
    store (t, k, y_audit_{t,i})
# per-stratum label-noise estimate (with Laplace smoothing to avoid zero counts)
for each stratum (t, k):
    ε_{t,k} = ( # { y_audit_{t,i} = 0 } + 1 ) / ( n_audit_{t,k} + 2 )

# Phase 1: free calibration-set construction (training-time, no extra labels)
for each calibration example (x, image, task k):
    run LLaDA-V denoising with full image processing at every step
    at each step t and each masked position i:
        a_{t,i}  = agreement( image-free prediction, image-conditioned prediction )   # LOWN statistic
        y_{t,i}  = 1[ image-conditioned prediction == ground-truth token ]            # observed (noisy) label
    store (t, k, a_{t,i}, y_{t,i})

# Phase 2: step-conditional Mondrian conformal threshold fitting with label-noise correction
for each stratum (t, k) with calibration pairs {(a_j, y_j)}_{j=1..n_{t,k}}:
    # nonconformity score: higher agreement => more confident reuse is safe
    s_j = 1 - a_j
    # conformal quantile at miscoverage α, finite-sample corrected
    q_{t,k} = the ⌈(n_{t,k}+1)(1-α)⌉-th smallest value of {s_j}
    # conservative label-noise inflation: shift quantile up by estimated oracle error rate
    q_corrected_{t,k} = min(1, q_{t,k} + ε_{t,k})
    τ_{t,k} = 1 - q_corrected_{t,k}

# Phase 3: inference-time routing with calibrated gate
for each denoising step t of a test example with task k:
    compute image-free prediction p_free and last image-conditioned prediction p_img
    a_t = agreement(p_free, p_img)
    if a_t ≥ τ_{t,k}:
        certify p_free for reuse; unmask |certified tokens| positions this step
    else:
        invoke image processing; unmask one token (standard LLaDA-V step)
```

Hyperparameters: α (user-specified, default 0.05), calibration-set size per stratum n_{t,k} (must exceed 1/α − 1), audit-set size per stratum n_audit_{t,k} (default 64, used only for ε_{t,k} estimation), and the exchangeability stratum definition (step index t binned into deciles when per-step calibration sets are too small).

**Why this design.** We chose a *step-conditional Mondrian* partition over a single global conformal threshold because agreement statistics are strongly non-stationary across denoising steps — early steps have noisier image-free predictions — so pooling steps would violate exchangeability and inflate the error rate at early steps; the cost is that each stratum needs enough calibration points, which we address by binning steps into deciles when n_{t,k} < 1/α − 1. We chose the *nonconformity score s = 1 − a* (monotone in agreement) over a learned recalibration head because conformal validity requires only exchangeability, not a well-specified model, and a learned head would reintroduce the calibration problem it is meant to solve; the cost is that the threshold is conservative when agreement is a weak predictor, yielding less speedup but never exceeding α. We chose to *reuse LOWN's agreement statistic unchanged* rather than design a new routing proxy because the contribution is the calibration wrapper, not the proxy; the cost is that SCCR inherits LOWN's proxy quality — if agreement is uninformative, the calibrated threshold will certify few tokens. We chose *finite-sample conformal quantiles* over asymptotic or temperature-scaling calibration because the latter require distributional assumptions that fail under task shift; the cost is that calibration sets must be refreshed when the task distribution changes. We chose to *estimate and inflate by the image-conditioned prediction's per-stratum error rate ε_{t,k}* rather than assume it is zero, because the observed label y_{t,i} is a noisy proxy for true correctness and the conformal guarantee is stated for the observed label; the cost is a small truly-labeled audit set and a slightly more conservative threshold (τ_{t,k} decreases as ε_{t,k} increases), which reduces speedup but keeps the realized error rate on reused tokens bounded by α + ε_{t,k}. A domain expert would not describe SCCR as a variant of LOWN, KLASS, or Elastic-Cache: LOWN uses a fixed agreement threshold with no error control, KLASS thresholds KL divergence without labels, and Elastic-Cache thresholds KV drift heuristically — none constructs a held-out (proxy, correctness) calibration set, estimates label noise, nor applies Mondrian conformal prediction, so the structural property SCCR establishes (finite-sample error control per step and task, robust to image-conditioned label noise) is absent from all three.

**Why this measures what we claim.** The agreement statistic a_{t,i} measures the probability that the image-free prediction is correct conditional on step and task because the calibration set pairs it with the image-conditioned prediction's equality to the ground-truth token, which is a near-oracle correctness label under the assumption that the image-conditioned prediction is correct whenever the model has converged on that token; this assumption fails when the image-conditioned prediction is itself wrong (e.g., dense visual grounding tasks where the model misreads the image), in which case a_{t,i} reflects agreement with a noisy oracle rather than true correctness — which is why SCCR estimates the per-stratum oracle error rate ε_{t,k} on a small truly-labeled audit set and inflates the conformal quantile by ε_{t,k}, so the realized error rate on reused tokens is bounded by α + ε_{t,k} rather than α alone. The conformal threshold τ_{t,k} measures the image-invocation error rate because Mondrian conformal prediction guarantees that, within each (step, task) stratum, the probability that a reused token is incorrect is at most α under the assumption that calibration and test examples are exchangeable within the stratum; this assumption fails when the test task distribution drifts from calibration (e.g., a new benchmark with different image-reliance), in which case the realized error rate can exceed α and the threshold must be recalibrated. The step-conditional partition measures transferability across steps because exchangeability is enforced separately per step, under the assumption that the agreement-correctness relationship is stationary within a step across tasks; this assumption fails when tasks differ in how informative agreement is at the same step, which is why we additionally stratify by task k.

## Contribution

(1) A step-conditional Mondrian conformal calibration wrapper for diffusion-VLM routing gates that converts any agreement-style statistic into a per-step, per-task threshold with a finite-sample guarantee on the image-invocation error rate, requiring no new model training and no human labels beyond the ground-truth tokens already available at calibration time. (2) A free calibration-set construction procedure that exploits the image-conditioned prediction as a near-oracle correctness label at every denoising step, making the calibration data essentially costless. (3) A design principle — stratum by (denoising step, task) to preserve exchangeability — that explains why fixed agreement thresholds fail to transfer and how to repair them without retraining.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Mixed multimodal benchmarks with per-task strata (grounding + VQA) | Tasks differ in image-reliance, testing per-task calibration |
| Primary metric | Realized image-invocation error rate at target α | Directly measures the certified error-control claim |
| Secondary metric | Per-stratum label-noise estimate ε_{t,k} and its correlation with realized error-rate inflation | Tests the load-bearing assumption that the image-conditioned prediction is near-oracle |
| Baseline | LLaDA-V full image processing every step | Upper-bound accuracy, no speedup, no error control |
| Baseline | LOWN fixed agreement threshold | Same proxy, but uncalibrated error rate |
| Baseline | KLASS KL-divergence threshold | Label-free proxy with no finite-sample guarantee |
| Ablation-of-ours | SCCR with single global threshold (no Mondrian steps) | Isolates value of step-conditional partitioning |
| Ablation-of-ours | SCCR without label-noise correction (ε_{t,k}=0) | Isolates value of the conservative noisy-label inflation |

### Why this setup validates the claim

The central claim is that SCCR provably limits the image-invocation error rate to α per step and task in finite samples, while retaining most of LOWN's speedup, and that this bound holds up to the estimated label noise ε_{t,k} of the image-conditioned prediction. The primary metric — realized error rate among reused tokens at the target α — is the only metric that directly tests this, because accuracy alone can be high while the error-control property is violated, and speedup alone can be high while accuracy collapses. Full-image LLaDA-V establishes the accuracy ceiling against which any routing loss is judged, so we can see whether SCCR's certified reuse stays close to that ceiling. LOWN with a fixed threshold is the critical baseline: it uses the identical agreement statistic, so any difference in realized error rate isolates the contribution of conformal calibration rather than proxy quality. KLASS tests whether a label-free divergence proxy could achieve the same control without calibration; if SCCR matches its speedup but bounds error while KLASS does not, the calibration wrapper is doing the work. The global-threshold ablation is the decisive internal control: if step-conditional Mondrian partitioning is necessary, removing it should produce error-rate inflation concentrated at early steps, exactly where agreement is non-stationary. The no-label-noise-correction ablation is the decisive test of the load-bearing assumption: if the image-conditioned prediction is truly near-oracle, removing the ε_{t,k} inflation should leave the realized error rate unchanged; if it is not, the uncorrected SCCR should exceed α on strata with high ε_{t,k}, and the corrected SCCR should restore the bound to α + ε_{t,k}. The secondary metric — per-stratum ε_{t,k} and its correlation with realized error-rate inflation — directly quantifies the label noise and its effect, as required to validate the assumption rather than assert it.

### Expected outcome and causal chain

**vs. LLaDA-V full image processing** — On a dense grounding case where the image disambiguates a referring expression, full processing invokes the image at every step and stays accurate but pays maximal latency. Our method invokes the image only when agreement falls below the calibrated threshold, and on this case early steps will route to the image while later steps certify reuse, so we expect near-parity accuracy with a clear latency reduction concentrated in the late-step regime, not uniform across all steps.

**vs. LOWN fixed agreement threshold** — On a VQA case where the fixed threshold was tuned on an easier task, LOWN certifies reuse at early steps where image-free predictions are still noisy, producing a handful of confidently wrong reused tokens and a realized error rate above α. Our method instead uses a stricter early-step threshold τ_{t,k} derived from the calibration quantile (and further inflated by ε_{t,k}), so it refuses those early certifications, and we expect the observable signal to be LOWN's error rate exceeding α on the early-step subset while SCCR sits at or below α + ε_{t,k}, with SCCR's speedup slightly lower but accuracy higher.

**vs. KLASS KL-divergence threshold** — On a task-shifted benchmark with different image-reliance, KLASS's heuristic divergence threshold has no label-grounded calibration and drifts, either over-invoking the image (losing speedup) or under-invoking it (inflating errors) depending on the shift direction. Our method's per-task Mondrian strata are fit on held-out (agreement, correctness) pairs, so the threshold adapts to the task's actual agreement-correctness relationship; we expect SCCR to hold error near α + ε_{t,k} across both original and shifted benchmarks, whereas KLASS shows a task-dependent error-rate swing.

**vs. SCCR without label-noise correction (ε_{t,k}=0)** — On a dense grounding stratum where the image-conditioned prediction is itself wrong (high ε_{t,k}), the uncorrected SCCR treats y_{t,i}=0 as a true negative and certifies reuse too aggressively, so its realized error rate exceeds α. The corrected SCCR inflates q_{t,k} by ε_{t,k}, lowering τ_{t,k} and refusing those certifications; we expect the corrected SCCR to bring the realized error rate back to α + ε_{t,k} at the cost of a measurable speedup reduction on that stratum, with the gap between corrected and uncorrected SCCR scaling with the measured ε_{t,k}.

### What would falsify this idea

If the realized image-invocation error rate exceeds α + ε_{t,k} at early steps even with step-conditional Mondrian thresholds and label-noise correction, or if the global-threshold ablation matches SCCR's error control across all strata, then the step-conditional calibration is not doing the work claimed and the central structural guarantee is wrong. A uniform gain across subsets rather than a concentration of the improvement at early steps and shifted tasks, or a failure of the measured ε_{t,k} to predict the uncorrected SCCR's error-rate inflation, would similarly indicate the mechanism is not the cause.

## References

1. Look Only When Needed: Certificate-Guided Visual Routing for Efficient Diffusion VLMs
2. Attention Is All You Need for KV Cache in Diffusion LLMs
3. Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles
4. KLASS: KL-Guided Fast Inference in Masked Diffusion Models
5. Simplified and Generalized Masked Diffusion for Discrete Data
6. Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data
7. Simple and Effective Masked Diffusion Language Models
8. Simple Guidance Mechanisms for Discrete Diffusion Models
