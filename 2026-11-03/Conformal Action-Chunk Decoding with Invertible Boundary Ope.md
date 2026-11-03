# Conformal Action-Chunk Decoding with Invertible Boundary Operators for Vision-Language-Action Models

## Motivation

Deterministic single-pass parallel decoders with L1 regression, as in OpenVLA-OFT, achieve large speed-ups but provide no mechanism to represent or certify coverage of a multimodal expert action distribution, and fixed-length action chunks carry no invariant tying the last action of chunk k to the first action of chunk k+1. The structural root cause is that the decoder is trained only to regress a point estimate per chunk, so mode collapse and inter-chunk discontinuity are not merely unmeasured — they are unrepresentable in the current architecture, and no acceptance signal in the OFT recipe can detect them. This is the new bottleneck implied by the field's convergence on parallel continuous decoding: the speed win is purchased with an unquantified mode-collapse and boundary-discontinuity liability.

## Key Insight

Coupling a distribution-free split-conformal coverage certificate on each action chunk with an invertible, Lipschitz-bounded learned transition operator that algebraically ties adjacent chunk boundaries makes mode collapse and boundary discontinuity jointly detectable and penalizable at training time, because the coverage radius and the transition-operator residual are both exact finite-sample quantities rather than learned proxies.

## Method

**What it is.** CADIBO (Conformal Action-chunk Decoding with Invertible Boundary Operators) is a fine-tuning recipe for a vision-language-action model (OpenVLA backbone) whose parallel decoder emits, for each action chunk, a point chunk plus a scalar conformal radius, and whose chunk boundaries are tied by a learned invertible transition operator. Inputs: image observations and language instruction. Outputs: a sequence of action chunks, each with a calibrated coverage radius, and a boundary residual that is checked at inference.

**How it works.**

```pseudocode
# Phase 1: single-pass chunk decoder (OFT-style backbone, L1 loss)
for each training step:
    h = VLA_encoder(images, language)          # shared prefix
    for k in 1..K:                             # K chunks per episode
        a_hat_k, r_k = ParallelHead(h, k)      # point chunk + radius head, chunk length H=8
        L_reg += ||a_hat_k - a_expert_k||_1    # OFT L1 regression

# Phase 2: split-conformal calibration (post-hoc, no retraining)
cal_set = held-out expert episodes disjoint from train (N_cal = 512 chunks)
for each calibration episode:
    compute nonconformity score s_k = ||a_hat_k - a_expert_k||_inf  per chunk
r_k = Quantile_{1-alpha}({s_k})                # alpha = 0.1, finite-sample coverage

# Phase 3: invertible boundary transition operator T
# T: R^d -> R^d, d = action dim (7 for LIBERO); parameterized as a composition of
# 4 coupling layers, each with a 2-layer MLP (hidden=128, GeLU) for g, and
# spectral-norm clamp to enforce Lipschitz constant L_g = 0.9 < 1.
#   T(a) = a + g(phi(a))  with g Lipschitz-bounded by L_g < 1
# T is fit in a separate stage on expert boundary pairs only (never on decoder outputs):
#   L_T = mean_{(a_k[-1], a_{k+1}[0]) in expert} ||T(a_k[-1]) - a_{k+1}[0]||_2
# After fitting, T is frozen for all subsequent decoder training.
# boundary identity enforced at train time (decoder training, T frozen):
    L_bound = ||T(a_hat_k[-1]) - a_hat_{k+1}[0]||_2
    L_total = L_reg + lambda_cov * mean(r_k) + lambda_bnd * L_bound
# lambda_cov = 0.1, lambda_bnd = 1.0

# Phase 4: inference-time acceptance check
# tau_bnd is calibrated on expert boundary residuals: tau_bnd = Quantile_{0.9}({||T(a_k[-1]) - a_{k+1}[0]||_2 : expert boundary pairs})
for each chunk k:
    emit a_hat_k with certified radius r_k
    if ||T(a_hat_k[-1]) - a_hat_{k+1}[0]||_2 > tau_bnd:
        flag boundary discontinuity; re-query decoder for chunk k+1 (max 1 re-query; if still flagged, emit original chunk)
```

**Why this design.** We chose a post-hoc split-conformal calibration over an end-to-end differentiable coverage loss because conformal prediction gives a finite-sample coverage certificate with no distributional assumption, accepting the cost that the radius is only valid on the calibration distribution and must be recomputed per deployment. We chose an invertible coupling-layer transition operator over a free MLP because invertibility plus a Lipschitz bound makes the boundary identity exactly checkable at inference (a free MLP can only be checked empirically), accepting the cost that the operator is restricted to a diffeomorphism family and cannot represent arbitrary boundary maps. We chose to keep the OFT single-pass parallel decoder rather than add a separate multimodal head because a separate head reintroduces the two-module routing problem that OFT deliberately collapsed, accepting the cost that coverage is certified around the point estimate rather than over an explicit mixture. A domain expert would not describe this as 'conformal prediction plus OFT': the load-bearing object is the algebraic coupling between the conformal radius r_k and the transition-operator residual, which makes the two quantities mutually constraining — the radius sets the boundary tolerance, and the boundary residual penalizes chunks whose coverage is achieved by collapsing to a single mode. To break the circularity that T could absorb decoder offsets, we freeze T after fitting on expert boundary pairs only, so the residual measures decoder deviation from expert-consistent transitions rather than T's ability to fit decoder error; and we calibrate tau_bnd on expert boundary residuals, decoupling the acceptance test from the coverage radius.

**Why it measures what we claim.** The conformal radius r_k measures coverage of the expert action support because the exchangeability of calibration and test episodes makes the empirical quantile of the nonconformity score an exact finite-sample upper bound on the miss rate; this assumption fails when the test episode's action distribution shifts (e.g., a new object layout), in which case r_k reflects coverage on the stale calibration distribution rather than the deployed one. The boundary residual ||T(a_hat_k[-1]) - a_hat_{k+1}[0]||_2 measures boundary continuity because invertibility of T guarantees that a zero residual implies the two boundary actions are the same point in the operator's coordinate system; this assumption fails when T is near-singular (Lipschitz constant near 1), in which case a small residual can hide a large discontinuity in the original action space. The joint penalty lambda_bnd * L_bound measures mode collapse because a collapsed decoder produces chunks whose boundaries are systematically offset by the same mode gap, which the transition operator cannot absorb without violating its Lipschitz bound; this assumption fails when the expert distribution is genuinely unimodal, in which case the penalty is vacuous and the method reduces to OFT with a coverage certificate. The acceptance threshold tau_bnd measures expert-consistent boundary residuals because it is calibrated on expert boundary pairs only; this assumption fails when the expert chunking is arbitrary and seams fall mid-mode, in which case tau_bnd is uninformative about mode collapse.

## Contribution

(1) A fine-tuning recipe for vision-language-action models that augments the OFT single-pass parallel decoder with a per-chunk split-conformal coverage radius and an invertible Lipschitz-bounded boundary transition operator, producing chunks that are individually coverage-certified and jointly boundary-consistent. (2) The design principle that the conformal radius and the boundary residual must be coupled — the radius sets the boundary tolerance — because coverage without boundary consistency yields individually valid but jointly discontinuous chunks, and boundary consistency without coverage yields smooth but mode-collapsed trajectories. (3) An inference-time acceptance check that flags boundary discontinuities using the certified radius as the tolerance, giving a per-chunk signal that the OFT recipe cannot produce.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | LIBERO (four task suites) | Standard manipulation benchmark with expert chunk support |
| Primary metric | Success rate + boundary-residual calibration gap | Success tests task; gap tests claimed coverage-boundary coupling |
| Baseline | Original OpenVLA | Tests whether recipe beats naive autoregressive VLA |
| Baseline | OFT fine-tuned OpenVLA | Tests whether conformal+boundary coupling adds over OFT |
| Baseline | OFT with free-MLP boundary head | Tests whether invertibility (not just a boundary loss) matters |
| Ablation-of-ours | CADIBO without boundary penalty (lambda_bnd = 0) | Tests whether r_k/boundary coupling is load-bearing |
| Ablation-of-ours | CADIBO with T fit on decoder outputs (not frozen on expert pairs) | Tests whether freezing T on expert boundaries is necessary to avoid circularity |

### Why this setup validates the claim

The central claim is that the conformal radius and the invertible-transition boundary residual are *mutually constraining*: the radius sets a per-chunk tolerance, and the boundary residual penalizes chunks whose coverage is bought by collapsing to one mode. LIBERO is the right testbed because it has expert action chunks with genuine multi-modality across layouts, so chunk boundaries are exactly where mode collapse and discontinuity should appear. Original OpenVLA tests the coarse claim that this recipe beats naive autoregressive decoding; OFT tests the finer claim that conformal calibration plus boundary coupling adds something beyond a strong single-pass decoder. The free-MLP boundary baseline isolates the invertibility assumption: if a free MLP with the same L_bound penalty performs equally, the invertibility argument is decorative. The no-boundary ablation isolates the coupling itself. The T-fit-on-decoder-outputs ablation isolates the circularity fix: if fitting T on decoder outputs performs equally, freezing T on expert pairs is unnecessary. Success rate is the primary metric because the claim is about task execution, but the calibration gap (empirical miss rate vs. nominal alpha on held-out layouts) is what actually detects the predicted effect: if the radius is doing real work, the gap should stay near alpha on in-distribution layouts and widen predictably on shifted ones, while the boundary-flagged re-query rate should concentrate on multi-modal episodes.

### Expected outcome and causal chain

**vs. Original OpenVLA** — On a case where a manipulation step requires a decisive multi-step action chunk (e.g., a grasp-then-lift where autoregressive decoding drifts mid-chunk), the baseline produces a chunk that commits early and cannot recover because its token-by-token decoder has no chunk-level coverage estimate. Our method instead emits a calibrated chunk with a radius that flags low-confidence regions and a boundary check that catches drift at the chunk seam, because the parallel head plus conformal radius bounds the miss rate per chunk. We expect a noticeable success gap concentrated on long-horizon multi-step tasks, with parity on short single-grasp tasks.

**vs. OFT fine-tuned OpenVLA** — On a case where the expert action distribution is bimodal (two valid approach directions to an object), the baseline produces chunks that average the modes and land in the gap between them, because L1 regression to a point estimate has no mechanism to detect or penalize mode-averaging at boundaries. Our method instead penalizes boundary residuals that a Lipschitz-bounded invertible operator cannot absorb, because a mode-averaged chunk produces a systematically offset seam that violates the bound. We expect a modest success gain overall but a concentrated gain on multi-modal-layout episodes, with the calibration gap staying near alpha where OFT has no coverage guarantee at all.

**vs. OFT with free-MLP boundary head** — On a case where the transition operator is near-singular, the baseline produces a small residual that hides a large action-space discontinuity, because a free MLP can only be checked empirically and has no Lipschitz guarantee. Our method instead certifies the residual as a true bound because invertibility plus the spectral-norm clamp makes small residual imply small discontinuity. We expect parity on most tasks but a detectable advantage specifically on episodes where the free-MLP residual is small yet task failure occurs — the diagnostic signature of a hidden discontinuity.

**vs. CADIBO with T fit on decoder outputs** — On a case where the decoder systematically offsets boundaries due to mode averaging, the baseline's T learns to absorb those offsets because it is fit on decoder outputs, making the residual small even when collapse occurs. Our method instead freezes T on expert boundary pairs, so the residual measures decoder deviation from expert-consistent transitions. We expect the frozen-T variant to show a higher boundary-flagged re-query rate on multi-modal episodes and a larger success gain, confirming that the circularity fix is load-bearing.

### What would falsify this idea

If the gain over OFT is uniform across uni-modal and multi-modal layouts rather than concentrated on multi-modal episodes, the mode-collapse mechanism is not what produces the improvement. If the free-MLP boundary baseline matches CADIBO on the hidden-discontinuity subset, the invertibility argument is not load-bearing and the method reduces to OFT with a coverage certificate. If the T-fit-on-decoder-outputs ablation matches CADIBO, the circularity fix is unnecessary and the coupling claim is weakened.

## References

1. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success
2. HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents
3. AlphaEvolve: A coding agent for scientific and algorithmic discovery
4. RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation
5. RoboMIND: Benchmark on Multi-embodiment Intelligence Normative Data for Robot Manipulation
6. GenSim2: Scaling Robot Data Generation with Multi-modal and Reasoning LLMs
7. Amplifying human performance in combinatorial competitive programming
8. Multi-objective Evolution of Heuristic Using Large Language Model
