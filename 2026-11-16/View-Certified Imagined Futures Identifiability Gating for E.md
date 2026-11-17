# View-Certified Imagined Futures: Identifiability Gating for Embodied Foundation-Model Reasoning

## Motivation

Agentic robotics harnesses such as the Embodied System 2 architecture of Envisioning Agentic Robotics (arxiv:2611.18359) treat the foundation model's imagined futures as if they were verified observations, and EnerVerse (venue:neurips:2025:0b10731d5c70) supplies those futures as multi-view video-diffused 4D spaces whose 3D structure is only partially constrained by the available camera views. The structural gap is that no existing embodied foundation model certifies that its imagined 4D future space is uniquely determined by the views it was conditioned on, so downstream reasoning consumes physically ambiguous imagined states as if they were grounded observations. This ambiguity is not a data-scarcity issue but a geometric one: with fewer than the required independent view constraints, multiple distinct 3D configurations project to the same multi-view evidence, and the diffusion prior silently picks one.

## Key Insight

Under a multi-view epipolar observability condition, the set of 3D future states consistent with a predicted multi-view video is an affine subspace whose dimension is computable in closed form from the camera geometry and the predicted depth/flow fields, so identifiability can be certified before reasoning rather than assumed.

## Method

**What it is.** VCI (View-Certified Imagination) is a wrapper around an EnerVerse-style multi-view video-diffusion future-space predictor that takes as input the predicted multi-view future video, the known camera intrinsics/extrinsics of the K conditioning views, the actually observed conditioning frames (real pixels and measured depth where available), and the predicted per-view depth and optical-flow fields, and outputs a per-future-chunk identifiability certificate (a calibrated probability of geometric error p_err and a binary certificate) that the agentic harness's reason stage must check before consuming the imagined future as evidence.

**Load-bearing assumption (stated explicitly).** The certificate is only meaningful if the linearized multi-view reprojection Jacobian A_t, built from the predicted depth/flow, has a null space equal to the true geometric ambiguity of the imagined 4D scene. This holds when the predicted depth is within a bounded error of true depth (assumption A: ||D_pred - D_true|| <= delta_D, with delta_D estimated per-scene from the sim-to-real calibration set) and the predicted flow is rigid-motion consistent. It fails when the diffusion prior hallucinates smooth, self-consistent depth in textureless or near-degenerate-baseline regions, in which case A_t is well-conditioned on a wrong linearization and the gate would PASS an underdetermined chunk. To prevent this self-certification, VCI anchors the constraint matrix and residuals on the actually observed conditioning views (real pixels, measured depth where available) rather than solely on the prior's predicted depth/flow, and calibrates all thresholds against a held-out set with known ground-truth 3D geometry.

**How it works.**

```pseudocode
# Inputs: K conditioning views with poses T_k, intrinsics K_k
#         observed conditioning frames O_k (real pixels) and measured depth M_k where available
#         predicted future video V (K views, T frames)
#         predicted depth D_k,t and flow F_k,t per view/frame
#         calibration set C_cal with known ground-truth 3D (used offline to fit thresholds)
# Output: calibrated error probability p_err in [0,1], certificate c in {PASS, REJECT}, residual rho, ambiguity basis B

# --- Offline calibration (once, on C_cal) ---
# Fit noise model: sigma_e, sigma_g, sigma_d from residuals on C_cal where GT 3D is known.
# Fit prior-bias term b_prior = E[||D_pred - D_true||] on C_cal (sim-to-real calibration).
# Set tau = 3 * sigma_d / sqrt(N)  (rank tolerance tied to depth noise, not fixed 1e-3)
# Set m_min, a_max, e_max, g_max by maximizing balanced accuracy of PASS/REJECT
#   against GT-identifiability labels on C_cal (labels: rank of GT Jacobian < 3N - 1).
# Fit logistic calibrator p_err = sigmoid(w0 + w1*m_t + w2*a_t + w3*e_bar + w4*g_bar + w5*b_prior)
#   on C_cal, with w fit by maximum likelihood; report Brier score and reliability curve.

1. For each future frame t and each pair (k, l):
     a. Warp observed view k at t to view l using measured depth M_k,t (where available)
        else predicted depth D_k,t, and pose T_l^{-1} T_k  ->  V_hat_{k->l}
     b. Photometric residual against real pixels: e_{k,l,t} = || O_l,t - V_hat_{k->l} ||^2
     c. Epipolar-consistency residual on F_k,t: distance of flow endpoint
        to the epipolar line induced by T_l^{-1} T_k   ->  g_{k,l,t}
     d. Prior-bias residual: b_{k,t} = || D_k,t - M_k,t ||^2 where M_k,t measured, else b_prior

2. Assemble the linearized multi-view constraint matrix A_t in R^{(K choose 2) x 3N}
   from the Jacobians of the warp w.r.t. per-point 3D coordinates (N=512 sampled points),
   using observed pixels and measured depth where available. Rank-revealing QR with
   tolerance tau = 3 * sigma_d / sqrt(N) (calibrated above) gives rank r_t and null basis B_t (3N - r_t columns).

3. Observability margin: m_t = sigma_{r_t}(A_t) / sigma_1(A_t)   (smallest retained /
   largest singular value).  Ambiguity energy: a_t = || B_t^T d_t ||^2 where d_t is the
   predicted depth-residual vector.  Aggregate residuals e_bar = mean_{k,l} e_{k,l,t},
   g_bar = mean_{k,l} g_{k,l,t}, b_bar = mean_k b_{k,t}.

4. Calibrated certificate: p_err,t = sigmoid(w0 + w1*m_t + w2*a_t + w3*e_bar + w4*g_bar + w5*b_bar)
   with weights fit on C_cal.  Binary decision:
     PASS  iff  p_err,t <= p_max (=0.5, calibrated to equal error rate on C_cal)
              AND  m_t >= m_min (calibrated)  AND  a_t <= a_max (calibrated)
              AND  e_{k,l,t} <= e_max (calibrated) for all pairs
              AND  g_{k,l,t} <= g_max (calibrated) for all pairs

5. Aggregate over the action-chunk horizon T_c: chunk passes iff >= ceil(0.8 T_c)
   frames pass.  If REJECT, return the null basis B (the unobservable directions)
   and p_err to the harness, which must either request an additional view or downgrade
   the future from 'observation' to 'hypothesis' before the reason stage.
```

**Why this design.** We chose a closed-form geometric certificate over a learned identifiability classifier because the ambiguity of a multi-view 3D reconstruction is exactly the null space of the linearized reprojection operator, a quantity with a known closed form, so a learned head would only add estimation error; the cost we accept is that the certificate is only as good as the predicted depth and flow, which we therefore must validate with the photometric and epipolar residuals and anchor on observed conditioning views. We chose to gate at the action-chunk level with an 80% frame-pass fraction rather than per-frame because a single ambiguous frame can be harmless if neighboring frames pin it down, accepting the cost that a systematically ambiguous sub-window can slip through if it is shorter than 20% of the chunk. We chose to return the null basis B rather than only a binary flag because the harness needs to know which direction is unobservable in order to decide whether an extra view would help, accepting the cost of a larger interface than a scalar score. We chose to calibrate tau, m_min, a_max, e_max, g_max and the logistic weights on a held-out set with known ground-truth 3D rather than fix them a priori, because the rank decision must be a function of measurement noise and prior bias, not a numerical artifact of tolerance choice. A domain expert would not describe this as a variant of EnerVerse or of a generic uncertainty estimator: EnerVerse predicts futures without any observability analysis, and standard uncertainty estimators (ensemble variance, diffusion log-likelihood) measure prediction spread, not geometric rank deficiency, so they cannot certify uniqueness.

**Why it measures what we claim.** The observability margin m_t measures multi-view identifiability because the assumption that the reprojection Jacobian is locally linear and that the predicted depth is within a bounded error delta_D of true depth (assumption A, estimated on the calibration set) makes the smallest retained singular value the exact inverse conditioning of the 3D reconstruction; this assumption fails when the predicted depth has a systematic bias (e.g., textureless regions where the diffusion prior hallucinates depth), in which case m_t reflects the conditioning of a wrong linearization rather than true observability, and the prior-bias term b_bar in the calibrated p_err is designed to flag exactly this regime. The ambiguity energy a_t measures the component of the predicted depth residual lying in the unobservable subspace because the assumption that the null basis B_t is correctly estimated by rank-revealing QR at tolerance tau = 3*sigma_d/sqrt(N) (tied to depth noise) makes B_t^T d_t the exact residual that cannot be explained by any view; this assumption fails when the rank threshold tau is mis-set relative to noise, in which case a_t reflects numerical rank choice rather than genuine ambiguity, which is why tau is calibrated rather than fixed. The epipolar residual g_{k,l,t} measures whether the predicted flow is geometrically consistent with the camera poses because the assumption that the flow is a projection of rigid 3D motion makes epipolar deviation a direct test of that assumption; this assumption fails under non-rigid or articulated motion, in which case g reflects motion non-rigidity rather than pose/depth error. The calibrated probability p_err measures the probability that the imagined future has geometric error because the assumption that the logistic model is fit on a calibration set representative of the test distribution makes p_err a calibrated estimate; this assumption fails under distribution shift, in which case p_err reflects the calibration set's bias rather than the test scene's true error probability.

## Contribution

['A view-certified imagination wrapper (VCI) that computes a closed-form multi-view observability margin and null-space ambiguity energy from predicted depth/flow and camera poses, producing a per-action-chunk identifiability certificate before any reasoning consumes the imagined future.', 'A design principle for agentic embodied foundation models: the reason stage must consume only certified-identifiable imagined futures, with the null basis returned to the harness to drive view acquisition or hypothesis downgrade, rather than treating imagined futures as verified observations.', "An interface and evaluation protocol that distinguishes 'physically ambiguous imagined future' from 'prediction error', enabling the harness to attribute downstream failures to observability rather than model quality."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Multi-view robot manipulation episodes with held-out ambiguous scenes, plus a synthetic ambiguous subset bootstrapped by subsampling conditioning views to create controlled near-degenerate baselines and masking textureless regions (rank deficiency known by construction) | Provides poses, depth, flow, and observed conditioning frames for certificate checks; synthetic subset labels ambiguous/well-conditioned without manual curation |
| Primary metric | Task success rate on geometrically ambiguous subset | Directly tests whether certificates prevent bad imagination |
| Secondary metric | Certificate false-PASS / false-REJECT rate vs. ground-truth identifiability labels on the synthetic subset | Quantifies operating envelope of the linearization assumption |
| Baseline | EnerVerse-style predictor, no certificate | Isolates value of gating imagined futures |
| Baseline | Diffusion ensemble-variance uncertainty gate | Tests whether prediction spread substitutes for geometric rank |
| Baseline | Monocular depth-confidence gate | Tests whether single-view confidence substitutes for multi-view identifiability |
| Baseline | Classical reprojection-conditioning-number gate from bundle adjustment (smallest/largest singular value of the reprojection Jacobian, fixed threshold) | Tests whether VCI's null-space-energy formulation adds value beyond a known geometric tool |
| Ablation-of-ours | VCI with binary flag only, no null basis | Isolates value of returning ambiguity directions |
| Ablation-of-ours | VCI with fixed thresholds (tau=1e-3, m_min=0.05, a_max=0.02, e_max=0.01, g_max=1.5px) vs. calibrated thresholds | Tests whether calibration against ground-truth 3D is necessary to avoid self-certification |
| Ablation-of-ours | VCI with predicted depth only vs. observed conditioning frames + measured depth where available | Tests whether anchoring on real observations prevents the self-referential failure mode |

### Why this setup validates the claim

The central claim is that multi-view geometric rank deficiency — not prediction spread or single-view confidence — is the quantity that determines whether an imagined future can be trusted as evidence for the harness's reason stage. The dataset choice makes this falsifiable because it must contain scenes where the conditioning views are genuinely rank-deficient (e.g., near-degenerate baselines, textureless surfaces, occluded geometry) alongside scenes where they are well-conditioned; the synthetic bootstrap makes the ambiguous/well-conditioned split known by construction, so the claim can be tested without manual scene curation. The primary metric — task success on the ambiguous subset — is the right detector because the predicted effect is specifically that gating on geometric identifiability prevents the harness from acting on hallucinated geometry, so any benefit should be concentrated there and should be near-parity on well-conditioned scenes. The EnerVerse-style baseline tests the sub-claim that a strong future predictor alone is insufficient; the ensemble-variance and depth-confidence baselines test the sub-claim that existing uncertainty proxies do not capture rank deficiency, since they can be confidently wrong when the diffusion prior is self-consistent but geometrically underdetermined. The classical reprojection-conditioning-number baseline tests whether VCI's specific null-space-energy formulation adds value beyond a known geometric tool, and we report where the two diverge. The null-basis ablation tests whether the interface's directional information, not merely the binary decision, drives the improvement — if the binary flag matches the full certificate, the null basis is decorative and the design rationale is wrong. The fixed-threshold and predicted-depth-only ablations test the load-bearing assumption directly: if calibration and observation anchoring do not improve false-PASS rates on the synthetic ambiguous subset, the self-certification concern is not mitigated and the method's certificate is not measuring true identifiability.

### Expected outcome and causal chain

**vs. EnerVerse-style predictor, no certificate** — On a case where the conditioning views have a small effective baseline and the target surface is textureless, the un-gated predictor produces a future video that is photometrically plausible in each view but geometrically underdetermined, so the harness consumes a hallucinated depth configuration as observation. Our method computes the reprojection Jacobian against observed conditioning frames, finds a near-zero smallest singular value, and rejects the chunk before the reason stage. We expect a noticeable success-rate gap on the ambiguous subset but near-parity on the well-conditioned subset, since the gate should rarely fire when rank is healthy.

**vs. Diffusion ensemble-variance uncertainty gate** — On a case where the diffusion prior is confidently self-consistent about a wrong 3D layout, ensemble variance stays low because all samples agree, so the baseline admits a geometrically ambiguous future as evidence. Our method instead measures the null-space energy of the predicted depth residual, which is large precisely when the layout is unconstrained by the views regardless of sample agreement. We expect the variance baseline to show low uncertainty on rejected-by-ours chunks, producing a clear divergence pattern: our gate fires where variance is quiet, and task success improves only on those chunks.

**vs. Monocular depth-confidence gate** — On a case where a single view has high per-pixel depth confidence but the multi-view configuration is rank-deficient along one direction, the monocular gate passes the future because it never checks cross-view consistency. Our method's epipolar and photometric residuals plus the observability margin detect the unconstrained direction and return the null basis, letting the harness request an extra view. We expect our method to outperform on ambiguous subsets while the monocular gate's advantage is confined to single-view noise cases, giving a subset-specific gap rather than a uniform one.

**vs. Classical reprojection-conditioning-number gate** — On a case where the reprojection Jacobian is well-conditioned overall but the predicted depth residual lies almost entirely in a low-energy null direction, the classical gate passes because its smallest singular value is above threshold, while our ambiguity energy a_t is large. We expect divergence on these chunks: our gate rejects where the classical gate passes, and task success improves only on the divergent set, showing that null-space energy adds value beyond the conditioning number.

**vs. VCI with fixed thresholds (tau=1e-3, m_min=0.05, a_max=0.02, e_max=0.01, g_max=1.5px)** — On the synthetic ambiguous subset where the diffusion prior hallucinates smooth, self-consistent depth, the fixed-threshold version computes a well-conditioned Jacobian from the predicted depth and passes the chunk (false PASS). The calibrated version, anchored on observed conditioning frames and with tau tied to depth noise, rejects the same chunk because the prior-bias term b_bar and the observed-pixel photometric residual are large. We expect the calibrated version to reduce false-PASS rate on the synthetic ambiguous subset while maintaining similar false-REJECT rate on well-conditioned scenes.

**vs. VCI with predicted depth only (no observed conditioning frames)** — On textureless regions, the predicted-depth-only version is self-referential and passes underdetermined chunks, while the observed-frame-anchored version detects the inconsistency between predicted depth and real pixels. We expect the anchored version to show lower false-PASS rate on the synthetic ambiguous subset, confirming that anchoring on real observations is necessary to avoid self-certification.

### What would falsify this idea

If the certificate's gain over the ensemble-variance, monocular-confidence, and classical conditioning-number baselines is uniform across ambiguous and well-conditioned subsets — rather than concentrated where geometric rank deficiency occurs — then the method is acting as a generic filter, not as a geometric identifiability certificate, and the central claim is wrong. Similarly, if the binary-flag ablation matches the full null-basis version, the claimed value of returning ambiguity directions is falsified; and if the calibrated-threshold and observed-frame-anchored versions do not reduce false-PASS rate on the synthetic ambiguous subset relative to the fixed-threshold and predicted-depth-only versions, the load-bearing assumption is not mitigated and the certificate is self-referential.

## References

1. Envisioning Agentic Robotics: A New Era of Robotic Systems
