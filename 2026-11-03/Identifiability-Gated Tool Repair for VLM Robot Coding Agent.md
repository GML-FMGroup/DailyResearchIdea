# Identifiability-Gated Tool Repair for VLM Robot Coding Agents

## Motivation

Agent-authored geometric perception tools in systems like HexaShell are validated solely by downstream binary task success, which is invariant to a global metric-frame error: a tool library that is self-consistent but metrically wrong (e.g., triangulating with an incorrect baseline or intersecting a ray with a mis-estimated plane) passes every test. This is a structural limitation of the entire agentic-coding-for-robotics paradigm: the VLM repair loop optimizes task-success correlates rather than a geometric identifiability condition on the recovered quantities, so the agent cannot distinguish a correct metric frame from a globally rescaled one. The root cause is that no existing agent-authored tool emits, or is required to satisfy, a certificate that the metric quantities it recovers are identifiable from the observations actually used under unknown camera intrinsics, extrinsics, and scene planes.

## Key Insight

Metric identifiability of a geometric tool is a rank/conditioning property of the joint observation Jacobian coupling camera parameters and scene geometry, and it can be certified on a held-out probe configuration without any ground-truth metric labels — so requiring a non-degenerate certificate turns VLM tool repair from an unconstrained search over task-success correlates into a constrained search over identifiable parameterizations.

## Method

```pseudocode
INPUT: tool library T = {triangulation, plane_intersection, homography, ...}
       VLM agent A, observation logs L, probe configurations P (held-out, no GT metric)
OUTPUT: repaired tool library T* where every tool carries a valid identifiability certificate

# Phase 0: Observation-model extraction (restricted DSL / AST pattern library)
# Coverage table over tool set: triangulation (2-view, N-view), plane_intersection,
# homography, PnP, essential-matrix decomposition. Each pattern maps AST nodes to
# symbolic observation equations; tools not matching any pattern are flagged
# 'unparsed' and excluded from certification (reported in results).
for each tool t in T:
    t.observation_model = symbolic form inferred from t's source code via AST pattern library
        # e.g. triangulation: u_i = pi(K, R_i, C_i, X) for i in {1,2}
    theta = [K, {R_i, C_i}, plane_pi, X]   # camera intrinsics, extrinsics, scene plane, 3D point
    J = d(observations)/d(theta)           # joint Jacobian, numerically differentiated at the log's operating point

# Phase 1: Gauge-quotiented Jacobian assembly (per tool invocation)
# Explicitly project out known gauge freedoms before computing null space:
#   - global scale: project out direction d_scale = [0, 0, 0, X] (or equivalent)
#   - projective frame (homography): project out 8-DOF projective gauge
#   - similarity gauge (SfM): project out 7-DOF similarity
# Gauge basis G is constructed from the tool's declared gauge group (from DSL pattern).
# J_gauge = J * (I - G * pinv(G))   # projected Jacobian, same dimensions as J

# Phase 2: Identifiability certificate (local + global check)
# Local check: smallest singular value of column-scaled J_gauge
# Global check: stacked Jacobian over a spanning set of probe operating points
#   (at least 3 distinct probes per tool, chosen to span parameter manifold)
#   J_stack = [J_gauge(p1); J_gauge(p2); J_gauge(p3)]
#   s_stack = smallest singular value of J_stack (after column scaling)
#   rank_witness = (s_stack > tau_abs) AND (condition_number(J_stack) < tau_cond)
#   tau_abs = 1e-3, tau_cond = 1e3 (same as original)
# Probe coverage: report certificate as conditional on probe set P; if P does not
# span deployment manifold (measured by condition number of probe parameter matrix),
# flag 'coverage_limited'.
for each probe configuration p in P:        # P is held out from the tool-authoring layouts
    J_p = J_gauge evaluated at p
    s_p = smallest singular value of J_p (after column scaling by parameter magnitudes)
    certificate[t][p] = (s_p, rank_witness_p, null_space_basis(J_p))
# Aggregate over probes: stack J_p for all p in P (or a spanning subset of size >=3)
J_stack = [J_p for p in P_span]   # P_span = minimal spanning subset, size >=3
s_stack = smallest singular value of J_stack
rank_witness_global = (s_stack > tau_abs) AND (condition_number(J_stack) < tau_cond)
certificate[t].global = (s_stack, rank_witness_global, null_space_basis(J_stack))

# Phase 3: Gated tool-repair loop
# Load-bearing assumption (stated explicitly): The joint observation Jacobian,
# assembled from the tool's source code with camera intrinsics, extrinsics, scene
# planes, and 3D structure as free unknowns, has a non-degenerate smallest singular
# value (after gauge projection and stacking over spanning probes) exactly when the
# recovered metric quantities are identifiable in the deployment setting. This is a
# local-to-global proxy: local rank at a point certifies weak observability; stacking
# over spanning probes tests global identifiability up to the probe coverage.
t = current tool; L = failure log
for iteration in 1..N_max:                 # N_max = 20
    cert = certificate[t] over all probes (local) and global stacked certificate
    if cert.global.rank_witness and all(cert[p].rank_witness for p in P):
        accept t into T*; break
    else:
        # the agent is told WHICH parameter directions are unidentifiable
        null_dirs = union of null_space_basis(J_p) over failing probes, plus null_space_basis(J_stack)
        prompt A with: failure log L, tool source, null_dirs, and the instruction
            "rewrite t so that the recovered metric quantities are identifiable
             from the observations actually used; you may add observations,
             fix a gauge (e.g., assume a known metric anchor), or restructure the estimator"
        t = A.rewrite(t, prompt)
        recompute J_gauge, certificate[t]
return T*
```

**Why this design.** We chose to certify identifiability on a *held-out probe configuration* rather than on the tool-authoring layouts because a tool can be accidentally identifiable on the layouts it was tuned for (e.g., a degenerate camera baseline that happens to work on one seed) while being structurally unidentifiable in general; this follows the same train/test discipline HexaShell uses for task layouts, accepting the cost that probes must be constructed to span the parameter space. We chose the *joint* Jacobian over camera parameters and scene geometry rather than separate per-tool Jacobians because the meta-gap is precisely that camera calibration and scene planes are treated as known in HexaShell, so identifiability must be assessed over the joint unknown; the cost is a larger Jacobian and more expensive numerical differentiation. We chose to feed the *null-space directions* back to the VLM agent rather than a scalar certificate score because a scalar gives no repair gradient and would collapse the loop into trial-and-error (the same failure mode as AlphaEvolve when the evaluator is a single scalar); the cost is that the agent must be able to interpret geometric null directions, which we mitigate by rendering them as parameter-space hints. A domain expert would not describe this as a variant of HexaShell's tool-repair loop: HexaShell accepts a tool when downstream task success is high, whereas here acceptance is gated by a rank witness on the joint observation Jacobian that is independent of task outcome. We additionally project out known gauge freedoms (scale, projective frame) before computing the null space so that the witness tests genuine ambiguity rather than gauge, and we stack the Jacobian over a spanning set of probes (at least 3) to test global identifiability rather than local conditioning at a single point; the cost is that the probe set must be certified as spanning the deployment parameter manifold, which we report as a coverage condition.

**Why this measures what we claim.** The smallest singular value `s_stack` of the column-scaled, gauge-quotiented, stacked joint Jacobian measures *metric identifiability* of the recovered quantities because, under the assumption that the observation model is locally linear and the noise is bounded, a non-negligible `s_stack` is equivalent to the parameter being recoverable up to the noise level across the probe set; this assumption fails when the observation model is strongly nonlinear near the operating point (e.g., a point near the camera's principal axis in triangulation), in which case `s_stack` reflects local conditioning rather than global identifiability. The `rank_witness_global` boolean measures the *existence* of an identifiable parameterization because it tests whether the null space of the gauge-quotiented stacked Jacobian is trivial; this assumption fails when the tool intentionally uses a gauge freedom (e.g., homography up to scale), in which case the rank witness must be computed on the gauge-fixed quotient, and if it is not, the witness reflects an unidentifiable gauge rather than a genuine geometric ambiguity. The held-out probe aggregation measures *structural* rather than *incidental* identifiability because probes are drawn independently of the tool-authoring layouts and stacked to span the parameter manifold; this assumption fails when the probe distribution does not cover the failure modes the tool will encounter at test time, in which case the certificate reflects identifiability on the probe manifold rather than on the deployment manifold, and we report the certificate as conditional on probe coverage.

## Contribution

['A calibration-identifiability closure layer for agent-authored geometric perception tools: a joint observation-Jacobian certificate that couples unknown camera intrinsics/extrinsics with unknown scene geometry and emits a rank/conditioning witness alongside every point estimate, computed on held-out probe configurations without ground-truth metric labels.', 'A constrained VLM tool-repair loop that accepts a tool only when its identifiability certificate is non-degenerate on every probe, and that feeds null-space directions back to the agent as a repair gradient — converting tool repair from search over task-success correlates into search over identifiable parameterizations.', 'A design principle for agentic robot coding: metric correctness of perception tools is a structural property of the observation model, not a task outcome, and can therefore be certified independently of downstream binary task success.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | HexaShell ten-layout manipulation suite, held-out probes (at least 3 spanning probes per tool) | Same task distribution, independent probe manifold |
| Primary metric | Fraction of tools with valid rank witness (gauge-quotiented, stacked over spanning probes) at deployment | Directly measures the claimed certificate property |
| Secondary metric | Ground-truth metric recovery error (available offline in simulation) | Breaks circularity between certificate and primary metric |
| Baseline | HexaShell task-success tool-repair loop | Tests whether outcome-gating alone suffices |
| Baseline | AlphaEvolve scalar-evaluator repair | Tests whether scalar feedback is enough |
| Baseline | Joint-Jacobian certificate with scalar score feedback | Isolates value of null-space directions |
| Baseline | Classical bundle-adjustment observability certificate (Schur-complement null-space analysis of joint BA Jacobian) | Tests whether the certificate itself is novel or the closed-loop repair is |
| Ablation-of-ours | Certify on tool-authoring layouts, not held-out probes | Tests whether held-out probes matter |
| Ablation-of-ours | Raw smallest-singular-value rank witness (no gauge projection, no stacking) | Tests whether gauge-quotient and global check are load-bearing |

### Why this setup validates the claim

The central claim is that gating tool acceptance on a rank witness of the joint observation Jacobian, evaluated on held-out probes and repaired via null-space directions, produces tools whose recovered metric quantities are structurally identifiable rather than accidentally identifiable. HexaShell's task-success loop tests the sub-claim that downstream success is a sufficient gate; if our method matches HexaShell on deployment identifiability, the certificate adds nothing. AlphaEvolve's scalar-evaluator repair tests whether the null-space directions are load-bearing: a scalar certificate gives no repair gradient, so if the scalar-feedback variant closes the gap, our geometric feedback is redundant. The classical bundle-adjustment observability baseline tests whether the certificate mechanism itself is novel: if the classical certificate matches our primary metric, the novelty is in the closed-loop repair-with-null-directions, not the certificate. The ablation isolates the held-out-probe design choice from the certificate mechanism itself, so a gain cannot be attributed to certification alone. The raw-witness ablation tests whether gauge projection and stacking are necessary for the certificate to measure structural identifiability. The primary metric — fraction of tools with a valid rank witness on deployment-time configurations — is the right detector because it is exactly the property the method claims to install, and it is evaluated independently of the task outcome that HexaShell optimizes, so any gap cannot be explained by task-success overfitting. The secondary metric — ground-truth metric recovery error, available offline in simulation — breaks the circularity between the certificate and the primary metric by providing an independent measure of whether the recovered quantities are metrically correct, not merely certified.

### Expected outcome and causal chain

**vs. HexaShell task-success tool-repair loop** — On a case where a triangulation tool is authored on a layout with a wide, well-conditioned camera baseline, the baseline accepts the tool because downstream task success is high, even though the tool's depth recovery is unidentifiable under the narrow baselines encountered at deployment. Our method instead rejects the tool at the certificate stage because the smallest singular value of the gauge-quotiented stacked joint Jacobian collapses along the depth direction on held-out probes, and feeds that null direction back so the agent adds a metric anchor. We expect a noticeable gap in valid-rank-witness fraction on narrow-baseline deployment layouts, with near-parity on wide-baseline layouts where HexaShell's accidental identifiability happens to hold. The secondary metric (ground-truth metric error) should show a corresponding gap: HexaShell-accepted tools should have higher depth error on narrow baselines despite high task success.

**vs. AlphaEvolve scalar-evaluator repair** — On a case where a plane-intersection tool is structurally unidentifiable because the scene plane is treated as known, the baseline receives a single scalar certificate score and must search blindly, consuming its iteration budget on rewrites that do not touch the ambiguous parameter. Our method instead hands the agent the null-space basis, which names the plane-normal direction as the unidentifiable quantity, so the rewrite directly fixes the gauge. We expect the scalar-feedback variant to plateau at a lower valid-certificate fraction within the same iteration budget, with the gap concentrated on tools whose null space is high-dimensional.

**vs. Joint-Jacobian certificate with scalar score feedback** — On a case where a homography tool has a legitimate scale gauge, the baseline's scalar score is low and the agent cannot tell gauge freedom from genuine geometric ambiguity, so it over-constrains the estimator and breaks valid tools. Our method instead renders the null direction as a parameter-space hint, so the agent gauge-fixes rather than over-constrains. We expect the scalar variant to show both missed repairs and spurious rewrites, while our method shows repairs concentrated on genuinely ambiguous tools and preservation of gauge-legitimate ones.

**vs. Classical bundle-adjustment observability certificate** — On a case where a triangulation tool is locally full-rank but globally ambiguous due to a critical motion (e.g., pure translation with unknown intrinsics), the classical Schur-complement null-space analysis of the joint BA Jacobian at a single operating point may report full rank, whereas our stacked certificate over spanning probes detects the global ambiguity. We expect the classical certificate to match our primary metric on locally-conditioned tools but to diverge on tools with critical-motion degeneracies, with our method showing higher valid-rank-witness fraction on those cases and lower ground-truth metric error after repair.

### What would falsify this idea

If the valid-rank-witness gain over HexaShell is uniform across deployment layouts rather than concentrated on the narrow-baseline and degenerate-plane subsets where the predicted identifiability failure occurs, the certificate is not measuring structural identifiability. Likewise, if the null-space-feedback variant matches the scalar-feedback variant on high-dimensional-null-space tools, the geometric repair gradient is not doing the work we claim. Additionally, if the ground-truth metric error does not correlate with the certificate's smallest singular value (e.g., correlation coefficient below 0.5 across tools), the certificate is not a sound proxy for metric identifiability.

## References

1. HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents
2. AlphaEvolve: A coding agent for scientific and algorithmic discovery
3. RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation
4. RoboMIND: Benchmark on Multi-embodiment Intelligence Normative Data for Robot Manipulation
5. GenSim2: Scaling Robot Data Generation with Multi-modal and Reasoning LLMs
6. Amplifying human performance in combinatorial competitive programming
7. Multi-objective Evolution of Heuristic Using Large Language Model
8. LLMatic: Neural Architecture Search Via Large Language Models And Quality Diversity Optimization
