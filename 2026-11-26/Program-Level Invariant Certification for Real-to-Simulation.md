# Program-Level Invariant Certification for Real-to-Simulation Transfer

## Motivation

Real2SimLab unifies program-execution and benchmark-native scoring across 6 benchmarks and 47 coding agents, but its output is a reward proxy, not a property certificate: a generated simulator can score well while violating the very invariants the downstream sim-to-real method relies on (e.g., SPiDR's pessimistic safety-set containment, DEAL's transition-dynamics identifiability, or OT distributional alignment in co-training). Because no existing platform checks these method-specific properties, a high Real2SimLab score is structurally uninformative about whether the generated simulator is usable by SPiDR, DEAL, or co-training pipelines — the platform certifies benchmark fit, not downstream-method compatibility.

## Key Insight

A generated simulator's downstream-method compatibility is a checkable property of its rollout traces, not a latent quality inferred from its benchmark reward, so running the program under the exact invariant each method requires yields a certificate that is logically independent of (and can contradict) the benchmark score.

## Method

**What it is.** PICR (Program Invariant Certification for Real2Sim) takes as input (i) a simulation program P produced by any Real2SimLab-supported method, (ii) a benchmark instance B with its native scorer s_B, (iii) a downstream method tag m ∈ {SPiDR, CoTrain, DEAL, InductiveSBI}, and (iv) a probe policy π_m — the data-collection distribution the downstream method itself uses (SPiDR: randomized-parameter exploration policy with domain-randomization ensemble of size 5; CoTrain: demonstration-collection policy that generated the 10 real demos; DEAL: Sys-Id excitation trajectory with sinusoidal parameter sweeps at 0.5 Hz; InductiveSBI: calibration-prior sampler over the latent prior). It outputs a certificate C = (score_B, invariant_id, value, pass/fail, witness_trace, probe_policy_id, support_overlap_estimate) alongside the native score. If π_m is unavailable, the certificate is emitted as `conditional_pass_under_probe_X` rather than an unconditional pass.

**How it works.**

```pseudocode
INPUT: program P, benchmark B, method tag m, probe policy π_m,
       rollout budget N=200, horizon H=200, tolerance tau
1. RUN P in Real2SimLab sandbox -> simulator S_P (env, dynamics, reset)
2. SCORE: r_B <- s_B(S_P)                       # native benchmark reward
3. EXECUTE N=200 rollouts of S_P under probe π_m, recording traces
   T_m = {(o_t, a_t, o_{t+1})} for t=1..H, H=200
   ALSO execute N=200 rollouts under benchmark policy π_B -> T_B
4. SUPPORT OVERLAP: fit density-ratio estimator d(o,a) between T_m and T_B
   via logistic regression (2-layer MLP, hidden=128, GeLU, 200 epochs, lr=1e-3);
   support_overlap_estimate <- importance-weight ESS = (sum w)^2 / sum w^2
   with w = d/(1-d), clipped at 10.0; report alongside pass/fail.
5. INVARIANT CHECK by m, evaluated on T_m (not T_B):
   if m == SPiDR:
       fit safety critic Q_safe on T_m via pessimistic TD (alpha=0.1,
       ensemble of 5 critics, each 2-layer MLP hidden=256 GeLU,
       50k gradient steps, batch=256)
       V_safe <- min over ensemble
       PASS iff for all visited unsafe-adjacent states, V_safe <= 0
       witness <- argmax V_safe state
   if m == CoTrain:
       embed T_m and held-out T_real (10 demos) via task encoder
       (ResNet-18 pretrained, frozen; 128-d projection)
       W <- unbalanced-OT coupling (marginal KL reg = 1.0, Sinkhorn
       iterations=100, epsilon=0.01)
       PASS iff W2(obs,act joint) <= tau=0.05 and marginal-gap ratio <= 1.2
       witness <- max-cost transport pair
   if m == DEAL:
       run DEAL's discriminator (2-layer MLP hidden=256, LeakyReLU 0.2)
       on T_m vs T_real transitions
       AUC <- 5-fold cross-validated AUC on held-out 20% of T_m
       PASS iff AUC in [0.45,0.55] AND permutation test (1000 shuffles)
       rejects null of no dynamics mismatch at p<0.05 AND parameter
       Fisher information rank >= 0.9 * dim(theta), computed on T_m
       witness <- least-identified parameter coordinate
   if m == InductiveSBI:
       split T_m into calibration (20%) and eval (80%) by TEMPORAL BLOCK
       (contiguous trajectory segments, block length=H/5=40 steps)
       train conditional normalizing flow (RealNVP, 6 coupling layers,
       hidden=128) on calibration via mini-batch OT (batch=64)
       PASS iff eval posterior coverage at 90% nominal in [0.85,0.95]
       and no simulation access needed at eval (amortization check)
       witness <- worst-calibrated latent
6. EMIT C = (r_B, invariant_id, value, pass, witness, probe_policy_id,
            support_overlap_estimate); attach to Real2SimLab record.
   If support_overlap_estimate < 0.1, downgrade pass to
   `conditional_pass_low_overlap`.
```

**Why this design.** We chose sandboxed rollout execution over static program inspection because the invariants (safety-set containment, transition identifiability, posterior coverage) are properties of the *executed* dynamics, not the source code, accepting the cost that certification requires N=200 rollouts under both π_m and π_B and thus ~200× the runtime of Real2SimLab's single scoring pass. We chose method-native invariant definitions (e.g., DEAL's own discriminator, SPiDR's pessimistic critic) over a single generic distribution-distance proxy because SPiDR, DEAL, co-training, and inductive SBI each fail under different geometric conditions — a single MMD score would not distinguish a simulator that is OT-aligned but safety-violating — accepting the cost that adding a new downstream method requires writing a new invariant module and supplying its probe policy. We chose to emit a witness trace alongside the binary pass/fail over a scalar certificate because a bare score cannot be audited or used for program repair, accepting the cost of larger certificate storage. We chose unbalanced OT with marginal-KL regularization for the co-training check over balanced OT because the real demonstrations are scarce (10) and balanced OT would silently over-weight them, accepting the cost of one extra regularization hyperparameter. We chose to collect traces under the downstream method's own probe policy π_m rather than the benchmark policy π_B because the certificate must be a statement about the distribution the downstream method will actually see; we report the support-overlap ESS between π_m and π_B so the reader can see how far the certificate is from a benchmark-policy statement. Critically, this is not a variant of Real2SimLab's scoring harness: Real2SimLab's scorer is a *reward proxy* aggregated into a single number, whereas PICR's invariants are *predicates over π_m rollout traces* whose truth value is independent of the reward — a program can pass the benchmark score and fail every invariant, which is precisely the failure Real2SimLab cannot detect.

**Why it measures what we claim.** The safety-containment check measures SPiDR-compatibility because pessimistic TD with an ensemble minimum on π_m traces is equivalent to the safety-set containment SPiDR requires *under the assumption that π_m is SPiDR's randomized-parameter exploration policy and the ensemble-min estimator is a consistent lower bound on the target-policy value under support overlap*; this assumption fails when π_m is unavailable and π_B is substituted, in which case the check reflects coverage of π_B's trajectory rather than SPiDR's safety set, and the certificate is downgraded to `conditional_pass_low_overlap`. The unbalanced-OT gap measures co-training compatibility because the joint (obs, act) transport cost on π_m traces is equivalent to the domain-invariant feature alignment the co-training loss optimizes *under the assumption that the task encoder used for the check matches the encoder co-training would learn and the 10 real demos are representative*; this fails when the held-out demos are unrepresentative, in which case the gap reflects demo-sampling noise. The discriminator AUC plus Fisher-rank check measures DEAL compatibility because AUC≈0.5 with full-rank Fisher information on π_m traces is equivalent to the transition-dynamics identifiability DEAL's Sys-Id targets *under the assumption that the discriminator architecture matches DEAL's and the permutation test has power to detect the relevant mismatch at N=200*; this fails when the discriminator overfits, in which case AUC reflects discriminator capacity rather than dynamics mismatch, which is why we report a permutation-test p-value alongside the AUC window. The posterior coverage check measures inductive-SBI compatibility because nominal-vs-empirical coverage on temporally blocked held-out traces is equivalent to amortized calibration *under the assumption that the blocked split removes temporal correlation*; this fails when blocks leak, in which case coverage reflects leakage.

## Contribution

["A property-certification layer (PICR) that extends Real2SimLab's program-execution harness with method-native invariant checks, emitting a per-program certificate (invariant value, pass/fail, witness trace) alongside the benchmark-native score.", 'A design principle that downstream-method compatibility is checkable as a predicate over rollout traces rather than inferable from benchmark reward, demonstrated by constructing four invariant modules whose pass/fail can contradict the benchmark score.', 'An open certification record format and witness-trace schema that makes Real2SimLab records auditable and usable for program repair.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Real2SimLab 6-benchmark suite, scoped to 2 benchmarks (MuJoCo locomotion + Franka manipulation) for Phase 1; remaining 4 in Phase 2 | Cross-benchmark programs with native scorers; Phase 1 keeps runtime tractable |
| Primary metric | Invariant-fail rate among score-passing programs, conditioned on probe policy π_m | Isolates failures native scoring cannot see, under the distribution the downstream method actually uses |
| Secondary metric | Downstream-degradation precision/recall: PICR failures vs. measured downstream performance drop (success rate or safety-violation rate) when running SPiDR/CoTrain on each program | Converts the certificate from diagnostic to decision tool |
| Baseline | Real2SimLab native scorer only | Tests whether reward alone detects invariant violations |
| Baseline | Generic MMD distribution-distance certificate | Tests whether one generic proxy suffices |
| Baseline | Random-invariant certificate | Tests whether method-native predicates matter |
| Baseline | PICR with π_B rollouts (no probe policy) | Tests whether probe-policy collection is load-bearing vs. benchmark-policy collection |
| Ablation-of-ours | PICR with method tag collapsed to single generic invariant | Tests necessity of per-method invariant modules |
| Ablation-of-ours | PICR with α ∈ {0.01, 0.1, 0.5}, τ ∈ {0.02, 0.05, 0.1}, ensemble size ∈ {3, 5, 10}, N ∈ {50, 100, 200, 400} | Calibration sweep to show pass/fail decisions are stable in a documented region rather than at a single operating point |
| Ablation-of-ours | Cross-method invariant transfer: certify under tag m, evaluate downstream degradation under m' ≠ m | Tests whether invariant geometry is method-specific rather than a generic quality signal |

### Why this setup validates the claim

The central claim is that a program can pass Real2SimLab's native benchmark score while violating the executed-dynamics properties that SPiDR, co-training, DEAL, and inductive SBI each depend on, and that method-native invariant predicates over probe-policy rollout traces detect this where a single reward number cannot. The native-scorer baseline directly tests the first half: if score-passing programs never fail invariants, there is nothing to certify. The generic MMD baseline tests whether one distribution-distance proxy can substitute for method-native predicates; if MMD-fail and PICR-fail sets coincide, the per-method machinery is unjustified. The random-invariant baseline calibrates the base rate of spurious failures, so any PICR advantage is not an artifact of emitting more predicates. The π_B-rollout baseline tests whether the probe-policy collection step is load-bearing: if PICR with π_B rollouts matches PICR with π_m rollouts, the adversarial alert's concern is empirically moot for these benchmarks; if not, the probe-policy design is justified. The collapsed-tag ablation tests whether the four invariant modules are individually load-bearing or interchangeable. The cross-method transfer ablation tests whether invariant geometry is method-specific. The calibration sweep documents the operating region where pass/fail decisions are stable. The primary metric — invariant-fail rate conditioned on score-passing — is the right detector because it measures exactly the gap the idea claims exists: programs that look good to the scorer but are unsafe, unaligned, unidentifiable, or miscalibrated downstream. The secondary downstream-degradation metric tests the missing equivalence: that a PICR failure predicts a method-specific performance loss, not merely a proxy violation.

### Expected outcome and causal chain

**vs. Real2SimLab native scorer only** — On a case where a coding agent writes a simulation program that reproduces benchmark reward by exploiting a scorer loophole (e.g. dynamics that satisfy the reward while admitting unsafe transitions), the baseline produces a high score with no signal of the violation because its scorer aggregates reward into one number and never executes the probe policy against safety predicates. Our method instead runs N=200 rollouts under π_m and evaluates the SPiDR pessimistic-critic containment check, so we expect a clear divergence: near-parity on benchmark score but a noticeable PICR-fail rate concentrated on programs whose reward comes from dynamics shortcuts. The downstream-degradation experiment should show that these same programs yield lower SPiDR success/safety when actually run, giving precision/recall above the native-scorer baseline (which has recall 0 by construction).

**vs. Generic MMD distribution-distance certificate** — On a case where a simulator is OT-aligned in observation marginals yet violates SPiDR's safety set or DEAL's parameter identifiability, the baseline produces a pass because a single marginal-distance proxy is blind to joint (obs, act) structure and to Fisher-rank collapse. Our method instead applies the method-native predicate — joint unbalanced-OT coupling for co-training, discriminator AUC plus permutation-test p-value plus Fisher rank for DEAL — so we expect the MMD certificate to pass a subset of programs that PICR fails, with the gap concentrated exactly on programs whose marginal statistics look right but whose transition or identifiability structure is wrong.

**vs. Random-invariant certificate** — On the full benchmark suite, the baseline produces a roughly uniform fail rate independent of program quality because its predicates carry no information about executed dynamics. Our method instead emits predicates whose truth values track the specific failure geometry each downstream method cares about, so we expect PICR's fail set to be strongly enriched on programs that later degrade under the corresponding downstream method, while the random certificate's fail set shows no such enrichment — a separation in fail-set precision rather than raw fail count.

**vs. PICR with π_B rollouts (no probe policy)** — On programs where the benchmark policy π_B visits a different state-action region than the downstream method's probe π_m, the π_B-rollout baseline produces a pass/fail that reflects π_B's coverage rather than π_m's. Our method instead collects under π_m and reports support-overlap ESS; we expect the two to disagree precisely on programs with low ESS (<0.1), and the downstream-degradation experiment to show that π_m-based PICR predicts method-specific degradation better than π_B-based PICR on those programs.

**vs. Cross-method invariant transfer (ablation)** — On programs certified under tag m but evaluated downstream under m' ≠ m, we expect method-native predicates to predict m-specific degradation better than cross-tag predicates (higher precision/recall for the matched tag), establishing that the invariant geometry is method-specific rather than a generic quality signal.

**vs. Calibration sweep (ablation)** — We expect pass/fail decisions to be stable across α ∈ {0.01, 0.1, 0.5}, τ ∈ {0.02, 0.05, 0.1}, ensemble size ∈ {3, 5, 10}, and N ∈ {100, 200, 400}, with variance of the invariant value decreasing as 1/√N; we will report the documented region where decisions do not flip.

### What would falsify this idea

If PICR's invariant-fail rate among score-passing programs is statistically indistinguishable from the random-invariant baseline, or if the generic MMD certificate matches PICR's fail set across all four method tags, then method-native predicates over probe-policy rollout traces add no detection power beyond existing scoring and the central claim is wrong. A second falsifier: if PICR failures are uniform across method tags rather than concentrated where each tag's predicted failure geometry occurs, or if PICR with π_B rollouts matches PICR with π_m rollouts on low-ESS programs, the per-method probe-policy design is not measuring what it claims.

## References

1. Real2SimLab: A Unified Platform for Real-to-Simulation Transfer
