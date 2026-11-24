# Drift-Closed On-Policy Distillation: Adaptive Support Expansion for Visual Agents

## Motivation

Trajectory-selection methods for visual agent distillation, such as the skill-context coverage selector of 'Distilling Visual Agents from Limited Interaction Experience' (E3), treat the selected trajectory pool as a static proxy for the full decision-context space and never re-check that proxy after the student begins optimizing. Because on-policy distillation moves the student's state-visitation distribution away from the recorded histories, the selected support can silently become incomplete: OPD then optimizes against trajectories that no longer cover the states the student actually reaches, so the reported recovery of the teacher gap (E3's 49.17%→82.08% on Navigation) is measured on a support that is not guaranteed to be closed under the student's own policy. The structural root cause is that coverage is a one-shot selection outcome rather than a dynamic, student-dependent property.

## Key Insight

Support closure is a fixpoint condition between the student's on-policy state-visitation distribution and the selected trajectory pool, so coverage must be re-established after every OPD update rather than certified once at selection time.

## Method

**What it is.** Drift-Closed OPD (DC-OPD) wraps the trajectory-selection + on-policy distillation loop of E3 with a closure monitor. Inputs: the recorded interaction pool $\mathcal{H}$, the teacher $\pi_T$, the student $\pi_\theta$, a latent encoder $\phi$. Outputs: an OPD-trained student plus a dynamically expanded support set $\mathcal{S}_t$ that is closed (within tolerance $\epsilon$) under $\pi_\theta$'s on-policy visitation.

**Load-bearing assumption (stated explicitly).** The monitor assumes that the frozen student visual encoder $\phi$ maps task-relevant out-of-support states (novel obstacle configurations the student's drifted policy actually visits) to low-density regions of the KDE $q_{ref}$ fitted on $\mathcal{S}_0$, so that the scalar drift statistics computed in $\phi$-latent space are a faithful proxy for genuine support incompleteness rather than encoder novelty or reference sparsity. This assumption is load-bearing: if $\phi$ conflates task-relevant novelty with task-irrelevant variation, the monitor will fire on the wrong states. We therefore (i) add a controlled diagnostic (Phase 0b) that perturbs task-irrelevant visual factors (lighting, texture) and task-relevant ones (obstacle layout) separately and reports $drift_t$ for each, and (ii) gate expansion on a decision-space closure signal (Phase 2) that does not rely solely on latent density.

**How it works.**

```pseudocode
Phase 0  INIT
  S_0 <- SelectTrajectories(H)          # E3 skill-context coverage + ordered visual-decision similarity
  phi   <- frozen student visual encoder; Z_ref <- phi(states in S_0)
  Fit density  q_ref(z) = KDE(Z_ref, bandwidth h)      # per-dim Silverman h
  Fit drift score  d(z) = -log q_ref(z)                # high d = outside current support

Phase 0b  DIAGNOSTIC (run once before OPD)
  # Controlled perturbation test: does d(z) separate task-relevant from task-irrelevant novelty?
  For each of N_diag=64 reference states s in S_0:
    s_irrel <- apply random lighting/texture perturbation to s   # task-irrelevant
    s_rel   <- render s with a novel obstacle layout             # task-relevant
    record d(phi(s_irrel)), d(phi(s_rel))
  Report AUC of d(z) separating {s_rel} from {s_irrel}.
  If AUC < 0.7: flag that latent density is confounded; proceed but report this in results.

Phase 1  OPD UPDATE (per iteration t)
  Roll out pi_theta on tasks in S_t -> on-policy states X_t, histories H_tau
  OPD loss: L = E[ KL( pi_T(.|h) || pi_theta(.|h) ) ]  over H_tau  (beta=1.0, no temperature)
  theta <- theta - eta * grad L                          # eta = 3e-5

Phase 2  CLOSURE MONITOR (after each update)
  Z_t <- phi(X_t)
  drift_t <- Quantile_{0.95}( d(z) for z in Z_t )       # 95th-pct latent drift
  mass_t  <- mean( 1[ d(z) > tau ] )  for z in Z_t       # tau = Quantile_{0.99}(d(Z_ref))
  # Decision-space closure signal (task-relevant, policy-coupled):
  kl_t    <- mean_{h in H_tau} KL( pi_T(.|h) || pi_theta(.|h) )   # teacher-student disagreement
  IF drift_t <= epsilon AND mass_t <= delta AND kl_t <= kappa:  (epsilon=0.5, delta=0.05, kappa=0.1)
      continue OPD
  ELSE:
      # targeted expansion: identify drift regions
      D_t <- cluster( {z in Z_t : d(z) > tau} )          # k-means, k=8
      FOR each cluster c in D_t:
          query pi_T on nearest recorded histories to c   # teacher-guided query
          OR fresh rollout pi_theta from c's centroid state (budget B=16 per cluster)
          append resulting trajectories to S_{t+1}
      Refit q_ref on S_{t+1}; recompute tau
      GOTO Phase 1
```

**Why this design.** We chose a *fixed* frozen-encoder latent density (KDE over $\phi$-states) over a jointly-trained drift classifier because a moving encoder would make drift non-comparable across iterations, accepting the cost that drift is measured in a representation that itself may not perfectly separate task-relevant states. We chose *quantile-thresholded* expansion ($\tau$ = 99th percentile of reference drift, $\epsilon$ on the 95th-percentile drift) over a fixed absolute drift cutoff because the latent scale is task-dependent, accepting the cost that the monitor inherits the KDE bandwidth's sensitivity. We chose *targeted expansion* (clustering drift states and querying the teacher or rolling out fresh) over uniform pool augmentation because uniform augmentation wastes the limited interaction budget that E3 explicitly optimizes, accepting the cost that clustering quality bounds expansion quality. Unlike the retrieval-style trajectory selector of E3 (which computes skill-context coverage once and never revisits it), DC-OPD's monitor is a *dynamic closure test* whose output is a decision to expand, not a similarity score — a domain expert would not describe this as a variant of E3's selector because E3 has no post-update feedback channel at all. Unlike a generic two-module controller, the monitor and the OPD update share the same surface: both operate on $\pi_\theta$'s on-policy states, and the monitor's output ($\mathcal{S}_{t+1}$) is the OPD update's input, so there is no separate router to train.

**Why this measures what we claim.** The quantity `drift_t = Quantile_{0.95}(d(z))` measures *support incompleteness* because the assumption is that the reference KDE $q_{ref}$ fitted on $\mathcal{S}_0$ approximates the student's on-policy visitation under a closed support; this assumption fails when the student's policy shifts to states the encoder maps to low-density regions that are nevertheless task-relevant (e.g., a novel obstacle configuration), in which case `drift_t` reflects *encoder novelty* rather than genuine coverage gap. The quantity `mass_t = mean(1[d(z)>tau])` measures the *fraction of on-policy visitation outside the selected support* because the assumption is that the 99th-percentile reference drift is a valid boundary of the selected support in latent space; this assumption fails when the reference set is itself sparse in a region the student later visits, in which case `mass_t` reflects *reference sparsity* rather than out-of-support visitation. The quantity `kl_t = mean KL(pi_T || pi_theta)` measures *decision-space closure* because the assumption is that teacher-student disagreement on the student's own histories is high exactly when the student visits states the support does not cover; this assumption fails when the teacher is itself uncertain on those states, in which case `kl_t` reflects *teacher noise* rather than coverage gap. The expansion trigger `(drift_t > epsilon OR mass_t > delta) AND kl_t > kappa` measures *closure violation* because the assumption is that a latent-density signal and a decision-space signal must jointly indicate incompleteness for the support to be genuinely incomplete; this assumption fails when drift is concentrated in a single high-mass cluster that the 95th-percentile misses, in which case the trigger reflects *tail insensitivity* and the monitor under-expands.

## Contribution

["A closure-monitoring mechanism (DC-OPD) that re-estimates the student's on-policy latent state-visitation distribution after each OPD update and triggers targeted support expansion, converting trajectory selection from a one-shot outcome into a dynamic, student-dependent property.", 'A drift-detection design principle combining tail (95th-percentile) and mass (fraction beyond 99th-percentile) statistics over a frozen-encoder KDE, with a quantile-adaptive threshold that avoids task-specific absolute cutoffs.', 'An expansion protocol that clusters drift states and routes each cluster to either teacher-guided queries or fresh student rollouts under a fixed per-cluster budget, preserving the limited-interaction constraint of the original distillation setting.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Navigation + Sokoban + spatial mapping panels (E3) | Matches the interaction-limited regime being tested |
| Dataset | Procedurally generated navigation suite with deliberately incomplete initial support | Tests closure beyond E3's fixed panels |
| Primary metric | Success rate on held-out panels | Directly reflects agent capability under limited interaction |
| Secondary metric | Cumulative interaction steps (env steps + teacher queries) | Makes efficiency claim falsifiable at matched budget |
| Baseline | E3 fixed-trajectory OPD | Tests whether dynamic closure beats static selection |
| Baseline | Matched SFT | Tests whether on-policy closure matters beyond imitation |
| Baseline | DAgger-style teacher query on student states (no KDE monitor) | Isolates the KDE monitor from generic on-policy querying |
| Baseline | 27B teacher reference | Upper bound anchoring the recoverable gap |
| Ablation-of-ours | DC-OPD with monitor disabled (no expansion) | Isolates the closure monitor's contribution |
| Ablation-of-ours | DC-OPD with latent-density signal only (no kl_t gate) | Tests whether decision-space closure adds beyond density |
| Ablation-of-ours | DC-OPD with kl_t gate only (no latent-density signal) | Tests whether density adds beyond decision-space closure |

### Why this setup validates the claim

The central claim is that a closure monitor converting static trajectory selection into a dynamic, feedback-driven support-expansion loop yields better on-policy distillation than E3's one-shot selector, especially when the student drifts into under-supported states. The E3 fixed-trajectory OPD baseline directly tests this: it shares the same teacher, student, and OPD loss, so any gain isolates the monitor. Matched SFT tests a competing explanation—that gains come merely from more supervised data rather than from on-policy closure—by holding data volume comparable. The DAgger-style baseline tests whether the KDE monitor adds anything beyond generic on-policy teacher querying. The 27B teacher anchors whether the student recovers a meaningful fraction of the measured gap. The ablation (monitor disabled, no expansion) is the sharpest test: if the monitor drives the effect, removing it should collapse performance back toward E3. The two signal ablations (density-only, kl-only) test whether the joint trigger is necessary. The procedurally generated suite with deliberately incomplete initial support tests whether closure generalizes beyond E3's panels. Success rate on held-out panels is the right metric because it directly measures transferable capability, which is exactly what support closure is supposed to protect; a metric like loss or drift magnitude would be a proxy that could improve without capability improving. The interaction-budget ledger and success-vs-cumulative-interaction curves make the efficiency claim falsifiable at matched cost.

### Expected outcome and causal chain

**vs. E3 fixed-trajectory OPD** — On a case where the student's on-policy rollouts drift into obstacle configurations absent from the four fixed trajectories, E3 produces stalled or degraded success because its selector computes coverage once and never revisits it, leaving those states unsupervised. Our method instead detects elevated 95th-percentile latent drift, out-of-support mass, and teacher-student KL, clusters the drift states, and queries the teacher or rolls out fresh from cluster centroids, so we expect a noticeable success-rate gap concentrated on panels whose layouts diverge from the initial selection, with near-parity on panels matching the initial support.

**vs. Matched SFT** — On a case where the student must recover from a state reached only by its own imperfect policy, matched SFT produces brittle behavior because it imitates recorded trajectories without ever correcting the student's own off-distribution states. Our method instead trains on the student's on-policy histories and expands support exactly where those histories leave the reference density, so we expect SFT to trail noticeably on recovery-heavy panels while our method closes most of that gap, with a smaller advantage on clean, in-distribution panels.

**vs. DAgger-style teacher query on student states (no KDE monitor)** — On a case where the student visits states that are task-relevant but already high-density under $q_{ref}$ (so the KDE monitor would not fire), DAgger queries the teacher on all student states and may over-query, wasting budget on redundant states. Our method instead gates expansion on the joint density-and-KL trigger, so we expect comparable success at lower cumulative interaction cost, with the advantage appearing as a leftward shift in the success-vs-interaction curve rather than a higher plateau.

**vs. DC-OPD with monitor disabled (ablation)** — On a case where the student's policy shifts mid-training toward a region the initial support set sparsely covers, the ablated variant produces a plateau because it has no post-update feedback channel to trigger expansion. Our method instead fires the closure trigger and adds targeted trajectories, so we expect the ablation to track E3 closely while full DC-OPD pulls ahead, with the divergence appearing only after the policy shift—evidence that the monitor, not the OPD loss, produces the gain.

**vs. DC-OPD with latent-density signal only (no kl_t gate)** — On a case where the frozen encoder $\phi$ maps a task-relevant novel obstacle configuration near in-support states (low $drift_t$), the density-only variant fails to fire and leaves those states unsupervised. Our method instead detects elevated teacher-student KL on those states and expands, so we expect the density-only variant to underperform on the diagnostic perturbation panels where $\phi$ conflates task-relevant and task-irrelevant novelty, with parity elsewhere.

**vs. DC-OPD with kl_t gate only (no latent-density signal)** — On a case where the teacher is itself uncertain on novel states (high teacher entropy), the KL-only variant may fire spuriously or fail to localize drift regions for clustering. Our method instead uses the latent-density signal to localize clusters for targeted expansion, so we expect the KL-only variant to expand less efficiently (more interaction for the same success) and to show noisier expansion decisions.

### What would falsify this idea

If DC-OPD's gain over E3 is uniform across panels rather than concentrated where the predicted drift-and-expansion failure mode occurs—or if the monitor-disabled ablation matches full DC-OPD, or if the DAgger-style baseline matches DC-OPD at equal interaction budget—then the closure monitor is not the causal driver and the central claim is wrong.

## References

1. Distilling Visual Agents from Limited Interaction Experience
