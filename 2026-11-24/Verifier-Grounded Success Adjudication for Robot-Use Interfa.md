# Verifier-Grounded Success Adjudication for Robot-Use Interface Benchmarks

## Motivation

Across the manipulation-benchmark lineage, success is adjudicated by the agent's own declaration rather than by an independent check of the simulator state, so declared success and observable task completion are not provably coupled under partial observability. RUIBench (E4) makes this structural problem explicit: it reports a verification gap in which agents declare success despite observable failures, and its 45%-to-81% interface-driven success swing is measured entirely through that self-declared signal. The same reliance on agent-reported outcomes recurs in benchmark papers such as ManiSkill-HAB (E2) and RoboTwin (E3), whose evaluations inherit whatever success criterion the agent or environment wrapper asserts; no branch of the lineage supplies an independent, simulator-state task-level verifier that re-adjudicates episodes across interfaces.

## Key Insight

Because the simulator exposes object poses and contact states that are causally upstream of task completion yet are never consulted by the agent's declaration, a predicate evaluated over those state variables yields a success label that is independent of the interface under test, so the declared-versus-verified gap becomes a measurable interface-invariant quantity rather than an unobservable assumption.

## Method

**What it is.** VISA (Verifier-grounded Interface Success Adjudication) takes as input a logged RUIBench episode (interface id, task id, per-step simulator state including object poses, gripper poses, contact pairs, and the agent's declared-success flag) and outputs a verified success label, a declared-minus-verified gap, and a per-suite/per-interface gap table. It is a re-auditing layer over RUIKit's 13 interfaces and 30 tasks, not a new agent.

**Load-bearing assumption (stated explicitly).** VISA assumes that for each task in the audited subset, the task's success condition is fully captured by a hand-specified conjunction of object-pose-in-region, required-contact-set, and no-forbidden-collision predicates evaluated over logged simulator state. This assumption is load-bearing: if it fails, `verified` reflects geometric reachability rather than true task success, and the gap table measures predicate-authoring error rather than interface-induced declaration inflation. To make the assumption testable rather than universal, VISA (i) restricts the interface-invariance audit to the subset of tasks whose goals are provably expressible as pose/contact conjunctions, (ii) reports excluded tasks explicitly as out of scope, (iii) calibrates tolerances per task from the simulator's own success-checker where one exists, and (iv) validates the predicate against human labels on a stratified sample.

**How it works.**

```pseudocode
INPUT: episode log L = {(s_t, a_t, declared_t)} for t=1..T,
       task spec tau (suite, goal predicate family, object set)
PARAM: pose_tol = 0.02 m (default; overridden per task by calibration),
       contact_force_min = 0.1 N (default; overridden per task by calibration),
       settle_steps = 5, goal_hold_steps = 3

PHASE 0 — Task expressibility screening (per task, once)
  for each task tau in the 30 RUIBench tasks:
      if tau.goal is expressible as a conjunction of
         (object pose in target region) AND
         (required contact set with force >= contact_force_min) AND
         (no forbidden collision pair active):
          mark tau as IN-SCOPE
      else:
          mark tau as OUT-OF-SCOPE and record the reason
          (e.g., semantic goal 'clean', 'correctly oriented for human use')
  # The interface-invariance audit (Phase 4) runs only on IN-SCOPE tasks.
  # OUT-OF-SCOPE tasks are reported in a coverage table and excluded from delta.

PHASE 1 — Goal-predicate instantiation and tolerance calibration (per in-scope task, once)
  for each IN-SCOPE task tau:
      if the simulator provides a native success-checker for tau:
          calibrate pose_tol_tau and contact_force_min_tau by grid search
          over {0.005, 0.01, 0.02, 0.05} m and {0.05, 0.1, 0.5} N to maximize
          agreement between P_tau and the native checker on a held-out set of
          50 logged episodes (20 declared-success, 30 declared-failure).
      else:
          use defaults pose_tol = 0.02 m, contact_force_min = 0.1 N.
      P_tau = compile_goal_predicate(tau, pose_tol_tau, contact_force_min_tau)
          # conjunction over
          # (object_i pose in target region within pose_tol_tau)
          # (contact state matches required grasp/contact set,
          #  force >= contact_force_min_tau)
          # (no forbidden collision pair active)
      store P_tau keyed by task id

PHASE 2 — State-only re-adjudication (per episode)
  verified = False
  hold = 0
  for t in 1..T:
      if P_tau(s_t) evaluates True:
          hold += 1
      else:
          hold = 0
      if hold >= goal_hold_steps and s_t stable over last settle_steps:
          verified = True; t_verified = t; break
  declared = any(declared_t == True for t in 1..T)

PHASE 3 — Gap accounting
  gap = declared AND NOT verified      # false-positive success
  miss = verified AND NOT declared     # false-negative success
  emit (task, interface, verified, declared, gap, miss, t_verified)

PHASE 4 — Interface-invariance audit (IN-SCOPE tasks only, matched task/seed pairing)
  # Assumption A (stated): all 13 interfaces are evaluated on the same IN-SCOPE
  # task subset with matched seeds and identical perception/control.
  # Failure mode F: unequal task sampling makes delta reflect difficulty
  # composition rather than interface behavior.
  # Enforcement: for each interface i, restrict to the intersection of tasks
  # and seeds evaluated by all 13 interfaces; report the size of this matched set.
  matched_tasks = intersection over i of tasks_evaluated[i]
  matched_seeds = intersection over i of seeds_evaluated[i]
  for each interface i in 13:
      gap_rate[i] = mean(gap) over matched_tasks x matched_seeds
      verified_rate[i] = mean(verified) over matched_tasks x matched_seeds
      declared_rate[i] = mean(declared) over matched_tasks x matched_seeds
  report delta[i] = declared_rate[i] - verified_rate[i]
  flag interface i as 'declaration-inflated' if delta[i] > 0.10
```

**Why this design.** We chose a hand-specified predicate over simulator state rather than a learned success classifier because the predicates are the only quantities that are provably independent of the interface under test, accepting the cost that predicate authoring requires per-task engineering and cannot cover tasks whose goals are not expressible as pose/contact conjunctions; Phase 0 makes this cost explicit by screening tasks and reporting out-of-scope ones. We chose a temporal hold requirement (`goal_hold_steps=3`, `settle_steps=5`) over a single-timestep check because a single frame can transiently satisfy a goal predicate during a failed trajectory, accepting the cost that some genuinely fast successes may be scored as failures. We chose to re-audit logged episodes rather than re-run the simulator because re-running would change the interface's behavior and confound the gap with rollout stochasticity, accepting the cost that we cannot verify states the logger did not record. We chose to report both false-positive (`gap`) and false-negative (`miss`) directions rather than only the false-positive gap, accepting extra bookkeeping, because an interface that suppresses declarations would otherwise look artificially good. We chose per-task tolerance calibration from the simulator's native success-checker where available (Phase 1) rather than a single fixed tolerance, accepting the cost that calibration requires a held-out episode set, because a fixed tolerance that is tight for a fine-grained interface is loose for a coarse one, so `verified` would not be interface-invariant by construction. A domain expert would not describe VISA as a variant of a learned reward model or a process reward model: those learn a scalar from trajectories and inherit the agent's own action distribution, whereas VISA evaluates a fixed predicate over privileged simulator state that the agent never observes, so its label is invariant to the interface by construction.

**Why it measures what we claim.** The quantity `verified` measures task completion because the goal predicate `P_tau` is a conjunction of object-pose-in-region, required-contact-set, and no-forbidden-collision conditions, under the assumption that the task's natural-language goal is fully captured by that conjunction; this assumption fails when a task's goal includes a semantic property (e.g., 'clean' or 'correctly oriented for human use') not expressible in pose/contact terms, in which case `verified` reflects geometric reachability rather than true task success. Phase 0 screens such tasks out of the invariance audit and reports them explicitly, so the assumption is conditional on the in-scope subset rather than universal. The quantity `gap = declared AND NOT verified` measures the verification gap because the agent's declaration is produced without access to the simulator state used by `P_tau`, under the assumption that the agent's policy and the verifier do not share the same state estimator; this assumption fails when the environment wrapper exposes ground-truth poses to the agent, in which case `gap` collapses to a measure of declaration timing rather than of misreporting. The quantity `delta = declared_rate - verified_rate` measures interface-induced declaration inflation because the 13 interfaces share perception and control and differ only in action abstraction and interaction form, under the assumption that any difference in `delta` across interfaces is attributable to the interface rather than to task sampling; this assumption fails when interfaces are evaluated on different task subsets, in which case `delta` reflects task difficulty composition rather than interface behavior. Phase 4 enforces matched task/seed pairing to make this assumption testable: the matched set size is reported, and delta is computed only on the intersection.

## Contribution

["VISA, an independent simulator-state task-level verifier and re-auditing layer that converts RUIBench's agent-declared success signal into a verified success label and a declared-minus-verified gap per episode, suite, and interface.", 'A design principle for manipulation benchmarks: success labels should be computed from privileged simulator state predicates that are invariant to the interface under test, and both false-positive and false-negative declaration directions should be reported.', "A per-interface 'declaration-inflated' flag derived from the gap table, enabling identification of which of RUIKit's 13 interfaces systematically over-report success."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | RUIBench: 30 tasks, 5 suites, 13 RUIKit interfaces; pilot on 8–10 tasks spanning all five suites for predicate authoring, then full in-scope subset for the audit | Directly the population VISA re-audits; pilot bounds feasibility and reports predicate-authoring time per task |
| Primary metric | Declared-minus-verified gap rate per interface (delta), computed on the matched task/seed intersection across all 13 interfaces | Isolates declaration inflation from raw success and from task-difficulty composition |
| Baseline | Agent's own declared-success flag | The unaudited signal VISA claims to correct |
| Baseline | Single-timestep predicate check (no hold/settle) | Tests whether temporal hold is load-bearing |
| Baseline | Learned trajectory reward model (2-layer MLP, hidden=256, GeLU, trained on logged (state, action) trajectories with declared-success as label) | Tests privileged-state independence claim |
| Ablation-of-ours | VISA without miss accounting (gap only) | Tests whether bidirectional reporting matters |
| Ablation-of-ours | VISA with fixed default tolerances (pose_tol=0.02 m, contact_force_min=0.1 N) instead of per-task calibration | Tests whether calibration is load-bearing for the invariance claim |
| Validity check | Human annotation of success/failure from video on a stratified sample of 100 episodes (20 per suite), reporting Cohen's kappa against P_tau | Tests the load-bearing assumption that P_tau captures task success |
| Coverage table | Per-task expressibility screening (IN-SCOPE vs OUT-OF-SCOPE with reason) | Makes the scope of the invariance claim explicit and bounded |

### Why this setup validates the claim

The central claim is that declared success inflates relative to interface-invariant, privileged-state-verified success, and that this inflation varies systematically across the 13 interfaces. The agent's own declaration baseline is the unaudited quantity; if VISA's verified label tracks it closely on every interface, the gap is negligible and the claim collapses. The single-timestep predicate baseline tests the temporal-hold sub-claim: if it produces the same gap pattern, the settle/hold machinery is unmotivated. The learned reward-model baseline tests the independence sub-claim: a learned scalar inherits the agent's action distribution, so if it reproduces VISA's gap table equally well, privileged-state predicate evaluation is unnecessary. The gap-only ablation tests whether false-negative accounting changes the interface ranking. The fixed-tolerance ablation tests whether per-task calibration is load-bearing for the invariance claim: if the fixed-tolerance variant produces the same delta table, calibration is unnecessary. The human-annotation validity check tests the load-bearing assumption directly: if Cohen's kappa between P_tau and human labels is low on in-scope tasks, the verified label is a geometric proxy rather than task success, and the gap table measures predicate error. The coverage table bounds the claim: delta is reported only on the in-scope subset, and out-of-scope tasks are named explicitly. The primary metric is right because delta is exactly the quantity the claim asserts is interface-induced; raw success rate cannot separate declaration behavior from task difficulty, whereas delta conditions on the same matched tasks, seeds, and perception/control.

### Expected outcome and causal chain

**vs. Agent's own declared-success flag** — On a case where an interface with a coarse action abstraction lets the agent terminate early after a near-miss grasp, the baseline records success because the agent declares it, since its declaration is produced without privileged state and cannot see the residual pose error. Our method instead evaluates the pose/contact conjunction over logged simulator state with a three-step hold, so it flags the episode as a gap. We expect a nonzero gap concentrated on coarse-abstraction interfaces, with near-zero gap on fine-grained ones, yielding a spread in the per-interface delta table rather than a uniform offset.

**vs. Single-timestep predicate check** — On a case where a trajectory transiently passes the goal region mid-motion before objects settle or topple, the baseline marks verified at that single frame because it has no temporal persistence requirement. Our method instead requires the predicate to hold for three consecutive steps after five settle steps, so transient satisfaction is rejected. We expect the single-timestep baseline to over-verify on dynamic and contact-rich suites, producing systematically lower gaps there, while the two agree on static placement tasks — a divergence concentrated in exactly the suites where motion is non-monotone.

**vs. Learned trajectory reward model** — On a case where the agent's policy visits states unlike the reward model's training distribution, the baseline mislabels because its scalar is fit to the agent's own action distribution and inherits its blind spots. Our method instead applies a fixed predicate over privileged state the agent never observes, so its label is invariant to which interface produced the trajectory. We expect the learned model's gap estimates to drift across interfaces in a way correlated with interface-specific state visitation, while VISA's gap table stays stable under interface swaps on identical tasks.

**vs. VISA without miss accounting (gap only)** — On a case where an interface suppresses declarations (agent fails to declare success despite verified completion), the gap-only ablation records no gap because it ignores false negatives, making the interface look artificially good. Our method instead reports both gap and miss, so the interface's delta is corrected by the miss rate. We expect the gap-only ablation to rank declaration-suppressing interfaces more favorably than the full VISA, changing the interface ordering on at least one suite.

**vs. VISA with fixed default tolerances** — On a case where a fine-grained interface achieves a near-exact placement that a coarse interface cannot, the fixed-tolerance variant may mark both as verified (if the tolerance is loose) or both as failed (if tight), flattening the delta table. Our method instead calibrates pose_tol_tau and contact_force_min_tau per task from the simulator's native success-checker where available, so the verified label tracks the task's own success criterion. We expect the fixed-tolerance variant to produce a delta table with lower spread across interfaces, and the calibrated variant to recover the interface-dependent spread, with the difference concentrated on tasks where the native checker exists and tolerances deviate from defaults.

**vs. Human annotation (validity check)** — On a stratified sample of 100 episodes, we report Cohen's kappa between P_tau and human success/failure labels. We expect kappa >= 0.7 on in-scope tasks (indicating P_tau captures task success) and kappa < 0.4 on out-of-scope tasks (confirming the need for Phase 0 screening). If kappa is low on in-scope tasks, the load-bearing assumption fails and the gap table is reinterpreted as a measure of predicate error rather than interface-induced inflation.

### What would falsify this idea

If the per-interface delta table is flat — gaps roughly equal across all 13 interfaces on the matched task/seed intersection — then declaration inflation is a property of the agent, not the interface, and VISA's interface-invariance audit adds nothing over a single global gap number. If the learned reward model reproduces VISA's gap ranking without privileged state, the privileged-state independence argument is unnecessary. If Cohen's kappa between P_tau and human labels is low (< 0.4) on in-scope tasks, the verified label is a geometric proxy rather than task success, and the gap table measures predicate error rather than interface behavior.

## References

1. RUIBench: What Makes a Good Robot-Use Interface?
