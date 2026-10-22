# Contract-Compiled Skills: Verifying Conditional Executability of Authored Agent Skills via Adversarial Contract Probing

## Motivation

Socratic Skill Learning (SSL) reports 0.801 utility on SkillTransferBench, but its central residual gap is that a successful task attempt leaves a Skill's conditions of use untested: the Student authors a Skill package from one trajectory and the held-out evaluation only checks whether the task succeeds, never whether the Skill's guards would have fired correctly under boundary conditions. This is a structural failure, not a data-scale failure — task-level success is a sufficient statistic for 'this run worked' but carries zero information about preconditions, postconditions, or what to do when a guard fails, so the same Skill can score high utility while being silently inapplicable outside the sampled trajectory. The same gap recurs in the static-descriptor branch of skill authoring, confirming the missing object is a formal execution contract, not more teaching data.

## Key Insight

Compiling each authored Skill into a conditional program with explicit guards and substitution clauses turns the untested conditions of use into obligations that an adversarial probe can falsify independently of task success, because a guard's truth value is observable at the moment of invocation while task-level utility is not.

## Method

**What it is.** Contract-Compiled Skill Learning (CCSL) takes as input an SSL-authored Skill package and outputs a *compiled contract skill*: a conditional program with an explicit precondition set, postcondition set, and a failure-recovery substitution table, plus a per-Skill contract-satisfaction vector. CCSL is a wrapper around the existing SSL interleaved loop; it does not change who operates the environment (the Student still does).

**Load-bearing assumption (stated explicitly).** CCSL assumes an LLM can compile an authored prose Skill into executable predicate functions over a fixed observation vocabulary whose truth values are semantically faithful to the Skill's intended conditions of use — i.e., that `guard_ok`/`post_ok`/`subst_ok` measure conditional executability rather than the compiler's own arbitrary predicate choices. This assumption is load-bearing because the satisfaction score is computed from these predicates; if they are tautological, ungrounded, or revised to fit probes, the score reflects predicate engineering rather than Skill quality. We make this assumption explicit and add a calibration gate (Phase 2b) that must pass before any predicate enters the satisfaction score.

**How it works.**

```pseudocode
def CCSL(family, budget_T=6, budget_P=4, k_probe=5, max_subst=2,
         calib_set=64, calib_thresh=0.9):
    # Phase 1: SSL teaching/practice as in baseline
    pkg = SSL_loop(family, rounds=budget_T)

    # Phase 2: compile authored prose Skill -> conditional program
    contract = compile(pkg, schema={
        pre:  list of predicates over (task_spec, env_state),
        post: list of predicates over (final_state, artifacts),
        subst: list of (guard_id -> recovery_procedure, expected_repair)})
    # compile is an LLM call constrained to emit executable predicate
    # functions over a fixed observation vocabulary; unparseable predicates
    # are rejected and re-emitted (max 3 retries), never silently dropped.

    # Phase 2b: predicate-calibration gate (NEW, addresses load-bearing assumption)
    # A separate verifier module (NOT the Student) authors and freezes the
    # postcondition predicates before probing. Each compiled predicate is
    # validated on a held-out calibration set of calib_set=64 invocation
    # states labeled by the environment (human- or simulator-provided
    # ground-truth pre/post labels). A predicate is accepted only if its
    # agreement with the labeled states is >= calib_thresh=0.9 AND it is
    # non-tautological (i.e., it evaluates False on at least 10% of the
    # calibration states). Rejected predicates are re-emitted (max 3 retries)
    # or the Skill is flagged as uncompilable and excluded from scoring.
    contract = calibrate(contract, calib_set, calib_thresh)

    # Phase 3: adversarial contract probing, interleaved with practice
    for probe in range(k_probe):
        # 3a. Educator proposes a boundary case or forced failure
        case = Educator.propose_boundary(contract, history)
        # 3b. Student executes the Skill on the case
        trace = Student.execute(contract, case)
        # 3c. Evaluate each contract clause on the trace
        guard_ok  = all(p(trace.invoke_state) == case.expected_pre for p in contract.pre)
        post_ok   = all(q(trace.final_state) == case.expected_post for q in contract.post)
        # 3d. If a guard was violated, force the substitution path
        if not guard_ok and contract.subst:
            sub = contract.subst[case.violated_guard]
            trace2 = Student.execute(sub.recovery, case)
            subst_ok = (sub.expected_repair(trace2) and post_ok_on(trace2, contract.post))
        else:
            subst_ok = None
        record(probe, guard_ok, post_ok, subst_ok,
               label_vocab_provenance=case.label_source)  # logged per probe
        # 3e. Student revises contract (not the prose Skill) on failure
        if not (guard_ok and post_ok):
            contract = Student.revise_contract(contract, probe, trace)

    # Phase 4: report contract-satisfaction, not task utility
    return contract, satisfaction(record, max_subst=max_subst)
```

Hyperparameters: `budget_T=6` SSL teaching rounds (matching the Fixed-Window SSL window), `budget_P=4` practice rounds, `k_probe=5` adversarial probes per Skill, `max_subst=2` substitution clauses per guard, `calib_set=64` held-out labeled invocation states for the calibration gate, `calib_thresh=0.9` minimum agreement for predicate acceptance. Contract-satisfaction score is the fraction of probes on which all three of {precondition match, postcondition match, substitution-preserves-postcondition} hold.

**Why this design.** We chose to compile Skills into an *executable predicate program* rather than keep them as natural-language descriptors because a descriptor cannot be falsified by a probe — a predicate can — accepting the cost that compilation may reject Skills whose prose is too vague to ground in the observation vocabulary. We chose to have the *Educator* generate boundary cases instead of sampling them uniformly from the task distribution because the whole failure mode is that boundary conditions are rare in the teaching trajectory, accepting the cost that probe quality is bounded by Educator competence (the same dependency SSL already has). We chose to make the *Student* the contract reviser rather than a separate verifier module because splitting authoring from verification would recreate the two-module controller anti-pattern and make the controller the hard problem, accepting the cost that a Student can revise a guard to fit a probe rather than fix the Skill — which is exactly why substitution clauses are scored by whether they preserve the *postcondition*, not by whether the Student claims they do, and why the calibration gate freezes postconditions before probing. We chose to report contract satisfaction *alongside* utility rather than replace utility, because a contract that is satisfied but useless is a real failure mode we want visible. A domain expert would not describe this as a variant of Trace2Skill or SKILL-KD distillation: those compress a trajectory into a reusable artifact, whereas CCSL adds an artifact (the contract) whose truth value is evaluated by intervening on the environment at invocation time — distillation has no analogue of a guard being *false*.

**Why it measures what we claim.** The contract-satisfaction score is claimed to measure *conditional executability* — whether the Skill is applicable exactly when its stated conditions hold. `guard_ok` measures precondition truth because we assume the probe's `expected_pre` label is itself derived from the same observation vocabulary the guard reads; this assumption fails when the Educator labels a boundary case using information the guard cannot observe, in which case `guard_ok` reflects label-observation mismatch rather than a genuine guard failure — we log a `label_vocab_provenance` field per probe to expose this. `post_ok` measures postcondition preservation because we assume the postcondition predicates are checked on the final state *after* the substitution path, not on the original trace; this fails when a substitution repairs the state but the postcondition predicate was written to accept the repaired state tautologically, in which case `post_ok` reflects predicate looseness — the calibration gate rejects tautological predicates (agreement < 0.9 or no False on ≥10% of calibration states) before they enter the score. `subst_ok` measures failure-recovery adequacy because we assume the forced-failure case genuinely violates the guard (the Educator's `violated_guard` label is checked against `guard_ok=False` before scoring); this fails when the forced failure is out-of-distribution in a way no substitution could repair, in which case `subst_ok` reflects probe infeasibility rather than recovery quality. `satisfaction` aggregates these three and therefore measures conditional executability only under the joint assumption that probes are in-vocabulary, postconditions are non-tautological, and forced failures are repairable — all three of which we log separately so a reader can see which assumption is doing the work.

**(D) Missing equivalences and failure modes.**
- (D1) `guard_ok` measures precondition truth; assumption A1: the Educator's `expected_pre` label is generated by the same observation vocabulary the guard reads (not by privileged simulator state); failure mode F1: the Educator labels using unobservable state, so `guard_ok` reflects label-observation mismatch. We log `label_vocab_provenance` per probe to detect F1.
- (D2) `satisfaction` measures conditional executability; assumption A2: the probe set is drawn from the same distribution as deployment-time invocations (in-vocabulary and repairable); failure mode F2: probes are out-of-distribution, so satisfaction reflects probe feasibility rather than executability. We test A2 with a held-out deployment-trace evaluation (see experiment).

## Contribution

(1) A contract-compilation layer that turns SSL-authored prose Skills into conditional programs with explicit preconditions, postconditions, and failure-recovery substitution clauses, together with a contract-satisfaction metric that is computed from adversarial probes rather than from task success. (2) The design finding that scoring substitution clauses by postcondition preservation (rather than by Student-reported repair) is what prevents contract revision from degenerating into fitting guards to probes. (3) An adversarial boundary-case and forced-failure probe protocol, interleaved with the existing SSL teaching/practice loop, that makes the untested conditions of use of an authored Skill observable.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | SkillTransferBench 22-family cohort (155 cases) | Held-out fresh-solver eval exposes conditional applicability |
| Primary metric | Contract-satisfaction score (with per-clause guard_ok, post_ok, subst_ok logged separately) | Directly falsifiable guard/post/subst truth, not utility proxy |
| Baseline | Adapted Trace2Skill distillation | Tests whether compression alone yields executable conditions |
| Baseline | Adapted SKILL-KD contrastive distillation | Tests whether contrastive artifact captures guards |
| Baseline | Fixed-Window SSL | Tests whether probing adds anything beyond teaching |
| Ablation-of-ours | CCSL with Educator boundary probes replaced by uniform sampling | Isolates adversarial probing as the causal driver |
| Ablation-of-ours | CCSL with calibration gate disabled | Tests whether the calibration gate is necessary to prevent tautological predicates from inflating satisfaction |
| Ablation-of-ours | CCSL with descriptor-only variant (natural-language guards, no executable predicates) | Tests whether contract compilation is the irreducible novelty |
| Cross-family transfer | Compile contract on family A, evaluate guards on family B's boundary cases | Tests whether contract-satisfaction is a reusable evaluation object |
| Probe-distribution shift | Evaluate contracts on held-out deployment traces (not Educator-proposed probes) | Tests assumption A2 (probes in-vocabulary and repairable) |

### Why this setup validates the claim

The central claim is that compiling prose Skills into executable predicate contracts, then adversarially probing them, produces *conditional executability* that trajectory-compression baselines cannot reach. Each baseline targets a distinct sub-claim. Trace2Skill and SKILL-KD test whether distillation artifacts — which compress trajectories into reusable descriptors — can be falsified at invocation time; they cannot, because a descriptor has no truth value to evaluate, so if they match CCSL on contract-satisfaction the compilation step is doing no work. Fixed-Window SSL tests whether the interleaved teaching loop alone already surfaces boundary conditions, isolating the probe phase as the delta. The uniform-sampling ablation is the load-bearing control: if CCSL's gain survives when Educator boundary proposals are swapped for random sampling, then the gain comes from compilation, not from adversarial probing, and the "why this design" argument collapses. The calibration-gate-disabled ablation tests the load-bearing assumption directly: if satisfaction is inflated without the gate, the metric is measuring predicate looseness rather than conditional executability. The descriptor-only variant tests the irreducible novelty: if a natural-language guard with oracle boundary cases cannot be falsified, the compilation step is the mechanism. The cross-family transfer tests whether contract-satisfaction is a reusable evaluation object rather than a per-family artifact. The probe-distribution shift tests assumption A2: if contracts evaluated on held-out deployment traces show different satisfaction than on Educator-proposed probes, the probe set is out-of-distribution and satisfaction reflects probe feasibility. Contract-satisfaction is the right metric because it is evaluated by intervening on the environment at invocation time — a guard being *false* is the observable that distillation has no analogue for — and it is logged per-clause (guard_ok, post_ok, subst_ok) so we can see which assumption carries the result rather than reading a single aggregate.

### Expected outcome and causal chain

**vs. Adapted Trace2Skill distillation** — On a case where the public variation spec permits a precondition the teaching trajectory never exercised (e.g., an object already in a target container), the baseline produces a plausible-sounding prose Skill that the Student applies unconditionally, because distillation compresses the trajectory into a descriptor with no representable guard to be false. Our method instead compiles that prose into a predicate over the observation vocabulary, and the Educator's boundary proposal surfaces the unexercised precondition as a guard failure, so we expect a visible gap on the boundary-case subset of the 155 cases and near-parity on the in-distribution subset — the gap should concentrate exactly where the trajectory was thin.

**vs. Adapted SKILL-KD contrastive distillation** — On a case where a substitution path is needed (the guard legitimately fails and recovery must preserve the postcondition), the baseline produces a contrastive artifact that names the failure mode but has no executable recovery table, so the Student improvises and postconditions drift. Our method instead scores `subst_ok` by whether the recovery procedure preserves the postcondition on the repaired trace, forcing the substitution table to be adequate rather than merely described. We expect the gap to appear specifically in the `subst_ok` component on forced-failure probes, with `guard_ok` roughly matched — a component-level signature, not a uniform lift.

**vs. Fixed-Window SSL** — On a case where the boundary condition is rare in the teaching trajectory, the baseline produces a Skill that transfers but silently misapplies at the boundary, because the interleaved loop never forces the Student to confront a guard violation. Our method instead spends `k_probe=5` adversarial probes per Skill and revises the contract on failure, so we expect CCSL to beat Fixed-Window SSL on contract-satisfaction while utility stays close — the predicted pattern is a metric divergence (satisfaction up, utility flat) rather than a uniform improvement.

**vs. CCSL with uniform-sampling ablation** — On a case where the boundary is a low-probability corner of the task distribution, uniform sampling rarely draws it within `k_probe=5`, so the ablation produces contracts with untested guards and inflated satisfaction on probes it happens to pass. Our method instead has the Educator propose cases conditioned on the contract's own clauses, so we expect the ablation to look competitive on aggregate satisfaction but to fail specifically on the boundary-concentrated subset — if the ablation matches CCSL everywhere, adversarial probing is not the mechanism.

**vs. CCSL with calibration gate disabled** — On a case where the compiled postcondition is tautological (e.g., "final state is valid"), the gate-disabled variant accepts it and `post_ok` is True regardless of Skill quality, inflating satisfaction. Our method rejects such predicates via the calibration gate (agreement < 0.9 or no False on ≥10% of calibration states), so we expect the gate-disabled variant to show higher satisfaction on tautology-prone Skills but lower agreement with held-out deployment traces — a signature that the gate is doing work.

**vs. CCSL descriptor-only variant** — On a case where the guard is expressed only in natural language, no probing protocol (even with oracle boundary cases) can falsify it because a descriptor has no truth value. Our method compiles the guard into an executable predicate, so we expect the descriptor-only variant to match CCSL on utility but fail to produce a falsifiable satisfaction score (or produce a score that cannot be checked against deployment traces), isolating compilation as the irreducible novelty.

**vs. Cross-family transfer** — On a case where a contract compiled on family A is evaluated on family B's boundary cases, we expect `guard_ok` to drop (preconditions differ across families) while `post_ok` and `subst_ok` may partially transfer, demonstrating that contract-satisfaction is a reusable evaluation object with component-level transfer signatures.

**vs. Probe-distribution shift** — On a case where contracts are evaluated on held-out deployment traces rather than Educator-proposed probes, we expect satisfaction to be lower if probes are out-of-distribution (assumption A2 fails) and comparable if probes are in-vocabulary. This tests whether the satisfaction metric reflects executability or probe feasibility.

### What would falsify this idea

If CCSL's contract-satisfaction gain over the distillation baselines is uniform across the boundary-case and in-distribution subsets — rather than concentrated on boundary cases and on the `subst_ok` component — then compilation and adversarial probing are not doing what the causal chain claims, and the gain is attributable to something else (e.g., extra LLM calls). Equivalently, if the uniform-sampling ablation matches CCSL on the boundary subset, the Educator's boundary proposals are not the driver and the central design choice is wrong. Additionally, if the calibration-gate-disabled variant matches CCSL on satisfaction and on held-out deployment traces, the load-bearing assumption (that predicates are semantically faithful) is not being protected by the gate, and the metric may be measuring predicate looseness rather than conditional executability.

## References

1. Socratic Skill Learning: Interleaving Teaching and Practice for Transferable Agent Skills
