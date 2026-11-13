# Belief-State Adjudication for Agent Recovery under Ambiguous Tool Outcomes

## Motivation

Existing recovery frameworks such as Evidence-Guided Recovery for AI Agents with Ambiguous Tool Outcomes and Partial Execution (EGR) pair observation-contingent planning with an effect-history adjudicator that requires tool outcomes to be resolvable to binary success/failure tokens. This structural assumption makes latent partial-execution states unidentifiable whenever public evidence (e.g., a timeout, a 500 response, a truncated log) is compatible with multiple execution outcomes, so the adjudicator must either discard ambiguous evidence or commit to a spurious resolution. The same residual gap recurs across ReAct-style replanning and Reflexion-style self-correction, confirming that no branch introduced a probabilistic observation model that maps ambiguous outputs to a distribution over execution states before adjudication.

## Key Insight

A factored belief-state estimator over latent execution states, whose likelihood factors are tied to the same admissibility predicates the adjudicator already enforces, makes recovery decisions identifiable under ambiguous evidence because the admissibility check acts as a hard constraint that zeroes out observationally-equivalent but inadmissible states, leaving a posterior whose support is exactly the set of states the adjudicator can safely act on.

## Method

**What it is.** We introduce BARCA (Belief-Adjudicated Recovery with Constrained Admissibility), a recovery controller that replaces the effect-history adjudicator's discrete evidence tokens with a factored belief state over latent partial-execution variables. Inputs: the current obligation set, provider capability list, public evidence stream (tool outputs, timestamps, log fragments), and the prior belief from the previous step. Outputs: a posterior belief over latent execution states and a recovery action selected by maximizing expected admissibility under that posterior.

**Load-bearing assumption (stated explicitly).** BARCA assumes the per-provider likelihood table P(e|f) is a calibrated observation model: the conditional probabilities in the table match the true empirical frequency of each evidence item e given each latent factor f, so that the constrained posterior concentrates on the true execution state as evidence accumulates. This assumption is load-bearing because the posterior mass on goal-satisfied states, the expected-admissibility score, and the terminal certificate all inherit its correctness; when it fails, the posterior concentrates on the KL projection of the true distribution onto the model family rather than on the true state (Berk 1966; White 1982). To make the assumption verifiable rather than asserted, BARCA maintains online-calibrated likelihood estimates and gates the terminal certificate on a calibration check, as specified below.

**How it works.**

```pseudocode
# Latent state: tuple (obligation_status, key_valid, auth_time_ok, deps_met, residual_effects)
# Each factor has a small discrete domain; factors are coupled only through shared evidence.
# Per-factor domains: obligation_status in {unmet, partially_met, met} (3);
#   key_valid in {valid, expired, unknown} (3); auth_time_ok in {ok, violated, unknown} (3);
#   deps_met in {met, violated, unknown} (3); residual_effects in {none, partial, full} (3).
# Joint state space per task family = 3^5 = 243 states (enumerated exactly; see complexity note).

function BARCA(prior_belief, evidence, obligations, capabilities, likelihood_counts, calibration_log):
    # 1. Online-calibrated observation model: for each evidence item e and each factor f,
    #    maintain a Dirichlet posterior over P(e|f) initialized from the pre-registered
    #    likelihood table (pseudocount strength alpha0=10 per cell) and updated from
    #    observed (e, f) pairs logged during execution. The point estimate used for
    #    inference is the Dirichlet posterior mean.
    likelihoods = {}
    for f in factors:
        likelihoods[f] = [dirichlet_mean(likelihood_counts[f][e], alpha0) for e in evidence]

    # 2. Constrained posterior update: multiply prior by likelihoods, then
    #    apply a large-but-finite penalty (lambda=1e-6) to any joint assignment
    #    violating an admissibility predicate (key expiry, authorization timing,
    #    dependency ordering, residual effects). The penalty is finite so that a
    #    mis-specified predicate cannot silently zero out the true state.
    unnormalized = {}
    for state in enumerate_joint_states(factors):
        if not admissible(state, obligations):
            unnormalized[state] = lambda_penalty * prior_belief[state] * prod(likelihoods[f][e] for f, e in zip(factors, evidence))
        else:
            unnormalized[state] = prior_belief[state] * prod(likelihoods[f][e] for f, e in zip(factors, evidence))
    posterior = normalize(unnormalized)

    # 3. Expected-admissibility action selection: for each candidate recovery action a,
    #    compute E_{s~posterior}[admissible_after(a, s)] and pick the argmax;
    #    tie-break by minimal residual-effect count.
    scores = {}
    for a in candidate_actions(obligations, capabilities):
        scores[a] = sum(posterior[s] * admissible_after(a, s) for s in posterior)
    best_action = argmax(scores, tie_break=min_residual_effects)

    # 4. Terminal certificate: goal is certified iff (a) posterior mass on
    #    goal-satisfied states exceeds threshold tau=0.95, (b) all such states
    #    pass the full admissibility check, and (c) the calibration check passes:
    #    the held-out posterior predictive coverage of the likelihood table is
    #    within [0.90, 0.98] over the last W=50 logged evidence items. If the
    #    calibration check fails, the certificate is withheld and the controller
    #    triggers conservative recovery (re-issue the last safe action).
    goal_mass = sum(posterior[s] for s in goal_states(posterior))
    all_admissible = all(admissible(s, obligations) for s in goal_states(posterior))
    calibration_ok = (0.90 <= posterior_predictive_coverage(calibration_log[-50:]) <= 0.98)
    certified = (goal_mass >= 0.95) and all_admissible and calibration_ok

    # 5. Fallback: if no admissible state has non-negligible mass (max posterior
    #    over admissible states < 0.01), do not certify; return conservative action.
    if max(posterior[s] for s in admissible_states(obligations)) < 0.01:
        certified = False
        best_action = conservative_recovery(obligations, capabilities)

    return posterior, best_action, certified
```

**Complexity note.** The joint state space is 3^5 = 243 states per task family, enumerated exactly in all 19 families; no mean-field approximation is needed. Per-step cost is O(243 * |evidence|) for the posterior update and O(|candidate_actions| * 243) for action selection, both tractable on CPU.

**Why this design.** We chose a factored likelihood table over a learned neural observation model because the provider capability metadata already encodes the relevant conditional probabilities (e.g., a timeout is more likely under non-execution than execution), and a learned model would require training data that the finite-study regime does not provide; the cost is that the likelihood table must be hand-specified per provider, which limits portability across new tool APIs. We chose to zero out inadmissible states during the posterior update rather than penalizing them softly because admissibility predicates are hard safety constraints (key expiry, authorization timing) that must never be violated, accepting that a single mis-specified predicate can eliminate the true state and force a conservative recovery. We chose expected-admissibility action selection over maximum-likelihood state selection because the adjudicator's goal is safe recovery, not state identification, and the expectation marginalizes over residual ambiguity; the cost is that the argmax can be dominated by a high-prior admissible action even when the posterior strongly favors an inadmissible one, which we mitigate with the terminal certificate threshold. A domain expert would not describe BARCA as a variant of EGR's adjudicator because EGR adjudicates discrete evidence tokens directly, whereas BARCA's posterior is constrained by the same admissibility predicates that EGR applies only after adjudication, so the constraint and the inference are coupled rather than sequential.

**Why this measures what we claim.** The posterior mass on goal-satisfied states measures the probability that the goal is actually satisfied under public evidence because the likelihood table is assumed to be a calibrated observation model for each provider capability, i.e., P(e|f) matches the empirical frequency of evidence e given latent factor f; this assumption fails when a provider changes its error semantics without updating the capability metadata, in which case the posterior reflects the stale likelihood table rather than the true execution state. The expected-admissibility score measures the safety of a recovery action because admissible_after(a, s) is assumed to be a deterministic predicate over the true latent state s, so the expectation is exact under the posterior; this assumption fails when the action's effect depends on unmodeled residual effects, in which case the score reflects the modeled admissibility rather than the true post-action safety. The terminal certificate threshold tau measures the confidence that the goal is truly satisfied because the posterior is assumed to concentrate on the true state as evidence accumulates (a martingale property of the constrained Bayesian update); this assumption fails when evidence is systematically ambiguous across all steps, in which case the certificate reflects the prior's bias rather than evidence-driven confidence. The calibration check measures whether the likelihood table remains calibrated because posterior predictive coverage is assumed to track the empirical frequency of held-out evidence; this assumption fails when the logged evidence distribution shifts, in which case the check withholds the certificate rather than reporting false confidence.

## Contribution

(1) A factored belief-state estimator over latent partial-execution variables whose likelihood factors are tied to provider capability metadata and whose posterior update is constrained by the same admissibility predicates used at recovery time, replacing discrete evidence-token adjudication with expected-admissibility action selection. (2) A terminal certificate that separates true goal satisfaction from safe public completion by requiring both posterior mass above a threshold and admissibility of all goal-satisfied states, providing a principled criterion for when recovery can stop under residual ambiguity. (3) A design principle for recovery controllers: coupling the observation model's support to the admissibility constraint makes latent execution states identifiable under ambiguous evidence, whereas sequential adjudication-then-planning does not.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Assignment-held-out partial-execution suite (58 paired units, 19 task families) | Paired units isolate recovery effect from task variation |
| Dataset (portability) | Second provider family with different error semantics (e.g., provider B with 429/503 instead of timeout/500), same 58-unit structure | Tests whether the likelihood table is portable across provider families |
| Primary metric | Safe-recovery rate (certified goal ∧ no admissibility violation) | Jointly captures safety and success, the core claim |
| Secondary metric | Calibration divergence: KL(hand-specified table || empirical frequencies) per provider capability | Grounds the calibration assumption empirically |
| Secondary metric | Posterior predictive coverage over held-out evidence | Verifies the calibration check used in the terminal certificate |
| Baseline | Evidence prompting | Tests whether constrained inference beats raw evidence reasoning |
| Baseline | EGR discrete-token adjudicator | Tests whether factored belief coupling beats sequential adjudication |
| Baseline | Unconstrained Bayesian update (no admissibility penalty) | Tests whether constraint-in-inference matters vs. post-hoc filtering |
| Ablation-of-ours | BARCA with soft admissibility penalties (lambda=1e-2 instead of 1e-6) | Tests hard-zeroing vs. soft penalty on the same posterior |
| Ablation-of-ours | BARCA with fixed hand-specified table (no online calibration) | Tests whether online calibration drives the gain |
| Ablation-of-ours | BARCA without calibration-gated certificate | Tests whether the calibration check prevents false certification |

### Why this setup validates the claim

The central claim is that coupling admissibility constraints into the posterior update, rather than applying them after adjudication, yields safer recoveries under ambiguous tool outcomes. The assignment-held-out suite is the right substrate because its paired units share task structure but vary in execution ambiguity, so differences in safe-recovery rate trace to the recovery mechanism rather than task difficulty. Evidence prompting tests the sub-claim that structured inference beats unstructured reasoning over the same public evidence. The EGR discrete-token adjudicator tests the sub-claim that a factored belief over latent variables outperforms sequential token adjudication when evidence is ambiguous. The unconstrained-update baseline isolates whether penalizing inadmissible states during inference—rather than filtering afterward—actually changes outcomes, which is the mechanism-specific claim. The soft-penalty ablation isolates hard constraints from soft ones on an otherwise identical posterior, so any gap is attributable to the constraint coupling itself. The fixed-table and no-calibration-gate ablations isolate the contribution of online calibration and the calibration-gated certificate to safe-recovery rate, directly testing the load-bearing assumption. The second provider family tests portability of the likelihood table. Safe-recovery rate is the correct metric because it is conjunctive: a method that certifies often but violates admissibility scores no better than one that recovers rarely, matching the design's safety-first objective.

### Expected outcome and causal chain

**vs. Evidence prompting** — On a case where a tool times out after partially executing a key-mutating step, prompting produces a recovery that re-issues the mutation because it reasons over the raw timeout text without tracking the latent key-validity factor. Our method instead penalizes key-expired states during the posterior update and selects the action maximizing expected admissibility, so the re-mutation is never a candidate. We expect a concentrated gap on timeout-after-partial-mutation units, with near-parity on clean-execution units.

**vs. EGR discrete-token adjudicator** — On a case where evidence is jointly ambiguous—a log fragment suggesting execution but a timestamp suggesting non-execution—the discrete-token adjudicator commits to one token reading and applies admissibility afterward, so it can certify a state that the timing predicate would have excluded. Our method instead couples the timing predicate into the joint posterior, eliminating that state before action selection. We expect lower false-certification on ambiguous-evidence units and parity elsewhere.

**vs. Unconstrained Bayesian update** — On a case where dependency ordering is violated but evidence is otherwise consistent, the unconstrained update puts high posterior mass on the violating state and picks an action admissible under it, only failing the post-hoc check. Our method instead penalizes the violating state in the posterior, so the action reflects only feasible states. We expect fewer admissibility violations on dependency-violation units, with similar certification rates.

**vs. BARCA with fixed hand-specified table** — On units where the provider's error semantics drift from the pre-registered table (e.g., a timeout becomes more likely under execution), the fixed-table variant mis-weights evidence and can certify a wrong state. Our online-calibrated variant updates the Dirichlet posterior from logged (e, f) pairs, so the likelihoods track the observed frequencies. We expect higher safe-recovery rate on drift units, with parity on stationary units.

**vs. BARCA without calibration-gated certificate** — On units where the likelihood table is mis-specified, the no-gate variant reports high posterior mass on a wrong state and certifies it, producing a silent false certification. Our calibration-gated variant withholds the certificate when posterior predictive coverage falls outside [0.90, 0.98], triggering conservative recovery. We expect fewer false certifications on mis-specified-table units, at the cost of lower certification rate on those units.

**Residual-effect leakage experiment** — On units with known residual-effect leakage (where admissible_after(a, s) diverges from true post-action safety because unmodeled residual effects are present), we expect the expected-admissibility score to overestimate safety, producing admissibility violations that the modeled predicate does not catch. This tests the named assumption A and failure mode F; we report the violation rate on these units separately.

### What would falsify this idea

If the safe-recovery gain over the EGR adjudicator is uniform across all units rather than concentrated on ambiguous-evidence and partial-mutation subsets where the constraint-coupling mechanism should bite, the central claim that coupling drives the improvement is wrong. Likewise, if the soft-penalty ablation matches hard-zeroing exactly, the hard-constraint-in-inference mechanism is not doing the work we attribute to it. If the online-calibrated variant does not outperform the fixed-table variant on drift units, the calibration repair does not address the load-bearing assumption.

## References

1. Evidence-Guided Recovery for AI Agents with Ambiguous Tool Outcomes and Partial Execution
