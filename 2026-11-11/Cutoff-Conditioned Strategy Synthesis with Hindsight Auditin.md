# Cutoff-Conditioned Strategy Synthesis with Hindsight Auditing for Simulated Financial Users

## Motivation

Open-world financial agent evaluations such as FINCOSMOS grade agents against a hidden user strategy that is revealed interactively, but they never establish that the strategy itself was temporally valid at each simulated decision time. Because FINCOSMOS samples strategies from a compositional grammar without a per-decision information cutoff, an agent's success cannot be attributed to valid temporal reasoning rather than to a user strategy that silently embeds hindsight (e.g., a rule referencing a drawdown regime only identifiable after the cutoff). The ancestor Time Machine GPT enforces a fixed historical cutoff only through prompt-level instruction, which cannot constrain a dynamically revealing simulated user whose strategy is instantiated across many decision times; the structural gap is that temporal validity is asserted, not constructed and audited.

## Key Insight

If every strategy clause is generated only from the information set that is provably available at its own decision time, then any clause that references post-cutoff information becomes a detectable violation of a per-clause provenance invariant rather than an unobservable property of the whole strategy, so temporal validity is enforced by construction and verified by audit instead of trusted.

## Method

**What it is.** We introduce CATCH (Cutoff-Audited Temporal Strategy Harness), a generator-plus-auditor that wraps FINCOSMOS's compositional strategy grammar. Inputs: a grammar, a decision-time schedule, and a per-time vintage-indexed information ledger of what data is publicly available. Outputs: a hidden strategy in which every clause carries a machine-checkable provenance tag, plus an audit report flagging any post-cutoff reference.

**Load-bearing assumption (stated explicitly).** CATCH's certificate is valid only if the ledger L[t] records the value of each field as it was observable at decision time t, not merely its nominal publication date. Concretely, we assume that for every field f in L[t], the value stored is the vintage-t value (i.e., the value that a point-in-time archive such as ALFRED or the FINCOSMOS suite's own as-of snapshots would have returned at t), and that no vendor backfill or revision has overwritten that value under an unchanged nominal publication date. If this assumption fails, the certificate is a nominal-availability certificate, not a temporal-validity certificate. We therefore report every certificate as conditional on ledger vintage integrity and include a ledger-integrity check (Phase 0) that verifies each field's vintage timestamp against the archive's revision history.

**How it works.**

```pseudocode
# Phase 0: vintage-ledger integrity check (new, makes the load-bearing assumption checked)
for each field f in ledger:
    vintages[f] = archive.vintage_history(f)   # list of (vintage_timestamp, value)
    if vintages[f] is empty or vintages[f][-1].vintage_timestamp > schedule_end:
        flag(f, reason="unverifiable_vintage")
        exclude f from L[t] for all t

# Phase 1: vintage-indexed information ledger construction
for each decision time t in schedule T:
    L[t] = { (record_id, vintage_timestamp, value) : vintage_timestamp <= t }
    # value is the vintage-t value, not the nominal-publication-date value
    L[t] = L[t] minus embargoed(records, lag=1 business day)

# Phase 2: cutoff-conditioned clause synthesis
for each decision time t in T (ascending):
    clause_set[t] = []
    for each grammar slot s in strategy_skeleton:
        candidates = grammar.expand(s, context=L[t])       # only L[t] visible
        c = select(candidates, seed=hash(s,t))              # deterministic, reproducible
        c.provenance = {field: (field_source(c), vintage_timestamp(field))}
        clause_set[t].append(c)

# Phase 3: hindsight audit (three independent checks)
for each clause c in clause_set[t]:
    # 3a. symbolic check: every field referenced must have a vintage_timestamp <= t in L[t]
    if any(field not in L[t] for field in c.provenance.field):
        flag(c, reason="post_cutoff_reference", evidence=field)
    # 3b. replay check: re-evaluate c on data truncated at t
    v_trunc = eval(c, truncate(data, t))
    v_full  = eval(c, data)
    if v_trunc != v_full:
        flag(c, reason="replay_divergence", delta=v_full-v_trunc)
    # 3c. sensitivity check: perturb only post-t fields and re-evaluate
    # (catches cancellation where post-t dependence is offset by pre-t dependence)
    for each post_t_field f in c.provenance.field with vintage_timestamp > t:
        data_perturbed = data with f replaced by a counterfactual value (e.g., f * (1+epsilon), epsilon=0.01)
        v_pert = eval(c, data_perturbed)
        if v_pert != v_full:
            flag(c, reason="post_t_sensitivity", field=f, delta=v_pert-v_full)

# Phase 4: strategy assembly with validity certificate
strategy = assemble(clause_set, tie_break=deterministic)
strategy.certificate = {t: audit_report[t] for t in T}
strategy.certificate.conditional_on = "ledger_vintage_integrity"
accept_strategy iff all audit_report[t] empty AND no unverifiable_vintage flags
```

**Why this design.** We chose per-clause provenance tagging over a single prompt-level cutoff instruction (the Time Machine GPT approach) because a prompt-level instruction cannot bind a dynamically revealing user whose clauses are instantiated at different times; the cost is that the grammar must be instrumented to emit field-level provenance, which restricts strategies to fields with resolvable vintage histories. We chose three independent audit checks (symbolic membership, truncated replay, and post-t sensitivity) over a single symbolic check because symbolic membership alone misses clauses that reference an aggregate whose constituents were revised after t, and truncated replay alone misses clauses whose post-t dependence is exactly offset by a pre-t dependence (cancellation); the cost is tripling audit compute. We chose deterministic seeded selection over stochastic sampling so that a run is reproducible and a flagged clause can be re-instantiated exactly, accepting reduced diversity of generated strategies. We chose to reject any strategy with a non-empty audit report over repairing flagged clauses, because repair could mask the very hindsight the harness is meant to detect; the cost is a lower yield of accepted strategies per grammar sample. A domain expert would not describe CATCH as a variant of Time Machine GPT's prompt-level cutoff, because CATCH constructs validity from a vintage-indexed per-clause information ledger and verifies it by truncated replay plus post-t sensitivity rather than asserting it in an instruction.

**Why it measures what we claim.** The per-clause provenance tag measures temporal validity of a strategy clause because the assumption that a field's vintage-timestamp upper-bounds its effective availability at decision time makes provenance membership equivalent to point-in-time availability; this assumption fails when a data vendor backfills or revises a series without changing its nominal publication date, in which case the tag reflects nominal rather than effective availability. The truncated-replay divergence measures hindsight leakage because the assumption that a clause whose value is invariant under truncation at t uses no post-t data makes replay equality equivalent to no post-cutoff dependence; this assumption fails when a clause's post-t dependence is exactly offset by a pre-t dependence (a cancellation), in which case replay equality reflects insensitivity to truncation rather than absence of hindsight. The post-t sensitivity check measures post-cutoff dependence because the assumption that perturbing only post-t fields changes the clause's value if and only if the clause depends on post-t information makes sensitivity equivalent to post-cutoff dependence; this assumption fails when the clause's dependence is non-monotonic or when the perturbation epsilon is too small to exceed numerical precision, in which case sensitivity reflects only local gradient rather than global dependence. The deterministic seed measures reproducibility because the assumption that the grammar expansion is a pure function of seed and ledger makes identical seeds yield identical clauses; this assumption fails when the ledger itself is updated between runs, in which case the seed reflects only partial reproducibility. The Phase 0 vintage-integrity check measures ledger validity because the assumption that the archive's vintage history is complete and accurate makes the check equivalent to ledger integrity; this assumption fails when the archive itself is incomplete, in which case the check reflects only archive coverage rather than true vintage integrity.

## Contribution

(1) CATCH, a cutoff-conditioned strategy generator with per-clause provenance certificates and a two-check hindsight auditor (symbolic membership plus truncated replay) that converts temporal validity from an asserted property into a constructed and verified invariant. (2) A design principle for simulated-user evaluation: temporal validity must be enforced at the clause level against a point-in-time information ledger, since prompt-level cutoffs cannot bind dynamically revealed strategies. (3) An audit taxonomy of hindsight failure modes (post-cutoff reference, replay divergence, revision-induced false validity) usable as veto conditions in open-world financial agent grading.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | FINCOSMOS hidden-strategy task suite, restricted to 30 series with explicit vintage history from a single vendor (ALFRED-style as-of snapshots) | Native point-in-time ledger and veto checks; vintage history makes the load-bearing assumption checkable |
| Primary metric | Hindsight-leak rate per accepted strategy | Directly measures the claim CATCH certifies |
| Secondary metric | Ledger-integrity coverage: fraction of fields with verifiable vintage history | Reports the scope over which the certificate is conditional |
| Baseline | Time Machine GPT (prompt-level cutoff) | Tests clause-level vs instruction-level binding |
| Baseline | FINCOSMOS unguarded agent | Tests whether audit adds anything over no guard |
| Baseline | Symbolic-only audit | Tests whether replay check is load-bearing |
| Baseline | Symbolic + replay audit (no sensitivity check) | Tests whether the post-t sensitivity check is load-bearing for cancellation cases |
| Ablation-of-ours | CATCH without replay check | Isolates contribution of truncated replay |
| Ablation-of-ours | CATCH without sensitivity check | Isolates contribution of post-t perturbation |
| Ablation-of-ours | CATCH with nominal-publication-date ledger (no vintage indexing) | Tests whether vintage indexing is load-bearing vs nominal dates |
| Injected-failure test | Synthetic backfill injection: for 10% of fields, overwrite the vintage-t value with a post-t value under the same nominal publication date | Confirms Phase 0 and symbolic check flag the backfill; measures false-negative rate of nominal-date ledger |
| Injected-failure test | Synthetic cancellation injection: construct clauses where post-t dependence is exactly offset by pre-t dependence | Quantifies how often cancellation occurs in the grammar and whether sensitivity check catches it |

### Why this setup validates the claim

The central claim is that per-clause provenance plus truncated replay plus post-t sensitivity yields strategies whose hindsight leakage is lower than any prompt-level or single-check alternative, and that the reduction is causally attributable to clause-level binding and vintage-indexed ledger integrity rather than to general caution. The FINCOSMOS suite restricted to 30 series with explicit vintage history is the right substrate because it supplies a hidden specification, a simulated user who reveals clauses at different times, and veto checks that flag post-cutoff references, while the vintage history makes the load-bearing assumption (that L[t] records the vintage-t value) checkable via Phase 0. Time Machine GPT's prompt-level cutoff tests the sub-claim that an instruction cannot bind dynamically revealed clauses; the unguarded FINCOSMOS agent tests whether auditing adds information beyond no guard; symbolic-only and the no-replay ablation test whether the second check is load-bearing; the no-sensitivity ablation and the cancellation injection test whether the third check is load-bearing for the cancellation failure mode; the nominal-date ablation tests whether vintage indexing is load-bearing versus nominal publication dates; and the backfill injection tests whether Phase 0 and the symbolic check catch the backfill failure mode. If CATCH's advantage is real, the leak rate must drop specifically on the subset of tasks where clauses are revealed across multiple decision times and where aggregates are revised post-cutoff, must stay at parity on single-decision-time tasks where no mechanism gap exists, and the injected-failure tests must show that the nominal-date ledger fails to flag backfill while the vintage-indexed ledger does. That concentration and the injected-failure results are what make the metric falsifiable rather than a generic win.

### Expected outcome and causal chain

**vs. Time Machine GPT (prompt-level cutoff)** — On a case where the simulated user reveals a margin clause at t=2 but the underlying revenue series was revised at t=4, the baseline produces a strategy that cites the revised revenue because its cutoff instruction is asserted once at prompt time and cannot re-bind a clause instantiated later. Our method instead tags each clause's fields against L[t=2] with vintage timestamps and flags the revised field symbolically, so we expect a noticeable leak-rate gap on multi-reveal tasks with revised aggregates and near-parity on single-reveal, unrevised tasks.

**vs. FINCOSMOS unguarded agent** — On a case where an agent opportunistically pulls a restated earnings figure because it is the most convenient web hit, the baseline produces a backtest that looks strong in-sample but fails the veto checks because nothing in its pipeline distinguishes vintage timestamp from retrieval date. Our method instead rejects any strategy with a non-empty audit report, so we expect the unguarded agent to show high leak rates broadly while CATCH concentrates its advantage on tasks whose fields have verifiable vintage history.

**vs. Symbolic-only audit** — On a case where a clause references a ratio whose numerator and denominator were both nominally published before t but revised afterward without a publication-date change, the baseline produces an accepted strategy because symbolic membership in L[t] passes. Our method instead re-evaluates the clause on data truncated at t and catches the replay divergence, so we expect a modest but consistent gap concentrated on aggregate-and-ratio clauses, with parity on raw single-field clauses.

**vs. Symbolic + replay audit (no sensitivity check)** — On a case where a clause's post-t dependence is exactly offset by a pre-t dependence (cancellation), the baseline produces an accepted strategy because v_trunc == v_full. Our method instead perturbs only post-t fields and catches the post-t sensitivity, so we expect a gap concentrated on the synthetic cancellation injection cases and on grammar slots where cancellation is possible (e.g., difference-of-aggregates clauses), with parity on clauses where no cancellation occurs.

**vs. CATCH without replay check** — On a case where a clause references an aggregate whose constituents were revised after t but whose nominal publication date is unchanged, the ablation accepts the strategy because symbolic membership passes. Our method catches the replay divergence, so we expect a gap concentrated on aggregate clauses.

**vs. CATCH without sensitivity check** — On the cancellation injection cases, the ablation accepts the strategy because replay equality holds. Our method catches the post-t sensitivity, so we expect a gap concentrated on cancellation cases and near-parity elsewhere.

**vs. CATCH with nominal-publication-date ledger** — On the backfill injection cases, the nominal-date ledger accepts the strategy because the nominal publication date is before t, while the vintage-indexed ledger flags the field in Phase 0. We expect the nominal-date ablation to show high leak rates on backfilled fields and near-parity on fields without backfill, quantifying the load-bearing role of vintage indexing.

### What would falsify this idea

If CATCH's leak-rate advantage over Time Machine GPT and the unguarded agent were uniform across single-reveal and multi-reveal tasks, or if removing the truncated-replay check produced no measurable change in leak rate on aggregate-and-ratio clauses, or if removing the sensitivity check produced no measurable change on cancellation injection cases, or if the nominal-date ledger matched the vintage-indexed ledger on backfill injection cases, the claim that clause-level binding, dual auditing, and vintage indexing are the causal mechanisms would be wrong.

## References

1. FINCOSMOS: An Open-World Evaluation Paradigm for Agents in Finance
