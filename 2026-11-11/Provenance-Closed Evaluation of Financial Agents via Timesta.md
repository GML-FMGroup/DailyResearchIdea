# Provenance-Closed Evaluation of Financial Agents via Timestamped Acquisition Ledgers

## Motivation

Open-world agent evaluations such as FINCOSMOS grade point-in-time correctness only on the final backtest window, while the simulated user's hidden strategy, the agent's Web acquisitions, and the elicited strategy fragments are never audited for look-ahead at the moment each influences a decision; a datum or strategy fragment acquired after the decision time can silently leak into the answer without any veto flag. This is the shared residual gap across Look-ahead bias work, Time Machine GPT, XBRL-agent pipelines, and Mind2Web-style interactive evaluation: temporal leakage identifiability is not established because provenance is recorded post hoc rather than enforced as a closure invariant over every influence channel.

## Key Insight

If every datum and every user-revealed strategy fragment carries an immutable acquisition timestamp and source identifier and the agent's decision is only permitted to read from the timestamp-closed subset of its own ledger, then look-ahead leakage becomes a decidable, per-decision property rather than a post-hoc heuristic, because the set of influences is exactly the set of ledger entries with timestamp ≤ decision time.

## Method

**What it is.** PACE (Provenance-Audited Closure Evaluation) is an evaluation harness that wraps a financial agent with a write-once acquisition ledger and a decision-time closure checker. Its inputs are the agent's tool calls (web fetch, user query, computation) and its outputs are a graded run in which every answer is accompanied by a closure certificate listing the ledger entries that were legally readable at decision time; any read outside that set is a hard veto.

**Load-bearing assumption (stated explicitly).** PACE's certificate attests to the set of ledger entries the agent *could* have read at decision time, not to the set the agent *reports* having used. The harness, not the agent, constructs each decision's readable context: before every `agent_step`, the harness materializes only `closed_subset(t_decision)` into the model's prompt and tool sandbox, and any ledger entry outside that subset is physically unreadable (not merely vetoed-if-named). The agent's `decision.influences` list is retained only as a diagnostic; it is not the load-bearing invariant. This assumption fails when the agent derives information from an unlogged computation over logged data or infers strategy content from the simulated user's refusals or timing; in those cases the certificate covers the materialized context but not the derived or inferred content, and the harness flags the run as `residual_unlogged` rather than certifying it as leakage-free.

**How it works.**

```pseudocode
LEDGER = []                       # append-only, entries never mutated
CLOCK  = monotone decision clock  # advances only on agent decision events

# Source availability windows (hyperparameters, set per source):
#   news: 24h publication lag; exchange_data: 0; user_reveal: 0;
#   compute: 0 (derived entries inherit max input acq_ts)

def record(kind, payload, source_id, wall_time, input_ids=None):
    entry = {id: len(LEDGER), kind, payload_hash, source_id,
             acq_ts: wall_time, decision_ts_at_acq: CLOCK.now(),
             input_ids: input_ids or []}   # provenance set for compute
    LEDGER.append(entry); return entry.id

def closed_subset(t_decision):
    # provenance closure: an entry is readable iff its acquisition
    # timestamp AND its source's declared availability window both
    # precede the decision time
    return {e for e in LEDGER
            if e.acq_ts <= t_decision and source_window(e.source_id).end <= t_decision}

def materialize_context(t_decision):
    # harness-controlled context boundary: only closed entries enter prompt/sandbox
    readable = closed_subset(t_decision)
    return {e.id: e.payload for e in readable}

def agent_step(observation, t_decision):
    context = materialize_context(t_decision)   # physical read barrier
    # diagnostic only: named influences are checked, but unnamed reads are
    # impossible because they are not in context
    for e_id in observation.referenced_entries:
        if e_id not in context:
            raise ClosureViolation(e_id, t_decision)  # hard veto, run rejected
    return agent.act(context, observation)

def user_reveal(fragment, t_decision):
    # strategy fragments are tagged at reveal time, not at authoring time
    return record(kind="strategy_fragment", payload=fragment,
                  source_id="simulated_user", wall_time=CLOCK.now())

def compute_tool(input_ids, fn):
    # instrumented computation: output carries provenance = union of input ids
    out = fn([LEDGER[i].payload for i in input_ids])
    return record(kind="compute", payload=out, source_id="compute",
                  wall_time=CLOCK.now(), input_ids=input_ids)

def grade(run):
    for decision in run.decisions:
        # certificate attests to materialized context, not self-report
        assert decision.context_ids == closed_subset(decision.t)
        if decision.has_unlogged_derivation:
            decision.flag = "residual_unlogged"
    recompute_metrics(run); check_strategy_against_hidden_spec(run)
    return ClosureCertificate(run, n_violations=0,
                              n_residual_unlogged=count_flags(run))
```

Hyperparameters: the source availability window is set per source (news: 24h publication lag; exchange_data: 0; user_reveal: 0; compute: 0, with derived entries inheriting `max(input acq_ts)`), and the decision clock granularity is one agent turn. The instrumented `compute_tool` wraps every computation call so that outputs carry a provenance set (union of input entry ids); an unlogged derivation is detected when the agent produces a computation output whose `input_ids` are not a subset of `closed_subset(t_decision)` at the decision turn, and is flagged `residual_unlogged` rather than silently certified.

**Why this design.** We chose an append-only ledger with a monotone decision clock over post-hoc timestamp filtering because post-hoc filtering (the FINCOSMOS approach) cannot distinguish a datum that was *available* at decision time from one that was *read* at decision time, accepting the cost that the agent must instrument every tool call. We chose to tag strategy fragments at *reveal* time rather than at authoring time because the simulated user's hidden strategy is authored before the run but only enters the agent's information set when elicited, accepting the cost that the harness must intercept user messages. We chose a hard veto over a soft penalty because a soft penalty lets a leaking run still produce a score, accepting the cost that some runs are discarded rather than graded. We chose a harness-controlled context boundary over agent self-reported influences because the model's context window is the real information channel and self-report is unreliable (Turpin et al., 2023; Lanham et al., 2023), accepting the cost that the harness must materialize context per turn and that derived or inferred content is flagged rather than certified. A domain expert would not describe this as a variant of FINCOSMOS's point-in-time check, because FINCOSMOS verifies the *backtest window* against a data cutoff, whereas PACE verifies the *materialized context of each decision* against a per-entry acquisition timestamp, which is a strictly finer-grained invariant that FINCOSMOS's veto checks do not compute.

**Why it measures what we claim.** `closed_subset(t_decision)` measures temporal provenance closure because the assumption that an entry's acquisition timestamp and its source's availability window jointly determine whether the datum could have been known at decision time; this assumption fails when a source back-dates or silently revises an entry, in which case the closure reflects the source's self-reported availability rather than true availability. `materialize_context(t_decision)` measures the agent's actual information set because the assumption that the harness controls every channel into the model's prompt and tool sandbox; this assumption fails when the agent derives information from an unlogged computation over logged data or infers strategy content from the simulated user's refusals or timing, in which case the certificate covers the materialized context but not the derived or inferred content, and the run is flagged `residual_unlogged` rather than certified leakage-free. `user_reveal` tagged at reveal time measures the elicitation boundary because the assumption that the simulated user only transmits strategy content through intercepted reveal events; this assumption fails when the agent infers strategy content from the user's *refusals* or timing, in which case the strategy fragment is legally absent from the ledger but informationally present, and the run is flagged `residual_unlogged`.

## Contribution

(1) A provenance-closure evaluation harness, PACE, that turns look-ahead leakage into a per-decision decidable invariant via an append-only timestamped ledger and a monotone decision clock, emitting a closure certificate for every graded run. (2) A design principle for open-world agent evaluation: strategy fragments from a simulated user must be tagged at reveal time, not authoring time, to make the elicitation boundary auditable. (3) A closure-certificate schema and veto protocol that extends FINCOSMOS-style point-in-time checks from the backtest window to the full influence set of each decision.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | FINCOSMOS hidden-strategy finance tasks | Provides simulated user, web acquisition, and hidden spec |
| Dataset | Synthetic leak-injection suite (50 hand-crafted cases with known ground-truth leak signatures) | Measures false-admit/false-reject rates without relying solely on FINCOSMOS's hidden spec |
| Primary metric | Leakage-free graded-run rate | Directly measures whether scores reflect only legally readable data |
| Secondary metric | False-negative rate on unlogged-derivation leaks | Quantifies the residual gap the certificate cannot cover |
| Baseline | FINCOSMOS point-in-time veto checks | Tests sub-claim that window-level checks miss per-decision leaks |
| Baseline | Post-hoc timestamp filtering | Tests sub-claim that availability ≠ read-time is undetectable post-hoc |
| Baseline | Uninstrumented agent (no ledger) | Tests sub-claim that naming influences is necessary for closure |
| Ablation-of-ours | PACE with soft penalty instead of hard veto | Isolates whether vetoing (vs. penalizing) leaks drives the gain |
| Ablation-of-ours | PACE with agent self-reported influences instead of harness-materialized context | Isolates whether the context boundary (vs. self-report) is load-bearing |

### Why this setup validates the claim

The central claim is that per-decision influence-set closure catches leakage that window-level and post-hoc checks cannot, and that hard vetoing converts that detection into trustworthy scores. FINCOSMOS tasks are the right substrate because they contain the exact failure triggers — a simulated user holding a hidden strategy, web data with publication lags, and programmatic grading against a hidden spec — so a leak, if it occurs, has a ground-truth signature. The synthetic leak-injection suite (50 cases) provides known ground-truth leak signatures so that false-admit and false-reject rates can be measured directly, including the unlogged-derivation case that the certificate flags as `residual_unlogged`. The three baselines each test a distinct sub-claim: FINCOSMOS veto checks test whether window-level invariants are too coarse; post-hoc filtering tests whether availability-time checks miss read-time leakage; the uninstrumented agent tests whether influence naming is load-bearing. The soft-penalty ablation isolates the veto mechanism from the detection mechanism, and the self-report ablation isolates the harness-controlled context boundary from the agent's introspection, so a gain cannot be attributed to ledger instrumentation alone. Leakage-free graded-run rate is the correct primary metric because it counts runs whose score is causally attributable to legally readable data — exactly the property PACE claims to guarantee — and it degrades detectably when any sub-claim fails; the false-negative rate on unlogged-derivation leaks quantifies the residual gap that the certificate explicitly does not cover.

### Expected outcome and causal chain

**vs. FINCOSMOS point-in-time veto checks** — On a case where the agent reads a news article published before the backtest window but after the decision turn that consumed it, the baseline passes the run because its veto compares the backtest window to a data cutoff and never inspects per-decision read times. Our method instead raises a ClosureViolation because the entry's acquisition timestamp exceeds the decision timestamp, so the run is vetoed rather than graded. We expect a noticeable gap in leakage-free graded-run rate concentrated on tasks with nonzero publication-lag sources, with parity on zero-lag exchange-data tasks.

**vs. Post-hoc timestamp filtering** — On a case where the agent fetches a datum that was available at decision time but only incorporates it into a later computation, the baseline cannot distinguish availability from read-time and either admits a stale influence or rejects a legitimate one, producing a score that misattributes provenance. Our method instead checks the materialized context against `closed_subset(t_decision)` at each decision, so the influence is admitted or vetoed at the exact turn it entered. We expect the baseline to show both false-admit and false-reject errors, while PACE shows near-zero false-admits and a modest false-reject rate only on derived-computation cases, which are flagged `residual_unlogged` rather than certified.

**vs. Uninstrumented agent (no ledger)** — On a case where the agent infers hidden-strategy content from the simulated user's refusals or response timing, the baseline has no record of the influence and grades a run whose score silently depends on leaked strategy content. Our method still cannot ledger the inference, but its reveal-time tagging and harness-controlled context boundary force the agent to declare or fail, surfacing the gap as a veto or an explicit `residual_unlogged` flag. We expect the uninstrumented baseline to show inflated scores on strategy-sensitive tasks and PACE to show either vetoed runs or a flagged residual, isolating the elicitation-boundary limitation.

**vs. PACE with agent self-reported influences (ablation)** — On a case where the agent uses an unlogged computation over logged data, the self-report variant produces a clean certificate because the agent does not name the derived influence, whereas the harness-materialized variant flags the run `residual_unlogged` because the computation output's `input_ids` are checked against `closed_subset(t_decision)`. We expect the self-report variant to show a higher false-admit rate on the synthetic leak-injection suite, isolating the context boundary as the load-bearing mechanism.

### What would falsify this idea

If PACE's gain in leakage-free graded-run rate over FINCOSMOS veto checks is uniform across all task subsets rather than concentrated on tasks with nonzero-lag sources and per-decision read-time mismatches, the claim that per-decision closure is the causal mechanism would be wrong. Equally, if the soft-penalty ablation matches the hard-veto variant, or if the self-report ablation matches the harness-materialized variant, the veto or context-boundary mechanism is not what produces trustworthy scores.

## References

1. FINCOSMOS: An Open-World Evaluation Paradigm for Agents in Finance
