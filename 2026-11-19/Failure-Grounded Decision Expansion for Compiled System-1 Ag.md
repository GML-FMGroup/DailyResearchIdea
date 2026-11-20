# Failure-Grounded Decision Expansion for Compiled System-1 Agents

## Motivation

Compiled dual-system agents such as AnyS1 mine a fixed, human-authored decision set from procedural skills and treat low-confidence fallback to System-2 as a routing event rather than a learning signal, so the System-1 decision space is never closed under the substitutions that execution actually requires. Structurally, the mined enumeration is bounded by what skill authors anticipated, and every fallback that resolves a choice outside that enumeration is discarded, guaranteeing that the same unanticipated decision recurs as a fallback indefinitely. This is the same fixed-enumeration bottleneck visible in modular agent search (AgentSquare searches a fixed four-module design space) and in fixed communication topologies (AgentPrune prunes a given graph but cannot add edges), so decision-space closure under observed failures is a field-level gap, not a quirk of one system.

## Key Insight

A System-2 fallback is a witnessed counterexample to the sufficiency of the enumerated decision set, and because the fallback trajectory is execution-grounded and the skill text names the enclosing decision, each fallback yields a verifiable new candidate decision that can be added to the pointer's label space without ever needing a reward model.

## Method

**What it is.** FGDE (Failure-Grounded Decision Expansion) is a runtime wrapper around a compiled AnyS1 workflow that turns each System-2 fallback into a supervised example for a new decision point. Inputs: a compiled workflow with a calibrated 4B pointer System-1, the skill text that authored each decision point, and the live execution trace. Outputs: an expanded decision set D*, an incrementally updated pointer head, and a fallback-rate ledger.

**Load-bearing assumption (stated explicitly).** The method assumes that a System-2 fallback action, clustered by action signature and localized to a skill clause, constitutes a verifiable new pointer output slot whose addition is causally responsible for any observed fallback-rate drop, rather than the drop being an artifact of threshold recalibration on an enlarged label space. To make this assumption testable, the method freezes the per-skill threshold τ at its pre-expansion value for one full batch after committing a new slot, reports fallback rate at fixed τ as the primary signal, and treats re-calibration as a separate, later step whose effect is reported independently. Clause localization is verified by execution-grounded counterfactual attribution rather than embedding cosine similarity alone.

**How it works.**

```pseudocode
# Phase 0: offline (inherited from AnyS1)
D0 = mine_decisions(skills)              # fixed enumerated set
pi1 = train_pointer(D0, Any-DP, label_smoothing=0.1)
tau = calibrate_threshold(pi1, heldout, alpha=0.05)  # per-skill conformal quantile
D = D0
fixed_tau = copy(tau)                    # frozen for post-expansion evaluation

# Phase 1: runtime fallback capture
for each step t in episode:
    c = pi1(state_t)                     # pointer over D, returns (choice, conf)
    if conf >= tau[skill_id]:
        execute(c)
    else:
        a2 = System2(state_t)            # large-model action
        execute(a2)
        log FallbackRecord(state_t, a2, skill_id, enclosing_clause, system2_entropy)

# Phase 2: failure-grounded expansion (batch, every B=50 fallbacks)
for skill_id in touched_skills:
    R = FallbackRecords(skill_id)
    # 2a. cluster fallback actions by normalized action signature
    clusters = agglomerative(R.actions, metric=action_signature, linkage=average)
    for cluster in clusters with support >= k=3:
        # 2b. execution-grounded clause attribution (counterfactual)
        #     For each candidate clause c in skill_text, mask c and re-run System-2 on state_t.
        #     The governing clause is the one whose masking changes the System-2 action.
        candidate_clauses = top_5_by_embedding_similarity(state_t, skill_text)
        changed = []
        for c in candidate_clauses:
            a2_masked = System2(state_t, masked_skill_text=remove_clause(skill_text, c))
            if action_signature(a2_masked) != action_signature(a2):
                changed.append(c)
        if len(changed) == 1:
            clause = changed[0]
        else:
            # ambiguous: quarantine, do not promote
            quarantine(cluster, state_t, reason='ambiguous_clause')
            continue
        # 2c. propose a new candidate decision point
        d_new = (clause, canonical_action(cluster))
        if d_new not in D:
            D = D + {d_new}
            # 2d. relabel: pointer gets a new output slot, initialized to cluster centroid
            pi1.add_slot(d_new, init=centroid_embedding(cluster))
    # 2e. incremental re-grounding: fine-tune only the added/affected slots
    pi1 = finetune(pi1, D, Any-DP + R, epochs=2, lr=1e-4, freeze_shared_trunk=True)
    # 2f. freeze tau for one batch: evaluate fallback rate at fixed_tau
    fallback_rate_fixed = measure_fallback_rate(pi1, heldout, tau=fixed_tau)
    # 2g. promotion gate: accept expansion only if fallback rate at fixed tau drops and accuracy holds
    if fallback_rate_fixed(skill_id) < fallback_rate_before(skill_id) and acc_heldout >= acc_before - 0.01:
        commit(D, pi1)
        # 2h. re-calibrate threshold on the enlarged label space (separate, later step)
        tau = calibrate_threshold(pi1, heldout, alpha=0.05)
        fixed_tau = copy(tau)  # update frozen reference for next batch
    else:
        rollback(D, pi1)
```

A fallback record is only promoted to a decision point when it has support ≥ 3 in its cluster and its enclosing clause is uniquely identified by the counterfactual masking test (exactly one candidate clause changes the System-2 action when masked); otherwise it stays in a quarantine log and is re-examined at the next batch. The pointer's shared trunk is frozen during incremental re-grounding so that previously learned decisions are not forgotten, and only the slot embeddings and a small output projection are updated.

**Why this design.** We chose execution-grounded fallback clustering over LLM-proposed new actions because the fallback action is already the action the System-2 model actually took in that state, so the new decision point is witnessed rather than hypothesized, accepting that expansion is limited to decisions the System-2 model happens to make. We chose clause-localized insertion over a global free-form label space because AnyS1's pointer is a classifier over a fixed enumeration, and adding a slot requires knowing which skill clause the decision belongs to; the trade-off is that a fallback whose clause cannot be localized is quarantined rather than learned. We chose frozen-trunk incremental fine-tuning over full retraining because full retraining would re-mine the entire decision set and destroy the cost advantage that motivates AnyS1 in the first place, accepting slower adaptation per new decision. We chose a promotion gate with rollback over unconditional expansion because a single spurious fallback could otherwise pollute the label space, accepting that some genuine decisions are delayed by one batch. We chose to freeze τ for one batch after expansion and report fallback rate at fixed τ as the primary signal, because otherwise the fallback-rate drop would be confounded with threshold recalibration on an enlarged label space; re-calibration is performed only after the gate accepts the expansion, and its effect is reported separately. We chose counterfactual clause masking over embedding cosine similarity for clause localization because similarity can attach a slot to the wrong decision when clauses share vocabulary, whereas masking directly tests whether the clause causally governs the System-2 action. A domain expert would not describe this as retrieval over a growing library (the Toolformer/Gorilla family) because the new slot is created by intervening on the pointer's output space and verified by a fallback-rate drop at fixed τ, not by nearest-neighbor matching against an existing library; nor is it the AgentSquare-style search over a fixed module space, because the space itself is what changes.

**Why it measures what we claim.** The cluster support count k measures decision recurrence because we assume a genuinely missing decision point will be hit repeatedly across episodes while a one-off System-2 idiosyncrasy will not; this assumption fails when the environment is non-stationary, in which case support reflects transient task drift rather than a stable missing decision. The counterfactual clause-masking test measures decision attribution because we assume the skill clause that causally governs a state is the one whose removal changes the System-2 action; this assumption fails when multiple clauses jointly determine the action, in which case the test returns ambiguous and the fallback is quarantined rather than misattached. The fallback-rate drop at fixed τ measures closure of the decision space because we assume a fallback that recurs after expansion at the same threshold indicates the new slot was not learned; this assumption fails when the System-2 model itself changes behavior across episodes, in which case the rate reflects System-2 variance rather than pointer coverage. To separate coverage-driven from variance-driven fallbacks, we log System-2 action entropy per skill and report a variance-decomposition control: fallbacks are attributed to coverage only when System-2 entropy is low (stable action) and the fallback recurs. The held-out accuracy delta measures non-forgetting because we assume the held-out set is drawn from the same skill distribution as the expansion; this assumption fails under distribution shift, in which case the gate accepts an expansion that overfits the fallback batch.

## Contribution

['A failure-grounded decision-space expansion mechanism (FGDE) that converts System-2 fallback traces into verified new decision points for a compiled System-1 pointer, including a clause-localization step, a support-and-margin promotion gate, and a frozen-trunk incremental re-grounding procedure.', 'A design principle for dual-system agents: fallback events are the correct supervision signal for closing an enumerated decision set, and closure should be gated on fallback-rate reduction with rollback rather than on raw confidence.', 'A fallback ledger and quarantine log artifact that records which skill clauses generated unanticipated decisions, enabling audit of where human-authored skills under-specify agent behavior.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Three unseen agent benchmarks (AnyS1 evaluation suite) | Same regime where AnyS1 fallback occurs |
| Primary metric | Fallback rate per skill at fixed τ (frozen for one batch post-expansion), paired with held-out task accuracy | Directly measures decision-space closure without accuracy loss, decoupled from recalibration |
| Baseline | Uncompiled System-2 agent | Upper-bound reliability, no cost savings |
| Baseline | Static AnyS1 (fixed D0, no expansion) | Tests whether expansion closes residual fallbacks |
| Baseline | Full-retrain AnyS1 (re-mine all decisions) | Tests frozen-trunk incremental path against retraining |
| Baseline | Fallback-to-retrieval (nearest-neighbor library of fallback (state, action) pairs, no pointer output-space change) | Isolates output-space intervention as the novel ingredient |
| Ablation-of-ours | FGDE without promotion gate (unconditional expansion) | Isolates whether gating prevents label-space pollution |
| Ablation-of-ours | FGDE with embedding-cosine clause localization (original 0.1 margin) instead of counterfactual masking | Tests whether execution-grounded attribution matters |
| Diagnostic | Expert-adjudicated fallback log (precision/recall of k=3 support rule and clause-localization against human labels) | Grounds clustering and localization in measured agreement |
| Control | System-2 action entropy per skill (variance-decomposition) | Separates coverage-driven from variance-driven fallbacks |

### Why this setup validates the claim

The central claim is that residual System-2 fallbacks in a compiled AnyS1 workflow are not irreducible noise but evidence of missing decision points, and that these can be added incrementally without retraining or forgetting. The static AnyS1 baseline tests the first half: if fallbacks are irreducible, static AnyS1 and FGDE should have identical fallback rates at fixed τ, so any divergence is attributable to expansion rather than threshold recalibration. The full-retrain baseline tests the second half: it reaches a comparable decision space via a costlier route, so FGDE should match its accuracy while using far less re-mining compute — if FGDE instead underperforms full retraining, frozen-trunk re-grounding is losing information. The fallback-to-retrieval baseline tests whether output-space intervention is necessary: if a nearest-neighbor library achieves the same fallback-rate reduction at equal cost, the pointer-slot mechanism is not the active ingredient. The no-gate ablation tests the promotion mechanism: without gating, spurious fallbacks enter the label space and held-out accuracy should degrade, isolating the gate as the cause of stability. The embedding-cosine ablation tests whether execution-grounded clause attribution matters: if cosine localization matches counterfactual masking, the counterfactual test is unnecessary. The expert-adjudicated diagnostic grounds the k=3 support rule and clause-localization in measured precision/recall against human labels, so the clustering and localization steps are validated rather than assumed. The System-2 entropy control separates coverage-driven from variance-driven fallbacks: a fallback-rate drop at fixed τ accompanied by low System-2 entropy is evidence of coverage, whereas high entropy indicates System-2 variance. Fallback rate at fixed τ is the right primary metric because it is the quantity the mechanism directly intervenes on and is decoupled from recalibration; accuracy is the paired guardrail ensuring closure is not achieved by mislabeling.

### Expected outcome and causal chain

**vs. Uncompiled System-2 agent** — On a case where a routine decision recurs (e.g. selecting a known candidate among a small enumerated set), the uncompiled agent invokes the large model at every step because it has no compiled pointer, incurring full episode cost and latency. Our method instead resolves that decision in one pointer forward pass and only escalates to System-2 when confidence falls below the calibrated threshold, and it converts those escalations into new pointer slots. We expect a large cost and model-call reduction on routine-heavy episodes with task success preserved, matching AnyS1's reported cost profile while showing fallback rate at fixed τ declining across batches.

**vs. Static AnyS1 (fixed D0)** — On a case where the workflow hits a decision the mined enumeration never anticipated, static AnyS1 falls back to System-2 every time it recurs because its pointer has no output slot for that choice, so the fallback rate at fixed τ plateaus at a nonzero floor. Our method instead clusters the recurring fallback actions, localizes the governing skill clause by counterfactual masking, and adds a slot initialized to the cluster centroid, so the same decision is resolved by the pointer next time. We expect fallback rate at fixed τ to drop noticeably on the subset of skills that generate recurrent fallbacks while staying flat on skills with only one-off fallbacks.

**vs. Full-retrain AnyS1** — On a case where a new decision appears after initial compilation, full retraining re-mines the entire decision set and re-trains from scratch, recovering accuracy but paying the original cost advantage away and risking drift in previously stable slots. Our method instead adds only the affected slots, freezes the shared trunk, and evaluates at fixed τ before re-calibrating, so previously learned decisions are preserved by construction. We expect comparable held-out accuracy to full retraining but at a fraction of the re-mining cost, with a smaller accuracy perturbation on unaffected skills.

**vs. Fallback-to-retrieval** — On a case where a fallback recurs, the retrieval baseline stores the (state, action) pair in a nearest-neighbor library and resolves it at inference without touching the pointer's output space. Our method instead creates a new output slot and re-grounds the pointer, so the decision is integrated into the classifier rather than matched post hoc. We expect FGDE to achieve a lower fallback rate at fixed τ than retrieval at equal inference cost, isolating output-space intervention as the novel ingredient; if retrieval matches FGDE, the slot-creation mechanism is not necessary.

**vs. FGDE without promotion gate** — On a case where a spurious one-off fallback clusters with an unrelated action, the ungated variant commits a new slot that mislabels nearby states because nothing checks whether the expansion actually helps. Our method instead quarantines low-support or ambiguously localized fallbacks and commits only when fallback rate at fixed τ drops without accuracy loss, rolling back otherwise. We expect the ungated variant to show a held-out accuracy dip concentrated on the polluted skills, while gated FGDE holds accuracy and still captures the genuine recurrent decisions.

**vs. FGDE with embedding-cosine clause localization** — On a case where two skill clauses share vocabulary, cosine localization with a 0.1 margin may attach the new slot to the wrong clause, whereas counterfactual masking tests whether removing the clause changes the System-2 action. We expect the cosine variant to show higher misattachment (measured by expert-adjudicated diagnostic) and a smaller fallback-rate drop at fixed τ on ambiguous-clause skills, while counterfactual masking quarantines ambiguous cases and achieves a cleaner drop.

### What would falsify this idea

If FGDE's fallback-rate reduction at fixed τ were uniform across skills rather than concentrated on the subset with recurrent, clause-localizable fallbacks, or if the System-2 entropy control showed that the drop coincided with high System-2 variance, the expansion mechanism would not be capturing missing decisions but merely shifting the threshold or reflecting System-2 drift. Equally, if FGDE matched full retraining in cost or underperformed it in accuracy, or if fallback-to-retrieval matched FGDE at equal cost, the frozen-trunk incremental claim or the output-space intervention claim would be wrong.

## References

1. Thinking Fast, Anywhere: One System-1 for the Decision Points of Any Agent Workflow
