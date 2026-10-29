# Certified Recoverability Scheduling for Multi-Agent LLM Systems

## Motivation

MAST (Cemri et al.) provides a 14-mode failure taxonomy and an LLM-as-a-Judge classifier, but the label is purely descriptive: it is emitted post hoc and is not coupled to any runtime state estimate or transition model, so no scheduler can act on it with a bounded effect. Consequently, Reflexion-style retry loops triggered by such labels have no notion of whether the retry returns the coordination state to a recoverable region or drives it into an unrecoverable one — the field has failure taxonomies but no observation model for a runtime controller, and therefore no closed-loop recoverability property for diagnosed failures.

## Key Insight

Treating the MAST failure label as a partial observation over a finite abstraction of coordination state makes the label actionable only when it indexes a pre-certified (action, invariant) pair whose post-state is guaranteed inside a recoverable set; the scheduler's dispatch rule is then a closed-form membership test over that set, which is a property no taxonomy-plus-retry composition can establish because neither component carries a transition relation.

## Method

**What it is.** CRS (Certified Recoverability Scheduler) is a runtime scheduler for multi-agent LLM systems. Its inputs are (i) the MAST failure label emitted by an LLM-as-a-Judge over the current trace window, and (ii) a low-dimensional coordination-state observation (which agent holds which subgoal, which messages are in flight, which subgoals are verified). Its outputs are a dispatched recovery action from a finite certified action set, or an escalation to a bounded fallback. The load-bearing artifact is a *recoverability certificate*: a table C[(failure_mode, coord_state_class)] -> (action, invariant, post_state_class), compiled offline and checked at execution time.

**Load-bearing assumption (stated explicitly).** The certificate's RECOVERABLE label is a property of the *replay distribution* over MAST-Data, and CRS assumes this replay distribution is predictive of the *on-policy* post-state distribution when the same action is dispatched by a live LLM agent. This assumption is not verified by the offline replay alone: substituting a different action changes the LLM's prefix, so the replayed continuation is off-policy by construction. CRS therefore (a) calibrates the certificate thresholds tau and H against measured replay-vs-on-policy agreement (see the calibration phase below) and (b) emits a *set-valued* certificate entry that is certified only when the post-state class is invariant across all observed replays and all sampled LLM continuations, so the guarantee holds regardless of the replay-vs-on-policy gap. The invariant check at runtime is a consistency check against this set-valued prediction, not an independent test of recoverability; the independent test is the calibration phase.

**How it works.**

```pseudocode
# ---------- OFFLINE: certificate compilation ----------
Inputs: MAST taxonomy M (14 modes), action library A (retry, reassign_subgoal,
        renegotiate_role, request_verification, restart_subtask, escalate),
        trace corpus T (MAST-Data), state abstraction phi (below)
1. Define phi: coordination trace -> coord_state_class in S, where
   S = (subgoal_ownership x in_flight_msgs x verified_set) coarsened to
   |S| <= 32 classes via bisimulation on the message-dependency DAG.
2. For each (m in M, s in S, a in A): roll out a in a counterfactual replay
   of the trace from s, observe post-state class s' = phi(trace after a).
   Collect n_replay = 20 replays per (m,s,a) cell.
3. Set-valued certification: for each (m,s,a), let S'(m,s,a) be the set of
   post-state classes observed across the n_replay replays AND across
   n_sample = 5 on-policy LLM continuations sampled from the live policy
   conditioned on the same prefix (the calibration sample, see below).
   Label (m,s,a) CERTIFIED iff |S'(m,s,a)| = 1 (post-state class invariant)
   AND the empirical frequency of that class over n_replay + n_sample >= tau.
4. Label s RECOVERABLE iff from s every MAST mode reachable within horizon
   H=3 has at least one CERTIFIED action a whose post-state class is in
   RECOVERABLE; else UNRECOVERABLE.
5. Emit certificate C[(m,s)] = (a*, inv, S') where a* is the CERTIFIED action
   with the highest lower-confidence-bound (Wilson 95% CI lower bound) on
   landing in RECOVERABLE, inv is the invariant 's' in RECOVERABLE and no new
   MAST mode of category inter-agent-misalignment appears in the post-trace
   window', and S' is the set-valued post-state prediction.

# ---------- CALIBRATION PHASE (derives tau and H) ----------
Inputs: held-out calibration split T_cal (20% of MAST-Data, disjoint from T),
        grid tau in {0.7, 0.8, 0.9, 0.95}, H in {2, 3, 4}
C1. For each (tau, H) on the grid, compile C on T and evaluate on T_cal:
    for each calibration trace, dispatch a* from C and record whether the
    on-policy post-state class phi(after a*) equals the certificate's
    predicted class in S'.
C2. Compute agreement(tau, H) = fraction of calibration dispatches where
    on-policy post-state class matches the certificate prediction, and
    false_certification_rate(tau, H) = fraction of CERTIFIED entries whose
    on-policy post-state class is NOT in S'.
C3. Select (tau*, H*) = argmax agreement subject to
    false_certification_rate <= 0.05. Use tau*, H* in the runtime certificate.

# ---------- RUNTIME: closed-loop scheduling ----------
At each scheduler tick:
  m_hat <- Judge(trace_window)                 # MAST label, may be 'none'
  s_hat <- phi(trace_window)                   # state abstraction
  if m_hat == 'none': dispatch(next_planned_action); continue
  (a*, inv, S') <- C[(m_hat, s_hat)]
  if s_hat not in RECOVERABLE:                 # pre-condition fails
      escalate(bounded_fallback)               # at most K=2 escalations/task
  else if not CheckInvariant(inv, s_hat):      # certificate mismatch
      escalate(bounded_fallback)
  else:
      dispatch(a*); record(post_state=phi(after a*))
      if phi(after a*) not in S': escalate(bounded_fallback)
```

The invariant check is a concrete predicate over the state abstraction (membership in RECOVERABLE plus absence of new inter-agent-misalignment labels in the post-trace window), not a learned score. The bounded fallback is a fixed escalation ladder (request_verification -> restart_subtask -> human handoff) capped at K=2 per task, so the scheduler cannot loop.

**Why this design.** We chose a *pre-compiled certificate table* over an online learned transition model because the MAST taxonomy is finite and stable (14 modes, kappa=0.88), so offline counterfactual replay over MAST-Data yields a finite, auditable object; the cost is that certificates are only as good as the replay distribution, which we accept because the calibration phase measures the replay-vs-on-policy gap and the set-valued certificate plus the invariant check catch distribution shift at runtime. We chose a *bisimulation-coarsened state abstraction* (|S| <= 32) over a raw trace embedding because recoverability is a property of the message-dependency DAG, not of surface text; the cost is loss of fine-grained state, accepted because the certificate's set-valued post-state check re-validates at execution time. We chose *membership in a recoverable set* as the dispatch predicate over a Reflexion-style self-critique score because membership is a closed-form test with a stated invariant, whereas a critique score has no transition semantics; the cost is that some recoverable states may be conservatively escalated. A domain expert would not describe CRS as 'MAST + Reflexion': Reflexion supplies neither a state estimate nor a transition relation, and CRS's dispatch rule is a set-membership test over a compiled certificate, not a retry heuristic — the certificate is the object Reflexion lacks.

**Why it measures what we claim.** The state abstraction `phi(trace)` measures *coordination state class* because bisimulation on the message-dependency DAG preserves the reachability relation among subgoal-ownership and in-flight-message configurations; this assumption fails when two traces are bisimilar on the DAG but differ in agent intent, in which case phi reflects structural coordination only. The certificate entry `C[(m,s)]` measures *actionable recoverability of failure mode m from state s* because counterfactual replay of action a from s plus the calibration sample is a valid intervention on the coordination state, and the set-valued certificate is certified only when the post-state class is invariant across replays and on-policy samples; this assumption fails when the calibration sample does not cover the on-policy continuation distribution, in which case the entry reflects replay-plus-calibration recoverability rather than full on-policy recoverability. The invariant check `CheckInvariant(inv, s_hat)` measures *closed-loop safety of the dispatch* because membership in RECOVERABLE plus absence of new inter-agent-misalignment labels is exactly the post-condition the certificate was compiled to guarantee; this assumption fails when the Judge misses a misalignment mode, in which case the check reflects judge-observed safety. The escalation cap K measures *boundedness of the fallback* because each escalation strictly reduces the remaining action set; this assumption fails when the fallback ladder itself re-enters a certified state, in which case K bounds scheduler steps rather than recovery attempts.

## Contribution

(1) A recoverability-certificate artifact: a per-(failure-mode, coordination-state-class) table of (action, invariant, post-state) compiled by counterfactual replay over MAST-Data, with a runtime membership test that makes the MAST label a control signal rather than a post-hoc description. (2) A closed-loop scheduling algorithm, CRS, whose dispatch rule is a closed-form membership test over a pre-computed recoverable set with a bounded escalation ladder, yielding a runtime-checkable recoverability invariant for every MAST failure mode. (3) A coordination-state abstraction (bisimulation-coarsened message-dependency DAG, |S| <= 32) that is the minimal state sufficient for certificate compilation, together with an analysis of which MAST modes are certifiable under it.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | MAST-Data trace corpus (held-out split T_cal, 20% of traces, disjoint from compilation split T) | Same distribution certificates were compiled over; T_cal is used only for calibration and evaluation, never for certificate compilation |
| Primary metric | Task success rate on traces with injected MAST failures | Directly tests recoverability, not raw capability |
| Secondary metric | Replay-vs-on-policy post-state class agreement on T_cal, as a function of n_replay in {5, 10, 20, 40}, tau in {0.7, 0.8, 0.9, 0.95}, H in {2, 3, 4} | Directly tests the load-bearing assumption that replay is predictive of on-policy post-state |
| Secondary metric | False-certification rate on T_cal (fraction of CERTIFIED entries whose on-policy post-state class is not in S') | Quantifies the cost of the replay-vs-on-policy gap |
| Secondary metric | Fraction of the 14 x 32 = 448 (mode, state) cells that are CERTIFIED vs escalated | Reports state-space coverage |
| Baseline | Reflexion-style self-critique retry | Tests whether structured transition semantics beat retry heuristics |
| Baseline | Static MAST-conditioned action policy (no certificate) | Tests whether the certificate/invariant adds value over label-to-action mapping |
| Baseline | Ungated LLM-as-Judge escalation | Tests whether membership gating beats free-form escalation |
| Ablation-of-ours | CRS without invariant check (certificate dispatch only) | Isolates the safety contribution of the invariant |
| Ablation-of-ours | CRS with point-valued certificate (single replay, no set-valued S', no calibration) | Isolates the contribution of set-valued certification and calibration |

### Why this setup validates the claim

The central claim is that a pre-compiled recoverability certificate plus a closed-form invariant check recovers multi-agent LLM tasks more reliably than retry or label-conditioned heuristics, and that the gain is attributable to the certificate's transition semantics rather than to extra compute or escalation. Using MAST-Data's held-out split T_cal keeps the evaluation on the same distribution the certificate was compiled from, so any gain cannot be attributed to distribution shift; failure injection lets us isolate the subset where recovery, not raw capability, is the bottleneck. The calibration study directly tests the load-bearing assumption: by measuring replay-vs-on-policy post-state class agreement as a function of n_replay, tau, and H on T_cal, and by deriving tau* and H* from the observed agreement curve subject to false-certification rate <= 0.05, we replace the a-priori tau=0.9 with a threshold justified by measured agreement. The set-valued certificate ablation isolates whether the set-valued S' and calibration sample are what make the certificate predictive of on-policy behavior. The Reflexion baseline tests the sub-claim that retry heuristics lack transition semantics; the static MAST-conditioned policy tests whether the certificate/invariant machinery adds anything beyond a label-to-action map; the ungated escalation baseline tests whether set-membership gating is what supplies boundedness and safety. The no-invariant ablation is the load-bearing control: if removing the invariant check does not degrade outcomes, the certificate's safety claim is unsupported. Task success rate on failure-injected traces is the right primary metric because it is the only quantity that directly reflects whether an action landed the system in a recoverable post-state; token-level or judge-score metrics would not distinguish a lucky retry from a certified dispatch.

### Expected outcome and causal chain

**vs. Reflexion-style self-critique retry** — On a case where an agent holding a subgoal emits a message contradicting a peer's verified subgoal, the baseline produces repeated self-critique and retry because its mechanism has no state estimate or transition relation, so it cannot detect that the coordination DAG is now cyclic. Our method instead looks up the (inter-agent-misalignment, conflicting-ownership) certificate entry and dispatches renegotiate_role because the certificate encodes the post-state class that resolves the conflict, so we expect a clear gap on the inter-agent-misalignment subset with parity on single-agent failure modes.

**vs. Static MAST-conditioned action policy** — On a case where the Judge labels a failure correctly but the current coordination state has already drifted (e.g., verification was lost after a retry), the baseline dispatches the mode's default action because it conditions only on the failure label, ignoring state. Our method instead checks membership of the observed state class in RECOVERABLE and escalates or picks the state-specific action because the certificate is keyed on (mode, state), so we expect a gap concentrated on traces where the label is correct but the state is off-distribution, with near-parity where label and state are aligned.

**vs. Ungated LLM-as-Judge escalation** — On a case where the Judge repeatedly flags the same failure, the baseline escalates unboundedly or loops because it has no cap and no post-condition. Our method instead escalates at most K=2 times along the fixed ladder because the invariant check plus the escalation cap bound the action set, so we expect fewer runaway traces and a modest success-rate advantage on long traces, with parity on short traces where the first escalation already succeeds.

**vs. CRS with point-valued certificate (ablation)** — On a case where the replay distribution and on-policy distribution diverge (stochastic continuation drift), the point-valued certificate predicts a single post-state class that the on-policy continuation may not reach, so the ablation dispatches a* and then fails the post-state check, escalating unnecessarily. Our method instead certifies only entries whose post-state class is invariant across replays and on-policy samples, so we expect the set-valued certificate to reduce false escalations and raise success rate on the subset of traces where the calibration study shows replay-vs-on-policy agreement is below 0.9, with parity where agreement is high.

### What would falsify this idea

If the calibration study shows replay-vs-on-policy post-state class agreement is below 0.7 for all (n_replay, tau, H) on the grid, the load-bearing assumption is false and the certificate's RECOVERABLE label does not transfer to the deployed system. If the no-invariant ablation matches full CRS on failure-injected traces, the invariant check is decorative and the certificate's safety claim is wrong. If CRS's gain over the static MAST-conditioned policy is uniform across all failure subsets rather than concentrated where state drift and inter-agent misalignment occur, the certificate's transition semantics are not the causal driver.

## References

1. Why Do Multi-Agent LLM Systems Fail?
