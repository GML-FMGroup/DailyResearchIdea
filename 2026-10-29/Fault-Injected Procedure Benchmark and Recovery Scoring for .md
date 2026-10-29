# Fault-Injected Procedure Benchmark and Recovery Scoring for Tool-Using Agents

## Motivation

Existing procedure-aware benchmarks such as TauIndianBankBench assume fault-free execution traces: their dual-check scorer verifies required steps and final records only when tools behave as expected, so an agent that silently retries, skips a step, or hallucinates a repair is scored identically to one that never encountered a fault. This structural blind spot recurs across tool-use evaluation suites (TRAJECT-Bench, ToolSandbox, ACEBench), which similarly lack a mechanism to attribute recovery behavior to detection, diagnosis, and repair rather than to luck or to final-record coincidence. The root cause is that fault injection is not part of the task representation, so there is no observable event to score against and no way to separate a recovered procedure from a never-faulted one.

## Key Insight

Faults can be represented as first-class, typed events in the procedure trace, and because each injected fault has a known ground-truth repair step, recovery can be scored as a deterministic alignment between the agent's post-fault actions and the procedure's repair subgraph — a property that fault-free traces cannot provide.

## Method

```pseudocode
# FARA: Fault-Augmented Recovery Assessment
Input: task suite T (from TauIndianBankBench), procedure spec P per task,
       fault catalog F = {tool_timeout, tool_error, customer_interrupt, wrong_record},
       injection rate r (default 0.3 per task), seed s
Output: recoverability score R per session, decomposition into D_detect, D_diag, D_repair

# (D) Load-bearing assumption, stated explicitly:
#   detect_score is claimed to measure fault detection under assumption A_detect:
#     the agent's first post-fault action referencing the fault (retry/ask/inspect)
#     cannot arise from a blind retry policy. Failure mode F_detect: a fixed
#     retry-on-error loop produces detect=1 with zero detection.
#   diag_score is claimed to measure diagnosis under assumption A_diag:
#     repair_step(f) is not the default next procedure step. Failure mode F_diag:
#     procedure momentum produces diag=1 without diagnosis.
#   repair_score is claimed to measure repair under assumption A_repair:
#     the final record would not have matched without the fault. Failure mode
#     F_repair: task luck produces repair=1.
#   Each assumption is operationalized below by a matched-control contrast.

1. Fault injection (offline, before session):
   For each task t in T:
     sample k ~ Poisson(r * |steps(t)|) faults from F
     assign each fault f a trigger step i_f (the procedure step whose tool call it preempts)
     attach to t a fault schedule S_t = [(i_f, f, repair_step(f))]
   Repair_step(f) is a ground-truth procedure step (e.g., re-invoke tool with corrected args, escalate to human).
   Derive repair_step(f) automatically from the procedure spec's error-handling subgraph where present;
   for residual steps without an error-handling edge, label manually and report inter-annotator agreement (target Cohen's kappa >= 0.8 on a 100-fault sample).

2. Matched-control schedule construction (offline, per fault):
   For each fault f at trigger i_f, construct a matched control session c(f) with the SAME task t and SAME fault f at i_f, but with the repair_step(f) replaced by a decoy repair step d(f) that is NOT the default next procedure step and is NOT the ground-truth repair.
   The control session is run with the same agent, same seed, same customer simulator.
   This yields paired traces (tau_faulted, tau_control) for every fault.

3. Session execution (agent interacts with simulated customer + tools):
   At each step i, if i matches a scheduled fault, the tool returns the fault response
   (timeout, error, or a customer interruption injected as a user turn).
   Agent emits action a_i (tool call, clarification, or message).
   Log full trace tau = [(i, a_i, tool_response_i)].
   Run both the faulted session and its matched control session.

4. Recovery scoring (dual-check extension, contrastive):
   For each fault f at trigger i_f, compare the agent's post-fault action a_{i_f+1} in the faulted trace to its action a_{i_f+1} in the matched control trace:
     detect_score = 1 if a_{i_f+1} in the faulted trace references the fault AND differs from a_{i_f+1} in the control trace by either (a) a changed tool argument, or (b) an explicit query/clarification turn, else 0.
       # This operationalizes A_detect: blind retry appears identically in both traces and is scored 0.
     diag_score  = 1 if a_{i_f+1} in the faulted trace matches repair_step(f) or a valid pre-repair probe AND the fault-free policy probability of that action at step i_f+1 (estimated from the fault-free baseline traces on the same task) is < 0.5, else 0.
       # Momentum-adjusted: subtracts the probability of the repair step under the fault-free policy.
     repair_score = 1 if repair_step(f) is executed AND final record matches expected AND the final record in the matched control trace does NOT match expected, else 0.
       # This operationalizes A_repair: task luck appears in both traces and is scored 0.
   R = mean over faults of (detect + diag + repair)/3, weighted by fault severity.
   Also compute procedure adherence and final-record checks as in TauIndianBankBench.
```

**Why this design**
We chose to inject faults at the procedure-step level rather than at the tool-call level because procedure steps are the unit the dual-check scorer already verifies, so a fault can be aligned to an expected repair step without re-annotating tasks; the cost is that faults not tied to a step (e.g., ambient latency) are out of scope. We chose a deterministic fault schedule per task (seeded Poisson) over adversarial or adaptive injection because it makes recoverability comparable across models and runs, accepting that it may under-sample rare fault combinations. We chose a three-way decomposition (detect/diag/repair) over a single binary recovery flag because a single flag conflates an agent that notices a fault but cannot repair it with one that never notices, and the decomposition is what lets TauIndianBankBench's procedure-aware scorer attribute failure to a specific phase; the cost is that the diagnostic step is harder to annotate and we restrict it to repair-step matching plus a small set of pre-repair probes. We chose to extend the existing dual-check scorer rather than build a new evaluator so that recoverability is directly comparable to the fault-free procedure score, accepting the dependency on TauIndianBankBench's task format. We added a matched-control session per fault so that each phase score is a differential between faulted and control traces, which is what makes the phase attribution causal rather than coincidental; the cost is doubling the number of sessions run per task.

**Why this measures what we claim**
The detect_score measures fault detection because we assume that an agent's first post-fault action referencing the fault (retry, ask, inspect) and differing from its matched-control action is equivalent to having noticed it; this assumption fails when an agent retries blindly without noticing, in which case detect_score reflects a lucky retry rather than detection — the matched-control contrast is what rules this out, because blind retry appears identically in both traces. The diag_score measures diagnosis because we assume that matching the ground-truth repair step or a pre-repair probe, when that action has fault-free policy probability < 0.5, is equivalent to having identified the fault cause; this assumption fails when the repair step is also the default next step, in which case diag_score reflects procedure momentum rather than diagnosis — the momentum adjustment is what rules this out. The repair_score measures repair because we assume that executing the repair step and matching the final record, while the matched control does not match, is equivalent to having repaired the fault; this assumption fails when the final record would have matched even without the fault, in which case repair_score reflects task luck rather than repair — the control-trace record check is what rules this out. The weighted average R measures recoverability under injected faults because we assume the seeded fault schedule is representative of the fault distribution the procedure is designed to handle; this assumption fails when the catalog omits a fault class, in which case R reflects recoverability on the covered faults only.

## Contribution

['FARA, a fault-injection and recovery-scoring extension to procedure-aware tool-use benchmarks that represents faults as typed events aligned to procedure steps and decomposes recovery into detection, diagnosis, and repair scores.', 'A design principle: recovery must be scored against a ground-truth repair subgraph attached to each injected fault, because final-record checks alone cannot distinguish a recovered procedure from a never-faulted one.', 'A fault catalog and injection schedule generator (tool timeout, tool error, customer interruption, wrong record) that is reusable across procedure-aware task suites.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | TauIndianBankBench sessions with seeded fault schedules and matched-control schedules | Procedure steps already scored, so faults align to repairs |
| Primary metric | Recoverability score R (severity-weighted, contrastive detect/diag/repair) | Decomposes recovery into phases, not one flag |
| Baseline | Fault-free procedure-aware score (upper bound) | Shows how much recovery headroom faults consume |
| Baseline | Outcome-only scoring under same faults | Tests whether final-record matching hides recovery failures |
| Baseline | Single binary recovery flag | Tests whether decomposition adds diagnostic signal |
| Baseline | Non-contrastive FARA (original single-action alignment, no matched control) | Tests whether matched-control contrast removes lucky-retry and momentum confounds |
| Ablation-of-ours | FARA without diag term (detect+repair only) | Isolates whether diagnosis carries independent signal |
| Ablation-of-ours | FARA without momentum adjustment (diag uses raw repair-step match) | Isolates whether momentum adjustment removes procedure-momentum inflation |

### Why this setup validates the claim

The central claim is that recoverability is a distinct, decomposable capability that procedure-aware scoring alone cannot see, and that the three phases fail for different reasons. The fault-free procedure-aware baseline establishes the ceiling: if injected faults cost nothing on R relative to fault-free adherence, the method is measuring noise rather than recovery. The outcome-only baseline tests the sub-claim that final-record matching masks recovery failures, since an agent can stumble to a correct record after a fault without ever detecting it. The binary-flag baseline tests whether the detect/diag/repair decomposition actually separates agents that notice-but-cannot-repair from agents that never notice, which is the whole reason for three terms. The non-contrastive FARA baseline tests the load-bearing assumption directly: if the matched-control contrast does not change phase scores relative to single-action alignment, then lucky retry and procedure momentum are not present in this agent population and the contrast is unnecessary; if it does change them, the contrast is doing the attribution work the claim requires. The diag ablation tests whether the diagnostic term is doing independent work or merely tracking repair. The momentum-adjustment ablation tests whether the fault-free policy probability threshold is what removes momentum inflation, or whether raw repair-step matching suffices. R is the right metric because it is severity-weighted and phase-attributed, so a predicted effect (gap concentrated in diag for weak agents, in detect for strong-but-brittle agents) is directly readable rather than averaged away. Together these form a falsifiable test: the claim predicts non-uniform phase profiles across baselines and a diag-specific gap, not a flat offset.

### Expected outcome and causal chain

**vs. Fault-free procedure-aware score** — On a case where a tool timeout preempts a balance-check step, the fault-free baseline never faces the fault, so its adherence score reflects only clean execution and cannot distinguish an agent that recovers from one that never encounters trouble. Our method injects the fault and scores detect/diag/repair against the annotated repair step and the matched control, so the same agent's recovery is measured directly. We expect a clear separation between fault-free adherence and R, with the gap largest for agents whose clean scores are high but whose R is low — the brittle-high-adherence pattern.

**vs. Outcome-only scoring under same faults** — On a case where a wrong_record fault returns a stale balance and the agent blindly proceeds, the outcome-only baseline can still award credit if the final record happens to match because the customer's request was unaffected. Our method instead flags detect_score=0 at the post-fault step because the agent's action is identical to its matched-control action, attributing the failure to detection even though the record matched. We expect outcome-only to show near-parity with our repair_score but a large deficit on detect and diag, i.e. a gap concentrated on the early phases rather than the final record.

**vs. Single binary recovery flag** — On a case where an agent notices a tool_error, asks the customer a clarifying question, but then re-invokes the tool with the same bad arguments, the binary flag records a failure indistinguishable from an agent that never reacted. Our method separates these: detect=1 (action differs from control by an explicit query), diag=0 (repair step not matched or policy probability >= 0.5), repair=0, versus detect=0 for the oblivious agent. We expect agents to cluster into distinct phase profiles (high-detect/low-diag vs low-detect) that the binary flag collapses, producing a visible bimodality in our decomposition that the flag cannot express.

**vs. Non-contrastive FARA (original single-action alignment)** — On a case where a blind retry policy fires on any tool error, the non-contrastive scorer records detect=1 because the retry action references the fault, while our matched-control contrast records detect=0 because the same retry appears in the control trace. We expect the non-contrastive baseline to overstate detect specifically on agents with fixed retry loops, producing a gap between full FARA and the non-contrastive baseline concentrated on those agents and near-parity elsewhere.

**vs. FARA without diag term** — On a case where the repair step coincides with the default next procedure step, the no-diag ablation credits repair via procedure momentum and reports near-full recovery. Our method's diag term requires a pre-repair probe or explicit repair-step match with fault-free policy probability < 0.5, so momentum-only recovery scores lower on diag. We expect the ablation to overstate recovery specifically on the subset where repair_step equals the default next step, producing a gap between full FARA and the ablation concentrated on that subset and near-parity elsewhere.

**vs. FARA without momentum adjustment** — On a case where the repair step coincides with the default next procedure step, the no-momentum-adjustment ablation credits diag=1 via raw repair-step match, while our momentum-adjusted diag requires fault-free policy probability < 0.5 and scores diag=0. We expect the ablation to overstate diag specifically on the subset where repair_step equals the default next step, producing a gap between full FARA and the ablation concentrated on that subset and near-parity elsewhere.

### What would falsify this idea

If our gain over outcome-only and binary-flag baselines is uniform across fault types and phases rather than concentrated in detect and diag on the predicted brittle-agent subset, the decomposition is not capturing distinct failure modes and the central claim fails. Likewise, if the diag ablation matches full FARA everywhere, the diagnostic term carries no independent signal and the three-way decomposition collapses to two; and if the non-contrastive FARA baseline matches full FARA everywhere, the matched-control contrast is unnecessary and the load-bearing assumption is unverified.

## References

1. TauIndianBankBench: Procedure-Aware, Action-Oriented Evaluation of Tool-Using Conversational Agents
2. Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents
3. TRAJECT-Bench:A Trajectory-Aware Benchmark for Evaluating Agentic Tool Use
4. CRMArena-Pro: Holistic Assessment of LLM Agents Across Diverse Business Scenarios and Interactions
5. ACEBench: Who Wins the Match Point in Tool Usage?
6. τ2-Bench: Evaluating Conversational Agents in a Dual-Control Environment
7. Non-Collaborative User Simulators for Tool Agents
8. ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities
