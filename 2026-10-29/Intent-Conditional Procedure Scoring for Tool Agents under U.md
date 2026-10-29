# Intent-Conditional Procedure Scoring for Tool Agents under Underspecified Instructions

## Motivation

Existing procedure-aware benchmarks such as TauIndianBankBench require a pre-specified ground-truth step sequence and score deviations as violations, so they cannot distinguish an unjustified skip from a step made redundant by an underspecified user request. This is a structural limitation, not a data-coverage one: the scorer's reference is a single sequence, but the space of user intents consistent with an ambiguous instruction is a set, and any deviation that is optimal for some intent in that set is wrongly penalized. The same gap recurs across ToolSandbox's 'Insufficient Information' category and ACEBench's 'Special' category, where ambiguity is tested but correctness is still judged against one canonical trajectory.

## Key Insight

A deviation from a required step is justified exactly when it is optimal under some user intent that remains consistent with the dialogue so far, so scoring reduces to a likelihood-ratio between the best intent-consistent explanation of the observed trajectory and the best intent-inconsistent one, which a probabilistic intent posterior over a finite procedure grammar makes computable rather than heuristic.

## Method

**What it is.** We introduce **AIPS** (Ambiguity-aware Intent-Probabilistic Scorer), which takes as input a dialogue prefix, a procedure specification (a finite set of steps with preconditions and effects), and an observed agent trajectory, and outputs a per-deviation justification score plus a session-level procedure-adherence score. It replaces the single reference sequence with a posterior over intents induced by the procedure grammar.

**Load-bearing assumption (stated explicitly).** AIPS assumes that the finite procedure grammar's enumerated intent hypotheses H contain the user's true intent with non-negligible prior mass, so that `post_ok/post_bad` reflects genuine intent-conditional justification rather than the grammar's own coverage of the dialogue. This is a closed-world assumption: the grammar enumerates subsets of the step vocabulary S, but the dialogue-to-subset parse is a bottom-up CFG parse over typed arguments, and an ambiguous banking utterance (e.g., 'just close it out') may map to many subsets or to none. To make this assumption verifiable rather than asserted, we (i) add an explicit residual/unknown-intent hypothesis `h_unk` with a learned prior mass, (ii) calibrate the coverage prior against human justification labels on a held-out slice, and (iii) report grammar coverage as a measured quantity (see experiment).

**How it works.**

```pseudocode
INPUT: procedure P = (steps S, preconditions Pre, effects Eff, constraints C)
       dialogue prefix D_t, observed trajectory tau = [a_1..a_T]
HYPER: alpha=0.5 (intent prior smoothing), beta=1.0 (deviation penalty), K=8 (intent samples),
       gamma=0.1 (residual-intent prior mass), lambda=0.5 (coverage-prior calibration weight)

# Phase 1: enumerate intent hypotheses consistent with D_t
H = EnumerateIntents(P, D_t)         # each h in H is a subset of S the user actually wants
H = H ∪ {h_unk}                      # residual hypothesis: intent not representable in grammar
for h in H \ {h_unk}:
    prior[h] = (1/|H|) * (alpha + coverage(h, D_t))   # dialogue coverage of h's steps
prior[h_unk] = gamma                 # fixed residual mass, renormalized with the rest
# Calibration: prior[h] is re-weighted by a logistic calibrator fit on a held-out slice
#   prior_cal[h] ∝ prior[h] * sigmoid(lambda * coverage(h, D_t) + b)
#   where (lambda, b) are fit to maximize agreement with human justification labels.

# Phase 2: for each hypothesis, compute the optimal trajectory under h
for h in H \ {h_unk}:
    tau_h* = Plan(P, h)              # shortest step sequence satisfying Pre/Eff for h
    L[h]  = exp(-beta * EditDistance(tau, tau_h*))   # trajectory likelihood under h
L[h_unk] = exp(-beta * |tau|)        # residual: no optimal trajectory, maximal penalty

# Phase 3: split hypotheses by whether observed deviations are consistent
H_ok  = { h in H : every deviation in tau is on tau_h* }
H_bad = H \ H_ok
post_ok  = sum_{h in H_ok}  prior_cal[h]*L[h]
post_bad = sum_{h in H_bad} prior_cal[h]*L[h]

# Phase 4: justification score per deviation
for each deviation d in tau:
    J[d] = log( post_ok / (post_bad + eps) )   # >0 means some intent justifies d

RETURN session_score = mean_d min(1, max(0, J[d]/tau_norm)),  {J[d]}
```

The intent posterior is updated turn-by-turn as the dialogue grows: at each turn, hypotheses whose required steps are contradicted by the user's utterance get prior 0, and the remaining mass is renormalized. The procedure grammar is a small context-free grammar over step names with typed arguments; `EnumerateIntents` is a bottom-up parse of the dialogue against this grammar, capped at K=8 hypotheses by prior mass. The residual hypothesis `h_unk` is never pruned by dialogue contradiction; its mass `gamma` is renormalized with the rest, so `post_bad` always includes the mass of intents the grammar cannot represent.

**Named equivalences and their breaking assumptions.**

(D1) *Edit-distance-to-optimal-trajectory as deviation severity.* `EditDistance(tau, tau_h*)` measures deviation severity under hypothesis h because we assume the optimal trajectory is unique up to step order and that all orderings are equally costly. This assumption fails when two step orders are both valid (e.g., a procedure with commutative preconditions), in which case edit distance over-penalizes a legitimate reordering and `J[d]` reflects order sensitivity rather than true severity. We mitigate by setting beta=1.0 and by reporting a reordering-only violation subtype separately in the experiment.

(D2) *Session-score clamp as procedure adherence.* The clamp `min(1, max(0, J[d]/tau_norm))` measures procedure adherence because we assume a session with all deviations justified is as good as a session with no deviations. This assumption fails when a justified deviation still leaves the final record wrong, in which case the score reflects procedure alone and must be paired with the benchmark's outcome check. We therefore report session_score alongside the benchmark's outcome label and never claim session_score alone is a correctness metric.

**Why this design.** We chose a finite procedure grammar over a free-form LLM intent classifier because the grammar gives a closed enumeration of hypotheses, which makes the denominator `post_bad` well-defined and the ratio comparable across sessions; the cost is that intents outside the grammar are scored as unjustified, which we accept because the benchmark's procedures already define the step vocabulary, and we bound this cost with the residual hypothesis `h_unk` and the calibration step. We chose edit distance against the optimal trajectory per hypothesis over step-set comparison because edit distance preserves order and detects reordering violations that set comparison would miss, accepting that it is sensitive to spurious extra steps (mitigated by beta=1.0 and the min/max clamp). We chose a likelihood ratio over a thresholded classifier because a ratio is scale-free across procedures and sessions, so one threshold works across the 1,000 banking tasks, accepting that the ratio is unstable when `post_bad` is near zero (handled by the eps floor and the tau_norm normalization). We chose to update the posterior per turn rather than scoring only at session end because ambiguity resolves mid-dialogue, and a session-end-only scorer would credit the agent for information it did not have; the cost is that early-turn deviations are scored against a wider hypothesis set and are thus more likely to be marked justified. This is not a variant of TRACE's evidence-bank scoring: TRACE accumulates step-level evidence against a reference-free bank and never constructs a counterfactual optimal trajectory under an alternative intent, so it cannot distinguish a justified skip from an error.

**Why it measures what we claim.** The quantity `post_ok` measures the probability mass of user intents under which the observed trajectory is optimal because the assumption is that the procedure grammar's step effects are deterministic and the user's intent is a subset of steps; this assumption fails when a user wants a step the grammar does not contain, in which case `post_ok` reflects grammar coverage rather than true justification (this is why we add `h_unk` and calibrate the prior). The quantity `EditDistance(tau, tau_h*)` measures the deviation severity under hypothesis h because the assumption is that the optimal trajectory is unique up to step order; this assumption fails when two step orders are both valid, in which case edit distance over-penalizes a legitimate reordering. The ratio `J[d]` measures per-deviation justification because the assumption is that the prior over intents is calibrated by dialogue coverage; this assumption fails when the user's utterance is itself ambiguous in a way the grammar cannot represent, in which case `J[d]` reflects the grammar's ambiguity resolution rather than the user's actual intent. The session score's clamp measures procedure adherence because the assumption is that a session with all deviations justified is as good as a session with no deviations; this fails when a justified deviation still leaves the final record wrong, in which case the score reflects procedure alone and must be paired with the benchmark's outcome check.

## Contribution

(1) AIPS, an ambiguity-aware procedure scorer that replaces the single ground-truth step sequence with a posterior over intent hypotheses induced by a finite procedure grammar, producing per-deviation justification scores via a likelihood ratio. (2) A design principle: procedure adherence under underspecified instructions is only identifiable relative to an intent hypothesis set, and the likelihood ratio over that set is the minimal statistic that separates justified from unjustified deviations. (3) A turn-by-turn posterior update mechanism that lets the scorer distinguish deviations made before versus after ambiguity resolves, enabling evaluation of agents that ask clarifying questions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | TauIndianBankBench (1,000 banking sessions) | Procedures with typed steps and outcome checks already defined. |
| Dataset (generalization) | ToolSandbox 'Insufficient Information' subset | Tests scorer outside a single closed banking grammar. |
| Primary metric | Per-deviation justification AUC vs. human labels | Directly tests whether J[d] separates justified from erroneous deviations. |
| Secondary metric | Grammar coverage (fraction of human-labeled intents representable in CFG) | Quantifies the load-bearing assumption. |
| Baseline | Outcome-only scoring | Isolates value of procedure-aware justification. |
| Baseline | TRACE evidence-bank scoring | Tests whether counterfactual intent posterior adds discriminative power. |
| Baseline | Step-set comparison (no order) | Tests whether edit-distance ordering matters. |
| Ablation-of-ours | AIPS with uniform intent prior (no dialogue update) | Isolates turn-by-turn posterior updating. |
| Ablation-of-ours | AIPS without residual hypothesis h_unk (gamma=0) | Isolates the effect of the open-set residual mass. |
| Ablation-of-ours | AIPS with uncalibrated coverage prior (lambda=0) | Isolates the prior-calibration step. |

**Annotation protocol.** Human justification labels are collected by 3 annotators per deviation on a stratified sample of 600 deviations (200 missing-step, 200 extra-step, 200 reordering-only) drawn from the 1,000 sessions. Annotators see the dialogue prefix, the procedure specification, and the deviation, and label it justified / unjustified. Inter-annotator agreement target: Cohen's kappa > 0.7; disagreements are resolved by majority vote. If human labeling is infeasible, we fall back to synthetic deviations injected into clean sessions: for each clean session we inject one deviation of each subtype and label it justified iff the injected deviation is optimal under some intent consistent with the dialogue prefix (computed by the same grammar, so this fallback tests internal consistency, not ground truth, and is reported separately).

**Calibration slice.** The prior calibrator (lambda, b) is fit on a held-out slice of 200 sessions disjoint from the 1,000 evaluation sessions; the same slice is used to estimate grammar coverage.

### Why this setup validates the claim

The central claim is that a per-deviation justification score derived from a counterfactual intent posterior, rather than a reference-free evidence bank, can distinguish a deviation the user's intent justifies from an error. TauIndianBankBench is the right substrate because its procedures define a closed step vocabulary (so the grammar's denominator is well-defined) and its sessions carry outcome labels that let us separate justified deviations from record-corrupting ones. The ToolSandbox 'Insufficient Information' subset tests whether the scorer generalizes beyond a single closed grammar, which is the load-bearing assumption made explicit in the method. Outcome-only scoring tests the sub-claim that procedure awareness adds information beyond the final record; TRACE tests the sub-claim that constructing an alternative optimal trajectory under a competing intent is what carries the signal, not merely accumulating step evidence; step-set comparison tests whether order-sensitive edit distance is doing real work. The uniform-prior ablation is the sharpest test: if the gain survives without dialogue-conditioned priors, the mechanism is not the posterior but the trajectory likelihood, which would reframe the claim. The h_unk and uncalibrated-prior ablations directly test the load-bearing assumption: if removing the residual mass or the calibration step does not change per-deviation AUC, then the grammar-coverage concern is not empirically binding on this benchmark. Per-deviation AUC against human justification labels is the right metric because the session score's clamp can mask per-deviation errors; AUC is threshold-free and detects whether the ranking of deviations is correct, which is exactly what the ratio is supposed to produce. Grammar coverage is reported as a measured fraction with a 95% bootstrap confidence interval over the 200-session calibration slice, and we report the Pearson correlation between coverage and |J[d] - human_label| to quantify whether coverage predicts J[d] error.

### Expected outcome and causal chain

**vs. Outcome-only scoring** — On a session where the customer asks for a balance check and the agent skips the mandated identity-verification step but still returns the correct balance, outcome-only scoring marks the session correct because it inspects only the final record. Our method instead computes that under every high-prior intent the verification step lies on the optimal trajectory, so `post_bad` dominates and `J[d]` goes negative for that deviation. We expect a large AUC gap concentrated on sessions whose outcome is correct but whose procedure is violated, with near-parity on sessions where both outcome and procedure are clean.

**vs. TRACE evidence-bank scoring** — On a case where the user's opening utterance already narrows intent to a single account-closure step and the agent legitimately skips the intermediate 'confirm address' step, TRACE accumulates step-level evidence against its reference-free bank and flags the skip as unsupported because it has no counterfactual trajectory under the narrowed intent. Our method instead renormalizes the prior after the opening turn, so the narrowed intent dominates `post_ok` and the skip receives a positive `J[d]`. We expect the gap to concentrate on sessions with early intent-narrowing utterances and to shrink toward parity on sessions whose intent stays ambiguous throughout.

**vs. Step-set comparison** — On a case where the agent performs the correct steps in the wrong order (e.g., debits before authorization), step-set comparison sees full coverage and reports no deviation. Our method's edit distance against the optimal ordered trajectory detects the reordering and assigns a negative `J[d]` because no intent's optimal trajectory permits that order. We expect a clear separation on reordering-only violations, with parity on missing-step and extra-step violations where both methods already fire.

**vs. AIPS without residual hypothesis h_unk (gamma=0)** — On a session whose user intent falls outside the grammar (e.g., 'just close it out' parsed to no subset), the gamma=0 variant assigns all mass to grammar-representable hypotheses, so `post_bad` is dominated by whichever hypotheses the CFG happens to parse and `J[d]` reflects grammar coverage. Our full method assigns residual mass gamma=0.1 to `h_unk`, so `post_bad` includes the unrepresentable-intent mass and `J[d]` is less overconfident. We expect the gap to concentrate on sessions with low grammar coverage (bottom quartile of the calibration slice), with near-parity on high-coverage sessions.

**vs. AIPS with uncalibrated coverage prior (lambda=0)** — On a session where the lexical-overlap coverage heuristic misranks hypotheses (e.g., a short utterance that overlaps many step names), the uncalibrated prior assigns mass by raw coverage and `J[d]` inherits the misranking. Our full method re-weights by the logistic calibrator fit to human justification labels, so `J[d]` tracks the calibrated posterior. We expect a modest AUC gain concentrated on sessions with short or lexically ambiguous utterances, and we report the Pearson correlation between grammar coverage and |J[d] - human_label| to quantify the residual coverage effect.

### What would falsify this idea

If the gain over TRACE and step-set comparison is uniform across all deviation subtypes rather than concentrated on early-intent-narrowing and reordering cases, then the counterfactual intent posterior is not the causal mechanism and the score is merely a noisier step-coverage count. Likewise, if the uniform-prior ablation matches full AIPS on per-deviation AUC, the dialogue-conditioned posterior is doing no work and the central claim about ambiguity resolution is wrong; and if the h_unk and uncalibrated-prior ablations match full AIPS, the load-bearing grammar-coverage assumption is not empirically binding on this benchmark.

## References

1. TauIndianBankBench: Procedure-Aware, Action-Oriented Evaluation of Tool-Using Conversational Agents
2. Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents
3. TRAJECT-Bench:A Trajectory-Aware Benchmark for Evaluating Agentic Tool Use
4. CRMArena-Pro: Holistic Assessment of LLM Agents Across Diverse Business Scenarios and Interactions
5. ACEBench: Who Wins the Match Point in Tool Usage?
6. τ2-Bench: Evaluating Conversational Agents in a Dual-Control Environment
7. Non-Collaborative User Simulators for Tool Agents
8. ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities
