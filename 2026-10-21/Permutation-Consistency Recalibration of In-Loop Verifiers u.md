# Permutation-Consistency Recalibration of In-Loop Verifiers under Tool-Use Distribution Shift

## Motivation

Across agentic-LM paper-trees, in-loop verifiers emit discrete, programmatically-checkable signals whose mapping to true task success is frozen at design time — Sage-A1.5's progress-checking and evidence-validation signals are the canonical instance, and Generalizing Verifiable Instruction Following's RLVR constraint modules inherit the same design-time-frozen assumption. Under tool-use distribution shift this mapping drifts, and because no branch re-estimates the signal→success relation online, a verifier that once meant 'success' can silently mean 'plausible-looking failure' while the agent keeps trusting it. The structural gap is that all these systems treat a programmatic signal as a sufficient proxy for ground-truth success independent of the execution distribution, and no existing method exploits the fact that the verifier is invoked many times per episode on overlapping sub-traces.

## Key Insight

When a verifier is repeatedly invoked on permuted sub-traces of the same episode, the exchangeability of those permutations forces the joint distribution of its discrete signals to satisfy a self-consistency invariant whose violation is a direct, label-free readout of the drift in the signal→success mapping.

## Method

**What it is.** PERCH (Permutation-Exchangeability ReCalibration Head) is an online recalibration layer that sits between an in-loop verifier and the agent's controller. Inputs: the stream of discrete verifier signals emitted on overlapping sub-traces of each episode, plus the permutation structure (which sub-trace is a permutation of which). Outputs: a per-signal reliability weight `w_s ∈ [0,1]` and a recalibrated success estimate `p̂_success` that the controller uses in place of the raw signal.

**Load-bearing assumption (stated explicitly).** PERCH assumes that when a verifier's discrete signal genuinely tracks task success, permuting or reordering a sub-trace leaves that signal unchanged, so within-group disagreement `r_g` is a label-free readout of drift in the signal→success mapping rather than of legitimate order-dependence in the task itself. This assumption is false for programmatic verifiers that enforce ordering, positional, or count/format constraints (e.g., IFBench/IFEval 'A before B'), which must return different signals on permutations of the same content; it also fails structurally for tool-use sub-traces whose calls have data dependencies, since permuting them produces traces outside the execution distribution. PERCH therefore does not use raw `r_g` as the drift statistic; it uses an order-invariance-controlled residual (Step 1 below) that subtracts each signal's baseline order-sensitivity estimated on in-domain traces where the signal→success map is known to be stable, and it restricts permutation groups to sub-traces that are genuinely exchangeable (no data dependency between tool calls).

**How it works.**

```pseudocode
# Per episode e, verifier V emits signals on sub-traces T_1..T_k
# Permutation groups G = {g}: sets of sub-traces that are permutations of each other
#   (same underlying content, different order / decomposition)
# Exchangeability filter: a group g is admitted only if no tool call in any member
#   consumes the output of another call in the same member (checked by the harness
#   from the tool-call dependency graph); groups failing this are dropped.
# Order-sensitivity class c(g) ∈ {order-sensitive, order-insensitive} is declared
#   by the constraint type (IFBench constraint taxonomy) and recorded per group.

# --- Step 0: baseline order-sensitivity (estimated once on in-domain stable traces) ---
# On a calibration set of N_cal = 512 in-domain episodes where the signal→success map
# is known to be stable (no tool-use distribution shift), compute for each signal value v
# and each order-sensitivity class c:
#   b_{v,c} = mean over admitted groups g with mode signal v and class c of r_g
# Store b_{v,c} as a fixed reference table (no online update).

for each episode e:
    collect signals s_{g,i} = V(T_i) for each permutation group g, member i
    # --- Step 1: order-invariance-controlled self-consistency residual ---
    for each admitted group g:
        r_g = disagreement( {s_{g,i}} )   # fraction of members whose signal differs from group mode
        v_g = mode signal of group g
        c_g = declared order-sensitivity class of g
        r_tilde_g = max(0, r_g - b_{v_g, c_g})   # excess disagreement beyond baseline
    # --- Step 2: reliability update (exponential-forgetting Beta) ---
    for each signal value v in {0,1}:
        n_v   <- (1 - lambda) * n_v   +  count of admitted groups where mode signal == v
        k_v   <- (1 - lambda) * k_v   +  count of admitted groups where mode==v AND r_tilde_g < tau
        w_v   <- k_v / (n_v + beta0)          # lambda=0.05, tau=0.2, beta0=2
    # --- Step 3: recalibrated success estimate ---
    p_hat = sum_v  w_v * 1[s == v] / sum_v w_v   # normalized over observed signals
    emit p_hat to controller; controller acts on p_hat, not raw s
```

Permutation groups are formed by the harness, which already re-invokes the verifier on reordered sub-traces for progress checking; PERCH reuses those invocations rather than adding new ones. `lambda`, `tau`, `beta0` are the only hyperparameters and are set as above across all tasks. The baseline table `b_{v,c}` is estimated once from 512 in-domain stable episodes and frozen; it is not a learned calibration head and carries no labels beyond the known-stable regime.

**Why this design.** We chose an order-invariance-controlled residual over a learned calibration head because the residual is computable from traces the agent already produces, accepting the cost that it only detects drift that manifests as order-sensitivity beyond the declared baseline. We chose exponential-forgetting Beta counts over a sliding window because drift is non-stationary and a window would discard the pre-drift baseline needed to detect the shift, accepting the cost that very old reliable signals are slowly forgotten. We chose to weight by group-mode agreement rather than per-invocation accuracy because a single invocation carries no label while a permutation group carries an internal consistency check, accepting the cost that groups with fewer than two members contribute nothing. We chose to restrict groups to genuinely exchangeable sub-traces and to subtract the declared order-sensitivity baseline because raw permutation disagreement fires on well-behaved order-sensitive verifiers and on data-dependent tool traces, accepting the cost that order-sensitive constraints and dependent tool calls yield fewer admitted groups. A domain expert would not describe PERCH as a variant of RLVR constraint verification (Generalizing Verifiable Instruction Following) or of Sage-A1.5's progress checker: those freeze the signal→success map and use the signal directly, whereas PERCH treats the signal as a noisy measurement and re-estimates the map from permutation structure alone, with no reward model and no offline labels.

**Why this measures what we claim.** The controlled residual `r_tilde_g` measures drift in the signal→success mapping because under the exchangeability assumption — that permuting an admitted sub-trace does not change the underlying task content — a verifier whose signal tracks true success must return the same signal on every member of a group beyond its declared baseline order-sensitivity; this assumption fails when a verifier is legitimately order-sensitive in a way not captured by the declared class, in which case `r_tilde_g` reflects residual order-dependence rather than drift. The reliability weight `w_v` measures the probability that signal value `v` implies true success because the Beta posterior over mode-agreement counts is a sufficient statistic for that probability under the assumption that within-group agreement is independent of the drift mechanism; this assumption fails when drift is correlated across groups within an episode, in which case `w_v` reflects episode-level rather than signal-level reliability. The recalibrated `p̂_success` measures the agent's best estimate of latent task success because it is a reliability-weighted average of the verifier's discrete readings under the assumption that the verifier is the only success-relevant measurement channel; this assumption fails when the verifier is systematically blind to a failure mode, in which case `p̂_success` reflects verifier coverage rather than true success.

## Contribution

(1) PERCH, an online recalibration layer that re-estimates the verifier signal→success mapping from permutation structure in the agent's own execution traces, requiring no offline synthesized labels and no additional verifier invocations. (2) A design principle: permutation-consistency residuals computed over overlapping sub-trace invocations are a sufficient, label-free statistic for detecting tool-use distribution shift in discrete verifier signals, and exponential-forgetting Beta reliability weights track that shift without a learned calibration head. (3) A drop-in interface for in-loop verifiers (Sage-A1.5-style progress checkers and RLVR constraint modules) that exposes per-signal reliability weights alongside the recalibrated success estimate.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | IFBench held-out constraints | Out-of-domain constraints expose verifier drift directly |
| Primary metric | Success-rate gap: calibrated vs raw signal | Directly measures recalibration benefit on controller decisions |
| Baseline | Raw verifier signal (no recalibration) | Tests whether recalibration adds anything at all |
| Baseline | Static Beta calibration (no forgetting) | Tests whether non-stationary drift handling matters |
| Baseline | Sliding-window calibration | Tests whether exponential forgetting beats window discard |
| Ablation-of-ours | PERCH without permutation residual (w_v uniform) | Isolates contribution of the self-consistency residual |
| Ablation-of-ours | PERCH with raw r_g (no baseline subtraction) | Tests whether order-invariance control is necessary |
| Diagnostic | P(success\|v) estimated on 512 labeled episodes | Validates w_v equivalence to true success probability |

### Why this setup validates the claim

The central claim is that PERCH recovers a drifting signal→success map from permutation structure alone, so the controller's decisions should improve precisely where the raw verifier's signal has decoupled from true success. IFBench's held-out constraints are the right dataset because they are the regime where a verifier trained on in-domain constraints is most likely to drift, making the predicted failure mode present rather than hypothetical. The raw-signal baseline tests the first sub-claim (recalibration beats no recalibration); the static-Beta baseline tests the second (forgetting is necessary under non-stationarity); the sliding-window baseline tests the third (exponential forgetting preserves a pre-drift baseline that a window discards). The uniform-weight ablation is the load-bearing control: if PERCH's gain survives when the self-consistency residual is removed, the residual is doing no work and the mechanism claim collapses. The raw-r_g ablation is the second load-bearing control: if PERCH's gain survives without baseline subtraction, the order-invariance control is doing no work and the stated load-bearing assumption is not actually needed. The P(success|v) diagnostic directly tests the equivalence between w_v and the true success probability on a labeled subset, so the reliability-weight claim is validated rather than assumed. The calibrated-vs-raw success-rate gap is the right primary metric because it directly measures whether the controller acts on a better success estimate, rather than measuring signal quality in isolation, which could improve without helping decisions.

### Expected outcome and causal chain

**vs. Raw verifier signal (no recalibration)** — On a case where an out-of-domain constraint is order-sensitive in a way the verifier was not trained for, the raw baseline passes the verifier's reading straight to the controller, so a spuriously high signal on a reordered sub-trace triggers premature commitment because the signal→success map is frozen and wrong. Our method instead sees excess within-group disagreement (r_tilde_g above tau) on that permutation group, down-weights the signal value via the Beta posterior, and emits a lower p̂_success, so the controller delays commitment. We expect a noticeable gap on the order-sensitive constraint subset but near-parity on order-insensitive constraints, where no drift is present to correct.

**vs. Static Beta calibration (no forgetting)** — On a case where the verifier's reliability shifts mid-stream after the agent enters a new constraint family, the static baseline's Beta counts are dominated by pre-shift observations, so it keeps trusting a signal value that has since decoupled and the controller over-commits because the posterior cannot move. Our method instead applies exponential forgetting, so pre-shift counts decay and the post-shift controlled residual dominates the reliability estimate, letting w_v track the new regime. We expect the static baseline to degrade progressively across the stream while PERCH stays flat, producing a widening gap concentrated in the later portion of episodes.

**vs. Sliding-window calibration** — On a case where drift is gradual rather than abrupt, the sliding-window baseline discards the pre-drift baseline exactly when it needs it to distinguish a genuine shift from noise, so it oscillates between over- and under-trusting the signal and the controller's decisions become erratic. Our method instead retains a decaying pre-drift baseline, giving a stable reference against which the residual shift is detected, so w_v moves smoothly. We expect lower variance in p̂_success across consecutive episodes for PERCH, with the window baseline showing larger swings on the gradual-drift subset and parity on abrupt-drift cases.

**vs. PERCH with raw r_g (no baseline subtraction)** — On the order-sensitive constraint subset, the raw-r_g ablation fires on well-behaved verifiers that correctly enforce ordering constraints, so it down-weights reliable signals and the controller under-commits; our method subtracts the declared baseline b_{v,c} and admits only exchangeable groups, so it does not fire on legitimate order-dependence. We expect the raw-r_g ablation to lose on order-sensitive constraints while matching PERCH on order-insensitive ones, confirming that the order-invariance control is necessary.

**vs. P(success|v) diagnostic** — On the 512 labeled episodes, we compare w_v to the empirical P(success|v) per signal value; if w_v tracks the empirical probability within binomial confidence intervals, the reliability-weight equivalence holds; if w_v is systematically above or below, the mode-agreement sufficient-statistic assumption is violated and the w_v claim is falsified.

### What would falsify this idea

If PERCH's gain over the raw-signal baseline is uniform across order-sensitive and order-insensitive constraint subsets rather than concentrated where the controlled permutation residual is predicted to fire, then the residual is not the causal driver and the central claim is wrong. Likewise, if the uniform-weight ablation matches full PERCH, the self-consistency mechanism contributes nothing and the design's rationale collapses; and if the raw-r_g ablation matches full PERCH, the order-invariance control is unnecessary and the load-bearing assumption is not actually load-bearing.

## References

1. Sage-A1.5: From Long-Horizon Execution to Scientific Exploration with Capability Composition
2. Generalizing Verifiable Instruction Following
3. DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
