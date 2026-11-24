# Composable Trajectory Progress Algebra for Physical Agency Benchmarks

## Motivation

WorldLeague (E3) reports per-dimension Interactive Score and physical success rate from robot trajectories, but never demonstrates that its trajectory-derived progress function composes across the WorldSeek→WorldShape→WorldShift sub-task decompositions; each suite is scored in isolation, so a system's composite goal progress under a multi-suite goal is undefined. This is a structural gap, not a tuning issue: without a stated composition operator and an identifiability condition, any aggregate of sub-task scores is an arbitrary heuristic, and cross-suite comparisons of Physical Agency Profiles cannot be justified. The same residual recurs across benchmark trees that report progress per decomposition without a composition law.

## Key Insight

Trajectory progress is additive under a shared control interface exactly when the sub-task invariants are disjoint and the interface state is a sufficient statistic for hand-off, so composite goal progress is identified as the path-ordered sum of sub-task progress increments rather than an unconstrained aggregate.

## Method

**What it is.** We introduce CAPA (Compositional Agency Progress Algebra), a scoring layer over WorldLeague that takes as input (i) a robot trajectory segmented into WorldSeek/WorldShape/WorldShift sub-task phases and (ii) per-phase progress functions, and outputs a composite goal-progress scalar with an explicit composition operator and a hand-off identifiability check.

**Load-bearing assumption (stated explicitly).** CAPA's certificate is valid only under assumption A: the per-phase progress function p_{k+1} is a function of the shared-interface state s_b alone, with no dependence on policy-internal memory (recurrent state, context window, or observation history not encoded in s_b). Under A, replay-invariance of p_{k+1} to prefix truncation from s_b is equivalent to s_b being a sufficient statistic for phase k+1. When A fails — the policy carries unobserved memory — a replay that re-runs the same policy reconstructs the same internal memory from the same observation stream, so the check returns 'invariant' precisely in the memory-dependent case it is designed to catch; C_b then reflects apparent rather than true sufficiency. We therefore (i) state A as a precondition, (ii) add a counterfactual-memory perturbation to the hand-off check so that hidden-memory dependence produces divergence, and (iii) when the policy's internal state cannot be reset or observed, downgrade the certificate to a necessary-but-not-sufficient screen and report it as such.

**How it works.**

```pseudocode
INPUT: trajectory τ = [(s_t, a_t)]_{t=1..T}, phase labels φ(t) ∈ {Seek, Shape, Shift}
       per-phase progress functions p_Seek, p_Shape, p_Shift ∈ [0,1]
       shared-interface state s_t
       policy internal-state handle h_t (recurrent state / context window), if exposed
OUTPUT: composite progress P(τ) ∈ [0,1], composition certificate C ∈ {pass, fail, screen}

1. Segment τ into contiguous phase intervals I_Seek, I_Shape, I_Shift by φ(t).
   Phase labels φ(t) are taken from WorldLeague's provided phase annotations when available;
   otherwise inferred by a fixed rule: a phase boundary is declared at the first timestep where
   the active sub-task goal id changes in the environment's observation dict.
2. For each phase k, compute increment Δ_k = p_k(s_end(k)) − p_k(s_start(k)).
3. Hand-off check at each boundary b between phase k and k+1:
     C_b^trunc = 1[ p_{k+1} invariant to replaying any prefix of τ before b ]
         (implemented as: rerun p_{k+1} from s_b with 3 random prefix truncations;
          require identical p_{k+1} output within ε=1e-3)
     C_b^mem   = 1[ p_{k+1} invariant to a matched perturbation of policy-internal memory ]
         (implemented as: at s_b, reset the policy's recurrent state / context window to a
          neutral value — zero vector for GRU/LSTM, empty context for transformer — and
          rerun p_{k+1}; require identical output within ε=1e-3)
     If the policy exposes no internal-state handle, set C_b^mem = unknown and
     downgrade the boundary certificate to C_b = screen.
     Otherwise C_b = 1 iff C_b^trunc = 1 AND C_b^mem = 1.
4. Composition operator: P(τ) = Σ_k w_k · Δ_k, with w_k = 1 (unweighted additive),
   and an order-sensitivity probe: recompute P under a permuted phase order τ';
   record |P(τ) − P(τ')| as the non-commutativity margin.
5. Certificate: C = pass iff all C_b = 1; C = screen iff any C_b = screen and none = 0;
   C = fail iff any C_b = 0. The non-commutativity margin is always reported
   (not required to be zero — reported to flag order-dependent goals).
6. Return (P(τ), C, {Δ_k}, margin).
```

**Why this design.** We chose an additive composition operator over a learned aggregator because additivity is the only operator whose validity is checkable from trajectory replays alone, accepting the cost that it cannot represent goals where sub-task progress is multiplicative (e.g., Shape only counts if Seek succeeded) — those are flagged by the non-commutativity margin rather than silently mis-scored. We chose a replay-based hand-off sufficiency check over a learned independence head because the latter would assert a statistical property the data cannot guarantee (the anti-pattern of disentanglement heads over coupled variables), accepting the cost of 3 extra rollouts per boundary. We added the counterfactual-memory perturbation because replay-invariance alone is confounded by policy-internal memory (assumption A), and we report a 'screen' label rather than a false 'pass' when the policy's internal state cannot be reset. We chose to report the non-commutativity margin rather than enforce commutativity because WorldShift is by construction order-sensitive, accepting that the algebra is a partial (order-annotated) algebra rather than a total one. A domain expert would not describe CAPA as a variant of retrieval-based or controller-glued benchmark aggregation (e.g., simple mean-of-suite-scores as in WorldLeague's per-dimension reporting) because CAPA's load-bearing object is the hand-off sufficiency certificate, which no prior aggregation computes.

**Why it measures what we claim.** The quantity Δ_k measures sub-task goal progress because we assume the per-phase progress function p_k is monotone in task-relevant state and zero at phase start; this assumption fails when a phase begins mid-progress, in which case Δ_k reflects within-phase improvement rather than absolute progress. The hand-off certificate C_b measures interface sufficiency because we assume (A) p_{k+1} is a function of the shared-interface state alone and (B) replay-invariance to prefix truncation and to memory perturbation is equivalent to the shared interface state being a sufficient statistic for the next phase; this assumption fails when p_{k+1} depends on unobserved memory carried in the policy rather than the interface state, in which case C_b reflects apparent rather than true sufficiency — the memory-perturbation sub-check is designed to make this failure produce divergence rather than a false pass. The non-commutativity margin measures order-dependence of the composite goal because we assume permuting phase order is a valid intervention on goal structure; this assumption fails when phases are physically non-permutable (Shape cannot precede Seek), in which case the margin reflects infeasibility rather than genuine order-sensitivity.

## Contribution

['A compositional progress algebra (CAPA) for trajectory-derived benchmark scoring, comprising per-phase progress increments, an additive composition operator, and a replay-based hand-off sufficiency certificate.', 'A concrete design principle: composite goal progress in physical-agency benchmarks is identified only when interface-state sufficiency holds at phase boundaries, and the non-commutativity margin quantifies where additivity is invalid.', "An open scoring layer applicable to WorldLeague's 60 tasks that exposes per-boundary certificates alongside Physical Agency Profiles."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | WorldLeague 60-task simulated suite | Native Seek/Shape/Shift phases with shared interface |
| Dataset (synthetic control) | 3-phase gridworld with known sufficient statistic (state = (x,y,carried_flag); hidden memory = last-seen color) | Ground-truth sufficiency for measuring certificate sensitivity/specificity |
| Primary metric | Hand-off certificate pass rate (pass/fail/screen) | Directly tests interface sufficiency claim |
| Secondary metric | Certificate sensitivity and specificity vs. synthetic ground truth | Quantifies detection of memory-dependent boundaries |
| Baseline | WorldLeague per-dimension mean aggregation | Tests whether additive composition alone suffices |
| Baseline | Interactive Score without certificate | Tests whether sufficiency check adds signal |
| Baseline | Physical success rate only | Tests whether progress algebra beats binary outcome |
| Ablation-of-ours | CAPA without hand-off check (additive only) | Isolates contribution of the certificate |
| Ablation-of-ours | CAPA with truncation-only check (no memory perturbation) | Isolates contribution of the counterfactual-memory sub-check |
| Ablation-of-ours | CAPA with memory-injection (plant hidden state at boundary) | Confirms certificate fires when memory dependence is present |

### Why this setup validates the claim

The central claim is that CAPA's load-bearing object — the hand-off sufficiency certificate — detects a failure mode that prior aggregation cannot, and that additive composition with a reported non-commutativity margin yields a more interpretable composite progress than per-dimension means. The WorldLeague suite is the right dataset because its three phase types share a single control interface, so a trajectory can be segmented and replayed at boundaries; without shared-interface trajectories the certificate is undefined. The synthetic gridworld is added because it provides a known sufficient statistic, so certificate sensitivity (does it fire when memory dependence is present?) and specificity (does it stay quiet when it is not?) can be measured against ground truth rather than inferred from expected concentration. The primary metric is certificate pass rate because that is the only quantity that distinguishes CAPA from mean-of-suite aggregation: if the certificate never fails, CAPA reduces to a weighted sum and the design's justification collapses. The two non-certificate baselines test whether the certificate adds signal beyond Interactive Score and beyond binary success, respectively — if CAPA's ranking matches both baselines exactly, the certificate is decorative. The ablation (additive-only, no certificate) isolates the certificate's contribution from the composition operator's contribution, so a gain in the full method can be attributed to the sufficiency check rather than to additivity per se. The truncation-only ablation isolates the counterfactual-memory sub-check from the prefix-truncation sub-check, and the memory-injection ablation deliberately plants hidden state to confirm the certificate fires when assumption A is violated.

### Expected outcome and causal chain

**vs. WorldLeague per-dimension mean aggregation** — On a task where WorldShift requires memory of a Seek-phase observation that is not carried in the shared interface state, the mean-aggregation baseline produces a high composite score because each per-dimension score is computed independently and the memory gap never surfaces. Our method instead flags a certificate failure at the Seek→Shift boundary because the memory-perturbation sub-check (resetting the policy's recurrent state at s_b) yields non-identical p_Shift outputs. We expect certificate failures concentrated on memory-dependent Shift tasks, with parity on stateless Shape tasks.

**vs. Interactive Score without certificate** — On a long-horizon task where the robot makes large Shape progress but never reaches the Seek target, Interactive Score reports substantial progress because it sums per-dimension increments without checking whether the shared interface state at the Seek→Shape hand-off is sufficient for Shape. Our method produces the same Δ_k values but attaches a failing certificate, so the composite is reported as a flagged partial rather than a clean score. We expect the two methods to agree numerically on Δ_k everywhere but diverge on the certificate label on a minority of boundary instances.

**vs. Physical success rate only** — On a task where the robot completes Seek and Shape but fails Shift, success rate reports zero and discards all partial-progress information. Our method reports a nonzero P(τ) with a passing certificate and a large non-commutativity margin, indicating the goal is order-sensitive rather than unreached. We expect CAPA to separate near-miss trajectories from early-failure trajectories that success rate collapses to the same zero.

**vs. CAPA with truncation-only check (ablation)** — On the synthetic gridworld where the true sufficient statistic is known and hidden memory is the last-seen color, the truncation-only variant returns 'invariant' on memory-dependent boundaries because replaying the same policy reconstructs the same internal memory. The full CAPA with memory perturbation returns 'fail' on those boundaries. We expect the truncation-only variant to have low sensitivity (many false passes) and the full CAPA to have high sensitivity, with specificity measured on stateless boundaries.

**vs. CAPA with memory-injection (ablation)** — On WorldLeague tasks where the policy exposes a recurrent state, we deliberately plant a hidden-state signal at the boundary (e.g., set the recurrent state to a value encoding a spurious observation). The full CAPA certificate should fire (C_b = 0) on these injected boundaries, confirming that the memory-perturbation sub-check is causally responsible for detection.

### What would falsify this idea

If certificate pass rate is near-uniform across all tasks and systems, or if CAPA's composite ranking is identical to per-dimension mean aggregation on every task, the claim that the hand-off certificate detects a distinct failure mode is wrong. A second falsifier: if certificate failures occur uniformly across memory-dependent and stateless tasks rather than concentrating on the predicted memory-dependent Shift subset, or if the synthetic control shows sensitivity no better than the truncation-only ablation, the assumed equivalence between replay-invariance and interface sufficiency is not doing the work we claim.

## References

1. WorldLeague: Benchmarking Physical Agency
