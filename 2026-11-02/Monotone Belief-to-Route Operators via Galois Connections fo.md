# Monotone Belief-to-Route Operators via Galois Connections for Routed Embodied Agents

## Motivation

Routed embodied runtimes such as REA store per-policy measured success in route memory and select policies by ranking that memory, but they provide no guarantee that refining the deliberative brain's belief leaves the induced policy ranking consistent — the router can oscillate between policies as beliefs sharpen, producing control chatter exactly when deliberation is most informative. This is a structural gap, not a tuning issue: route memory is a flat success table with no order-theoretic relation to belief states, so any refinement of belief can arbitrarily permute the ranking. The same failure recurs across any brain-cerebellum architecture that routes on cached statistics (REA being the concrete instance), making monotone consistency a meta-level requirement for stable routed control.

## Key Insight

If route memory is lifted from a flat success table to a belief-indexed ranking that is the right adjoint of a refinement map on beliefs, then monotonicity of the router's policy ranking under belief refinement holds by construction of the Galois connection rather than by post-hoc calibration.

## Method

**What it is.** We introduce **MONARCH** (MONotone Adjunction Router with Constrained Hypotheses), a belief-refinement operator that replaces REA's flat route-memory lookup with a pair of order-preserving maps between a belief lattice and a policy-ranking lattice, so that every deliberative update that refines belief can only move the induced ranking upward in a fixed refinement order.

**Load-bearing assumption (stated explicitly).** MONARCH's monotonicity certificate is with respect to the hand-specified atom lattice order `<=_B`, not with respect to the brain's continuous neural belief. The method assumes the three-level atom lattice (task_phase × failure_signature × boundary_flag) is a faithful, sufficiently fine abstraction of the deliberative brain's belief state, such that the adjunction verified on logged route memory transfers to online belief refinements the brain actually emits. This is a simulation-relation assumption (the partition must be stable under the brain's refinement dynamics) and is never assumed away: it is checked at runtime by the abstraction-fidelity gate in Phase 3 step 7a, and inversion rates are reported separately for representable and non-representable refinements.

**How it works.**

```pseudocode
Inputs: belief state b in belief poset (B, <=_B); route memory M: (policy, context) -> success count
Outputs: total preorder R_b on policies Pi, with R_b <=_R R_b' whenever b <=_B b'

# Phase 1: belief lattice construction (offline)
1. Discretize belief into atoms a = (task_phase, failure_signature, boundary_flag)
   - task_phase in {approach, grasp, transport, release} (4 values)
   - failure_signature in {none, slip, collision, occlusion} (4 values)
   - boundary_flag in {0, 1} (2 values)
   => 32 atoms total; lattice depth = 3 (one level per coordinate)
2. Order atoms by refinement: a <=_B a' iff a'.context entails a.context
   (e.g., 'grasp-slip at boundary' <=_B 'grasp-slip')
   - entailment is coordinate-wise: a <=_B a' iff for each coordinate c, a'[c] = a[c] or a[c] = '*'
   - '*' denotes the unrefined (top) value at that coordinate

# Phase 2: Galois connection (offline, from logged route memory)
3. Define alpha: B -> R,  alpha(b) = ranking induced by M restricted to atoms <=_B b
   - ranking induced by a success table S: sort policies by S(policy) descending; ties broken by policy id ascending (fixed, logged)
4. Define gamma: R -> B,  gamma(r) = join of atoms whose induced ranking is <=_R r
   - join computed in the atom lattice under <=_B
5. Verify adjunction alpha(b) <=_R r  iff  b <=_B gamma(r) on held-out logs
   - held-out split: 20% of logged (b, r) pairs, stratified by boundary_flag
   - violation = any pair where exactly one side of the iff holds
   - tolerance: success-rate residual <= 1e-3 counts as equality (absorbs PAVA rounding)
6. If violated, project M onto the largest sub-relation satisfying it
   (isotonic regression over success counts with monotone constraint)
   - PAVA applied per policy across the atom lattice, monotone in <=_B
   - tolerance 1e-3 on success-rate residuals
   - projection is applied only to atoms with >= 5 logged observations; atoms below that threshold inherit the nearest ancestor's projected value and are flagged 'low-support'

# Phase 3: runtime routing (online, per boundary event)
7. On judge-gate trigger, brain emits refined belief b' with b <=_B b'
   7a. Abstraction-fidelity gate: check whether b' is representable as a join of logged atoms
       - representable iff every coordinate of b' is either '*' or a value observed in >= 1 logged atom below b'
       - if representable: proceed to step 8 with b'
       - if not representable: fall back to the nearest logged atom a* <=_B b' (maximal such atom by <=_B), set b' := a*, and set flag unverified := True
       - log (b', representable, unverified, a*) for audit
8. Router computes R_{b'} = alpha(b') via adjunction; ranking is monotone by step 5 on the representable sub-lattice
9. Select argmax policy under R_{b'}; log (b, b', R_b, R_{b'}, tie_break_used) for audit
   - tie_break_used records whether argmax was unique or resolved by policy-id ascending

# Hyperparameters (inline)
- lattice depth: 3 (phase x signature x boundary); sensitivity over depth 2 (phase x boundary) and 4 (phase x signature x boundary x sub-phase)
- isotonic projection: PAVA, tolerance 1e-3 on success-rate residuals
- low-support threshold: 5 logged observations per atom
- judge-gate threshold: boundary-flag only (inherited from REA)
- adjunction held-out split: 20%, stratified by boundary_flag
```

**Why this design.** We chose a Galois connection over a learned ranking head because the adjunction condition `alpha(b) <=_R r iff b <=_B gamma(r)` is a *checkable* structural constraint on logged data, whereas a learned head (the retrieval-style router in REA and in Gorilla/Toolformer-style selectors) offers no such certificate and can permute rankings under refinement — accepting the cost that the belief lattice must be hand-specified and may be coarser than a neural belief. We chose isotonic projection (PAVA) over retraining the success table because it minimally edits route memory to restore monotonicity while preserving measured success counts, accepting the cost that a genuinely non-monotone environment will be forced into a monotone surrogate and lose some ranking fidelity. We chose to trigger refinement only at REA's boundary judge gate rather than continuously, because continuous refinement reintroduces the oscillation the method targets, accepting the cost that failures away from boundaries are not refined. Unlike Toolformer/Gorilla-style retrieval routers, MONARCH's output is constrained by an adjunction rather than by descriptor similarity, so a domain expert would not describe it as a variant of nearest-neighbor policy selection: the load-bearing object is the order relation, not the descriptor. The abstraction-fidelity gate (step 7a) is included precisely because the hand-specified lattice may be coarser than the neural belief; it confines the monotonicity certificate to the representable sub-lattice and makes the transfer assumption testable rather than assumed.

**Why it measures what we claim.** The computational quantity `alpha(b)` measures *the ranking implied by belief b* because the assumption that route-memory success counts restricted to atoms below b are a sufficient statistic for policy quality under b; this assumption fails when success counts are confounded by context drift (e.g., a policy succeeds only in a phase not captured by the atom), in which case `alpha(b)` reflects stale success rather than current belief-conditional quality. The quantity `b <=_B b'` measures *belief refinement* because the assumption that atom entailment is a sound refinement order (a' entails a implies a' is more specific); this assumption fails when atoms are non-comparable but semantically overlapping, in which case refinement is partial and monotonicity is vacuous on that pair. The adjunction check `alpha(b) <=_R r iff b <=_B gamma(r)` measures *monotone consistency of the router* because the assumption that the ranking lattice order is the policy-preference order used at selection time; this assumption fails when tie-breaking at argmax ignores the preorder, in which case the check certifies a ranking the router does not actually follow — the `tie_break_used` log in step 9 is what tests this equivalence, and the representable/unverified flag from step 7a is what separates certified from uncertified refinements when reporting inversion rates.

## Contribution

["MONARCH, a belief-refinement operator that lifts REA's flat route memory into a Galois connection between a belief lattice and a policy-ranking lattice, yielding monotone routing under deliberation by construction.", 'An isotonic-projection procedure (PAVA with monotone constraint) that repairs logged route memory into the largest sub-relation satisfying the adjunction, giving a concrete recipe for retrofitting existing routed runtimes.', 'A design principle — order-theoretic consistency between belief refinement and policy ranking — that replaces per-failure calibration of routing signals with a checkable structural constraint on logged route memory.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | REA boundary-event logs with route-memory success tables | Contains the refinement events MONARCH targets |
| Dataset (transfer) | Tool-selection agent logs with per-tool success tables | Tests transfer beyond REA's hand-specified lattice |
| Primary metric | Ranking-inversion rate across belief refinements, reported separately for representable vs. non-representable refinements (per step 7a) | Directly measures the monotonicity the claim asserts, conditioned on the fidelity gate |
| Secondary metric | Policy-switch frequency per episode and task success rate | Quantifies downstream control chatter |
| Secondary metric | PAVA argmax-change rate and ranking-fidelity loss (Kendall tau vs. raw success table) | Measures projection's cost |
| Baseline | REA flat route-memory router | The system MONARCH replaces |
| Baseline | Gorilla/Toolformer-style descriptor-similarity selector | Tests whether order structure beats similarity matching |
| Baseline | Learned ranking head over belief features (2-layer MLP, hidden=256, GeLU, trained on same logged (b, r) pairs) | Tests adjunction certificate vs. unconstrained learning |
| Ablation-of-ours | MONARCH without isotonic projection (raw success counts) | Isolates the projection's contribution to monotonicity |
| Ablation-of-ours | MONARCH with lattice depth 2 (phase × boundary) and depth 4 (phase × signature × boundary × sub-phase) | Tests whether guarantee is an artifact of coarse lattice |
| Ablation-of-ours | MONARCH with abstraction-fidelity gate disabled (all refinements treated as representable) | Isolates the gate's contribution to reported inversion rate |

### Why this setup validates the claim

The central claim is that constraining routing through a Galois connection between a hand-specified belief lattice and a policy-ranking lattice eliminates order-violating ranking changes under belief refinement, and that this constraint is what produces stable policy selection at boundary events. The primary metric — ranking-inversion rate across refinement steps, i.e., how often the induced preorder moves downward or permutes when belief strictly refines — is precisely the quantity the claim asserts is near-zero for MONARCH and nonzero for unconstrained routers; accuracy alone would not distinguish a router that is accurate-but-inconsistent from one that is accurate-and-monotone. Reporting this metric separately for representable vs. non-representable refinements (per the step-7a gate) makes the load-bearing abstraction-fidelity assumption testable: if MONARCH's inversion rate is near-zero on representable refinements but rises on non-representable ones, the certificate is confined to the sub-lattice where it is established, exactly as the method states. REA's flat router tests the sub-claim that flat memory lookup permits refinement-driven permutation. The descriptor-similarity selector tests the sub-claim that the load-bearing object is the order relation rather than descriptor proximity: if similarity matching already achieves low inversion, the adjunction is unmotivated. The learned ranking head tests whether a checkable structural constraint outperforms an unconstrained learned mapping on the same features. The no-projection ablation isolates whether the isotonic step, not merely the lattice specification, is what restores monotonicity — if removing PAVA leaves inversion rates unchanged, the projection is decorative and the mechanism story is wrong. The depth-2 and depth-4 ablations test whether the guarantee is an artifact of a coarse hand-specified lattice. The PAVA argmax-change rate and Kendall tau against the raw success table quantify the ranking-fidelity cost the method concedes. The tool-selection transfer experiment tests whether the guarantee transfers to a second routed runtime. Policy-switch frequency and task success quantify downstream control chatter beyond inversion rate.

### Expected outcome and causal chain

**vs. REA flat route-memory router** — On a case where a grasp-slip is first observed away from a boundary and later at a boundary, the baseline's flat lookup returns the same policy ranking for both because its memory key does not distinguish the refined atom, so the ranking can invert relative to the earlier decision. Our method instead maps the refined belief through alpha, whose restriction to atoms below the refined atom includes the boundary-specific counts, so the ranking moves only upward in the refinement order. We expect a clear inversion-rate gap concentrated on boundary-refined events, with near-parity on non-boundary events where no refinement occurs, and the gap concentrated on representable refinements (non-representable refinements reported separately and expected to show the baseline's inversion rate, since the gate falls back to the nearest logged atom).

**vs. Gorilla/Toolformer-style descriptor-similarity selector** — On a case where two policies have near-identical descriptors but divergent success under a specific failure signature, the baseline selects by descriptor proximity and can swap its top choice when the belief is refined because the descriptor does not encode the refinement. Our method instead constrains selection by the adjunction, so the refined belief's ranking is bounded below by the coarser belief's ranking. We expect the baseline to show inversions on the descriptor-ambiguous subset while MONARCH shows near-zero inversions there, with both comparable on descriptor-distinct subsets.

**vs. Learned ranking head over belief features** — On a case where training data under-represents a boundary signature, the baseline's learned head extrapolates freely and can place a policy above one it previously ranked lower, with no certificate preventing it. Our method instead inherits monotonicity from the verified adjunction, so the ranking cannot move downward under refinement regardless of coverage. We expect the baseline's inversion rate to rise on the under-represented signatures while MONARCH's stays flat on representable refinements, producing a gap that widens as signature coverage thins; on non-representable refinements MONARCH's gate flags the event and its inversion rate is expected to track the baseline's, which is the falsifiable signature of the abstraction-fidelity assumption.

**vs. MONARCH without isotonic projection** — On atoms with sparse counts, raw success counts can violate the adjunction, so the no-projection variant should show higher inversion rates than full MONARCH on low-support atoms (those below the 5-observation threshold). We expect the PAVA argmax-change rate to be small (few policies change top rank) and Kendall tau against the raw table to be high, indicating the projection restores monotonicity at modest ranking-fidelity cost.

**vs. MONARCH at lattice depth 2 and depth 4** — Depth 2 is coarser and should show lower inversion rate but also lower task success (less refined routing); depth 4 is finer and should show higher inversion rate on non-representable refinements (more atoms to fall outside the logged sub-lattice) but higher task success on representable ones. We expect the depth-3 setting to sit between them on both metrics, indicating the guarantee is not an artifact of a single coarse lattice.

**vs. MONARCH with abstraction-fidelity gate disabled** — With the gate disabled, non-representable refinements are routed through alpha(b') with an empty or stale restriction, so inversion rate should rise to near the baseline's on those events. We expect the gate-enabled variant to show a clear inversion-rate advantage on the non-representable subset, confirming the gate is what confines the certificate.

**Transfer to tool-selection agent** — On tool-selection logs with per-tool success tables, we expect the same pattern: MONARCH's inversion rate near-zero on representable refinements, baseline's nonzero, with policy-switch frequency and task success showing the downstream control-chatter reduction.

### What would falsify this idea

If MONARCH's ranking-inversion rate is statistically indistinguishable from the unconstrained learned head on the boundary-refined representable subset, or if the no-projection ablation shows no increase in inversions on low-support atoms, then the adjunction constraint is not what produces monotonicity and the central claim is wrong. A uniform gain across all subsets rather than concentration where refinement actually occurs would likewise indicate the effect is not mechanism-driven; and if the gate-disabled variant shows no increase in inversion rate on non-representable refinements, the abstraction-fidelity assumption is not load-bearing and the gate is decorative.

## References

1. REA: A Routed Embodied Agent Runtime for Asynchronous Brain-Cerebellum Control
