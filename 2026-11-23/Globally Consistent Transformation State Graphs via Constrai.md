# Globally Consistent Transformation State Graphs via Constraint-Satisfaction Split/Merge Assignment

## Motivation

Training-free transformation-tracking pipelines such as TIPO (arxiv:2611.27398) decide split/merge events per-proposal: they generate candidates from disjoint mask components, score them with mask-decoder overlap priors, and query an MLLM only on the single top-scoring proposal. Because each decision is made in isolation, the resulting transformation state graph has no mechanism that enforces node-identity closure — a confirmed split can leave an orphaned parent while a later merge re-uses the same mask region, silently duplicating or dropping entities. The structural root cause is that global invariants over the whole graph (every entity has exactly one parent/child role per event, no duplicated or orphaned nodes) are never represented as constraints; they are only hoped for as a side effect of local top-1 classification.

## Key Insight

Split/merge acceptance is a joint assignment over all candidate transformation events whose feasible set is exactly the set of state graphs satisfying node-identity closure, so solving the assignment and the graph update as one constrained optimization makes the no-duplicated/no-orphaned invariant a hard property of the output rather than a hoped-for byproduct of per-proposal MLLM voting.

## Method

**What it is.** We introduce GC-TIPO (Globally Consistent Transformation Inference via Proposals and Overlaps), a training-free framework that takes as input a stream of frames with SAM-style instance masks, a candidate split/merge proposal set, mask-decoder overlap priors, and MLLM verification scores, and outputs at each event a *feasible* transformation state graph together with semantic labels for accepted events. Unlike TIPO, no single proposal is ever accepted on its own; acceptance is the solution of a constrained joint assignment.

**Load-bearing assumption (stated explicitly).** The method assumes that the true transformation state graph at each event window is representable within the ILP's feasible set: each physical entity plays at most one parent role and at most one child role per window, no genuine event requires a shared mask region to participate in two accepted events, and every surviving entity either retains mask support or is consumed by an accepted event. Projecting onto this feasible set can therefore only remove TIPO's errors and never removes the correct event. This assumption is load-bearing because the hard constraints are what make node-identity closure a property of the output; if the true graph falls outside the feasible set, the solver defaults to an identity event and suppresses all correct transformations in that window. We make this assumption explicit and add a calibration/verification story (Phase 5) to test it empirically on VISTA.

**How it works.**

```pseudocode
INPUT per event window t:
  masks M_t (SAM-style instance masks)
  proposals P_t = {p_i}  # split/merge candidates from disjoint mask components
  prior s_i in [0,1]    # inner/external overlap score vs mask-decoder priors
  MLLM verification v_i in [0,1] and label l_i (queried for top-K=5 proposals)
  current state graph G_{t-1} = (V, E)  # entities, parent/child roles

PHASE 1 — Constraint construction
  Build binary variables x_i in {0,1} for each proposal p_i in P_t.
  For every entity e in V touched by any p_i, build the role-incidence row:
    sum_{i: e is parent in p_i} x_i  <= 1     (no duplicated parent role)
    sum_{i: e is child  in p_i} x_i  <= 1     (no duplicated child role)
  For every mask region r shared by two proposals p_i, p_j:
    x_i + x_j <= 1                             (mutual exclusion)
  For every entity e that would be orphaned (no role in any accepted event
  and no surviving mask support):
    sum_{i: e gains a role in p_i} x_i >= 1     (no orphaned node)

PHASE 2 — Soft evidence term
  w_i = alpha * s_i + beta * v_i,  alpha=0.4, beta=0.6
  (MLLM score is evidence, not a decision rule)

PHASE 3 — Solve joint assignment (ILP, branch-and-bound)
  maximize  sum_i w_i * x_i
  subject to constraints from Phase 1
  solver: CBC via PuLP, time limit 2s per event window

PHASE 4 — Graph update and labeling
  For each accepted x_i = 1: apply split/merge to G_{t-1},
  attach semantic label l_i from the MLLM query.
  Recompute orphan check; if infeasible, fall back to identity event
  (no transformation accepted) and log a constraint violation.

PHASE 5 — Calibration and verification (offline, on VISTA training split)
  (a) Reliability of w_i: bin proposals by w_i into 10 equal-frequency bins;
      compute empirical event frequency per bin; report reliability diagram
      and AUC of w_i against ground-truth event labels.
  (b) Component AUC: report AUC of s_i alone and v_i alone against ground-truth
      event labels to check monotonicity of the weighted sum.
  (c) Multi-way event rate: count VISTA windows where a ground-truth event has
      >1 parent or >1 child role for a single entity; report fraction of windows.
  (d) Mask-support loss rate: fraction of entity-windows where mask support is
      lost without a ground-truth event (occlusion proxy).

OUTPUT: feasible G_t with closed node-identity invariants.
```

**Why this design.** We chose a hard-constrained ILP over TIPO's greedy top-1 MLLM acceptance because the invariant we care about (no duplicated or orphaned nodes) is a *global* property that no per-proposal rule can enforce — accepting the best-scoring split can still orphan its parent if a later merge is also accepted; the ILP makes this impossible by construction, accepting the cost that infeasibility can force an identity event where TIPO would have committed to *some* transformation. We chose to keep the MLLM score as a *soft* linear term (beta=0.6) rather than as a hard gate because MLLM verification is noisy on occluded frames; the trade-off is that a confidently-wrong MLLM can still bias the objective, which we accept because the hard constraints bound the damage to *which* feasible graph is chosen, not *whether* the graph is feasible. We chose CBC branch-and-bound with a 2s cap over a differentiable relaxation because the event windows are small (typically <30 proposals) and exact feasibility matters more than gradient flow; the trade-off is that we cannot back-propagate through acceptance, which we accept because the framework is training-free by inheritance from TIPO. A domain expert would *not* describe this as a variant of TIPO: TIPO's decision surface is a single scalar per proposal, whereas ours is a feasible-set projection over the joint proposal vector, and the MLLM's role is demoted from judge to feature.

**Why it measures what we claim.** The ILP objective `sum_i w_i x_i` measures *transformation-event plausibility* because we assume the weighted sum of mask-overlap prior and MLLM verification is monotone in the true event probability; this assumption fails when the MLLM is systematically biased toward visually salient but semantically wrong splits (e.g., occlusion mistaken for disassembly), in which case the objective reflects visual salience rather than physical transformation. The constraint rows `sum x_i <= 1` on parent/child roles measure *node-identity closure* because we assume each physical entity participates in at most one parent role and one child role per event window — an assumption that fails when a genuine simultaneous multi-way assembly occurs within a single window, in which case the constraint artificially forbids the true event and the solver defaults to identity. The orphan constraint `sum x_i >= 1` measures *entity persistence* because we assume every entity in G_{t-1} either survives via mask support or is consumed by an accepted event; this fails when mask support is lost to occlusion, in which case the constraint forces a spurious acceptance to avoid orphaning. The infeasibility fallback to identity events measures *conservatism under ambiguity* because we assume an infeasible constraint set indicates ambiguous evidence; this fails when the infeasibility is caused by a single corrupted mask rather than genuine ambiguity, in which case the fallback suppresses a correct transformation. Phase 5 provides the concrete calibration and verification steps to test these assumptions on VISTA: reliability diagrams and AUC for w_i, s_i, and v_i against ground-truth event labels; empirical rate of simultaneous multi-way assemblies; and rate of mask-support loss without a true event.

## Contribution

(1) GC-TIPO, a training-free transformation-tracking framework that replaces TIPO's per-proposal top-1 MLLM acceptance with a joint ILP assignment over split/merge proposals, making node-identity closure a hard output invariant. (2) A constraint formulation (role-incidence rows, mask-region mutual exclusion, orphan-avoidance rows) that operationalizes state-graph closure as a feasible-set projection, together with a demotion of MLLM verification from decision rule to soft evidence term. (3) An analysis of when hard-constraint feasibility and MLLM evidence disagree, yielding an identity-event fallback policy that trades transformation recall for graph integrity.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VISTA (state graphs, assembly/disassembly events) | Only benchmark with ground-truth transformation state graphs |
| Primary metric | State-graph feasibility rate + transformation F1 | Directly measures invariants we claim to enforce |
| Baseline | TIPO (greedy top-1 MLLM acceptance) | The mechanism our joint assignment replaces |
| Baseline | SAM 2/3 tracklet-only tracker | Appearance-robust but identity-static reference |
| Baseline | Failure-triggered disassembly search | Prior work, disassembly-only, search-based |
| Ablation-of-ours | GC-TIPO minus ILP constraints (soft-argmax) | Isolates the global-feasibility claim |
| Calibration study | Reliability diagram + AUC of w_i, s_i, v_i on VISTA train split | Tests monotonicity assumption of w_i |
| Runtime table | Per-window CBC solve time, % hitting 2s cap, % identity fallback | Confirms 2s budget and fallback rate |

### Why this setup validates the claim

The central claim is that enforcing global feasibility constraints over the joint proposal vector yields state graphs that are closed under node identity, and that this closure is what TIPO's per-proposal greedy rule cannot deliver. VISTA is the right testbed because it is the only benchmark that ships ground-truth state graphs with annotated assembly and disassembly events, so feasibility of a predicted graph is checkable rather than proxied. TIPO is the decisive baseline: it shares our proposal set, overlap priors, and MLLM queries, so any gap between it and GC-TIPO is attributable to the decision rule (greedy scalar versus constrained joint assignment) rather than to representation or evidence. The SAM 2/3 tracklet-only baseline tests the orthogonal sub-claim that identity-static trackers cannot update semantic identities through physical change, and the failure-triggered search baseline tests whether our always-on inference beats expensive reactive search. The soft-argmax ablation is the linchpin: it keeps our evidence term and proposal set but drops the constraint rows, so if the gain persists without constraints, the constraints are not doing the work we claim. Transformation F1 detects whether we are simply conservative, while feasibility rate detects whether accepted graphs are internally consistent; together they separate "accepts fewer but correct events" from "accepts the same events more coherently." The calibration study directly tests the load-bearing assumption that w_i is monotone in true event probability by reporting reliability diagrams and AUC of w_i, s_i, and v_i against ground-truth event labels on the VISTA training split; it also reports the empirical rate of simultaneous multi-way assemblies and mask-support loss without a true event, which quantify how often the hard constraints could forbid a correct event. The runtime table reports per-window CBC solve time, the fraction of windows hitting the 2s cap, and the fraction falling back to identity, confirming that the 2s budget is realistic and that the fallback does not dominate results.

### Expected outcome and causal chain

**vs. TIPO (greedy top-1 MLLM acceptance)** — On a window where a high-scoring split and a later high-scoring merge both touch the same parent entity, TIPO accepts both because each proposal is scored in isolation, producing a state graph with a duplicated parent role and an orphaned daughter node. Our method instead rejects the incompatible pair because the role-incidence rows forbid two parent assignments for one entity, so we expect a visible feasibility-rate gap concentrated on multi-event windows, with near-parity on single-event windows.

**vs. SAM 2/3 tracklet-only tracker** — On an assembly where two tracked masks fuse into one, the baseline maintains two tracklets through appearance memory and never creates the merged entity, because its identity model has no mechanism to retire and spawn nodes. Our method instead treats the fusion as a merge proposal, verifies it against overlap priors, and rewrites the state graph, so we expect a large transformation-F1 gap on assembly events and little gap on pure-occlusion events where no true transformation occurs.

**vs. Failure-triggered disassembly search** — On a disassembly that succeeds visually but corrupts the state graph, the baseline never triggers its expensive search because tracking did not fail, leaving the graph stale. Our method instead runs the ILP every window and catches the inconsistency via the orphan constraint, so we expect gains on disassembly events that are visually clean but graph-inconsistent, with parity on cases where tracking actually breaks.

**vs. GC-TIPO minus ILP constraints (soft-argmax)** — On the same multi-event windows, the ablation picks the argmax of the weighted score independently per proposal and reproduces TIPO's duplicated-role errors, because without the constraint rows the objective is separable. Our method instead projects onto the feasible set, so we expect the ablation to recover most of the raw detection recall but lose the feasibility-rate advantage, isolating the constraints as the source of the gain.

**Calibration and runtime expectations** — On the VISTA training split, we expect w_i to show a monotone reliability diagram with AUC > 0.7 against ground-truth event labels, and we expect s_i and v_i to have lower individual AUCs, supporting the weighted sum. We expect the empirical rate of simultaneous multi-way assemblies to be low (e.g., <5% of windows), justifying the degree-1 constraint, and we expect mask-support loss without a true event to be non-negligible (e.g., 10–20% of entity-windows), which would quantify the orphan-constraint risk. For runtime, we expect median CBC solve time <0.5s per window, <5% of windows hitting the 2s cap, and <10% falling back to identity, confirming the 2s budget is realistic.

### What would falsify this idea

If GC-TIPO's improvement over TIPO is uniform across single-event and multi-event windows rather than concentrated on windows where role conflicts or orphaning are possible, then the constraints are not the causal mechanism and the central claim is wrong. Likewise, if the soft-argmax ablation matches GC-TIPO on feasibility rate, the global-assignment story collapses to a scoring change. Additionally, if the calibration study shows that w_i is non-monotone (reliability diagram non-monotone or AUC ≤ 0.5) or that simultaneous multi-way assemblies occur in more than 5% of windows, the load-bearing assumption is violated and the hard constraints would need to be relaxed.

## References

1. Taking Apart and Putting Together: Tracking Objects Through Physical Transformations
