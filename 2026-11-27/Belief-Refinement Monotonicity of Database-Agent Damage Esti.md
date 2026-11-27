# Belief-Refinement Monotonicity of Database-Agent Damage Estimates

## Motivation

Layer B of the database-agent damage study (arxiv:2611.30483) compares six small open-weight models and one 35B MoE across nine engines on hazard-containing tasks, but treats model identity as the treatment variable: it never places models on an ordered scale of how accurately each represents schema and permission state, so the reported damage estimator has no comparability guarantee across models and cannot answer whether better schema/permission understanding weakly lowers expected damage. This is the same structural gap visible in adjacent agent-safety work — ShieldAgent (ICML 2025) and GuardAgent (ICML 2025) both build guard layers that check action trajectories but neither orders the protected agents by belief accuracy, so their measured guardrail accuracy is not tied to any monotonicity property of the guarded agent's world model. Without an ordering, capability and safety are confounded and any cross-model damage comparison is a comparison of unspecified quantities.

## Key Insight

If each model is assigned a belief-refinement rank by a behavioral schema/permission-probe battery whose items are nested supersets, then expected damage is a monotone non-increasing function of rank exactly when the damage estimator's conditioning set is closed under the refinement partial order — a structural closure property that retrieval-style or guard-layer comparisons cannot supply.

## Method

**What it is.** BRMO (Belief-Refinement Monotonicity Oracle) takes as input a set of agent models, the frozen Layer-B hazard task suite, and the schema/permission probe battery, and outputs (i) a belief-refinement partial order over models (reported as a poset, with a verified chain when one exists), (ii) a per-rank expected-damage curve, and (iii) a monotonicity verdict with a bootstrap confidence interval. Its mechanism is a probe-ordered partial order plus a refinement-closed damage estimator, not a learned disentanglement head and not a guard layer.

**Load-bearing assumption (stated explicitly).** BRMO assumes that the set of probe items a model answers exactly correctly, Correct(m,e) ⊆ P, is a valid proxy for that model's belief about schema and permission state, and that the refinement relation m' refines m iff Correct(m,e) ⊇ Correct(m',e) for all engines e is a genuine partial order on the evaluated models. This assumption is checked, not asserted: Phase 2 reports the poset and the fraction of model pairs that are incomparable under ⊆, and the monotonicity test is run only along verified chains (comparable pairs). If the correctness sets are not nested, the contribution is stated as 'damage is monotone along a verified refinement chain' rather than 'damage is monotone in a score'.

**How it works.**

```pseudocode
INPUT: models M = {m_1..m_7}; hazard tasks T (Layer B, 9 engines);
       probe battery P (nested schema/permission items, depth d=5);
       permission arms A = {read, write, ddl, delete, admin}

PHASE 1 — Probe battery construction (nested, not free-form)
  For depth k = 1..5:
    P_k = items asking the agent to report the CURRENT schema
          (tables, columns, types) and the CURRENT permission set
          under permission arm a in A, on engine e.
  Constraint: P_k ⊂ P_{k+1} (each deeper level adds columns/roles,
    never removes).  Each item has a ground-truth answer from the engine
    catalog, so scoring is exact-match, not LLM-judged.
  Battery size: |P| = 5 depths × 5 arms × 9 engines × 12 items = 2,700 items.

PHASE 2 — Belief-refinement partial order (verified, not quantile-binned)
  For each model m and engine e:
    Correct(m,e) = { p ∈ P : answer(m,p) == catalog_truth(p) }   (exact match)
  Define the refinement relation on models:
    m' ⪰ m  iff  Correct(m',e) ⊇ Correct(m,e)  for all engines e.
  Build the poset (M, ⪰).  Report:
    - the Hasse diagram of the poset;
    - the fraction of model pairs that are incomparable under ⊆;
    - every maximal chain C = m_1 ⪰ m_2 ⪰ ... ⪰ m_k (k ≥ 3 required
      for a monotonicity test; if no chain of length ≥ 3 exists, report
      'no verified refinement chain' and stop).
  Rank r(m) is the position of m on the longest verified chain C*
  (r = 1 lowest / coarsest, r = |C*| highest / most refined).
  Models not on C* are excluded from the monotonicity test and reported
  separately with their poset relations.

PHASE 3 — Refinement-closed damage estimation
  For each model m, run the Layer-B hazard tasks under each arm a in A.
  For a trial t, damage D(t) = 1[destructive statement committed]
    × severity_weight(engine, arm)   (severity from Layer-A canonical
    mistake library, scripted, no model).
  Estimator along chain C*:
    D_hat(r) = (1/|T_r|) Σ_{m ∈ C*: rank(m)=r} Σ_{t∈T_r} D(t,m)
  where T_r = ⋂_{m' ∈ C*, rank(m') ≥ r} T(m')  and T(m') is the set of
  tasks whose hazard signature is expressible under the schema/permission
  state that m' correctly reports (i.e., every schema/permission atom in
  the hazard signature is in Correct(m',e)).  T_r is the refinement-closed
  conditioning set: it is the intersection over the chain from rank r
  upward, so T_r ⊇ T_{r+1} and the estimator is a function of the
  refinement class, not of a scalar score.

PHASE 4 — Monotonicity test
  Compute D_hat(1..|C*|). Test H0: D_hat non-increasing in r.
  Statistic: max over adjacent pairs of (D_hat(r+1) - D_hat(r))_+.
  Null distribution: permutation of model→rank assignment within C*
    (10,000 shuffles), 95% CI via percentile bootstrap (1,000 resamples).
  Report per-engine breakdown and per-arm breakdown.
  Power analysis (pre-registered): given per-rank n_r = |T_r| × |{m: rank(m)=r}|
    and the observed per-trial damage variance, compute the minimum
    detectable adjacent-rank increase at 80% power (α=0.05); if any
    n_r < 30, apply stratified sampling over (engine, arm) cells to
    reach n_r ≥ 30 before running.

PHASE 5 — Template-memorization control (tests the catalog-equals-belief assumption)
  Construct a held-out schema variant set P_heldout: same item types as P
    but with table/column/role names and permission sets unseen in the
    models' training data (randomized identifiers, e.g. tbl_a7f3, role_q9x2).
  Re-run Phase 2 on P_heldout.  Report Δ = accuracy(P) - accuracy(P_heldout)
    per model.  If Δ is large (> 0.20) for a model, its Correct set on P
    reflects template memorization rather than catalog belief; flag that
    model and report the monotonicity verdict both with and without it.

OUTPUT: poset (M, ⪰) + Hasse diagram + incomparable-pair fraction;
        verified chain C* and ranks r(m); curve D_hat(r);
        monotonicity verdict + CI; per-engine/per-arm decomposition;
        template-memorization Δ per model and flagged-model sensitivity.
```

**Why this design.** We chose a nested probe battery with exact-match scoring over a free-form capability benchmark because the refinement partial order requires set inclusion of correct items to be verifiable, accepting the cost that the battery cannot probe beliefs outside its fixed schema/permission vocabulary. We chose a directly verified partial order (set inclusion of Correct sets) over a scalar-score-plus-quantile-cut ranking because a higher fraction correct does not imply set inclusion — a model can score higher on a disjoint item subset — so a quantile cut would impose a total order that the data do not support; the cost is that some models may be incomparable and excluded from the chain, reducing the number of ranks. We chose a refinement-closure restriction on the task set over using all hazard tasks because without it a rank-r model could be evaluated on tasks whose hazard signature it cannot even represent, which would break the monotonicity claim for reasons unrelated to belief accuracy; the cost is a smaller effective task set per rank, which we address with the pre-registered power analysis and stratified sampling. We chose a permutation null over a parametric monotonicity model because the damage estimator is a bounded binary-severity sum whose distributional form is unknown, accepting the cost of lower power on small per-rank samples. We added the template-memorization control because the equivalence 'probe score = belief about schema/permission state' fails if agents answer from memorized templates; the control uses unseen schema identifiers to test that equivalence empirically. A domain expert would not describe this as a variant of ShieldAgent (ICML 2025) or GuardAgent (ICML 2025): those methods add an external guard layer that checks trajectories against rules, whereas BRMO adds no guard — it re-orders the evaluated agents themselves and restricts the damage estimator's conditioning set to make the estimator a function of the refinement rank, which is a property of the evaluation design, not of any added module.

**Why it measures what we claim.** The probe score s(m,e) measures schema/permission belief accuracy because exact-match against the engine catalog is equivalent to correct belief about schema and permission state under the assumption that the catalog is the ground truth the agent must act on; this assumption fails when an agent can act correctly without representing the catalog (e.g., memorized task templates), in which case s reflects task familiarity rather than belief — the Phase 5 template-memorization control tests this assumption directly by comparing accuracy on seen vs. unseen schema identifiers. The refinement-closure restriction T_r measures the set of tasks on which rank-r belief is sufficient because a task is included exactly when its hazard signature is expressible under the schema/permission state that all rank-≥r models on the verified chain correctly report (intersection over the chain); this assumption fails when a task's hazard depends on state outside the probe vocabulary, in which case T_r under-covers and D_hat(r) reflects only the probed slice. The monotonicity statistic measures whether better belief weakly lowers expected damage because permutation of rank assignment within the verified chain is exchangeable under the null that damage is independent of belief rank; this assumption fails when rank correlates with an unmeasured capability (e.g., instruction-following), in which case the statistic reflects capability confounding rather than belief refinement.

## Contribution

(1) A refinement-closed damage estimator paired with a nested schema/permission probe battery that assigns each agent an ordinal belief-refinement rank and yields a monotonicity verdict with a permutation null, replacing model identity as the treatment variable. (2) A design principle: damage comparability across agents requires the estimator's conditioning set to be closed under the belief-refinement partial order, otherwise cross-model comparisons are comparisons of unspecified quantities. (3) A reusable probe battery and rank-assignment procedure that can be layered onto the Layer-B task suite of arxiv:2611.30483 without modifying the agent or the engines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Layer-B hazard task suite (9 engines, 9,468 trials) | Frozen, engine-spanning, already hazard-annotated |
| Dataset (transfer) | Held-out DDL/DML hazard suite on the same 9 engines, generated independently of Layer B (n=1,200 trials, 133 per engine) | Tests whether the monotonicity verdict replicates beyond the single frozen suite |
| Primary metric | Rank-ordered expected-damage curve D_hat(r) along the verified refinement chain C* | Directly encodes the monotonicity claim |
| Secondary metric | Fraction of model pairs incomparable under ⊆; template-memorization Δ per model | Reports whether the poset is a chain and whether probe accuracy reflects catalog belief |
| Baseline | Ungated small open-weight agent (Layer B reference) | Tests claim without belief conditioning |
| Baseline | ShieldAgent-style guard layer | Tests whether external rules beat belief refinement |
| Baseline | GuardAgent-style plan-checking guard | Tests trajectory-rule checking alternative |
| Ablation-of-ours | BRMO without refinement-closure restriction (all T) | Isolates the conditioning-set mechanism |
| Ablation-of-ours | BRMO with quantile-binned scalar rank (original Phase 2) instead of verified poset chain | Tests whether the verified partial order is doing the work vs. a score-based total order |

### Why this setup validates the claim

The central claim is that damage falls monotonically with schema/permission belief refinement once the estimator is conditioned on what each rank can actually represent, where refinement is a verified set-inclusion relation on correct probe items rather than a scalar score. The ungated Layer-B reference tests the raw sub-claim that belief accuracy alone (unaided) predicts damage; if BRMO's curve is flat against this reference, the refinement rank carries no signal. The ShieldAgent and GuardAgent baselines test the competing sub-claim that the effect is really about added external checking rather than about re-ordering the agents themselves — if guard layers match or beat BRMO's monotonicity, our mechanism is not doing the work. The no-closure ablation is the decisive control: it holds the models and tasks fixed while removing only the refinement-closure restriction, so any collapse in monotonicity there is attributable to the conditioning set rather than to model quality. The quantile-bin ablation tests whether the verified poset chain is necessary: if the scalar-rank version produces the same monotone curve, the partial-order machinery is not doing the work and the claim reduces to ordinary capability ranking. The held-out transfer suite tests whether the verdict is an artifact of the frozen Layer-B tasks. The rank-ordered damage curve is the right metric because the claim is ordinal and shape-based (non-increasing), not a point estimate; a mean-damage comparison could hide a non-monotone curve behind a favorable average.

### Expected outcome and causal chain

**vs. Ungated small open-weight agent** — On a case where a 4B model under the delete arm drops a column whose dependency it never represented, the ungated baseline commits the destructive statement because nothing in its pipeline forces schema belief to be checked before action. Our method instead places that model at a low refinement rank on the verified chain and restricts its damage estimate to tasks whose hazard signature is expressible under the state it correctly reports, because the refinement-closure restriction (intersection over the chain) excludes tasks it cannot represent. We expect a steep damage drop across the low-to-mid ranks and near-parity at the top rank, i.e. a clearly non-increasing curve rather than a flat one.

**vs. ShieldAgent-style guard layer** — On a case where a hazard requires a multi-step permission escalation across two statements, the guard-layer baseline admits the trajectory because its rule circuits are matched per-action and the escalation is only visible in the composed sequence, a mechanism gap in trajectory-level rule coverage. Our method instead ranks the driving model by its exact-match permission belief on the verified poset and conditions the damage set on that rank, because the rank is a property of the agent's state representation rather than of a checker. We expect BRMO to show monotonicity across all ranks on the verified chain while the guard baseline shows a damage floor at mid ranks where escalation hazards concentrate.

**vs. GuardAgent-style plan-checking guard** — On a case where the safety guard request is under-specified for a DDL hazard on a non-relational engine, the plan-checking baseline generates guardrail code that misses the engine-specific destructive primitive, because its plan is derived from the request text rather than from engine catalog state. Our method instead uses the engine catalog as exact-match ground truth in the probe battery, so a model that misreports the engine's permission set is ranked low on the poset and its damage estimate is restricted accordingly. We expect a visible gap on the non-relational engine subset but parity on relational engines where request text is more complete.

**vs. BRMO without refinement-closure restriction** — On a case where a low-rank model is evaluated on a hazard task whose signature requires schema atoms it does not correctly report, the no-closure ablation includes that task in the low-rank damage estimate, inflating or deflating D_hat(1) for reasons unrelated to belief accuracy. Our method excludes it via the intersection T_r, so the curve is a function of the refinement class. We expect the no-closure ablation to produce a flatter or non-monotone curve, with the gap concentrated at low ranks.

**vs. BRMO with quantile-binned scalar rank** — On a case where two models have equal probe accuracy but disjoint correct-item sets, the quantile-bin version assigns them the same rank and treats them as comparable, while the verified poset reports them as incomparable and excludes them from the chain. Our method's monotonicity test is run only along comparable pairs, so the verdict is not contaminated by a false total order. We expect the quantile-bin ablation to show a weaker or noisier monotonicity signal, with the incomparable-pair fraction (reported in Phase 2) explaining the gap.

### What would falsify this idea

If the refinement-closure ablation produces the same monotone curve as full BRMO, the conditioning set is not doing the work and the claim reduces to ordinary capability ranking. If the quantile-bin ablation matches the verified-poset version, the partial-order machinery is unnecessary; if the template-memorization control shows large Δ for most models, probe accuracy reflects task familiarity rather than catalog belief and the refinement interpretation is unsupported. If the held-out transfer suite fails to replicate the monotonicity verdict, the result is an artifact of the frozen Layer-B tasks.

## References

1. When an AI agent makes a mistake in a database, what decides the damage?
