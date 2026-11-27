# Counterfactual Hazard Lattice: Identifying Model, Guard, and Engine Contributions to Database-Agent Damage

## Motivation

The frontier study on database-agent damage (arXiv:2611.30483) reports Layer A (scripted, model-free mistakes) and Layer B (model-driven agent mistakes) as two separate factorial slices, so the relative contribution of the model, the guard layer, and the database engine to damage remains unidentified: each slice is descriptive, and neither provides a shared estimator or counterfactual contrast. This is not a missing number but a structural gap — because Layer A and Layer B never share a hazard distribution or an intervention lattice, no contrast between them can be interpreted causally, and the field's central question ('what decides the damage?') cannot be answered by accumulating more slices of the same kind. The same meta-level pattern recurs in adjacent agent-safety work: ShieldAgent (venue:icml:2025:7abadac5f324) and GuardAgent (venue:icml:2025:3b41a0b1beb5) evaluate guard layers on their own benchmarks without a shared intervention surface that would let a guard's marginal effect be separated from the model's or the environment's, so guard-layer benefit is reported as a within-benchmark delta rather than an identified component.

## Key Insight

If scripted (model-free) and model-generated mistakes are re-expressed as interventions on the same hazard lattice, then the model's contribution becomes identifiable as a contrast between two arms that share the guard layer and engine — an invariance that separate factorial slices cannot supply.

## Method

**What it is.** CHAL (Counterfactual Hazard Lattice) is an estimator, not a new agent: it takes the union of Layer A's 22 scripted mistakes and Layer B's model-generated mistake traces, re-parameterizes both onto a common intervention lattice with four axes (hazard type h, permission arm p, recovery arm r, engine e), and outputs a joint decomposition of expected damage into engine, guard-layer, and model components with contrast-level uncertainty.

**Load-bearing assumption (stated explicitly).** CHAL's model component is identified under an exclusion restriction: the model m is assumed to affect expected damage only through which canonical hazard type h it selects, not through within-hazard severity. In plain language: once a hazard type is fixed (e.g., an unbounded DELETE), a model's execution of that hazard is assumed to produce the same damage distribution as the scripted canonical mistake. This is a no-direct-effect assumption in a mediation analysis. It is not assumed true; it is bounded. The method therefore reports the model component as an interval whose width is driven by a pre-registered sensitivity parameter for the within-hazard severity path, following the Imai-Keele-Tingley / VanderWeele sensitivity framework, rather than declaring that path unidentified and dropping it.

**How it works.**

```text
INPUT: Layer A cells {(h,p,r,e, damage)} with h in 22 scripted hazards, no model
       Layer B trials {(m,h,p,r,e, damage)} for models m in 6 small open-weight models
       A hazard-canonicalization map: each Layer B trace -> one of the 22 hazard types h
         (matched by SQL/NoSQL operation class + target object class, adjudicated by two annotators,
          kappa target 0.8; unmatched traces go to a residual hazard class h*)

STEP 1  Build lattice L = H x P x R x E, H = 22 canonical + h*, P = 5 permission arms, R = 2 recovery arms,
        E = 9 core engines (secondary 3 engines mapped to nearest core engine, per E1).
        Each cell (h,p,r,e) receives: n_A = scripted replications, n_B(m) = model replications.

STEP 2  Fit a single damage model on the union:
        damage ~ g( h, p, r, e, m_indicator, m_indicator x h )
        where m_indicator = 0 for Layer A (no model) and = m for Layer B.
        g is a fixed-effects log-linear model on damage magnitude with cell-level random intercepts;
        m_indicator enters through a model-by-hazard interaction term (the model is allowed to change
        WHICH hazards are realized) AND through a bounded within-hazard severity term s(m,h)
        constrained by a pre-registered sensitivity parameter lambda in [0,1]:
        |s(m,h)| <= lambda * sd(damage | h, p, r, e, m_indicator=0).
        lambda = 0 recovers the interaction-only restriction; lambda = 1 allows the severity path
        to be as large as the observed scripted within-hazard standard deviation.

STEP 3  G-computation contrast on L (the shared counterfactual):
        (a) Engine component:  E_e[ damage | do(h), do(p), do(r) ] averaged over h,p,r, comparing engines.
        (b) Guard component:   E[ damage | do(p=guarded) ] - E[ damage | do(p=unguarded) ] holding h,r,e fixed.
        (c) Model component:   E[ damage | do(m=m'), h~q_model ] - E[ damage | do(m=0), h~q_model ],
            where q_model is the empirical hazard distribution induced by model m on Layer B tasks
            (this is the only quantity Layer B contributes that Layer A cannot).
            Reported as an interval [L_low, L_high] obtained by sweeping lambda over {0, 0.25, 0.5, 0.75, 1.0}
            and taking the envelope of the resulting point estimates.
        (d) Mediation contrast: fraction of the model component transmitted through hazard-type choice
            q_model vs. through within-hazard damage severity, via a two-stage g-computation.
            The severity-path fraction is reported as a bounded interval under the same lambda sweep.

STEP 4  Uncertainty: nonparametric bootstrap over cells (B=1000) with the canonicalization map held fixed;
        report 95% intervals and a pre-registered falsification check (see verification).
        The lambda sweep is nested inside each bootstrap replicate so that the reported model-component
        interval reflects both sampling uncertainty and sensitivity-parameter uncertainty.

OUTPUT: per-component point estimates + intervals on the shared lattice, plus a per-(h,p,r,e) attribution table.
        The model component is reported as a point estimate at lambda=0 plus a sensitivity interval over lambda.
```

**Why this design.** We chose a single joint damage model over two separately-fit Layer A and Layer B models because separate fits cannot share a counterfactual contrast — the whole point is that the model component must be evaluated against the same hazard distribution the scripted layer samples, accepting the cost that the joint model must assume the canonicalization map is stable across layers (we hold it fixed and report its kappa). We chose g-computation over a learned disentanglement head because the components here are defined by interventions on the lattice (do-operators on p, r, e, and hazard reweighting for m), not by statistical independence — the components are operationally coupled by design, and a learned independence head would be estimating a quantity the design does not produce; the cost is that g-computation requires correct specification of g, which we mitigate with cell-level random intercepts and a pre-registered specification check. We chose to let m enter through a model-by-hazard interaction plus a bounded severity term rather than through a free model main effect because the model cannot change what a scripted hazard destroys once the hazard is fixed — this is the invariant that makes the decomposition identifiable, and the bound on the severity term makes the exclusion restriction explicit and testable rather than assumed; the cost is that the model component is an interval, not a point, and the interval width is governed by lambda. We chose bootstrap intervals over closed-form because the canonicalization step is non-differentiable, accepting the cost of wider intervals. A domain expert would not describe CHAL as a variant of ShieldAgent or GuardAgent: those systems build a guard and report its within-benchmark accuracy, whereas CHAL builds no guard and instead estimates the marginal contribution of a guard arm inside a shared lattice — the deliverable is an attribution, not a shield.

**Why it measures what we claim.** The g-computation engine contrast measures the engine's causal contribution to damage because the do(p), do(r), do(h) interventions hold the guard and hazard fixed across engines, assuming no engine-specific selection into hazard types; this assumption fails when an engine's API surface makes certain hazards unreachable (e.g., no multi-statement transactions), in which case the contrast reflects reachable-hazard coverage rather than engine destructiveness. The guard contrast measures the guard layer's marginal effect because do(p=guarded) vs. do(p=unguarded) is applied within the same (h,r,e) cell, assuming permission arms are exchangeable within a cell; this fails when a guarded arm changes which hazards the agent attempts (behavioral adaptation to the guard), in which case the contrast reflects both blocking and deterrence. The model component measures the model's contribution to damage because it reweights the scripted hazard distribution to the model's empirical hazard distribution q_model while holding the engine and guard fixed, assuming q_model estimated from Layer B trials transports to the scripted lattice and that the within-hazard severity path is bounded by lambda; this fails when Layer B tasks under-sample a hazard class or when the true severity path exceeds the lambda bound, in which case the model component interval reflects the task distribution and the sensitivity bound rather than the model alone. The mediation split measures whether the model acts through hazard choice or within-hazard severity because the two-stage g-computation decomposes the total model contrast into a hazard-reweighting path and a severity path, assuming no unmeasured mediator-outcome confounding between hazard type and damage; this fails when hazard type and severity are jointly determined by an unobserved task feature, in which case the split is descriptive rather than causal and the lambda sweep reports the range of splits consistent with the data.

## Contribution

['A shared counterfactual hazard lattice (CHAL) that re-parameterizes scripted and model-generated database-agent mistakes onto one intervention surface and yields an identified g-computation decomposition of damage into engine, guard-layer, and model components.', "A design principle for agent-safety attribution: the model's contribution is identifiable only through hazard-distribution reweighting against a model-free reference, not through a model main effect, because the model cannot redefine what a fixed hazard destroys.", 'A reusable attribution table and canonicalization protocol (hazard-type map with inter-annotator kappa) that lets future guard-layer work (e.g., ShieldAgent-style shields) report marginal contribution against a shared lattice rather than a private benchmark.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Layer A + Layer B union on shared lattice | Only source with scripted and model hazards on same engines |
| Primary metric | Calibration of component contrasts against held-out cell damage | Tests attribution, not just prediction |
| Baseline | Separate Layer A / Layer B fits | Tests whether joint counterfactual is necessary |
| Baseline | Model main-effect decomposition | Tests the hazard-fixing invariant |
| Baseline | GuardAgent/ShieldAgent-style guard-accuracy reporting | Tests attribution vs. shield framing |
| Baseline | CHAL with lambda=0 (interaction-only restriction) | Tests whether the bounded severity term changes conclusions |
| Ablation-of-ours | CHAL without canonicalization map (raw trace IDs) | Tests whether hazard canonicalization carries the signal |
| Ablation-of-ours | CHAL with h* split into k=3 sub-classes | Tests stability under alternative canonicalizations |
| Sensitivity | Lambda sweep over {0, 0.25, 0.5, 0.75, 1.0} | Reports model-component interval under exclusion-restriction violation |

### Why this setup validates the claim

The central claim is that expected database-agent damage decomposes into engine, guard, and model components on a shared intervention lattice, and that the model component is identifiable as a hazard-reweighting contrast against the scripted hazard distribution once the within-hazard severity path is bounded by a pre-registered sensitivity parameter. Each baseline targets one sub-claim. The separate Layer A / Layer B fits test whether a shared counterfactual is actually needed: if two independent fits reproduce CHAL's component estimates, the joint lattice is unnecessary. The model main-effect decomposition tests the invariant that the model cannot redefine a hazard once fixed; if a free model main effect improves held-out calibration, the interaction-only restriction is wrong. The guard-accuracy baseline tests whether within-benchmark guard accuracy predicts the guard arm's marginal damage contribution; if it does, CHAL's attribution adds nothing over existing guard reporting. The lambda=0 baseline tests whether the bounded severity term is load-bearing: if CHAL at lambda=0 and CHAL at lambda=1 produce overlapping model-component intervals on held-out cells, the exclusion restriction is not doing work and the simpler interaction-only model is preferred. The canonicalization ablation removes the 22-hazard map, testing whether it is the load-bearing structure or whether raw trace identity suffices; the h*-split ablation tests whether the residual class is hiding heterogeneity that would change component estimates. The metric is held-out cell damage calibration because attribution claims are only meaningful if the decomposed components predict damage in cells not used to fit them; a component estimate that does not transfer is an artifact of the fit, not a causal contribution. This combination is falsifiable because each baseline can win on a specific diagnostic subset.

### Expected outcome and causal chain

**vs. Separate Layer A / Layer B fits** — On a case where a model induces a hazard class that Layer A scripts at low frequency, the baseline produces a model component confounded with the hazard distribution because its two fits never share a counterfactual and the model fit inherits Layer B's task sampling. Our method instead reweights the scripted hazard distribution to q_model inside one lattice, so the model contrast is measured against the same hazard set the scripted layer samples. We expect a visible gap on hazard classes where Layer B under-samples relative to Layer A, with near-parity on hazard classes sampled densely in both.

**vs. Model main-effect decomposition** — On a case where a model changes which hazards it attempts but not how much damage a fixed hazard causes, the baseline produces a nonzero model main effect that absorbs hazard-choice variation into a severity term because it lets m enter freely. Our method instead restricts m to a model-by-hazard interaction plus a lambda-bounded severity term, so hazard-choice effects stay in the reweighting path and any within-hazard severity effect is reported as a bounded interval rather than a free parameter. We expect the baseline to over-attribute to within-hazard severity and under-attribute to hazard reweighting, with the discrepancy concentrated on models whose hazard distributions diverge most from the scripted one.

**vs. GuardAgent/ShieldAgent-style guard-accuracy reporting** — On a case where a guarded permission arm blocks a hazard but the agent compensates by attempting a different reachable hazard, the baseline produces high guard accuracy alongside unchanged damage because it scores blocking within its own benchmark rather than marginal damage inside a shared lattice. Our method instead measures the guard contrast within the same (h,r,e) cell, so deterrence and blocking separate. We expect guard accuracy to be high while the measured guard component is attenuated on engines with wide reachable-hazard coverage, and closer agreement on engines with narrow API surfaces.

**vs. CHAL with lambda=0 (interaction-only restriction)** — On a case where a model executes a fixed hazard with systematically larger scope (e.g., a wider DELETE) than the scripted canonical mistake, the lambda=0 baseline attributes the entire damage difference to hazard reweighting because it forbids any within-hazard severity effect. Our method at lambda>0 allows the severity path to absorb part of that difference, so the model component interval widens and the mediation split shifts toward the severity path. We expect the lambda=0 and lambda=1 intervals to overlap on hazards where model and scripted executions are similar in scope, and to separate on hazards where execution-level factors (batching, transaction wrapping, retry behavior) differ, with the separation concentrated on models whose within-hazard damage variance is largest.

**vs. CHAL without canonicalization map (raw trace IDs)** — On a case where two Layer B traces from different models share the same operation class and target object class but differ in surface syntax, the raw-trace baseline treats them as distinct hazards because it has no canonicalization map, so the model component is absorbed into trace identity and the lattice becomes sparse. Our method instead maps both traces to the same canonical hazard h, so the model contrast is estimated within a shared cell. We expect the ablation to produce wider component intervals and to fail the held-out calibration check on hazard classes with high surface-syntax diversity, with near-parity on hazards whose traces are syntactically homogeneous.

**vs. CHAL with h* split into k=3 sub-classes** — On a case where the residual hazard class h* contains heterogeneous traces that differ in damage severity, the unsplit h* baseline may mask a model effect by averaging over sub-classes. Our method with h* split into three sub-classes (by operation class and target object class, adjudicated by the same two annotators) tests whether component estimates are stable under a finer canonicalization. We expect the model-component interval to remain within the bootstrap 95% interval of the unsplit version if h* is homogeneous, and to shift outside it if h* hides a model-specific hazard that the 22 canonical types do not cover.

### What would falsify this idea

If held-out cell damage calibration is no better under the joint lattice than under separate fits, or if the guard-accuracy baseline predicts the guard component as well as CHAL does, or if the lambda=0 and lambda=1 model-component intervals overlap on all held-out cells, then the shared-counterfactual attribution and the bounded severity term are not doing work. If the ablation without canonicalization matches CHAL's component estimates, the hazard map is not load-bearing and the decomposition reduces to trace identity.

## References

1. When an AI agent makes a mistake in a database, what decides the damage?
