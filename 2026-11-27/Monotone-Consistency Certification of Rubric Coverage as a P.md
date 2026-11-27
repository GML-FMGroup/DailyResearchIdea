# Monotone-Consistency Certification of Rubric Coverage as a Proxy for Answer Utility

## Motivation

Rubric-based RAG evaluation, as instantiated by DRBench Pro, treats the satisfaction of knowledge-tree relations as a sufficient proxy for downstream answer utility, inheriting the component-isolated scoring logic of SePer, which quantifies retrieval utility through post-retrieval semantic perplexity reduction without ever testing whether that reduction composes into end-to-end generation quality. The structural root cause is that no existing branch imposes a compositionality invariant tying retrieval-level or rubric-level scores to generation utility, so a rubric can be fully satisfied while the answer remains useless, and the field has no way to detect this failure because coverage and utility are never jointly constrained. This gap recurs across the whole evaluation tree: SePer isolates the retriever, RAGGED isolates reader stability, and DRBench Pro isolates coverage, none of them certifying that the isolated quantity is monotone in the thing users actually care about.

## Key Insight

If rubric satisfaction is genuinely a proxy for answer utility, then the mapping from rubric-coverage vectors to held-out generation utility must be monotone, and fitting that mapping under an explicit isotonic constraint both certifies when the proxy holds and localizes exactly which rubric relations violate it.

## Method

**What it is.** MONA (MONotone certification of rubric proxies) takes as input, for each evaluated agent trajectory, a rubric-coverage vector `c ∈ [0,1]^R` produced by DRBench Pro-style relation judgments plus a held-out generation-utility score `u` from an answer-utility judge on evidence-grounded probe questions derived from the same knowledge-tree node; it outputs a certified monotone mapping `f: c → u`, a per-relation violation report, and a scalar certification score `κ` measuring how far the observed data departs from monotonicity.

**Load-bearing assumption (stated explicitly).** MONA assumes that a held-out probe question built from the same knowledge-tree edge as a rubric relation measures the downstream answer utility attributable to that relation, so that the coverage-to-utility map is a well-defined function of the coverage vector alone rather than of the probe's own multi-hop demands. This is assumption A: the edge is a sufficient and isolated context, so probe performance is attributable to relation r alone. The failure mode F is that cross-edge synthesis or shared evidence makes the probe answer reflect multi-hop reasoning or reader noise rather than r's contribution; when F occurs, U[t] is not a monotone function of c[t], κ collapses toward zero for reasons unrelated to proxy validity, and viol[r] attributes the failure to the wrong relation. To detect F, MONA uses a matched probe pair (single-edge and cross-edge) and certifies monotonicity on the residual utility difference, subtracting out cross-edge synthesis demand, plus a probe-answer evidence-attribution check that flags when F occurs.

**How it works.**

```pseudocode
# Phase 1: probe construction (per knowledge-tree node n)
for relation r in node n's rubric relations:
    probe_q_single[r] = evidence-grounded question built from the SAME tree edge as r
    probe_q_cross[r]  = evidence-grounded question requiring synthesis across the SAME edge and one adjacent edge
    # probes are held out: never shown to the agent, never in the rubric

# Phase 2: utility measurement (per trajectory t)
for each probe_q_single[r] and probe_q_cross[r]:
    a_single = evaluated_agent.generate(probe_q_single[r])   # frozen agent
    a_cross  = evaluated_agent.generate(probe_q_cross[r])    # frozen agent
    u_single[r,t] = judge(a_single, gold_evidence[r])        # answer-utility judge, 1-5
    u_cross[r,t]  = judge(a_cross,  gold_evidence[r])        # answer-utility judge, 1-5
    u[r,t] = u_single[r,t] - u_cross[r,t]                    # residual utility difference
U[t] = mean_r u[r,t]                                         # trajectory utility

# Phase 2b: evidence-attribution check (flags failure mode F)
for each probe answer a_single and a_cross:
    attribution[r,t] = fraction of cited evidence spans that fall on edge r
    flag_F[r,t] = 1 if attribution[r,t] < 0.5 else 0
# report flag_F rate per relation; relations with flag_F rate > 0.2 are marked F-suspect

# Phase 3: monotone-consistency fit (pooled over N trajectories)
# Constraint: if c_i >= c_j componentwise then f(c_i) >= f(c_j)
# Fit isotonic regression of U on c with NO interaction terms:
f = IsotonicRegression(increasing=True, out_of_bounds='clip')  # sklearn, quadratic loss
residuals[t] = U[t] - f(c[t])

# Phase 4: per-relation violation audit
for relation r:
    # partial dependence: hold all other c fixed at median, sweep c[r]
    pd[r] = [f(median_except_r, v) for v in grid(0,1,step=0.05)]
    viol[r] = max(0, max_k(pd[r][k] - pd[r][k+1]))   # downward steps

# Phase 5: certification score
kappa = 1 - (sum_t residuals[t]^2) / (sum_t (U[t]-mean(U))^2)
report: kappa, {viol[r]}, top-k violating relations, flag_F rates
```

Hyperparameters: probe count per node = number of rubric relations (1:1, no sampling); judge = a fixed rubric-grounded LLM judge with 1–5 ordinal scale, temperature 0; isotonic fit uses quadratic loss with no regularization; partial-dependence grid step 0.05; evidence-attribution threshold = 0.5; F-suspect threshold = 0.2.

**Why this design.** We chose a *held-out probe derived from the same tree edge* rather than re-using the rubric question itself, because re-using the rubric would make utility trivially correlated with coverage by construction, accepting the cost that probe construction adds a generation step per relation. We chose *isotonic regression with no interaction terms* over a flexible neural regressor because the whole point is to test whether coverage alone (a low-dimensional sufficient statistic) explains utility, accepting the cost that a genuinely non-monotone but real relationship will register as a violation rather than being absorbed by capacity. We chose a *per-relation partial-dependence audit* over a single global κ because κ alone says the proxy fails but not which relation is responsible, accepting the cost of R separate sweeps. We chose a *matched probe pair* (single-edge and cross-edge) and certify on the residual utility difference, because this subtracts out cross-edge synthesis demand and reader noise, accepting the cost of doubling probe generation. A domain expert would not describe MONA as a variant of SePer or RAGGED: SePer fits a perplexity-reduction metric to human preference and RAGGED fits reader-noise curves, whereas MONA imposes an ordering constraint that is falsifiable — it can output 'the proxy is broken at relation r', which no prior scoring method can produce.

**Why it measures what we claim.** The certification score κ measures *proxy validity* because it equals the fraction of utility variance explained by a monotone function of coverage alone; this assumes coverage is a one-dimensional sufficient statistic for utility, and the assumption fails when two trajectories have equal coverage but different utility due to answer organization, in which case κ reflects residual non-monotone variance rather than a clean certification. The per-relation violation `viol[r]` measures *which rubric relation is a bad proxy* because it is the largest downward step in the fitted partial dependence on c[r]; this assumes the isotonic fit is locally faithful, and the assumption fails when relations are strongly correlated in the sampled trajectories, in which case the violation is attributed to the wrong relation. The probe utility `u[r,t]` measures *downstream answer utility attributable to relation r* because the probe question is built from the same knowledge-tree edge as the rubric relation and the residual difference `u_single - u_cross` subtracts cross-edge synthesis demand; this assumes the edge is a sufficient and isolated context, and the assumption fails when the probe requires cross-edge synthesis, in which case u reflects multi-hop reasoning rather than the relation's contribution. The evidence-attribution check flags this failure mode F when the fraction of cited evidence spans on edge r falls below 0.5.

## Contribution

(1) MONA, a monotone-consistency certification layer that fits an isotonic mapping from rubric-coverage vectors to held-out generation utility and emits a scalar certification score κ plus a per-relation violation audit, turning rubric coverage from an assumed proxy into a testable one. (2) A per-relation violation localization procedure based on partial-dependence sweeps of the fitted isotonic map, which identifies which knowledge-tree relations break the coverage–utility monotonicity rather than only reporting aggregate failure. (3) A held-out probe construction protocol that pairs each rubric relation with an evidence-grounded question from the same tree edge, providing the utility signal that rubric-based evaluation currently lacks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | DRBench Pro knowledge trees (144 cases, 24 domains) | Relation-linked rubrics give coverage vectors plus edge-matched probe sources |
| Primary metric | Certification score κ with per-relation viol[r] audit | Directly quantifies monotone proxy validity, not raw agent quality |
| Baseline | DRBench Pro coverage score alone | Tests whether raw coverage predicts utility without ordering constraint |
| Baseline | SePer-style perplexity-reduction utility ranking | Tests whether a preference-aligned utility metric already encodes monotonicity |
| Baseline | RAGGED-style reader-noise stability sweep | Tests whether stability metrics subsume proxy-validity certification |
| Ablation-of-ours | MONA with rubric-question probes (not held-out) | Tests whether held-out probes are what make κ informative |
| Ablation-of-ours | MONA with single-edge probes only (no matched pair) | Tests whether the residual difference is what removes cross-edge synthesis demand |

### Why this setup validates the claim

The central claim is that rubric-coverage proxies can be *certified* as monotone in downstream utility, and that when they fail, the failure is localized to specific relations. DRBench Pro is the right substrate because it is the only source here whose rubrics are tied to explicit knowledge-tree relations, so a held-out probe can be built from the same edge as each rubric relation — without that edge structure, MONA's probe construction has no anchor. The coverage-only baseline tests the sub-claim that raw coverage is insufficient: if coverage alone already ranks trajectories by utility, κ is uninformative and MONA adds nothing. The SePer-style baseline tests whether an existing preference-aligned utility metric already implies monotonicity, which would make MONA's ordering constraint redundant. The RAGGED-style baseline tests whether reader-noise stability metrics already capture what κ captures. The ablation with rubric-question probes is the load-bearing one: if MONA's κ stays high when probes are the rubric questions themselves, then the held-out probe design — the piece we argued is essential — is doing no work, and the certification is circular. The single-edge-only ablation tests whether the matched probe pair is necessary to subtract cross-edge synthesis demand; if κ is unchanged without the cross-edge probe, the residual difference is not doing work. κ is the right primary metric because it is a variance-explained quantity under a monotone constraint, so a real failure of the proxy shows up as a *drop* in κ concentrated on the relations we predict are bad, not as a diffuse performance change.

### Expected outcome and causal chain

**vs. DRBench Pro coverage alone** — On a case where two trajectories cover the same relations but one organizes them into a coherent causal chain and the other lists them as isolated facts, the coverage baseline assigns near-identical scores because it counts relation hits with no ordering or sufficiency notion. Our method instead fits an isotonic map from coverage to held-out probe utility, so the trajectory whose coverage does not translate into answerable probes lands as a large residual and lowers κ, because the probe is built from the same tree edge and therefore tests whether the covered relation is *usable*. We expect a clear κ gap concentrated on high-coverage/low-utility trajectories, with near-parity on low-coverage ones where both methods agree the proxy is weak.

**vs. SePer-style perplexity-reduction utility** — On a case where a relation is covered verbatim in the report but the probe answer requires the model to *apply* that relation, the SePer-style metric scores high because post-retrieval perplexity drops on the surface text, yet the probe answer is wrong. Our method instead treats the probe utility as the target and asks whether coverage is monotone in it, so the verbatim-but-unusable relation surfaces as a viol[r] spike rather than being absorbed into a favorable perplexity number. We expect the two methods to agree on relations that are both covered and usable, and to diverge sharply on the verbatim-only subset, with MONA flagging those relations as violations.

**vs. RAGGED-style reader-noise stability** — On a case where a reader is robust to noise but the rubric still over-rewards a shallow relation, the RAGGED-style sweep reports stable performance because it varies retrieval depth and measures reader degradation, not proxy validity. Our method instead holds the reader fixed and audits the coverage-to-utility map, so a relation whose partial dependence is non-monotone produces a downward step in viol[r] even when the reader is perfectly stable. We expect MONA to localize violations that the stability sweep reports as flat, with the divergence concentrated on relations where coverage and utility decouple rather than where retrieval depth changes.

**vs. MONA with rubric-question probes** — On a case where the rubric question is reused as the probe, the probe answer is trivially correlated with coverage because the agent was already rewarded for covering that relation. Our method instead uses held-out probes, so κ reflects whether coverage generalizes to answerable questions rather than whether the agent can repeat the rubric. We expect the rubric-question ablation to produce inflated κ (near 1) and near-zero viol[r] across all relations, while the held-out version shows lower κ and localized violations, confirming that the probe design is load-bearing.

**vs. MONA with single-edge probes only** — On a case where the single-edge probe requires cross-edge synthesis, the single-edge-only version attributes multi-hop difficulty to the relation, inflating residuals and misattributing violations. Our method instead uses the residual difference between single-edge and cross-edge probes, subtracting cross-edge synthesis demand. We expect the single-edge-only ablation to show lower κ and a different set of top violating relations than the matched-pair version, with the divergence concentrated on relations whose probes have high cross-edge demand (flag_F rate > 0.2).

### What would falsify this idea

If κ is uniformly high across all relations and the per-relation viol[r] audit finds no relation with a downward partial-dependence step, then coverage is already a monotone sufficient statistic and the certification machinery is solving a problem that does not exist. Equally fatal: if the ablation with rubric-question probes produces κ indistinguishable from the held-out-probe version, the probe design is doing no work and the claimed mechanism is wrong.

## References

1. DRBench Pro: A Knowledge-Centric Benchmark for Professional Deep Research Agents
