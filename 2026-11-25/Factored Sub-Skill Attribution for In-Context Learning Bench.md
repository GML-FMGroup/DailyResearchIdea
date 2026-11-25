# Factored Sub-Skill Attribution for In-Context Learning Benchmarks via Probe-Conditioned Counterfactual Cell Analysis

## Motivation

Aggregate success-rate benchmarks for in-context learning — exemplified by ICLWorld's reported 82%/76% computer-use vs 27%/26% robot-use gap — collapse a crossed experimental design (3 rule versions × 3 generalization levels) into a single scalar per model, leaving the causal locus of failure unidentifiable. The structural problem recurs across the field: XLand-100B reports that in-context RL baselines 'struggle to generalize to novel and diverse tasks' but its dataset offers no per-factor decomposition, and HELMET shows that aggregate scores over diverse categories exhibit low correlations and unreliable rankings, yet still reports category-level means rather than separable latent sub-skill estimates. Because rule-extraction, rule-application, and transfer are operationally entangled in every existing ICL evaluation, no benchmark can say which sub-skill a given model's failure is attributable to.

## Key Insight

When each crossed cell of the rule-version × generalization-level design is paired with a probe variant that holds all but one sub-skill fixed by construction (rule supplied verbatim, rule supplied programmatically, rule held fixed across instances), the resulting per-cell success matrix admits a contrastive difference-in-differences decomposition whose terms are identifiable sub-skill deficits rather than confounded aggregate rates.

## Method

**What it is.** FALCON (Factored Attribution of Latent sub-skill COmponents) is a diagnostic scoring layer that sits on top of the ICLWorld harness. Its input is a model's full run matrix over (task t, rule-version v, generalization-level g, probe-condition c) cells; its output is a per-model attribution profile: a 3×3 sub-skill success matrix plus a signed decomposition of the digital-vs-physical gap into per-sub-skill contributions with bootstrap confidence intervals.

**Load-bearing assumption (stated explicitly).** FALCON assumes that the exec-only probe — in which the rule is injected programmatically into environment state rather than inferred from context — measures the same rule-application sub-skill as the full ICL condition, i.e., the agent's policy is invariant to the provenance/format of the rule. This is a prompt/format-sensitivity equivalence claim. It is not assumed silently: it is measured and subtracted via the provenance-calibration condition described in Phase 1 and Phase 3.

**How it works.**

```pseudocode
INPUT: tasks T (|T|=160), rule versions V={text, video, counterintuitive-text},
       generalization levels G={replay, new-instance, unseen-env},
       probe conditions C={full, verbatim-recall, verbatim-in-context-exec, exec-only, transfer-fixed}
       model M, ICLWorld harness H, bootstrap reps B=1000

# Phase 1: probe construction (offline, per task)
for t in T:
  for v in V:
    emit probe(t,v,'full')                    # original ICLWorld cell
    emit probe(t,v,'verbatim-recall')         # ask M to restate rule; score exact-match
    emit probe(t,v,'verbatim-in-context-exec')# rule supplied verbatim in context, no inference required;
                                              # agent must act
    emit probe(t,v,'exec-only')               # inject rule programmatically into env state;
                                              # agent must act, not infer
    emit probe(t,v,'transfer-fixed')          # hold rule text constant, swap instance/env

# Phase 2: run collection
S[t,v,g,c] = H.run(M, probe(t,v,c), level=g)  # binary success

# Phase 3: sub-skill estimation (per model, per world w in {digital, physical})
for w:
  # rule extraction: exact-match restatement
  R[w] = mean_{t,v} S[t,v,'replay','verbatim-recall']
  # provenance-calibrated rule application:
  #   raw exec-only score minus the provenance/format effect measured by
  #   the verbatim-in-context-exec control (same task-rule cell, rule present
  #   in context verbatim, no inference).
  A_raw[w] = mean_{t,v} S[t,v,'replay','exec-only']
  P[w]     = mean_{t,v} S[t,v,'replay','verbatim-in-context-exec']
  A[w]     = A_raw[w] - (P[w] - mu_full[w])   # residualized against control;
                                              # mu_full[w] = grand mean over full cells in world w
  # transfer: hold rule text constant, swap instance/env
  X[w] = mean_{t,g} S[t,v,'transfer-fixed',g]
  # residual interaction: full condition minus additive prediction
  for (v,g):
    pred[v,g] = R[w] + A[w] + X[w] - 2*mu[w]  # mu = grand mean over full cells
    resid[v,g] = S_full[v,g] - pred[v,g]

# Phase 4: gap attribution
gap = mean_w=digital(full) - mean_w=physical(full)
contrib_skill = {k: (R_or_A_or_X[digital]-R_or_A_or_X[physical]) for k in {R,A,X}}
contrib_resid = mean resid[digital] - mean resid[physical]
# bootstrap over t (B=1000) for CIs on each contribution
```

Hyperparameters: B=1000 bootstrap resamples over tasks; exact-match scoring for verbatim-recall (no LLM judge); binary success elsewhere inherited from ICLWorld's oracle. Probe conditions are constructed once and reused across all six frontier models. The provenance-calibration control (`verbatim-in-context-exec`) is run on the same task-rule cells as `exec-only` so that the difference P[w] − mu_full[w] is a within-cell estimate of the format/provenance effect.

**Why this design.** We chose *programmatic rule injection* for the exec-only probe over letting the model re-derive the rule because injection is the only way to make rule-application a sufficient statistic for the execution sub-skill, accepting the cost that the agent may behave differently when a rule appears as environment state versus as context. We chose *difference-in-differences against the additive prediction* over a learned disentanglement head because the additive model is a falsifiable null (residuals are directly observable) rather than an unidentifiable latent-variable assumption, accepting the cost that any true sub-skill interaction will be absorbed into the residual rather than attributed to a named skill. We chose *per-task bootstrap over tasks* over per-run bootstrap because tasks are the exchangeable unit in ICLWorld's 160-task design, accepting wider intervals in exchange for correct coverage. We chose to *residualize the exec-only score against a verbatim-in-context-exec control* rather than treat exec-only as a pure execution measure because the provenance/format-sensitivity literature (HELMET; ICLWorld's own digital-vs-physical gap) shows that equivalent task framings can diverge sharply; the control makes the provenance effect an explicit, measured quantity that is subtracted rather than silently absorbed into A. A domain expert would not describe FALCON as a variant of HELMET's category-mean reporting or XLand-100B's baseline-suite comparison: those methods report per-category or per-baseline averages, whereas FALCON's load-bearing quantity is the *residual* of a crossed design under an additive null, which neither prior method computes.

**Why it measures what we claim.** The verbatim-recall score R measures *rule extraction* because exact-match restatement is equivalent to extraction under the assumption that a model that can restate a rule has encoded it from context; this assumption fails when a model can paraphrase a rule it cannot apply, in which case R reflects surface memorization rather than usable extraction. The provenance-calibrated exec-only score A measures *rule application* because programmatic injection removes the inference step and the verbatim-in-context-exec control removes the format/provenance effect under the assumption that the agent treats injected state as equivalent to inferred state after controlling for format; this assumption fails when the agent's policy conditions on the *provenance* of the rule in a way not captured by the verbatim-in-context control, in which case A reflects residual prompt-format sensitivity rather than execution skill. The transfer-fixed score X measures *transfer* because holding the rule constant isolates generalization under the assumption that instance and environment shifts are the only varying factors; this assumption fails when the fixed rule text leaks instance-specific cues, in which case X reflects rule-instance coupling. The residual resid[v,g] measures *unmodeled interaction* under the assumption that sub-skills combine additively in log-odds space; this assumption fails when sub-skills are multiplicatively coupled or when the probability-space link function is mismatched, in which case the residual reflects the coupling or link mismatch rather than noise.

**Contribution.** (1) FALCON, a probe-conditioned factored scoring layer that converts a crossed benchmark design into identifiable per-sub-skill attribution profiles with bootstrap intervals. (2) A falsifiable additive-null model of ICL sub-skills whose residuals are directly observable, providing a design principle for diagnostic benchmark scoring beyond aggregate pass rates. (3) A reusable probe-construction protocol (five conditions per task-rule cell, including a provenance-calibration control) applicable to any ICL benchmark with a rule × generalization design.

## Contribution

(1) FALCON, a probe-conditioned factored scoring layer that decomposes each benchmark run into separable rule-extraction, rule-application, and transfer estimates with bootstrap confidence intervals and a signed gap-attribution profile. (2) A falsifiable additive-null model of ICL sub-skills whose per-cell residuals are directly observable, establishing a design principle for diagnostic benchmark scoring that replaces scalar pass rates. (3) A reusable four-condition probe-construction protocol (full, verbatim-recall, exec-only, transfer-fixed) applicable to any ICL benchmark with a rule × generalization design.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | ICLWorld 160-task crossed design | Native rule × generalization × world structure |
| Primary metric | Digital–physical gap attribution share | Directly targets the causal claim |
| Baseline | ICLWorld aggregate pass-rate ranking | Tests whether factored attribution adds signal |
| Baseline | HELMET category-mean reporting | Tests whether residual beats category averages |
| Baseline | XLand-100B baseline-suite comparison | Tests whether attribution beats suite deltas |
| Ablation-of-ours | Additive-only FALCON (residual dropped) | Isolates contribution of the residual term |
| Ablation-of-ours | Uncalibrated FALCON (exec-only without verbatim-in-context-exec control) | Isolates contribution of the provenance-calibration control |

### Why this setup validates the claim

The central claim is that the digital–physical gap decomposes into identifiable sub-skill contributions plus an observable interaction residual, and that this decomposition is more diagnostic than aggregate pass rates. ICLWorld's crossed (task, rule-version, generalization-level) design is the only source that supplies the exchangeable task structure and the two-world contrast needed to define the gap in the first place, so it is not replaceable. The aggregate pass-rate baseline tests sub-claim one: if FALCON's attribution profile merely re-ranks models the way pass rates do, the decomposition adds nothing. HELMET's category-mean reporting tests sub-claim two: if per-sub-skill means reproduce HELMET-style category averages, the residual is doing no work. XLand-100B's baseline-suite comparison tests sub-claim three: if signed contributions do not localize the gap better than suite-level deltas, attribution is cosmetic. The additive-only ablation is load-bearing: it removes the residual term while keeping the three named skills, so any diagnostic advantage that survives tells us whether the residual or the skill decomposition carries the signal. The uncalibrated-FALCON ablation is load-bearing for the provenance assumption: it removes the verbatim-in-context-exec control while keeping the exec-only probe, so any change in the attribution profile tells us whether the format/provenance effect was materially confounding A. Attribution share is the right primary metric because it is a signed, per-skill quantity that can be wrong in a directional way — a uniform gain across skills is distinguishable from a concentrated one, which aggregate accuracy cannot express.

### Expected outcome and causal chain

**vs. ICLWorld aggregate pass-rate ranking** — On a case where a model scores well on digital replay but collapses on physical unseen-environment tasks, the aggregate baseline reports a single high mean and a single low mean because it averages over all cells and cannot say which sub-skill drove the drop. Our method instead runs the verbatim-recall, verbatim-in-context-exec, exec-only, and transfer-fixed probes on the same task cells and attributes the gap to a specific skill because the crossed design separates extraction, application, and transfer. We expect a large positive contribution from the transfer skill for physical-world tasks, with the extraction contribution near zero — a pattern invisible in the aggregate ranking.

**vs. HELMET category-mean reporting** — On a case where a model restates counterintuitive rules perfectly but fails to act on them, HELMET-style category means report a strong instruction-following category score because the category average pools extraction and application cells. Our method instead contrasts verbatim-recall against provenance-calibrated exec-only within the same task-rule cell because the two probes differ only in whether the rule must be inferred (with the format/provenance effect subtracted via the verbatim-in-context-exec control), so we expect a visible extraction-minus-application gap concentrated in the counterintuitive-text rule version and near parity on text rules.

**vs. XLand-100B baseline-suite comparison** — On a case where two models tie on the suite-level mean but fail on different sub-skills, the suite comparison reports them as equivalent because it compares baseline deltas rather than decomposing them. Our method instead reports signed per-skill contributions with bootstrap intervals over tasks because the additive null makes each contribution a directly observable quantity, so we expect the two models to separate on opposite-signed extraction and transfer contributions despite matching suite means.

**vs. uncalibrated FALCON (exec-only without verbatim-in-context-exec control)** — On a case where the agent's policy conditions on rule provenance, the uncalibrated variant attributes the format effect to the execution sub-skill because it treats exec-only as a pure execution measure. Our method instead subtracts the verbatim-in-context-exec control within the same task-rule cell, so we expect the calibrated A to be lower than the uncalibrated A on cells where provenance sensitivity is high, and the calibrated attribution profile to shift mass from application to the residual or to extraction.

### What would falsify this idea

If the attribution profile's per-skill contributions are uniform across all rule versions and generalization levels — i.e., the digital–physical gap is not concentrated in any named sub-skill and the residual term is indistinguishable from zero — then the additive-null decomposition carries no diagnostic information beyond aggregate pass rates, and the central claim fails. A second falsifier: if the additive-only ablation matches FALCON's attribution exactly on held-out tasks, the residual is noise rather than coupling, and the load-bearing quantity is not doing the work we claim. A third falsifier: if the calibrated and uncalibrated FALCON profiles are identical (i.e., the verbatim-in-context-exec control is indistinguishable from the full condition), the provenance assumption is empirically supported and the calibration step is unnecessary; if they diverge in the direction predicted by format sensitivity, the calibration is doing the work we claim.

## References

1. ICLWorld: Benchmarking In-Context Learning of Agents from Digital to Physical Worlds
