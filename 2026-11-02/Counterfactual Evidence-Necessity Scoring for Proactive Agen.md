# Counterfactual Evidence-Necessity Scoring for Proactive Agent Arenas

## Motivation

Benchmarks for proactive agents, including HALO, award credit whenever an agent's assistance decision matches the human-acceptable verdict, but this success-based rubric never tests whether the retrieved evidence was causally responsible for the decision. HALO's own finding that over-proactivity dominates failures is precisely the symptom of this structural gap: an agent can retrieve irrelevant or post-hoc evidence and still be scored correct if its action level happens to align, so the benchmark cannot distinguish grounded proactivity from lucky guessing. This outcome-correctness-suffices assumption recurs across the ancestor branches (all three treat the acceptability verdict as the terminal signal), and no single branch closed it, leaving a shared residual gap in what arena scores actually certify.

## Key Insight

Requiring the decision to flip when retrieved evidence is masked converts an outcome-correctness rubric into a necessity test, because causal responsibility of evidence — not its co-occurrence with a correct verdict — is what distinguishes grounded proactivity from accidental alignment.

## Method

**A. What it is.** We introduce NEO-HALO (Necessity-Enforced Oracle for HALO), a scoring protocol that wraps HALO-Arena scenarios. For each agent decision it runs a paired counterfactual: the original retrieval-conditioned run and a content-swapped run in which each retrieved evidence span is replaced, in-place, by a semantically unrelated span of the same type, length, and formatting slot (e.g., a DB row replaced by another DB row from a disjoint table; a document span replaced by a span of equal token length from a disjoint document). Necessity credit is awarded only when the assistance decision is withheld or flips under content swap. Inputs: scenario, retrieved evidence set, original decision; output: a per-decision necessity score in {0, 1} plus a continuous swap-margin.

**B. How it works.**
```pseudocode
for each scenario s in HALO-Arena:
    E = agent.retrieve(s)                      # evidence spans, with provenance
    d_orig = agent.decide(s, context=E)        # action level + scope
    for each evidence unit e_i in E:            # leave-one-unit-out content swap
        e_i_swap = sample_unrelated_span(type(e_i), len(e_i))  # same type, length, slot
        E_swap_i = E with e_i replaced by e_i_swap
        d_cf_i = agent.decide(s, context=E_swap_i)
        flip_i = 1[action_level(d_cf_i) < action_level(d_orig)]
    # necessity credit: decision must depend on the content of the evidence it cited
    necessity = 1[mean(flip_i) >= tau]         # tau = 0.5 (majority flip)
    # continuous margin for graded scoring
    margin = mean(flip_i) - tau
    # cross-check: content swap must not degrade task accuracy spuriously
    if necessity == 1 and d_orig == human_acceptable(s):
        award(1)
    else:
        award(0)
```
The protocol runs leave-one-unit-out content swap over evidence units (structured DB rows or document spans), records action-level flips, thresholds at tau=0.5, and gates credit on both necessity and human acceptability. A control condition performs a same-slot content swap on a random non-retrieved span of equal length and type to rule out generic context perturbation. Decoding is fixed at temperature 0 for all runs; to stabilize flip_i under any residual stochasticity, each decision is evaluated with k=3 independent samples and flip_i is the majority flip across samples. The total inference budget is O(|E| * k) extra calls per decision; for HALO-Arena with average |E|=5 and k=3, this is 15 extra calls per decision.

**C. Why this design.** We chose leave-one-unit-out content swap over full-context masking because full masking conflates necessity with generic context starvation (the agent may fail for reasons unrelated to the cited evidence), accepting the cost of O(|E|) extra inference calls per decision. We chose an action-level flip criterion over a verdict-string match because action levels are the ordinal quantity HALO already annotates, accepting that a scope-only change is not credited as a flip. We chose to gate credit on necessity AND human acceptability rather than necessity alone, because a decision could flip under content swap for the wrong reason (e.g., the agent becomes under-active), accepting that the score is conservative and reports necessity rates below raw acceptability. A domain expert would not describe this as a variant of process reward models or SCST-style Monte-Carlo prefix rewards: those score intermediate steps by their marginal contribution to eventual success, whereas NEO-HALO intervenes on the evidence input itself and requires the decision to be causally dependent on it, which no reward-model or retrieval-selection baseline tests.

**D. Why it measures what we claim.** The per-unit flip indicator `flip_i` measures causal necessity of evidence unit e_i because the content-swap intervention holds position, length, and formatting slot fixed, so the only difference between original and counterfactual is the evidential content; this assumption fails when the agent's decision is stochastic or when the swapped span introduces a new semantic cue, in which case `flip_i` reflects content sensitivity rather than evidence necessity. The aggregate `necessity` thresholded at tau=0.5 measures decision-level evidence dependence because a majority of units must individually flip the action level; this fails when evidence units are redundant, in which case each unit may be individually unnecessary while the set is necessary, and the metric under-credits. The same-slot content-swap control condition measures whether flips are attributable to evidence content rather than context perturbation because it matches length, position, and type; it fails when the swapped span happens to be semantically adjacent to the scenario, in which case the control absorbs some true signal. The conjunction with human acceptability ensures the score measures grounded proactivity rather than mere sensitivity; it fails when the human verdict is itself noisy, in which case the score inherits that noise.

## Contribution

(1) A necessity-grounded scoring protocol (NEO-HALO) that replaces HALO's outcome-correctness rubric with a paired counterfactual evidence-removal test, operationalizing causal evidence responsibility as a computable per-decision score. (2) A design principle — necessity-gating with a random-span control — that separates grounded proactivity from accidental action-level alignment, directly targeting the over-proactivity failure mode HALO diagnoses. (3) A re-scoring analysis of HALO-8K responses that quantifies the gap between success-based acceptability and necessity-grounded credit across models.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HALO-Arena scenarios with retrieved evidence | Ground-truth action levels and human acceptability annotations. |
| Primary metric | Necessity-gated acceptability rate | Jointly rewards causal evidence dependence and grounded decisions. |
| Baseline | Raw HALO acceptability | Tests whether necessity adds signal beyond plain acceptability. |
| Baseline | Verdict-string match | Tests whether action-level flips beat surface answer matching. |
| Baseline | Full-context masking | Tests whether leave-one-unit-out avoids conflating necessity with starvation. |
| Baseline | Occlusion-based faithfulness (comprehensiveness/sufficiency) | Tests whether action-level flips add beyond standard occlusion metrics. |
| Ablation-of-ours | NEO-HALO without same-slot content-swap control | Tests whether credit is attributable to evidence content. |
| Ablation-of-ours | NEO-HALO with set-level masking (mask all of E) | Tests redundancy assumption and detects substitutable evidence. |

### Why this setup validates the claim

The central claim is that a decision should earn credit only when it causally depends on the evidence it cited, and that this dependence is best isolated by leave-one-unit-out content swap. Raw HALO acceptability tests the sub-claim that necessity is not redundant with acceptability: if the two rank models identically, necessity adds nothing. Verdict-string match tests the sub-claim that action-level flips capture dependence that surface agreement misses. Full-context masking tests the sub-claim that unit-level intervention is needed to avoid attributing generic context starvation to evidence necessity. Occlusion-based faithfulness baselines test whether the action-level flip criterion adds beyond standard occlusion metrics (comprehensiveness/sufficiency), ensuring we are not re-deriving existing attribution scores. The ablation without the same-slot content-swap control tests whether observed flips are content-driven rather than generic context perturbation. The set-level masking ablation tests the non-redundancy assumption: if masking all of E produces a flip but leave-one-out does not, evidence units are substitutable and the metric under-credits, which we report as a redundancy diagnostic. Necessity-gated acceptability is the right primary metric because it directly operationalizes the conjunction the method asserts — sensitivity to evidence plus human-grounded correctness — so a model that is merely sensitive or merely acceptable cannot score well. Together these choices form a falsifiable contrast: the method should win specifically where cited evidence is load-bearing and lose parity elsewhere.

### Expected outcome and causal chain

**vs. Raw HALO acceptability** — On a case where an agent retrieves the right document but decides to act on a plausible prior rather than the retrieved span, the baseline awards full acceptability because the final action matches the human verdict. Our method instead swaps that span with an unrelated one, observes no action-level flip, and withholds necessity credit because the decision did not depend on the cited evidence. We expect a noticeable gap on scenarios where cited evidence is load-bearing but the prior is strong, with near-parity on scenarios where evidence is trivially necessary.

**vs. Verdict-string match** — On a case where swapping a span changes the action level from proactive to passive but the verdict string still contains the same keyword, the baseline credits the decision because the surface answer matches. Our method instead records the action-level flip and awards necessity credit, because ordinal action change is exactly the signal the baseline discards. We expect our method to separate models that are superficially correct from those genuinely grounded, producing a clear gap on the subset where action level and verdict string diverge.

**vs. Full-context masking** — On a case where removing all context starves the agent into passivity for formatting reasons unrelated to the cited evidence, the baseline counts that flip as necessity and over-credits. Our method instead swaps one unit at a time and requires a majority of units to flip, so a single formatting-cue collapse does not trigger credit. We expect the baseline to inflate necessity rates on long-context scenarios while our method stays stable, yielding a concentrated gap on scenarios with many evidence units.

**vs. Occlusion-based faithfulness (comprehensiveness/sufficiency)** — On a case where standard occlusion metrics assign high comprehensiveness to a span that co-occurs with the correct verdict but is not causally necessary, the baseline credits the span because removing it changes the output distribution. Our method instead requires an action-level flip under content swap, which is a stricter causal test. We expect our method to show lower necessity rates for spurious spans and higher correlation with human-annotated evidence load-bearingness, producing a gap on scenarios with distractor evidence.

**vs. NEO-HALO without same-slot content-swap control** — On a case where a formatting cue (e.g., a list terminator) is removed, the ablation counts the resulting flip as necessity and over-credits. Our method instead uses a same-slot content swap that preserves formatting, so the control and treatment are exchangeable under the null. We expect the ablation to inflate necessity rates on scenarios with structured evidence, yielding a gap that isolates the content-specific effect.

**vs. NEO-HALO with set-level masking (mask all of E)** — On a case where evidence units are redundant (each individually unnecessary but the set necessary), leave-one-out yields zero flips and under-credits, while set-level masking yields a flip. Our method reports both: the primary necessity score uses leave-one-out, and the set-level masking condition serves as a redundancy diagnostic. We expect to observe scenarios where set-level masking flips but leave-one-out does not, quantifying the redundancy rate and justifying the conservative nature of the primary metric.

### What would falsify this idea

If necessity-gated acceptability ranks models identically to raw HALO acceptability across all subsets — no gain where cited evidence is load-bearing — then the causal-dependence signal is redundant and the central claim is wrong. Likewise, if the ablation without the same-slot content-swap control shows the same scores, the flips are generic context perturbation rather than evidence necessity.

## References

1. HALO: Human-Grounded Granular Evaluation of Proactive Agents
