# Prior-Null Falsification: Certifying LLM-Synthesized Principles via Prior-Entailment Closure

## Motivation

Existing LLM-for-science pipelines treat the generator's own output distribution as if it were evidence: Principia aggregates recurring cross-hypothesis relationships into principles without any evidence-generating step, so a principle is accepted whenever the LLM finds it plausible, and Co-Scientist ranks hypotheses through tournament critique that is itself produced by the same model family. LLM-SRBench and ScienceAgentBench score artifacts against fixed ground truth or expert-validated task outputs, which certifies in-distribution fit but never certifies that the recovered artifact is identifiable beyond the generator's prior — a memorized equation or a plausible-sounding principle passes both regimes. The shared missing mechanism is a closure operator that forces the generator to commit to an intervention whose outcome its own prior cannot produce, turning the prior into a null model rather than an oracle.

## Key Insight

If the generator must first commit a distribution over intervention outcomes and acceptance is defined as the observed outcome landing outside that distribution's support, then the generator's own prior becomes the null hypothesis, so acceptance certifies information the generator did not already contain rather than agreement with the generator's beliefs.

## Method

**What it is.** Prior-Null Falsification (PNF) takes an LLM-synthesized principle `p` together with the hypothesis set `H = {h_1..h_n}` that generated it, and returns either a *falsification certificate* `(x*, y*, supp_0)` — a discriminating intervention `x*` whose observed outcome `y*` lies outside the generator's committed prior support `supp_0` — or a *prior-entailment rejection* stating that no such intervention could be constructed. It never returns a fit score.

**Load-bearing assumption (stated explicitly).** PNF assumes that the frozen outcome-support set `PRIOR(x)` faithfully and completely covers the outcomes entailed by the hypotheses in `H`, so that any outcome falling outside `PRIOR(x)` is genuinely unentailed by the generator's prior rather than merely omitted by an under-enumerating prompt. Because acceptance is defined as `y_obs not in PRIOR(x)`, any enumeration omission — not any genuine prior-escape — would manufacture a certificate. PNF therefore does not treat a single enumeration as the prior; it constructs `PRIOR` via an auditable completeness procedure (Phase 1 below) and conditions every ACCEPT on the outcome also being out-of-support for a separately elicited, prompt-varied prior.

**How it works.**

```pseudocode
Input: hypothesis set H (LLM-generated, with stated scope S), synthesized principle p, budget K
Output: ACCEPT with certificate (x*, y*, supp_0) | REJECT(prior-entailed)

# Phase 1 — Prior commitment (must precede any outcome observation)
# Completeness procedure: E = 3 independent, adversarially-prompted enumerations per h
for h in H:                       # each hypothesis is a causal claim over scope S
    for e in 1..3:                # e=1: direct; e=2: "list outcomes you would NOT expect";
                                  # e=3: "list edge-case / boundary outcomes"
        D_h^e <- LLM(h, prompt=enum_prompt[e],
                     "enumerate the outcomes you expect under EVERY
                      admissible intervention in S; assign each a support flag")
    D_h <- union_e D_h^e          # outcome-level support set per hypothesis
PRIOR <- union_h D_h              # outcome-level support set, frozen, hashed
freeze(PRIOR); log hash          # generator can no longer revise its prior
# Held-out prior-prober (separate model, prompt-varied):
PRIOR_probe <- LLM_probe(H, prompt="enumerate expected outcomes under S")
freeze(PRIOR_probe); log hash

# Phase 2 — Principle-implied prediction
D_p <- LLM(p, prompt="for each intervention in S, predict outcome distribution")

# Phase 3 — Discriminating-intervention search (K candidates)
candidates <- LLM(p, PRIOR, prompt="propose interventions x whose D_p outcome
              is HIGH-probability under p but OUTSIDE PRIOR's support")
for x in top-K candidates ranked by  score(x) = D_p(x)[y_hat] * 1[y_hat not in PRIOR(x)]:
    y_hat <- argmax D_p(x)
    if y_hat in PRIOR(x): continue          # not discriminating; skip
    # Phase 4 — Execute the intervention (simulator / held-out data / wet lab)
    y_obs <- execute(x)
    # Dual-prior check: outcome must escape BOTH the primary and the prober prior
    if y_obs not in PRIOR(x) and y_obs not in PRIOR_probe(x):
        return ACCEPT, certificate(x, y_obs, PRIOR(x))
    else:
        record(x, y_obs)                     # prior was right; p gains nothing
return REJECT(prior-entailed)                # no intervention escaped the prior
```

Hyperparameters: `K = 8` candidate interventions per principle; `|H|` capped at 16 hypotheses to keep the prior tractable; `E = 3` independent adversarially-prompted enumerations per hypothesis (direct, negated, edge-case); support flags are binary (in-support / out-of-support) rather than probabilities, because the acceptance rule is set membership, not likelihood. The prior-prober is a separate model instance (different prompt template, same family permitted) whose support set is frozen and hashed independently of `PRIOR`.

**Why this design.** We chose a *frozen, hashed prior commitment* over a post-hoc likelihood comparison because a generator that sees the outcome first can always rationalize it, so the prior must be immutably fixed before execution — accepting the cost that the prior is coarser (binary support) than a full distribution. We chose *intervention search scored by prior-escape* over scoring by agreement with `p` because agreement is exactly what Principia and Co-Scientist already optimize, and a principle that merely agrees with its generator is the failure mode we are trying to detect; the cost is that PNF will reject many true-but-uninformative principles. We chose *set-membership acceptance* over a thresholded probability because set membership is the only criterion that does not require calibrating LLM log-probs (which Kadavath et al. show are miscalibrated for correctness), accepting the cost that a single lucky out-of-support draw is sufficient for acceptance. We chose the *multi-prompt union plus held-out prior-prober* over a single enumeration because a single enumeration is a sparse sample of the prior, not the prior; the cost is `E+1` times the enumeration calls and a stricter ACCEPT rule that will reduce certificate yield. A domain expert would not describe PNF as a variant of Principia: Principia's loop revises hypotheses using observations the LLM itself produced, whereas PNF's acceptance signal is defined by the *complement* of the generator's committed prior, so the generator cannot be the source of its own evidence.

**Why it measures what we claim.** `PRIOR(x)`, the frozen outcome-support set, measures *the generator's own hypothesis prior* because we assume the LLM's multi-prompt enumeration under each `h in H` faithfully covers what that hypothesis entails; this assumption fails when the LLM under-enumerates (omits an outcome it would have predicted even under adversarial prompting), in which case `PRIOR` reflects the LLM's enumeration diligence rather than its actual prior — which is why we measure the enumeration-omission rate against the true equation's support and require the dual-prior check before ACCEPT. The quantity `1[y_hat not in PRIOR(x)]` measures *discriminating power of the intervention* because we assume an outcome outside the union of per-hypothesis supports cannot be produced by any `h in H`; this fails when two hypotheses have correlated blind spots, in which case the score reflects shared training-data gaps rather than genuine discrimination. The final acceptance `y_obs not in PRIOR(x) and y_obs not in PRIOR_probe(x)` measures *external validity of p* because we assume executing `x` yields an outcome the generator did not control; this fails when the executor is itself an LLM from the same family, in which case the certificate reflects executor agreement rather than world-grounded evidence — which is why PNF requires a non-LLM executor (simulator, held-out dataset, or instrument).

## Contribution

(1) A closure operator, Prior-Null Falsification, that maps (hypothesis set, principle) to a discriminating intervention and returns a falsification certificate defined by prior-support escape rather than a fit score, with an immutable prior-commitment protocol that prevents post-hoc rationalization. (2) The design principle that an LLM hypothesis generator's own committed prior can serve as the null model for external validity, converting acceptance from 'the model finds it plausible' into 'the model could not have produced this outcome'. (3) A reproducible acceptance protocol with a frozen-support hash and a non-LLM executor requirement that makes certificates auditable across runs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | LLM-SRBench (239 problems, 4 domains) | Ground-truth equations give non-LLM executability for held-out outcomes |
| Primary metric | Fraction of ACCEPT certificates whose held-out outcome reproduces under the true equation | Set-membership acceptance is the claim, not fit |
| Secondary metric | Enumeration-omission rate: fraction of true-equation-supported outcomes absent from `PRIOR(x)` on a held-out intervention set | Directly bounds the load-bearing assumption |
| Baseline | Principia-style iterative revision | Tests whether post-hoc revision rationalizes outcomes |
| Baseline | Co-Scientist-style agreement scoring | Tests whether optimizing agreement misses prior-escape |
| Baseline | Direct LLM self-consistency falsification | Tests whether an unfrozen prior is exploitable |
| Baseline | Pre-registered prediction-market / conformal OOD baseline | Tests whether PNF is just pre-registration rebranded |
| Ablation-of-ours | PNF with prior unfrozen (revisable) | Isolates the causal role of the frozen hash |
| Ablation-of-ours | PNF with single-shot enumeration (E=1, no prober) | Isolates the completeness procedure |
| Control | Shuffled-principle null: same intervention `x*` re-run against a principle `p'` sampled from a different problem | Estimates lucky-draw false-certificate rate |

**Pre-registration of subsets.** Before running PNF, we stratify problems using an oracle check against the true equation: for each problem, we sample 64 admissible interventions from scope `S`, evaluate the true equation, and label the problem *prior-incomplete* if at least one true outcome is absent from the single-shot `PRIOR` enumeration, else *prior-complete*. Subset definitions are frozen and hashed before any PNF certificate is generated.

### Why this setup validates the claim

The central claim is that a principle is externally valid only when an executed intervention yields an outcome outside the generator's *committed* prior, and that this signal cannot be produced by the generator itself. LLM-SRBench is the right substrate because each problem has a ground-truth equation: we can execute any proposed intervention `x` against the true dynamics and observe `y_obs` without an LLM in the executor role, directly satisfying PNF's non-LLM-executor requirement and closing the "executor agreement" failure mode. The primary metric — the fraction of ACCEPT certificates whose held-out outcome *reproduces* under the true equation — measures exactly what PNF claims to certify (external validity), not fit. The secondary metric — enumeration-omission rate — directly bounds the load-bearing assumption: if the multi-prompt union plus prober leaves a high omission rate, certificates are conditioned on that measured rate rather than asserted complete. Principia tests the sub-claim that iterative revision of hypotheses against self-produced observations cannot distinguish valid from rationalized principles; Co-Scientist tests the sub-claim that agreement-optimization is the failure mode PNF targets; self-consistency falsification tests the sub-claim that an unfrozen prior is exploitable; the pre-registered prediction-market / conformal OOD baseline tests whether PNF's mechanism is distinguishable from pre-registration. The ablation (unfrozen prior) isolates the frozen-hash mechanism: if the hash is doing the causal work, unfreezing should collapse certificate reproduction toward the rationalization baselines while leaving candidate-intervention counts unchanged. The single-shot-enumeration ablation isolates the completeness procedure: if the multi-prompt union and prober are doing work, removing them should raise the omission rate and inflate false certificates. The shuffled-principle null control estimates the lucky-draw rate: if a certificate reproduces equally often when the principle is shuffled, the certificate reflects sampling noise rather than principle informativeness. Together these form a falsifiable test: if PNF's certificates reproduce no more often than Co-Scientist's top principles on the same problems, the mechanism claim is dead regardless of downstream scores.

### Expected outcome and causal chain

**vs. Principia-style iterative revision** — On a problem where the LLM's initial hypothesis set `H` omits a regime (e.g., a saturation term the LLM never enumerated), Principia's loop observes an outcome, then revises the principle to accommodate it, producing a principle that fits but was never at risk of being wrong. Our method instead freezes `PRIOR` before any outcome is observed, so the omitted regime is exactly where a discriminating intervention can escape the prior and yield a certificate. We expect Principia to show high fit but low certificate-reproduction on the pre-registered *prior-incomplete* subset, and near-parity on the *prior-complete* subset — a gap concentrated where the predicted failure mode lives.

**vs. Co-Scientist-style agreement scoring** — On a problem where the true equation and a plausible LLM prior agree on most interventions, agreement scoring ranks the principle highly even though it makes no novel prediction; the generator and the scorer share the same training-data blind spots. Our method instead scores candidates by `1[y_hat not in PRIOR(x)]`, which is zero for any intervention inside the prior regardless of agreement, so it is forced to search the complement. We expect Co-Scientist to produce many high-agreement principles with low reproduction on held-out interventions, while PNF produces fewer ACCEPTs but a higher reproduction fraction — a precision/volume trade-off, not a uniform gain.

**vs. Direct LLM self-consistency falsification** — On a problem with a subtle sign or scaling error, an unfrozen-prior LLM asked to falsify its own principle will generate an outcome, notice it is consistent with its (revised) prior, and declare no falsification; the revision happens silently after the fact. Our method's hashed prior makes post-hoc revision impossible, so the same intervention either escapes the frozen support (certificate) or does not (REJECT). We expect the self-consistency baseline to report near-zero falsifications across all subsets, while PNF reports ACCEPTs concentrated on the *prior-incomplete* subset — the diagnostic signature that the frozen hash, not the LLM's diligence, is doing the work.

**vs. Pre-registered prediction-market / conformal OOD baseline** — A pre-registered prediction market elicits a probability for each intervention outcome and accepts when the observed outcome falls outside a conformal prediction set at level α=0.1. This baseline shares PNF's pre-commitment but lacks the hypothesis-set entailment structure: it treats the generator's probability as the null rather than the union of per-hypothesis supports. We expect the conformal baseline to accept more often on *prior-complete* problems (where its calibrated set is wider) but to reproduce less often on *prior-incomplete* problems, because it has no mechanism to force search into the complement of the hypothesis-entailed support. If PNF's reproduction advantage over conformal is uniform rather than concentrated on *prior-incomplete* problems, the mechanism is pre-registration, not prior-escape.

**vs. Shuffled-principle null control** — For each ACCEPT certificate `(x*, y*)` produced by PNF, we re-execute `x*` and check whether `y*` is also out-of-support for a principle `p'` drawn from a different problem. Under the null, the certificate carries no principle-specific information. We expect the shuffled-null reproduction rate to be near the base rate of out-of-support outcomes on that problem (estimated from the 64-intervention oracle sample), and PNF's reproduction rate to exceed it by a margin concentrated on the *prior-incomplete* subset. If PNF's reproduction rate is statistically indistinguishable from the shuffled null, the certificate reflects lucky draws rather than principle informativeness.

### What would falsify this idea

If PNF's certificate-reproduction fraction is uniform across the pre-registered *prior-incomplete* and *prior-complete* subsets rather than concentrated on the incomplete subset, the mechanism is not prior-escape but something else (e.g., easier problems overall). Equivalently, if the unfrozen-prior ablation reproduces certificates at the same rate as the frozen version, or if the shuffled-principle null reproduces at the same rate as PNF, the frozen commitment and the principle-specific escape are causally inert and the central claim fails.

## References

1. Principia: Test-Time Hypotheses to Principle Construction for Scientific Discovery
2. Accelerating scientific discovery with Co-Scientist
3. LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models
4. ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery
