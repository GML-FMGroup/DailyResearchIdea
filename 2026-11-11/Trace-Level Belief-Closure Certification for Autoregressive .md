# Trace-Level Belief-Closure Certification for Autoregressive Language-Model Benchmarks

## Motivation

Existing multitask benchmarks such as MMDM score a single structured decision emitted in one forward pass, and even long-context benchmarks like LongBench v2 reduce correctness to a terminal-answer predicate over one output string. This terminal-predicate design structurally cannot distinguish a model whose reasoning trajectory genuinely commits to the answer from one whose trajectory contradicts itself and then lands on the right token, so no benchmark currently certifies that an autoregressive decision process is belief-closed. The root cause is a representational one: correctness is defined on a terminal string, so there is no object in the harness that could carry trace-level information even if it were collected.

## Key Insight

A model's intermediate commitments are recoverable as answer-distribution snapshots at each reasoning step, so monotone agreement between the step-wise commitment sequence and the terminal answer is a checkable invariant that a terminal-answer predicate simply cannot express, and violations of it are exactly the silent revisions that inflate benchmark scores.

## Method

**What it is.** We introduce BCC (Belief-Closure Certification), a scoring harness that takes a model, a benchmark item, and a step-boundary segmenter, and returns a per-item certificate `(terminal_correct, closure_score, violation_evidence, readable_boundaries)` by probing the model's answer distribution at every reasoning-step boundary and testing whether later commitments contradict earlier ones in a way that the terminal answer conceals.

**Load-bearing assumption (stated explicitly).** The entire certificate rests on the assumption that appending an answer-elicitation suffix to a partial reasoning trace yields a distribution `P_t` whose argmax is the model's *carried-forward commitment* at step `t` — i.e., the suffix is a neutral readout that does not itself cause the model to re-reason or change its belief. This assumption is not assumed globally; it is tested per boundary via the suffix-invariance check in Phase 2b, and boundaries that fail the check are marked unreadable and excluded from revision mass rather than treated as commitments.

**How it works.**

```pseudocode
def BCC(model, item, segmenter, k=8, tau=0.15, alpha=0.05, n_suffixes=2):
    # Phase 1: generate full trace once, segment into reasoning steps
    trace = model.generate(item.prompt, temperature=0.0)
    steps = segmenter.split(trace)            # S_1 .. S_T, step boundaries
    terminal = extract_answer(trace, item.format)

    # Phase 2a: snapshot answer distribution at each step boundary under
    #   n_suffixes=2 semantically equivalent answer-elicitation suffixes
    #   (suffix_A: "Therefore, the answer is"; suffix_B: "Final answer:")
    #   plus a no-suffix continuation baseline (suffix_C: empty string).
    #   Each probe = item.prompt + S_1..S_t + suffix; k=8 samples per probe.
    P_t = {}
    for t in 1..T:
        P_t[t] = [model.distribution(probe(steps[:t], item, suffix=s), k=k)
                  for s in [suffix_A, suffix_B, suffix_C]]

    # Phase 2b: suffix-invariance check — a boundary is *readable* iff
    #   argmax is identical across suffix_A and suffix_B (the two
    #   semantically equivalent elicitations). The no-suffix baseline is
    #   recorded for the neutrality validation experiment but does not
    #   gate readability (it is a continuation, not an elicitation).
    readable = [t for t in 1..T
                if argmax(P_t[t][0]) == argmax(P_t[t][1])]
    unreadable = [t for t in 1..T if t not in readable]

    # Phase 3: commitment sequence and revision detection (readable only)
    c_t = argmax(P_t[t][0]) for t in readable   # suffix_A as primary readout
    support_t = P_t[t][0][c_t]
    revisions = [(t, c_{t-1}, c_t) for consecutive readable t where c_t != c_{t-1}]

    # Phase 4: contradiction test — a revision is a *violation* iff
    #   the pre-revision commitment is later abandoned while the terminal
    #   answer equals the pre-revision commitment's competitor, i.e. the
    #   trace revisits and settles, OR the terminal answer differs from the
    #   majority commitment with no intervening evidence-bearing step.
    violations = []
    for (t, a, b) in revisions:
        # evidence-bearing step: does S_t contain a span that the model
        # itself scores as raising P(b) by > tau under leave-one-step-out?
        delta = P_t[t][0][b] - model.distribution(probe(steps[:t-1] + [S_t removed], item, suffix=suffix_A), k=k)[b]
        if delta < tau and terminal != a and terminal == b:
            violations.append((t, a, b, delta))

    # Phase 5: closure score = 1 - (violation mass / total revision mass)
    #   Revision mass is computed only over readable boundaries; unreadable
    #   boundaries contribute neither to numerator nor denominator.
    rev_mass = sum(abs(P_t[t][0][c_t] - P_{t-1}[0][c_{t-1}]) for revisions)
    viol_mass = sum(abs(P_t[t][0][a] - P_t[t][0][b]) for violations)
    closure = 1.0 if rev_mass == 0 else 1.0 - viol_mass / rev_mass

    # Phase 6: certificate — terminal correctness gated by closure
    certified = terminal_correct and closure >= 1 - alpha
    return Certificate(terminal_correct, closure, certified, violations, readable, unreadable)
```

Step boundaries are produced by a fixed segmenter (newline-delimited reasoning paragraphs, or explicit step markers when present); `k=8` samples are drawn per probe to estimate `P_t` stably, `tau=0.15` is the minimum evidence-bearing probability shift, `alpha=0.05` is the tolerated closure deficit for certification, and `n_suffixes=2` is the number of semantically equivalent elicitation suffixes used for the per-boundary readability check.

**Why this design.** We chose step-boundary distribution snapshots over sampling multiple full traces because a full-trace sample conflates the model's commitment at step `t` with everything that follows it, so it cannot localize a revision; the cost is that probing requires `T` extra forward passes per item. We chose a leave-one-step-out evidence test over a simple "did the answer change" detector because answer flips are legitimate when a step actually carries evidence, and the only way to separate legitimate revision from silent revision is to intervene on the step itself; the cost is that the leave-one-step-out probe is itself a distributional estimate and inherits sampling noise. We chose a ratio-of-masses closure score over a binary violation flag because a single late violation in a 40-step trace should not be scored the same as a trace that thrashes throughout, and mass-weighting preserves that distinction; the cost is that the score is sensitive to how revision mass is normalized. We chose to gate terminal correctness rather than replace it because the benchmark's existing predicate is the community's unit of comparison, and a certified-correctness rate that can be reported alongside raw accuracy is more adoptable than a new metric; the cost is that certified accuracy is strictly lower and requires per-item trace collection. We chose a suffix-invariance check over a single elicitation suffix because the neutrality of the suffix is the load-bearing assumption of the entire certificate, and a per-boundary readability test makes that assumption falsifiable item-by-item rather than assumed globally; the cost is that `n_suffixes=2` doubles the number of forward passes per boundary and some boundaries will be marked unreadable and excluded from revision mass. A domain expert would not describe this as a variant of process reward models or step-level value estimation, because BCC does not learn a value function or reward any step — it checks a consistency invariant between the model's own answer distribution at adjacent step boundaries and its terminal answer, with no training signal.

**Why this measures what we claim.** The quantity `c_t = argmax(P_t)` measures the model's *commitment after step t* because the answer-elicitation suffix removes the model's freedom to keep reasoning, making the elicited distribution a readout of its current answer belief under the assumption that the suffix is a neutral intervention that does not itself change the belief; this assumption fails when the model treats the elicitation suffix as a new prompt and re-reasons, in which case `c_t` reflects a fresh inference rather than a carried commitment — which is why Phase 2b marks such boundaries unreadable when the argmax is not stable across the two semantically equivalent suffixes. The quantity `delta` (leave-one-step-out probability shift) measures whether step `S_t` is *evidence-bearing for the revision* because removing a step that genuinely supports answer `b` should depress `P[b]` under the assumption that the model's answer distribution is causally sensitive to the content of the step that immediately precedes it; this assumption fails when the model's later steps dominate the distribution and mask the contribution of `S_t`, in which case `delta` reflects the redundancy of `S_t` rather than its evidential role. The quantity `closure = 1 - viol_mass/rev_mass` measures *belief closure* because a belief-closed trajectory should exhibit no revision whose probability mass is unsupported by an evidence-bearing step, under the assumption that legitimate multi-step reasoning always produces a detectable probability shift at the step where the evidence arrives; this assumption fails when evidence is gradual and no single step crosses `tau`, in which case `closure` reflects step granularity rather than closure. The gate `certified = terminal_correct and closure >= 1-alpha` measures *certified correctness* because it requires the terminal answer to be both right and reachable without silent revision, under the assumption that silent revisions are the dominant failure mode inflating terminal accuracy; this assumption fails when models reach correct answers through uniformly low-confidence guessing, in which case the certificate reflects trace consistency but not reasoning soundness.

## Contribution

(1) A trace-level correctness predicate and scoring harness, BCC, that converts a terminal-answer benchmark into a per-item belief-closure certificate by probing answer distributions at reasoning-step boundaries and testing leave-one-step-out evidence for each revision. (2) A design principle for benchmark evaluation: certified accuracy (terminal correctness gated by closure) separates models that reach answers through consistent commitment from those that reach them through unsupported revision, a distinction terminal predicates cannot express. (3) An open harness and certificate format that can be layered onto existing multitask benchmarks without modifying their item formats, together with a violation-evidence record that localizes each silent revision to a specific step.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MMDM (16 task categories, multi-format) | Long multi-step decision traces with variable option spaces |
| Primary metric | Certified accuracy (terminal_correct ∧ closure ≥ 0.95) | Gates correctness on belief closure, exposing silent revision |
| Baseline | Raw terminal accuracy | Community unit; ignores revision structure entirely |
| Baseline | Full-trace self-consistency voting | Samples whole traces; cannot localize step-level revisions |
| Baseline | Answer-flip detector (no evidence test) | Flags all revisions, conflating legitimate and silent ones |
| Ablation-of-ours | BCC without leave-one-step-out evidence test | Isolates whether evidence grounding drives the gain |
| Validation (neutrality) | Suffix-invariance readability rate + correlation of suffix_A P_t vs. no-suffix continuation logit | Tests the load-bearing elicitation-suffix neutrality assumption |
| Stress test (gradual evidence) | Synthetic traces with evidence accumulated over 3–5 steps | Quantifies false-violation rate under gradual evidence |

### Why this setup validates the claim

The central claim is that terminal accuracy conceals silent revisions, and that gating correctness on an evidence-grounded closure invariant will separate genuinely sound traces from lucky ones. MMDM is the right substrate because its long, multi-format decision traces (up to 64,817 tokens, option spaces 2–1,024) give many step boundaries where commitments can drift, and its 16 categories let us test whether the predicted failure mode concentrates where we say it does. Raw terminal accuracy is the correct primary contrast: it is exactly the quantity BCC claims is inflated, so a certified-accuracy gap that is larger on revision-prone categories than on short single-hop categories is the falsifiable signal. The full-trace voting baseline tests whether localization matters at all (if whole-trace sampling suffices, BCC's per-step probing is unnecessary). The answer-flip detector tests whether the evidence test is doing real work: if it matches BCC, then "did the answer change" already captures silent revision and the leave-one-step-out machinery is decorative. The ablation confirms the same point from inside our method. The neutrality validation directly tests the load-bearing assumption: if the suffix_A readout does not correlate with the no-suffix continuation logit at readable boundaries, then `c_t` is not measuring a carried commitment and the certificate is invalid regardless of the certified-accuracy gap. The gradual-evidence stress test quantifies the false-violation rate under the known failure mode where evidence accumulates over 3–5 steps and no single step crosses `tau`. If BCC's certified accuracy exceeds all three baselines precisely on the predicted subset, and the neutrality validation shows high readability and correlation, and the gradual-evidence false-violation rate is low, the causal story holds; if the gap is uniform or the neutrality validation fails, the mechanism is wrong.

### Expected outcome and causal chain

**vs. Raw terminal accuracy** — On a case where the model commits to option A at step 4, later drifts to B without any evidence-bearing step, and terminates on B which happens to be correct, raw accuracy scores this as a clean win because its predicate sees only the final token match. Our method instead snapshots `P_t` at each boundary, detects the A→B revision, finds `delta < tau` under leave-one-step-out, and records a violation, so closure drops and the item is not certified. We expect raw accuracy to sit noticeably above certified accuracy overall, with the gap concentrated in the long-trace, high-option-count categories where drift has room to occur, and near-parity on short single-hop items.

**vs. Full-trace self-consistency voting** — On a case where the model's trace thrashes through three candidate answers and lands on the right one, majority voting across full-trace samples reports high confidence because the terminal answer is stable across samples, yet the instability is invisible since each sample conflates commitment with everything after it. Our method instead reads the commitment at each boundary and localizes exactly where the revisions occur, so it can attribute the thrash to specific steps and penalize closure. We expect voting to match raw accuracy closely (it inherits the same blindness) while BCC's certified rate diverges downward on thrashing traces, giving a detectable gap on categories with the most multi-step structure and parity on categories where traces are short.

**vs. Answer-flip detector (no evidence test)** — On a case where a step genuinely introduces decisive evidence and the model legitimately revises from A to B, the flip detector flags this as a violation because it only checks whether the answer changed, wrongly penalizing sound reasoning. Our method instead runs leave-one-step-out: removing that step depresses `P[b]` by more than `tau`, so `delta ≥ tau` marks the revision as evidence-bearing and it is not counted. We expect the flip detector to under-certify correct traces (false violations concentrated on evidence-rich categories) while BCC recovers those items, producing a gap in certified accuracy in BCC's favor that is largest exactly where legitimate mid-trace revisions are common.

**vs. BCC without leave-one-step-out (ablation)** — On a case where a step quietly shifts the commitment with no evidential content, the ablation has no way to distinguish this from a legitimate evidence-driven revision, so it either penalizes both or neither depending on its fallback rule, and closure becomes a noisy proxy for step count. Our method instead grounds each revision in a measured probability shift, so silent revisions are penalized and evidence-bearing ones are spared. We expect the ablation to land between raw accuracy and full BCC, with its certified rate degrading most on categories whose traces mix legitimate and silent revisions, confirming that the evidence test—not the probing alone—drives the separation.

**vs. Suffix-invariance neutrality validation** — On a case where the model treats the elicitation suffix as a new prompt and re-reasons, the argmax of `P_t` under suffix_A and suffix_B will differ, and the boundary is marked unreadable. We expect the readability rate to be high (≥ 0.90) on short single-hop categories and lower on long-trace categories where the model is more likely to re-reason, and we expect the correlation between suffix_A `P_t` and the no-suffix continuation logit at readable boundaries to be positive and substantial (Pearson r ≥ 0.5). If readability is low or correlation is near zero, the load-bearing assumption fails and the certificate is invalid.

**vs. Gradual-evidence stress test** — On synthetic traces where evidence accumulates over 3–5 steps and no single step crosses `tau`, the leave-one-step-out test will mark each revision as a violation because `delta < tau` at every individual step, producing false violations. We expect the false-violation rate to increase with the number of accumulation steps and to be highest when the per-step probability shift is just below `tau`. This quantifies the known failure mode where `closure` reflects step granularity rather than closure.

### What would falsify this idea

If BCC's certified-accuracy gap over raw accuracy is uniform across MMDM categories rather than concentrated in long-trace, high-option-count, revision-prone categories, the claim that silent revision is the dominant inflation mechanism is wrong. Likewise, if the no-evidence-test ablation matches full BCC, then the leave-one-step-out grounding is not doing the causal work and the closure score is merely counting answer flips; and if the suffix-invariance readability rate is low or the correlation with the no-suffix continuation logit is near zero, the load-bearing elicitation-suffix neutrality assumption fails and the certificate is invalid.

## References

1. MMDM: Measuring Massive Multitask Decision Making
