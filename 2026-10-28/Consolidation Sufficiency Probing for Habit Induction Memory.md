# Consolidation Sufficiency Probing for Habit Induction Memory

## Motivation

HABIT-Bench reveals that even the best memory method reaches only 33.61% macro accuracy on habit induction from longitudinal weak evidence, and performance remains limited even when annotated evidence is provided. This gap persists because memory systems like LightMem's sleep-time consolidation compress raw session evidence into summaries without ever validating whether those summaries preserve the distributed, individually weak cues that habit induction requires. The field treats consolidation as a lossless preprocessing step, but HABIT-Bench's four-choice accuracy cannot detect whether the consolidation mechanism itself is the bottleneck, leaving the sufficiency of consolidated summaries structurally unverified.

## Key Insight

Consolidation sufficiency is identifiable as the gap between habit-inference accuracy from raw evidence and from consolidated summaries, and this gap is attributable to consolidation only when probe questions confirm that the consolidated summary is the sole information channel available to the inference model.

## Method

**What it is.** We introduce **CSP** (Consolidation Sufficiency Probing), an ablation-based evaluation protocol that runs on HABIT-Bench to isolate whether a memory system's consolidation mechanism preserves the distributed weak evidence required for habit induction. CSP takes as input a memory system with a consolidation stage (e.g., LightMem), a set of HABIT-Bench longitudinal histories, and a fixed inference LLM, and outputs a sufficiency score plus a per-habit evidence-retention profile.

**How it works.**

```pseudocode
Input: histories H (HABIT-Bench), memory system M with consolidation C, inference LLM L
Output: sufficiency gap S, retention profile R

1. For each history h in H:
   a. Build two memory channels:
      - RAW(h): all session turns, no consolidation, truncated to 32k tokens with a fixed sliding window (stride 16k tokens); each question q is paired with the window that contains the earliest supporting cue for q, or the first window if no cue is found.
      - CON(h): C(h) — consolidated summaries only
   b. For each habit-inference question q in HABIT-Bench(h):
      - acc_raw(q)  = L(q | RAW(h))
      - acc_con(q)  = L(q | CON(h))
   c. Evidence-existence probe (prior-controlled, forced-choice): for each weak cue c in the habit graph behind q, construct a matched absent foil c' of the same cue type (same domain, same semantic category, but never mentioned in h). Ask L a two-alternative forced-choice probe: "Which of the following is supported by the memory: (A) <c> or (B) <c'>? Answer A or B." Randomize the order of A/B per probe with a fixed seed. Score endorsement as 1 if the probe selects the present cue, 0 if it selects the foil.
      - e_raw(c) = endorsement of c over c' given RAW(h)
      - e_con(c) = endorsement of c over c' given CON(h)
   d. Sufficiency gap for q: S(q) = acc_raw(q) - acc_con(q)
   e. Retention for cue c: R(c) = e_con(c) / max(e_raw(c), eps)
2. Aggregate: S = mean_q S(q); R_profile = {mean R(c) by cue type}
3. Report per-domain (food, finance, software, travel) and per-variant
   (exception, change, unaccepted-suggestion) breakdowns.
```

Hyperparameters: probe format is two-alternative forced-choice with randomized option order (seed=42); eps = 0.05; aggregation is macro-average across domains, matching HABIT-Bench. The probe is designed to be prior-controlled by pairing each present cue with a matched absent foil of the same type, so that a prior-driven 'yes' bias affects both options equally and cancels in the endorsement score.

**Why this design.** We chose a paired-channel ablation (RAW vs CON) over simply comparing consolidation variants because the paired design holds the inference LLM and question set fixed, so any accuracy difference is attributable to the consolidation channel rather than to model capacity or question difficulty, accepting the cost that RAW histories can exceed context windows and must be truncated with a fixed sliding-window budget (we cap at 32k tokens). We chose explicit evidence-existence probes over inferring retention from answer accuracy alone because accuracy conflates "evidence absent" with "evidence present but not used," accepting the cost that probes themselves may be noisy and require the 0.5 threshold calibration we report. We chose per-cue retention ratios over a single scalar sufficiency score because habit induction depends on aggregating many individually weak cues, and a scalar would hide whether consolidation drops specific cue types (e.g., exceptions) while preserving frequent ones, accepting the cost of a richer but harder-to-summarize profile. This differs from MemoryAgentBench and LongMemEval, which evaluate retrieval and long-range understanding but never isolate the consolidation stage; a domain expert would not describe CSP as a variant of those benchmarks because CSP's unit of analysis is the information channel, not the question.

**Why it measures what we claim.** The sufficiency gap S(q) measures consolidation-induced information loss because it assumes the inference LLM is identical across channels and that RAW(h) is a superset of CON(h)'s information; this assumption fails when the LLM's context-window truncation drops raw evidence that consolidation would have preserved, in which case S(q) reflects truncation loss rather than consolidation loss. The retention ratio R(c) measures whether a specific weak cue survives consolidation because it assumes the forced-choice probe is answered from memory content alone and not from the LLM's prior; this assumption fails when the LLM endorses the present cue over the matched foil from world knowledge even when the cue is absent, in which case R(c) reflects prior-driven hallucination rather than retention. The per-cue-type breakdown measures selective evidence loss because it assumes cue types are independently probed and not conflated by the probe wording; this assumption fails when a single probe elicits evidence for multiple cue types, in which case the breakdown reflects probe leakage rather than selective loss.

## Contribution

['A paired-channel ablation protocol (CSP) that isolates the consolidation mechanism in memory systems by holding the inference model and question set fixed while varying only the memory channel.', 'An evidence-existence probing scheme that produces per-cue retention profiles, enabling detection of selective loss of weak evidence types (e.g., exceptions) that aggregate accuracy masks.', 'An empirical analysis on HABIT-Bench quantifying the sufficiency gap between raw evidence and consolidated summaries across four domains and three habit variants.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | HABIT-Bench longitudinal histories | Purpose-built for weak-evidence habit induction |
| Primary metric | Sufficiency gap S (RAW minus CON accuracy) | Directly isolates consolidation-induced loss |
| Baseline | LightMem consolidation | Representative consolidation stage under test |
| Baseline | No-consolidation full-context agent | Upper bound on evidence availability |
| Baseline | MemoryAgentBench retrieval pipeline | Tests whether retrieval, not consolidation, is the gap |
| Ablation-of-ours | CSP without evidence probes (accuracy-only) | Checks whether retention profile adds signal |

### Why this setup validates the claim

The central claim is that consolidation in memory systems silently drops distributed weak evidence, and that this loss is separable from model capacity and question difficulty. The paired RAW/CON design holds the inference LLM and question set fixed, so the sufficiency gap S is attributable to the consolidation channel alone; this is why the primary metric is a gap rather than an absolute accuracy. The LightMem baseline tests the sub-claim that a real, published consolidation mechanism exhibits measurable loss. The no-consolidation full-context baseline tests the sub-claim that the gap is not an artifact of our channel construction: if RAW itself fails, the gap is uninformative. The retrieval baseline tests the competing hypothesis that the deficit is a retrieval problem rather than a consolidation problem, since MemoryAgentBench-style pipelines retrieve from stored content without summarizing. The accuracy-only ablation tests whether the per-cue retention profile R adds diagnostic signal beyond a scalar gap, which is the second half of our claim. Together these form a falsifiable test: the claim predicts a gap concentrated on exception/change/unaccepted-suggestion variants and on rare cue types, not a uniform deficit.

### Expected outcome and causal chain

**vs. LightMem consolidation** — On a history where a user's habit is evidenced by scattered weak cues (e.g., a food preference implied across many sessions, with one exception mentioned once), LightMem's consolidation compresses those turns into summaries that preserve the frequent pattern but drop the single exception mention, so the CON channel answers the exception-variant question wrongly while RAW answers it correctly. Our method makes this visible as a large S(q) on that question and a low R(c) for the exception cue specifically, because the forced-choice probe asks directly whether the exception evidence survives (by contrasting it with a matched absent foil). We expect a noticeable gap concentrated on exception and unaccepted-suggestion variants, with near-parity on the frequent-pattern variants, and a retention profile where exception cues show markedly lower R than frequency cues.

**vs. No-consolidation full-context agent** — On a long history exceeding the 32k truncation budget, the full-context agent drops early-session evidence that consolidation would have preserved, so it fails questions whose supporting cues lie in the truncated region; this is the truncation-loss confound our design must separate from consolidation loss. Our method handles this by reporting S alongside a truncation flag, so questions whose RAW evidence is truncated are excluded from the consolidation-loss estimate rather than inflating it. We expect the full-context baseline to match or beat CON on untruncated histories (confirming RAW is an upper bound) but to degrade on truncated ones, and we expect our gap to remain positive after excluding truncated questions — if it does not, the gap is truncation, not consolidation.

**vs. MemoryAgentBench retrieval pipeline** — On a case where the needed weak cue is stored but not retrieved because the query embedding favors the dominant habit pattern, the retrieval baseline answers from the retrieved frequent-pattern evidence and misses the exception, producing the same wrong answer as consolidation but for a different reason. Our method distinguishes the two by probing evidence existence directly: if the cue is present in CON but the answer is still wrong, the failure is retrieval or inference, not consolidation. We expect the retrieval baseline to show a smaller sufficiency gap than LightMem but a similar per-cue-type bias toward frequent cues, and our retention profile to show high R with low accuracy on those questions — the signature of retained-but-unused evidence rather than dropped evidence.

### What would falsify this idea

If the sufficiency gap S is uniform across cue types and across the exception/change/unaccepted-suggestion variants — rather than concentrated on rare and exception cues with parity on frequent ones — then consolidation is not selectively dropping distributed weak evidence, and the central claim is wrong. Likewise, if the accuracy-only ablation reproduces the full retention profile's diagnostic conclusions, the per-cue probing machinery adds nothing and the method's core contribution fails.

## References

1. HABIT-Bench: Evaluating Habit Induction from Longitudinal Weak Evidence in Agent Memory
2. Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions
3. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory
