# Closure-Verified Request Rewriting for Failure Recovery in Tool-Calling Vision-Language Navigation

## Motivation

Tool-calling VLN agents such as ZeroVLN expose a semantic action interface whose request vocabulary is fixed at design time, so when the action expert rejects or cannot execute a natural-language movement request there is no defined channel for that failure to re-enter the VLM brain's planning state; the brain simply re-samples from the same unconstrained language space with no guarantee the re-issued request is executable or consistent with prior commitments. This is a field-level structural gap rather than a ZeroVLN-specific bug: LH-VLN's long-horizon subtask benchmarks and MGDM-style memory arbitration both presuppose that failures propagate into planning, yet neither defines the algebraic property (closure of the request set under failure substitution) that would make such propagation well-posed, so multi-stage decision consistency remains unverified at the language-request layer.

## Key Insight

If every executable request is tagged with a finite precondition signature and every failure is mapped through a substitution grammar that rewrites only the violated precondition, then the set of re-issuable requests is closed by construction and each rewrite is a minimal, auditable edit whose consistency with stored navigation memory can be checked without re-invoking the action expert.

## Method

**What it is.** Closure-Verified Request Rewriting (CVRR) is a wrapper layer around ZeroVLN's brain–expert tool-calling interface. Input: the brain's proposed natural-language request `r`, the action expert's rejection signal `f` (a failure code plus the expert's observed precondition violation), and the agent's navigation memory `M`. Output: either an accepted request forwarded to the expert, or a rewritten request `r'` drawn from a pre-registered closed vocabulary `V`, together with a memory-consistency verdict.

**Load-bearing assumption (stated explicitly).** CVRR assumes that every action-expert rejection is attributable to exactly one precondition drawn from a fixed five-bit signature `(reachable, clearance, heading_ok, landmark_visible, budget_ok)`, and that for every violated precondition the vocabulary `V` contains at least one template differing from `r` in exactly the clause governing that precondition. This assumption is not asserted as true; it is measured in the experiment (see `### Evaluation Setup`, rows 'Rejection-cause coverage' and 'Single-precondition rate'). If the measured single-precondition rate is low, CVRR's achievable gain is upper-bounded by that rate and the closure claim holds only over the modeled subset of rejections.

**How it works.**

```pseudocode
# Pre-registered once, offline, by mining rejection logs from a pilot ZeroVLN run
# (see 'Vocabulary construction protocol' below)
V = finite set of request templates, each tagged with precondition signature
    sig(v) = (reachable, clearance, heading_ok, landmark_visible, budget_ok)
REJECT_CODES = {UNREACHABLE, BLOCKED, HEADING_MISMATCH, LANDMARK_ABSENT, BUDGET}

# Per navigation step
1. brain proposes r; expert returns ACCEPT or (f, violated_precondition p)
2. if ACCEPT: forward r, append (r, ACCEPT) to M
3. if reject: p = f.precondition
4. SUBSTITUTE(r, p):
     candidates = { v in V : sig(v)[p] = True and v differs from r only in the
                  clause that governs p }        # minimal-edit substitution
     if candidates empty: emit NO_CLOSED_REQUEST -> trigger backtrack
5. MEMORY-ARBITRATE(candidates, M):
     for each v in candidates:
         score(v) = w1 * novelty(v, M)            # w1 = 1.0
                  - w2 * repeat_count(v, M)       # w2 = 2.0
                  + w3 * subgoal_progress(v, M)   # w3 = 1.5
     keep candidates with score >= tau (tau = 0.0)
6. if exactly one survives: issue it; else issue argmax score, log tie
7. append (r, f, r', verdict) to M as a short-term entry; entries older than
   K = 20 steps are compressed into long-term graph nodes (MGDM-style)
```

```
brain --r--> [CVRR] --r'--> action expert
              ^   |
              |   +-- reject (f,p) --> SUBSTITUTE --> MEMORY-ARBITRATE
              |                                    |
              +------- M (short-term buffer + long-term graph) <--+
```

**Vocabulary construction protocol (pre-registration).** `V` is not hand-authored. It is mined from a pilot ZeroVLN run on the VLN-CE train split (no test-split data): (i) log every expert rejection with its failure code and the request string; (ii) cluster logged requests by the clause that differs from the nearest accepted request in the same episode; (iii) for each cluster, register one template per distinct clause edit, tagging it with the five-bit signature computed from the expert's logged state at rejection time; (iv) cap `|V|` at 200 templates (documented in the supplement) to bound effort and keep the closure check tractable. If a rejection's violated precondition is not representable by any template in `V`, the `NO_CLOSED_REQUEST` branch fires and the step is logged as a coverage failure.

**Why this design.** We chose a *pre-registered finite template vocabulary with explicit precondition signatures* over letting the brain freely paraphrase the failed request, because free paraphrase gives no closure guarantee and makes the re-issued request un-auditable, accepting the cost that CVRR cannot express requests outside `V` and must emit NO_CLOSED_REQUEST to fall back to backtracking. We chose *minimal-edit substitution keyed to the single violated precondition* over global re-planning of the whole request, because global re-planning destroys the correspondence between failure cause and edit and thereby destroys auditability, accepting the cost that coupled precondition violations require sequential single edits across steps. We chose a *deterministic weighted memory score with fixed weights and a fixed threshold* over a learned arbitration policy, because a learned policy would reintroduce the very SFT-on-navigation-data that ZeroVLN deliberately avoids and would make the closure claim untestable, accepting the cost that the weights are hand-set and may be suboptimal per environment. A domain expert would not describe CVRR as a variant of retrieval-based tool selection (Toolformer/Gorilla/RAG): those systems select a tool by descriptor similarity from an open library, whereas CVRR *intervenes on the request* under a closure constraint and then arbitrates the intervened request against episodic memory — the object being computed is a minimal substitution, not a nearest neighbor.

**Why it measures what we claim.** `candidates = {v : sig(v)[p]=True, minimal edit}` measures *closure of the request set under failure substitution* because the assumption is that every executable request's precondition set is fully captured by the five-bit signature and that failures are single-precondition; this assumption fails when a rejection is caused by an unmodeled precondition (e.g., a dynamic obstacle absent from the signature), in which case the candidate set reflects only the modeled preconditions and closure is claimed over a strict subset of reality. `score(v) = w1*novelty - w2*repeat + w3*progress` measures *memory-consistent re-issuance* because the assumption is that novelty/repetition/subgoal-progress computed over `M` are sufficient statistics for whether a re-issued request contradicts prior commitments; this assumption fails when memory entries are sparse (early episodes), in which case the score reflects recency rather than consistency. The `NO_CLOSED_REQUEST` trigger measures *genuine inexpressibility in the current vocabulary* because the assumption is that `V` is rich enough that non-emptiness of candidates implies expressibility; this fails when `V` is under-specified, in which case the trigger reflects vocabulary poverty rather than task impossibility. The per-rejection diagnostic (substitution-acceptance rate vs. arbitration-choice quality) separates these two failure modes empirically: substitution-acceptance rate is the fraction of rejections for which `candidates` is non-empty and the issued `r'` is accepted by the expert; arbitration-choice quality is the fraction of multi-candidate steps where the issued `r'` is the one that maximizes subgoal progress in hindsight.

## Contribution

(1) A closure-verified request-rewriting layer (CVRR) that equips a tool-calling VLN interface with a finite, precondition-tagged request vocabulary and a minimal-edit substitution grammar, making the set of re-issuable requests closed under single-precondition failure by construction. (2) A memory-arbitration rule that scores candidate rewrites against MGDM-style short-term/long-term navigation memory with fixed, auditable weights, yielding a deterministic and inspectable failure-recovery decision rather than a re-sampled paraphrase. (3) An audit trace format (request, failure code, rewrite, verdict) that makes failure propagation into the VLM brain's planning state observable and replayable on VLN-CE episodes.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | VLN-CE (continuous environments) | ZeroVLN's native benchmark; continuous expert rejections exist |
| Primary metric | Success Rate (SR) on rejection-triggering episodes | Isolates the population CVRR actually intervenes on |
| Secondary metric | Rejection-cause coverage rate (fraction of logged rejections whose violated precondition is representable in the five-bit signature) | Makes the load-bearing assumption an empirical result |
| Secondary metric | Single-precondition rate (fraction of logged rejections attributable to exactly one signature bit) | Upper-bounds CVRR's achievable gain |
| Secondary metric | Substitution-acceptance rate (fraction of rejections where a non-empty candidate set yields an expert-accepted `r'`) | Separates substitution quality from arbitration quality |
| Baseline | ZeroVLN (unmodified brain–expert interface) | Tests whether closure-verified rewriting beats raw tool-calling |
| Baseline | ZeroVLN + free-paraphrase retry | Tests closure/auditability vs unconstrained re-issuance |
| Baseline | SFT-based hierarchical VLN method | Tests whether CVRR closes the SFT gap without SFT |
| Baseline | Constrained-decoding retry (same grammar as `V`, no memory arbitration) | Isolates closure guarantee from template substitution alone |
| Ablation-of-ours | CVRR minus MEMORY-ARBITRATE (issue first candidate) | Isolates memory-consistency contribution from substitution |
| Ablation-of-ours | CVRR with `V` mined from pilot logs vs. hand-authored `V` | Tests whether the construction protocol matters |

### Why this setup validates the claim

The central claim is that a finite, precondition-signature-indexed substitution vocabulary, arbitrated by episodic memory, converts action-expert rejections into executable re-issued requests — recovering navigation progress that raw tool-calling loses — without any navigation-domain SFT of the VLM brain. VLN-CE is the right arena because ZeroVLN already defines the brain–expert interface there and the continuous action expert produces genuine precondition rejections, so the intervention population is real rather than synthetic. The unmodified ZeroVLN baseline tests the sub-claim that rejections are a recoverable loss, not noise: if CVRR beats it, the gain must come from acting on rejections. The free-paraphrase baseline tests the closure sub-claim: it can issue any request, so if it does *not* outperform CVRR, the finite vocabulary plus auditability is doing the work rather than expressive freedom. The constrained-decoding baseline tests whether the closure guarantee itself (not just template substitution) is load-bearing: it enforces the same grammar as `V` but has no memory arbitration, so any CVRR advantage over it isolates the arbitration step. The SFT-based baseline tests the no-SFT sub-claim: if CVRR approaches it, the wrapper substitutes for navigation-domain adaptation. The ablation isolates memory arbitration from substitution: if removing it collapses the gain, the score is load-bearing rather than decorative. The rejection-cause coverage and single-precondition rate metrics make the load-bearing assumption an empirical result: if coverage is low, CVRR's gain is upper-bounded by the covered fraction and the paper reports that bound rather than claiming full closure. SR on rejection-triggering episodes is the right primary metric because it is the only slice where the mechanism can plausibly act; overall SR would dilute the predicted effect against episodes where no rejection ever occurs.

### Expected outcome and causal chain

**vs. ZeroVLN (unmodified)** — On a case where the brain requests a waypoint through a doorway the expert flags as BLOCKED, unmodified ZeroVLN forwards the rejected request, receives no motion, and either stalls or re-proposes the same geometric goal because the brain has no signal distinguishing *why* execution failed. Our method reads the violated precondition, substitutes the minimal clause governing clearance from `V`, and issues a request the expert can execute, so we expect a noticeable SR gap concentrated on rejection-triggering episodes with near-parity on episodes that never reject. The measured single-precondition rate sets the upper bound on this gap: if only a fraction of rejections are single-precondition, the gap is capped at that fraction.

**vs. ZeroVLN + free-paraphrase retry** — On a case where the brain paraphrases a HEADING_MISMATCH failure into a longer instruction, the free-paraphrase baseline can drift into requests whose preconditions are unverifiable and re-issues near-duplicates of the failed request, inflating repeat loops without closure. Our method restricts edits to the clause governing the violated precondition and scores candidates against memory, so we expect CVRR to show fewer repeated-request cycles and a modest SR edge, with the gap widening as episode length grows.

**vs. Constrained-decoding retry** — On a case where the brain re-issues a request that satisfies the grammar but repeats a prior commitment already logged in `M`, the constrained-decoding baseline has no mechanism to penalize the repeat and will loop. Our method's memory score penalizes repeats (`w2 = 2.0`) and rewards subgoal progress (`w3 = 1.5`), so we expect CVRR to show a higher substitution-acceptance rate and fewer repeat cycles, isolating the arbitration step's contribution from the closure grammar alone.

**vs. SFT-based hierarchical VLN** — On a case where the SFT baseline's learned action head predicts a geometric goal that is locally plausible but contradicts the instruction's landmark ordering, the SFT baseline executes confidently and fails late in the episode because its adaptation traded general visual grounding for navigation-domain fitting. Our method keeps the brain's general capabilities and intervenes only at rejection points, so we expect CVRR to trail SFT slightly on easy in-distribution episodes but close or exceed it on long-horizon and OOD-style episodes where confident-but-wrong geometric prediction dominates.

### What would falsify this idea

The idea is falsified if CVRR's SR gain is uniform across rejection-triggering and non-rejection episodes rather than concentrated where the predicted failure mode occurs, or if the free-paraphrase baseline matches CVRR — the latter would show the closure vocabulary and memory arbitration are not what produce the effect, only that retrying helps. It is also falsified if the measured rejection-cause coverage rate is low enough that the achievable gain bound is near zero, in which case the paper should pivot to characterizing rejection structure as the contribution.

## References

1. ZeroVLN: Learning to Navigate without Supervised Fine-Tuning of the VLM Brain
