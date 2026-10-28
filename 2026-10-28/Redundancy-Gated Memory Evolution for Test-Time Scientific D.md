# Redundancy-Gated Memory Evolution for Test-Time Scientific Discovery

## Motivation

Test-time memory evolution (TTME) distills each discovery step's experience into an external memory without weight updates, but its update rule has no redundancy control, so memory can fill with near-duplicate experiences that crowd out diverse, task-relevant guidance and let degenerate memory growth masquerade as adaptation. This is the same structural gap that FunSearch avoids only by relying on an external evaluator to score and select candidates, and that ShinkaEvolve-style search avoids via explicit rejection sampling; TTME has neither, so its memory closure is left uncontrolled. The meta-level problem is that memory-based test-time adaptation has no intrinsic criterion for what belongs in the memory set, making adaptation capacity unmeasurable and unbounded.

## Key Insight

Rejecting a candidate experience whose embedding lies within a task-calibrated cosine radius of an existing memory entry makes the memory set a maximal epsilon-separated packing, so its capacity is bounded by the covering number of the experience manifold rather than by the number of discovery steps.

## Method

**What it is.** Redundancy-Gated Memory Evolution (RGME) is a drop-in replacement for the TTME memory update: each candidate experience (a distilled lesson plus its embedding) passes through a similarity gate against current memory before insertion, and only accepted experiences condition subsequent LLM discovery. Inputs: candidate experience from the current discovery step, current memory set M, a running similarity-statistics buffer. Output: updated memory M' and the conditioning context for the next step.

**How it works.**

```pseudocode
# per discovery step t: LLM proposes solution, evaluator scores it, lesson distilled
c_t = distill(solution_t, score_t)          # experience text + its embedding e_t

# Phase 1: redundancy gate (two-signal: cosine pre-filter + LLM equivalence check)
if |M| < 8:                                  # bootstrap: accept until memory seeded
    accept = True
else:
    sims = { cos(e_t, e_m) : m in M }
    s_max = max(sims)
    tau_t = mu_t + 1.0 * sigma_t             # mu_t, sigma_t from similarity buffer
    if s_max < tau_t:
        accept = True
    else:
        # LLM-based equivalence check on the top-1 match m* = argmax_m sims
        # prompt: 'Are these two lessons semantically equivalent in their operative clause? Answer YES or NO.'
        # temperature=0, max_tokens=3, model = same LLM used for discovery
        equiv = llm_equivalence_check(c_t, M[m*])
        accept = not equiv

# Phase 2: insertion or merge
if accept:
    M  <- M ∪ {(c_t, e_t)}
    mu_t, sigma_t <- running_mean_std(sims ∪ buffer)   # update buffer, window 64
else:
    m* = argmax_m sims
    M[m*] <- merge(M[m*], c_t)               # append novel clause only
    e_{m*} <- embed(M[m*].text)              # re-embed merged entry to reflect appended clause
    # diagnostic: log merge event with s_max, tau_t, and LLM equivalence verdict
    log_merge_event(t, m*, s_max, tau_t, equiv)

# Phase 3: conditioning (unchanged from TTME)
context_t+1 = top_k(M, query=e_t, k=3)       # cosine retrieval
```

Hyperparameters: bootstrap threshold 8, gate margin 1.0 (tau = mu + 1.0 sigma), similarity buffer window 64, retrieval k=3, LLM equivalence check temperature=0, max_tokens=3.

**Why this design.** We chose a *running* similarity distribution (mu_t, sigma_t from the last 64 accepted-step similarity statistics) over a fixed cosine threshold because experience embeddings drift as the LLM visits different regions of the task, so a static cutoff either blocks early diverse lessons or admits late near-duplicates; the cost is that the gate is non-stationary and can briefly admit a redundant entry right after a distribution shift. We chose *merge-on-reject* over *discard-on-reject* because discarding throws away the novel clause inside a near-duplicate lesson, whereas merging preserves it at the price of a slightly noisier single memory entry. We chose cosine similarity of the distilled-lesson embedding over code-AST diff because lessons are natural-language distillations rather than programs, so AST diff is undefined, accepting that embedding cosine is a coarser signal than structural diff. We added an LLM-based equivalence check as a second signal because the load-bearing assumption that cosine similarity is a monotone proxy for actionable redundancy is known to fail on fine-grained operative-clause distinctions (SBERT, MTEB, anisotropy); the check is a cheap, deterministic (temperature=0) verification that prevents false merges of lexically similar but operatively distinct lessons. We re-embed merged entries so that the epsilon-separation/covering-number bound remains valid after clause appending. A domain expert would NOT describe this as FunSearch: FunSearch's filtering is an *external evaluator scoring programs*, whereas RGME's filter is an *intrinsic geometric gate over the memory set itself* that operates without any evaluator and bounds memory capacity directly; nor is it TTME, which has no gate at all.

**Why it measures what we claim.** `s_max`, the maximum cosine similarity of the candidate embedding to existing entries, measures *redundancy of the candidate experience* because the assumption that distilled-lesson embeddings place semantically equivalent lessons within a cosine radius makes s_max a monotone proxy for semantic overlap; this assumption fails when two lessons share vocabulary but differ in the operative clause (e.g., both mention 'modular arithmetic' but one prescribes a different construction), in which case s_max reflects lexical/topical overlap rather than actionable redundancy. To mitigate this failure, the LLM equivalence check acts as a second signal that verifies operative-clause equivalence before classifying a candidate as redundant. `tau_t = mu_t + 1.0 sigma_t` measures *the task's current novelty bar* because the assumption that the similarity distribution of recent accepted steps tracks the local density of the experience manifold makes a one-sigma-above-mean cutoff a density-relative separation radius; this assumption fails when the LLM's proposals become globally homogeneous (all sims high), in which case tau_t rises and the gate degenerates to accepting almost nothing, reflecting proposal collapse rather than genuine memory saturation. The resulting memory set measures *epsilon-separated experience coverage* because the assumption that accepted entries are pairwise below tau_t makes the set an epsilon-packing whose size is bounded by the covering number; this fails when merge-on-reject silently raises an entry's effective radius, in which case set size understates true coverage. Re-embedding merged entries updates their effective radius to reflect the appended clause, preserving the packing interpretation.

## Contribution

(1) A redundancy-gated memory update rule (RGME) that turns TTME's unbounded memory into an epsilon-separated packing with a running, task-calibrated similarity threshold and merge-on-reject insertion. (2) The design principle that memory capacity in test-time adaptation should be bounded by experience-manifold covering number rather than by step count, with the gate's threshold derived from the empirical similarity distribution of accepted steps rather than fixed. (3) A diagnostic of degenerate memory growth — the ratio of accepted to proposed experiences over time — that separates genuine adaptation from repetitive memory accumulation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Five mathematical discovery tasks (cap set, bin packing, etc.) | Matches TTME/TTT comparison regime directly |
| Dataset | One non-math discovery task (e.g., combinatorial heuristic search or program synthesis) | Tests generality beyond math tasks (reviewer suggestion) |
| Primary metric | Best score achieved vs. FLOPs-to-threshold | Captures quality-per-compute of memory evolution |
| Secondary metric | Correlation between s_max and human/LLM-judged semantic redundancy on a held-out set of 512 lesson pairs | Grounds the redundancy proxy claim (reviewer suggestion) |
| Baseline | Test-Time Training (TTT) | Tests whether gated memory beats gradient adaptation |
| Baseline | Ungated TTME | Tests whether the gate itself drives gains |
| Baseline | Discard-on-reject TTME | Tests whether merge-on-reject preserves novel clauses |
| Ablation-of-ours | RGME with fixed cosine threshold | Tests whether running distribution beats static cutoff |
| Ablation-of-ours | RGME with cosine-only gate (no LLM equivalence check) | Tests whether the second signal prevents false merges |
| Ablation-of-ours | RGME with merge-on-reject but no re-embedding | Tests whether re-embedding preserves packing validity |

### Why this setup validates the claim

The central claim is that redundancy-gating the memory set bounds capacity without discarding novel content, so RGME should reach equal-or-better best scores with fewer discovery steps and lower FLOPs than ungated TTME, and beat TTT outright on tasks where adaptivity matters. TTT is the right primary baseline because it isolates the test-time adaptation mechanism: if RGME only matched inference-only frameworks, the claim would be trivial, but matching TTT while spending far less adaptation compute is the substantive win. Ungated TTME is the decisive control — it shares everything with RGME except the gate, so any gap is attributable to gating. Discard-on-reject isolates the merge design choice, and the fixed-threshold ablation isolates the running-statistics choice. The cosine-only gate ablation isolates the contribution of the LLM equivalence check, and the no-re-embedding ablation tests the packing validity. Best-score-vs-FLOPs is the right metric because it simultaneously exposes quality and the efficiency the gate is supposed to buy; a pure final-score metric would hide the compute story entirely. The calibration study directly measures the correlation between s_max and judged semantic redundancy, grounding the proxy assumption rather than leaving it untested. The non-math task checks whether the gate's benefit transfers beyond the math family.

### Expected outcome and causal chain

**vs. Test-Time Training (TTT)** — On a task where the useful lesson is discovered early but must be retained across many later proposals (e.g. a cap-set construction insight), TTT overwrites its parameters on each new batch and loses the early lesson, because gradient adaptation has no explicit experience store. Our method instead keeps the distilled lesson in memory and reconditions retrieval on it, so we expect a moderate best-score gap concentrated on tasks with late-stage refinement, with near-parity on short-horizon tasks.

**vs. Ungated TTME** — On a task where the LLM repeatedly proposes near-identical lessons (e.g. bin-packing heuristics converging on the same sorting rule), ungated TTME floods memory with duplicates, diluting top-k retrieval so the conditioning context becomes redundant and discovery stalls. Our gate rejects and merges those duplicates, keeping the retrieval set diverse, so we expect RGME to reach the same best score in noticeably fewer discovery steps and lower FLOPs, with the gap widening on longer runs.

**vs. Discard-on-reject TTME** — On a case where a near-duplicate lesson carries one novel operative clause (e.g. a construction differing only in the modulus choice), discard-on-reject throws the clause away, so the framework never exploits it. Our merge-on-reject appends the novel clause to the matched entry, preserving it for later retrieval, so we expect a small but consistent best-score edge on tasks where incremental clause-level refinements matter, and parity elsewhere.

**vs. RGME with cosine-only gate** — On cases where two lessons are lexically similar but operatively distinct (e.g., both mention 'modular arithmetic' but prescribe different constructions), the cosine-only gate would falsely merge them, corrupting the packing argument. The LLM equivalence check prevents this, so we expect the full RGME to achieve higher best scores on tasks with fine-grained clause distinctions, with the gap concentrated on merge events flagged as suspected false merges by the diagnostic.

**vs. RGME with no re-embedding** — When a merged entry's embedding is not updated, its effective radius understates the appended clause, violating the epsilon-separation bound. Re-embedding corrects this, so we expect the full RGME to show better retrieval diversity and slightly higher best scores on tasks with many merges.

**vs. RGME with fixed cosine threshold** — The running distribution adapts to embedding drift, while a fixed threshold either blocks early diverse lessons or admits late near-duplicates. We expect RGME to outperform the fixed-threshold ablation on tasks where the similarity distribution shifts over time, with the gap widening on longer runs.

### What would falsify this idea

If RGME's advantage over ungated TTME were uniform across all tasks rather than concentrated on the long-horizon, duplicate-heavy tasks where the gate is predicted to bind, the causal story that redundancy-gating drives the gain would be wrong. Likewise, if the fixed-threshold ablation matched RGME everywhere, the running-distribution mechanism would be unnecessary; and if the cosine-only gate matched full RGME, the LLM equivalence check would be redundant.

## References

1. Can Test-Time Memory Evolution Outperform Test-Time Training for Scientific Discovery?
2. Mathematical discoveries from program search with large language models
3. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery
4. Language Models Can Teach Themselves to Program Better
5. New lower bounds for cap sets
