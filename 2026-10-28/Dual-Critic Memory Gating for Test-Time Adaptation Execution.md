# Dual-Critic Memory Gating for Test-Time Adaptation: Execution-Grounded and Novelty-Filtered Experience Storage

## Motivation

Test-time memory evolution methods like TTME (Can Test-Time Memory Evolution Outperform Test-Time Training for Scientific Discovery?) update external memory by distilling experience from each discovery step, but they rely on the LLM as the sole judge of whether an experience is valid and non-redundant. This creates a shared residual gap: misjudged experiences (e.g., plausible but incorrect reasoning) are stored and poison future retrieval, while redundant entries accumulate and dilute the memory's utility. The structural root cause is that LLM self-judgment conflates semantic plausibility with task-specific correctness and lacks any mechanism for redundancy detection, so validation invariance and redundancy closure cannot be enforced by the same critic.

## Key Insight

A memory entry is admissible if and only if it passes two independent, task-grounded filters—one checking correctness against an executable or proof-checkable ground truth, and one checking novelty against the existing memory via a set-cover-style redundancy test—so that the stored memory is exactly the set of verified, non-redundant experiences, a property that LLM-as-sole-judge cannot guarantee because it conflates plausibility with correctness and never computes redundancy.

## Method

# Dual-Critic Memory Gating (DCMG)
# Inputs: base LLM M, task-specific verifier V (execution/proof checker),
#         novelty threshold tau (default 0.85), memory capacity K (default 500),
#         probe suite P (fixed set of 64 probe inputs per task, generated once from the task's
#         public test distribution and held fixed across all runs),
#         behavioral signature function sig(e) = hash of V.run(e, P) outputs (e.g., tuple of
#         pass/fail and output values on each probe input),
#         conservative pre-filter threshold tau_pre (default 0.95, tuned on held-out split)
# Output: evolving memory store S

Initialize S = empty set
For each discovery step t = 1, 2, ...:
    # 1. LLM proposes a candidate experience e_t (reasoning trace + outcome)
    e_t = M.propose(context, S)

    # 2. Critic-1 (Validation): task-specific verifier checks e_t
    #    - For code/math: execute e_t or check proof with V
    #    - For non-executable: use a held-out test suite (see design)
    valid_t = V.check(e_t)  # binary: True if passes ground-truth check
    if not valid_t:
        discard e_t; continue

    # 3. Critic-2 (Novelty/Redundancy): compute redundancy score
    #    - First, cheap pre-filter: embed e_t and each s in S with a frozen sentence encoder
    #      (all-MiniLM-L6-v2, 384-dim, L2-normalized).
    #    - Compute max cosine similarity to any s in S: sim_max = max_{s in S} cos(emb(e_t), emb(s))
    #    - If sim_max <= tau_pre (default 0.95), e_t is considered novel (no further check).
    #    - If sim_max > tau_pre, apply behavioral redundancy test:
    #         For each s in S with cos(emb(e_t), emb(s)) > tau_pre:
    #             if sig(e_t) == sig(s): mark redundant_t = True; break
    #         If no behavioral match, e_t is novel (redundant_t = False).
    #    - If sim_max > tau (default 0.85) but <= tau_pre, e_t is considered novel (pre-filter conservative).
    #    - If sim_max > tau_pre and behavioral match found, redundant_t = True.
    #    - If sim_max > tau_pre and no behavioral match, redundant_t = False.
    #    - If sim_max <= tau, redundant_t = False.
    #    (Note: tau=0.85 is the original threshold; tau_pre=0.95 is a conservative pre-filter.
    #     The behavioral test is the primary redundancy criterion; embedding similarity is only a
    #     cheap pre-filter to avoid expensive behavioral checks on clearly distinct entries.)
    if redundant_t:
        discard e_t; continue

    # 4. Store only entries passing both critics
    S = S ∪ {e_t}
    # 5. Capacity management: if |S| > K, evict the entry with
    #    lowest validation confidence (V.score) among stored entries
    if |S| > K:
        evict argmin_{s in S} V.score(s)

Return S

**Why this design**
We chose a hard binary gate from an external verifier (Critic-1) over LLM self-judgment because semantic plausibility is not equivalent to task correctness, accepting the cost that some valid but non-executable experiences are discarded; to mitigate this, we require that every task provides a verifier (execution for code/math, proof checker for formal proofs, or a held-out test suite for empirical tasks). We chose a behavioral redundancy test (execution-grounded signature) as the primary novelty gate, with a frozen sentence-encoder cosine-similarity pre-filter, over LLM-based redundancy judgment because LLM redundancy judgments are inconsistent and expensive, accepting the cost that semantically equivalent but behaviorally distinct experiences may slip through; the pre-filter threshold tau_pre is tuned on a held-out set to balance precision and recall, and the behavioral signature is computed on a fixed probe suite P of 64 inputs. We chose capacity-based eviction by validation confidence over FIFO or random eviction because it preserves high-quality experiences, accepting the cost that early low-confidence but foundational experiences may be evicted; we set K=500 based on memory scalability experiments. Unlike FunSearch's single evaluator, which only checks program correctness and does not manage redundancy, our dual-critic explicitly enforces both validation invariance and redundancy closure, and unlike TTME's LLM-as-sole-judge, our gates are task-grounded and deterministic given the verifier and encoder.

**Why it measures what we claim**
The computational quantity `valid_t = V.check(e_t)` measures **validation invariance** (the property that stored experiences are correct) because we assume the verifier V is sound and complete for the task domain—i.e., it returns True iff the experience is correct; this assumption fails when the verifier is incomplete (e.g., a proof checker that rejects valid but non-canonical proofs), in which case `valid_t` reflects verifier strictness rather than true correctness. The computational quantity `redundant_t = (sim_max > tau_pre and behavioral_match)` measures **redundancy closure** (the property that no two stored entries are behaviorally equivalent) because we assume the behavioral signature `sig(e)` (outputs on the fixed probe suite P) is a monotone proxy for semantic equivalence—i.e., two experiences are semantically redundant iff they produce identical outputs on P; this assumption fails when the probe suite P is not representative of the task's input distribution (e.g., two different cap-set constructions that agree on P but diverge on other inputs), in which case `redundant_t` reflects probe coverage rather than true semantic redundancy. The embedding pre-filter `sim_max > tau_pre` measures **candidate redundancy** because we assume cosine similarity in the frozen encoder space is a conservative upper bound on behavioral equivalence—i.e., if sim_max <= tau_pre, the experiences are behaviorally distinct; this assumption fails when the encoder conflates distinct experiences with similar surface forms (e.g., two different math proofs with similar notation), in which case the pre-filter may let redundant entries through to the behavioral test, but the behavioral test then correctly rejects them. The eviction rule `argmin V.score(s)` measures **experience quality** because we assume V.score is calibrated across experiences; this assumption fails when V.score is not comparable across different task types, in which case eviction may remove high-quality entries from one task while retaining low-quality entries from another.

## Contribution

['A dual-critic memory gating framework (DCMG) that replaces LLM-as-sole-judge with two independent, task-grounded filters—a validation critic using an external verifier and a novelty critic using a frozen encoder—to enforce validation invariance and redundancy closure in test-time memory evolution.', 'A concrete instantiation of DCMG for scientific discovery tasks, including a protocol for constructing task-specific verifiers (execution for code/math, proof checking for formal proofs, held-out test suites for empirical tasks) and a capacity-based eviction rule that preserves high-confidence experiences.', "An empirical demonstration that DCMG, when plugged into TTME's memory evolution loop, reduces the storage of misjudged and redundant experiences, leading to improved downstream discovery performance on mathematical tasks compared to LLM-as-sole-judge baselines."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | FunSearch cap-set + bin-packing task suite | Executable verifier gives ground-truth correctness |
| Primary metric | Valid-novel yield per discovery step | Directly reflects both gates working |
| Baseline | FunSearch single-evaluator | Tests value of redundancy gating |
| Baseline | TTME LLM-as-sole-judge | Tests value of task-grounded validation |
| Baseline | TTT adaptation | Tests memory gating vs weight updates |
| Ablation-of-ours | DCMG minus Critic-2 (novelty off) | Isolates redundancy closure contribution |
| Ablation-of-ours | DCMG with embedding-only Critic-2 (tau=0.85, no behavioral test) | Isolates behavioral redundancy test contribution |
| Calibration study | Labeled redundancy probe set (200 pairs) | Validates monotonicity assumption of behavioral signature |

### Why this setup validates the claim

The central claim is that dual-critic gating yields a memory store that is simultaneously valid and non-redundant, and that this store improves downstream discovery efficiency more than single-signal adaptation. The FunSearch task suite is the right testbed because it supplies an executable verifier, so Critic-1's soundness assumption actually holds and we can attribute any gain to gating rather than to verifier noise. FunSearch's single evaluator isolates the redundancy sub-claim: it validates but never deduplicates, so if our gain over it is real, it must come from Critic-2. TTME's LLM-as-sole-judge isolates the validation sub-claim: its gate is semantic, not executable, so if our gain over it is real, it must come from Critic-1. TTT tests the broader claim that memory gating beats weight-level adaptation under fixed compute. The ablation (novelty off) is the sharpest test: it makes the same valid-only store as FunSearch, so any gap between it and full DCMG is attributable purely to redundancy closure. The additional ablation (embedding-only Critic-2) isolates the contribution of the behavioral redundancy test over the embedding pre-filter alone, directly testing the load-bearing assumption that behavioral signatures are a better redundancy proxy than cosine similarity. The calibration study measures the precision and recall of the behavioral signature against human-annotated semantic redundancy on 200 pairs, providing empirical grounding for the monotonicity assumption. Valid-novel yield per step is the right primary metric because it counts exactly the quantity both critics are designed to produce; a metric like raw task score would confound gating quality with base-model capability.

### Expected outcome and causal chain

**vs. FunSearch single-evaluator** — On a case where the LLM repeatedly proposes near-identical cap-set constructions re-expressed with permuted variable names, FunSearch accepts each as a fresh program because its evaluator only checks correctness and never compares against the store, so its population fills with semantic duplicates and search stalls on a narrow region. Our method instead rejects each duplicate at Critic-2 because the behavioral signature (outputs on the 64-probe suite) matches an existing entry, so the store stays diverse and the LLM is pushed toward unexplored constructions. We expect a noticeable gap on the late-stage discovery subset (where diversity matters) but near-parity on early steps where the store is still sparse.

**vs. TTME LLM-as-sole-judge** — On a case where the LLM emits a plausible-looking but subtly incorrect bin-packing heuristic (e.g. a tie-break rule that violates capacity on adversarial inputs), TTME admits it because the LLM judge rates semantic plausibility highly, so the flawed heuristic contaminates the store and propagates into later proposals. Our method instead discards it at Critic-1 because the executable verifier returns False on the adversarial test, so the store remains sound. We expect a concentrated gap on the adversarial-input subset with near-parity on the easy subset, and a visible divergence in store contamination rate over steps.

**vs. TTT adaptation** — On a case where the task distribution shifts mid-run (cap-set dimension increases), TTT updates weights on early experiences and overfits to the old regime because gradient updates are not selectively gated by correctness or novelty. Our method instead keeps a discrete store that can be queried without overwriting weights, so early-regime experiences are retained but not forced into the model. We expect our method to hold roughly flat performance across the shift while TTT degrades, with the gap widening as the shift grows.

**vs. DCMG minus Critic-2 (novelty off)** — This ablation makes the same valid-only store as FunSearch, so any gap between it and full DCMG is attributable purely to redundancy closure. We expect full DCMG to outperform this ablation on late-stage discovery steps where redundancy accumulates, with near-parity on early steps, confirming that Critic-2 is the causal driver of the gain.

**vs. DCMG with embedding-only Critic-2 (tau=0.85, no behavioral test)** — This ablation uses only the frozen-encoder cosine similarity gate with tau=0.85, without the behavioral signature test. We expect it to underperform full DCMG because it will wrongly discard lexically similar but behaviorally distinct experiences (e.g., two different cap-set constructions with identical FunSearch scaffolding) and wrongly admit lexically different but behaviorally identical heuristics. The gap between this ablation and full DCMG directly measures the value of the behavioral redundancy test and validates the load-bearing assumption that behavioral signatures are a better redundancy proxy than cosine similarity.

### What would falsify this idea

If DCMG's gain over FunSearch is uniform across early and late steps rather than concentrated where redundancy accumulates, then Critic-2 is not the causal driver and the central claim collapses. Likewise, if the novelty-off ablation matches full DCMG, redundancy closure contributes nothing and the dual-critic design is unjustified. Additionally, if the embedding-only Critic-2 ablation matches full DCMG, the behavioral redundancy test adds no value and the load-bearing assumption that behavioral signatures are a better proxy than cosine similarity is false.

## References

1. Can Test-Time Memory Evolution Outperform Test-Time Training for Scientific Discovery?
2. Mathematical discoveries from program search with large language models
3. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery
4. Language Models Can Teach Themselves to Program Better
5. New lower bounds for cap sets
