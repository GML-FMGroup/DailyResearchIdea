# Mode-Gated Backtracking Stacks: Coupling Reasoning Depth to a Committed-Prefix Consistency Invariant

## Motivation

Across the dual-process literature the fast/slow mode token and the search structure are treated as independent axes: ACPO (Incentivizing Dual Process Thinking) learns a system-aware depth-budget token but the reasoning stream remains a single monotone chain, and TS-LLM grafts AlphaZero-style tree search on top via an external value function and MCTS controller rather than letting the mode itself define where a commitment can be revoked. The structural consequence is that mode switching selects *how long* to think but never *where* a fast-path commitment is reversible, so step-level credit must be imported from an external process reward model or search controller instead of being exposed by the generation surface itself. This meta-gap — mode decoupled from search state — recurs across the ACPO, CoRe generator/verifier, and TS-LLM trees.

## Key Insight

If the mode token's emission is defined as the unique event that pushes a bounded stack frame and simultaneously freezes its fast-path prefix as a candidate commitment, then backtracking becomes a syntactic consequence of the token's own semantics and the commit/reject decision reduces to a checkable identity between the frozen prefix and the slow-path branch — no external tree controller or value function is needed to decide where to branch.

## Method

**What it is.** MGBS (Mode-Gated Backtracking Stacks) is a decoding-and-training scheme in which a single reserved mode token `⟨MODE:d, k⟩` (depth d ∈ {fast, slow}, stack slot k) is the only operation that pushes a frame onto a bounded backtracking stack of size B=3. Emitting the token freezes the current fast-path prefix P_f into frame k; the model then either emits `⟨COMMIT k⟩` (pop, accept) or `⟨REVOKE k⟩` (pop, discard P_f and continue from the frame's saved spine). Inputs: prompt + base LM logits. Outputs: a reasoning trace interleaved with mode/stack tokens and a final answer.

**How it works.**
```pseudocode
state: spine (committed tokens), stack (max depth B=3), frame k holds (spine_snapshot, fast_prefix P_f)

def step(logits, spine, stack):
    t = sample(logits)                      # ordinary token
    if t == <MODE:d,k>:                     # only legal if len(stack)<B and k==len(stack)
        frame[k] = (copy(spine), fast_prefix := spine)
        stack.push(frame[k]); spine = spine   # freeze, do NOT advance answer
        return ENTER_SLOW(d)
    if t == <COMMIT k>:                     # only legal if top==k
        # identity check: does slow-path branch S_k entail P_f's conclusion?
        if CONSISTENT(P_f=frame[k].fast_prefix, S_k=spine_since_push):
            stack.pop(); return ACCEPT(spine)
        else: return REJECT_SIGNAL()        # logit penalty, force REVOKE next
    if t == <REVOKE k>:
        spine = frame[k].spine_snapshot; stack.pop(); return CONTINUE_FAST()
    spine.append(t); return t

def CONSISTENT(P_f, S_k):                   # mode-conditional consistency identity
    # identity: answer(P_f) == answer(S_k) AND every entity/number in P_f appears in S_k
    # AND every intermediate quantity/operation in P_f is re-executed in S_k
    return exact_match(answer_of(P_f), answer_of(S_k)) \
           and subset(entities(P_f), entities(S_k)) \
           and re_execute_match(ops(P_f), ops(S_k))

def re_execute_match(ops_Pf, ops_Sk):
    # ops_Pf: list of (op, operands, result) extracted from P_f by a fixed regex over arithmetic expressions
    # ops_Sk: same extraction from S_k
    # returns True iff for every op in ops_Pf there exists an op in ops_Sk with identical (op, operands) and identical result
    return all(any(op == op2 and operands == operands2 and result == result2 for (op2, operands2, result2) in ops_Sk) for (op, operands, result) in ops_Pf)
```

**Assumption A (explicit).** `CONSISTENT(P_f, S_k)` measures *derivational consistency* under assumption A: the answer slot plus the set of intermediate arithmetic operations is a sufficient statistic for the reasoning conclusion. A fails when the model reaches the right answer via a wrong intermediate entity (e.g., misreads a rate as a total but the final number coincidentally matches); in that case the identity reflects answer-slot and operation-surface agreement rather than genuine derivational consistency. We make A explicit and test it in the calibration study below.

**Training:** two-stage. (1) SFT cold-start on traces where `⟨MODE:d,k⟩` is inserted at genuine decision points mined by a cheap offline heuristic: for each training problem, sample 8 fast-path completions at temperature 0.7; a decision point is a token position where at least 2 of the 8 completions diverge in their next arithmetic operation (detected by the same regex as `re_execute_match`) and the majority fast-path answer disagrees with the gold answer on a held-out sample of 512 problems. (2) GRPO with a reward that is the sum of task correctness and a per-frame term `+1` for a `⟨COMMIT⟩` whose `CONSISTENT` check passes, `-0.5` for a `⟨REVOKE⟩`, and `-1` for a `⟨COMMIT⟩` that fails the identity check — this yields step-level credit localized to the frame rather than the whole trajectory. Stack depth B=3, slow-path budget 256 tokens per frame.

**Why this design.** We chose a *bounded* stack of B=3 over an unbounded one because unbounded branching reintroduces the MCTS controller we are trying to eliminate (the search tree would again need an external scheduler), accepting that traces with more than three nested revocations are out of distribution. We chose the mode token itself as the *only* push operation over a separate `⟨BRANCH⟩` token because a dedicated branch token would reproduce TS-LLM's decoupling of mode from search — a domain expert would describe that design as TS-LLM with a cheaper controller, whereas here the mode token is load-bearing for the stack invariant. We chose the exact-match + entity-subset + re-execution consistency identity over a learned verifier (as in CoRe) because a learned verifier is exactly the external PRM whose absence is the point of the contribution; the cost is that the identity is brittle on free-form answers, which we mitigate by normalizing to a canonical answer slot and by the re-execution check that catches entity substitutions.

**Why it measures what we claim.** `CONSISTENT(P_f, S_k)` measures *revocability of the fast-path commitment* because the exact-match-plus-entity-subset-plus-re-execution identity is equivalent to "the slow branch did not overturn the fast conclusion" under assumption A that the answer slot plus intermediate operations is a sufficient statistic for the reasoning conclusion; this assumption fails when the model reaches the right answer via a wrong intermediate entity, in which case the identity reflects answer-slot and operation-surface agreement rather than genuine derivational consistency. The per-frame reward `+1/-0.5/-1` measures *step-level credit* because it is assigned to the frame that owns the prefix, under the assumption that the frame boundary is the only point where the prefix can be revoked; this fails when a revocation should have occurred at an earlier frame, in which case the reward reflects local rather than global credit. Stack depth `len(stack)` measures *reasoning depth allocation* because the mode token is the sole push operation, under the assumption that depth is chosen at push time; this fails if the model emits depth tokens at non-decision points, in which case depth reflects surface verbosity.

## Contribution

(1) A decoding-and-training framework, MGBS, in which a single mode token is the sole operation that pushes a bounded backtracking frame, making branch/revoke/commit events syntactic consequences of mode semantics rather than outputs of an external search controller. (2) A mode-conditional consistency identity (exact-match answer plus entity-subset containment between the frozen fast prefix and the slow branch) that supplies step-level credit localized to a frame, removing the need for a process reward model or MCTS value function. (3) A two-stage SFT+GRPO recipe with a per-frame reward that trains the model to place mode tokens at genuine revocation points rather than uniformly across the trace.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MATH500 + GSM8K mixed-difficulty split | Contains genuine decision points where fast path errs |
| Primary metric | Accuracy on frames where fast/slow disagree | Directly isolates revocability benefit |
| Baseline | Vanilla CoT decoding | Tests whether backtracking adds anything over flat reasoning |
| Baseline | Best-of-N with self-consistency (N=8) | Tests whether external search beats internal stack |
| Baseline | CoRe-style learned verifier | Tests whether identity check can replace external PRM |
| Ablation-of-ours | MGBS with CONSISTENT always-true | Tests whether identity check, not stack, drives gains |
| Ablation-of-ours | MGBS with CONSISTENT = exact-match only (no re-execution) | Tests whether re-execution check is load-bearing |
| Calibration set | 512 hand-labeled problems with (fast prefix, slow branch) pairs | Measures divergence between answer-slot agreement and derivational consistency |

### Why this setup validates the claim

The central claim is that a mode token load-bearing for a bounded stack yields localized, revocable fast-path commitments without an external controller. The dataset choice matters because MATH500 and GSM8K contain problems where a plausible fast-path answer is wrong for a specific intermediate reason — exactly the decision points the offline heuristic mines. The primary metric — accuracy restricted to the disagreement subset — is the right detector because it conditions on the cases where the mechanism is supposed to fire; a uniform gain on the agreement subset would indicate the gains come from extra compute, not revocability. Vanilla CoT tests whether the stack adds anything beyond flat generation. Best-of-N with self-consistency (N=8) tests whether external sampling search dominates the internal bounded stack, directly probing the "no external scheduler" claim. The CoRe-style learned verifier tests whether the identity check is a viable substitute for an external PRM. The always-true CONSISTENT ablation is the load-bearing control: if the stack alone produces the gains, the identity check is decorative and the contribution collapses to "bounded backtracking," not "revocability." The exact-match-only ablation isolates the contribution of the re-execution check. The calibration set of 512 hand-labeled problems measures how often answer-slot agreement diverges from derivational consistency, directly testing assumption A.

### Expected outcome and causal chain

**vs. Vanilla CoT decoding** — On a case where the fast path commits to a wrong intermediate quantity (e.g. misreading a rate as a total), vanilla CoT produces a confidently wrong final answer because it has no mechanism to freeze and revisit the prefix. Our method instead emits `⟨MODE:slow,k⟩` at the mined decision point, runs the slow branch, and either `⟨COMMIT⟩`s (if consistent) or `⟨REVOKE⟩`s back to the spine. We expect a concentrated accuracy gap on the disagreement subset with rough parity on the agreement subset, since only the former exercises the stack.

**vs. Best-of-N with self-consistency (N=8)** — On a case where the fast path is wrong but the model's sampled diversity is low (all N samples share the same misread), best-of-N produces the same wrong answer because its search is external and blind to the specific prefix it needs to revoke. Our method instead localizes the revocation to the exact frame via the mode token, so the slow branch starts from the saved spine rather than resampling from scratch. We expect our method to beat best-of-N at matched token budget on the disagreement subset, with the gap widening as N grows and diversity collapses.

**vs. CoRe-style learned verifier** — On a case where the verifier was trained on a distribution that under-represents the failure mode (e.g. entity-substitution errors), CoRe produces a false accept because its verifier scores surface plausibility rather than derivational consistency. Our method instead applies the exact-match-plus-entity-subset-plus-re-execution identity, which catches entity substitutions by construction. We expect our method to show a specific advantage on entity-substitution cases and near-parity elsewhere, plus a lower false-accept rate on `⟨COMMIT⟩` frames.

**vs. MGBS with CONSISTENT always-true** — On a case where the slow branch restates the fast prefix's entities but does not re-derive its intermediate operations, the always-true ablation accepts the COMMIT and propagates a wrong intermediate quantity. Full MGBS rejects it via the re-execution check. We expect full MGBS to outperform the always-true ablation on the disagreement subset, with the gap concentrated on entity-substitution cases.

**vs. MGBS with CONSISTENT = exact-match only** — On a case where the slow branch reaches the same final answer via a different intermediate operation (e.g., coincidental cancellation), exact-match-only accepts the COMMIT while full MGBS rejects it. We expect full MGBS to show a lower false-accept rate on the calibration set's right-answer-wrong-reason pairs.

### What would falsify this idea

If the always-true CONSISTENT ablation matches full MGBS on the disagreement subset, the identity check is not doing the work and the claim that revocability is measured by the identity collapses. If gains are uniform across agreement and disagreement subsets rather than concentrated on the disagreement subset, the extra compute of the slow path — not the stack mechanism — explains the improvement. If the calibration study shows that answer-slot agreement diverges from derivational consistency on more than 20% of the 512 hand-labeled problems, assumption A fails and the identity is not a faithful proxy.

## References

1. Incentivizing Dual Process Thinking for Efficient Large Language Model Reasoning
2. Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
3. Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs
4. Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws
5. Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training
6. Broken Neural Scaling Laws
7. Solving Math Word Problems via Cooperative Reasoning induced Language Models
8. Solving Quantitative Reasoning Problems with Language Models
