# Residual-Typed Effect Completion: Co-Synthesizing Operations and Checkers from Failure Residuals in Agentic Orchestrators

## Motivation

Every current agentic orchestration stack — Sage-A1.5's subagent delegation with progress/evidence verification, the constraint-verification modules of Generalizing Verifiable Instruction Following (IFBench), and the agentic synthesis pipelines of DeepSeek-V3.2 — assumes a fixed vocabulary: subtasks decompose into a pre-defined set of modular operations, each paired with a discrete verifier drawn from an available library. When a subtask fails because neither the operation nor its checker exists in that vocabulary, the failure signal is consumed only to retry or re-plan *within* the vocabulary, so genuinely novel task demands fall outside the operation set and the orchestrator can never extend its own capability/verification surface. The root cause is structural: capability and verification are assembled from two independent fixed sets, so there is no object from which a *new* operation and its *matching* checker could be derived together.

## Key Insight

A failure, read as a typed residual between a subtask's declared effect signature and the observed evidence, is the unique object whose minimal completion simultaneously determines the missing operation's type and its checker's predicate — because the checker is the dual (negation-as-obligation) of the same residual, capability and verification are co-generated rather than retrieved, making the vocabulary closed under failure by construction.

## Method

**What it is.** RETEC (Residual-Typed Effect Completion) is an orchestration layer that maintains a *shared typed effect signature* Σ = (O, C) of operations o: (Pre, Post) and checkers c: Post → {pass, fail, residual}. Given a subtask with declared postcondition P and observed evidence e that fails, RETEC computes the typed residual r = P ⊖ e, synthesizes the minimal operation o* whose postcondition is the minimal completion of r, and synthesizes its checker c* as the dual of the same r — one object, two artifacts.

**Load-bearing assumption (stated explicitly).** The residual r is exactly the vocabulary gap only if (A1) the declared postcondition P is a faithful effect term (neither over- nor under-specified) and (A2) WITNESS(e) is a sound under-approximation of what the evidence establishes. When A1 or A2 fails, r conflates a missing operation with a specification error or an evidence under-observation, and DUAL(r) would encode the wrong obligation. RETEC therefore adds a residual-attribution step before co-generation and a checker-validation step before admission, as specified below.

**How it works.**

```pseudocode
# Σ = (O, C): shared typed effect signature; types are first-order effect terms
# over a lattice (⊥, value sorts, ∧, ∨, ¬-as-obligation), each op has Pre/Post
# Effect sorts restricted to S = {ordering, formatting, cardinality} for milestone 1.

function STEP(subtask s, evidence e, signature Σ):
    P = s.declared_post                      # effect term asserted by orchestrator
    if CHECK(P, e) == pass: return SUCCESS
    r = RESIDUAL(P, e)                        # typed difference P ⊖ e, minimal in lattice
    if r == ⊥: return RETRY(s)                # failure is execution noise, not vocabulary gap
    # --- residual attribution: separate vocabulary gap from spec error / evidence noise ---
    attr = ATTRIBUTE(r, s, e)                 # returns GAP, SPEC_ERROR, or AMBIGUOUS
    if attr != GAP: return ESCALATE(r, attr)
    # --- co-generation from the single residual object r ---
    o_star = MINIMAL_COMPLETION(r, Σ)         # op whose Post = r ∧ frame; fresh symbol
    c_star = DUAL(r)                          # checker predicate = ¬r as an obligation
    # validate checker against known instances before admission
    if not VALIDATE_CHECKER(c_star, r): return ESCALATE(r, CHECKER_FAIL)
    # type-check the pair against Σ before admission (conservative extension)
    if not CONSERVATIVE_EXT(Σ, o_star, c_star): return ESCALATE(r)
    Σ = Σ ∪ {(o_star, c_star)}                # vocabulary grows, closed under failure
    return EXECUTE(o_star, c_star, s)

function RESIDUAL(P, e):
    # P, e are effect terms; residual is the lattice meet of P with ¬e,
    # projected to the maximal sub-term of P not witnessed by e
    return PROJECT_MAXIMAL(P ∧ ¬WITNESS(e))

function ATTRIBUTE(r, s, e):
    # A1 check: re-derive P' from an independent evidence-grounded source
    # (a second, independently sampled evidence trace e2 for the same subtask,
    #  or a held-out probe subtask s_probe with the same declared P).
    # If P' ⊖ e2 yields a residual r2 with r ∧ r2 == ⊥ (disjoint), the residual
    # is not consistent across traces -> SPEC_ERROR or AMBIGUOUS.
    # A2 check: require r to be witnessed as a consistent residual across
    # at least two independent evidence traces (e and e2).
    # If r is consistent across both traces and r ∧ r2 != ⊥, return GAP.
    # If r is inconsistent (r ∧ r2 == ⊥), return SPEC_ERROR.
    # If only one trace is available or traces are partially observable, return AMBIGUOUS.
    return GAP if CONSISTENT(r, r2) else (SPEC_ERROR if DISJOINT(r, r2) else AMBIGUOUS)

function MINIMAL_COMPLETION(r, Σ):
    # smallest op term whose Post entails r and whose Pre is satisfiable in Σ
    # search over effect-term lattice bottom-up; depth ≤ D (default 3)
    return ARGMAX_{o : Post(o) ⊒ r, Pre(o) satisfiable} SIZE(o)  # minimal size

function DUAL(r):
    # checker is the obligation form of ¬r: c(e') = pass iff e' ⊨ ¬r
    return OBLIGATION(¬r)

function VALIDATE_CHECKER(c_star, r):
    # c_star must pass on a known-positive instance e_pos that satisfies ¬r
    # and fail on a known-negative instance e_neg that does not satisfy ¬r.
    # Instances are drawn from the held-out probe set (size 8: 4 pos, 4 neg).
    # If c_star misclassifies any instance, return False -> ESCALATE.
    return c_star(e_pos) == pass for all e_pos AND c_star(e_neg) == fail for all e_neg

function CONSERVATIVE_EXT(Σ, o, c):
    # o,c must not retract any existing postcondition; c must be decidable
    # on the evidence sort of o's Post (checked by sort inference)
    return DECIDABLE(c, sort(Post(o))) and NO_RETRACTION(Σ, o)
```

**Why this design.** We chose to derive both the operation and the checker from a *single* residual object rather than running two independent synthesis steps (tool synthesis + verifier synthesis, as in the DeepSeek-V3.2 agentic synthesis pipeline and the IFBench constraint-verification modules) because two independent steps can drift: a synthesized tool may not be checkable by a separately synthesized verifier, leaving the orchestrator unable to confirm the new capability — accepting the cost that the residual lattice must be expressive enough to admit a dual, which restricts us to decidable effect sorts. We chose *minimal completion* over retrieval of the nearest existing operation (the Toolformer/Gorilla/RAG-style tool-selection baseline) because retrieval cannot produce an operation that is absent from the library by definition, accepting the cost of a bounded lattice search (depth ≤ 3) that may fail on residuals requiring deep compositions, in which case we escalate rather than hallucinate. We chose a *conservative extension* admission test over unrestricted signature growth because unchecked growth lets the orchestrator invent unfalsifiable checkers that always pass, accepting the cost that some legitimate but non-monotone vocabulary extensions are rejected. We chose the lattice-residual `PROJECT_MAXIMAL(P ∧ ¬WITNESS(e))` over a free-text LLM-generated gap description because a typed residual is the only form whose dual is mechanically well-defined; a natural-language gap would reintroduce the drift problem. A domain expert would not describe this as a variant of tool-synthesis-plus-verification pipelines because those pipelines synthesize from a *task description*, whereas RETEC synthesizes from a *typed failure residual against a declared effect*, and the checker is not a separate model but the lattice dual of the same object.

**Why it measures what we claim.** The computational quantity `r = PROJECT_MAXIMAL(P ∧ ¬WITNESS(e))` measures *the vocabulary gap* (the missing capability) because the assumption that the declared postcondition P is a faithful effect term and that WITNESS(e) is a sound under-approximation of what the evidence establishes makes r exactly the part of P not yet achieved; this assumption fails when the orchestrator's declared P is itself wrong (over- or under-specified), in which case r reflects a specification error rather than a genuine missing operation. The quantity `o_star = MINIMAL_COMPLETION(r, Σ)` measures *the minimal new capability* because the assumption that the effect-term lattice is closed under the operations the task domain requires makes the minimal-size term entailing r the canonical completion; this fails when the domain needs an effect not expressible in the lattice, in which case o_star is the closest expressible approximation and the checker may pass on a weaker condition than intended. The quantity `c_star = DUAL(r)` measures *verification soundness for the new capability* because the assumption that ¬r as an obligation is decidable on the evidence sort of Post(o_star) makes c_star pass exactly when the residual is discharged; this fails when evidence is noisy or the sort is only partially observable, in which case c_star reflects discharge of the observable residual rather than true task success. The admission test `CONSERVATIVE_EXT` measures *vocabulary integrity* because the assumption that no-retraction plus decidability prevents unfalsifiable checkers makes admitted pairs genuinely checkable; it fails when a checker is decidable but trivially satisfiable, in which case integrity is preserved syntactically but not semantically. The residual-attribution step `ATTRIBUTE(r, s, e)` measures *whether r is a genuine vocabulary gap* because the assumption that a consistent residual across two independent evidence traces indicates a missing operation (rather than a spec error or evidence noise) makes GAP the correct label; this fails when both traces share the same systematic evidence bias, in which case ATTRIBUTE may mislabel a spec error as GAP and route it to co-generation, which is why VALIDATE_CHECKER provides a second line of defense.

## Contribution

(1) A residual-typed effect-completion framework (RETEC) in which a single typed residual object co-generates both a new operation and its checker as lattice duals, giving an orchestrator a capability/verification vocabulary that is closed under failure rather than fixed. (2) The design principle that capability synthesis and verifier synthesis must share one object to avoid drift, operationalized as a conservative-extension admission test over a shared typed effect signature. (3) A reproducible residual-lattice construction (effect terms, PROJECT_MAXIMAL, MINIMAL_COMPLETION, DUAL) with a bounded search depth, instantiable on top of any orchestrator that emits structured subtask boundaries and boundary evidence such as Sage-A1.5.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Held-out IFBench constraint suite (58 OOD constraints) | Tests generalization to unseen verifiable constraints |
| Dataset (second domain) | A code-editing agent benchmark (e.g., 40 held-out tasks from a code-editing suite) | Tests whether the mechanism generalizes past a single constraint suite |
| Primary metric | Pass rate on first-attempt constraint discharge | Directly detects vocabulary-gap closure |
| Secondary metric | Fraction of real failures that fall inside vs. outside the decidable lattice (sorts: ordering, formatting, cardinality) | Measures coverage of the residual lattice |
| Diagnostic metric | % of subtasks with faithful P (A1 holds) and % of residuals expressible in the lattice (A2 holds) | Measures when the load-bearing assumption holds vs. breaks |
| Baseline | Retrieval tool-selection (Toolformer/Gorilla-style) | Tests whether retrieval alone closes gaps |
| Baseline | Independent tool+verifier synthesis | Tests whether co-generation avoids drift |
| Baseline | Free-text LLM gap description | Tests whether typed residual matters |
| Ablation-of-ours | RETEC without conservative-extension admission | Tests whether admission prevents fake checkers |
| Ablation-of-ours | RETEC without residual-attribution (ATTRIBUTE always returns GAP) | Tests whether attribution separates spec error from vocabulary gap |
| Ablation-of-ours | RETEC without checker validation (VALIDATE_CHECKER always returns True) | Tests whether checker validation catches mis-specified P |
| Diagnostic experiment | Inject deliberately mis-specified P (over- and under-specified) on 20 hand-constructed cases | Shows the residual degrades predictably and attribution routes to ESCALATE |
| Pilot experiment | 10-15 hand-constructed vocabulary-gap cases | Validates the residual/dual pipeline before scaling to the full 58-constraint suite |

### Why this setup validates the claim

The central claim is that deriving operation and checker from a single typed residual closes vocabulary gaps that retrieval and independent synthesis cannot, and that this shows up specifically where the required capability is absent from the library. IFBench's held-out constraints are the right dataset because they are verifiable and out-of-domain, so a model cannot have memorized the needed operation; a gap is genuine. The second, structurally different domain (code-editing) tests whether the mechanism generalizes past a single constraint suite and lets us report the fraction of real failures that fall inside vs. outside the decidable lattice. The retrieval baseline tests the sub-claim that selection cannot produce absent operations; the independent-synthesis baseline tests the drift sub-claim; the free-text-gap baseline tests the typed-residual sub-claim. The ablation without conservative-extension admission isolates the admission test, so any gain attributable to co-generation is separated from gains from merely growing the signature. The ablation without residual-attribution and the ablation without checker validation isolate the two new guards that make the load-bearing assumption testable: if attribution is doing the work of separating spec error from vocabulary gap, removing it should increase ESCALATE-to-wrong-synthesis errors; if checker validation is catching mis-specified P, removing it should increase silently wrong checkers. The diagnostic experiment injecting deliberately mis-specified P shows the residual degrades predictably (r reflects the spec error, ATTRIBUTE returns SPEC_ERROR or AMBIGUOUS, and the case is escalated rather than co-generated). The diagnostic metrics (% faithful P, % residuals expressible in the lattice) measure how often the load-bearing assumption holds on the held-out suite, turning the assumption into a testable precondition. First-attempt pass rate is the right primary metric because it captures whether the newly synthesized capability is both executable and checkable in one shot — exactly the joint property RETEC claims to secure — rather than rewarding retries that mask checker drift.

### Expected outcome and causal chain

**vs. retrieval tool-selection** — On an IFBench constraint requiring an operation absent from the tool library (e.g., a novel ordering-plus-format obligation), retrieval returns the nearest existing tool, which satisfies a weaker postcondition and silently passes a loose check, so the constraint fails. RETEC computes the residual against the declared postcondition, finds no entailing operation, and synthesizes a minimal completion with its dual checker, so the gap closes. We expect a noticeable pass-rate gap concentrated on constraints whose required operation is absent from the library, with near-parity on constraints solvable by existing tools.

**vs. independent tool+verifier synthesis** — On a case where the synthesized tool's postcondition is expressible but the separately synthesized verifier is defined over a different evidence sort, the pair drifts: the verifier cannot confirm the tool, and the orchestrator either accepts unverified output or loops. RETEC's checker is the lattice dual of the same residual, so its sort matches the operation's postcondition by construction. We expect the independent-synthesis baseline to show high retry counts and checker-mismatch failures, while RETEC shows first-attempt discharge concentrated where the drift would otherwise occur.

**vs. free-text LLM gap description** — On a case where the failure is a subtle partial discharge (most of the postcondition met, one conjunct missing), a free-text gap description is vague and the downstream synthesis targets the wrong sub-goal, producing an operation that re-solves the satisfied part. RETEC's PROJECT_MAXIMAL residual isolates the maximal unwitnessed sub-term, so synthesis targets exactly the missing conjunct. We expect a gap on partially-discharged cases and parity on total failures, where any description suffices.

**vs. RETEC without residual-attribution** — On a case where the declared P is over-specified (requires an effect not actually needed), the residual r reflects the over-specification, and without ATTRIBUTE the system would co-generate an operation and checker for a non-existent gap, producing a checker that passes on the wrong obligation. With ATTRIBUTE, the inconsistency across two evidence traces routes the case to ESCALATE. We expect the no-attribution ablation to show higher rates of admitted-but-wrong checkers and lower first-attempt pass rate on the diagnostic mis-specified-P cases.

**vs. RETEC without checker validation** — On a case where P is under-specified (misses a required conjunct), the residual r is smaller than the true gap, and DUAL(r) yields a checker that passes on the under-specified obligation. Without VALIDATE_CHECKER, this checker is admitted and silently accepts wrong outputs. With VALIDATE_CHECKER, the checker fails on the known-negative instances and the case is escalated. We expect the no-validation ablation to show higher false-pass rates on the diagnostic under-specified-P cases.

### What would falsify this idea

If RETEC's gain over the independent-synthesis and retrieval baselines is uniform across all constraint subsets rather than concentrated where the predicted failure mode (absent operation, sort mismatch, partial discharge) occurs, the causal story is wrong and the gain is a generic capability effect. Equally, if the no-admission ablation matches RETEC's pass rate on held-out constraints, the conservative-extension test is not doing the integrity work claimed; and if the diagnostic metrics show that A1 or A2 holds on fewer than a majority of subtasks, the load-bearing assumption is not satisfied often enough for the residual to measure the vocabulary gap.

## References

1. Sage-A1.5: From Long-Horizon Execution to Scientific Exploration with Capability Composition
2. Generalizing Verifiable Instruction Following
3. DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
