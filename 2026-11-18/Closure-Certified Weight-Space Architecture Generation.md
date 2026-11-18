# Closure-Certified Weight-Space Architecture Generation

## Motivation

LLM-authored weight-space programs (the table+procedure models of 'Weight-Space Programming Doesn't Need Transformers') are accepted purely on in-distribution accuracy: the LLM emits a sparse fact table and a fixed query procedure, and a held-out accuracy number decides whether the architecture is kept. Nothing in the generative loop forces the emitted procedure to be algebraically well-defined on fact combinations it never saw, so an architecture can score well by memorizing rows while its composition operator is undefined or wrong on unseen pairs, chains, and distractor facts. This is the same structural gap that recurs across LLM-guided NAS trees (RZ-NAS E3, NADER E4): architecture quality is judged by a scalar score on the visible task, never by whether the generated computation is closed under composition, so no evolution step is ever forced to certify it.

## Key Insight

Compositional closure is a property of the emitted procedure's algebra over the fact table, not of its accuracy on the visible facts, so it can be certified by a generator that synthesizes only fact combinations the procedure was never shown and checks the procedure's own composition rule against the algebraic identities it must satisfy.

## Method

**What it is.** CLOSURE-CERT (Compositional-cLosure Certificate for LLM-authored weight-space programs) is a search-time gate placed between the LLM's architecture emission and the acceptance step. Inputs: an LLM-authored program P = (fact table T, fixed query procedure Q, declared composition operator ⊕) produced from k=10 labeled examples in a sandbox; a held-out composition test generator G; a closure threshold τ=0.95. Output: a certificate C(P) ∈ {pass, fail} together with a per-identity closure profile, and a re-emission request when fail.

**Load-bearing assumption (stated explicitly).** The certificate is valid only if the task's fact algebra has a declared binary composition operator ⊕ whose output on a pair (a,b) is not equal to the payload of any single row in T — i.e., the correct answer to a held-out composition requires combining two rows' payloads. If the domain is a pure lookup where Q(a,b) is nearest-row retrieval, then Q is trivially commutative and distractor-invariant and the identity probes certify the retrieval operator's symmetries, not closure. We therefore require the LLM to emit ⊕ alongside Q and restrict τ to tasks where the generator can construct held-out pairs whose composed value is provably absent from any single row (verified by the generator's unit test, below).

**How it works.**

```pseudocode
def closure_certificate(P=(T,Q,⊕), G, tau=0.95, max_repairs=3):
    # Phase 1: generate held-out compositions G never used to author P
    pairs    = G.sample_unseen_pairs(T, n=256)      # (a,b) with a,b in T, (a,b) not queried during authoring
    chains   = G.sample_chains(T, depth=3, n=256)   # a -> b -> c compositions
    distract = G.inject_distractors(T, n=256)       # add m=8 irrelevant rows, re-query known pairs
    composed = G.sample_composed_absent(T, n=256)   # pairs whose correct value is absent from any single row
    # Phase 2: probe the procedure's own composition rule on each family
    score = {}
    score['identity']   = agree(Q(a,b), Q(b,a)) for pairs          # commutativity of ⊕
    score['assoc']     = agree(Q(Q(a,b),c), Q(a,Q(b,c))) for chains # associativity on depth-3 chains
    score['invariance']= agree(Q(a,b), Q_distract(a,b)) for distract# distractor rows must not change the answer
    score['coverage']  = fraction of held-out pairs whose Q returns a defined (non-abstain) value
    score['rule']      = agree(⊕(a,b), Q(a,b)) for composed        # declared operator vs. procedure output
    C = min(score['identity'], score['assoc'], score['invariance'], score['coverage'], score['rule'])
    if C >= tau: return ('pass', score)
    # Phase 3: failure is fed back as a repair prompt; the architecture is NOT accepted
    for r in range(max_repairs):
        P = llm.reemit(P, score, failed_family=argmin(score))
        C, score = closure_certificate(P, G, tau, max_repairs=0)
        if C >= tau: return ('pass', score)
    return ('fail', score)
```

The generator G is built once per task domain from the same k=10 labeled examples by enumerating all fact atoms, forming the cross-product of atoms not co-occurring in any labeled example, and holding out a fixed 20% slice as the closure probe; it is never shown to the LLM during authoring. G's construction is a fixed script with a unit test that verifies (i) no probe pair co-occurs in any labeled example and (ii) every pair in `composed` has a correct value that is absent from every single row of T. τ is set to 0.95 on each identity family. The gate sits inside the free-design regime: only programs that pass are admitted to the accuracy evaluation, so closure becomes a search-time constraint rather than a post-hoc number. Total LLM calls per candidate are capped at 1 (authoring) + 3 (repairs) = 4, matching the RZ-NAS baseline's call budget.

**Why this design.** We chose a *generated* held-out composition probe over a held-out accuracy split because accuracy on new questions conflates composition with retrieval — a program can answer an unseen question by matching a near-duplicate row — whereas the algebraic identities (commutativity, associativity, distractor invariance) plus the declared-operator check are evaluated on the procedure's own outputs and on pairs whose correct value is absent from any single row, so they cannot be satisfied by row matching alone; the cost is that a program with a genuinely non-commutative but correct composition rule is penalized, which we accept by restricting τ to tasks whose fact algebra is commutative (ARC-style fact lookup). We chose to feed failures back as a *family-specific* repair prompt (naming which identity failed) over a generic retry because the failure profile localizes the defect to a single procedure branch, trading extra LLM calls (up to 3) for a much higher repair rate. We chose to gate *before* accuracy evaluation over gating after because a program that fails closure but scores well would otherwise dominate the search and re-introduce the memorization failure the certificate is meant to eliminate; the cost is that some high-accuracy programs are discarded. A domain expert would not describe this as a variant of RZ-NAS (E3) or NADER (E4): those methods use LLM reflection as a *heuristic* to improve a scalar score, whereas here the reflection is driven by a *certificate over algebraic identities of the emitted procedure*, and the accepted object is the certificate, not the score.

**Why it measures what we claim.** The identity-agreement score measures compositional closure because it assumes the fact algebra is commutative and associative on the task's true fact set — an assumption that fails when the domain has order-dependent facts (e.g., temporal chains), in which case the score reflects order sensitivity rather than closure. The distractor-invariance score measures algebraic well-definedness because it assumes irrelevant rows are truly irrelevant to every valid query — an assumption that fails when a distractor row is a near-duplicate of a relevant one, in which case the score reflects table redundancy rather than closure. The coverage score measures definedness of the composition operator because it assumes abstention on a held-out pair indicates an undefined rule rather than a conservative procedure — an assumption that fails when a correct procedure abstains on low-confidence pairs, in which case coverage reflects caution rather than closure. The rule-agreement score measures whether the emitted procedure implements its declared composition operator because it assumes the declared ⊕ is the task's true operator on the held-out composed pairs — an assumption that fails when the LLM declares a plausible but incorrect ⊕, in which case the score reflects operator mis-declaration rather than closure. Finally, the repair-loop pass rate measures whether closure is *achievable* by re-emission because it assumes the LLM can localize the failed identity from the family label — an assumption that fails when the defect is distributed across branches, in which case the pass rate reflects prompt-following rather than closure.

## Contribution

['A search-time closure certificate (CLOSURE-CERT) that operationalizes compositional closure of LLM-authored weight-space programs as a conjunction of algebraic identity checks over a held-out composition generator, admitted as a gate before accuracy evaluation.', 'A design principle: closure failures localize to a single identity family and can be repaired by family-conditioned re-emission, making compositional generalization a constraint the generative loop must satisfy rather than a post-hoc metric.', 'A held-out composition generator that synthesizes unseen pairs, depth-3 chains, and distractor-injected queries from the same k labeled examples used for authoring, without leaking probe facts to the LLM.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ARC multiple-choice fact-lookup tasks (k=10 examples) | Matches free-design regime where composition failure appears |
| Dataset (boundary) | Temporal-chain order-sensitive task (k=10 examples) | Tests graceful degradation where commutativity assumption fails |
| Primary metric | Accuracy on held-out compositional question subset, with minimum atom-distance ≥2 from labeled examples | Directly tests whether closure gating improved composition, not retrieval |
| Secondary metric | Retrieval-vs-composition accuracy decomposition per candidate | Verifies predicted gap is concentrated on compositional subset |
| Baseline | Free-design WSP, no gate (E6 setup) | Tests whether gating adds anything over raw LLM design |
| Baseline | Post-hoc accuracy gate (accept if held-out accuracy high) | Tests whether closure gate beats accuracy-only selection |
| Baseline | RZ-NAS-style reflective scalar feedback (E3) | Tests certificate-vs-score reflection distinction |
| Baseline | Generic metamorphic-relation test suite (random input perturbation + output-consistency checks) | Tests whether identity-family localization drives the gain |
| Ablation-of-ours | CLOSURE-CERT with repair loop disabled (fail = discard) | Isolates contribution of family-specific repair feedback |
| Ablation-of-ours | CLOSURE-CERT with rule-agreement probe removed | Isolates contribution of declared-operator check over identity symmetries |

### Why this setup validates the claim

The central claim is that a certificate over algebraic identities of the LLM-emitted procedure filters out programs that merely match near-duplicate rows, admitting instead programs whose composition rule is genuinely defined. The free-design WSP baseline (E6) tests the sub-claim that ungated search admits memorizing programs; if our gate matters, the gap must appear on compositional questions rather than on questions whose answers are near-duplicates of labeled examples. The post-hoc accuracy gate tests the sub-claim that accuracy alone cannot discriminate closure from retrieval — both gates see the same candidate pool, so any advantage must come from the certificate's access to procedure-level identities rather than to more data. The RZ-NAS-style reflective scalar baseline tests whether certificate-driven, family-localized feedback outperforms generic scalar-improvement reflection. The generic metamorphic-relation baseline tests whether the gain comes specifically from identity-family localization rather than from any output-consistency testing. The repair-disabled ablation isolates whether the pass-rate gain comes from re-emission feedback or merely from discarding bad programs. The rule-agreement ablation isolates whether the declared-operator check adds discrimination beyond the symmetry probes. The primary metric — accuracy on held-out compositional questions with minimum atom-distance ≥2 — is the right detector because the predicted effect is specifically compositional: a uniform accuracy gain across all subsets would be consistent with generic improvement and would not support the closure mechanism. The temporal-chain dataset tests the stated boundary: if the certificate degrades gracefully (lower pass rate but no false accepts) rather than collapsing, the scoped claim is supported.

### Expected outcome and causal chain

**vs. Free-design WSP, no gate** — On a case where a held-out question composes two facts whose atoms each appear separately in the ten labeled examples, the ungated baseline produces a program that answers by matching a near-duplicate row and returns the wrong composed value because its table has no defined composition rule. Our method instead rejects that program at the gate because its rule-agreement or associativity score collapses on unseen pairs, forcing re-emission toward a procedure with an explicit rule. We expect a noticeable accuracy gap concentrated on the compositional subset, with near-parity on retrieval-style questions.

**vs. Post-hoc accuracy gate** — On a case where a memorizing program happens to score well on the small held-out accuracy split because that split is also near-duplicate-heavy, the accuracy gate admits it and it then fails on genuinely novel compositions. Our method instead fails it on distractor-invariance, coverage, and rule-agreement probes that accuracy cannot see, because those probes query the procedure's own outputs and its declared operator rather than labels. We expect the accuracy gate to match us on retrieval subsets but lose several points on compositional subsets, with the gap widening as composition depth increases.

**vs. RZ-NAS-style reflective scalar feedback** — On a case where the emitted program has one defective branch, the scalar-feedback baseline re-emits with generic "improve performance" guidance and often perturbs unrelated branches, because its feedback carries no localization. Our method instead names the failed identity family and targets the single branch, so we expect a higher repair pass rate within the three re-emission budget and a smaller fraction of accepted programs that later fail composition probes.

**vs. Generic metamorphic-relation test suite** — On a case where a lookup program is trivially commutative and distractor-invariant, the generic metamorphic suite passes it because random input perturbations and output-consistency checks are satisfied by nearest-row retrieval. Our method instead fails it on the rule-agreement probe, because the held-out composed pairs require combining two rows' payloads and the declared ⊕ does not match Q's output. We expect the metamorphic baseline to match us on symmetry families but lose on the compositional subset, confirming that identity-family localization plus the declared-operator check drives the gain.

### What would falsify this idea

If the gain over the post-hoc accuracy gate is uniform across retrieval and compositional subsets rather than concentrated on compositional questions, the certificate is not doing what we claim and the central mechanism is unsupported. Likewise, if the repair-disabled ablation matches the full method, the family-specific feedback contributes nothing and closure is not achievable by re-emission as argued; and if the rule-agreement ablation matches the full method, the declared-operator check adds no discrimination beyond the symmetry probes.

## References

1. Weight-Space Programming Doesn't Need Transformers
