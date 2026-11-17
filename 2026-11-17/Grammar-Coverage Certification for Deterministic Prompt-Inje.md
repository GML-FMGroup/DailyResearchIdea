# Grammar-Coverage Certification for Deterministic Prompt-Injection Sanitizers

## Motivation

Runtime defenses increasingly rely on a deterministic sanitizer that strips instruction-like spans by fixed pattern rules before the model reads tool output—SPILLWAY is the clearest instance, and its own framing concedes that the sanitizer is a fixed-rule component with no stated relation between what its rules match and what an attacker can actually write. The structural gap is that the sanitizer is evaluated only empirically (zero ASR on templated injections), so its protection is a property of the observed attack distribution rather than of the space of injectable instructions; a novel format outside the rule set is silently passed through, and nothing in the pipeline reports that this has happened. This recurs across the tree: Spotlighting-style boundary marking and PI detectors make the same implicit assumption that a fixed surface transformation covers the attacker's instruction space, so the field has no notion of sanitizer coverage at all.

## Key Insight

If the injectable-instruction surface is defined as a formal grammar over tool-output text and each sanitizer rule is compiled to an automaton, then language inclusion between the rule union and the grammar is a decidable check, so the sanitizer's coverage becomes a certified quantity and every uncovered span is localized rather than silently admitted.

## Method

**What it is.** GRAMMAR-COVERAGE CERTIFICATION (GCC) takes a deterministic sanitizer's fixed rule set R and an instruction grammar G (a regular grammar over tool-output tokens, authored per deployment from the tool's output schema and the injection formats the operator intends to exclude) and returns a three-valued verdict per input span—COVERED, RESIDUAL, or OUT-OF-GRAMMAR—plus a machine-checkable coverage certificate for R relative to G. Inputs: rule set R, grammar G, raw tool-output string s. Outputs: sanitized string s', the set of removed spans with their verdicts, and a residual-risk certificate enumerating the residual language L(G) \ L(R).

**Load-bearing assumption (stated explicitly).** The certificate is conditional on G: it certifies that R covers L(G), not that R covers the attacker's full instruction space. Soundness of the over-approximation (every attacker-writable instruction lies in L(G)) is assumed, not proven; the OUT-OF-GRAMMAR verdict and the measured OUT-OF-GRAMMAR rate on held-out attack families are the honest signals when this assumption fails. COVERED_ALL is claimed only for the typed-field sublanguage of G where regularity is defensible; the free-text wildcard class is explicitly uncertifiable and any span matched only by that class is reported as OUT-OF-GRAMMAR rather than COVERED.

**How it works.**

```pseudocode
# Phase 1: compile rules and grammar to automata (once per deployment)
A_R = union_of_dfas(compile_regex_to_dfa(r) for r in R)      # sanitizer coverage automaton
A_G = dfa_from_regular_grammar(G)                            # injectable-instruction surface
# G construction: typed-field subgrammar G_typed from tool schema (enumerated field names,
# value patterns, delimiters) unioned with free-text wildcard class G_free; A_G = A_G_typed ∪ A_G_free

# Phase 2: coverage certification (once per (R, G) pair)
A_res = product_minus(A_G, A_R)          # DFA for L(G) \ L(R): uncovered instruction language
if A_res.accepts_empty():
    certificate = COVERED_ALL           # R recognizes a superset of L(G)
else:
    witnesses = shortest_paths(A_res, k=5)   # k=5 shortest uncovered instruction strings
    certificate = RESIDUAL(witnesses, A_res)
# Additionally compute A_res_typed = product_minus(A_G_typed, A_R) and report
# COVERED_ALL_TYPED separately; free-text coverage is never certified.

# Phase 3: per-input sanitization with verdicts
def sanitize(s, R, A_G, A_res):
    spans = scan_with_rules(s, R)                 # fixed-rule matches, as in SPILLWAY
    removed = [sp for sp in spans if sp.matched]
    verdicts = {}
    for span in removed:
        verdicts[span] = COVERED if A_R.accepts(span.text) else RESIDUAL
    residual = [w for w in find_substrings(s, A_res) if w not in removed]
    for w in residual:
        verdicts[w] = RESIDUAL
    for tok in tokens(s):
        if tok not in L(A_G) and tok not in removed:
            verdicts[tok] = OUT_OF_GRAMMAR
    return strip(s, removed), verdicts, certificate
```

Hyperparameters: k=5 witness strings per residual certificate; grammar G is a regular grammar with a fixed token alphabet derived from the tool's typed-field schema plus a free-text wildcard class; no model calls anywhere in the certification path. Automata toolkit: automata-lib (Python) for DFA construction and product; regex-to-DFA via Thompson construction followed by subset construction, with a hard cap of 10,000 DFA states per rule and a fallback to a coarse character-class approximation (reported in the certificate) if the cap is exceeded. Compilation cost and DFA state counts are logged per (R, G) pair.

**Why this design.** We chose a regular grammar G rather than a context-free or LLM-authored grammar because language inclusion against a DFA is decidable in polynomial time, accepting the cost that some nested or long-range instruction formats fall outside G and are reported as OUT-OF-GRAMMAR rather than certified—this is the right trade because the certificate's value depends on being checkable without a model. We chose to compile sanitizer rules to automata and take language inclusion rather than sampling attacks and measuring ASR because sampling measures the attack distribution while inclusion measures the instruction space, accepting the cost that a grammar the operator writes badly yields a vacuous certificate. We chose to localize residual spans per input and emit them as evidence rather than to refuse inputs wholesale because SPILLWAY's auditability property is worth preserving, accepting the cost that downstream consumers must act on RESIDUAL verdicts. We chose a fixed k=5 witness enumeration over exhaustive residual enumeration because the residual language can be infinite, accepting the cost that witnesses are illustrative rather than complete. A domain expert would not describe this as a variant of SPILLWAY: SPILLWAY's sanitizer is a fixed-rule stripper with no relation to an instruction grammar, whereas GCC's load-bearing object is the inclusion check between two automata, and the sanitizer's rules are an input to a decision procedure rather than the defense itself.

**Why it measures what we claim.** The language-inclusion verdict `A_res.accepts_empty()` measures coverage of the injectable-instruction surface by the sanitizer's rules because it is equivalent to L(G) ⊆ L(R) under the assumption that G is a sound over-approximation of the attacker's instruction space (every instruction an attacker can write is in L(G)); this assumption fails when the attacker writes an instruction outside G, in which case the certificate reflects coverage of G only and the OUT-OF-GRAMMAR verdict is the honest signal rather than a guarantee. The per-span verdict `A_R.accepts(span.text)` measures whether a removed span was in the sanitizer's rule language because rule compilation is exact, under the assumption that the regex-to-DFA compilation preserves the deployed rule semantics; this fails when rules use backreferences or lookaround that the DFA compilation approximates, in which case the verdict reflects the compiled approximation rather than the deployed rule. The residual witness set measures the residual risk surface because shortest paths through A_res enumerate the cheapest uncovered instructions under the assumption that attacker cost correlates with string length; this fails when short instructions are not the easiest to inject, in which case the witnesses illustrate the residual language rather than rank attacker difficulty. The stratified ASR by verdict measures whether RESIDUAL verdicts predict executed injections because it conditions attack success on the certificate's own labels under the assumption that attacker success correlates with membership in L(G)\L(R); this fails when attackers write instructions outside G or exploit non-linguistic channels (encoded payloads, tool-parameter smuggling), in which case coverage can be high while ASR remains nonzero and the certificate reflects grammar coverage rather than injection resistance.

## Contribution

(1) A certification procedure that compiles a deterministic sanitizer's fixed rules and a deployment-authored regular instruction grammar into automata and decides language inclusion, returning either a full-coverage certificate or a residual-risk certificate with witness strings. (2) A three-valued per-span verdict scheme (COVERED / RESIDUAL / OUT-OF-GRAMMAR) that turns the sanitizer from an opaque stripper into an auditable component whose uncovered surface is localized on every input. (3) A design principle for prompt-injection sanitizers: coverage of the instruction space is a checkable property of the rule set relative to a declared grammar, and should be reported alongside sanitization rather than inferred from attack-success benchmarks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | AgentDojo suites with SPILLWAY-style typed tool returns | Typed fields separate targets from free text |
| Primary metric | Residual-language coverage of G by R | Directly measures the certificate's claim |
| Secondary metric | ASR stratified by COVERED / RESIDUAL / OUT-OF-GRAMMAR verdicts | Tests whether RESIDUAL verdicts predict executed injections |
| Secondary metric | OUT-OF-GRAMMAR rate on held-out attack families (ChatInject, PISmith variants) | Quantifies soundness of the over-approximation |
| Baseline | SPILLWAY | Fixed-rule sanitizer without grammar inclusion check |
| Baseline | DRIFT | Dynamic rule validator with no coverage certificate |
| Baseline | MELON | Re-execution detection, not rule-space coverage |
| Ablation-of-ours | GCC without inclusion check (rules only) | Isolates the certificate's contribution |

### Why this setup validates the claim

The central claim is that GCC converts a fixed-rule sanitizer into a checkable coverage guarantee over an instruction grammar, and that this guarantee—not the sanitizer itself—is what closes residual injection paths. AgentDojo with typed tool returns is the right dataset because SPILLWAY's own evidence shows target values arrive in typed fields while injected instructions arrive as free text, giving us a clean substrate where the grammar G is well-defined and the residual language is meaningful. The primary metric is residual-language coverage rather than attack success rate because ASR samples the attacker distribution while coverage measures the instruction space; if GCC's inclusion check is sound, coverage should predict which attack families slip through. The secondary metrics are necessary because coverage alone is self-referential: stratified ASR by verdict tests whether RESIDUAL spans actually execute, and the OUT-OF-GRAMMAR rate on held-out families tests whether G soundly bounds the attacker's writable space. SPILLWAY tests whether a fixed-rule sanitizer alone suffices; DRIFT tests whether dynamic validation without a certificate catches what static rules miss; MELON tests whether re-execution detection substitutes for rule-space analysis. Each baseline probes a different sub-claim: that fixed rules are enough (SPILLWAY), that runtime deviation monitoring is enough (DRIFT), and that behavioral detection is enough (MELON). The ablation—GCC with the inclusion check removed—is the decisive control: if the certificate contributes nothing, the ablation should match full GCC; if the certificate is load-bearing, removing it should produce residual injections that full GCC flags as RESIDUAL before they execute.

### Expected outcome and causal chain

**vs. SPILLWAY** — On a case where an injected instruction is phrased in a format the fixed rule set does not match (e.g., an instruction-like string with unusual punctuation or a synonym for a tool name), SPILLWAY's sanitizer leaves it in place because its rules are a fixed enumeration with no relation to an instruction grammar, so the agent reads and may act on it. GCC instead reports the span as RESIDUAL or OUT-OF-GRAMMAR against A_G, because the inclusion check between A_R and A_G surfaces exactly the strings the rules miss. We expect a noticeable gap on the subset of injections whose surface form departs from the rule set's templates, with near-parity on injections that match templated formats. Stratified ASR should show that RESIDUAL-flagged spans have higher ASR than COVERED spans under SPILLWAY, and that GCC's early flagging reduces executed injections on the RESIDUAL subset.

**vs. DRIFT** — On a case where a malicious instruction arrives through memory retrieval and is semantically consistent with the user's task (so the dynamic validator sees no privilege or intent deviation), DRIFT admits it because its validator monitors plan deviation rather than the instruction language itself. GCC instead flags the span against A_res because the coverage certificate is computed over the grammar independent of whether the instruction looks plan-consistent. We expect a gap concentrated on memory-stream injections that preserve apparent task intent, with parity on injections that visibly deviate from the planned function trajectory.

**vs. MELON** — On a case where a short injected instruction redirects the agent to an action that is also plausible under the masked user prompt (so re-execution produces similar actions), MELON's similarity check fails to flag the attack because masking does not change the generated action. GCC instead marks the instruction span as RESIDUAL against A_R regardless of the agent's behavioral response, because the verdict is computed over the string, not the trajectory. We expect a gap on short, task-plausible injections, with parity on injections whose actions diverge sharply under masking.

### What would falsify this idea

If full GCC and the no-inclusion-check ablation produce indistinguishable residual-injection rates across the subsets where the certificate predicts RESIDUAL spans—i.e., the certificate's verdicts do not concentrate failures where the predicted failure mode occurs—then the inclusion check is not load-bearing and the central claim is wrong. A uniform gain across all subsets, rather than a gain concentrated on grammar-residual injections, would likewise indicate the effect comes from something other than coverage certification. Additionally, if the OUT-OF-GRAMMAR rate on held-out attack families (ChatInject, PISmith) is high (e.g., >30%) while ASR remains nonzero on COVERED spans, the soundness assumption fails and the certificate reflects grammar coverage rather than injection resistance.

## References

1. SPILLWAY: Separating Target Admission from Sanitization to Defend Tool-Using Agents against Prompt Injection
