# Identifiability-Gated Target Admission with Typed Re-resolution for Tool-Using Agents

## Motivation

Runtime prompt-injection defenses that rely on provenance separation quietly assume that every legitimate action target is recoverable from a typed, provenance-tagged field — SPILLWAY, for instance, admits a call only when each target value appearing in tool output but not in the user task was returned by a read-only tool in a typed field or covered by a fixed delegation rule. When a legitimate target (a recipient named only in free text, a payee given in prose) is absent from typed fields, this admission rule cannot distinguish 'legitimate but untyped' from 'injected and untyped', so the defense either refuses useful actions or falls back to content inspection — the very assumption the background ancestor made and that SPILLWAY leaves unchecked.

## Key Insight

Target identifiability is a decidable precondition rather than an assumed property: an action is safe to admit iff its target is either typed-and-provenanced or resolvable to exactly one entry in a typed store by a deterministic, evidence-retaining resolution, so the admission floor becomes conditional on a checked predicate instead of on silent typed-field coverage.

## Method

**What it is.** IDRA (Identifiability-Gated Re-resolution Admission) takes as input the user task, the raw tool-return stream (typed fields plus free-text regions), and a typed contact/account store, and outputs an admission decision per proposed action together with an identifiability certificate naming the typed field or store entry that grounds the action's target. The typed store is a table with columns `key` (string), `entry_id` (string), `write_provenance` ∈ {`user_authored`, `readonly_tool_typed`, `untrusted`}, and `write_timestamp` (episode-relative integer). Entries are enrolled only from (i) user-authored text at task start, (ii) typed fields returned by read-only tools, or (iii) prior tool returns; entries from (iii) are tagged `untrusted` unless the returning tool is on a fixed read-only allowlist, in which case they are tagged `readonly_tool_typed`. The load-bearing assumption is stated explicitly: **A free-text action target that uniquely re-resolves to exactly one store entry is legitimate only if that entry's `write_provenance` is trusted (`user_authored` or `readonly_tool_typed`) and its `write_timestamp` is ≤ task start; unique resolvability is a valid proxy for non-injected provenance only under this store-integrity precondition.**

**How it works.**

```pseudocode
IDRA(task, tool_outputs, typed_store, proposed_action):
    # Phase 1: extract candidate targets from the action
    targets = extract_targets(proposed_action)          # recipient, payee, path, URL host

    # Phase 2: identifiability predicate per target
    for t in targets:
        if t in task.user_supplied_values:               # user-authored, trusted
            cert[t] = ("user_task", span_of(t, task), provenance="user_authored")
        elif typed_field_contains(tool_outputs, t):       # typed, provenance-tagged
            cert[t] = ("typed_field", field_id_of(t, tool_outputs), provenance=field_provenance(t))
        elif t in removed_spans(tool_outputs):            # target lies in sanitized text
            cert[t] = REFUSE                              # never restorable
        else:
            # Phase 3: mandatory typed-field re-resolution, gated on store integrity
            matches = resolve(t, typed_store, task.start_time)
            if len(matches) == 1:
                cert[t] = ("re_resolved", matches[0].entry_id,
                           edit_distance(t, matches[0].key),
                           provenance=matches[0].write_provenance,
                           write_timestamp=matches[0].write_timestamp)
            else:
                cert[t] = REFUSE                          # 0 or >1 matches: unidentifiable

    # Phase 4: admission
    if any(cert[t] is REFUSE for t in targets):
        return REFUSE, cert
    return ADMIT, cert

resolve(t, typed_store, task_start):
    # deterministic, no model call; store-integrity gate applied before matching
    trusted = [e for e in typed_store
               if e.write_provenance in {"user_authored", "readonly_tool_typed"}
               and e.write_timestamp <= task_start]
    candidates = [e for e in trusted if normalize(e.key) == normalize(t)]
    if len(candidates) == 0:
        candidates = [e for e in trusted
                      if edit_distance(normalize(t), normalize(e.key)) <= 2]
    return candidates
```

Hyperparameters: edit-distance threshold = 2 (set by a held-out validation split of legitimate-but-untyped targets; sensitivity reported for thresholds 0–3). Normalization strips case, whitespace, and common honorifics. Store entries with `write_provenance = untrusted` or `write_timestamp > task_start` are excluded from `resolve` entirely, so a poisoned contact written during the episode cannot be matched and the target falls to REFUSE with a certificate that records the excluded entry's provenance and timestamp as evidence.

**Why this design.** We chose a deterministic re-resolution step against a typed store over a model-based judge because the judge reintroduces the exact attack surface (a model reading attacker text) that SPILLWAY's admission check was designed to avoid, accepting the cost that re-resolution fails when the store lacks the target. We chose to refuse on ambiguous matches (multiple store entries) rather than pick the highest-similarity entry because ambiguous re-resolution is precisely the case where an attacker can steer a free-text target toward a plausible-but-wrong entry, accepting the cost of false refusals on genuinely ambiguous names. We chose to retain the removed-span evidence and make it non-restorable, matching SPILLWAY's sanitizer-evidence design, because it gives an auditable trace of why an action was refused, accepting the cost of storing spans. We chose to gate re-resolution on store-entry provenance and write timestamp rather than treat the store as trusted by default, because a store populated from tool returns is exactly the memory stream that ASB and DRIFT identify as attacker-influenceable; the cost is that legitimate entries written during the episode (e.g., a contact added mid-task by the user) are refused until re-enrolled with trusted provenance. A domain expert would not describe IDRA as a variant of SPILLWAY: SPILLWAY's admission predicate presupposes typed-field coverage, whereas IDRA makes coverage a checked predicate with an explicit re-resolution fallback and an explicit store-integrity precondition, so the two differ in what happens when a legitimate target is untyped — the case SPILLWAY cannot handle.

**Why it measures what we claim.** The identifiability certificate measures target identifiability because the assumption that a target recoverable from a typed field or a unique trusted store entry is equivalent to a legitimate (non-injected) target holds only when the typed store is itself trusted and the user task is the sole source of legitimate targets; this assumption fails when the attacker can write to the typed store (e.g., a poisoned contact written during the episode), in which case the certificate reflects store integrity rather than target legitimacy — the store-integrity gate in `resolve` makes this failure visible by refusing the poisoned entry and recording its provenance and timestamp in the certificate rather than silently admitting it. The edit-distance threshold measures the boundary between typo-tolerant matching and attacker-steerable fuzzy matching because the assumption that legitimate untyped targets are within edit distance 2 of their store entry holds only for natural-language name variants; this assumption fails for abbreviations or nicknames, in which case re-resolution refuses a legitimate action and the metric reflects conservative refusal rather than identifiability.

## Contribution

["IDRA, an admission framework that replaces SPILLWAY's presupposed typed-field coverage with an explicit identifiability predicate and a deterministic typed-store re-resolution fallback, producing a per-action identifiability certificate.", 'A design principle: refusing on ambiguous re-resolution (0 or >1 matches) is necessary to prevent attacker-steered free-text targets from resolving to plausible-but-wrong store entries, at the cost of false refusals on genuinely ambiguous names.', 'An audit artifact: the identifiability certificate records the grounding source (user task, typed field, or re-resolved store entry) for every admitted action.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AgentDojo (standard + PISmith regimes) | Injection targets on typed tools, plus adaptive attacker |
| Dataset | ASB scenarios (e-commerce, finance, memory-retrieval) | Second tool-use benchmark; includes memory-poisoning attack |
| Primary metric | In-scope ASR at matched utility | Separates target-redirection failures from utility loss |
| Secondary metric | Untyped-legitimate-target prevalence (fraction of benign tasks whose action target is absent from typed fields) | Establishes the gap is first-order, not a corner case |
| Secondary metric | Poisoned-store ASR (attacker writes a contact entry during the episode) | Tests store-integrity gate degrades gracefully |
| Baseline | SPILLWAY | Tests whether re-resolution fallback beats typed-only admission |
| Baseline | DRIFT | Tests whether deterministic gating beats plan-deviation validation |
| Baseline | MELON | Tests whether certificate gating beats masked re-execution detection |
| Ablation-of-ours | IDRA without re-resolution (typed-field only) | Isolates the contribution of the fallback phase |
| Ablation-of-ours | IDRA without store-integrity gate (trusts all store entries) | Isolates the contribution of the provenance/timestamp precondition |
| Ablation-of-ours | IDRA with edit-distance threshold ∈ {0,1,2,3} | Reports sensitivity; threshold pre-registered on held-out split |

### Why this setup validates the claim

The central claim is that making typed-field coverage a checked predicate with a deterministic re-resolution fallback admits legitimate untyped targets that SPILLWAY refuses, without opening a redirection channel, and that this admission is sound only under a checked store-integrity precondition. AgentDojo's standard and PISmith regimes are the right stress test because they place attacker-controlled free text in tool returns and then attempt to redirect an action's target — exactly the boundary IDRA's certificate is supposed to police. ASB adds a memory-poisoning attack and a memory-retrieval stage, which is the concrete failure mode the store-integrity gate is meant to block. In-scope ASR at matched utility is the correct metric because a defense that refuses everything trivially lowers ASR; conditioning on utility forces the comparison onto the region where the claim lives. The untyped-legitimate-target prevalence metric establishes whether the coverage gap is first-order. SPILLWAY tests the sub-claim about coverage gaps (does re-resolution recover legitimate actions SPILLWAY refuses?), DRIFT tests the sub-claim about determinism (does a certificate gate avoid plan-deviation blind spots?), and MELON tests the sub-claim about attack-surface reintroduction (does a non-model check beat a re-execution detector?). The typed-field-only ablation is the load-bearing control for re-resolution: if IDRA's gain over SPILLWAY comes from re-resolution, removing it should collapse IDRA back to SPILLWAY-like behavior on the untyped subset while leaving the typed subset unchanged. The no-store-integrity-gate ablation is the load-bearing control for the store-integrity precondition: if the gate matters, removing it should raise poisoned-store ASR while leaving clean-store ASR unchanged. That pair of patterns is the falsifiable signature.

### Expected outcome and causal chain

**vs. SPILLWAY** — On a case where a legitimate payee arrives as free text ("send to Dana Whitfield") but the typed store has a unique trusted `dana.whitfield@…` entry, SPILLWAY refuses because its admission predicate requires typed-field coverage and no fixed delegation rule applies. Our method instead re-resolves the free-text target against the trusted subset of the typed store, finds a unique normalized-key match, and admits with a `re_resolved` certificate recording the entry's provenance and timestamp. We expect a visible utility gain on the untyped-legitimate subset with ASR parity on the typed subset — i.e., the gap should be concentrated where SPILLWAY's refusal floor bites, not spread uniformly.

**vs. DRIFT** — On a case where the attacker appends an instruction to a tool return that redirects a payment to an attacker account while staying within the planner's privilege envelope, DRIFT's Dynamic Validator sees a plan-consistent deviation and admits because its check compares against the user query rather than grounding the target value. Our method instead reads the payee's provenance: it is neither user-supplied nor a typed field nor a unique trusted store entry, so it lands in REFUSE with a certificate naming the ungrounded target. We expect a noticeable ASR reduction on target-redirection attacks that preserve plan shape, with utility parity on benign tasks whose targets resolve cleanly.

**vs. MELON** — On a case where the injected task is short and stylistically similar to the user's task, MELON's masked re-execution produces actions close to the original and its similarity test misses the attack, because detection depends on a behavioral divergence that a careful attacker suppresses. Our method instead never inspects behavior: the attacker's target fails identifiability regardless of how similar the trajectory looks, so it is refused by construction. We expect MELON's ASR to rise under PISmith's adaptive variants while IDRA's stays near its floor, with the divergence appearing specifically on low-divergence injections rather than across the board.

**vs. poisoned-store attacker (ASB memory-poisoning)** — On a case where the attacker injects a contact entry `Dana Whitfield <attacker@evil>` into the store during the episode, the no-store-integrity-gate ablation admits the attacker's target because unique resolvability is treated as sufficient. Full IDRA instead excludes the entry (provenance `untrusted` or `write_timestamp > task_start`), refuses the action, and records the excluded entry's provenance and timestamp in the certificate. We expect poisoned-store ASR near zero for full IDRA and elevated for the no-gate ablation, with clean-store ASR unchanged between the two — the signature that the gate, not re-resolution, is what blocks the poisoned-store channel.

### What would falsify this idea

If IDRA's gain over SPILLWAY is uniform across typed and untyped subsets rather than concentrated on the untyped-legitimate subset, then re-resolution is not the causal mechanism and the coverage-gap claim is wrong. Likewise, if the typed-field-only ablation matches full IDRA on ASR under PISmith, the re-resolution phase is an attack surface rather than a recovery mechanism; and if the no-store-integrity-gate ablation matches full IDRA on poisoned-store ASR, the store-integrity precondition is not load-bearing and the central claim fails.

## References

1. SPILLWAY: Separating Target Admission from Sanitization to Defend Tool-Using Agents against Prompt Injection
