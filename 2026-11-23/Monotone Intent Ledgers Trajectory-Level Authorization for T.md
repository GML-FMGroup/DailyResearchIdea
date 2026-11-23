# Monotone Intent Ledgers: Trajectory-Level Authorization for Tool-Using LLM Agents

## Motivation

Per-step authorization interfaces such as Intent Contracts (E6) validate each contract in isolation, so a planner can escalate harm across a trajectory through a sequence of individually-valid contracts that each widen scope incrementally; the frontier's trusted-dispatch split relocates authority per call but leaves cross-step composition unconstrained. This is a structural gap: any per-step checker that does not maintain a stateful relation between the current contract and a single user-authorized root objective cannot rule out escalation spread over several steps, and no existing benchmark (AgentHarm, SafeArena, E1/E4/E7/E11) or guardrail agent (GuardAgent, E5/E10) enforces such a trajectory-level invariant. The problem is meta-level: it recurs wherever sequential authorization is composed without a composition rule.

## Key Insight

A monotone containment invariant over a stateful ledger — each new contract's goal must be a refinement of the single root objective established at trajectory start, with any scope widening requiring explicit re-authorization — makes cross-step escalation a checkable property of the ledger rather than an emergent property of individually-valid steps.

## Method

**A. What it is.** We introduce **MIL (Monotone Intent Ledger)**, a trajectory-level authorization layer for tool-using LLM agents. Inputs: the user's initial request and, at each step, a planner-submitted Intent Contract (goal, verbatim user quotation, scope fields) as in E6. Outputs: a per-step admit/reject/re-authorize decision plus an append-only ledger record; the trusted dispatcher writes the tool call only when the ledger's monotone containment check passes.

**B. How it works.**

```pseudocode
# State: L = [c_0]  (root contract, fixed at trajectory start)
#       root = c_0.goal; root_scope = c_0.scope
#       auth_tokens = {}  (scope-widening approvals tied to verbatim user quotes)
#       containment_calibration = human-labeled set of 512 (contract, root) pairs
#                                with REFINES/EQUAL/WIDENS labels (see D)

on_step(contract c_t, user_utterance u_t):
    # 1. Verbatim-quote check (inherited from E6)
    if c_t.quotation not substring_of (u_0 + u_t): reject
    # 2. Containment check against root, NOT against previous step
    #    Trusted model emits (verdict, cited_scope_elements) where cited_scope_elements
    #    is the list of root_scope fields it believes c_t refines.
    (verdict, cited) = trusted_model.check_containment(c_t.goal, c_t.scope, root, root_scope)
    #    verdict in {REFINES, EQUAL, WIDENS, ABSTAIN}
    #    ABSTAIN is returned when the model's confidence < 0.7 (calibrated on containment_calibration)
    # 3. Structured verification of cited elements against machine-comparable scope fields
    if verdict in {REFINES, EQUAL}:
        if not all(e in root_scope.typed_fields for e in cited):
            verdict = ABSTAIN   # cited element not in root scope -> cannot verify containment
    if verdict == REFINES or verdict == EQUAL:
        L.append(c_t); dispatch(c_t)
    elif verdict == WIDENS or verdict == ABSTAIN:
        if c_t.quotation in auth_tokens:        # prior explicit user approval
            L.append(c_t); dispatch(c_t)
        else:
            ask_user_for_explicit_widening_approval(c_t)
            if approved: auth_tokens.add(c_t.quotation); L.append(c_t); dispatch(c_t)
            else: reject(c_t); L.append(marked_rejected(c_t))
    # 4. Ledger audit (offline / per-trajectory)
    assert all(is_refinement(L[i].goal, root) or L[i].quotation in auth_tokens for i in range(len(L)))
```

**C. Why this design.** We chose a *single root objective* as the containment anchor over chaining each contract to its immediate predecessor because chaining-to-predecessor permits drift (each step refines the last, so a sequence of small widenings composes into large escalation), accepting the cost that a genuinely legitimate mid-trajectory goal change forces an explicit user re-authorization. We chose a *small trusted model* for the containment check over a rule-based scope-subsumption checker because scope fields are natural-language and rule-based subsumption is brittle on paraphrase, accepting the cost that the trusted model is itself an LLM whose containment judgments must be audited. We chose *append-only logging with offline assertion* over blocking dispatch on a full ledger replay because replay at every step would multiply trusted-model calls per step, accepting the cost that some escalation is detected post-hoc rather than prevented. A domain expert would not describe MIL as a variant of Intent Contracts (E6): E6's mechanism is per-step contract validation with no stateful composition rule, whereas MIL's load-bearing mechanism is the monotone containment invariant over a ledger — a property E6's per-step checker structurally cannot express.

**D. Why it measures what we claim.** The containment verdict `REFINES/EQUAL/WIDENS` measures *absence of cross-step escalation* because we assume that refinement of the root goal is a sufficient condition for a step to lie within the user-authorized objective; this assumption fails when the root goal is itself under-specified and a step legitimately narrows it in a way that later enables escalation, in which case the verdict reflects local refinement rather than global safety. The `auth_tokens` set measures *explicit user authorization of scope widening* because we assume a verbatim user quotation is a faithful record of consent; this assumption fails when the user's utterance is ambiguous, in which case the token reflects surface-string match rather than informed consent. The offline ledger assertion measures *trajectory-level monotone containment* because we assume the ledger is complete (every dispatched call has a logged contract); this assumption fails if the dispatcher can be invoked outside the ledger, in which case the assertion reflects only the logged subset. The structured verification of `cited_scope_elements` measures *verifiable containment* because we assume that the trusted model's cited root-scope elements are the actual basis for its verdict; this assumption fails when the model cites elements that do not appear in its internal reasoning, in which case the verification reflects post-hoc rationalization rather than the decision procedure. The `ABSTAIN` band measures *calibrated uncertainty routing* because we assume the model's confidence is calibrated on `containment_calibration`; this assumption fails when the calibration set is not representative of AmPermBench contract pairs, in which case the abstain band reflects calibration-set bias rather than true uncertainty. The primary metric (cross-step escalation rate) measures *trajectory-level safety* because we assume that root-scope containment of every changed object is necessary and sufficient for trajectory-level safety (i.e., no harm arises from in-scope object changes composed across steps); this assumption fails when a sequence of individually in-scope changes to the same object jointly constitutes harm, in which case the metric reflects per-step containment rather than global harm.

## Contribution

['A trajectory-level authorization object — the Monotone Intent Ledger — with a monotone containment invariant (refinement of a single root objective, with scope widening gated on explicit re-authorization) that makes cross-step escalation a checkable property of the ledger.', 'A design principle: cross-step escalation in tool-using agents is prevented by anchoring every step to the root objective rather than chaining to the previous step, and this anchoring is what per-step contract validation structurally cannot provide.', 'An audit procedure (offline ledger assertion) that turns the invariant into a per-trajectory safety certificate, together with a re-authorization protocol tied to verbatim user quotes.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | AmPermBench task families (E6) | Tool-use tasks with measurable object-change provenance |
| Dataset (failure-mode family) | AmPermBench-F: new family instantiating in-scope composition harm (see D) | Tests assumption A: in-scope changes composing to harm |
| Primary metric | Cross-step escalation rate: dispatched calls whose changed objects escape root scope | Directly measures monotone containment violation |
| Secondary metric | Containment-check error rate: REFINES/WIDENS misclassification vs. human labels on 512-pair calibration set | Quantifies soundness of the trusted-model oracle |
| Baseline | Per-step Intent Contracts reviewer (E6) | Tests whether stateful ledger beats stateless checking |
| Baseline | Reasoning-blind permission classifier | Tests whether goal/scope visibility matters |
| Baseline | Goal+quotation reviewer | Strongest per-step reviewer; isolates ledger contribution |
| Ablation-of-ours | MIL without offline ledger assertion | Tests whether append-only audit adds detection beyond gating |
| Ablation-of-ours | MIL with injected false-REFINES rate swept 0%–20% | Tests how escalation-rate results degrade under oracle error |
| Ablation-of-ours | MIL without structured verification of cited_scope_elements | Tests whether verifiable containment adds beyond raw LLM verdict |

### Why this setup validates the claim

The central claim is that a monotone containment invariant anchored to a single root objective prevents cross-step escalation that per-step contract validation cannot express. AmPermBench is the right substrate because it records which objects each dispatched call changes, letting us compute escalation directly rather than through proxy harm labels. The three baselines decompose the claim: the reasoning-blind classifier tests whether scope information is necessary at all; the goal+quotation reviewer is the strongest per-step checker, so beating it isolates the ledger as the load-bearing mechanism rather than richer reviewer inputs; the Intent Contracts reviewer tests the specific structural claim that per-step validation without a composition rule is insufficient. The ablation removes the offline assertion to separate prevention (gating) from detection (audit), so a gain concentrated in post-hoc detection does not get credited to the invariant. The primary metric is the right one because it is exactly the quantity the monotone invariant bounds; a method that improves other safety scores while leaving escalation rate flat would not support the claim. The containment-check error rate and the injected-false-REFINES sweep directly measure the load-bearing assumption that the trusted model's containment verdict is a sound proxy for trajectory safety, converting the guarantee from assumed to quantified. The AmPermBench-F family instantiates the named failure mode of assumption A (in-scope composition harm), so a flat escalation rate there would show the metric bounds what it claims to bound.

### Expected outcome and causal chain

**vs. Per-step Intent Contracts reviewer (E6)** — On a case where the planner issues a sequence of individually plausible contracts, each quoting a real user phrase but each widening scope slightly, the baseline admits every step because each contract passes local validation against the user's words with no memory of prior scope. Our method instead checks each contract against the fixed root scope, so the first widening triggers an explicit user re-authorization gate. We expect a noticeable escalation-rate gap concentrated on multi-step task families where drift requires three or more steps, and near-parity on single-step families where local and global checks coincide.

**vs. Reasoning-blind permission classifier** — On a case where the user's words are ambiguous and the planner's command changes an object the user never named, the baseline admits the call because it sees only surface user text and the command, missing the planner's inferred goal. Our method instead requires the contract's goal and quotation to be contained in the root scope before the trusted dispatcher writes the call. We expect a large gap on ambiguous-request subsets and smaller gaps on explicit-request subsets, since explicit requests leave little room for the classifier to fail.

**vs. Goal+quotation reviewer** — On a case where the planner composes several individually narrow steps into a broad effect, the baseline admits each step because it evaluates goal and quotation per step with no cross-step composition rule. Our method instead rejects any step whose scope is not a refinement of the root, regardless of how plausible the local goal looks. We expect a modest but consistent gap on composition-heavy families and parity elsewhere, since this baseline already captures most single-step risk.

**vs. MIL without offline ledger assertion** — On a case where the planner's contract is admitted by the gating check but the ledger is not audited, the baseline (no assertion) cannot detect post-hoc that a dispatched call's changed object escaped root scope. Our method's offline assertion flags such cases after the trajectory. We expect the ablation to show higher undetected escalation rate, isolating the audit's independent contribution from gating.

**vs. MIL with injected false-REFINES rate swept 0%–20%** — On a case where the trusted model mislabels a WIDENS contract as REFINES, the baseline (0% injection) admits only true refinements, while higher injection rates admit widening steps. We expect escalation rate to increase monotonically with false-REFINES rate, quantifying how the invariant's soundness degrades under oracle error and bounding the claim's robustness.

**vs. MIL without structured verification of cited_scope_elements** — On a case where the trusted model returns REFINES but cites a root-scope element not present in the typed scope fields, the baseline (no verification) admits the call, while our method routes to ABSTAIN and re-authorization. We expect a small but consistent reduction in escalation rate on ambiguous contracts, showing the structured check adds beyond the raw LLM verdict.

### What would falsify this idea

If MIL's escalation-rate advantage over the goal+quotation reviewer is uniform across task families rather than concentrated on multi-step composition-heavy families where per-step checking structurally cannot see drift, the claim that the ledger invariant is the load-bearing mechanism is wrong. Likewise, if removing the offline ledger assertion leaves escalation rate unchanged while only the gating ablation moves it, the append-only audit is doing no independent work and the monotone-invariant framing collapses to per-step checking. If the containment-check error rate on the 512-pair calibration set exceeds 15% (false-REFINES) and the injected-error sweep shows escalation rate rising above the goal+quotation reviewer's at that error level, the load-bearing assumption that the trusted model is a sound containment oracle is falsified.

## References

1. Intent Contracts: Taking Tools Away from Agents
