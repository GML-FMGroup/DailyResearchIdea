# Precondition-Indexed Skill Certificates with Revocation Semantics for Drift-Aware Computer-Use Agents

## Motivation

Across the computer-use agent literature, a skill artifact or synthesized environment is validated exactly once at construction time and then treated as static: VISUALSKILL validates that a retrieved topic faithfully represents the live UI/docs at build time, and ASI programmatically verifies induced skills before caching them, yet neither certificate records which environmental predicates the artifact's execution actually depended on. When the UI, API, or documentation drifts, the certificate remains 'valid' while the artifact silently becomes unusable, so the agent's subsequent failure is indistinguishable from genuine agent incompetence — a structural blindness that recurs in every skill-store and environment-synthesis tree. The root cause is that the certificate is a scalar verdict over a static artifact rather than a function of the artifact's operational preconditions, so no execution-time observation can be mapped back to a specific revoked dependency.

## Key Insight

If a certificate is constructed as a set of precondition-indexed obligations — each bound to a concrete, runtime-observable UI/API/doc predicate that the artifact's execution demonstrably touched during construction — then any single observed predicate violation is a certificate revocation event that localizes artifact staleness separately from agent failure, turning one-shot validation into a drift-detectable invariant without any new model or classifier.

## Method

**What it is.** PRISM (Precondition-Indexed Skill Certification) takes a skill artifact (a VISUALSKILL-style per-topic file, or an ASI-style induced program) plus the construction-time execution trace that produced it, and emits a *precondition-indexed certificate*: a set of obligations `{(p_i, witness_i, action_i, necessity_i)}` where `p_i` is a runtime-observable predicate over the live environment, `witness_i` is the construction-time observation that grounded it, `action_i` is the artifact step that touched it, and `necessity_i` is a boolean flag recording whether the predicate passed the construction-time necessity test (defined below). At execution time PRISM re-evaluates the obligations against the live environment and emits a revocation event per violated `p_i` before the artifact is used.

**How it works.**

```pseudocode
# ---- Construction phase ----
def certify(artifact, trace, env, sandbox):
    obligations = []
    for step in trace.steps:                       # each step: action + obs
        touched = extract_env_reads(step)          # UI nodes, API fields, doc anchors read
        for r in touched:
            pred = predicate_for(r, env)            # e.g. exists(selector), schema_has(field)
            if pred.holds(env):
                obligations.append( (pred, r.witness, step.index) )
    obligations = dedup_and_minimize(obligations)   # drop predicates implied by others
    # ---- Necessity test (closes the lucky-path gap) ----
    for (pred, witness, step_idx) in obligations:
        necessity_i = necessity_test(artifact, pred, sandbox)
    return Certificate(artifact.id, obligations, env.fingerprint)

def necessity_test(artifact, pred, sandbox):
    # Counterfactual removal: perturb the sandbox so pred is false,
    # then replay the artifact and check whether execution fails.
    sandbox_perturbed = sandbox.perturb(pred)       # e.g. rename selector, drop schema field
    result = replay(artifact, sandbox_perturbed)
    return not result.succeeded                    # True iff removal breaks execution

# ---- Execution phase ----
def guard(cert, env, agent):
    revoked = []
    for (pred, witness, step_idx, necessity_i) in cert.obligations:
        if not pred.holds(env):
            revoked.append( (pred, witness, step_idx, necessity_i) )
    if revoked:
        return REVOCATION(cert.artifact.id, revoked)   # artifact staleness, not agent failure
    return agent.execute(cert.artifact)                # certificate still valid
```

Obligations are minimized by a subsumption pass (`dedup_and_minimize`) that removes any predicate whose witness is entailed by a stronger retained predicate, so the certificate stays small (target ≤ 20 obligations per topic). Re-validation runs once per artifact load, not per action. The necessity test is run once per retained obligation at certification time; its cost is one sandbox replay per obligation (bounded by the ≤ 20 target), and it is the only step that distinguishes a *necessary* read from a merely *observed* read. Obligations with `necessity_i = False` are retained in the certificate but flagged, so the execution phase can report spurious-revocation rate separately from genuine staleness.

**Why this design.** We chose to derive obligations from the *construction-time execution trace* rather than from a static diff of the artifact against the environment, because a trace records which predicates the artifact's execution actually read, whereas a diff records everything that changed and cannot distinguish relevant from irrelevant drift; the cost is that obligations only cover predicates the construction run happened to touch, so an artifact whose trace was shallow gets a weak certificate. We chose a *set of per-predicate obligations* over a single aggregate validity score because a scalar cannot localize which dependency broke, and localization is exactly what separates staleness from agent failure; the cost is that certificates are larger and must be stored alongside the artifact. We chose *revocation* (artifact is quarantined and re-certification is triggered) over silent fallback to no-skill execution because silent fallback would mask drift and prevent the skill store from ever being repaired; the cost is that a revoked artifact is temporarily unavailable until re-certified. We added the *necessity test* because the trace alone cannot distinguish a read the artifact actually depends on from a read that happened to be present on a lucky path; the cost is one sandbox replay per obligation at certification time, and the benefit is that `necessity_i` makes the over-approximation measurable rather than hidden. A domain expert would not describe PRISM as a variant of ASI's programmatic verification, because ASI verifies a skill *once* against a static environment and caches the verdict, whereas PRISM's certificate is a function of the environment and carries the dependency structure needed to invalidate it later — the verification object itself is different.

**Why it measures what we claim.** `extract_env_reads` measures *operational dependence* because we assume every environmental read performed during the construction trace is a necessary condition for the artifact's observed success; this assumption fails when the construction run took a lucky path that read a predicate it did not need (e.g., a selector that happened to be present but was never required), in which case the obligation over-approximates dependence and may revoke a still-valid artifact. The `necessity_test` measures *demonstrated necessity* because we assume that if removing a predicate in the sandbox causes the artifact's replay to fail, then the predicate is load-bearing for that artifact; this assumption fails when the sandbox perturbation is not faithful to the real drift (e.g., renaming a selector in a way that also changes unrelated DOM structure), in which case `necessity_i` may be a false positive and the obligation is retained despite being spurious. `pred.holds(env)` at execution time measures *drift* because we assume the predicate's truth value is stable under the environment's own versioning within a session; this assumption fails under mid-session A/B UI variants, in which case a revocation may reflect a transient variant rather than genuine staleness. `dedup_and_minimize` measures *minimal sufficient dependence* because we assume predicate entailment is decidable over the predicate grammar (selector existence, schema field presence, doc anchor presence); this assumption fails for semantic doc drift where the anchor still exists but its content changed, in which case the certificate reflects syntactic rather than semantic validity and a semantic re-certification pass is required.

## Contribution

(1) A precondition-indexed certificate object for skill artifacts and synthesized environments, whose construction and execution semantics are defined by a single invariant — the certificate is valid iff every obligation it indexes holds in the live environment — with revocation as the defined response to violation. (2) A trace-derived obligation extraction and subsumption-minimization procedure that converts a construction-time execution trace into a small set of runtime-checkable predicates, giving the first mechanism that localizes artifact staleness separately from agent failure in a computer-use skill store. (3) An execution-time re-validation protocol that quarantines revoked artifacts and triggers re-certification, providing a design principle for drift-aware skill stores rather than one-shot validated caches.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | WebArena subset (50 tasks) with live site mutations | Realistic drift, multi-step web tasks |
| Primary metric | Task success rate after drift injection | Directly measures staleness handling |
| Secondary metric | Spurious-revocation rate (revocations where `necessity_i = False`) | Quantifies over-approximation from lucky-path reads |
| Secondary metric | Skill-store repair rate over 3 drift cycles | Tests long-run store health, not just per-task success |
| Baseline | No-skill agent | Tests whether any skill helps under drift |
| Baseline | ASI static-verified skill | Tests one-time verification vs. indexed revalidation |
| Baseline | VISUALSKILL text-skill | Tests text-only vs. precondition-indexed certification |
| Baseline | Version-hash invalidation (cache invalidation on env fingerprint change) | Tests whether dependency tracking beats coarse invalidation |
| Ablation | PRISM without dedup_and_minimize | Tests whether obligation minimization matters |
| Ablation | PRISM without necessity_test (all reads retained) | Tests whether necessity filtering reduces spurious revocations |

Drift-injection taxonomy (pre-registered, per-type ground-truth labels): (1) selector rename, (2) selector removal, (3) API schema field removal, (4) API schema field rename, (5) semantic doc edit (anchor present, content changed). Each drifted task is labeled with the type(s) injected so success and revocation can be broken down per type.

### Why this setup validates the claim

This design isolates the central claim that precondition-indexed certification converts silent skill failure into localized revocation. The no-skill baseline establishes the floor: any gain over it must come from skill use, not from the agent's raw capability. The ASI static-verified baseline tests the specific sub-claim that one-time verification is insufficient under drift; if ASI matches PRISM despite drift, the motivation for indexed revalidation collapses. The VISUALSKILL text-skill baseline tests whether the gain is merely from having a richer artifact, or specifically from the certificate's dependency structure. The version-hash baseline tests whether the mechanism is just coarse cache invalidation under a new name: if version-hash invalidation matches PRISM on success and spurious-revocation rate, the dependency structure is not load-bearing. The no-dedup ablation tests whether minimization is load-bearing: if removing dedup_and_minimize yields identical success, the certificate's compactness is cosmetic. The no-necessity-test ablation tests whether the necessity filter reduces spurious revocations: if spurious-revocation rate is identical with and without the test, the lucky-path gap is not empirically material on this benchmark. Success rate after drift injection is the right primary metric because it directly exposes the predicted failure mode—silent wrong execution—rather than proxy signals like step count or token use. The spurious-revocation rate is the right secondary metric because it directly measures the over-approximation the necessity test is meant to control. The skill-store repair rate over 3 drift cycles tests the long-run claim that revocation signals actually improve store health rather than only per-task success. A method that revokes correctly will recover success via re-certification; one that silently falls back will fail on exactly the drifted subsets, producing a localized gap rather than a uniform one.

### Expected outcome and causal chain

**vs. No-skill agent** — On a case where a UI selector or API field has been renamed between construction and execution, the no-skill agent re-derives a plan each time and can adapt, but it lacks the skill's procedural knowledge and flounders on long-horizon tasks. Our method revokes the stale artifact and triggers re-certification, then executes the repaired skill, so we expect a clear gap on drifted long-horizon tasks but parity on short or undrifted tasks.

**vs. ASI static-verified skill** — On a case where a doc anchor still exists but its surrounding schema changed, ASI's one-time verification cached a passing verdict and the agent executes the skill against the changed environment, producing silent wrong behavior. Our method's obligation over that field fails at load time and emits a revocation before execution, so we expect ASI to show a sharp success drop on drifted subsets while PRISM holds near its undrifted rate.

**vs. VISUALSKILL text-skill** — On a case where a visual selector is present but a required API field is absent, the text-skill baseline has no predicate-level dependency record and executes blindly, failing mid-trajectory. Our method's certificate localizes the broken dependency and revokes before the first action, so we expect a concentrated gain on tasks with partial structural drift and near-parity on purely visual drift.

**vs. Version-hash invalidation** — On a case where an irrelevant DOM node changes (e.g., a banner added) but no depended-on predicate is violated, version-hash invalidation revokes the artifact and forces re-certification, while PRISM's obligations all still hold and the artifact executes. We expect PRISM to show fewer spurious revocations and higher success on tasks with irrelevant drift, with parity on tasks where the depended-on predicate itself changed.

**vs. PRISM without necessity_test** — On a case where the construction trace read a present-but-unused selector that later drifts, the no-necessity-test variant revokes a still-valid artifact (spurious revocation), while the full method's `necessity_i = False` flag on that obligation lets the execution phase report it as spurious rather than treating it as genuine staleness. We expect the full method to show a lower spurious-revocation rate on the selector-rename and selector-removal drift types, with parity on schema-field drift where reads are more likely to be necessary.

### What would falsify this idea

If PRISM's advantage over ASI and VISUALSKILL is uniform across drifted and undrifted subsets rather than concentrated on the drifted subsets where the predicted silent-failure mode occurs, the central claim that precondition-indexed revocation localizes staleness is wrong. Equivalently, if the no-dedup ablation matches PRISM on success and certificate size, the minimization claim is unsupported; if the no-necessity-test ablation matches PRISM on spurious-revocation rate, the necessity filter is not empirically load-bearing on this benchmark.

## References

1. VISUALSKILL: Multimodal Skills for Computer-Use Agents
2. Inducing Programmatic Skills for Agentic Tasks
3. Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents
4. Agent Workflow Memory
5. Proposer-Agent-Evaluator(PAE): Autonomous Skill Discovery For Foundation Model Internet Agents
6. Agent S: An Open Agentic Framework that Uses Computers Like a Human
7. AgentStore: Scalable Integration of Heterogeneous Agents As Specialized Generalist Computer Assistant
8. OS-ATLAS: A Foundation Action Model for Generalist GUI Agents
