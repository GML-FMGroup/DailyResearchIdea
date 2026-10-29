# Behavior-Taxonomy Customer Simulation with Information-Preservation Auditing for Tool-Agent Evaluation

## Motivation

Existing tool-agent benchmarks that rely on simulated customers treat the simulator as a fixed, cooperative (or at best loosely persona-conditioned) component: CRMArena-Pro varies personas for multi-turn realism but does not systematically cover non-collaborative behavior types, and Non-Collaborative User Simulators covers only four ad-hoc categories with no guarantee that task-relevant information actually reaches the agent. The root cause is structural — simulators are validated by downstream task success rather than by an independent check that the required intents and slot values were conveyed, so a 'hard' session and an 'information-starved' session are indistinguishable, and behavior coverage is anecdotal rather than a controlled space. This makes any claim about agent robustness to uncooperative users non-reproducible across the behavior space.

## Key Insight

If each non-collaborative behavior is implemented as a reversible dialogue-level transformation whose inverse is applied to the simulator's own utterance stream, then information preservation becomes a checkable invariant on the transformed transcript rather than a hoped-for byproduct of task success.

## Method

**What it is.** TAXPRESERVE is a simulator wrapper that (i) samples a customer behavior profile from a fixed taxonomy, (ii) renders each required information unit through a behavior-specific surface transformation, and (iii) audits the emitted transcript with the inverse transformation to certify that every task-relevant unit was conveyed. Inputs: a task specification (required intents, slot-value pairs, constraints) and a behavior profile; output: a multi-turn customer utterance stream plus a per-session information-preservation certificate.

**How it works.**

```pseudocode
TAXONOMY = {evasive, contradictory, impatient, over-sharing, topic-switching, vague-referential}
# each behavior b has: transform(b, unit) -> surface form ; inverse(b, surface) -> unit-set
# each (b, transform) pair is validated once against human annotations (see CODEC_VALIDATION)

function SIMULATE(task, profile):
    units = EXTRACT_UNITS(task)          # {intent_i} ∪ {slot:value} ∪ {constraint}
    delivered = {}                        # unit -> turn index
    transcript = []
    for turn in range(MAX_TURNS=20):
        b = profile.next_behavior()       # profile is a fixed ordered list, not sampled per turn
        pending = units - delivered.keys()
        if pending and b.allows_delivery:
            u = SELECT(pending)           # priority: constraints > slots > intent
            surface = transform(b, u)     # e.g. evasive: embed u in a hedge + deflection
            transcript.append((b, surface))
            recovered = inverse(b, surface)
            if u in recovered: delivered[u] = turn   # only certify if inverse recovers u
        else:
            transcript.append((b, FILLER(b)))       # behavior-only turn, no unit
    cert = AUDIT(transcript, units)
    return transcript, cert

function AUDIT(transcript, units):
    recovered = union over turns of inverse(b_t, surface_t)
    return {unit: (unit in recovered) for unit in units}

function CODEC_VALIDATION(behavior, transform, n_samples=100):
    # one-time calibration per (b, transform) pair
    surfaces = [transform(b, u) for u in SAMPLE_UNITS(n_samples)]
    human_extracted = [HUMAN_ANNOTATE(s) for s in surfaces]  # 3 annotators, majority vote
    inverse_extracted = [inverse(b, s) for s in surfaces]
    agreement = PRECISION_RECALL(inverse_extracted, human_extracted)  # per unit type
    return agreement  # if recall < 0.9 for a unit type, restrict certificate to that type
```

A session is admitted to the benchmark only if `cert` marks all units True; otherwise it is regenerated with the same profile up to `MAX_RETRIES=3`, and persistent failures are logged as taxonomy defects rather than silently dropped. Before any session is generated, each (behavior, transformation) pair is validated via `CODEC_VALIDATION`: we sample 100 units per pair, collect human annotations of which units are extractable from each surface (3 annotators, majority vote), and compute inverse-vs-human precision/recall per unit type (constraint, slot, intent). If recall for a unit type falls below 0.9, the certificate for that unit type is disabled (sessions are still generated but not certified on that type), and the transformation is flagged for revision. This ensures the certificate measures human-faithful preservation, not codec self-consistency.

**Why this design.** We chose a fixed, ordered behavior profile over per-turn random behavior sampling because a fixed profile makes the non-collaborative condition a controlled independent variable — the same behavior sequence is replayed against every agent — accepting the cost that sessions are less diverse within a profile. We chose behavior-specific surface transformations with explicit inverses over free-form LLM persona prompting because inverses give us a checkable certificate rather than a plausibility judgment, accepting the cost that transformations must be hand-specified per behavior (six behaviors, ~3 transformations each). We chose a priority ordering (constraints > slots > intent) over uniform unit selection because constraint omission is the failure mode that most changes the correct agent action, accepting the cost that early turns are slightly more informative than a natural customer would be. This differs from the persona-conditioned simulator of CRMArena-Pro and the four-category simulator of Non-Collaborative User Simulators: those systems generate non-collaboration as a style and verify it only through downstream task outcome, whereas TAXPRESERVE treats information delivery as an auditable channel with an inverse codec, so behavior and information are separated by construction rather than confounded.

**Why it measures what we claim.** `recovered = inverse(b, surface)` measures information preservation because the named assumption is that the transform/inverse pair is a lossless codec for the unit set — i.e., any unit a human reader could extract from the surface is exactly what the inverse extracts; this assumption fails when a transformation paraphrases a slot value into a form the inverse's pattern does not match (e.g., an evasive hedge replaces a date with 'sometime soon'), in which case the audit flags a false negative and the session is regenerated, meaning the certificate reflects codec coverage rather than human interpretability. `profile.next_behavior()` measures controlled coverage of the non-collaborative space because the assumption is that the six taxonomy entries span the behaviors that change the agent's required tool sequence; this fails for behaviors that alter information content rather than surface form (e.g., a customer who supplies a genuinely wrong account number), in which case the session measures adversarial input handling, not non-collaboration. `delivered[u]` measures conveyance because the assumption is that an agent observing the transcript sees the same surface the inverse sees; this fails when the agent's tool calls truncate or summarize the transcript, in which case the certificate measures simulator-side delivery, not agent-side reception — which is why we report both and treat divergence as a separate diagnostic. `CODEC_VALIDATION` measures human-faithful preservation because the assumption is that the inverse's extraction matches what a competent human reader would extract; this fails when the inverse pattern-matches a token that humans cannot ground (e.g., 'sometime soon' for a date), in which case the certificate would falsely certify a lossy transformation — which is why we calibrate against human annotations and disable certification for unit types with recall < 0.9.

## Contribution

(1) A behavior taxonomy with paired reversible surface transformations that turns non-collaborative customer simulation into a controlled independent variable with a per-session information-preservation certificate. (2) A design principle — audit-by-inverse — that separates 'the simulator was hard' from 'the simulator withheld information', enabling regeneration of defective sessions instead of confounding them with agent failure. (3) An open simulator wrapper and certificate format that can be layered onto existing tool-agent benchmarks such as TauIndianBankBench-style procedure-aware suites without changing their scoring.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | TAXPRESERVE sessions over banking task specs (200 tasks, 6 profiles, 3 seeds) | Auditable delivery certificates isolate behavior from information. |
| Primary metric | Constraint-preservation rate under non-collaborative profiles | Directly detects the failure mode the method targets. |
| Baseline | CRMArena-Pro persona-conditioned simulator | Tests whether auditable codec beats style-prompted non-collaboration. |
| Baseline | Non-Collaborative User Simulators (four-category) | Tests whether six-behavior controlled coverage adds signal. |
| Baseline | Free-form LLM persona prompting | Tests whether explicit inverses beat plausibility-only generation. |
| Ablation-of-ours | TAXPRESERVE without inverse audit (regenerate on plausibility) | Isolates the certificate mechanism from the transform design. |
| Ablation-of-ours | TAXPRESERVE with uniform unit selection (no priority ordering) | Controls the priority-ordering confound: tests whether the effect is driven by ordering rather than behavior. |
| Diagnostic | Simulator-side vs. agent-side divergence rate | Measures the assumption that inverse-recovery equals agent-side reception (tool-call summarization may drop units). |
| Diagnostic | Inverse-vs-human agreement (precision/recall per unit type) | Validates the codec against human annotations, ensuring the certificate measures human-faithful preservation. |

### Why this setup validates the claim

The central claim is that separating behavior from information via an inverse codec produces non-collaborative sessions whose difficulty is attributable to surface transformation rather than to lost or corrupted content, and that agents degrade predictably as a function of which units are delivered late and under which behavior. The persona-conditioned and four-category baselines test the sub-claim that style-level non-collaboration confounds behavior with information: if those baselines produce sessions where constraint omission is entangled with behavior identity, an agent's failure cannot be attributed to either. The free-form persona baseline tests whether checkable inverses matter at all versus plausibility judgment. The no-audit ablation is the load-bearing control: it keeps the same six behaviors and transformations but drops the certificate, so any gain of full TAXPRESERVE over it must come from the audit admitting only information-preserving sessions, not from the transformations themselves. The uniform-selection ablation controls the priority-ordering confound: if the effect persists with uniform selection, the ordering is not driving the result. The divergence diagnostic measures whether the certificate reflects agent-side reception: we compare the set of units recovered by the inverse to the set of units the agent's tool calls actually reference, and report the divergence rate as a separate metric. The codec-validation diagnostic ensures the certificate is not self-referential: we report inverse-vs-human agreement per unit type, and if recall is below 0.9 for a unit type, we disable certification for that type and note it. Constraint-preservation rate is the right primary metric because the design's own priority ordering predicts that constraint delivery is the unit most sensitive to behavior sequencing; a metric that averaged over slots and intents would wash out exactly the effect the causal story predicts.

### Expected outcome and causal chain

**vs. CRMArena-Pro persona-conditioned simulator** — On a case where an evasive customer must convey a transfer limit before an impatient follow-up, the persona-conditioned baseline produces a fluent hedge that drops the numeric limit because its generator optimizes stylistic plausibility with no delivery check. Our method instead renders the limit through the evasive transform and certifies recovery via the inverse before admitting the session, so the limit is provably present. We expect a noticeable gap in constraint-preservation rate concentrated on evasive and topic-switching profiles, with near-parity on impatient profiles where style and content rarely collide.

**vs. Non-Collaborative User Simulators (four-category)** — On a case requiring a contradictory customer to state an account number twice with a reconcilable surface difference, the four-category baseline produces contradiction as a global session label, so the agent cannot tell which utterance carries the authoritative value and the simulator cannot report whether both statements were delivered. Our method instead assigns each unit to a specific behavior turn and audits both, so the agent faces a contradiction whose information content is known. We expect the gap to appear as higher constraint-preservation under contradictory and vague-referential profiles, with little difference on the two behaviors both taxonomies share.

**vs. Free-form LLM persona prompting** — On a case where an over-sharing customer buries a date inside tangential detail, free-form prompting sometimes paraphrases the date into a relative expression that no reader can ground, and plausibility judgment accepts it. Our method instead regenerates when the inverse fails to recover the date, logging the transformation as a taxonomy defect rather than admitting an ungrounded session. We expect the observable signal to be a lower regeneration-admitted rate alongside a higher constraint-preservation rate, with the gap largest on over-sharing and evasive profiles and vanishing on impatient profiles where surface forms stay literal.

**vs. TAXPRESERVE without inverse audit** — The no-audit ablation keeps the same six behaviors and transformations but regenerates on plausibility rather than on inverse recovery. We expect full TAXPRESERVE to show higher constraint-preservation rate across all profiles, with the largest gap on evasive and over-sharing profiles where the inverse catches lossy paraphrases that plausibility judgment misses. If the gap is uniform across profiles, the certificate is doing no causal work.

**vs. TAXPRESERVE with uniform unit selection** — This ablation replaces the priority ordering (constraints > slots > intent) with uniform random selection of pending units. We expect the constraint-preservation rate to be similar between the two, indicating that the priority ordering does not drive the effect; if uniform selection shows a significantly lower constraint-preservation rate, the ordering is a confound and we will report it as such.

### What would falsify this idea

If full TAXPRESERVE and the no-audit ablation show the same constraint-preservation rate across all six profiles, the certificate is doing no causal work and the claim that auditable delivery separates behavior from information is wrong. Likewise, if the gain over persona-conditioned baselines is uniform across profiles rather than concentrated on evasive, contradictory, and over-sharing sessions where the predicted codec failures occur, the mechanism story is falsified. Additionally, if inverse-vs-human agreement is low (recall < 0.9) for a unit type, the certificate for that type is invalid and the claim is weakened for that type.

## References

1. TauIndianBankBench: Procedure-Aware, Action-Oriented Evaluation of Tool-Using Conversational Agents
2. Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents
3. TRAJECT-Bench:A Trajectory-Aware Benchmark for Evaluating Agentic Tool Use
4. CRMArena-Pro: Holistic Assessment of LLM Agents Across Diverse Business Scenarios and Interactions
5. ACEBench: Who Wins the Match Point in Tool Usage?
6. τ2-Bench: Evaluating Conversational Agents in a Dual-Control Environment
7. Non-Collaborative User Simulators for Tool Agents
8. ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities
