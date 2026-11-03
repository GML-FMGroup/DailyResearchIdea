# Typed Defect Algebras for Closed Repair Operators in Video-Generation Agents

## Motivation

Across iterative-operator paper-trees, the operator's coverage is established only on observed failure trajectories: Self-Evolving Coding Agents via Harness-Policy Optimization validates its H-step harness proposal against the hard tasks RL happened to surface, and PegasusAgent's execution-free repair loop is validated only against AMD Radeon defects it encountered. The structural root cause is that the failure/defect class is represented implicitly by the trajectory set, so an unseen failure mode that is a substitution instance of a seen one has no guarantee of being repairable, and coverage stays empirical. A finitely generated typed algebra with substitution would make the class explicit and its closure checkable, which is the bottleneck the field now hits as per-instance validation saturates.

## Key Insight

If the repair operator's proposal distribution is factored through the substitution structure of a finitely generated typed defect operad, then closure under substitution becomes a structural invariant of the operator's output type rather than an empirical property of the trajectories seen, because every unseen defect is syntactically a substitution instance whose repair is the corresponding substitution instance of a generator repair.

## Method

**What it is.** FARA (Finitary Algebra of Repair Actions) is a repair operator for video-generation agents whose output space is a typed term algebra over defect constructors. Input: a failed video-generation trajectory (prompt, sampled frames, agent tool calls) plus the agent's self-diagnosis. Output: a repair term `r` in the algebra together with a repair certificate `cert(r)` that is itself an algebra element, and the repaired harness/agent configuration obtained by substituting `r` into the current configuration.

**Load-bearing assumption (stated explicitly).** FARA assumes that every video-generation failure the agent will encounter is syntactically a substitution instance over a fixed finite generator set `G` (|G| = 16), so that a frozen per-generator repair table lifted by substitution yields a well-typed repair for unseen defects. This is a syntactic-closure assumption, not a semantic-coverage guarantee: substitution closure guarantees only that the repair term is well-typed, not that it fixes the defect. The method therefore (i) flags any defect term not in `T(G)` as out-of-algebra rather than silently repairing it, and (ii) adds an execution-based semantic validity check on the lifted repair (Stage 3b) so that the certificate is decoupled from the construction. The assumption fails for genuine new generators (e.g., temporal aliasing, identity drift from motion blur, physics violations not expressible over `G`); in that case the closure check reflects only repairability relative to `G`.

**How it works.**

```pseudocode
# Stage 0: typed defect signature (finitely generated, |G| = 16)
# Sorts: Frame, Motion, Identity, Prompt, Tool, Cfg
# Constructors (generators), each with arity and coarity over sorts:
#   TemporalDrift   : Frame x Frame -> Motion
#   IdentitySwap    : Identity x Frame -> Identity
#   PromptFrameGap  : Prompt x Frame -> Prompt
#   ToolTimeout     : Tool x Cfg -> Tool
#   PhysicsViol     : Motion x Frame -> Motion
#   ... (fixed, finite generator set G, |G| = 16)

# Stage 1: diagnose failure trajectory -> defect term d in T(G)
def diagnose(traj, agent):
    spans = agent.segment(traj)                  # tool-call spans + frame spans
    atoms = [classify(span) for span in spans]   # each atom is a generator in G
    d = unify(atoms)                              # substitution instance over G
    return d, substitution_witness(d)            # witness: which generator at which position

# Stage 2: repair proposal factored through the algebra
def propose_repair(d):
    # structural recursion on d, NOT free-form generation
    if d is a generator g:
        return R[g]                               # generator repair table, fixed offline
    if d == sigma(d1, ..., dk):                   # substitution instance
        return sigma(R[d1], ..., R[dk])           # substitution-lifted repair

# Stage 3: certificate construction + closure check
def certify(d, r):
    cert = Certificate(d, r, substitution_witness(d))
    assert type_of(r) == type_of(d)               # sort-preservation check
    assert r == lift(R, substitution_witness(d))  # closure check
    return cert

# Stage 3b: execution-based semantic validity check
def semantic_check(d, r, cfg):
    cfg' = substitute(r, cfg)
    new_traj = agent.run(cfg')
    return passes(new_traj)                       # binary pass/fail on held-out rubric

# Stage 4: apply repair to configuration, run agent
cfg' = substitute(r, cfg)
new_traj = agent.run(cfg')
```

Generator repair table `R` is built once per generator (|G| = 16 entries) by running the agent's own repair loop on canonical witnesses, then frozen; substitution lifting is pure syntax. The closure check in Stage 3 is what makes the guarantee structural rather than empirical: any defect term `d` — including unseen ones — is a substitution instance, so `r = lift(R, witness(d))` is guaranteed to be well-typed and in the algebra. The semantic check in Stage 3b is what makes the certificate non-circular: it validates that the repaired trajectory actually passes, decoupling the certificate from the construction.

**Why this design.** We chose a fixed finite generator set `G` over an open-ended learned defect taxonomy because finiteness is what makes the substitution structure enumerable and the repair table buildable once, accepting the cost that defects outside `G` are rejected rather than repaired (a coverage boundary, not a silent failure). We chose structural recursion on the defect term over free-form LLM repair proposal because free-form generation has no type discipline and cannot certify its own output, accepting the cost that the repair table for each generator must be curated and may be suboptimal. We chose to make the certificate an algebra element over a natural-language justification because a certificate in the same algebra is checkable by the same type system that generates it, accepting the cost that the certificate is less human-readable than prose. We added the execution-based semantic check (Stage 3b) because sort-preservation alone does not certify effectiveness; this accepts the cost of one additional agent run per repair. A domain expert would not describe FARA as a variant of Self-Evolving Coding Agents via Harness-Policy Optimization or PegasusAgent: those operators generate repairs in an untyped output space and validate per-instance, whereas FARA's repair is a substitution lift of a frozen generator table — the operator cannot emit an ill-typed repair, which is a property the H-step and execution-free repair loop do not have.

**Why it measures what we claim.** The closure check `r == lift(R, substitution_witness(d))` measures *repairability of the full defect class* because it assumes that every defect term in the trajectory is a substitution instance over `G` — i.e., that `G` is a generating set for the observed defect language; this assumption fails when a failure is a genuine new generator (e.g., a video-specific artifact like temporal aliasing not in `G`), in which case the check reflects only *repairability relative to G* and the defect is flagged as out-of-algebra. The type check `type_of(r) == type_of(d)` measures *sort-preservation of repair* because it assumes the sorts (Frame/Motion/Identity/Prompt/Tool/Cfg) are a sound abstraction of the video-generation failure surface; this assumption fails when a defect spans two sorts (e.g., identity drift caused by motion blur), in which case the check reflects only *single-sort well-typedness* and the repair may be well-typed yet ineffective. The substitution witness measures *structural correspondence between defect and repair* because it assumes the witness is a faithful parse of the failure trajectory; this assumption fails when the agent's segmentation is wrong, in which case the witness reflects the agent's *diagnosis* rather than the true defect structure. The semantic check `passes(new_traj)` measures *semantic effectiveness of the lifted repair* because it assumes the held-out rubric is a faithful proxy for the intended video quality; this assumption fails when the rubric is incomplete or gameable, in which case the check reflects only *rubric-passing* rather than true defect repair.

## Contribution

(1) A typed defect algebra (finitely generated operad with substitution) used as the output space of an iterative repair operator for video-generation agents, with a substitution-lifted repair table and a certificate that is itself an algebra element. (2) A design principle: coupling the repair proposal distribution to the algebra's substitution structure makes closure under substitution a structural invariant checkable at repair time, rather than an empirical property of the trajectory set. (3) An open generator set and repair table for video-generation failure modes (temporal drift, identity swap, prompt-frame gap, tool timeout, physics violation) with the associated type system.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Held-out video-gen failure trajectories with typed defect labels (n=500; 350 in-algebra, 150 out-of-algebra) | Tests repair on defects matching generator set |
| Primary metric | Fraction of defects repaired to passing, split by in-algebra vs out-of-algebra | Separates coverage boundary from repair efficacy |
| Secondary metric | Out-of-algebra rate on a realistic trajectory corpus (n=1000) | Quantifies coverage boundary on real failures |
| Baseline | Free-form LLM repair proposal | Untyped repair with no certificate, per-instance validation |
| Baseline | Self-HPO harness-only optimization | Harness search without typed defect algebra |
| Baseline | PegasusAgent execution-free repair loop | Execution-free repair in untyped output space |
| Baseline | Hand-written typed rule engine (same |G|=16, no substitution-lift) | Isolates operadic substitution from typing alone |
| Ablation-of-ours | FARA without closure check (free-form proposal, typed certificate only) | Isolates contribution of substitution-lift guarantee |
| Ablation-of-ours | FARA with flat per-generator lookup (no structural recursion) | Separates recursion from substitution-lift |
| Ablation-of-ours | FARA without semantic check (Stage 3b removed) | Isolates contribution of execution-based validation |

### Why this setup validates the claim

The central claim is that FARA cannot emit an ill-typed repair because its repair is a substitution lift of a frozen generator table, and that this structural guarantee converts into a measurable repair-success advantage over untyped operators. The dataset must therefore contain both in-algebra defects (substitution instances over G) and out-of-algebra defects (genuine new generators), so that the metric's split directly exposes the coverage boundary the design accepts rather than hiding it. The free-form LLM baseline tests the claim that untyped repair proposal has no type discipline and fails on structurally novel substitution instances; the Self-HPO harness-only baseline tests whether typed repair beats unconstrained harness search when the defect is a composition rather than a single harness setting; PegasusAgent tests whether execution-free untyped repair loses on composition depth. The hand-written typed rule engine baseline isolates whether the operadic substitution structure (not typing alone) produces any gain. The ablation without the closure check keeps the typed certificate but removes the substitution-lift closure check, so if the guarantee — not merely the certificate artifact — drives the gain, the ablation should recover most of the free-form baseline's failures on deep substitution instances. The flat per-generator lookup ablation separates structural recursion from substitution-lift. The no-semantic-check ablation isolates the contribution of execution-based validation. The in-algebra/out-of-algebra split is the right metric because a uniform gain would be consistent with generic agent improvement, whereas a gain concentrated on in-algebra compositional defects and parity on out-of-algebra defects is the signature the mechanism predicts. The out-of-algebra rate on a realistic corpus quantifies the significance claim rather than asserting it as an accepted cost.

### Expected outcome and causal chain

**vs. Free-form LLM repair proposal** — On a case where the defect is a nested composition such as identity swap caused by temporal drift across three frame spans, the baseline produces a plausible-sounding patch that addresses only the surface identity symptom because its untyped generation has no substitution structure to recurse over and validates only the single instance it sees. Our method instead recurses structurally on the defect term, lifting the identity and drift generator repairs through the same substitution witness, because the repair is a syntactic lift of a frozen table. We expect a noticeable gap on multi-generator compositional defects but rough parity on single-generator defects.

**vs. Self-HPO harness-only optimization** — On a case where the failure requires coordinated changes across tool timeout and prompt-frame gap, the baseline produces a harness setting that fixes one axis while leaving the other, because harness-only search treats configuration as a flat vector with no typed defect decomposition to guide which axes co-vary. Our method instead emits a repair term whose constructors name both axes and substitutes them jointly into the configuration, because the defect term exposes the composition explicitly. We expect a clear advantage on multi-sort defects and parity on single-sort harness defects.

**vs. PegasusAgent execution-free repair loop** — On a case where a physics violation is only visible in sampled frames after a tool call, the baseline produces a repair validated without execution and therefore misses the frame-level manifestation, because its execution-free loop has no typed link between tool sort and frame sort. Our method instead carries the defect as a Motion x Frame constructor and lifts the repair through the witness that ties the tool span to the frame span, because the algebra's sorts encode that cross-sort dependency. We expect a moderate gap concentrated on cross-sort defects, with parity on single-sort tool defects.

**vs. Hand-written typed rule engine (same |G|=16, no substitution-lift)** — On a case where the defect is a substitution instance of depth 3, the baseline applies a flat rule lookup per generator without lifting through the substitution witness, so it produces a repair that is well-typed but does not compose the generator repairs according to the defect structure. Our method instead lifts the generator repairs through the substitution witness, because the repair is a syntactic lift of a frozen table. We expect a gap on deep substitution instances, with parity on single-generator defects.

**vs. FARA without closure check (free-form proposal, typed certificate only)** — On a case where the defect is a deep substitution instance, the ablation produces a repair that is typed but not the substitution lift, because it lacks the closure check to enforce structural correspondence. Our method instead enforces `r == lift(R, substitution_witness(d))`, because the closure check is the structural guarantee. We expect the ablation to recover most of the free-form baseline's failures on deep substitution instances, confirming the closure check is load-bearing.

**vs. FARA with flat per-generator lookup (no structural recursion)** — On a case where the defect is a composition of two generators, the ablation applies a flat lookup that ignores the composition structure, because it lacks structural recursion. Our method instead recurses structurally on the defect term. We expect a gap on compositional defects, with parity on single-generator defects, separating recursion from substitution-lift.

**vs. FARA without semantic check (Stage 3b removed)** — On a case where a well-typed repair is semantically ineffective (e.g., cross-sort defect), the ablation certifies the repair as valid because it lacks the execution-based semantic check. Our method instead runs the repaired trajectory and validates it passes, because Stage 3b decouples the certificate from the construction. We expect the ablation to over-report repair success on cross-sort defects, confirming the semantic check is necessary.

### What would falsify this idea

If the ablation without the closure check matches FARA's repair success on in-algebra compositional defects, the substitution-lift guarantee is not what produces the gain and the central claim is wrong. Likewise, if FARA's advantage is uniform across in-algebra and out-of-algebra defects rather than concentrated where the predicted structural failure mode occurs, the mechanism is not the cause. If the hand-written typed rule engine matches FARA, the operadic substitution structure adds nothing beyond typing.

## References

1. Self-Evolving Coding Agents via Harness-Policy Optimization
2. Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution
3. Eliminating Meta Optimization Through Self-Referential Meta Learning
4. Large Language Models Are Human-Level Prompt Engineers
5. Execution-free Agentic Program Repair for Enterprise-Scale Development
6. LocAgent: Graph-Guided LLM Agents for Code Localization
7. SweRank: Software Issue Localization with Code Ranking
8. Training Software Engineering Agents and Verifiers with SWE-Gym
