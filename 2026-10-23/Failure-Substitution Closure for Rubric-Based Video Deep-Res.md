# Failure-Substitution Closure for Rubric-Based Video Deep-Research Evaluation

## Motivation

Rubric-based video deep-research benchmarks such as OmniDR-Bench (Beyond Watching and Searching) score every atomic binary criterion independently and treat an unsatisfiable criterion as a hard zero, so an agent that cannot ground a video segment or retrieve external multimodal evidence is penalized for a failure that is structurally unavoidable given the available evidence, not for shallow research. Because Omni-ARTs rubric trees assume criteria are independently satisfiable, aggregate scores are not comparable across different failure patterns: two agents with identical genuine research depth can receive different totals purely because they hit different unavoidable failures, which corrupts both fairness and discriminativeness at the benchmark level.

## Key Insight

If every rubric criterion is paired with a predefined substitute criterion whose satisfaction is semantically entailed by the original criterion's intent, then the aggregate score is invariant to which unavoidable failures occurred, making scores comparable across failure patterns without weakening the test of genuine deep research.

## Method

**What it is.** FSC (Failure-Substitution Closure) is a rubric-tree evaluation layer that takes as input an agent's report plus its recorded failure events on a video deep-research task, and outputs a failure-closed aggregate score in which each unavoidable failure is replaced by a predefined substitute criterion rather than scored as zero. It operates on the Omni-ARTs tree structure (3 dimensions, 23 shared directions, atomic binary criteria) by augmenting each leaf with a substitution rule, an entailment-justification record, and a failure-attribution record.

**How it works.**

```pseudocode
INPUT: rubric tree T with atomic leaves L; agent report R; failure log F (per-leaf: grounded? evidence_retrieved? attempted?)
OUTPUT: closed score S_closed in [0,1], plus substitution trace

# Phase 0: substitution-map calibration (offline, before any agent runs)
# For each leaf c, author names substitute s (same direction tag, strictly weaker evidence requirement)
# and writes an entailment justification J(c,s).
# Calibration set: 120 leaves sampled from the 23 shared directions (>=5 per direction),
# each evaluated by 3 domain experts on held-out agent reports.
# Expert labels each (c,s) pair: ENTAILED (satisfying s entails original intent) or NOT_ENTAILED.
# Rule accepted iff >=2/3 experts label ENTAILED (majority).
# Rejected rules are re-authored once; if still rejected, leaf c is marked NO_SUBSTITUTE
# and behaves as in raw Omni-ARTs (unavoidable failure = zero).
# Report calibration agreement (Fleiss' kappa) and accepted-rule count per dimension.

# Phase 1: failure attribution (which leaves failed for structural reasons)
for leaf c in L:
    if c.evaluated(R) == 1: status[c] = SATISFIED
    elif c in F and F[c].reason in {NO_VIDEO_SEGMENT, NO_EXTERNAL_EVIDENCE} and F[c].attempted == True:
        status[c] = UNAVOIDABLE
    else: status[c] = GENUINE_FAIL

# Phase 2: substitution application (predefined, calibrated rule per leaf)
for leaf c with status[c] == UNAVOIDABLE:
    if c has NO_SUBSTITUTE: status[c] = GENUINE_FAIL; continue
    s = substitute[c]            # s is a sibling/cousin leaf with same direction tag
    if s.status == SATISFIED and same_direction(c, s):
        status[c] = SATISFIED_BY_SUBSTITUTION
    else:
        status[c] = GENUINE_FAIL   # closure rule: substitution only credits if substitute is met

# Phase 3: aggregation with closure weights (weights fixed at rubric construction, w_c = 1/|L| per dimension)
S_closed = sum_{c in L} w_c * 1[status[c] in {SATISFIED, SATISFIED_BY_SUBSTITUTION}]

# Phase 4: comparability audit (report-level diagnostic, not a re-score)
report substitution_rate per dimension and flag tasks where substitution_rate > tau=0.30
```

Substitution rules are authored during Audited Rubric Construction: for each leaf c, the rubric author names a substitute leaf s in the same shared direction whose evidence requirement is strictly weaker but whose satisfaction entails the same research skill (e.g., if the agent could not ground moment m, the substitute requires citing the nearest grounded moment within the same scene and stating the temporal gap). The substitution map is fixed before any agent is run, so it cannot be gamed by the agent. **Load-bearing assumption (stated explicitly):** each accepted substitute leaf s is a semantically valid weakening of the original leaf c's intent within the same shared direction, such that crediting s preserves the measurement of the same research skill rather than substituting a different, easier skill. This assumption is not taken on faith: it is checked empirically in Phase 0 on a 120-leaf calibration set with 3-expert majority entailment labeling (Fleiss' kappa reported), and substitution credit is additionally gated on the agent having attempted the original leaf (F[c].attempted == True), so an agent that never grounds cannot harvest substitutes.

**Why this design.** We chose a fixed pre-authored substitution map over a learned or agent-proposed fallback because a learned fallback would let the agent's own failure pattern define its scoring criterion, which destroys cross-agent comparability; the cost is that substitution rules must be manually authored and may be coarse for rare failure modes. We chose failure attribution based on recorded evidence-availability flags (NO_VIDEO_SEGMENT, NO_EXTERNAL_EVIDENCE) rather than on the agent's self-reported confidence because self-reported confidence is not calibrated for task correctness (Kadavath et al.), accepting that attribution depends on the benchmark harness logging availability correctly. We chose to require the substitute leaf to be SATISFIED for substitution credit, rather than granting automatic credit on failure, because automatic credit would reward failure; the cost is that an agent that fails both the original and the substitute still receives zero, which is the intended conservative behavior. We added the Phase 0 calibration gate and the attempted-original gate because without them the substitution map is an unvalidated authoring artifact and an agent could satisfy every substitute by citing a nearby timestamp without ever grounding; the cost is a one-time 120-leaf expert calibration pass (3 experts, ~2 hours per expert) and the possibility that some leaves receive NO_SUBSTITUTE and revert to raw scoring. A domain expert would not describe this as a variant of standard rubric evaluation (e.g., Omni-ARTs) because the aggregate is computed over a closure of the leaf set under a substitution relation, not over the leaf set itself; the score is invariant to which unavoidable failures occurred, a property Omni-ARTs does not have.

**Why this measures what we claim.** The substitution rate per dimension measures the fraction of failures that are structurally unavoidable rather than genuine, because the named assumption is that the harness's evidence-availability flags are a sufficient statistic for whether a criterion was satisfiable given the retrieved evidence; this assumption fails when the harness fails to log an available segment, in which case the substitution rate reflects harness logging error rather than agent capability. The closed score S_closed measures genuine deep-research depth comparably across failure patterns because the named assumption is that each substitute leaf is semantically entailed by the original leaf's intent within the same shared direction (the rubric author's substitution rule is a valid weakening, verified by the Phase 0 3-expert majority entailment check on 120 leaves); this assumption fails when the substitute is only superficially related to the original, in which case S_closed reflects rubric-authoring quality rather than research depth. The comparability audit flag (substitution_rate > 0.30) measures whether a task's score is dominated by substitution rather than direct satisfaction, because the named assumption is that high substitution rates indicate a task whose evidence is structurally scarce; this assumption fails when an agent deliberately triggers substitution by avoiding hard grounding, in which case the flag reflects agent gaming rather than task difficulty — mitigated by the attempted-original gate, which prevents an agent that never grounds from reaching the substitution branch at all.

## Contribution

(1) A failure-substitution closure operator over rubric trees that makes aggregate scores invariant to which unavoidable failures occurred, implemented as a fixed pre-authored substitution map with a conservative satisfaction requirement. (2) A design principle for fair rubric-based agent benchmarks: criteria that can fail for structural evidence-availability reasons must carry a predefined substitute, and substitution rates must be reported as a separate diagnostic rather than folded into the headline score. (3) An audited substitution-map construction procedure compatible with existing Omni-ARTs trees, yielding a failure-closed variant of OmniDR-Bench scoring.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | OmniDR-Bench (50 tasks, 19 subcategories) | Video deep-research tasks with recorded evidence-availability flags |
| Primary metric | S_closed (failure-closed rubric score) | Directly measures the closure-adjusted aggregate we claim |
| Baseline | Omni-ARTs raw rubric score | Standard leaf-set aggregate without closure |
| Baseline | Self-reported-confidence-weighted scoring | Tests that evidence flags beat confidence attribution |
| Baseline | Automatic-credit-on-failure scoring | Tests that substitution-only-if-satisfied is necessary |
| Ablation-of-ours | FSC without comparability audit | Isolates the diagnostic from the scoring mechanism |
| Ablation-of-ours | FSC without Phase 0 calibration gate (unvalidated substitution map) | Tests whether the entailment check is load-bearing for construct validity |
| Ablation-of-ours | FSC without attempted-original gate | Tests whether the attempted gate is necessary to prevent substitute harvesting |
| Calibration set | 120 leaves (>=5 per shared direction), 3 domain experts, majority entailment label | Provides the empirical entailment check for the load-bearing assumption |
| Expert audit set | 200 leaves sampled across 50 tasks, 3 experts label UNAVOIDABLE vs GENUINE_FAIL and rate substitute entailment (ENTAILED / NOT_ENTAILED) | Validates flag-based attribution and substitution entailment against human judgment |

### Why this setup validates the claim

The central claim is that a fixed, pre-authored substitution closure makes the aggregate score comparable across agents with different failure patterns while remaining conservative (it never rewards failure) and while preserving the measured construct (substitute satisfaction entails original intent). Omni-ARTs raw scoring is the direct control: it computes over the leaf set itself, so any agent whose evidence retrieval is structurally blocked is penalized identically to an agent that genuinely failed, which is exactly the confound we claim to remove. The confidence-weighted baseline tests the attribution sub-claim — if self-reported confidence were an adequate substitute for harness evidence flags, its score would track S_closed, and our attribution choice would be unmotivated. The automatic-credit baseline tests the conservatism sub-claim: it is the degenerate version of closure that grants credit on failure, so any gain it shows over us on gaming-prone tasks would reveal that our SATISFIED-substitute requirement is too strict. The two new ablations directly test the load-bearing assumption: removing the Phase 0 calibration gate leaves the substitution map unvalidated, so if S_closed without calibration matches S_closed with calibration on expert re-scoring, the entailment check is not doing construct-validity work; removing the attempted-original gate tests whether an agent that never grounds can harvest substitutes. The 120-leaf calibration set and 200-leaf expert audit set provide the human labels needed to compute inter-annotator agreement (Fleiss' kappa) and precision/recall of the flag-based UNAVOIDABLE attribution against expert labels, which is the only way to distinguish a genuine comparability gain from a rubric-authoring artifact. S_closed is the right metric because the predicted effect is a reordering of agents on structurally-scarce tasks specifically, and only a closure aggregate that is invariant to which unavoidable failures occurred can express that reordering without also moving on tasks where evidence was available.

### Expected outcome and causal chain

**vs. Omni-ARTs raw rubric score** — On a case where the agent correctly reasons about a moment but the harness logged no video segment for it, the raw score produces a zero on that leaf because its aggregate has no notion of satisfiability, so a capable agent and a lazy agent look identical on that leaf. Our method instead marks the leaf UNAVOIDABLE (only if F[c].attempted == True), checks whether the pre-authored weaker substitute (cite nearest grounded moment in scene plus temporal gap) is satisfied and was accepted in Phase 0 calibration, and credits it only then, because the substitution map is fixed before any agent runs. We expect a noticeable gap on the structurally-scarce task subset (low evidence availability) and near-parity on evidence-rich tasks, since closure only moves leaves that were unavoidable.

**vs. Self-reported-confidence-weighted scoring** — On a case where the agent is confidently wrong about a temporal ordering it could not ground, the confidence-weighted baseline produces inflated credit because uncalibrated confidence is used as the attribution signal, and confidence does not track correctness. Our method instead attributes failure from the harness's NO_VIDEO_SEGMENT / NO_EXTERNAL_EVIDENCE flags, which are independent of what the agent claims, because we deliberately reject self-report as an attribution channel. We expect the confidence baseline to show high variance across agents on the same task and a weaker correlation with expert re-scoring, while S_closed stays stable — a signal concentrated on tasks with many ungroundable leaves.

**vs. Automatic-credit-on-failure scoring** — On a case where an agent skips hard grounding entirely and fails both the original leaf and its substitute, the automatic-credit baseline produces full credit because it grants substitution on failure alone, so avoiding hard evidence becomes a winning strategy. Our method instead requires the substitute leaf to be SATISFIED before granting credit and requires F[c].attempted == True, because rewarding failure would destroy the score's meaning. We expect the automatic-credit baseline to score highest on the tasks where agents strategically dodge grounding, while S_closed penalizes exactly those agents — an inversion of ranking on the gaming-prone subset with parity elsewhere.

**vs. FSC without Phase 0 calibration gate** — On a case where a substitute is only superficially related to the original (e.g., citing any nearby timestamp without stating the temporal gap), the uncalibrated variant credits it because the substitution map was never expert-checked, so S_closed reflects rubric-authoring quality rather than research depth. Our method rejects such rules in Phase 0 (majority expert NOT_ENTAILED) and marks the leaf NO_SUBSTITUTE, reverting to raw scoring. We expect the uncalibrated variant to show higher S_closed on gaming-prone tasks and lower correlation with expert re-scoring, while calibrated S_closed tracks expert labels; the gap is concentrated on leaves where the calibration set flagged NOT_ENTAILED.

**vs. FSC without attempted-original gate** — On a case where an agent never grounds any moment but always cites a nearby timestamp and notes a gap, the ungated variant lets it satisfy every substitute and harvest credit without attempting the original leaf. Our method requires F[c].attempted == True before entering the substitution branch, so an agent that never grounds receives GENUINE_FAIL. We expect the ungated variant to score highest on the gaming-prone subset, while gated S_closed penalizes exactly those agents — an inversion of ranking on that subset with parity elsewhere.

### What would falsify this idea

If S_closed's advantage over raw Omni-ARTs scoring were uniform across all tasks rather than concentrated on the structurally-scarce subset, the closure mechanism would not be doing what we claim, and the score would just be a rescaled raw score. Likewise, if the automatic-credit baseline matched S_closed on gaming-prone tasks, our conservatism requirement would be false; and if the Phase 0 calibration gate did not change S_closed's correlation with expert re-scoring, the load-bearing entailment assumption would be unverified and the closure would measure rubric-authoring quality rather than research depth.

## References

1. Beyond Watching and Searching: Benchmarking Agents for Video-centered Deep Research Reports
