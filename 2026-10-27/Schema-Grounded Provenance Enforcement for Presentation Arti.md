# Schema-Grounded Provenance Enforcement for Presentation Artifacts in LLM Work Agents

## Motivation

Work-agent frameworks such as the one surveyed in 'Work Agents: Systems, Capabilities, and Optimization' treat verifiable outcomes and stateful workspaces as assumptions rather than enforced invariants, and artifact-generating pipelines like PPTAgent assume an extractable slide schema without ever instantiating a mechanism that binds each generated artifact to a validated schema derived from workspace state. The structural root cause is that the persistence/openness taxonomy is descriptive: nothing in the generation loop makes schema lineage a precondition for artifact acceptance, so an artifact can be emitted whose structure is untraceable to any workspace-derived schema and still pass as a valid 'presentation' work product. This leaves the presentation work form unevaluable against the framework's own verification story.

## Key Insight

If every artifact is required to carry a provenance chain whose terminal node is a schema mechanically derived from the workspace state and re-validated by an independent conformance checker, then artifact validity becomes a decidable function of workspace state rather than a judgment about the generator's output, which is the property that descriptive taxonomies and edit-based pipelines structurally cannot provide.

## Method

**What it is.** SGPC (Schema-Grounded Provenance Checker) is a generation-time gate for the 'presentation' work form. Its inputs are (i) the workspace state W (files, prior artifacts, declared constraints), (ii) a candidate artifact A produced by an LLM agent, and (iii) the agent's claimed provenance record P. Its output is an accept/reject verdict plus, on reject, a typed defect report that is fed back into the agent's next turn.

**Load-bearing assumption (stated explicitly).** SGPC assumes DERIVE_SCHEMAS deterministically extracts from W a schema set S_W that is complete and sound over the structure the agent legitimately needs, such that any provenance node absent from S_W is genuinely ungrounded rather than merely unextracted. This assumption is load-bearing: if it fails, UNGROUNDED_NODE conflates extraction incompleteness with genuine ungrounding. To make the assumption testable rather than assumed, DERIVE_SCHEMAS emits a coverage certificate C(W) and UNGROUNDED_NODE is split (see Phase 1 and Phase 2).

**How it works.**

```pseudocode
INPUT: workspace state W, candidate artifact A, claimed provenance P, budget k=3
OUTPUT: verdict in {ACCEPT, REJECT(defects)}, accepted lineage L*

# Phase 1: derive the ground schema set (not the agent's schema)
S_W, C_W = DERIVE_SCHEMAS(W)     # extract structural constraints from W:
                                 # field names/types from source files, ordering
                                 # constraints from prior accepted artifacts,
                                 # cardinality bounds from declared task spec
                                 # C_W = coverage certificate: for each region r of W,
                                 #   C_W[r] in {PARSED, UNPARSED} with byte offsets

# Phase 2: check provenance closure of P against S_W
for each provenance edge e = (A -> n) in P:
    if n is not in S_W:
        if region(n) in C_W and C_W[region(n)] == UNPARSED:
            emit defect UNEXTRACTED_REGION(n)   # routed to adjudication, not counted as grounding defect
        else:
            emit defect UNGROUNDED_NODE(n)      # node absent from a PARSED region
    else if not EDGE_WITNESSED(e, W):  emit defect UNWITNESSED_EDGE(e)
                                 # edge must correspond to an actual read/
                                 # derivation step recorded in W's access log

# Phase 3: independent conformance check of A against the terminal schema
s* = terminal(P)
D  = CONFORM(A, s*)             # structural diff: missing fields, type
                                 # mismatches, cardinality violations, ordering
if D nonempty: emit defect SCHEMA_VIOLATION(D)

# Phase 4: closure verdict
if no defects: return ACCEPT, lineage L* = P
else:
    for t in 1..k:               # repair loop, k=3
        A', P' = AGENT_REPAIR(A, P, defects, W)
        re-run Phases 2-3 on (A', P')
        if clean: return ACCEPT, L* = P'
    return REJECT(defects)
```

Hyperparameters: repair budget k=3; conformance is exact structural matching (no fuzzy scoring); schema derivation is deterministic extraction, not LLM-generated. Coverage certificate C_W is emitted by the same deterministic extractor and records, per region of W, whether the region was parsed (PARSED) or left unparsed (UNPARSED) with byte offsets; UNEXTRACTED_REGION defects are routed to a human/LLM adjudication path and are not counted as grounding defects in the primary metric.

**Why this design.** We chose to derive schemas deterministically from workspace state rather than letting the agent propose its own schema, because an agent-proposed schema makes the grounding condition circular (the generator certifies itself); the cost is that DERIVE_SCHEMAS must handle heterogeneous workspace formats, which we accept by restricting the presentation work form to workspaces with machine-readable source structure. We chose a witnessed-edge requirement over a simple schema-membership check because membership alone permits an agent to cite a schema it never used, which is exactly the failure mode PPTAgent's edit-based pipeline leaves open; the cost is that the harness must maintain an access log, which constrains the agent's runtime to a logged tool interface (as in SWE-agent's ACI). We chose an independent conformance checker over trusting the agent's self-report because self-reported conformance is the unverifiable claim the taxonomy already makes; the cost is that the checker is a separate component whose extraction rules must be maintained. We added the coverage certificate and the UNEXTRACTED_REGION split because the load-bearing completeness assumption would otherwise be untestable: the certificate makes explicit which regions of W were parsed, so UNGROUNDED_NODE is emitted only for nodes absent from a PARSED region, and nodes falling in UNPARSED regions are routed to adjudication rather than counted as grounding defects. This is not a variant of retrieval-based tool selection (Toolformer/Gorilla) nor of the descriptive framework in 'Work Agents': those systems select or describe, whereas SGPC rejects artifacts whose lineage cannot be witnessed, a decision that no retrieval or descriptive taxonomy performs.

**Why it measures what we claim.** The quantity `UNGROUNDED_NODE` measures schema-lineage traceability because a node absent from S_W within a PARSED region cannot have been derived from workspace state under the assumption that DERIVE_SCHEMAS is complete over W's PARSED regions; this assumption fails when W contains structure only expressible in natural language, in which case the node falls in an UNPARSED region and is emitted as UNEXTRACTED_REGION (routed to adjudication) rather than as UNGROUNDED_NODE, so the defect reflects extraction incompleteness only if the coverage certificate mislabels a region. The quantity `UNWITNESSED_EDGE` measures actual derivation because the access log records every read the agent performed under the assumption that the agent cannot fabricate log entries; this assumption fails when the agent has write access to the log, in which case the metric reflects log integrity rather than provenance. The quantity `SCHEMA_VIOLATION` measures conformance to the workspace-derived schema because exact structural matching is equivalent to satisfying the schema's constraints under the assumption that the schema fully specifies the artifact's required structure; this assumption fails for artifacts with legitimate free-form regions, in which case the violation reflects over-specification rather than artifact defect. The repair-loop acceptance rate measures the operational evaluability of the presentation work form because repeated rejection under a fixed checker is equivalent to the artifact being unverifiable under the assumption that the agent's repair attempts are representative of its capability; this assumption fails when the agent's repair policy is degenerate, in which case the rate reflects repair-policy quality rather than work-form evaluability.

## Contribution

(1) SGPC, a generation-time schema-grounding closure mechanism that makes artifact acceptance a decidable function of workspace state via witnessed provenance edges and an independent conformance checker. (2) The design principle that presentation-work-form evaluability requires deterministic schema derivation from workspace state rather than agent-proposed schemas, operationalizing the persistence/openness taxonomy as a verification condition. (3) A defect taxonomy (ungrounded node, unwitnessed edge, schema violation) with a bounded repair protocol that yields a reproducible acceptance metric for artifact-generating work.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | 60 presentation tasks over machine-readable workspaces (files, specs, prior artifacts); 20 per failure-mode subset | Exercises schema derivation and witnessed-edge checks on structured sources |
| Primary metric | Defect-typed rejection rate (UNGROUNDED_NODE / UNWITNESSED_EDGE / SCHEMA_VIOLATION), with UNEXTRACTED_REGION reported separately | Directly observes the three grounding failures the claim targets |
| Baseline | Self-certified agent (agent proposes schema, self-reports conformance) | Tests whether independent derivation beats generator self-certification |
| Baseline | Membership-only checker (schema membership, no edge witnessing) | Tests whether witnessed edges add power beyond membership |
| Baseline | PPTAgent-style edit pipeline (reimplementation of its edit-consistency check) | Tests against the pipeline whose failure mode the design names |
| Ablation-of-ours | SGPC without witness requirement (membership only, deterministic schema) | Isolates the witnessed-edge contribution from schema derivation |
| Control | Fixed non-adaptive repair policy (re-run same repair template, no defect feedback) | Bounds the repair-loop acceptance rate against degenerate repair |

### Why this setup validates the claim

The central claim is that a generation-time gate which derives schemas deterministically from workspace state, requires witnessed provenance edges, and checks conformance independently will reject artifacts whose lineage cannot be traced, and that this rejection is causally attributable to grounding failures rather than to agent weakness. Each baseline isolates one sub-claim. The self-certified agent tests whether independent schema derivation matters: if self-certification yields lower rejection rates on the same artifacts, the circularity argument is wrong. The membership-only checker tests whether the witnessed-edge requirement adds discriminating power beyond schema membership, which is the specific gap attributed to edit-based pipelines. The PPTAgent-style pipeline tests the end-to-end claim against the named failure mode. The ablation removes the witness requirement while keeping deterministic schema derivation, so any gap between it and full SGPC is attributable to witnessing alone. The fixed non-adaptive repair control bounds the repair-loop acceptance rate: if the adaptive repair loop does not exceed this control, the rate reflects repair-policy quality rather than work-form evaluability. The primary metric is defect-typed rejection rate because it is the only quantity that separates the three grounding failure modes; a single aggregate rejection rate would conflate ungrounding, unwitnessed derivation, and schema violation, and would not let us confirm that the predicted failure mode is the one detected. The dataset is chosen so that workspaces have machine-readable structure, which is the scope condition under which DERIVE_SCHEMAS is assumed complete; this makes the falsification test fair rather than rigged against the baselines. The coverage certificate C_W is logged for every workspace, and UNEXTRACTED_REGION defects are reported separately so that extraction incompleteness is visible rather than conflated with ungrounding.

### Expected outcome and causal chain

**vs. Self-certified agent** — On a case where the agent cites a schema field it invented to match its artifact rather than one present in the workspace, the self-certified baseline accepts the artifact because its schema is proposed by the generator and its conformance is self-reported, so the grounding condition is circular. Our method instead derives S_W from W and flags the invented field as UNGROUNDED_NODE because the node is absent from a PARSED region of the deterministically extracted schema set. We expect a noticeable rejection-rate gap concentrated on artifacts that reference non-existent fields, with parity on artifacts whose fields all trace to W.

**vs. Membership-only checker** — On a case where the agent cites a field that does exist in the workspace schema but never read it (no access-log entry), the membership-only baseline accepts because membership is satisfied, even though the derivation step did not occur. Our method instead requires EDGE_WITNESSED against the access log and emits UNWITNESSED_EDGE because no read corresponds to the cited edge. We expect a gap concentrated on artifacts citing plausible-but-unread fields, with parity on artifacts whose citations match logged reads.

**vs. PPTAgent-style edit pipeline** — On a case where the edit pipeline produces a structurally plausible artifact by copying surface form from prior artifacts without a witnessed derivation from source files, the baseline accepts because it checks edit consistency, not lineage. Our method instead runs independent CONFORM against the terminal schema and checks each provenance edge against W, emitting SCHEMA_VIOLATION or UNWITNESSED_EDGE because the copied structure lacks a witnessed source. We expect a gap concentrated on artifacts assembled from prior-artifact surface form, with parity on artifacts derived directly from source files.

**vs. Fixed non-adaptive repair control** — On a case where the agent's repair policy is degenerate (repeats the same repair template without using the defect report), the adaptive repair loop may accept artifacts that the control rejects, or vice versa. We expect the adaptive loop's acceptance rate to exceed the control's on the three failure-mode subsets, and we report the gap as the bound on repair-policy quality; if the gap is near zero, the repair-loop acceptance rate reflects repair-policy quality rather than work-form evaluability.

### What would falsify this idea

If the defect-typed rejection rate of full SGPC is statistically indistinguishable from the membership-only ablation across all three defect types, the witnessed-edge requirement is doing no work and the central claim about lineage traceability collapses. If instead SGPC's rejections are uniform across subsets rather than concentrated on the predicted failure-mode subsets (invented fields, unread citations, surface-form copying), the gate is rejecting for a reason other than grounding, and the causal attribution is wrong.

## References

1. Work Agents: Systems, Capabilities, and Optimization
2. ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery
3. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
