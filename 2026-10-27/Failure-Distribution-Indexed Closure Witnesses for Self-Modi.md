# Failure-Distribution-Indexed Closure Witnesses for Self-Modifying Vision-Language Tool Sets

## Motivation

Across both HexaShell-style robot-coding agents and LightZeroNav-style VLM navigation agents, the artifact set (tools / actions) is certified once by a small seed evaluation and then allowed to grow or be substituted with no invariant tying the patched set back to the failure distribution that motivated the patch. HexaShell explicitly builds tools on ten layouts of one seed per task, freezes them, and then evaluates on ten new layouts — a one-shot certification with no correctness-preservation check on subsequent patches; LightZeroNav similarly removes its frozen VLM ambiguity detector without any substitution-closure rule. The structural meta-gap is that neither lineage formalizes a closure operator over the self-modified artifact set, so 'the set still covers what it was built to cover' is never checkable against the perception substrate the agent already uses to act, and patches silently regress coverage of the very failures they were meant to fix.

## Key Insight

A patch is only certified if a witness re-derives, from the agent's own perception substrate, that the patched set still covers the specific perception-derived failure region that motivated it — making closure a property of the failure distribution rather than of a frozen seed suite, so substitution and growth cannot silently drop coverage.

## Method

```pseudocode
# FDC-Witness: Failure-Distribution-Indexed Closure Witness
# Inputs:  current artifact set A = {a_1..a_n} (tools/actions), perception substrate P
#          (calibrated RGB streams + geometry primitives the agent already uses to act)
#          signature tuple S = (object_identity, view_geometry, occlusion_state)
#          (hand-specified; its sufficiency is ASSUMED, not proven — see (D))
# Output:  certified A' or rejection with a localized coverage gap

procedure PATCH_AND_CERTIFY(failure_log L, proposed patch p):
    # 1. Extract the failure distribution that motivated the patch
    F_motivate <- EXTRACT_FAILURE_DIST(L, P)      # perception-derived failure region
    # 2. Apply patch to get candidate set
    A' <- APPLY(A, p)                              # grow or substitute an artifact
    # 3. Build a closure witness: re-derive coverage on F_motivate
    W <- { }
    for each failure cell c in F_motivate:         # cells = (object, view, occlusion) tuples
        # re-render / re-observe c through the SAME perception substrate P
        percept <- SAMPLE_SUBSTRATE(P, c)
        covered <- ANY(a in A' : a.SUCCEEDS_ON(percept))
        W[c] <- covered
    # 4. Closure invariant: every cell in F_motivate must be covered
    if ALL(W[c]) and NO_REGRESSION(A', A, F_motivate):
        return CERTIFY(A', W)
    else:
        return REJECT(A', LOCALIZE_GAP(W))

procedure EXTRACT_FAILURE_DIST(L, P):
    # group failures by perception-derived signature, not by task id
    return GROUP_BY(L, key = (object_identity, view_geometry, occlusion_state))

procedure NO_REGRESSION(A', A, F):
    # the patched set must not lose coverage A had on F
    return FOR_ALL cells c in F: A'.covers(c) >= A.covers(c)
```

**What it is.** FDC-Witness is a closure operator that indexes every self-modification of a VLM agent's tool/action set to the perception-derived failure distribution that motivated it, and emits a witness — a per-failure-cell coverage certificate checked against the agent's own perception substrate rather than a frozen seed suite.

**How it works (diagram).**

```
 failure log L ──► EXTRACT_FAILURE_DIST ──► F_motivate (cells)
                                               │
 proposed patch p ──► APPLY(A,p) ──► A'        │
                                      │        ▼
                                      └──► COVERAGE CHECK on F_motivate
                                               │  (re-sample via P)
                                               ▼
                                    witness W (per-cell covered?)
                                               │
                              ALL covered AND NO_REGRESSION?
                                    │yes              │no
                                    ▼                 ▼
                              CERTIFY(A',W)     REJECT + LOCALIZE_GAP
```

**Why this design.** We chose to index closure to the *failure distribution* rather than to a held-out seed suite (as HexaShell does with its ten frozen layouts) because a seed suite is a fixed sample that the patch is not required to relate to, so a patch can pass the suite while regressing exactly the failures that motivated it; the cost is that F_motivate must be re-materialized per patch, which is more expensive than a single frozen evaluation. We chose to re-sample each failure cell through the agent's *own* perception substrate P rather than through a simulated oracle because the substrate is the surface the agent actually acts on, so the witness is checked against the same distribution of views/occlusions the agent will face; the cost is that P must expose geometry primitives (calibration, plane intersections) — the very assumption HexaShell already relies on. We chose a per-cell all-covered plus no-regression condition over a scalar coverage score because a scalar can be gamed by a patch that trades one failure cell for another, whereas the all-covered conjunct forces closure on every cell that motivated the patch; the cost is brittleness when a single cell is genuinely infeasible, which we handle by localizing and reporting the gap rather than silently passing. This differs from a nearest-neighbor retrieval of tools (Toolformer/Gorilla-style) in that the witness is a *counterfactual coverage check on an intervened artifact set*, not a similarity match to a library entry — an expert would not describe it as retrieval because no library lookup occurs; the artifact set is modified and then re-verified against a distribution derived from the agent's own failures.

**Why it measures what we claim.** `F_motivate` (the failure distribution extracted from L) measures *the failure region that motivated the patch* because failures are grouped by perception-derived signature (object identity, view geometry, occlusion state) under the assumption that two failures with the same signature are the same failure mode; this assumption fails when two distinct root causes share a signature (e.g., a mislocalized object and a missed grasp both appear as 'occluded wrist view'), in which case F_motivate conflates them and the witness certifies coverage of a coarser region than intended. `W[c]` (per-cell coverage via re-sampling P) measures *closure of the patched set over the motivating failures* because re-observing the same cell through the agent's own substrate is assumed to reproduce the perception the agent will encounter at test time; this assumption fails under distribution shift in the substrate (e.g., lighting change between patch time and deployment), in which case W reflects coverage of the patch-time substrate rather than deployment. `NO_REGRESSION` measures *correctness preservation* because it assumes the pre-patch set's coverage on F_motivate is a lower bound the patch must not lower; this fails when the pre-patch set was itself already wrong on F_motivate, in which case no-regression preserves a pre-existing error rather than true correctness.

**(D) Load-bearing assumptions made explicit.**

- **(D1) Signature sufficiency.** The tuple S = (object_identity, view_geometry, occlusion_state) is assumed to be a sufficient statistic for failure mode: two failures sharing S are the same root cause, and covering the S-cell implies covering the failure. This is *assumed*, not proven. It fails under perceptual aliasing (visually near-identical observations mapping to different action failures) and under VLM nuisance sensitivity (typographic/viewpoint perturbations that leave S unchanged but change the failure). When it fails, F_motivate is a coarsened partition and the witness certifies closure over the wrong partition. We measure the conflation rate on a labeled subset: for each cell in a held-out labeled set of 200 failures with known root-cause labels, we report the fraction of cells whose members span >1 root cause. We do not replace S with a learned partition; we report the conflation rate so the assumption is measured rather than assumed.

- **(D2) Substrate exchangeability.** `W[c]` measures deployment-time closure only under the assumption A that patch-time substrate samples are exchangeable with deployment perception. This fails under lighting/occlusion shift between patch time and deployment (failure mode F). When A breaks, W reflects coverage of the patch-time substrate, not deployment. We report the shift rate as the fraction of cells whose coverage flips between a patch-time substrate sample and a held-out deployment-time substrate sample on a labeled subset of 100 cells.

- **(D3) Pre-patch correctness.** `NO_REGRESSION` measures correctness preservation only under the assumption that the pre-patch set's coverage on F_motivate is a lower bound the patch must not lower. This fails when the pre-patch set was itself already wrong on F_motivate; then no-regression preserves a pre-existing error. We report the pre-patch error rate on F_motivate on the same labeled subset, so the reader can see how often the lower bound is itself wrong.

## Contribution

(1) A failure-distribution-indexed closure operator (FDC-Witness) that certifies every patch or substitution of a VLM agent's tool/action set against a perception-derived failure distribution re-sampled through the agent's own substrate, with an all-covered plus no-regression invariant and a localized coverage-gap report on rejection. (2) The design principle that closure must be indexed to the motivating failure distribution rather than a frozen seed suite, demonstrated by the mechanism's ability to reject patches that pass a HexaShell-style frozen-layout evaluation while regressing the specific failures that motivated them. (3) A reusable witness schema (per-failure-cell coverage certificates) that makes 'the set still covers what it was built to cover' a checkable, loggable object for self-modifying vision-language agents.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Held-out manipulation layouts from HexaShell task suite (10 layouts × 10 seeds = 100 episodes) | Same substrate and geometry primitives the method assumes |
| Primary metric | Per-failure-cell closure rate on re-materialized F_motivate (cells = S-tuples) | Directly measures the closure invariant the method certifies |
| Secondary metric | Signature conflation rate on 200 labeled failures (D1) | Measures whether S is a sufficient statistic |
| Secondary metric | Substrate shift rate on 100 labeled cells (D2) | Measures patch-time vs deployment exchangeability |
| Secondary metric | Pre-patch error rate on F_motivate (D3) | Measures whether no-regression lower bound is itself correct |
| Baseline | Frozen seed-suite gate (HexaShell ten-layout protocol) | Tests whether relating patch to failure distribution matters |
| Baseline | Scalar coverage score gate | Tests whether per-cell conjunction beats aggregate score |
| Baseline | Nearest-neighbor tool retrieval (Toolformer/Gorilla-style) | Tests whether counterfactual re-verification beats similarity match |
| Ablation-of-ours | FDC-Witness without NO_REGRESSION check | Isolates whether the no-regression conjunct carries weight |
| Patch corpus | Pre-registered 40 patches with known motivating cells | Bounds per-patch re-materialization cost |

### Why this setup validates the claim

The central claim is that indexing closure to the perception-derived failure distribution that motivated a patch, rather than to a frozen seed suite, catches patches that pass a fixed sample while regressing the failures they were meant to fix. The frozen seed-suite baseline tests exactly this sub-claim: if a patch can pass ten held-out layouts while reopening the motivating failure cell, the seed-suite gate will certify it and our method will reject it, so the two gates should diverge precisely on those patches and agree elsewhere. The scalar coverage baseline tests the second sub-claim, that a per-cell all-covered conjunction resists coverage-trading in a way an aggregate score cannot; we expect the scalar gate to accept patches that fix one cell while breaking another, which our witness localizes. The retrieval baseline tests whether the witness is doing real work beyond similarity matching: retrieval can propose a tool that looks like the right one but is never re-verified against the actual failure region, so its closure rate should track library proximity rather than the motivating failures. The ablation isolates the no-regression conjunct, telling us whether correctness preservation is load-bearing or whether all-covered alone suffices. The primary metric, per-cell closure rate on the re-materialized F_motivate, is the right detector because it is computed on the very cells the method claims to certify; a gain that does not show up here would mean the witness is not measuring what it claims. The three secondary metrics (D1–D3) make the load-bearing assumptions measurable: conflation rate on 200 labeled failures tells us how often S conflates distinct root causes; substrate shift rate on 100 labeled cells tells us how often patch-time samples fail to predict deployment; pre-patch error rate tells us how often the no-regression lower bound is itself wrong. These are reported alongside the primary metric so the reader can see the conditions under which the closure claim holds.

### Expected outcome and causal chain

**vs. Frozen seed-suite gate (HexaShell ten-layout protocol)** — On a case where a patch fixes an occlusion-handling tool but subtly breaks a specific wrist-view grasp cell that the ten frozen layouts do not exercise, the baseline certifies the patch because its fixed sample never instantiates that cell. Our method instead re-materializes that exact cell through the agent's own substrate and finds it uncovered, so it rejects and localizes the gap. We expect a concentrated gap: near-parity on layouts the seed suite covers, but a noticeable acceptance-rate divergence on patches whose motivating failure cells sit outside the frozen sample. We also report the D1 conflation rate on the 200 labeled failures; if it is high, the divergence may be attenuated because F_motivate is coarser than the true failure partition.

**vs. Scalar coverage score gate** — On a case where a patch trades a mislocalized-object cell for a missed-grasp cell, the baseline's aggregate score rises or holds flat because the two cells cancel, so it certifies a patch that has merely moved the failure. Our method's per-cell all-covered conjunction sees one cell flip to uncovered and rejects. We expect the scalar gate to accept a visible fraction of coverage-trading patches that our witness rejects, with the divergence concentrated on patches whose motivating distribution spans multiple cells. The D2 substrate shift rate bounds how much of this divergence is attributable to patch-time vs deployment mismatch rather than to the conjunction itself.

**vs. Nearest-neighbor tool retrieval (Toolformer/Gorilla-style)** — On a case where the failure log describes an occluded wrist view and retrieval returns a library tool that matches on task description but was built for a different occlusion geometry, the baseline installs it without re-verification and the motivating cell stays uncovered. Our method applies the patch and then re-samples that cell through the agent's substrate, catching the mismatch. We expect our closure rate to exceed retrieval's specifically on cells whose signature is shared across distinct root causes, with parity on cells that retrieval happens to match exactly. The D1 conflation rate tells us how often the shared-signature case actually occurs in the labeled subset.

**vs. FDC-Witness without NO_REGRESSION** — On a case where the pre-patch set already covers a cell correctly and the patch would drop it, the ablation certifies because all-covered holds on the remaining cells, while our full method rejects. We expect the ablation to accept a fraction of patches that regress pre-existing coverage; the D3 pre-patch error rate tells us how often the no-regression conjunct is protecting a correct lower bound versus preserving a pre-existing error.

### What would falsify this idea

If our gain over the frozen seed-suite gate is uniform across all patches rather than concentrated on patches whose motivating failure cells fall outside the seed sample, the mechanism is not doing what we claim and the improvement comes from something else. Likewise, if the no-regression ablation shows no difference in accepted-patch correctness, the closure invariant reduces to all-covered and the design rationale collapses. Additionally, if the D1 conflation rate on the 200 labeled failures is high (e.g., >30%), the signature-sufficiency assumption is empirically false and the closure claim is over a coarsened partition.

## References

1. HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents
2. AutoEval: Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real World
3. Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents
4. AlphaEvolve: A coding agent for scientific and algorithmic discovery
5. Autonomous Improvement of Instruction Following Skills via Foundation Models
6. ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI
7. Multi-objective Evolution of Heuristic Using Large Language Model
8. Amplifying human performance in combinatorial competitive programming
