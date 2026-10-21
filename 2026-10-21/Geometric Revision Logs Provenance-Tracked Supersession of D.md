# Geometric Revision Logs: Provenance-Tracked Supersession of Depth and Normal Cues for 3D Spatial Reasoning

## Motivation

Existing geometric-evidence pipelines for spatial reasoning treat single-image depth and normal estimates as unconditional premises with no order relation among cues, so a hallucinated depth cannot be contradicted, only compounded. Geo3R extracts geometric evidence to reduce spatial hallucination but has no stateful substrate in which one cue can retire another; S-Agent 2.0 supplies persistent scene memory whose entries are monotone under supersession, yet its supersession is recency-ordered rather than geometrically grounded, so it cannot decide which of two conflicting depth readings is the surviving one. The field-level bottleneck is that geometric evidence extraction and error correction live in separate objects (a memory plus a checker), leaving the belief state over 3D cues without a well-defined, auditable frontier.

## Key Insight

Making supersession itself a geometric test — a cue is retired only when a later reading reprojects consistently across views and agrees with the surviving surface normal — turns the evidence store into a partially-ordered set whose frontier is always a geometrically consistent 3D interpretation, so correction and state are the same object rather than a memory plus a checker.

## Method

**What it is.** GRL (Geometric Revision Log) is a training-free evidence substrate for Geo3R-style 3D reasoning. Its input is a stream of tool readings (monocular depth maps, surface-normal maps, camera poses) tagged with the view and tool that produced them; its output is a revision log whose surviving frontier — the set of non-superseded entries — is the only evidence any downstream 3D reasoning chain may cite.

**How it works.**

```pseudocode
GRL(readings R, poses P, consistency threshold tau=0.85, normal-agreement eta=15deg,
    pose_sigma=0.01 (relative translation), depth_tol=0.05, min_views=2):
  log = []                      # entries: (id, view, tool, claim, depth, normal, provenance)
  for each reading r in R (in acquisition order):
    claim = interpret(r)        # plain-language: "surface at pixel p is ~d m away, facing n"
    entry = Entry(id, r.view, r.tool, claim, r.depth, r.normal, provenance=r)
    conflicts = [e for e in log if overlaps(e, entry)]
    for e in conflicts:
      score, residual, bias_pred = unify(e, entry, P, pose_sigma, depth_tol)
      # retire e only if entry's residual exceeds predicted estimator bias AND
      # score >= tau AND normal agreement holds AND at least min_views independent views agree
      if score >= tau and angle(e.normal, entry.normal) <= eta \
         and residual > bias_pred and independent_views(e, entry, log) >= min_views:
        mark_superseded(e, by=entry.id)   # retire, keep provenance link
      else if score < tau:
        entry.support.append(e.id)        # entry cannot retire e; record as unresolved
    log.append(entry)
  frontier = [e for e in log if not e.superseded]
  scene = rederive_abstraction(frontier)   # rebuild 3D scene from surviving cues only
  return log, scene

def unify(e, entry, P, pose_sigma, depth_tol):
  # reproject e's depth into entry's view via P; compare per-pixel depth
  reproj = warp(e.depth, P[e.view], P[entry.view])
  # pose-marginalized residual: integrate over pose uncertainty
  residual = mean_over_pose_samples( |reproj - entry.depth| / entry.depth )
  depth_consistency = fraction(|reproj - entry.depth| < depth_tol * entry.depth)
  normal_consistency = cos(e.normal, warp_normal(entry.normal))
  score = depth_consistency * normal_consistency   # scalar in [0,1]
  # predicted estimator bias for this viewing geometry (grazing angle, discontinuity)
  bias_pred = bias_model(e.view, entry.view, e.normal, entry.normal)  # calibrated from held-out set
  return score, residual, bias_pred

def independent_views(e, entry, log):
  # count distinct views (excluding e.view and entry.view) whose entries agree with entry
  # within depth_tol and normal eta; used to avoid correlated-error consensus
  return count of v in log where v.view not in {e.view, entry.view} and agrees(v, entry)
```

**Why this design.** We chose a single revision log over a separate memory module plus a consistency checker because a checker that runs after storage has no principled way to decide which of two conflicting cues to keep, accepting the cost that every insertion must pay a reprojection test. We chose geometric unification (cross-view reprojection plus depth-normal agreement) over recency ordering as the supersession rule because recency is not a valid contradiction witness — a later view can be wrong too — accepting the cost that supersession now depends on camera-pose accuracy. We chose to re-derive the scene abstraction from the surviving frontier on every supersession rather than patch the existing abstraction incrementally because patching leaves stale derived facts that are not in the log, accepting the cost of recomputation. This is not a variant of S-Agent 2.0's memory or of Geo3R's evidence extraction: S-Agent 2.0 retires entries by recency and Geo3R never retires them at all, whereas GRL's retirement is gated by a geometric consistency score that neither system computes.

**Why it measures what we claim.** The reprojection consistency score measures cross-view geometric agreement because it assumes the camera poses are accurate to within the depth tolerance; this assumption fails when pose estimation drifts, in which case the score reflects pose error rather than cue error. The depth-normal agreement term measures surface plausibility because it assumes a locally planar surface within a patch; this assumption fails at depth discontinuities, in which case it reflects occlusion boundaries rather than hallucination. The surviving-frontier cardinality measures the auditable belief state because it assumes every contradiction is eventually witnessed by some overlapping view; this assumption fails in single-view regions, in which case the frontier reflects unverified rather than verified evidence. The pose-marginalized residual and predicted bias term measure whether a dissenting cue is genuinely erroneous rather than merely disagreeing due to correlated estimator biases; this assumption fails when the bias model is mis-calibrated for the viewing geometry, in which case the gate may retire a correct cue that disagrees with a consensus of correlated errors.

## Contribution

(1) A geometric revision log that stores every depth/normal cue with full provenance and retires cues under a reprojection-and-normal-agreement supersession rule, yielding a partially-ordered evidence set with a well-defined surviving frontier. (2) The design principle that supersession must be geometrically grounded rather than recency-ordered, demonstrated by contrasting frontier stability under deliberately corrupted depth readings against a recency-ordered log. (3) A plug-in interface that lets any Geo3R-style reasoning chain draw only from the surviving frontier, with per-entry provenance sufficient to audit which view and tool produced each surviving claim.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MMSI-Bench + ReVSI multi-view subsets | Multi-view overlap is where revision actually fires |
| Primary metric | Frontier-consistency accuracy on conflicting-cue items | Directly measures whether surviving evidence is correct |
| Baseline | Same agent without tools | Isolates whether evidence substrate matters at all |
| Baseline | S-Agent 2.0 recency-retirement memory | Tests supersession rule against recency rival |
| Baseline | Geo3R evidence extraction (no retirement) | Tests whether retiring stale cues helps |
| Ablation-of-ours | GRL with recency supersession instead of geometry | Isolates the geometric-consistency gate |
| Ablation-of-ours | GRL with fixed tau/eta (no pose-marginalization, no bias model) | Tests whether calibrated gate is necessary |
| Ablation-of-ours | GRL with min_views=1 (no independent-view requirement) | Tests whether correlated-error consensus is a real failure mode |

### Why this setup validates the claim

The central claim is that a revision log whose supersession is gated by cross-view geometric consistency yields a more trustworthy evidence frontier than either no retirement (Geo3R) or recency-based retirement (S-Agent 2.0), and that downstream 3D reasoning accuracy tracks that frontier quality. Each baseline targets one sub-claim. The no-tools baseline establishes the floor: if GRL does not beat an agent with no perception at all, the substrate is doing nothing. S-Agent 2.0's recency memory tests whether geometric unification is genuinely a better contradiction witness than acquisition order — this is the sharpest rival because it also retires entries, so any gain must come from the *rule*, not from retirement per se. Geo3R's non-retiring extraction tests whether retirement itself helps, isolating the value of supersession. The ablation (GRL with recency supersession) is the decisive control: it holds the log structure, provenance, and re-derivation fixed and varies only the gate, so a gap between it and full GRL attributes the effect to geometric unification rather than to logging overhead. The additional ablations (fixed tau/eta, min_views=1) test whether the pose-marginalized residual and independent-view requirement are necessary to avoid correlated-error consensus, as the adversarial alert suggests. The primary metric — accuracy on items where two views genuinely conflict — is chosen because that is exactly the population where the competing retirement rules diverge; on non-conflicting items all methods should be near-parity, so concentrating measurement on conflict items makes the predicted effect detectable rather than diluted by easy cases. Conflict items are pre-registered: two views are deemed to genuinely conflict if their depth maps disagree by more than 10% at overlapping pixels after pose-based reprojection, and the ground-truth surviving cue is determined by manual inspection of the 3D scene.

### Expected outcome and causal chain

**vs. Same agent without tools** — On a case where the question asks about an object seen from a second viewpoint, the tool-free agent must answer from the single rendered view and confabulates the occluded side because it has no depth or pose evidence to reproject. Our method instead has depth, normals, and poses in the frontier and can re-derive the scene from surviving cues, so we expect a large gap on viewpoint-change and occlusion items and near-parity on single-view perspective items where the base agent already suffices.

**vs. S-Agent 2.0 recency memory** — On a case where an early accurate depth reading is contradicted by a later, drifted reading from a grazing-angle view, the recency baseline retires the good entry and keeps the bad one because acquisition order, not geometry, decides supersession. Our method instead runs the reprojection-and-normal test, finds the later reading inconsistent with the earlier view, and refuses to retire it, so we expect a gap concentrated on items whose later views are grazing-angle or pose-noisy, with parity on items where the latest view is also the most reliable.

**vs. Geo3R evidence extraction (no retirement)** — On a case where two views disagree about a surface's depth, the non-retiring system keeps both cues and the downstream chain cites whichever it encounters first, producing an answer that silently mixes incompatible geometry. Our method instead marks the losing entry superseded and re-derives the scene from the surviving frontier, so we expect higher frontier-consistency accuracy on conflict items and a smaller gap on non-conflict items, where keeping both cues is harmless.

**vs. GRL with fixed tau/eta (no pose-marginalization, no bias model)** — On a case where two cues from the same estimator family agree with each other but are both wrong (correlated error), the fixed-gate variant will retire a correct dissenting cue because it lacks the bias model and pose-marginalized residual. Our method instead requires the dissenting cue's residual to exceed the predicted bias for that viewing geometry and checks for independent views, so we expect a gap concentrated on items with correlated-error consensus, with parity on items where errors are independent.

**vs. GRL with min_views=1 (no independent-view requirement)** — On a case where only two views overlap and both are from the same estimator, the min_views=1 variant may retire a correct cue based on a single conflicting view. Our method requires at least two independent views to agree before retirement, so we expect higher accuracy on items with limited view overlap, with parity on items with many independent views.

### What would falsify this idea

If the ablation with recency supersession matches full GRL on conflict items, the geometric-consistency gate is not what produces the gain and the central claim is wrong. Likewise, if GRL's advantage is uniform across conflict and non-conflict subsets rather than concentrated on conflict items, the improvement is not coming from revision at all. If the fixed-gate ablation matches full GRL, the pose-marginalized residual and bias model are unnecessary and the simpler gate suffices.

## References

1. S-Agent 2.0: Spatial Reasoning with a General Coding Agent
2. Geo3R: Mitigating Spatial Reasoning Hallucination in Multimodal Large Language Models
3. Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation
