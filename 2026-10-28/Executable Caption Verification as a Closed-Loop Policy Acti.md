# Executable Caption Verification as a Closed-Loop Policy Action in Collaborative Video Planning

## Motivation

The CodeCap-Bench work establishes that executable code captions preserve video information more faithfully than natural language, yet current MLLMs struggle to generate faithful code captions, and the VideoChat-M1 collaborative policy-planning loop has no mechanism to enforce that a proposed plan's code caption is syntactically valid or semantically aligned with the video before execution. Structurally, the loop treats video understanding as multi-agent natural-language dialogue, so validity is never a precondition for accepting a plan — a code caption can be syntactically broken or semantically misaligned and still be accepted, because no policy action exists that can fail and thereby veto the plan. This is the shared residual gap: executable semantics and round-trip fidelity are assumed rather than enforced at the point of plan acceptance.

## Key Insight

A code caption is an executable program whose syntactic validity is a decidable property and whose semantic alignment can be measured by differential execution against the video's observable state, so making caption generation and verification a mandatory policy action before plan acceptance converts an unverifiable natural-language agreement into a decidable accept/reject gate.

## Method

**What it is.** We introduce ECV (Executable Caption Verification), a closed-loop policy action inside the VideoChat-M1 planner: before any plan is accepted, the planner must emit a code caption, execute it in a sandboxed interpreter, and pass both a syntactic-validity check and a semantic-alignment check against the video's observable state; failure returns the execution trace to the planner as a rejection signal and the plan is revised. Inputs are the video and the candidate plan; outputs are an accepted plan with its verified code caption, or a rejection with a diagnostic trace.

**Load-bearing assumption (stated explicitly).** The semantic gate assumes that a code caption's rendered program states can be compared pixel-and-edge-wise to sampled video frames such that normalized distance is a monotone proxy for caption faithfulness, i.e., that the primitive library can render the video's observable content closely enough that low alignment means a wrong caption rather than an inexpressive library. This is the analysis-by-synthesis identifiability problem: rendering fidelity is bounded by the primitive library and scene parameterization, so raw distance conflates caption error with rendering coverage. To make the gate measure caption-attributable error rather than library coverage, we calibrate alignment against a per-video renderer ceiling and require the diff map to localize to caption-relevant primitives, as specified below.

**How it works.**
```pseudocode
ECV(video V, candidate plan P, planner M, budget B=3):
  # --- renderer calibration (once per video, before gating) ---
  # fit an oracle program C_oracle to V by black-box search over the primitive library
  # (random restarts + coordinate descent on primitive parameters, 200 iterations)
  frames_oracle = sandbox_execute(C_oracle, t=0..T, dt=1)
  residual_floor = mean_t d(frames_oracle[t], frames_obs[t]) / d_max   # per-video ceiling
  # tau_align is now a RELATIVE threshold on residual above the floor
  tau_rel = 0.18   # candidate must reduce residual by >=18% relative to floor

  for attempt in 1..B:
    C = M.generate_code_caption(V, P)          # policy action: caption as program
    # --- syntactic gate (decidable) ---
    ast = parse(C)
    if ast is None:
      M.update(reject="syntax", trace=parse_error(C)); continue
    # --- semantic gate (differential execution, calibrated) ---
    frames_hat = sandbox_execute(C, t=0..T, dt=1)   # rendered program states
    frames_obs = sample_frames(V, t=0..T, dt=1)
    residual = mean_t d(frames_hat[t], frames_obs[t]) / d_max   # d = normalized pixel+edge distance
    # relative alignment: improvement over the renderer ceiling
    align_rel = (residual_floor - residual) / max(residual_floor, 1e-6)
    if align_rel < tau_rel:
      M.update(reject="semantic", trace=diff_map(align_rel, t)); continue
    # --- localization check: diff must be attributable to caption primitives ---
    # ablate each primitive call in C and re-render; require residual to increase
    # for at least one primitive (i.e., the caption is doing causal work)
    delta_residual = []
    for prim in C.primitives:
      C_abl = C.without(prim)
      frames_abl = sandbox_execute(C_abl, t=0..T, dt=1)
      delta_residual.append( mean_t d(frames_abl[t], frames_obs[t])/d_max - residual )
    if max(delta_residual) < 0.02:   # no primitive improves fit -> caption not attributable
      M.update(reject="localization", trace=delta_residual); continue
    # --- round-trip gate ---
    C_rev = M.generate_code_caption(V, P, condition=C)  # re-caption conditioned on accepted C
    if edit_distance(C, C_rev) > tau_rt (0.15):
      M.update(reject="roundtrip", trace=edit_ops(C,C_rev)); continue
    return accept(P, C, align_rel, roundtrip=edit_distance(C,C_rev))
  return reject(P, trace=last_trace)   # planner must re-plan, not re-caption
```
The three gates are ordered by cost: syntax is O(parse), semantic is O(T·render) plus O(|C|·T·render) for the localization ablation, round-trip is O(re-caption). Rejection traces are appended to the planner's dialogue state so the next plan is conditioned on the specific failure mode, not just a scalar score.

**Why this design.** We chose a decidable syntactic gate over a learned validity classifier because syntax is exactly checkable and a learned classifier would reintroduce the calibration problem that makes natural-language self-verification unreliable; the cost is that syntactically valid but semantically wrong captions pass the first gate and must be caught downstream. We chose differential execution against sampled frames over a learned alignment scorer because rendering the program produces an observable state that can be compared pixel-wise, giving a signal that is grounded in the video rather than in the model's self-report; the cost is that rendering fidelity is bounded by the primitive library, so domains outside the library's expressiveness will show artificially low alignment — which is why we calibrate against a per-video renderer ceiling and require primitive-level localization, so the gate measures caption-attributable error rather than library coverage. We chose a round-trip re-caption gate over a single-pass confidence check because edit distance between two independently conditioned generations is an external consistency signal that does not rely on the model's log-probs, which prior work (Kadavath et al.) shows are poorly calibrated for correctness; the cost is one extra generation per plan, tripling caption cost in the worst case. We chose to feed rejection traces back into the planner's dialogue state rather than restarting the planner because the trace localizes the failure (which frame, which edit), turning rejection into actionable revision; the cost is that the planner may overfit to the trace and produce captions that pass gates without improving the plan. A domain expert would not describe ECV as a variant of Toolformer or a retrieval-based tool selector: those methods choose among existing tools by likelihood, whereas ECV constructs a new program per plan and gates it on execution against the video, a signal no retrieval library contains.

**Why it measures what we claim.** The syntactic gate measures executable validity because a successful parse under the grammar is equivalent to the caption being a well-formed program; this assumption fails when the grammar is too permissive, in which case the gate reflects grammar membership rather than semantic correctness. The calibrated alignment score measures semantic alignment with the video because the relative improvement of rendered program states over a per-video renderer ceiling, together with the requirement that ablating a caption primitive increases residual, is equivalent to the caption's observable consequences matching the video beyond what the library can achieve by coverage alone, under the assumption that the oracle program fit captures the library's expressiveness ceiling; this assumption fails when the oracle search underfits (e.g., too few restarts), in which case alignment reflects search quality rather than caption faithfulness. The round-trip edit distance measures caption stability because two generations conditioned on the same video and plan but different contexts (with vs. without the accepted caption) agreeing in edit distance is equivalent to the caption being determined by the video rather than by sampling noise, under the assumption that the conditioning context is the only difference between the two generations; this assumption fails when the planner's context window truncates the video, in which case edit distance reflects context loss rather than instability. Together these three quantities operationalize the motivation's two concepts — executable semantics (syntax + calibrated alignment + localization) and round-trip fidelity (edit distance) — with the named assumptions making each equivalence explicit.

## Contribution

(1) A closed-loop policy action, ECV, that makes code-caption generation, sandboxed execution, and dual syntactic/semantic verification a mandatory precondition for plan acceptance in a collaborative video-planning loop. (2) A design principle for validity enforcement in multi-agent video planning: order gates by decidability and cost (syntax → differential execution → round-trip), and feed localized rejection traces back into the planner's dialogue state rather than scalar scores. (3) An analysis of when differential execution against sampled frames is a valid alignment signal, identifying the primitive-library expressiveness boundary as the binding condition.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | CodeCap-Bench video set, restricted to videos with library coverage ≥ 0.70 (measured as fraction of frames renderable by the primitive library; see coverage measurement below) | Pairs code captions with reconstructable video states; restriction prevents low alignment from being absorbed as caption error |
| Primary metric | Plan acceptance rate with verified captions | Directly measures gated executable validity |
| Secondary metric | Downstream reconstruction/QA accuracy on accepted plans | Tests whether passing gates implies better plan quality |
| Baseline | Natural-language caption planner | Tests whether executable form beats prose |
| Baseline | Single-pass self-verification planner | Tests whether external gates beat log-prob confidence |
| Baseline | Toolformer-style tool selector | Tests whether construction beats retrieval |
| Baseline | Learned verifier (reward model trained on CodeCap-Bench accept/reject labels) | Tests whether decidable gates beat a learned verifier on the same planner |
| Baseline | Budget-matched no-gate control (same number of caption generations as ECV, no gates) | Tests whether gain is from gating, not extra sampling |
| Ablation-of-ours | ECV without semantic gate | Isolates differential execution contribution |
| Ablation-of-ours | ECV with absolute alignment threshold (tau_align=0.82) instead of calibrated relative threshold | Isolates the effect of renderer calibration |

**Coverage measurement.** Before evaluation, we measure library coverage on CodeCap-Bench: for each video, we fit an oracle program by black-box search (200 iterations of random restarts + coordinate descent on primitive parameters) and record the residual floor. Coverage is defined as the fraction of frames where the oracle residual is below 0.25 (normalized pixel+edge distance). Videos with coverage < 0.70 are excluded from the primary evaluation set and reported separately as a coverage-limited subset. This precondition is stated explicitly so that low alignment on out-of-library videos is not silently treated as caption error.

**Calibration study.** On a held-out subset of 200 videos, we collect human faithfulness judgments (3 annotators, majority vote) for 5 candidate captions per video (1,000 captions total) and compute Spearman rank correlation between the calibrated alignment score (align_rel) and human judgments. We also report the threshold tau_rel that maximizes agreement (F1 of accept/reject against human labels) and use that value in the main experiment; if the optimal tau_rel differs from 0.18, we report both.

### Why this setup validates the claim

The central claim is that gating a planner's code caption on decidable syntax, calibrated differential execution against sampled frames, and round-trip stability produces more executable, video-grounded plans than unverified generation. CodeCap-Bench is the right dataset because it already supplies videos whose content is expressible as programs and a reconstruction/QA substrate, so a caption's executable semantics can be checked rather than asserted; restricting to videos with library coverage ≥ 0.70 ensures that the semantic gate measures caption-attributable error rather than rendering coverage, directly addressing the load-bearing assumption. The natural-language baseline tests the sub-claim that executable form is necessary; the single-pass self-verification baseline tests the sub-claim that the model's own confidence is insufficient and external gates are required; the Toolformer-style selector tests the sub-claim that constructing a program per plan beats retrieving an existing tool by likelihood; the learned verifier baseline tests whether the decidable gates are a genuine novelty or just a different verifier; and the budget-matched no-gate control tests whether any gain is due to gating rather than to the extra caption generations. The ablation removing the semantic gate isolates differential execution, and the ablation using an absolute threshold isolates the effect of renderer calibration. Plan acceptance rate with verified captions is the primary metric because it counts exactly the events the method claims to improve — plans that survive all three gates — and it will move only if the gates actually reject bad captions and admit good ones; downstream reconstruction/QA accuracy on accepted plans is the secondary metric that tests whether passing gates implies better plan quality, guarding against gates tuned to admit easy captions.

### Expected outcome and causal chain

**vs. Natural-language caption planner** — On a case where the plan requires a temporal operation like "the object enters after the light changes," the natural-language baseline produces a fluent sentence with no executable structure, so downstream reconstruction cannot render the described state and the plan's semantic content is unverifiable. Our method instead emits a program whose primitive calls render frame states that are compared pixel-wise to the video, because the caption is forced into an executable grammar. We expect a clear acceptance-rate gap concentrated on temporally compositional videos, with near-parity on single-object static scenes where prose is already unambiguous.

**vs. Single-pass self-verification planner** — On a case where the model generates a syntactically valid but semantically wrong caption, the self-verification baseline accepts it because its confidence is high and uncorrelated with correctness, so the wrong plan proceeds. Our method instead catches it at the semantic gate, where calibrated residual exceeds the relative threshold and the diff map returns a rejection trace that conditions the next attempt. We expect the gap to appear specifically on the subset where the baseline's confidence is high but calibrated alignment is low, and to shrink toward parity on cases where confidence and correctness coincide.

**vs. Toolformer-style tool selector** — On a case where no retrieved tool matches the video's specific event sequence, the selector picks the closest existing tool by likelihood and produces a caption that under-describes the scene. Our method instead constructs a new program conditioned on the plan and gates it on execution against this video, because the primitive library composes rather than retrieves. We expect a gap that grows with the compositional novelty of the video, and near-parity on videos whose content matches a single library primitive.

**vs. Learned verifier (reward model)** — On cases where the learned verifier's training distribution does not cover the failure mode (e.g., syntactically valid but semantically wrong captions that are rare in CodeCap-Bench accept/reject labels), the learned verifier may accept the wrong caption because it generalizes from surface features. Our method instead applies a decidable syntactic check and a calibrated execution check that do not rely on labeled accept/reject data. We expect the decidable gates to win on out-of-distribution failure modes and the learned verifier to win on in-distribution cases where its training labels are dense; we will report the split.

**vs. Budget-matched no-gate control** — The control generates the same number of captions as ECV (up to 3 per plan) but accepts the first syntactically valid caption without semantic or round-trip gates. If ECV's acceptance rate exceeds the control, the gain is attributable to gating rather than to extra sampling. We expect ECV to outperform the control on the subset where the first syntactically valid caption is semantically wrong, and to match the control on easy videos where the first caption is correct.

**Ablation: ECV without semantic gate** — Removing the semantic gate leaves syntax and round-trip gates. We expect acceptance rate to increase (fewer rejections) but downstream reconstruction/QA accuracy on accepted plans to decrease, because semantically wrong captions are admitted. This isolates the contribution of differential execution.

**Ablation: ECV with absolute alignment threshold** — Replacing the calibrated relative threshold with the original absolute tau_align=0.82 tests whether renderer calibration matters. We expect the absolute threshold to reject many correct captions on videos with high residual floor (low coverage) and to accept some wrong captions on videos with low residual floor, leading to lower acceptance rate and lower downstream accuracy than the calibrated version, with the gap concentrated on the coverage-limited subset.

### What would falsify this idea

If the acceptance-rate gain over the self-verification baseline is uniform across all subsets rather than concentrated where confidence is miscalibrated and calibrated alignment is low, the gates are not doing the causal work we claim. Likewise, if removing the semantic gate leaves acceptance and downstream reconstruction unchanged, or if the budget-matched no-gate control matches ECV, differential execution is not the mechanism and the central claim is wrong.

## References

1. CodeCap-Bench: Can Code Serve as a Good Video Caption?
