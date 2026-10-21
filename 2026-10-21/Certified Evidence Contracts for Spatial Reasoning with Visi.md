# Certified Evidence Contracts for Spatial Reasoning with Vision-Language Agents

## Motivation

Existing spatial reasoning agents treat perception tool outputs as oracle inputs: S-Agent 2.0 consumes CLI 3D percepts without any failure-substitution rule, and Geo3R assumes single-image depth/normal estimates are reliable enough to condition reasoning. Structurally, both lineages lack a per-evidence success predicate that couples tool invocation to downstream reasoning invariants, so a corrupt depth patch or non-integrable normal field silently enters the language model's context and produces confident but ungrounded spatial claims.

## Key Insight

A certificate computed from a tool's raw output plus the known camera/scene model—such as metric consistency of a depth patch under the camera projection or integrability of a normal field around a closed loop—is a structural invariant that is monotone in evidence validity and requires no learned confidence head, so gating reasoning on certified evidence converts silent oracle failures into explicit, substitutable rejections.

## Method

**What it is.** CERT (Certified Evidence for Reasoning Tasks) is a contract layer between a vision-language agent's perception tools and its spatial reasoning context. Inputs: a tool call (depth, normal, segmentation, or CLI probe) with its raw output and the known camera/scene model. Output: either a certified evidence token that may enter the reasoning context, or a rejection that triggers a counterfactual re-query whose admissibility is established by the same certificate.

**How it works.**

```pseudocode
def CERT_contract(tool_call, raw_output, camera_model, scene_memory):
    # Phase 1: compute certificate from raw output + model only (no learned head)
    cert = compute_certificate(tool_call.tool, raw_output, camera_model)
    # depth: reproject patch to 3D, check metric consistency under camera_model
    #   (residual = ||proj(K * T * X) - pixel||_2, threshold tau_depth)
    # normals: integrate over closed loop, check residual < tau_int
    # segmentation: check mask area vs. expected solid-angle under camera_model
    #   (residual = |area_mask - area_expected| / area_expected, threshold tau_seg)
    # global anchor: cross-tool consistency between depth and segmentation solid-angle
    #   (residual = |depth_scale * area_depth - area_seg| / area_seg, threshold tau_anchor)

    # Phase 2: admissibility gate
    if cert.passes(tau):
        return CertifiedEvidence(raw_output, cert, tool_call)

    # Phase 3: counterfactual re-query — same certificate, different invocation
    alt_call = counterfactual_requery(tool_call, cert.failure_mode)
    # e.g., re-query depth with shifted patch / different scale; re-query normals with
    # different integration path; re-query segmentation with different prompt
    alt_output = execute(alt_call)
    alt_cert = compute_certificate(alt_call.tool, alt_output, camera_model)

    if alt_cert.passes(tau):
        return CertifiedEvidence(alt_output, alt_cert, alt_call)
    else:
        return RejectedEvidence(tool_call, cert, alt_cert)
    # Rejected evidence is NOT consumed; the LM's spatial claim is blocked

# In the agent loop:
# 1. Agent proposes spatial claim C conditioned on evidence set E
# 2. For each e in E: e must be CertifiedEvidence; otherwise C is not allowed
# 3. If any e is RejectedEvidence, agent must either re-query or abstain
```

**Threshold calibration.** The thresholds `tau = {tau_depth, tau_int, tau_seg, tau_anchor}` are not fixed geometric constants. They are fit per tool and per failure mode on a held-out calibration set of 512 tool outputs (256 known-good, 256 known-bad, labeled by ground-truth validity from the dataset's sensor annotations or manual inspection). For each tool and failure mode, we select the threshold that maximizes the F1 score of `cert.passes(tau)` against the binary validity label on this calibration set. This calibration makes explicit the load-bearing assumption: **A certificate computed purely from a tool's raw output plus the known camera/scene model is both necessary and sufficient to determine whether that evidence is admissible for downstream spatial reasoning, i.e., structural consistency with the camera model is equivalent to evidence validity.** This assumption fails when the camera model itself is wrong or when the tool output is self-consistent but globally biased (e.g., monocular depth with unknown scale). To mitigate the latter, we include the global anchor check (cross-tool consistency) and report certificate precision/recall against ground-truth validity on a separate test set of 512 outputs, so the equivalence is measured rather than assumed.

**Why this design.** We chose a certificate computed solely from the tool's raw output and the known camera/scene model over a learned confidence head because a learned head cannot manufacture the structural coupling between tool output and reasoning invariant that the motivation demands; accepting the cost that the certificate is only as strong as the camera model's accuracy. We chose a single contract surface (same certificate for both initial and counterfactual evidence) over a router between verifier and reasoner because a router would reintroduce the two-module anti-pattern and leave the hard problem—deciding when to trust—unspecified; accepting the cost that the contract must be expressive enough to cover heterogeneous tools. We chose counterfactual re-query with the same certificate over a fixed fallback tool because the same certificate establishes admissibility of the substitute, making the substitution rule auditable; accepting the cost that some failures may require multiple re-queries. Unlike Toolformer or Gorilla, which retrieve tools by descriptor similarity, CERT does not select tools—it certifies whether a tool's output is admissible for reasoning, a property retrieval cannot compute.

**Why it measures what we claim.** `cert.passes(tau)` measures evidence validity because the certificate is a structural invariant (metric consistency, integrability, solid-angle consistency, cross-tool anchor) that is equivalent to the tool output being consistent with the known camera/scene model; this assumption fails when the camera model itself is wrong or when the tool output is self-consistent but globally biased, in which case the certificate reflects consistency with a false model or a biased output rather than truth. `counterfactual_requery` measures substitutability because it intervenes on the tool invocation while holding the certificate fixed, so a passing re-query establishes that admissible evidence exists under the same invariant; this assumption fails when the failure mode is systematic across all invocations, in which case the re-query reflects persistent tool bias rather than a local failure. `RejectedEvidence` measures the absence of admissible evidence for the claim because it is returned only when both initial and counterfactual certificates fail; this assumption fails when the certificate is too strict, in which case rejection reflects over-conservatism rather than true invalidity. The calibration procedure (fitting `tau` on 512 labeled outputs) and the reported certificate precision/recall directly measure how well `cert.passes(tau)` tracks ground-truth validity, making the equivalence assumption testable.

## Contribution

["A per-evidence admissibility contract (CERT) that couples each perception tool call to a downstream reasoning invariant via a certificate computed from the tool's raw output and the known camera/scene model, with no learned confidence head.", 'A counterfactual re-query mechanism that establishes substitutability of failed evidence using the same certificate, converting silent oracle failures into explicit, auditable rejections that block spatial claims.', 'A design principle: structural certificates (metric consistency, integrability, solid-angle consistency) are monotone in evidence validity and therefore provide a gate that is auditable and tool-agnostic, unlike learned confidence heads.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | MMSI-Bench + ReVSI spatial subsets | Stress metric depth, normals, and occlusion reasoning where certificates bite |
| Primary metric | Certified-evidence precision vs. spatial-claim accuracy | Directly tests admissibility→correctness coupling |
| Baseline | S-Agent 2.0 (tool-using agent) | Tests whether certification beats raw tool access |
| Baseline | Geo3R hallucination mitigation | Tests whether structural certificates beat learned correction |
| Baseline | APC mental-imagery prompting | Tests whether camera-model invariants beat simulated perspective |
| Baseline | Learned confidence head (2-layer MLP, hidden=256, GeLU, trained on 512 labeled tool outputs to predict validity) | Tests whether the certificate provides gains beyond a learned gate, especially under distribution shift |
| Ablation-of-ours | CERT with certificate gate disabled (accept all tool outputs) | Isolates the certificate as the causal ingredient |
| Ablation-of-ours | CERT with fixed geometric thresholds (no calibration) | Tests whether calibration is necessary for the certificate to track validity |
| Ablation-of-ours | CERT with global anchor check removed | Tests whether cross-tool consistency is needed to catch self-consistent-but-biased outputs |

### Why this setup validates the claim

The central claim is that a certificate computed from a tool's raw output plus the known camera/scene model is both necessary and sufficient to make a tool output admissible for spatial reasoning, and that rejected outputs are the right ones to block. S-Agent 2.0 tests the first sub-claim: it has the same perception tools but no admissibility gate, so any gain CERT shows over it must come from certification, not from tool availability. Geo3R tests whether structural invariants outperform a learned hallucination-correction mechanism on exactly the perspective/orientation/viewpoint tasks where its method was designed. APC tests whether a camera-model invariant is a stronger substitute for mental imagery than prompted simulation. The learned confidence head baseline tests whether the certificate's structural coupling provides gains beyond a learned gate, particularly under distribution shift (e.g., reflective surfaces, low-texture regions) where a learned head may overfit to training tool outputs. The ablation—CERT with the gate disabled—is the sharpest probe: it holds tool access, counterfactual re-query plumbing, and prompt fixed while removing only the certificate, so if the certificate is the causal ingredient, the gap between full CERT and this ablation should be large and concentrated on outputs the certificate actually rejects. The fixed-threshold ablation tests whether calibration is necessary for the certificate to track validity, and the anchor-removed ablation tests whether cross-tool consistency is needed to catch self-consistent-but-biased outputs. Certified-evidence precision (the fraction of admitted evidence whose downstream spatial claim is correct) is the right primary metric because it is the quantity the certificate is supposed to control; raw accuracy can rise from better tools alone, but only certification should tighten the admitted-evidence→claim relationship. We also report certificate precision/recall against ground-truth validity on a held-out test set of 512 tool outputs, directly measuring the equivalence assumption.

### Expected outcome and causal chain

**vs. S-Agent 2.0** — On a case where a depth probe returns a plausible-looking but metrically inconsistent patch (e.g., a reflective or low-texture surface producing a smooth but wrong depth gradient), the baseline consumes it directly and asserts a wrong relative distance because it has no invariant tying depth to the camera model. Our method reprojects the patch, detects the metric inconsistency, and either rejects or re-queries with a shifted patch, so we expect a noticeable accuracy gap on reflective/low-texture subsets but rough parity on well-textured subsets where the baseline's tools are already clean.

**vs. Geo3R** — On a case where a viewpoint change makes an object's orientation ambiguous and the learned corrector over-smooths the hallucination into a confident but wrong orientation, Geo3R produces a fluent wrong answer because its correction is a learned prior, not a scene-model constraint. Our method instead checks solid-angle consistency of the segmentation mask under the known camera, rejects the inadmissible mask, and blocks the claim or re-queries, so we expect a gap concentrated on the orientation and viewpoint-change tasks where the learned prior and the geometric invariant disagree, with parity on perspective tasks where both succeed.

**vs. APC** — On a case requiring an alternative-viewpoint visibility judgment, APC simulates the viewpoint in language and can fabricate occluders, producing a wrong visibility call because its simulation is not anchored to the actual camera model. Our method certifies the depth and normal evidence against the true camera model before the claim is allowed, so on visibility tasks with a known camera we expect a clear gap, while on tasks without a reliable camera model (where our certificate is only as good as the model) we expect convergence with APC.

**vs. Learned confidence head** — On in-distribution tool outputs, the learned head may match the certificate because both can learn to reject obvious failures. However, under distribution shift (e.g., reflective surfaces not seen in training), the learned head is expected to overfit to spurious features and admit biased outputs, while the certificate's structural checks (including the global anchor) remain sensitive to metric inconsistency. We expect the certificate to outperform the learned head on shifted subsets, with parity on in-distribution subsets.

### What would falsify this idea

If the gain over S-Agent 2.0 is uniform across subsets rather than concentrated on the subsets where the certificate actually rejects evidence, the certificate is not the causal ingredient and the claim fails. Likewise, if the gate-disabled ablation matches full CERT on admitted-evidence precision, certification is not controlling admissibility as claimed. Additionally, if certificate precision/recall against ground-truth validity is low (e.g., precision < 0.8), the equivalence assumption is violated and the certificate does not measure validity as claimed.

## References

1. S-Agent 2.0: Spatial Reasoning with a General Coding Agent
2. Geo3R: Mitigating Spatial Reasoning Hallucination in Multimodal Large Language Models
3. Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation
