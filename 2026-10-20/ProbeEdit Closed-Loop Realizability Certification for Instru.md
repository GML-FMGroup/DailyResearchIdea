# ProbeEdit: Closed-Loop Realizability Certification for Instruction Decomposition via Editor Response Marginals

## Motivation

Frontier instruction-based editing pipelines such as the CoT-planner/MLLM-region/hint-guided-editor architecture of 'Instruction-Based Image Editing with Planning, Reasoning, and Generation' (and similarly SmartEdit) delegate instruction decomposition to a foundation model trained on synthetic instruction-image pairs, then forward every sub-prompt to the editor without ever checking that the source image supports it or that the editor's own output distribution moves in the claimed direction. The structural failure is that decomposition faithfulness is an open-loop assumption: a sub-prompt like 'make the jacket red' is emitted even when no jacket region exists, and the error propagates silently because nothing in the pipeline grounds a sub-prompt against the editor's conditional response. This is a meta-gap shared across planner-based and MLLM-bridged editing trees — both assume a foundation model's decomposition is trustworthy rather than measuring it against the source image and the editor's capability.

## Key Insight

The editor's own conditional response to a minimal, reversible probe edit is a sufficient statistic for sub-prompt realizability, because a sub-prompt is realizable exactly when the editor's output distribution shifts monotonically along the edit direction on a witness region while remaining invariant off it — a property the planner cannot assert but the editor can be interrogated for.

## Method

```pseudocode
INPUT: source image x0, instruction u, planner P, region network R, editor E (differentiable), probe budget K, margin threshold tau, witness area floor a_min
OUTPUT: certified sub-prompt list S*, per-sub-prompt certificates (m_i, W_i), re-decomposition log

# Phase 1: open-loop decomposition (unchanged baseline)
S = P(u)                                  # sub-prompts s_1..s_n
for each s_i:
    M_i = R(x0, s_i)                       # soft region mask, reused as probe support

# Phase 2: reversible probe edit on the witness support
for each s_i:
    W_i = top-quantile(M_i, q=0.9)         # witness region = high-confidence mask support
    eps = alpha * mean|x0[W_i]|            # alpha=0.05, small reversible perturbation
    x_probe = x0; x_probe[W_i] += eps * d_i   # d_i = unit direction from s_i's CLIP embedding delta
    y_edit  = E(x_probe, s_i)              # editor conditional output
    y_rev   = E(x_probe, s_i_reverse)       # reverse sub-prompt (antonym / negation)
    delta_in  = cos( f(y_edit[W_i]) - f(x0[W_i]),  f(y_rev[W_i]) - f(x0[W_i]) )   # f = CLIP image encoder
    delta_out = cos( f(y_edit[~W_i]) - f(x0[~W_i]), f(y_rev[~W_i]) - f(x0[~W_i]) )
    m_i = delta_in - delta_out              # realizability margin
    cert_i = (m_i, W_i)

# Phase 3: closed-loop gating
S* = []
for each (s_i, cert_i):
    if cert_i.m_i >= tau and area(W_i) >= a_min:
        S*.append(s_i)
    else:
        S_sub = P.re_decompose(s_i, x0, feedback=cert_i)   # planner sees the failing margin + witness
        S*.extend(S_sub)                                   # recurse once; drop if still failing
return E(x0, S*), {cert_i}
```

**Why this design.** We chose a *reversible perturbation probe on the region-mask support* over a learned realizability classifier because the classifier would be trained on the same synthetic distribution that caused the open-loop failure, accepting the cost that probing adds K editor forward passes per sub-prompt (K=2 here: forward and reverse). We chose a *contrastive forward-vs-reverse margin* over a raw output-shift magnitude because magnitude conflates realizability with edit strength, accepting the cost that we must synthesize a reverse sub-prompt for each decomposition (cheap, LLM-generated). We chose to *reuse the region mask as the probe support* rather than a fresh segmentation because the mask already localizes where the editor should act, accepting the cost that a wrong mask yields a wrong witness (mitigated by the a_min area floor and the off-region invariance term). We chose *one re-decomposition recursion* over unbounded replanning to bound latency, accepting the cost that some unrealizable prompts are dropped rather than repaired. A domain expert would not describe this as a variant of InstructPix2Pix or SmartEdit: those methods condition the editor on an instruction and trust it, whereas here the editor is used as its own oracle via a counterfactual probe, and the planner is constrained by a measured certificate rather than by its training prior.

**Why it measures what we claim.** The margin m_i = delta_in - delta_out measures *sub-prompt realizability* because the named assumption is that a realizable sub-prompt induces a monotone, region-localized shift in the editor's output distribution under forward-vs-reverse conditioning; this assumption fails when the editor is insensitive to the reverse prompt (e.g., symmetric edits like 'rotate 90 degrees'), in which case m_i reflects edit-direction asymmetry rather than realizability. The witness region W_i measures *spatial grounding of the sub-prompt* because the assumption is that the region network's high-confidence support coincides with the image content the editor will modify; this fails when the region network hallucinates a mask on absent content, in which case W_i reflects planner-region agreement rather than source-image support. The area floor a_min measures *non-degeneracy of the certificate* because the assumption is that a realizable edit requires a minimum spatial extent; this fails for point edits (single-pixel color changes), in which case a_min rejects a genuinely realizable sub-prompt. The re-decomposition feedback measures *planner correction* because the assumption is that exposing the failing margin and witness to the planner yields a sub-prompt whose certificate passes; this fails when the planner lacks the vocabulary to rephrase, in which case the sub-prompt is dropped rather than repaired.

## Contribution

(1) A closed-loop realizability verifier that certifies each decomposed sub-prompt by probing the editor with a reversible forward/reverse edit on the region-mask witness and returning a scalar margin plus witness region, converting decomposition faithfulness from an open-loop assumption into a measurable per-sub-prompt property. (2) A certificate-constrained re-decomposition loop in which the planner consumes failing margins and witnesses, yielding a design principle that planner output should be gated by editor-response evidence rather than trusted by training prior. (3) A per-sub-prompt certificate format (margin, witness, area) that is editor-agnostic and plugs into existing planner/region/editor pipelines without retraining the editor.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Reason-Edit complex-instruction split | Targets multi-sub-prompt instructions where open-loop fails |
| Primary metric | Sub-prompt realizability pass rate (certified) | Directly counts gated sub-prompts meeting margin+area |
| Baseline | InstructPix2Pix | Tests single-shot editor without planning or gating |
| Baseline | SmartEdit | Tests MLLM planner trusting prior, no probe oracle |
| Baseline | Chain-of-model planner (LLM+seg+edit) | Tests decomposition without certificate gating |
| Ablation-of-ours | Ours w/o probe (magnitude-only gate) | Isolates contrastive margin vs raw output shift |

### Why this setup validates the claim

The central claim is that a reversible forward-vs-reverse probe on the region-mask support yields a *measured* realizability certificate that lets the planner gate and repair sub-prompts the open-loop pipeline would silently accept. Reason-Edit is the right dataset because its complex instructions decompose into multiple sub-prompts where some are unrealizable for the editor, so the failure mode we target is actually present and countable. InstructPix2Pix tests the no-planning sub-claim: if a single editor call were sufficient, our gating would add nothing. SmartEdit and the chain-of-model planner test the no-oracle sub-claim: both decompose and trust their prior, so they should accept unrealizable sub-prompts at the same rate as open-loop. The ablation isolates the contrastive margin from mere edit-strength magnitude — if magnitude alone sufficed, the ablation would match ours, falsifying our mechanism story. The primary metric counts sub-prompts that pass the certificate, which is precisely the quantity our mechanism is supposed to raise; image-fidelity metrics would conflate realizability with edit quality and cannot detect the predicted effect.

### Expected outcome and causal chain

**vs. InstructPix2Pix** — On a case where the instruction bundles a realizable edit with an unrealizable one (e.g. "make the car red and rotate the sky 90 degrees"), the baseline produces a plausible-looking but partially wrong image because it conditions once on the full instruction with no per-sub-prompt check, so the unrealizable clause is absorbed into a generic shift. Our method instead decomposes, probes each sub-prompt forward-vs-reverse on its witness region, and drops or repairs the clause whose margin falls below tau, so we expect a clear gap on multi-clause Reason-Edit subsets and near-parity on single-clause instructions.

**vs. SmartEdit** — On a case where the MLLM planner emits a sub-prompt the editor cannot realize (e.g. "restore the original texture" after a prior edit), the baseline produces an output that silently ignores the clause because the planner trusts its training prior and never queries the editor as an oracle. Our method instead runs the reversible probe on the region support and, when the forward-vs-reverse contrast collapses, feeds the failing margin and witness back for one re-decomposition, so we expect higher certified pass rate concentrated on the complex-instruction subsets where planner-prior and editor-capability disagree.

**vs. Chain-of-model planner (LLM+seg+edit)** — On a case where the region network hallucinates a mask on absent content (e.g. "brighten the window" when no window is visible), the baseline produces an edit on a spurious region because segmentation and editing are chained without any cross-check. Our method instead reuses the mask as probe support and the off-region invariance term delta_out penalizes edits that leak outside the witness, so a hallucinated witness produces a low margin and is gated out; we expect the gap to concentrate on hallucinated-region cases while matching the baseline when masks are correct.

### What would falsify this idea

If our certified pass-rate gain over the chain-of-model planner were uniform across subsets rather than concentrated on multi-clause and hallucinated-region cases — or if the magnitude-only ablation matched the contrastive margin — then the probe is not measuring realizability and the central claim is wrong.

## References

1. Instruction-Based Image Editing with Planning, Reasoning, and Generation
2. SmartEdit: Exploring Complex Instruction-Based Image Editing with Multimodal Large Language Models
3. InstructPix2Pix: Learning to Follow Image Editing Instructions
