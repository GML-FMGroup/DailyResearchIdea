# Certified Dynamics Subspace Probing for Faithful Counterfactual Video-Language Supervision

## Motivation

Counterfactual video-language supervision, as introduced by V-JEPA-LM (arxiv:2611.17831), assumes that pixel transformations isolate dynamics-relevant change, but never certifies that the frozen encoder's shared dynamics subspace is invariant to nuisance photometric/geometric factors; a linear probe can therefore track transformation-induced nuisances rather than true dynamics. Because no mechanism ties the language model's read-out to the same subspace the counterfactual supervision rewards, the intervention is not causally faithful by construction, and the persistent gap between probe accuracy (87%) and VLM chance performance may persist or be closed for the wrong reason.

## Key Insight

A transformation family is admissible into counterfactual supervision if and only if the frozen encoder's dynamics probe subspace is invariant under that family, and this same certified subspace can serve as a fixed regression target that anchors the language model's internal read-out, making the intervention faithful by a shared-subspace identifiability condition rather than by post-hoc correlation.

## Method

**What it is.** We introduce **CASP** (Certified-Admissible Subspace Probing), a training framework that (i) certifies each pixel-transformation family by measuring the invariance of a frozen V-JEPA dynamics subspace under that family, and (ii) trains the language model with an auxiliary probe-consistency loss that regresses the LM's internal dynamics read-out onto the certified probe's prediction.

**Load-bearing assumption (stated explicitly).** CASP assumes that if the frozen encoder's probe subspace is invariant under a transformation family f, then f changes only dynamics-relevant attributes and not nuisance photometric/geometric factors. This is a subspace-coverage assumption, not an identifiability guarantee: a transformation whose effect lies largely orthogonal to the top-k probe directions can yield a high invariance score even if it also perturbs nuisances. To make this assumption testable rather than assumed, we add a nuisance-controlled certificate and a held-out nuisance-probe validation (Phase 1b) that directly measures whether high C(f) can be obtained by subspace coverage alone.

**How it works.**

```pseudocode
# Phase 1: Subspace extraction (frozen V-JEPA encoder E, linear probe p)
Given public video pairs (v, v') where v' = T_f(v) for transformation family f
For each f in {speed, direction, order, state}:
    Collect frozen features {E(v)} and {E(v')}
    Fit linear probe p_f on speed/direction/order/state labels from E(v)
    Compute subspace S_f = row-space(p_f) projected onto top-k (k=32) principal directions
    Compute raw invariance score C_raw(f) = 1 - ||P_{S_f} (E(v) - E(v'))|| / ||E(v) - E(v')||

# Phase 1b: Nuisance-controlled certificate (new, makes load-bearing assumption testable)
    Build nuisance basis N_f: apply a fixed set of photometric/geometric augmentations A = {brightness ±0.2, contrast ±0.2, hue ±0.1, random crop 0.8–1.0, horizontal flip} to each v, collect features {E(a(v))}, and take the top-m (m=16) principal directions of the resulting feature differences as N_f.
    Compute residual difference d_f = (I - P_{N_f})(E(v) - E(v'))
    Compute nuisance-controlled certificate C(f) = 1 - ||P_{S_f} d_f|| / ||d_f||
    Admit f into counterfactual pipeline iff C(f) >= tau (tau=0.85)

# Phase 1c: Certificate validation (held-out nuisance probe)
    Fit a held-out linear probe q_f on S_f to predict the transformation identity (which f was applied) from E(v) vs E(v').
    Report nuisance-leakage score L(f) = accuracy(q_f) - chance. A high C(f) with high L(f) indicates subspace coverage, not nuisance isolation; we require L(f) <= 0.10 for admission alongside C(f) >= tau.

# Phase 2: Counterfactual supervision with certified families
For each admitted f and public video v:
    Generate answer-balanced variants v_f^+ , v_f^- via pixel transformation T_f
    Construct QA pairs whose answer is set by f and differs only in the change
    LM input: 32 visual tokens per frame from E(v), text question

# Phase 3: Probe-consistency regularizer
For each training batch:
    LM produces internal dynamics read-out h(v) at a designated [DYN] token
    Frozen probe p_f (for the family f used to generate the pair) produces target y = p_f(E(v))
    Auxiliary loss L_probe = || MLP(h(v)) - y ||^2, where MLP is a 2-layer MLP with hidden size 256 and GeLU activation
    Total loss L = L_QA + lambda * L_probe   (lambda=0.1)
```

**Why this design.** We chose to certify transformation families via a subspace-invariance score rather than admit all pixel transformations as V-JEPA-LM does, because the structural assumption that any pixel transformation isolates dynamics is false when the frozen encoder's probe subspace is not invariant to that family; accepting the cost that some families may be rejected and training data shrinks. We chose a fixed linear probe on frozen features as the regression target for the LM's internal read-out rather than a learned consistency head, because a fixed probe provides an external anchor that cannot co-adapt with the LM; accepting the cost that the LM is constrained to mimic a linear function of frozen features, which may limit expressivity. We chose a single designated [DYN] token for the read-out rather than pooling all visual tokens, because it forces the LM to concentrate dynamics information in a measurable locus; accepting the cost that the LM may underutilize other tokens for dynamics. A domain expert would not describe this as a variant of V-JEPA-LM because V-JEPA-LM trains only on QA loss with unverified transformations, whereas CASP introduces a certification gate and an auxiliary regression that ties the LM to a certified subspace — a coupling between the supervision signal and the encoder's identifiability that V-JEPA-LM does not establish.

**Why this measures what we claim.** The nuisance-controlled invariance certificate C(f) measures the admissibility of a transformation family for counterfactual supervision because it assumes that if the frozen encoder's probe subspace is invariant under f after projecting out measured nuisance directions, then f changes only dynamics-relevant attributes and not nuisance factors; this assumption fails when the probe subspace is low-rank and the transformation's effect lies outside it, in which case C(f) reflects probe subspace coverage rather than true nuisance isolation. The probe-consistency loss L_probe measures the faithfulness of the LM's dynamics read-out to the certified subspace because it assumes that regressing the LM's internal representation onto the frozen probe's prediction forces the LM to encode the same dynamics information the probe reads; this assumption fails when the MLP adapter in L_probe absorbs the regression burden without changing the LM's internal representation, in which case L_probe reflects adapter capacity rather than LM faithfulness.

## Contribution

["A transformation-invariance certificate that scores each pixel-transformation family by the invariance of a frozen encoder's linear probe subspace, providing a concrete admissibility criterion for counterfactual supervision pipelines.", "A probe-consistency regularizer that anchors a language model's internal dynamics read-out to a certified frozen-feature probe, operationalizing the shared-subspace identifiability condition as a training loss.", "An empirical design principle: the set of transformation families admissible for faithful counterfactual video-language supervision is determined by the frozen encoder's subspace geometry, not by the transformation's semantic label."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Counterfactual dynamics QA built from public video (speed/direction/order/state) | Answer set by transformation, isolates dynamics reading |
| Primary metric | Accuracy on answer-balanced counterfactual variants | Balanced answers force reading change, not attribute retrieval |
| Baseline | V-JEPA-LM (unverified counterfactual supervision) | Tests whether certification gate adds value |
| Baseline | Frozen V-JEPA + linear probe only (no LM) | Upper reference for dynamics read-out |
| Baseline | SigLIP-based video LLM | Tests whether text-aligned encoders lack shared dynamics subspace |
| Baseline | Learned consistency head (co-adapting MLP) instead of fixed probe | Tests whether fixed-probe anchor is necessary |
| Baseline | Probe fit on augmented features (nuisance-residualized probe) | Tests whether nuisance control alone explains gains |
| Ablation-of-ours | CASP without probe-consistency loss (certification only) | Isolates contribution of L_probe from certification |
| Ablation-of-ours | CASP with high-capacity MLP adapter (hidden=1024) in L_probe | Tests whether adapter absorbs regression without changing h(v) |
| Diagnostic | Synthetic rendered clips with known ground-truth dynamics and nuisance (speed vs. brightness) | Validates C(f) ranks families by true nuisance leakage |
| Diagnostic | Held-out linear probe on LM's h(v) (bypassing MLP adapter) | Confirms LM representation, not adapter, carries dynamics signal |

### Why this setup validates the claim

The central claim is that CASP's certification gate plus probe-consistency loss makes the language model actually read dynamics rather than retrieve named attributes. The answer-balanced counterfactual dataset is the right test because it removes the shortcut of guessing from static attributes: on each item the two variants differ only in the transformation, so accuracy can only come from reading the change. The V-JEPA-LM baseline tests the sub-claim that unverified transformations admit families whose effect lies outside the probe subspace, producing supervision that does not isolate dynamics; if CASP beats it, the certification gate is doing work. The frozen-probe baseline tests the sub-claim that the LM can be tied to a certified subspace without losing the probe's read-out fidelity; parity with it on admitted families shows the LM is not falling below the external anchor. The SigLIP baseline tests the sub-claim that text-aligned encoders lack a shared dynamics subspace, so CASP's gains should not appear there. The learned-consistency-head baseline tests whether the fixed-probe external anchor is necessary or whether a co-adapting head achieves the same effect, directly addressing the novelty concern. The augmented-features-probe baseline tests whether nuisance control alone (without the certification gate) explains any gains. The no-L_probe ablation isolates whether the auxiliary regression, rather than certification alone, is responsible for any LM-side faithfulness gain. The high-capacity-adapter ablation tests the failure mode where the MLP absorbs the regression without changing h(v). The synthetic diagnostic with known ground-truth dynamics and nuisance validates that C(f) ranks families by true nuisance leakage rather than subspace coverage. The held-out linear probe on h(v) directly tests whether the LM's internal representation, not the adapter, carries the dynamics signal.

### Expected outcome and causal chain

**vs. V-JEPA-LM** — On a case where the transformation is a speed change whose effect falls largely outside the low-rank probe subspace, V-JEPA-LM trains on the pair as if the answer were dynamics-determined, so the LM learns a spurious cue correlated with the transformation and answers at chance on balanced variants because its supervision never isolated the change. CASP instead rejects that family at the certification gate (low C(f) or high L(f)), so its training pairs come only from families whose probe subspace is invariant and whose nuisance leakage is low, and the LM sees supervision where the answer truly tracks the change. We expect a noticeable gap on items from high-invariance families and parity or slight loss on rejected families, with the overall gap concentrated where C(f) is high and L(f) is low.

**vs. Frozen V-JEPA + linear probe only** — On a case requiring compositional or multi-step dynamics reasoning, the frozen probe reads speed or direction from features at high accuracy but cannot answer the question because it has no language interface and no way to combine attributes. CASP's LM receives the same frozen features and is anchored to the probe's prediction through L_probe, so it should match the probe on single-attribute items while exceeding it on compositional items where the LM's text side can integrate. We expect near-parity on single-attribute subsets and a clear advantage on compositional subsets, with the probe as a ceiling on the former.

**vs. SigLIP-based video LLM** — On a case where the answer depends on direction or order, the SigLIP encoder's frame-wise text-aligned features lack a shared dynamics subspace, so the LLM retrieves the attribute named in the question rather than reading the change and answers at chance on balanced variants. CASP's V-JEPA encoder carries dynamics in a shared subspace, and the certification gate admits only families that subspace supports, so the LM's read-out is anchored to a signal that actually encodes the change. We expect a large gap on dynamics subsets for CASP and chance-level performance for SigLIP, confirming the encoder-subspace diagnosis rather than a generic model-size effect.

**vs. Learned consistency head (co-adapting MLP)** — On a case where the LM's internal representation h(v) is not forced to align with the certified subspace, the learned head can co-adapt with the LM and satisfy the consistency loss without changing h(v), so the LM may still rely on spurious cues. CASP's fixed probe provides an external anchor that cannot co-adapt, so the LM must adjust h(v) to reduce L_probe. We expect CASP to outperform the learned-head baseline on items from high-C(f) families, with the gap concentrated where the probe's prediction is most reliable.

**vs. Probe fit on augmented features (nuisance-residualized probe)** — On a case where the transformation family has high nuisance leakage (high L(f)), a probe fit on augmented features may still admit the family because it does not explicitly certify invariance of the dynamics subspace under f. CASP's nuisance-controlled certificate C(f) and held-out nuisance probe L(f) jointly reject such families. We expect CASP to match the augmented-probe baseline on low-leakage families and outperform it on high-leakage families, demonstrating that certification adds value beyond nuisance residualization alone.

**vs. CASP without probe-consistency loss (certification only)** — On a case where the LM's internal read-out h(v) is not anchored to the certified subspace, the LM may learn to answer from other cues even when supervision is clean. CASP's L_probe forces h(v) to regress onto the certified probe's prediction. We expect full CASP to outperform the certification-only ablation on items requiring fine-grained dynamics reading, with the gap concentrated on compositional subsets where the LM must integrate multiple attributes.

**vs. CASP with high-capacity MLP adapter** — On a case where the MLP adapter has enough capacity to absorb the regression target without changing h(v), L_probe can be driven to zero while h(v) remains unchanged. We expect the high-capacity adapter to achieve low L_probe but no improvement on the held-out linear probe on h(v), and no gain over the certification-only ablation on QA accuracy. This tests the failure mode where L_probe reflects adapter capacity rather than LM faithfulness.

**Synthetic diagnostic** — On rendered clips with independently varied speed (dynamics) and brightness (nuisance), we compute C(f) for speed and brightness transformations and compare their ranking to the true nuisance leakage (measured by the effect of brightness on the probe's speed prediction). We expect C(speed) > C(brightness) and L(speed) < L(brightness), matching the ground-truth ordering. If C(f) ranks families by subspace coverage rather than true nuisance leakage, the synthetic diagnostic will reveal the discrepancy.

**Held-out linear probe on h(v)** — On a held-out set, we fit a linear probe on the LM's internal h(v) to predict the dynamics label and compare its accuracy to the probe on E(v). We expect the h(v) probe to achieve accuracy close to the E(v) probe for CASP but not for the certification-only ablation or the high-capacity-adapter variant, confirming that the LM representation, not the adapter, carries the dynamics signal.

### What would falsify this idea

If CASP's gain over V-JEPA-LM is uniform across admitted and rejected transformation families rather than concentrated on high-C(f) families, the certification gate is not doing the work claimed and the central coupling argument is wrong. If the no-L_probe ablation matches full CASP on LM-side faithfulness, the probe-consistency loss is not the mechanism producing the effect. If the held-out linear probe on h(v) shows no improvement over the certification-only ablation, L_probe is not changing the LM's internal representation. If the synthetic diagnostic shows C(f) does not rank families by true nuisance leakage, the certificate is invalid.

## References

1. V-JEPA-LM: Video-Language Models That Read Dynamics Beyond Static, Text-Aligned Representations
