# Domain-Identifiable Reverse Encoding for Cross-Domain Generative Representation Learning

## Motivation

Visual representation learning methods increasingly rely on pretrained encoders whose semantic axes are assumed to be stable across domains, yet no existing work enforces this stability explicitly. RevFlow (Generation as Reverse Encoding) demonstrates that frozen-encoder intermediate representations can serve as generative targets, but reports results only on ImageNet 256x256 and provides no mechanism to ensure that the semantic axes it generates remain identifiable when the training distribution shifts. This is a meta-level failure: SPGym shows RL representation methods degrade as image-pool diversity grows, CIDiffuser is validated on only two editing benchmarks, and RevFlow is single-domain — each branch implicitly assumes its training distribution spans the needed semantic variation, so no explicit domain-shift identifiability constraint is imposed on the representation space.

## Key Insight

A semantic axis is identifiable under domain shift exactly when its generative reconstruction from the frozen encoder's deep representation is invariant to domain-specific nuisance factors, and this invariance can be enforced by a domain-adversarial regularizer applied to the stage-transition inputs of HyC-DiT rather than to the encoder itself.

## Method

**What it is.** DIRA (Domain-Identifiable Reverse Adapter) augments RevFlow's HyC-DiT stage transitions with a domain-adversarial identifiability regularizer. Inputs: a frozen pretrained encoder E (RevFlow's ImageNet-pretrained encoder), a multi-domain image pool D = {D_1,...,D_K} (K=3 domains: ImageNet, sketches, stylized), and RevFlow's stage-wise generative targets. Outputs: a HyC-DiT whose stage transitions map deep semantic representations to shallow appearance-rich representations in a way that is invariant to domain identity, plus a domain classifier used only during training.

**Load-bearing assumption (stated explicitly).** DIRA assumes that domain identity is linearly decodable from the frozen encoder's deep semantic representation if and only if the semantic axes carry domain-specific nuisance, so that a gradient-reversal classifier applied at the deep-to-mid transition can strip domain nuisance while preserving cross-domain semantic content. This assumption fails when domains differ only in label marginal (not in nuisance factors), in which case the classifier reflects label-distribution shift rather than semantic-axis contamination; it also fails when the frozen encoder itself was pretrained on a single domain, in which case the mid-level manifold reflects that domain's biases and the reconstruction term preserves them. To detect the label-marginal confound, we include a control that reports classifier accuracy on a label-shuffled domain split (domain labels randomly permuted across images within each domain) — if the classifier achieves above-chance accuracy on the shuffled split, the domain signal is driven by label marginals rather than nuisance.

**How it works.**

```pseudocode
# Phase 1: freeze encoder, extract stage targets
for each image x in D:
    z_deep, z_mid, z_shallow = E(x)          # frozen encoder hierarchy
    d = domain_label(x)                      # domain id 1..K (K=3)

# Phase 2: train HyC-DiT with domain-adversarial regularizer
for each stage transition t in {deep->mid, mid->shallow, shallow->pixel}:
    h_t = HyC_DiT_stage_t(z_target_t, condition=c_t)   # shared DiT backbone, stage-specific condition injection
    # reconstruction loss (native matching in representation space)
    L_rec = || h_t - z_target_{t-1} ||^2
    # domain-adversarial head (applied ONLY to deep->mid transition input)
    if t == deep->mid:
        d_hat = DomainClassifier(h_t, lambda_grl=0.1)   # gradient reversal layer
        L_dom = CE(d_hat, d)
    else:
        L_dom = 0
    L_total = L_rec + alpha * L_dom    # alpha = 0.5
    update HyC_DiT_stage_t via L_total

# Phase 3: few-step generation with native matching
sample z_deep ~ N(0,I); run stage transitions; decode pixels
```

**Why this design.** We chose to apply the domain-adversarial head only to the deep-to-mid transition rather than to all stages because the deep semantic representation is the level at which RevFlow's generative targets are most abstract and least tied to appearance, so domain nuisance factors are most likely to leak into the semantic axes at exactly that boundary; applying the regularizer to shallower stages would force invariance over appearance details that are legitimately domain-specific, accepting the cost that we do not explicitly regularize mid-to-shallow transitions. We chose a gradient-reversal layer over a separate domain classifier trained post-hoc because the reversal couples the domain classifier's loss directly into the stage-transition parameters, making the invariance constraint active during representation learning rather than a diagnostic; the cost is that gradient reversal introduces training instability at high alpha, which we mitigate by fixing alpha=0.5 and using a warm-up of 2 epochs before enabling the reversal. We chose native matching in representation space (inherited from RevFlow) over pixel-space reconstruction for the stage losses because it avoids the high-frequency interference that RevFlow's abstract explicitly identifies as a cost of image reconstruction, accepting that the semantic axes are then defined by the frozen encoder's geometry rather than by pixel fidelity. A domain expert would not describe this as a variant of RevFlow because RevFlow has no domain-shift mechanism at all — its stage transitions are trained on a single domain and its objective contains no invariance term; DIRA changes the training signal at the deep-to-mid boundary specifically.

**Why it measures what we claim.** The domain classifier's accuracy on held-out domains measures domain-identifiability of the deep semantic representation because the gradient-reversal layer makes the stage-transition parameters minimize the classifier's ability to recover domain identity from the generated mid-level representation, under the assumption that domain identity is linearly decodable from the deep representation if and only if the semantic axes carry domain-specific nuisance; this assumption fails when domains differ only in label marginal (not in nuisance factors), in which case the classifier reflects label-distribution shift rather than semantic-axis contamination. The reconstruction loss ||h_t - z_target||^2 measures semantic preservation because native matching in representation space is equivalent to requiring that the generated mid-level representation lie on the same manifold as the frozen encoder's mid-level output, under the assumption that the frozen encoder's mid-level manifold encodes the semantic content shared across domains; this assumption fails when the frozen encoder itself was pretrained on a single domain, in which case the manifold reflects that domain's biases and the reconstruction term preserves them. The stage-specific condition injection c_t measures stage identity because it is a one-hot vector distinguishing deep-to-mid from mid-to-shallow transitions, under the assumption that the shared DiT backbone needs only this discrete signal to route computation; this assumption fails when stage transitions require different receptive fields, in which case the shared backbone underfits the shallower transition.

## Contribution

['A domain-adversarial identifiability regularizer applied to the deep-to-mid stage transition of HyC-DiT, which enforces that the semantic axes used as generative targets in reverse encoding remain invariant to domain identity without modifying the frozen encoder.', 'A design principle for reverse-encoding generative models: invariance constraints should be applied at the most abstract stage transition (deep-to-mid) rather than uniformly across stages, because shallower transitions legitimately carry domain-specific appearance detail.', 'An empirical protocol for evaluating cross-domain identifiability of generative representation targets using held-out domain classification accuracy on generated mid-level representations.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Multi-domain image pool: ImageNet (D1), sketches (D2), stylized (D3) | Forces domain nuisance into deep semantics; three domains with one held-out (sketches) |
| Primary metric | Held-out-domain classifier accuracy vs. FID | Jointly tests invariance and semantic preservation |
| Baseline | RevFlow (no domain mechanism) | Tests whether invariance term is needed at all |
| Baseline | RevFlow + post-hoc domain classifier probe | Tests whether active reversal beats diagnosis |
| Baseline | RevFlow + domain-adversarial on all stages | Tests whether deep-only placement is correct |
| Ablation-of-ours | DIRA with alpha=0 (no reversal) | Isolates the invariance term's contribution |
| Control | Label-shuffled domain split classifier accuracy | Detects label-marginal confound (assumption failure) |

### Why this setup validates the claim

The central claim is that domain nuisance leaks into RevFlow's deep-to-mid boundary and that an actively coupled invariance constraint there improves cross-domain generation without sacrificing semantic fidelity. The multi-domain pool is essential because single-domain ImageNet cannot expose the failure: if all domains share nuisance statistics, the invariance term has nothing to remove and the test is vacuous. The primary metric is deliberately a pair: held-out-domain classifier accuracy detects whether invariance was actually achieved (the mechanism's direct effect), while FID on held-out domains detects whether invariance was achieved by destroying semantics (the mechanism's predicted cost). Reading them together makes the claim falsifiable in both directions. RevFlow tests the necessity of any domain mechanism; the post-hoc probe baseline tests whether the gradient-reversal coupling matters versus mere diagnosis; the all-stages adversarial baseline tests the placement argument, since if regularizing every stage matched or beat deep-only, our stated design rationale would be wrong. The alpha=0 ablation isolates the reversal term from everything else in DIRA, so any gain can be attributed to the invariance signal rather than to the multi-domain training pool itself. The label-shuffled control directly tests the load-bearing assumption: if the classifier achieves above-chance accuracy on shuffled domain labels, the domain signal is driven by label marginals rather than nuisance, and the invariance term is removing label-relevant directions.

### Expected outcome and causal chain

**vs. RevFlow (no domain mechanism)** — On a case where a held-out domain (e.g., sketch) shares semantic content with natural images but differs sharply in appearance statistics, RevFlow's stage transitions produce mid-level representations that still encode domain identity, because its objective contains no term that penalizes domain recovery and its targets inherit the frozen encoder's domain-biased geometry. Our method instead drives the deep-to-mid transition to strip domain-identifiable axes via the reversed classifier gradient, so we expect a clear drop in held-out-domain classifier accuracy alongside a modest FID improvement concentrated on held-out domains, with near-parity on in-domain FID. The label-shuffled control should show near-chance accuracy for both methods, confirming that the domain signal is nuisance-driven rather than label-marginal.

**vs. RevFlow + post-hoc domain classifier probe** — On the same sketch case, the post-hoc probe accurately reports that domain identity is decodable from deep representations but cannot change them, because the classifier is trained after the transition parameters are frozen and its loss never reaches them. Our method instead couples the classifier loss into the transition parameters during training, so the representation is actively pushed toward non-identifiability rather than merely measured. We expect the post-hoc baseline to show high probe accuracy with RevFlow-level FID, while DIRA shows lower probe accuracy and better held-out FID — a divergence between diagnosis and intervention.

**vs. RevFlow + domain-adversarial on all stages** — On a case where appearance details are legitimately domain-specific (e.g., stroke texture in sketches), the all-stages baseline forces invariance at the shallow boundary too, erasing appearance information the decoder needs and degrading reconstruction fidelity. Our method instead regularizes only the deep-to-mid transition, leaving shallow appearance structure intact, so we expect the all-stages variant to show lower held-out classifier accuracy but worse FID, while DIRA achieves comparable invariance with better FID — a pattern where the two methods separate on fidelity rather than on invariance.

### What would falsify this idea

If DIRA's gain over RevFlow is uniform across in-domain and held-out-domain FID rather than concentrated on held-out domains, the claimed domain-nuisance mechanism is not what produces the improvement. Likewise, if the all-stages adversarial baseline matches DIRA on both classifier accuracy and FID, the deep-only placement argument is wrong and the design rationale collapses. Additionally, if the label-shuffled control shows above-chance classifier accuracy, the load-bearing assumption fails and the invariance term is removing label-relevant directions rather than nuisance.

## References

1. Generation as Reverse Encoding
