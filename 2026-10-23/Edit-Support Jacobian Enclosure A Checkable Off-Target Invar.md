# Edit-Support Jacobian Enclosure: A Checkable Off-Target Invariance Certificate for Parametric Memory Edits

## Motivation

Localized parametric interventions—null-space projected edits (AlphaEdit) and circuit-restricted edits (MCircKE, CaKE)—are validated only on the distribution they were fit to, while the field assumes that restricting the locus of the edit (a preserved-knowledge null space, a causal circuit, a fused memory slot) confines its functional effect on the complement set. This localization-as-preservation fallacy recurs across both paper-trees: AlphaEdit's guarantee holds only for the preserved-knowledge matrix it computed, and MCircKE/CaKE verify circuit edits only on the benchmark they were mapped on, with no bound on off-circuit or out-of-distribution behavior. The missing object is an invariant, checkable at edit time from the edit operator and local model structure alone, that bounds functional change on inputs the edit was never fit to.

## Key Insight

For a smooth model, the first-order functional change induced by an edit is exactly the Jacobian-vector product J·Δθ, so an off-target invariance certificate reduces to bounding the edit's directional derivative on the complement—a quantity that can be enclosed by a block-diagonal Gershgorin/interval analysis of the local Jacobian over the edit's support, without ever enumerating complement inputs.

## Method

**What it is.** ESJE (Edit-Support Jacobian Enclosure) takes as input a parameter edit operator E: θ → θ + Δθ with a declared support set S (the parameters E touches, e.g., an FFN block, a circuit, or a fused memory slot) and a local model structure (the Jacobian blocks ∂h/∂θ_S evaluated at a small anchor set A of forward passes). It outputs a scalar certificate radius r(E) and a binary certificate flag: certified if the first-order functional change on any complement input is bounded by r(E), where r(E) is computed from Δθ and the Jacobian enclosure alone, with no forward pass on complement data.

**Load-bearing assumption (stated explicitly).** The certificate rests on the assumption that the separability ratio κ—computed as the ratio of cross-Jacobian norm ||∂f/∂θ_S̄||_F to within-support norm ||∂f/∂θ_S||_F at the anchor set A—is a valid proxy for how much a finite edit Δθ_S leaks onto complement inputs, so that κ ≤ κ_max certifies off-target invariance. This assumption is load-bearing because the entire certificate is withheld or granted based on κ; if κ does not track functional leakage, the certificate is decorative. The assumption fails when the edit's effect on downstream activations is mediated by layer Jacobian products rather than by raw parameter-space cross-blocks (as in deep networks where localized MLP edits produce large activation shifts at downstream layers and attention heads), and when the anchor set A is used to estimate a worst-case bound over the entire complement input distribution—a distribution-shift extrapolation that no finite anchor sample can certify. In such cases κ reflects residual parameter-space correlation rather than true functional separability, and the certificate is correctly withheld (or, if κ is small but leakage is large, the certificate is unsound—a failure mode we make explicit in block D).

**How it works.**
```pseudocode
INPUT: model f_θ, edit E with support S and delta Δθ_S, anchor set A (k=16 inputs, forward-only), tolerance ε, κ_max=0.1
OUTPUT: (certified: bool, radius r)

1. ANCHOR JACOBIAN BLOCKS
   for each anchor a in A:
       compute J_S(a) = ∂f_θ(a)/∂θ_S        # shape [d_out x |S|], via backward-mode AD on the support only
   # block partition: partition S into b=8 contiguous blocks (layers/heads/FFN rows)
   for each block j in 1..b:
       G_j = J_{S_j}(a)ᵀ J_{S_j}(a)           # local Gram block, |S_j| x |S_j|
       λ_max_j(a) = power_iteration(G_j, iters=20)
   λ_max_j = max_a λ_max_j(a)                 # worst-case over anchors

2. EDIT DIRECTIONAL ENERGY
   for each block j:
       e_j = ||Δθ_{S_j}||_2
   E_dir = sqrt( Σ_j λ_max_j · e_j² )        # first-order bound on ||J·Δθ||_2

3. SUPPORT-COMPLEMENT SEPARABILITY CHECK (the certificate's load-bearing step)
   # verify S is a functional separator: for each anchor, the cross-Jacobian
   # between S and its complement S̄ must be small relative to within-S blocks
   for each anchor a:
       C(a) = || ∂f_θ(a)/∂θ_{S̄} ||_F / (|| ∂f_θ(a)/∂θ_S ||_F + 1e-8)
   κ = max_a C(a)                             # separability ratio
   # if κ > κ_max (default 0.1), S is NOT a separator: abort as uncertifiable

4. CERTIFICATE
   r = E_dir · (1 + κ)                        # first-order bound inflated by leakage
   certified = (r ≤ ε) and (κ ≤ κ_max)
   return (certified, r)
```

**Why this design.** We chose a first-order Jacobian enclosure over a full second-order or sampling-based bound because the edit operator and local Jacobian are the only quantities available at edit time without complement data, accepting the cost that the certificate is exact only for infinitesimal edits and must be inflated for finite Δθ (we use the (1+κ) inflation to absorb first-order truncation). We chose block-diagonal Gershgorin/power-iteration over a full SVD of the support Jacobian because |S| can be millions of parameters and full SVD is intractable at edit time, accepting that block-diagonal ignores cross-block curvature and therefore yields a looser (conservative) radius. We chose the separability ratio κ as an explicit abort condition rather than folding it silently into r because a reviewer's first objection to any localization claim is that the support is not actually a separator—making κ a hard gate converts the localization heuristic into a falsifiable precondition, accepting that some edits (e.g., AlphaEdit projections that are not circuit-aligned) will be rejected as uncertifiable. Unlike AlphaEdit, whose null-space projection guarantees invariance only for the preserved-knowledge matrix it was fit on, ESJE certifies invariance on the complement from the edit geometry itself; unlike MCircKE/CaKE, which verify circuit edits only on the benchmark they were mapped on, ESJE yields a checkable radius before any complement evaluation. A domain expert would not describe ESJE as a variant of AlphaEdit because AlphaEdit's guarantee is a linear-algebraic projection onto a fixed null space, whereas ESJE's guarantee is a Jacobian-enclosure bound that subsumes null-space projection (when Δθ lies in the null space, E_dir = 0 and r = 0) and circuit restriction (when S is a verified separator, κ ≈ 0) as special cases.

**Why it measures what we claim.** The directional energy E_dir measures the first-order functional change on the complement because the chain rule makes ∂f/∂θ · Δθ the exact directional derivative for infinitesimal edits, and the block-diagonal Gram bound encloses this derivative without enumerating inputs; this assumption fails when Δθ is large enough that second-order terms dominate, in which case E_dir reflects a lower bound that the (1+κ) inflation only partially corrects. The separability ratio κ measures whether the support S is a functional separator because the cross-Jacobian norm ||∂f/∂θ_S̄||_F quantifies how much complement parameters influence the same outputs, and κ ≤ κ_max is the named assumption that S and S̄ act on disjoint output directions; this assumption fails when the model has strong cross-layer coupling (e.g., a circuit that shares attention heads with off-circuit computation), in which case κ reflects residual leakage rather than true separability and the certificate is correctly withheld. The anchor set A measures local Jacobian structure because we assume the Jacobian is approximately constant over the input neighborhood spanned by A; this assumption fails for highly input-dependent circuits (e.g., retrieval-conditioned memory fusion where the gating varies sharply with context), in which case λ_max reflects the anchor distribution rather than the worst-case complement, and the radius is optimistic. **Failure mode F (explicit):** κ measures functional separability of S from S̄, and the assumption A that ||∂f/∂θ_S̄||_F small relative to ||∂f/∂θ_S||_F implies complement outputs are unaffected by edits to S fails when large-magnitude but low-rank cross-Jacobian directions leak despite small Frobenius norm—i.e., a single dominant cross-Jacobian singular direction can carry substantial leakage while the Frobenius norm remains small; in that case κ underestimates leakage and the certificate is unsound. We do not claim to detect this failure mode with the current κ; it is a stated limitation of the certificate.

## Contribution

['ESJE, a checkable off-target invariance certificate for localized parametric edits that computes a radius from the edit operator and block-diagonal Jacobian enclosure alone, with a hard separability precondition (κ) that falsifies the localization-as-preservation assumption when the support is not a functional separator.', 'A unification result: null-space projection (AlphaEdit) and circuit restriction (MCircKE/CaKE) are recovered as the two limiting cases of the same enclosure (E_dir = 0 and κ ≈ 0 respectively), converting two separate heuristics into instances of one invariant.', 'A diagnostic protocol that, for any edit operator, reports whether it is certifiable and at what radius, enabling edit-time rejection of edits whose off-target effect cannot be bounded—demonstrated on knowledge-editing and chunk-memory-fusion edits.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | MQuAKE multi-hop editing suite | Tests generalization beyond the edited fact |
| Primary metric | Certified-edit rate at fixed ε, with multi-hop accuracy on certified edits | Separates certificate validity from raw edit strength |
| Baseline | MEMIT | Layer-localized editing without circuit support |
| Baseline | CaKE | Circuit-aware editing verified only on mapped benchmark |
| Baseline | AlphaEdit | Null-space projection invariant on preserved matrix |
| Ablation-of-ours | ESJE with κ-gate disabled (κ folded into r) | Tests whether hard separability gate carries the claim |
| Ablation-of-ours | ESJE with anchor set size k=4 and k=64 | Tests sensitivity to the anchor-sample estimate of κ and λ_max |

### Why this setup validates the claim

The central claim is that ESJE yields a checkable radius before any complement evaluation, and that this radius is *predictive*: edits it certifies should preserve complement behavior, and edits it rejects should be the ones that leak. MQuAKE is the right substrate because multi-hop questions expose complement leakage directly—an edit that corrupts an off-circuit fact surfaces as a wrong intermediate hop, not as a wrong edited fact. The primary metric deliberately pairs certified-edit rate with downstream multi-hop accuracy on the certified subset: a method that certifies everything trivially would show high certified rate but degraded accuracy on certified edits, so the pair is jointly falsifiable. MEMIT tests the sub-claim that layer-localized support is not a separator; CaKE tests whether benchmark-mapped circuits are looser than Jacobian-verified separators; AlphaEdit tests whether null-space invariance is subsumed rather than matched. The κ-disabled ablation isolates the load-bearing step: if removing the hard gate does not change the certified/uncertified accuracy gap, the separability precondition is decorative and the idea fails. The anchor-size ablation (k=4, k=64) tests the stated assumption that the Jacobian is approximately constant over the anchor neighborhood: if certified rate and downstream accuracy are stable across k, the anchor-sample estimate is robust; if they vary sharply, the certificate is an anchor-sample heuristic rather than a worst-case bound, as conceded in block D.

### Expected outcome and causal chain

**vs. MEMIT** — On a multi-hop case where the edited fact requires an off-circuit attention head to compose with the edited layer, MEMIT produces correct edited-fact recall but wrong final-hop reasoning because its layer-local support is not a functional separator and the edit perturbs shared output directions. Our method instead computes κ for that support, finds κ above κ_max, and withholds the certificate on exactly those instances because the cross-Jacobian reveals the leakage. We expect a noticeable certified-accuracy gap concentrated on multi-hop subsets requiring off-circuit composition, with near-parity on single-hop edits.

**vs. CaKE** — On a case where the circuit mapped from the benchmark does not match the circuit active at inference for a rephrased query, CaKE produces confident but incorrect multi-hop answers because its circuit verification is benchmark-bound and silently assumes the mapped support is the operative one. Our method instead re-evaluates the anchor Jacobian at inference-time anchors and rejects the edit when the separability ratio exceeds κ_max, because κ is computed from the model geometry rather than the benchmark mapping. We expect CaKE to show a degradation concentrated on rephrased/paraphrased multi-hop items, while ESJE's certified subset shows parity there.

**vs. AlphaEdit** — On a case where the edit direction has a component outside the fitted null space, AlphaEdit produces preserved-matrix invariance but complement drift because its guarantee is linear-algebraic over a fixed preserved matrix, not over the full complement. Our method instead bounds the directional energy E_dir against the Jacobian enclosure and inflates by (1+κ), so when the edit leaks outside the null space the radius grows and the certificate is withheld. We expect ESJE to certify a strict subset of AlphaEdit-accepted edits, and within that subset to show equal or slightly better complement accuracy, with the gap appearing only on edits AlphaEdit accepts but ESJE rejects.

### What would falsify this idea

If the certified/uncertified accuracy gap is uniform across multi-hop subsets rather than concentrated where κ exceeds κ_max, the separability gate is not tracking the actual leakage mechanism and the central claim is wrong. Equivalently, if the κ-disabled ablation matches the full method on both certified rate and downstream accuracy, the hard gate is decorative and the certificate does not measure what it claims.

## References

1. Deep Fusion Memory: A Chunk-Level Parametric Memory for Large Language Models
2. Addressing the Reasoning Gap: Mechanistic Circuit-Based Knowledge Editing in Large Language Models
3. CaKE: Circuit-aware Editing Enables Generalizable Knowledge Learners
4. AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models
5. Uncovering Overfitting in Large Language Model Editing
6. PMET: Precise Model Editing in a Transformer
