# Modality-Native Concept Basis for Causal Lensing in Vision-Language Models

## Motivation

MC-Lens (Probing the "Conscious" Mind) establishes causal sufficiency only for concepts expressible as decoder-vocabulary tokens, because its readout carrier is anchored to the language decoder's output vocabulary; consequently, the intervention set is not closed over non-tokenizable visual features such as spatial relations and textures. This is a structural limitation of any token-anchored lens: the shared semantic space is defined by tokenizable outputs, so visual concepts without a lexical label cannot be read out or intervened upon. The field's trajectory toward mechanistic interpretability of VLMs requires a concept basis that is native to the visual modality and not constrained by the language head's vocabulary.

## Key Insight

A concept basis learned directly from visual encoder activations (via a sparse autoencoder dictionary) provides an intervention algebra that is closed over non-tokenizable visual features, because its atoms are defined by reconstruction of visual representations rather than by alignment to output tokens.

## Method

**What it is.** We introduce VN-Lens (Vision-Native Lens), a modality-native concept lens for vision-language models that replaces MC-Lens's decoder-vocabulary readout carrier with a learned visual-concept dictionary $D \in \mathbb{R}^{K \times d}$ obtained by training a sparse autoencoder (SAE) on the vision encoder's intermediate activations. VN-Lens takes as input a VLM (e.g., LLaVA) and an image-question pair, and outputs (i) per-layer concept activation vectors over the dictionary, (ii) a concept-intervention operator that edits these activations, and (iii) causal effect scores for each dictionary atom on the model's output.

**How it works.**

```pseudocode
# Phase 1: Train modality-native dictionary (offline)
Input: vision encoder activations H_v = {h_l} for l in layers L_v, collected on image corpus
For each layer l in L_v:
    Train SAE_l: encoder E_l (d -> K), decoder D_l (K -> d)
    Loss = ||h_l - D_l(ReLU(E_l(h_l)))||^2 + lambda * ||ReLU(E_l(h_l))||_1
    with K = 4096, lambda = 1e-3, sparsity target 5% active atoms
Output: dictionary D_l and encoder E_l for each layer

# Phase 2: Concept readout and intervention at inference
Input: image x, question q, VLM f, layer l, atom index k
1. Run f on (x, q) up to layer l; extract vision-token activations h_l
2. Concept codes: c = ReLU(E_l(h_l))  # sparse codes over K atoms
3. Intervention operator I_{k,alpha}: modify c_k <- c_k + alpha, then reconstruct h'_l = D_l(c)
   and substitute h'_l for h_l in the forward pass
4. Causal effect: CE(k) = D_KL( f(x,q | h_l) || f(x,q | I_{k,alpha}(h_l)) )
   or task-specific output change (e.g., answer logit difference)
5. Select atoms with CE(k) > tau; these form the causally sufficient concept set
```

**Why this design.** We chose a sparse autoencoder dictionary over slot-attention prototypes because SAEs provide an overcomplete, sparse basis that empirically disentangles features in vision transformers (as shown in vision SAE literature), accepting the cost that dictionary atoms may not be perfectly monosemantic and require post-hoc interpretation. We chose to intervene in the vision encoder's activation space rather than in the language decoder's residual stream because the motivation requires concepts native to the visual modality, accepting the cost that interventions may not propagate cleanly to the final answer and thus require measuring causal effect at the output. We chose a per-layer dictionary rather than a single shared dictionary across layers because vision features are known to be layer-specific (cf. Cross-modal Information Flow, E1/E4, showing lower layers encode general visual features and middle layers encode object-specific features), accepting the cost of training K dictionaries. We chose a reconstruction-based intervention (modify sparse code then decode) over direct activation patching because it respects the dictionary's linear generative model, accepting the cost that reconstruction error may dilute the intervention; we control this by reporting reconstruction fidelity. A domain expert would not describe VN-Lens as a variant of MC-Lens because MC-Lens's readout carrier is a token set in the decoder vocabulary, whereas VN-Lens's carrier is a learned visual dictionary whose atoms are not constrained to be tokenizable — the intervention algebra is defined over dictionary atoms, not tokens.

**Why this measures what we claim.** The causal effect CE(k) measures causal sufficiency of concept atom k for the model's output because we assume that intervening on the sparse code and reconstructing the activation is a valid intervention on the underlying visual feature represented by atom k; this assumption fails when the SAE dictionary is not monosemantic (an atom mixes multiple features), in which case CE(k) reflects the mixed feature's effect rather than a single concept's effect. The reconstruction fidelity (||h_l - D_l(c)||) measures the faithfulness of the dictionary basis because we assume that low reconstruction error implies the dictionary spans the relevant activation subspace; this assumption fails when the dictionary overfits to frequent features and ignores rare but causally relevant ones, in which case fidelity reflects coverage of common features only. The sparsity of c measures concept selectivity because we assume that sparse codes correspond to few active concepts; this assumption fails when the SAE learns dense codes for complex visual inputs, in which case sparsity reflects input complexity rather than concept count. The comparison of CE(k) for tokenizable vs. non-tokenizable atoms measures the closure gap of the token-anchored lens because we assume that atoms with no token-correlate correspond to non-tokenizable visual features; this assumption fails when an atom is non-tokenizable merely due to dictionary training artifacts, in which case the gap reflects training noise rather than a fundamental limitation.

**(D) Assumption block: reconstruction-intervention-to-causal-sufficiency equivalence.**

- **Assumption A (reconstruction fidelity + monosemanticity):** $D_l(E_l(h_l))$ reconstructs $h_l$ with negligible error (we operationalize 'negligible' as normalized reconstruction error $\|h_l - D_l(c)\|^2 / \|h_l\|^2 < 0.05$ on held-out images) and atom $k$ is monosemantic (we operationalize 'monosemantic' as: the top-100 activating images for atom $k$ share a single human-labeled visual attribute with $\geq 80\%$ agreement, measured by the auto-interpretation diagnostic below).
- **Failure mode F:** If reconstruction error exceeds 0.05 or atom $k$ is polysemantic (top-100 images span $\geq 2$ distinct attributes), then $CE(k)$ reflects a mixed feature or an off-manifold perturbation rather than a single concept's effect.
- **Diagnostics reported jointly with CE(k):** (i) per-layer normalized reconstruction error on a held-out set of 2,000 images; (ii) per-atom monosemanticity score = fraction of top-100 activating images whose auto-interpretation label matches the atom's assigned attribute; (iii) fraction of activation variance outside the SAE span, computed as $1 - \mathrm{Var}(D_l(c)) / \mathrm{Var}(h_l)$ per layer. These three numbers bound the closure claim: atoms with high monosemanticity and low reconstruction error are the ones for which CE(k) is interpreted as single-concept causal sufficiency; atoms failing either threshold are flagged and their CE(k) is reported separately as 'mixed-feature effect'.

## Contribution

['VN-Lens: a modality-native concept lens that replaces the decoder-vocabulary readout carrier with a sparse-autoencoder dictionary trained on vision encoder activations, providing an intervention algebra closed over non-tokenizable visual features.', "A causal-effect metric over dictionary atoms that quantifies sufficiency for visual features without requiring token alignment, and an empirical demonstration that the token-anchored lens's intervention set is not closed under the original formalism.", 'An open-source implementation and evaluation protocol for measuring closure gaps in concept lenses for VLMs.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | VQA-v2 image-question pairs (train split for SAE training: 80k images; eval split: 5k image-question pairs) | Standard vision-language inputs with object-level questions |
| Primary metric | Causal effect concentration on question-relevant atoms, defined as the fraction of total $\sum_k CE(k)$ carried by the top-5 atoms per question, averaged over the eval set | Directly tests whether atoms are causally sufficient |
| Secondary metric | Per-layer normalized reconstruction error $\|h_l - D_l(c)\|^2 / \|h_l\|^2$ and per-atom monosemanticity score (fraction of top-100 activating images whose auto-interpretation label matches the atom's assigned attribute) | Bounds the closure claim and tests Assumption A |
| Baseline | MC-Lens decoder-vocabulary lens | Tests whether token-anchored readout misses visual concepts |
| Baseline | Random-dictionary SAE (shuffled atoms) | Tests whether learned structure drives causal effects |
| Baseline | Direct activation patching | Tests whether reconstruction-based intervention preserves causal signal |
| Ablation-of-ours | Single shared dictionary across layers | Tests whether layer-specific dictionaries are necessary |

### Why this setup validates the claim

The central claim is that a vision-native learned dictionary yields concept atoms that are more causally sufficient for VLM outputs than token-anchored readout, and that this gain is concentrated on visual features that lack a token correlate. The MC-Lens baseline tests the first sub-claim: if a decoder-vocabulary carrier already captures the same causal structure, our dictionary adds nothing. The random-dictionary baseline tests whether any causal effect we observe is a genuine property of learned atoms rather than of the intervention operator itself — if shuffled atoms produce comparable causal effects, the dictionary learned nothing. The direct-patching baseline isolates the cost of reconstruction-based intervention: if patching beats reconstruction uniformly, our intervention algebra is the wrong choice, but if reconstruction wins specifically where patching breaks the forward-pass distribution, that supports our design. The shared-dictionary ablation tests the layer-specificity premise drawn from E1/E4, which report that lower layers encode general visual features and middle layers encode object-specific ones; if one dictionary suffices, that premise is unsupported. The primary metric — concentration of causal effect on question-relevant atoms — is the right detector because it distinguishes a lens that finds many weakly-causal atoms from one that finds few strongly-causal ones, which is exactly what a modality-native basis should improve. The secondary metrics (reconstruction error and monosemanticity) make Assumption A testable: if reconstruction error exceeds 0.05 or monosemanticity falls below 0.8 for the atoms driving the effect, then CE(k) cannot be interpreted as single-concept causal sufficiency and the closure claim is bounded accordingly. Together these form a falsifiable test: the claim predicts a specific interaction (our advantage concentrated on non-tokenizable, object-specific features), not a uniform gain.

### Expected outcome and causal chain

**vs. MC-Lens decoder-vocabulary lens** — On a case where the question asks about a visual attribute with no clean token correlate (e.g., a subtle texture or spatial relation the decoder vocabulary does not name), MC-Lens produces a readout carrier whose nearest token set collapses the attribute into a coarse verbal category, because its atoms are constrained to be tokenizable and cannot represent the residual visual feature. Our method instead reads out the attribute through learned visual atoms whose codes are not forced onto the vocabulary, because the SAE dictionary is fit directly to encoder activations. We expect a noticeable causal-effect gap on non-tokenizable-attribute questions but rough parity on questions whose answers are well lexicalized (e.g., object identity), producing an interaction rather than a main effect.

**vs. Random-dictionary SAE (shuffled atoms)** — On any case, the shuffled-atom baseline produces causal-effect scores that are diffuse and near-uniform across atoms, because its atoms are random directions in activation space and no single atom aligns with a coherent feature, so intervening on one perturbs a mixture. Our method instead yields a sparse, peaked causal-effect profile where a small set of atoms carries most of the output effect, because trained atoms align with sparse feature directions in the encoder. We expect our causal-effect distribution to show markedly higher concentration (a few atoms above tau, most near zero) while the shuffled baseline stays flat, and the gap should hold across layers rather than only at one layer.

**vs. Direct activation patching** — On a case where the target activation lies off the encoder's natural manifold (e.g., an unusual object pose), direct patching produces a large but often incoherent output change, because it substitutes an activation that the downstream layers were not trained to receive, injecting off-manifold noise that contaminates the causal estimate. Our method instead modifies the sparse code and decodes back through the dictionary, keeping the perturbed activation near the encoder's manifold, because the decoder's linear generative model constrains the edit. We expect reconstruction-based intervention to yield cleaner, more monotone dose-response curves (effect growing with alpha) and a better separation between relevant and irrelevant atoms, with the advantage largest where patching's off-manifold error is greatest.

### What would falsify this idea

If our causal-effect advantage over MC-Lens were uniform across all question types rather than concentrated on non-tokenizable, object-specific features, the claim that a visual-native dictionary captures concepts the token lens cannot would be wrong. Likewise, if the shared-dictionary ablation matched per-layer dictionaries on causal concentration, the layer-specificity premise motivating per-layer training would be unsupported.

## References

1. Probing the "Conscious" Mind: Selective Access and Global Workspace in Vision-Language Models
