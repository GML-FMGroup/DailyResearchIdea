# Counterfactual Region Sufficiency for Composition Judgment Grounding

## Motivation

Existing composition understanding benchmarks, such as the PICD dataset and benchmark from Can Machines Understand Composition?, evaluate models by whole-image categorical labels or scalar aesthetic scores without any structural property that identifies which image regions license the judgment. This whole-image mapping persists across both specialized embedding models and MLLM evaluation, meaning a model can score well on composition categories while providing no identifiable region-level attribution. The root cause is that composition is treated as an unidentifiable whole-image function, so there is no formal condition under which a judgment can be attributed to a specific region set.

## Key Insight

A composition judgment is structurally identifiable only when a minimal region set exists whose removal flips the frozen LLM's judgment while all other regions are held fixed, and this counterfactual flip condition can be enforced as a contrastive training signal on a lightweight interface between frozen encoders and frozen LLMs.

## Method

**What it is.** We introduce **CRSG** (Counterfactual Region Sufficiency Grounding), a lightweight trainable interface that sits between a frozen vision encoder (CLIP ViT-L/14, patch size 14, producing a 16x16 grid of 256 patch features of dimension d=1024) and a frozen LLM (LLaVA-1.5-7B with its frozen vision projector, or Flan-T5-XL with a frozen linear projector). Given an image and a composition judgment query, CRSG outputs a binary region mask over the fixed 16x16 patch grid and a scalar counterfactual sufficiency score, without modifying either frozen model. CRSG has 0.66M trainable parameters (grounding head only).

**How it works.** CRSG is trained with a contrastive objective that enforces the predicted mask to be both necessary and sufficient for the frozen LLM's judgment. The load-bearing assumption is stated explicitly: *zeroing the pixels of a predicted patch set while leaving all other pixels intact is treated as a valid, judgment-preserving intervention, so that the change in the frozen LLM's judgment under removal isolates the causal necessity of that region set rather than an out-of-distribution artifact of the masked input.* Because this assumption can fail (zeroed patches are OOD for frozen encoders/LLMs), we (i) replace hard zeroing with an in-distribution counterfactual intervention (patch-shuffle infill from a matched reference pool, see step 4), and (ii) calibrate the flip signal per-image against a matched-area random-mask OOD control (step 5b), so L_nec measures judgment-relevant removal net of the intervention artifact.

```pseudocode
Input: image I, composition judgment query q, frozen encoder E, frozen LLM L
Hyperparameters: patch grid G=16x16 (256 patches), mask temperature tau=0.1 (annealed 1.0->0.1 over first 5k steps), margin m=0.5, lambda_nec=1.0, lambda_suf=1.0, lambda_sparse=0.01, lambda_cal=0.5, K_rand=4 random-control masks per image, reference pool size R=512 images from PICD train split

1. Encode I with E -> patch features F in R^{256 x 1024}
2. Grounding head H (2-layer MLP: 1024->256->1, GeLU activation, hidden 256) maps each patch feature to a logit
3. Sample mask M ~ Bernoulli(sigmoid(H(F)/tau)) via Gumbel-Sigmoid (straight-through), M in {0,1}^{256}
4. In-distribution counterfactual intervention (replaces pixel-zeroing):
     I_masked = I with patches where M=1 replaced by the corresponding patch from a reference image I_ref drawn uniformly from the R=512-image reference pool (patch-shuffle infill); this keeps the masked input on the frozen encoder/LLM manifold
     (Pixel-zeroing variant I_zero is retained only as an OOD-control probe, not for training)
5. Compute frozen-LLM judgments and calibrated flip probability:
     y_full = L(E(I), q)                         # full image
     y_masked = L(E(I_masked), q)               # image with masked regions infilled
     s(.) = temperature-scaled logit of the LLM's 'yes' judgment token for query q (temperature fit on a 512-example calibration set from PICD train, held out from grounding-head training)
     flip = sigmoid((s(y_full) - s(y_masked)) / T_cal)   # calibrated flip probability, T_cal from calibration set
5b. OOD-control calibration (per-image):
     For k=1..K_rand (K_rand=4): draw random mask M_k with area matched to |M|/256; compute flip_k = sigmoid((s(y_full) - s(y_masked_k)) / T_cal) under the same infill intervention
     flip_ctrl = mean_k flip_k
     L_nec = max(0, m - (flip - flip_ctrl))   # necessity net of intervention artifact
6. Sufficiency loss (contrastive, both arguments are judgments of the SAME masked image under different queries):
     L_suf = -log( exp(s(y_masked, q)/tau) / sum_{q' in Q_alt} exp(s(y_masked, q')/tau) )
     where Q_alt = the 23 other PICD composition categories (semantically disjoint from q by construction of the 24-category label set), and s(y_masked, q') is the frozen LLM's judgment of the SAME masked image I_masked under query q'
7. Sparsity loss: L_sparse = mean(M)
8. Total loss: L = lambda_nec * L_nec + lambda_suf * L_suf + lambda_sparse * L_sparse + lambda_cal * L_cal
     where L_cal = (flip - flip_ctrl)^2 is a calibration regularizer that keeps the net flip signal well-scaled
9. Output at inference: M (thresholded at 0.5), sufficiency score = flip - flip_ctrl (calibrated net flip probability)

Training: AdamW, lr=1e-3, batch size 16 images, 20 epochs on PICD train split, single A100 40GB (~6 GPU-hours). Frozen-LLM judgments are cached for a fixed candidate mask pool (256 masks/image) to avoid re-running the LLM each step; a small differentiable surrogate (linear probe on frozen E features, 1024->1) is distilled from the frozen LLM's judgments and verified to agree with the frozen LLM on a held-out 512-example set (agreement >= 0.90) before being used to train H; final 5 epochs use the frozen LLM directly.
```

**Why this design.** We chose a frozen-encoder/frozen-LLM interface over end-to-end fine-tuning because the goal is to learn only the spatial grounding, not to alter the judgment function; this avoids retraining and enables cross-model transfer, accepting the cost that the interface cannot correct systematic LLM composition biases. We chose a Gumbel-Sigmoid straight-through mask over a continuous soft mask because the counterfactual flip condition requires a discrete removal operation (a soft mask would not actually remove regions), accepting the cost of higher-variance gradients that we mitigate with tau=0.1 annealing. We chose a contrastive sufficiency loss over a simple binary cross-entropy on the mask because sufficiency requires the masked judgment to be discriminative for the target query against other queries, not just to flip; this is the key difference from prior region-proposal or saliency methods such as Grad-CAM or RISE, which identify important regions but do not enforce a counterfactual sufficiency condition, and a domain expert would not describe CRSG as a variant of those because it optimizes a joint necessity-and-sufficiency contrastive objective against a frozen LLM's judgments. We chose a 16x16 patch grid over pixel-level masks to keep the grounding head lightweight and the counterfactual intervention tractable, accepting coarser spatial resolution. We chose patch-shuffle infill over pixel-zeroing for the intervention because zeroing produces OOD inputs whose judgment changes may reflect the artifact rather than removal of judgment-relevant content, accepting the cost of a reference pool and a per-image random-mask control.

**Why it measures what we claim.** The calibrated net flip signal (flip - flip_ctrl) measures causal necessity of the predicted region set because the infill intervention holds all other regions fixed and the per-image random-mask control subtracts the OOD artifact of the intervention itself; this assumption fails when the frozen LLM's judgment is sensitive to global context outside the mask (e.g., overall scene layout) or when the reference pool is distributionally mismatched, in which case the net flip reflects a partial intervention plus residual artifact rather than true necessity. The contrastive sufficiency loss L_suf measures sufficiency of the masked region set for the target judgment because the softmax over the 23 alternative PICD categories enforces that the masked image is more predictive of the target query than of others; this assumption fails when alternative queries are semantically overlapping (e.g., 'rule of thirds' vs. 'centered composition'), in which case L_suf reflects query discriminability rather than sufficiency for the specific judgment. The sparsity loss L_sparse measures minimality of the region set because it penalizes mask area; this assumption fails when the frozen LLM requires a large region to flip, in which case L_sparse reflects a trade-off with necessity rather than true minimality. The sufficiency score (flip - flip_ctrl) measures the counterfactual sufficiency of the region set because it quantifies the calibrated judgment change under removal net of the intervention artifact; this assumption fails when the LLM's score is not calibrated across masked and full inputs, in which case the score reflects a raw logit difference rather than a calibrated flip probability.

## Contribution

['A novel region-level attribution identifiability condition for composition judgments, formalized as a counterfactual flip condition over minimal region sets, together with CRSG, a lightweight grounding interface between frozen vision encoders and frozen LLMs that outputs region masks and counterfactual sufficiency scores trained with a joint necessity-and-sufficiency contrastive objective.', 'A design principle showing that enforcing both necessity (flip under removal) and sufficiency (discriminability against alternative queries) as a contrastive objective yields region masks that are structurally attributable, unlike saliency or region-proposal baselines that optimize only importance.', 'A benchmark protocol and evaluation metric for region-level composition attribution on the PICD dataset, measuring mask necessity, sufficiency, and sparsity against human-annotated region sets.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | PICD composition benchmark (24 categories) | Composition-specific judgments with category structure for contrastive queries |
| Primary metric | Calibrated net counterfactual flip rate (flip - flip_ctrl) on held-out queries | Directly tests necessity+sufficiency net of OOD artifact |
| Secondary metric | Pixel-zeroing vs. patch-shuffle flip agreement on 512-example probe set | Validates intervention operator consistency |
| Baseline | Grad-CAM region grounding | Saliency-only, no counterfactual sufficiency objective |
| Baseline | RISE perturbation masks | Importance-only, no frozen-LLM contrastive necessity |
| Baseline | Random 16x16 mask (matched area) | Controls for sparsity/area confound |
| Baseline | Pixel-zeroing CRSG (OOD-control) | Isolates whether infill intervention removes artifact |
| Ablation-of-ours | CRSG minus L_suf (necessity only) | Isolates whether contrastive sufficiency drives gains |
| Ablation-of-ours | CRSG minus L_cal (no OOD-control calibration) | Isolates whether calibration removes artifact |

### Why this setup validates the claim

The central claim is that a region set which is jointly necessary and sufficient for a frozen LLM's composition judgment can be learned by a lightweight interface, and that this differs from saliency. PICD is the right substrate because it supplies composition judgments across 24 categories, giving the contrastive sufficiency loss genuine alternative queries to discriminate against — without semantic variety, L_suf would be vacuous. The primary metric, calibrated net counterfactual flip rate under infill removal, is the right detector because it directly operationalizes the causal intervention the method optimizes while subtracting the per-image OOD artifact via the matched-area random-mask control; a method that merely highlights important-looking regions (Grad-CAM, RISE) can score high on saliency-overlap metrics without producing masks whose removal actually flips the judgment. The secondary metric, pixel-zeroing vs. patch-shuffle flip agreement on a 512-example probe set, validates the intervention operator itself: if the two interventions yield consistent flips, the flip signal is not an artifact of zeroing. Grad-CAM tests whether attribution alone suffices; RISE tests whether perturbation-importance alone suffices; the random-area mask controls for the trivial explanation that any mask of similar sparsity flips judgments; the pixel-zeroing CRSG variant tests whether the infill intervention is doing the claimed work. The necessity-only and no-calibration ablations are the decisive controls: if the full CRSG beats them, the gain is attributable to the sufficiency objective and the calibration, which are the paper's distinguishing mechanisms. Together these form a falsifiable test: the claim predicts a specific ordering (CRSG > necessity-only > saliency baselines > random) concentrated on categories where sufficiency discrimination matters, and CRSG > pixel-zeroing CRSG on categories where the OOD artifact is largest.

### Expected outcome and causal chain

**vs. Grad-CAM** — On a case where the composition judgment hinges on a spatially diffuse cue, such as leading lines whose relevance depends on where the eye exits the frame, Grad-CAM produces a hotspot on high-contrast edges because its mechanism is gradient saliency, which has no notion of whether removing the region changes the judgment. Our method instead selects the region whose removal flips the frozen LLM's judgment because L_nec explicitly optimizes that counterfactual (net of the random-mask control), so we expect a clear flip-rate gap on diffuse-cue categories (leading lines, balance) but near-parity on focal-object categories where saliency and necessity coincide.

**vs. RISE** — On a case where a large salient object dominates but the composition query concerns framing, such as 'rule of thirds' on a centered subject, RISE produces a mask covering the subject because its importance scores track prediction confidence under occlusion, not judgment-specific necessity. Our method instead masks the off-subject regions whose removal degrades the target judgment relative to alternative queries because L_suf forces the masked image to be discriminative for that query, so we expect a moderate gap on framing categories and a larger gap precisely where saliency and composition diverge.

**vs. Random 16x16 mask** — On a case where any sufficiently large region removal perturbs the LLM, a random mask of matched area occasionally flips the judgment by chance, producing a non-trivial floor. Our method instead concentrates the mask on judgment-relevant patches because the joint necessity-sufficiency objective rewards targeted removal, so we expect CRSG to exceed the random floor by a wide margin while the necessity-only ablation lands between them, confirming the floor is real but the gain is mechanism-driven.

**vs. Pixel-zeroing CRSG** — On a case where zeroed patches create black holes that violate scene layout, the pixel-zeroing variant's flip signal is inflated by the OOD artifact because the frozen LLM responds to the perturbation itself rather than to removal of judgment-relevant content. Our method instead uses patch-shuffle infill and subtracts the per-image random-mask control because L_nec is defined on the calibrated net flip, so we expect the infill CRSG to show a smaller but more category-concentrated flip-rate advantage, with the largest gap on categories where the OOD artifact is strongest (balance, leading lines).

### What would falsify this idea

If CRSG's flip-rate advantage over Grad-CAM and RISE is uniform across all 24 categories rather than concentrated on categories where saliency and composition necessity are predicted to diverge, the sufficiency mechanism is not doing the claimed work. Equally, if the necessity-only ablation matches full CRSG, the contrastive sufficiency loss is inert and the central claim collapses to ordinary saliency; and if pixel-zeroing CRSG matches infill CRSG, the OOD-control calibration is inert and the intervention-validity assumption is unverified.

## References

1. Can Machines Understand Composition? Dataset and Benchmark for Photographic Image Composition Embedding and Understanding
2. Automatic Image Composition Optimization and Visual Perception Simulation Evaluation for Architectural Interior Design
3. Visual Instruction Tuning
4. Scaling Instruction-Finetuned Language Models
5. Self-Instruct: Aligning Language Models with Self-Generated Instructions
