# Noise-Consistent Schema Alignment for Omni-Modal Parsing

## Motivation

OSAD (Youtu-Parsing-Omni) trains its unified encoder on clean modality inputs with token-level targets from a reference-conditioned copy, so the shared encoder is never forced to produce the same OmniSchema under corrupted inputs; VITA-1.5's observed vulnerability to noisy audio/image inputs is a symptom of this structural gap, not a data-scarcity problem. Because alignment is only ever optimized on the clean manifold, the encoder can satisfy the objective while learning a schema representation that is not invariant to modality corruption — the training signal contains no term that penalizes schema drift caused by noise. This is a meta-level failure: the field's self-evolving post-training recipes (OSAD, reference-conditioned rollouts) assume their own clean rollouts are the only supervision needed, leaving robustness to be hoped for rather than trained.

## Key Insight

If the same clean input is passed through two independent corruption draws and the shared encoder is required to map both corrupted encodings to the clean schema embedding while the schema decoder is frozen on the clean target, then the only degree of freedom left is the encoder's noise-invariant subspace — so schema consistency becomes an invariant the encoder must satisfy, not a property the decoder can absorb.

## Method

**What it is.** NCSA (Noise-Consistent Schema Alignment) is a post-training add-on to OSAD. Inputs: a clean multimodal document (text tokens, image patches, audio frames, video frames) with its OmniSchema token-level target; outputs: an updated Youtu-Omni-Encoder whose schema embedding is stable under controlled modality corruption.

**Load-bearing assumption (stated explicitly).** NCSA assumes the corruption operators are *label-preserving*: for every training sample, the clean OmniSchema target y must remain fully recoverable from each corrupted view x_k, so that collapsing z_1 and z_2 cannot destroy schema-relevant information. This assumption is not free — occlusion of a table cell or audio corruption masking a spoken figure can remove information that is genuinely absent, and a pure L2 term would then push toward an under-determined schema. We therefore (i) bound corruption magnitudes (SNR ≥ 5 dB, occlusion ≤ 30%, token dropout ≤ 10%), (ii) add a variance-preservation term so the encoder cannot collapse to a constant representation, and (iii) gate the consistency loss per-sample by an information-retention check, so the encoder is never asked to collapse variance that encodes genuinely absent schema content.

**How it works.**

```pseudocode
# Per training step, given clean sample x = (x_txt, x_img, x_aud, x_vid) and OmniSchema target y
# Encoder E (shared weights), schema decoder D, reference-conditioned copy R (frozen per OSAD round)
# Corruption operators: C_img (occlusion, p=0.3), C_aud (white/segmental corruption, SNR 5-15 dB), C_txt (token dropout, p=0.1)

1. z_clean = E(x)                                   # clean encoding
2. y_hat_clean = D(z_clean)                         # main OSAD loss below
3. L_osad = token_role_CE(y_hat_clean, y)           # existing OSAD objective, unchanged

4. For k in {1,2}:                                  # two independent corruption draws
5.     x_k = (C_txt(x_txt), C_img(x_img), C_aud(x_aud), x_vid)   # same corruption family, independent noise seed
6.     z_k = E(x_k)
7.     y_hat_k = D(z_k)

# --- information-retention gate (per sample) ---
8.  g = 1[ CE(y_hat_1, y) <= tau AND CE(y_hat_2, y) <= tau ]   # tau = 1.5 * CE(y_hat_clean, y), clipped to [0.5, 2.0]
#     i.e. apply the consistency term only where both corrupted views still decode to the clean target
#     within a margin; otherwise the corruption is treated as non-label-preserving and L_cons is skipped.

# --- variance-preserving invariance (VICReg-style) ---
9.  L_inv   = || z_1 - z_2 ||_2^2
10. L_var   = mean_j hinge(1 - std_batch(z_1[:,j])) + mean_j hinge(1 - std_batch(z_2[:,j]))
#     std_batch over the minibatch (batch size 64); hinge(a) = max(0, a); target std = 1 per dim
11. L_cons  = g * ( L_inv + mu * L_var )            # mu = 1.0

12. L_schema = g * ( CE(y_hat_1, y) + CE(y_hat_2, y) )   # schema target consistency under noise, gated identically
13. L_total  = L_osad + lambda_c * L_cons + lambda_s * L_schema
# hyperparameters: lambda_c = 0.5, lambda_s = 0.25, mu = 1.0, tau = 1.5 * CE(y_hat_clean, y) clipped to [0.5, 2.0],
# corruption p_img=0.3, p_txt=0.1, SNR_aud in [5,15] dB, batch size 64
# D is frozen during the L_cons/L_schema branch so the encoder cannot push the invariance burden onto the decoder
```

The mechanism is: two independent draws from the same corruption family produce two corrupted encodings `z_1, z_2`; the L2 consistency term forces the encoder to collapse their difference, the variance term prevents the constant-representation collapse that a pure L2 objective admits, and the schema-CE term forces both to remain decodable to the clean target. The per-sample gate `g` enforces the label-preservation assumption at the sample level: where a corrupted view no longer decodes to y within margin tau, the corruption has removed genuinely needed information and the consistency term is switched off for that sample. Freezing D during the consistency branch is what makes the invariance land on E rather than being absorbed by a more expressive decoder.

**Why this design.** We chose two independent corruption draws over a single noisy/clean pair because a single pair admits a degenerate solution where the encoder learns a fixed offset that maps noisy to clean without learning invariance; two independent draws force the encoder to collapse the *variance* of the corruption distribution, accepting the cost of 2× encoder forward passes per step. We chose to freeze the schema decoder D on the consistency branch rather than fine-tune it jointly because a jointly trained decoder can absorb schema drift by becoming noise-aware, which would leave the shared encoder unchanged — the exact failure VITA-1.5 exhibits; the cost is that the decoder cannot adapt to genuinely new schema tokens during this phase. We chose an explicit L2 consistency loss over a contrastive InfoNCE loss because the clean/noisy pair is a *paired* observation (same underlying schema), so the positive is known exactly and InfoNCE's negative sampling would inject spurious negatives from other documents; the cost is that L2 is less robust to outlier corruptions, which is why the corruption magnitudes are bounded (SNR ≥ 5 dB, occlusion ≤ 30%) and why we add the VICReg-style variance term (mu = 1.0) and the per-sample retention gate g — a pure L2 term on augmented views admits the constant-representation solution that Barlow Twins and VICReg explicitly guard against, and the gate prevents the encoder from being asked to collapse variance that encodes genuinely absent schema content. A domain expert would not describe this as "OSAD + noise augmentation": standard augmentation changes the input distribution but has no term coupling the two corrupted encodings, and OSAD's own reference-conditioned copy is a *clean* self-teacher — NCSA adds a paired-encoding invariance constraint that OSAD structurally lacks.

**Why it measures what we claim.** `||z_1 - z_2||_2^2` measures *noise-invariant schema alignment* because it is the empirical variance of the encoder output under the corruption distribution, and the assumption that makes this equivalent to alignment robustness is that the corruption operators are label-preserving (they do not change the OmniSchema target); this assumption fails when occlusion removes an entire table cell or audio corruption masks a spoken figure, in which case the L2 term reflects the encoder's inability to recover information that is genuinely absent, and the loss pushes toward a schema that is *under-determined* rather than robust — which is why the per-sample gate g disables L_cons exactly on those samples and the variance term L_var blocks the constant-representation collapse. `CE(y_hat_1, y) + CE(y_hat_2, y)` measures *schema fidelity under noise* because it is the token-role cross-entropy of the noisy branch against the clean target, and the assumption is that the reference-conditioned copy R used to generate y is itself noise-free; this fails when R's own rollouts contain schema errors, in which case the CE term reinforces those errors under noise. Freezing D measures *encoder-side invariance* because it removes the decoder's capacity to compensate, and the assumption is that the frozen D is a faithful readout of the clean schema; this fails if D is under-trained at the start of post-training, in which case the consistency loss is applied to a decoder that cannot distinguish schema tokens.

## Contribution

['A noise-consistency post-training objective (NCSA) for OSAD that adds a paired-corruption encoder-invariance term and a schema-CE term under a frozen decoder, making noise-invariant schema alignment a directly trained invariant rather than a hoped-for side effect.', "The design principle that freezing the schema decoder during the consistency branch is what forces the invariance onto the shared encoder, distinguishing this from standard modality augmentation and from OSAD's clean self-teacher.", 'A controlled corruption protocol (bounded occlusion, SNR-bounded audio corruption, token dropout) that defines the label-preserving noise manifold on which schema consistency is a valid training signal.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | OmniSchema parsing suite (10 tasks) | Covers all four modalities under one schema |
| Primary metric | Token-role schema accuracy under held-out corruption | Directly measures invariance the claim asserts |
| Diagnostic metric | Per-token-role schema-embedding drift: cosine distance between clean and corrupted schema embeddings, computed per token role and correlated (Pearson r) with the L2 term | Validates the proxy: tests the assumption that low encoder-output variance implies preserved token-role decodability |
| Diagnostic metric | Label-preservation audit: fraction of samples where the per-sample gate g fires (CE(y_hat_k, y) ≤ tau) per corruption family and magnitude | Directly measures how often the load-bearing label-preservation assumption holds |
| Baseline | Youtu-Parsing-Omni (OSAD, no NCSA) | Isolates the added invariance term |
| Baseline | OSAD + standard noise augmentation | Tests whether coupling term matters |
| Baseline | VITA-1.5-style joint decoder fine-tune | Tests the decoder-absorption failure path |
| Ablation-of-ours | NCSA with single noisy/clean pair (no two-draw) | Tests the degenerate fixed-offset solution |
| Ablation-of-ours | NCSA without variance term (mu = 0) | Tests whether the constant-representation collapse the VICReg term guards against actually occurs |
| Ablation-of-ours | NCSA without per-sample gate (g = 1 always) | Tests whether the label-preservation gate matters on non-label-preserving corruptions |

### Why this setup validates the claim

The central claim is that paired two-draw consistency with a frozen decoder produces encoder-side schema invariance that neither plain OSAD nor ordinary augmentation can produce. The dataset matters because OmniSchema's ten tasks span text, image, audio, and video under one token-role target, so a single corruption protocol can be applied per modality while keeping the label fixed — which is exactly the assumption the L2 term relies on. The primary metric is accuracy of the decoded schema under corruption the model never saw at training magnitude, because that is where an encoder that merely memorized clean inputs and one that collapsed corruption variance diverge. The per-token-role drift diagnostic tests the equivalence directly: if cosine drift between clean and corrupted schema embeddings correlates with the L2 term (Pearson r reported per token role), the proxy is validated; if not, low L2 variance does not imply preserved decodability. The label-preservation audit reports how often the gate g fires, so the load-bearing assumption is measured rather than asserted. OSAD-without-NCSA tests the sub-claim that the invariance term is necessary; OSAD-with-augmentation tests the sub-claim that the *coupling* term, not the input distribution shift, is what does the work; the VITA-1.5-style joint-decoder baseline tests the sub-claim that freezing D is what forces invariance onto E. The single-pair ablation tests the degenerate-solution argument directly: if a single noisy/clean pair matches two independent draws, our stated mechanism is wrong. The mu = 0 and no-gate ablations test the two guards added for the load-bearing assumption. Together these form a falsifiable test because each baseline removes exactly one design choice, so a gain that survives all of them but vanishes in the ablation localizes the effect to the claimed mechanism.

### Expected outcome and causal chain

**vs. Youtu-Parsing-Omni (OSAD, no NCSA)** — On a document where a table cell is partially occluded (within the 30% bound) and the surrounding text still determines the cell's role, plain OSAD produces a schema in which that cell's role token flips or defaults, because its encoder has no term coupling two views of the same document and so treats the occlusion as a distributional fact about the input rather than noise to be marginalized. Our method instead encodes two independent occlusion draws and the L2 term forces their difference toward zero while the schema-CE keeps both decodable, so the encoder learns to route around the occluded patch using the surviving text. We expect a clear accuracy gap on the corrupted-eval subset, concentrated on documents with partial occlusion, and near-parity on clean documents; the per-token-role drift diagnostic should show lower cosine drift on the roles whose surrounding text determines them.

**vs. OSAD + standard noise augmentation** — On an audio segment where segmental corruption at 10 dB masks a spoken figure, the augmentation baseline produces a schema that mislabels the figure's role, because augmentation changes the marginal input distribution but never couples two corrupted encodings, so the encoder can satisfy the loss by fitting noisy inputs independently and never learns to collapse their variance. Our method applies the same corruption family but draws it twice and penalizes the encoder-output difference, so the encoder is pushed toward a representation that is constant across the corruption distribution. We expect the gap to be modest on mild corruption and to widen noticeably as SNR drops toward 5 dB, where variance collapse matters most; the label-preservation audit should show the gate firing more often as SNR drops, indicating where the assumption begins to fail.

**vs. VITA-1.5-style joint decoder fine-tune** — On a video frame sequence where a caption's temporal marker is corrupted, the joint-decoder baseline produces a schema that drifts because its noise-aware decoder absorbs the corruption, leaving the shared encoder unchanged and therefore still noise-sensitive. Our method freezes D on the consistency branch, so the decoder cannot compensate and the invariance burden lands on E, which must produce encodings that the frozen clean readout can still decode. We expect the joint-decoder baseline to look competitive on clean data but to show a distinct accuracy drop on corrupted video and audio-visual subsets, while our method holds closer to its clean-data accuracy.

**vs. NCSA without variance term (mu = 0)** — If the constant-representation collapse the VICReg term guards against is real, the mu = 0 ablation should show degraded clean-data accuracy (the encoder collapses variance that also carries clean schema content) even if corrupted-eval accuracy looks similar; we expect the full NCSA to dominate on the clean subset.

**vs. NCSA without per-sample gate (g = 1 always)** — On samples where occlusion removes the only carrier of a cell's role token, the ungated variant is forced to collapse variance that encodes genuinely absent content, pushing toward an under-determined schema; we expect the gated variant to show higher accuracy on exactly the samples where the audit reports the gate firing, and near-parity elsewhere.

### What would falsify this idea

If the two-draw ablation matches full NCSA on corrupted-eval accuracy, the claimed variance-collapse mechanism is not what produces the gain, and the single-pair degenerate solution is sufficient. If the per-token-role drift diagnostic shows no correlation between the L2 term and cosine drift, the proxy does not measure schema invariance and the claim is unsupported. If the gain is uniform across clean and corrupted subsets rather than concentrated on the corruption-family subsets the mechanism targets, the improvement is a generic capacity effect rather than noise-invariant schema alignment.

## References

1. Youtu-Parsing-Omni: One Encoder, One Schema, All Modalities
2. Qwen3-Omni Technical Report
3. CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models
4. Qwen2-Audio Technical Report
5. Qwen Technical Report
6. Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models
7. Finite Scalar Quantization: VQ-VAE Made Simple
8. Speak While You Think: Streaming Speech Synthesis During Text Generation
