# Text-Gated Visual Feature Selection for Trainable-Backbone Vision-Language-Action Models

## Motivation

Across the VLA literature the same structural failure recurs: the semantic alignment between visual features and language that makes VLMs useful for control is only preserved when the vision-language encoder is frozen, and is silently discarded once the backbone is fine-tuned end-to-end. OTTER (E3) makes this explicit — it keeps pre-trained encoders frozen precisely because independent visual/language feature feeding 'degrad[es] the pre-trained semantic alignments' — while ScaleVLA (E5, E8) shows the field's trajectory is toward fully trainable frontier backbones (552B), where OTTER's freezing guarantee is no longer available. Thus the meta-level problem is that task-relevant visual feature identifiability is currently an artifact of a frozen encoder rather than a property the architecture can maintain under joint action-expert training.

## Key Insight

If the text-conditioned selection gate — not the encoder weights — is made the invariant, then semantic alignment between selected visual features and the instruction is enforced by a contrastive objective at every training step, so it survives backbone fine-tuning that would otherwise dissolve the frozen-encoder guarantee.

## Method

**What it is.** We introduce TVG-VLA (Text-Gated Visual selection for VLA), which inserts a lightweight text-aware selection gate between a trainable 552B DeepSeek-V4.1-Flash visual encoder and a trainable action DiT. It takes (image tokens V, language instruction L) and outputs a sparse, instruction-aligned visual feature set Z that is the sole visual input to the action DiT, plus a contrastive alignment loss.

**Load-bearing assumption (stated explicitly).** The gate — not the encoder weights — can carry the instruction-conditioned identifiability property that OTTER obtains only by freezing. Concretely, we assume that a contrastive alignment objective on the selected visual features, trained jointly with the action loss on a moving 552B backbone, is a strong enough counter-pressure to preserve semantic alignment between Z and L. This assumption is load-bearing because the entire claim rests on it; it is unverified because (i) pooled contrastive objectives can collapse to batch-level centroids without localizing (GLIP, DenseCL), (ii) joint training of a VLM with an action expert can degrade semantic knowledge unless insulated (E4/E11), and (iii) a hard top-k mask can create a self-confirming loop if the gate initially selects wrong patches. The method below specifies the concrete calibration and verification steps that test this assumption inside the existing training procedure.

**How it works.**

```pseudocode
Inputs: image tokens V ∈ R^{N×d} (N≈256 patches), instruction embedding e_L = VLM_text(L) ∈ R^d
Gate parameters: W_q, W_k ∈ R^{d×d}; temperature τ=0.07; keep-ratio r=0.25
Backbone adapter: low-rank adapter on the 552B backbone with rank 16, scaling factor 0.1 (knowledge-insulation-style separation per E4/E11)

1. Instruction query projection:
     q = W_q · e_L                      # one query vector per instruction
2. Per-patch relevance scores (over ALL N patches, before selection):
     s_i = <q, W_k · v_i> / sqrt(d)     # i = 1..N
3. Sparsemax over s with keep-ratio r:
     α = Sparsemax(s / τ) restricted to top-k, k = floor(r·N)
4. Selected features:
     Z = { α_i · v_i : α_i > 0 }        # typically k≈64 tokens
5. Action generation:
     A = ActionDiT(Z, e_L)              # action chunk, unchanged from ScaleVLA
6. Per-token alignment loss (replaces pooled InfoNCE):
     L_align = (1/N) Σ_{i=1..N} [ -log( exp(<v_i, e_L>/τ) / Σ_{L'∈batch} exp(<v_i, e_{L'}>/τ) ) ]
     # computed over ALL N patches, so unselected patches still receive alignment gradient
7. Coverage regularizer on α (prevents premature collapse of selection set):
     L_cov = -H(α) = Σ_i α_i log α_i    # entropy of the sparsemax distribution
8. Total loss: L = L_action + λ · L_align + β · L_cov,  λ = 0.1, β = 0.01
```

Training alternates: the 552B backbone is updated by L_action gradients that flow only through selected tokens (α acts as a gradient mask), while L_align updates the gate parameters W_q, W_k and the per-token projection, and L_cov updates the gate. To prevent L_align from being overwhelmed by action gradients, the backbone update uses a stop-gradient on the alignment pathway: L_align gradients do not flow into the 552B backbone weights; they flow only into the gate and the per-token projection. The backbone is trained solely by L_action through the low-rank adapter (rank 16, scaling 0.1), which insulates the pretrained VLM semantics from the action expert's continuous-output pressure (E4/E11). The gate is ~2M parameters; the backbone and DiT are trained jointly as in ScaleVLA, with the adapter and stop-gradient separation described above.

**Why this design.** We chose a *sparsemax top-k gate* over a dense attention re-weighting because OTTER's frozen-encoder guarantee came from selecting a small, task-relevant subset — a dense soft-attention would let irrelevant patches leak into the DiT and re-couple visual noise with action, accepting the cost that hard selection makes the gate non-differentiable at the boundary and requires the straight-through estimator. We chose a *per-token contrastive objective over all N patches* over a pooled InfoNCE on mean(Z) because pooled objectives constrain only the batch-level centroid and do not localize (GLIP, DenseCL), whereas per-token alignment directly penalizes each patch that fails to distinguish instructions, including unselected patches, accepting the cost that batch composition (negative sampling) now matters and small batches weaken the signal. We chose to *keep the 552B backbone trainable* rather than freeze it as OTTER does, because ScaleVLA's scaling gains (E5) depend on end-to-end adaptation, accepting the cost that the gate must absorb the alignment burden and that compute is not reduced relative to ScaleVLA. We added a *low-rank adapter and stop-gradient separation* because E4/E11 show that naively adding untrained continuous action-expert parameters to a pretrained VLM significantly harms training speed and knowledge transfer, and a 2M-parameter gate trained by a λ=0.1 auxiliary loss is a far weaker counter-pressure than the action gradient; the adapter insulates the backbone while the stop-gradient ensures L_align cannot be overwhelmed. We added a *coverage regularizer* because a hard top-k mask can create a self-confirming loop if the gate initially selects wrong patches; the entropy term prevents premature collapse of the selection set. A domain expert would not describe this as 'OTTER + ScaleVLA' because OTTER's alignment is a *consequence of frozen weights* (a static property), whereas TVG-VLA's alignment is an *actively enforced training invariant* on a moving backbone — the selection mechanism, not the encoder, carries identifiability, which is a structural property the composition of OTTER and ScaleVLA does not produce.

**Why it measures what we claim.** The quantity `L_align` (per-token contrastive loss over all N patches) measures *instruction-conditioned identifiability of visual features* because the assumption that the batch contains matched (v_i, L) pairs and mismatched (v_i, L') pairs makes minimizing L_align equivalent to maximizing mutual information between each patch and its instruction; this assumption fails when two instructions in a batch are semantically near-duplicate (e.g., 'pick up the red cup' vs. 'grasp the crimson mug'), in which case L_align reflects lexical rather than referential discrimination. The keep-ratio r and sparsemax support measure *task-relevance sparsity* because the assumption that patches outside the top-k carry no action-relevant information makes the mask equivalent to a relevance indicator; this assumption fails under contact-rich manipulation where peripheral context (table edge, gripper shadow) matters, in which case the mask discards causally necessary features. The gradient mask through α measures *credit assignment to selected features* because the assumption that unselected tokens do not influence the DiT makes the mask a faithful attribution; this assumption fails when the DiT's cross-attention attends to residual unselected tokens, in which case the mask overstates selection's causal role. The coverage regularizer `L_cov` measures *selection-set diversity* because the assumption that a collapsed selection set is suboptimal makes entropy a proxy for coverage; this assumption fails when the optimal selection is genuinely sparse and low-entropy, in which case L_cov fights the sparsemax objective.

## Contribution

["(1) TVG-VLA, a text-gated sparse visual feature selection module with a contrastive alignment objective that maintains instruction-feature identifiability under joint training of a 552B backbone and action DiT, replacing OTTER's frozen-encoder guarantee with an actively enforced training invariant.", "(2) A design principle — 'selection-as-invariant' — showing that semantic alignment in trainable-backbone VLAs can be carried by a ~2M-parameter gate rather than by freezing the encoder, decoupling alignment preservation from backbone scale.", '(3) A diagnostic protocol (gate-selection overlap with ground-truth referent masks and alignment-vs-action-loss trajectory tracking) for auditing whether alignment is preserved or silently degraded during fine-tuning.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | LIBERO-PRO + real-world precise manipulation suite | Matches ScaleVLA's three-capability comparison surface |
| Primary metric | Success rate split by instruction-ambiguity and contact-richness subsets | Detects where alignment vs. sparsity should matter |
| Secondary metric | Alignment drift: CKA between selected-feature/instruction similarity matrices at init vs. final | Directly tests whether the gate maintains an invariant that freezing merely inherits |
| Secondary metric | Attention-mass diagnostic on unselected tokens in the DiT's cross-attention | Tests the assumption that unselected tokens do not influence the DiT |
| Baseline | ScaleVLA (552B, no gate) | Tests whether gating adds anything over scale alone |
| Baseline | OTTER (frozen VLM + text-aware extraction) | Tests whether active alignment beats frozen-encoder alignment |
| Baseline | ScaleVLA + dense soft-attention gate | Tests whether hard sparsity is load-bearing |
| Baseline | ScaleVLA-small (2B backbone, no gate) | Scaled-down primary comparison for feasibility |
| Baseline | OTTER-small (2B frozen VLM + text-aware extraction) | Scaled-down primary comparison for feasibility |
| Ablation-of-ours | TVG-VLA without L_align (gate trained by L_action only) | Isolates the contrastive alignment contribution |
| Ablation-of-ours | TVG-VLA without L_cov (β=0) | Isolates the coverage regularizer contribution |
| Ablation-of-ours | TVG-VLA with frozen backbone + gate | Isolates whether the trainable backbone is actually necessary |

### Why this setup validates the claim

The central claim is that instruction-conditioned identifiability of selected visual features, actively enforced on a trainable 552B backbone, produces gains that neither raw scale (ScaleVLA) nor frozen-encoder alignment (OTTER) can produce. Each baseline tests one sub-claim. ScaleVLA tests whether the gate contributes beyond scale: if TVG-VLA only matches it, selection is decorative. OTTER tests whether *active* alignment beats alignment-as-a-byproduct-of-freezing: OTTER should be strong on novel-object generalization but weak where the moving backbone drifts from pretrained semantics, so a gain concentrated on instruction-ambiguous cases would confirm the mechanism claim rather than a generic capacity claim. The dense-attention baseline tests whether *hard sparsity* (the straight-through top-k) is load-bearing versus mere re-weighting — if dense attention matches, our stated design rationale is wrong. The no-L_align ablation is the sharpest test: it holds architecture fixed and removes only the enforced invariant, so any gap is attributable to L_align rather than the gate's parameter count. The frozen-backbone + gate variant isolates whether the trainable backbone is necessary: if it matches full TVG-VLA, the backbone need not be trainable and the alignment burden is entirely on the gate. The alignment-drift metric (CKA between selected-feature/instruction similarity matrices at init vs. final) directly measures whether the gate maintains an invariant that freezing merely inherits: we expect OTTER's CKA to be fixed by construction (frozen encoder) while TVG-VLA's CKA remains high despite a moving backbone, and the no-L_align ablation to show large drift. The attention-mass diagnostic on unselected tokens tests the assumption that the DiT's cross-attention assigns zero weight to unselected tokens; if residual attention mass is high, the gradient mask overstates selection's causal role. Success rate split by ambiguity and contact-richness is the right metric because our mechanism predicts a *non-uniform* gain pattern; a uniform gain would be consistent with generic improvement and would not discriminate our causal story from a capacity story. To address feasibility, the primary experiment uses 2B-7B backbones (ScaleVLA-small, OTTER-small, TVG-VLA-small) with the 552B run as an optional confirmation; we specify a compute budget of 4×A100-80GB for 2 weeks per 2B run, 3 random seeds, and a reduced baseline set that fits within 3-6 months on available hardware.

### Expected outcome and causal chain

**vs. ScaleVLA (552B, no gate)** — On a case where the instruction is lexically ambiguous ('pick up the red cup' amid several cups) and the 552B backbone's visual tokens are dominated by task-irrelevant patches, ScaleVLA feeds all 256 patches to the DiT, so instruction-relevant signal is diluted among distractors and the policy commits to the wrong object. Our gate selects ~64 instruction-aligned patches, so the DiT receives a cleaner conditioning set. We expect a noticeable success gap on the high-ambiguity subset but near-parity on unambiguous single-object scenes, where dilution is not the bottleneck. The alignment-drift metric should show TVG-VLA maintaining high CKA between selected-feature/instruction similarity matrices at init vs. final, while ScaleVLA (no gate) shows large drift.

**vs. OTTER (frozen VLM + text-aware extraction)** — On a case where the robot's embodiment-specific visual statistics (gripper shadow, table edge) diverge from web-pretraining, OTTER's frozen encoder cannot adapt, so its text-aware extraction selects features that are semantically aligned but physically miscalibrated, and the policy misjudges contact. Our trainable backbone updates through the selected tokens, so the encoder drifts toward task-relevant visual structure while L_align keeps selection instruction-faithful. We expect our gain concentrated on contact-sensitive manipulation (e.g. insertion) and on novel-object zero-shot, with OTTER competitive on standard LIBERO-PRO scenes. The frozen-backbone + gate ablation should match OTTER on contact-sensitive tasks, confirming that the trainable backbone is necessary for the gain.

**vs. ScaleVLA + dense soft-attention gate** — On a case where peripheral context is genuinely irrelevant but visually salient (a bright off-task object), the dense gate assigns diffuse non-zero weight across many patches, letting that distractor leak into the DiT and re-couple visual noise with action. Our hard top-k mask zeroes those tokens, so the DiT's conditioning is strictly instruction-relevant. We expect the dense variant to trail on distractor-heavy scenes while matching us on clean scenes, and the no-L_align ablation to trail specifically on ambiguous-instruction cases, isolating L_align's contribution. The attention-mass diagnostic on unselected tokens should show lower residual attention for the hard top-k than for the dense gate, confirming that the mask is load-bearing.

**vs. TVG-VLA without L_cov** — On a case where the gate initially selects wrong patches, the no-L_cov variant may collapse to a small, self-confirming selection set, reinforcing alignment of an already-mis-selected set. Our coverage regularizer prevents premature collapse, so we expect the no-L_cov variant to trail on ambiguous-instruction cases and to show lower entropy of α during training.

### What would falsify this idea

If TVG-VLA's gain over ScaleVLA is uniform across ambiguity and contact-richness subsets rather than concentrated where the predicted failure modes occur, the alignment-and-selection causal story is wrong and the improvement is generic capacity. Equally, if the no-L_align ablation matches full TVG-VLA, then L_align is not the mechanism carrying identifiability and the central claim collapses. If the alignment-drift metric shows TVG-VLA's CKA dropping to ScaleVLA's level despite the gate, the load-bearing assumption is false.

## References

1. ScaleVLA: Scaling Vision-Language-Action Models to 552 Billion Parameters
