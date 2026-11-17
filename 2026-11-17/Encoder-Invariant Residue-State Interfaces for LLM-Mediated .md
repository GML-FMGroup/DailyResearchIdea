# Encoder-Invariant Residue-State Interfaces for LLM-Mediated Protein Folding

## Motivation

Current LLM-mediated folding systems such as Calpha couple a language model's residue states to a structure head through a fixed ESM2 feature channel, so the reported folding competence cannot be attributed to language reasoning over protein sequence rather than to the specific encoder's inductive bias; this is the same identifiability gap that DiMA sidesteps by demonstrating architecture-level generalization across encoders without ever testing whether a downstream controlled property is encoder-invariant. The field is converging on PLM-mediated design pipelines (steering in E1/E7, concept bottlenecks in E2/E8, latent diffusion in E5/E9) where each downstream head inherits one encoder's representation, and no existing work audits whether the controlled property survives a backbone swap.

## Key Insight

If a structure head's inputs are constrained to a shared residue-state subspace that is statistically sufficient for the folding property across independently trained PLM backbones, then folding accuracy becomes invariant to encoder identity rather than an artifact of ESM2's feature geometry.

## Method

**What it is.** We introduce **XANTHUS** (eXchAngeable residue-state iNterface for sTructural Head Under Swapping), an encoder-agnostic residue-state adapter that maps residue states from any of several PLM backbones (ESM-2, ESMc, SaProt) into a shared latent channel consumed by a frozen SimpleFold structure head, paired with an encoder-invariance audit that measures whether the folding property is preserved under backbone swap. The load-bearing assumption is stated explicitly: *a single frozen SimpleFold head, fed only through a shared 512-d linear-projected, layer-normalized residue-state channel trained with an L2 alignment penalty, can recover folding accuracy from ESM-2, ESMc, and SaProt states that is statistically invariant to which backbone produced them — i.e., the folding-relevant information in these three encoders' residue states is linearly mappable into one common subspace without loss of the per-encoder signal the head needs.* We test this assumption directly via a non-degeneracy constraint and a competence-matched audit (Phases 2b and 3b below).

**How it works.**

```pseudocode
# Phase 1: Shared residue-state subspace construction
for backbone b in {ESM-2, ESMc, SaProt}:
    H_b = backbone_b(sequence)               # per-residue states, dim d_b (ESM-2: 1280, ESMc: 960, SaProt: 1280)
    P_b = linear_proj_b(H_b)                 # dim d_b -> d_shared=512
    Z_b = layer_norm(P_b)                    # shared residue-state channel

# Phase 2: Adapter training (frozen SimpleFold head)
for each (sequence, coords) in train_set:
    Z = Z_{b(sequence)}                      # one backbone per sequence, round-robin
    pred = SimpleFold_frozen(Z)              # head weights frozen; input dim contract verified in Phase 0
    loss = TM_soft(pred, coords) + lambda_align * ||Z_b - Z_{b'}||_2^2 + lambda_ret * L_ret
    update linear_proj_b only                # lambda_align = 0.1, lambda_ret = 0.5

# Phase 2b: Information-retention constraint (prevents collapse)
# L_ret = ||H_b - Dec_b(Z_b)||_2^2, where Dec_b is a per-backbone linear decoder (d_shared -> d_b)
# trained jointly with linear_proj_b (same optimizer, same round-robin batches).
# A held-out folding-relevant probe (linear ridge regression from Z_b to per-residue relative solvent accessibility, trained on 512 sequences) must achieve R^2 >= 0.6 for each backbone; otherwise training restarts with lambda_ret increased by 0.1 (max 3 restarts).

# Phase 3: Encoder-invariance audit (no further training)
for each test sequence s:
    preds = { SimpleFold_frozen(Z_b(s)) : b in backbones }
    report TM(preds[b], coords) for each b
    report Var_b[ TM(preds[b], coords) ]    # invariance score
    report pairwise RMSD between preds[b]   # prediction stability
    report cross-backbone state diversity: mean pairwise cosine distance between Z_b(s) across b (constant-collapse detector; threshold < 0.01 flags collapse)
    report probe R^2 per backbone on held-out set (non-degeneracy check)

# Phase 3b: Competence-matched stratification
# For each test sequence, compute per-backbone baseline TM (using each backbone's own unaligned adapter trained in Phase 2 with lambda_align=0).
# Stratify Var_b[TM] into quartiles of per-backbone baseline TM spread; report Var_b[TM] within each quartile.
# Invariance claim holds only if Var_b[TM] is low in the top quartile (matched competence) and the residual variance in lower quartiles is attributable to competence gaps.

# Phase 4: Falsification control
repeat Phases 1-3 with lambda_align = 0 and with shuffled Z_b assignments
# Additionally: repeat Phase 2 with the SimpleFold head lightly fine-tuned (last 2 layers unfrozen, lr=1e-5) per backbone as a control to isolate alignment from head scale-sensitivity.
```

**Why this design.** We chose a frozen SimpleFold head with per-backbone linear projections over retraining the head jointly with each backbone because joint training would let the head absorb encoder-specific idiosyncrasies, which is exactly the confound Calpha's fixed ESM2 channel already introduces; freezing the head forces the adapter to carry the entire encoder-agnostic burden, accepting the cost that the shared channel must be expressive enough for three very different representation geometries. We chose a shared 512-d subspace with an explicit alignment penalty (lambda_align = 0.1) over unconstrained per-backbone projections because without alignment the audit would only test head robustness to input distribution shift, not encoder invariance of the property; the cost is that alignment may collapse useful backbone-specific signal, which we monitor via a held-out probe and a reconstruction decoder (lambda_ret = 0.5) that penalizes loss of per-backbone information. We chose round-robin single-backbone training over ensembling all three backbones per sequence because ensembling leaks cross-backbone information into the adapter and would make the audit circular; the cost is noisier per-step gradients. A reviewer would not describe XANTHUS as 'DiMA plus Calpha' because DiMA's encoder-agnosticism is architectural (same diffusion architecture, different encoders, no shared channel) whereas XANTHUS's contribution is the identifiability audit of a controlled downstream property under encoder swap, which DiMA does not perform and Calpha's fixed-channel design cannot express.

**Why it measures what we claim.** The variance of TM-score across backbones (`Var_b[TM]`) measures encoder invariance of the folding property because it assumes that the three PLM backbones span the relevant representation space for folding, so any residual variance is attributable to encoder identity rather than to sequence difficulty; this assumption fails when a backbone is systematically weaker on a protein family (e.g., SaProt on disordered regions), in which case the variance reflects backbone competence rather than interface non-invariance — we therefore report Var_b[TM] stratified by per-backbone baseline TM quartiles (Phase 3b) so that competence-matched subsets isolate invariance. The pairwise RMSD between predictions measures prediction stability because it assumes that coordinate-space agreement is a sufficient proxy for shared structural commitment; it fails when two backbones produce different but equally valid folds, in which case RMSD reflects conformational degeneracy rather than interface failure. The alignment penalty `||Z_b - Z_{b'}||^2` measures shared-channel sufficiency because it assumes that L2 distance in the projected space tracks semantic equivalence of residue states; it fails when projections collapse to a constant, in which case the penalty reflects trivial alignment rather than sufficiency — we therefore report cross-backbone state diversity (mean pairwise cosine distance) and a reconstruction loss `L_ret` to detect and penalize collapse. The frozen-head ablation with `lambda_align = 0` measures whether alignment is causally necessary for invariance because it assumes that frozen head weights isolate the adapter's contribution; it fails when the frozen head is itself sensitive to input scale, in which case the ablation reflects head brittleness rather than adapter necessity — we therefore include a lightly fine-tuned head control (last 2 layers, lr=1e-5) to separate alignment from head scale-sensitivity.

## Contribution

(1) XANTHUS, an encoder-agnostic residue-state adapter that maps ESM-2, ESMc, and SaProt states into a shared 512-d channel consumed by a frozen structure head, with a per-backbone projection and alignment penalty. (2) An encoder-invariance audit protocol that quantifies whether a downstream structural property is preserved under backbone swap, providing a design principle: controlled properties in PLM-mediated pipelines should be reported with an invariance score, not a single-backbone number. (3) A falsification control showing whether alignment is causally necessary for invariance, establishing that encoder-agnosticism requires an explicit shared-subspace constraint rather than architectural analogy alone.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | FoldBench-334 + CASP16 domains | Matches Calpha scoring; diverse folds stress encoder swap |
| Primary metric | Var_b[TM-score] across backbones, stratified by per-backbone baseline TM quartiles | Directly quantifies encoder-invariance of folding property while controlling for backbone competence |
| Secondary metric | Cross-backbone state diversity (mean pairwise cosine distance between Z_b) and probe R^2 per backbone | Detects degenerate collapse of the shared channel |
| Baseline | Calpha fixed ESM2 channel | Fixed-encoder reference for the same head |
| Baseline | Per-backbone unaligned adapters (lambda_align = 0) | Tests whether alignment penalty is causally needed |
| Baseline | Concatenation of all three backbones' states into the head (encoder-agnostic control) | Cheap test of whether shared channel adds value over naive fusion |
| Ablation-of-ours | XANTHUS with lambda_align = 0 | Isolates alignment contribution to invariance |
| Ablation-of-ours | XANTHUS with lambda_ret = 0 | Isolates information-retention constraint against collapse |
| Control | XANTHUS with lightly fine-tuned SimpleFold head (last 2 layers, lr=1e-5) | Separates alignment effect from head scale-sensitivity |

### Why this setup validates the claim

The central claim is that a shared residue-state channel makes the folding property identifiable under backbone swap, so the decisive test is not average TM-score but the dispersion of TM-score across ESM-2, ESMc, and SaProt on the same sequences, conditional on non-degenerate state diversity and competence matching. Calpha's fixed ESM2 channel establishes the reference point: it can only express one encoder's view, so it cannot even produce the variance statistic that defines our claim. The unaligned per-backbone adapter baseline tests whether the alignment penalty is what produces invariance rather than mere projection capacity, and the concatenation baseline tests whether a shared channel adds value over naive fusion of all three backbones. The lambda_align = 0 and lambda_ret = 0 ablations are the causal tests: if removing alignment does not degrade invariance, the shared subspace is doing no work; if removing retention collapses state diversity without hurting Var_b[TM], the invariance is trivial. The competence-matched stratification (Phase 3b) ensures that low Var_b[TM] is not an artifact of a systematically weaker backbone inflating variance on hard folds. Var_b[TM] is the right primary metric because it is exactly the quantity the claim asserts should shrink, and FoldBench-334 plus CASP16 provide the fold diversity needed for variance to be meaningful rather than noise.

### Expected outcome and causal chain

**vs. Calpha fixed ESM2 channel** — On a sequence where ESM-2's residue states underrepresent a family (e.g., a fold ESMc encodes more sharply), the fixed-channel baseline produces a single prediction with no way to expose encoder dependence, because its architecture cannot express an alternative encoder view. Our method instead routes all three backbones through the shared channel and folds each, because the adapter is trained to map heterogeneous states into one latent space, so we expect a low Var_b[TM] on well-represented folds but a visible spread on the family where ESM-2 is weakest, revealing where the fixed channel was silently biased.

**vs. Per-backbone unaligned adapters** — On a sequence whose backbones produce differently scaled residue states, the unaligned baseline produces predictions that drift with encoder identity because each projection is free to adopt its own geometry, so TM dispersion stays near the raw cross-encoder gap. Our method instead pulls Z_b toward a common subspace via the alignment penalty, because the L2 term couples the projections during round-robin training, so we expect a markedly tighter Var_b[TM] than the unaligned baseline, with the gap concentrated on sequences where raw encoder disagreement was largest.

**vs. Concatenation of all three backbones' states into the head** — On a sequence where one backbone is noisy, the concatenation baseline produces a prediction dominated by the noisiest channel because the head receives all three states without a shared bottleneck, so TM dispersion reflects the worst backbone rather than a controlled property. Our method instead constrains all backbones into one channel before a single frozen head, because the head cannot absorb encoder idiosyncrasies, so we expect lower Var_b[TM] and lower pairwise RMSD than concatenation, with the gap concentrated on sequences where one backbone's raw states are outliers.

### What would falsify this idea

If Var_b[TM] under XANTHUS is statistically indistinguishable from the unaligned-adapter baseline and from the lambda_align = 0 ablation across all fold subsets, the shared-channel alignment is not causally responsible for invariance and the central claim fails. Equally fatal: if the invariance gain is uniform across subsets rather than concentrated where raw cross-encoder disagreement is largest, or if low Var_b[TM] coincides with cross-backbone state diversity below 0.01 (collapse), the effect is head smoothing or trivial alignment rather than encoder-agnostic residue-state transfer.

## References

1. Calpha: Toward Native Biomolecular Intelligence in a General-Purpose Language Model
