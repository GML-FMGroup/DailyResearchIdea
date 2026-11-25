# Seam-Locked Blockwise Autoregressive Diffusion for High-Resolution Image Synthesis

## Motivation

Blockwise high-resolution synthesis such as Qwen-Image-HD generates each 1K tile conditioned only on high-resolution upper and left neighbors plus a compressed thumbnail, so the agreement between a newly generated tile and its already-generated right and lower neighbors is never constrained during generation and is left to be learned empirically from data. This is a recurring structural failure across the field, not a local quirk: Infinity's unified bitwise token stream (E1) also provides no invariant forcing adjacent regions to agree at their seams, because a single stream ordering does not couple a token to regions generated later in a different spatial order. The root cause is that seam coherence is treated as an emergent property of training rather than as an explicit constraint on the boundary variables shared by adjacent tiles, so residuals concentrate exactly on the shared boundaries where no conditioning signal exists.

## Key Insight

A tile boundary is a variable shared by two adjacent predictions, so enforcing equality of the boundary values under the shared-canvas coordinate map turns seam coherence from an empirical byproduct into a constraint that is satisfied by construction at every generation step.

## Method

**What it is.** We introduce SLAB (Seam-Locked Autoregressive Blocks), a blockwise autoregressive diffusion procedure that augments Qwen-Image-HD's tile prediction with a differentiable seam-consistency operator. Inputs are the pretrained Qwen-Image 2.0 backbone, the shared-canvas coordinate map, and the partially generated canvas; outputs are tiles whose shared boundaries provably agree with already-generated neighbors up to a tolerance.

**How it works.**

```pseudocode
# canvas C, tile size 1024, stride 1024, overlap band w = 32 px
# shared-canvas coordinate map Phi maps every pixel to global coords
# backbone latent space Z (Qwen-Image 2.0 VAE latent, 8x downsampled)
# seam refinement operates in Z, not raw pixels
for each tile position (i, j) in row-major order:
    cond = {upper tile C[i-1,j], left tile C[i,j-1], thumbnail T(C[:i,:j])}
    z0  = backbone.encode(backbone.predict_tile(cond, Phi))  # 1024x1024 proposal -> latent

    # --- seam-consistency operator S (on-manifold projection) ---
    # extract boundary strips in shared-canvas coords, in latent space
    B_top  = z0[0:w//8, :]        ; N_top  = encode(C[i-1,j])[-w//8:, :]
    B_left = z0[:, 0:w//8]        ; N_left = encode(C[i,j-1])[:, -w//8:]
    # right/lower neighbors do not yet exist at this step;
    # they are locked when the next tile is generated (see lock step)

    # boundary-agreement loss, applied as test-time refinement
    # K=8 latent-space gradient steps, eta=0.05
    for k in 1..K (K = 8):
        L_seam = || B_top  - N_top  ||_1
               + || B_left - N_left ||_1
        z0 = z0 - eta * grad_{z0} L_seam     # eta = 0.05, gradient descent on latents

    # on-manifold projection: re-denoise z0 for 4 steps with the neighbor strips
    # as hard conditioning (inpainting-style mask on the boundary band)
    z0 = backbone.denoise(z0, cond, mask=boundary_band, steps=4)

    C[i,j] = backbone.decode(z0)
    # lock step: when tile (i, j+1) is generated, its left strip is forced
    # to equal C[i,j]'s right strip by the same operator, so the boundary
    # between (i,j) and (i,j+1) is pinned by construction

return C
```

The operator is applied at every tile, and the lock step means each interior boundary is pinned exactly once, by the later of its two adjacent tiles, using the earlier tile as the reference. The same operator is also used as a training-time auxiliary loss on the local tile objective, with weight lambda = 0.1, so the backbone learns to propose tiles whose boundaries are already close to their neighbors before refinement.

**Why this design.** We chose a latent-space differentiable seam operator with a re-denoising projection over a learned discriminator on seams because the discriminator would reintroduce the empirical dependence we are trying to remove, accepting the cost that latent-space refinement is slower per tile than a feed-forward head. We chose to pin each boundary once, by the later tile, over a symmetric joint optimization of both tiles because joint optimization would require re-running earlier tiles and break the autoregressive order that gives Qwen-Image-HD its compute advantage, accepting the cost that the earlier tile is treated as fixed reference and cannot be corrected. We chose a fixed overlap band of 32 px over a learned adaptive band because a fixed band makes the pinned set of pixels exactly the shared-canvas boundary set, accepting the cost that very high-frequency detail within the band is smoothed. A domain expert would not describe this as a variant of Qwen-Image-HD's blockwise conditioning: Qwen-Image-HD conditions on neighbors but never constrains the output boundary, whereas SLAB adds a constraint that is enforced by construction, and it is not a variant of seam-blending post-processing because the constraint is applied inside the autoregressive loop and propagated through the lock step.

**Why this measures what we claim.** The boundary-agreement loss `L_seam` measures seam coherence because it is the L1 distance between the boundary values of two adjacent tiles expressed in the same shared-canvas coordinate frame; this equivalence assumes the shared-canvas map `Phi` is exact, and it fails when the map has sub-pixel rounding error at tile corners, in which case `L_seam` reflects coordinate misalignment rather than visual discontinuity. The lock step measures structural invariance because pinning a boundary once makes the two adjacent tiles share that variable exactly; this assumes the later tile's refinement does not move the pinned strip, and it fails when refinement gradients leak into the pinned strip, in which case the invariance is approximate rather than exact. The training-time auxiliary loss measures whether the backbone's proposals are already seam-consistent; this assumes the local tile objective is representative of full-canvas generation, and it fails when tile statistics differ from canvas statistics, in which case the loss reflects local rather than global coherence. Seam-band LPIPS measures seam coherence because it computes LPIPS only on the 32-px band centered on pinned boundaries; this assumes the band contains only boundary-discontinuity signal and not general texture-fidelity signal, and it fails when high-frequency texture inside the band dominates LPIPS independent of seam alignment, in which case seam-band LPIPS reflects texture fidelity rather than seam coherence.

## Contribution

(1) A seam-consistency operator embedded in the blockwise autoregressive loop, with a lock step that pins each interior boundary exactly once, turning seam agreement into a structural invariant of generation rather than an emergent property. (2) A design principle: boundary variables shared by adjacent tiles should be constrained at generation time in the shared-canvas frame, and the constraint can be enforced by pixel-space gradient refinement without retraining the backbone. (3) An auxiliary training loss that aligns the backbone's tile proposals with the seam constraint, reducing the refinement burden at inference.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Qwen-Image-Bench at 4K | Matches source benchmark for 4K synthesis |
| Primary metric | Seam-band LPIPS at pinned boundaries | Directly measures boundary coherence claimed |
| Control metric | Interior-band LPIPS at matched distance from seams | Separates seam coherence from general texture fidelity |
| Baseline | Qwen-Image-HD (unmodified) | Isolates effect of seam operator |
| Baseline | FLUX-3-Image-Klein direct 4K | Tests against strong direct-generation alternative |
| Baseline | Full-image 4K SFT of Qwen-Image 2.0 | Tests against full-canvas fine-tuning |
| Ablation-of-ours | SLAB without lock step | Tests whether pinning causes the gain |
| Ablation-of-ours | SLAB without latent re-denoising projection (pixel-space L1 only) | Tests whether on-manifold projection is necessary |

### Why this setup validates the claim

The central claim is that enforcing boundary agreement by construction inside the autoregressive loop reduces seam artifacts beyond what neighbor-conditioning alone achieves. Qwen-Image-HD (unmodified) is the key baseline because it shares the exact backbone, shared-canvas map, and blockwise order, so any gain isolates the seam operator and lock step rather than architecture or data. FLUX-3-Image-Klein direct 4K tests whether the blockwise-plus-constraint route beats a strong monolithic generator on the same benchmark, and full-image 4K SFT tests whether local constrained training outperforms naive full-canvas fine-tuning. The ablation without the lock step is the discriminator: if the gain is caused by pinning boundaries once, removing the lock should collapse the seam-band metric toward the unmodified baseline even though the same L_seam refinement remains. The ablation without latent re-denoising projection tests whether the on-manifold projection is necessary to avoid off-manifold artifacts from pixel-space L1 descent. Seam-band LPIPS is the right primary metric because it concentrates measurement exactly where the mechanism operates; a whole-image metric would dilute a boundary-localized effect and could not distinguish seam coherence from general fidelity. The interior-band LPIPS control at matched distance from seams is included to verify that any seam-band gain is not merely a general texture-fidelity improvement, addressing the failure mode where high-frequency texture inside the band dominates LPIPS independent of seam alignment.

### Expected outcome and causal chain

**vs. Qwen-Image-HD (unmodified)** — On a case where a tile's upper neighbor ends in a high-frequency texture that continues across the shared boundary, the unmodified baseline proposes a tile whose boundary strip is only softly conditioned on the neighbor, so the two strips disagree and a visible seam appears because nothing constrains the output boundary. Our method instead runs the differentiable seam operator inside the loop and pins the strip via the lock step, so the boundary is forced to agree with the earlier tile. We expect a clear seam-band LPIPS reduction concentrated at pinned interior boundaries, with near-parity on interior-band LPIPS where the mechanism does not act.

**vs. FLUX-3-Image-Klein direct 4K** — On a case requiring long-range structural continuity across a 4K canvas, direct generation must commit the entire canvas in one pass, so local texture statistics can drift between distant regions and the model exhibits repetitive striping or inconsistent structure because it has no tiling reference to anchor local detail. Our method instead generates each tile against explicit upper, left, and thumbnail conditions with locked seams, so local structure is anchored to already-generated content. We expect SLAB to show its advantage on structure-preservation and seam metrics at 4K while direct generation may retain an edge on global aesthetic scores, giving a split signal rather than uniform dominance.

**vs. Full-image 4K SFT of Qwen-Image 2.0** — On a case with limited training compute, full-image 4K fine-tuning must fit long image sequences, so it produces unnatural repetitive stripes in local regions because gradient signal is spread across the whole canvas. Our method instead trains on local tiles with the seam auxiliary loss and generates blockwise with pinned boundaries, so each prediction covers a manageable 1K target with explicit neighbor references. We expect SLAB to match or exceed full-image SFT on seam-band and structure metrics at lower training cost, with the gap largest on high-frequency textured regions.

### What would falsify this idea

If the seam-band LPIPS gain over unmodified Qwen-Image-HD is uniform across interior and non-boundary regions rather than concentrated at pinned boundaries, or if removing the lock step leaves the seam-band metric unchanged, or if the interior-band LPIPS control shows a comparable gain, then the claimed causal mechanism is not what produces the improvement and the central claim is wrong.

## References

1. Qwen-Image-HD: Autoregressive Diffusion for High Definition Image Generation
