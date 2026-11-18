# Fixed-Point Dynamic/Static Decomposition for 4D Scene Reconstruction

## Motivation

Existing 4D reconstruction methods treat the dynamic/static split as a one-shot prediction: π^4 predicts moving-object masks and static geometry jointly in a single forward pass, and CAT4D optimizes a deformable Gaussian representation without ever requiring the static geometry to be invariant to mask refinement. This structural one-shotness means the reported static geometry is a function of a single mask estimate rather than of a self-consistent mask–geometry pair, so mask errors propagate silently into the reconstruction and cannot be corrected by re-estimation. The field's trajectory toward feed-forward foundation models (π^4, Flux4D) amplifies this because the mask and geometry are produced by coupled branches whose outputs are never tested for mutual consistency.

## Key Insight

If static-region geometry is truly static, it must be a fixed point of the alternating map (mask update → static-geometry update), so requiring the alternation to converge makes the reported static geometry invariant to mask initialization and exposes—rather than hides—mask–geometry inconsistencies that one-shot prediction cannot detect.

## Method

**What it is.** We introduce FIX4D (Fixed-Point 4D decomposition), a wrapper that takes any base 4D reconstructor providing a moving-object mask head and a static-geometry head (we instantiate it on π^4) and replaces its one-shot output with the converged limit of an alternating mask–geometry refinement. Inputs: a video sequence or unordered RGB views. Outputs: a converged moving-object mask sequence M*, converged static geometry G* (dense depth + camera poses + metric scale), and a convergence certificate (residual trajectory, iteration count, and a multi-initialization stability report).

**Load-bearing assumption (stated explicitly).** FIX4D assumes the alternating map (mask update → static-geometry update) has a unique, attracting fixed point that the damped iteration converges to, so that the converged static geometry is invariant to mask initialization and the residual trajectory is a valid measure of mask–geometry inconsistency. This assumption is not guaranteed: a moving object can be explained either as static-with-camera-motion or as moving-with-static-camera, and the map may therefore have multiple fixed points. FIX4D does not assume uniqueness away; it tests for it via the multi-initialization stability report (below) and reports the spread of converged G* across basins. When the spread exceeds a threshold, the convergence certificate is marked 'basin-dependent' and the residual trajectory is interpreted as reflecting the chosen basin rather than mask error alone.

**How it works.**

```pseudocode
# Inputs: RGB observations X; base model f (π^4 backbone);
#         max_iters=8, tol=1e-3, damping=0.5, n_init=4, perturb_frac=0.05,
#         basin_tol=1e-2
# Outputs: M*, G*, residual history r[0..K], stability report S

# --- single-run alternation (the core loop) ---
def run_alternation(X, M_init):
    M_0  = M_init
    G_0  = f.geometry_head(X, mask=M_0)
    r[0] = static_residual(G_0, M_0)
    for k = 1 .. max_iters:
        # (a) mask update: re-estimate moving mask using CURRENT static geometry
        M_k = f.mask_head(X, static_geometry=G_{k-1})
        # (b) damped mask step to avoid oscillation
        M_k = (1 - damping) * M_{k-1} + damping * M_k
        # (c) static-geometry update: re-fit geometry on pixels where M_k==static
        G_k = f.geometry_head(X, mask=M_k)
        r[k] = static_residual(G_k, M_k)
        if r[k] < tol and ||G_k - G_{k-1}||_F / ||G_{k-1}||_F < tol:
            break
    return M_K, G_K, r[0..K]

# --- main entry: multi-initialization stability test ---
M_base = f.mask_head(X)                       # base one-shot mask
runs = []
runs.append(run_alternation(X, M_base))       # run 0: base initialization
for i = 1 .. n_init-1:                        # runs 1..n_init-1: perturbed inits
    M_pert = perturb_mask(M_base, frac=perturb_frac)   # flip a random 5% of pixels
    runs.append(run_alternation(X, M_pert))

# stability report S: spread of converged geometry across runs
G_star_list = [run.G_K for run in runs]
S.spread = max_{i,j} ||G_star_i - G_star_j||_F / ||G_star_j||_F
S.basin_dependent = (S.spread > basin_tol)
S.residual_trajectories = [run.r for run in runs]
S.mask_flip_rates = [mean(M_k != M_{k-1}) for each run, each k]

# report the run with lowest final residual as the primary output
best = argmin_run run.r[-1]
return best.M_K, best.G_K, best.r[0..K], S
```

`static_residual(G, M)` is the photometric + geometric reprojection error of G restricted to pixels where M==static, normalized by the number of static pixels: this is the quantity the fixed point must drive to its floor. The damping factor 0.5 is fixed (not tuned per scene) to smooth the mask iterate; we do not claim it guarantees contraction on the mask simplex, because the mask head is a classifier whose argmax can flip discretely at decision boundaries. Instead, we report per-iteration mask-flip rates and the multi-initialization spread S.spread to empirically characterize whether the damped iterate stabilizes and whether the converged geometry is basin-dependent. We choose damping rather than a learned step because a learned step would reintroduce the one-shot prediction we are trying to remove.

**Why this design.** We chose an *alternating fixed-point iteration with a fixed damping coefficient* over a single feed-forward pass because the base model's mask and geometry heads are trained jointly and therefore mutually consistent only at the training distribution; at test time, when the mask is wrong, the geometry head has no mechanism to signal the error back. The alternation makes the mask a function of the current geometry and vice versa, accepting the cost of K× inference (K≈3–8 in practice) to gain a consistency check that one-shot π^4 and CAT4D lack. We chose *damped* updates over plain alternation because the mask head is a classifier whose argmax can flip discretely between iterations, producing limit cycles; damping smooths the iterate, accepting slower convergence (more iterations) to favor a stable limit over oscillation. We chose to *report the residual trajectory and multi-initialization spread* as part of the output rather than only the final reconstruction because the residual trajectory and spread are the only observables that distinguish a genuine fixed point from a stalled or basin-dependent iterate; accepting the cost of exposing a diagnostic that a one-shot method would hide. A domain expert would not describe FIX4D as a variant of CAT4D's deformable-Gaussian optimization or π^4's feed-forward coupling: those methods never evaluate the map at a second iterate, so the fixed-point property is not even defined for them.

**Why it measures what we claim.** The computational quantity `static_residual(G_k, M_k)` measures *mask–geometry consistency* because we assume that if the static mask is correct, the static geometry fitted to it must reproject with near-zero error onto the static pixels (the standard static-scene assumption); this assumption fails when the scene contains unmodeled non-rigid deformation (e.g., cloth, water) that is neither static nor a coherent moving object, in which case the residual reflects unmodeled deformation rather than mask error. The quantity `||G_k - G_{k-1}||_F` measures *fixed-point stability of static geometry* because we assume that a genuine fixed point implies the geometry iterate contracts to a unique limit independent of initialization; this assumption fails when the alternating map has multiple fixed points (e.g., a moving object can be explained either as static-with-camera-motion or as moving-with-static-camera), in which case the norm reflects which basin the initialization fell into rather than convergence to the correct decomposition. The multi-initialization spread `S.spread` measures *basin-dependence of the converged decomposition* because we assume that if the fixed point is unique and attracting, perturbed initial masks converge to the same G*; this assumption fails when the map has multiple attractors, in which case S.spread is large and the convergence certificate is marked basin-dependent. The iteration count K measures *how far the one-shot prediction was from self-consistency* because we assume the base model's initial mask is a perturbation of the fixed point; this assumption fails when the base model is systematically biased (e.g., consistently under-segments thin structures), in which case K reflects the bias magnitude rather than random initialization noise.

## Contribution

['A fixed-point formulation of dynamic/static decomposition (FIX4D) that replaces one-shot mask–geometry prediction with a damped alternating iteration and reports a convergence certificate, applicable as a wrapper around any base reconstructor exposing a mask head and a geometry head.', 'The empirical finding that static geometry under one-shot prediction is sensitive to mask initialization, and that the converged fixed point reduces this sensitivity—establishing mask–geometry consistency as a measurable property distinct from per-frame reconstruction error.', "A stability protocol that perturbs the converged mask and extends the input sequence to test whether the fixed point is preserved, providing a diagnostic for whether a 4D reconstructor's static geometry is genuinely static or merely plausible at one iterate."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | DyCheck + outdoor driving sequences | Motion-rich, extreme viewpoint; stresses mask–geometry coupling |
| Primary metric | Static-region reprojection residual after convergence | Directly measures the consistency the fixed point claims |
| Secondary metric | Multi-initialization spread S.spread (n_init=4, perturb_frac=0.05) | Tests basin-dependence of converged geometry |
| Secondary metric | Per-iteration mask-flip rate | Substantiates oscillation/damping argument |
| Baseline | π^4 one-shot | Tests whether iteration beats single feed-forward pass |
| Baseline | CAT4D | Tests dynamic-Gaussian optimization against iterative decomposition |
| Baseline | ViDAR | Tests diffusion-supervised reconstruction against our consistency loop |
| Ablation-of-ours | FIX4D without damping (plain alternation) | Isolates damping's role in reaching a stable limit |
| Ablation-of-ours | FIX4D single-init (n_init=1) | Isolates the multi-init stability test's contribution |

### Why this setup validates the claim

The central claim is that a one-shot mask–geometry prediction is not self-consistent, and that alternating to a fixed point recovers consistency that one-shot π^4, CAT4D, and ViDAR cannot reach. The dataset choice matters because DyCheck's extreme viewpoint variation and the driving sequences both create the exact condition under which mask and geometry heads disagree: large apparent motion that can be explained ambiguously as object motion or camera motion. π^4 one-shot tests the sub-claim that a single pass leaves residual on the table; CAT4D and ViDAR test whether the gain is specific to our alternation or merely a byproduct of any iterative/optimization-based dynamic reconstruction. The undamped ablation is the sharpest test: if the gain comes from damping-driven stabilization rather than iteration itself, the ablation should show oscillation and a worse or non-converging residual, confirming the mechanism. The multi-initialization spread S.spread directly tests the load-bearing assumption: if the map has a unique attracting fixed point, perturbed initial masks should converge to the same G* (small S.spread); if S.spread is large, the convergence certificate is marked basin-dependent and the residual trajectory cannot be interpreted as mask error alone. The primary metric is the right detector because it is exactly the quantity the fixed point is defined to minimize; a method that improves novel-view aesthetics but not static-region residual has not demonstrated the claimed consistency.

### Expected outcome and causal chain

**vs. π^4 one-shot** — On a case where a thin moving structure (a pole, a limb) sits against a distant static background, one-shot π^4 produces a mask that bleeds the structure into the static set because its mask and geometry heads are jointly trained and mutually consistent only near the training distribution, with no error signal fed back. Our method instead re-estimates the mask conditioned on the current geometry, so the misassigned pixels surface as elevated static residual and get pulled out on the next iteration. We expect a noticeable residual drop on motion-rich, thin-structure subsets but near-parity on largely static scenes, where the one-shot output is already close to a fixed point. We also expect S.spread to be small on these thin-structure cases, indicating a unique attractor, whereas on scenes with large object-vs-camera ambiguity S.spread may be large and the certificate marked basin-dependent.

**vs. CAT4D** — On a case where a monocular video contains a moving object viewed from a novel angle, CAT4D's deformable-Gaussian optimization fits the synthesized multi-view signal but has no explicit mask–geometry consistency check, so mask errors are absorbed into deformation rather than corrected. Our method instead treats the mask as a variable in an alternating map and drives static residual toward its floor, so mask error cannot hide inside deformation. We expect a modest but consistent residual gap on dynamic-region subsets, with the gap concentrated where the moving object is large and the static background is thin. We expect the multi-init stability report to reveal basin-dependence precisely on these large-object cases, where the object-vs-camera ambiguity is strongest.

**vs. ViDAR** — On a case where diffusion-based pseudo multi-view supervision is spatio-temporally inconsistent, ViDAR's diffusion-aware loss mitigates but does not eliminate the inconsistency, leaving static-region residual tied to the diffusion prior's noise. Our method instead uses the reconstructor's own geometry head as the consistency oracle, so the residual floor is set by the data rather than the generative prior. We expect ViDAR to look competitive on appearance-heavy metrics but to trail on static-region residual, especially on scenes with extreme viewpoint change. We expect our multi-init spread to be smaller than ViDAR's run-to-run variance under different diffusion seeds, indicating that our fixed point is more stable than the generative prior's output.

### What would falsify this idea

If the residual trajectory is essentially flat from iteration 0 (r[0] already at its floor) on the motion-rich subsets where the base model's one-shot mask is predicted to be wrong, then the one-shot prediction was already self-consistent and the central claim collapses. Equally, if the undamped ablation converges as well as the damped version with no oscillation, then damping is not doing the work we attribute to it and the gain, if any, comes from something other than the fixed-point mechanism. Finally, if the multi-initialization spread S.spread is large (basin-dependent) on the majority of scenes, then the converged geometry is not invariant to mask initialization and the residual trajectory cannot be interpreted as mask–geometry inconsistency.

## References

1. π^4: A Foundation Model for 4D Scene Geometry and Motion
