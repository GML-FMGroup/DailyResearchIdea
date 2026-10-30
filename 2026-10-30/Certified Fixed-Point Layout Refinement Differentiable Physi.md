# Certified Fixed-Point Layout Refinement: Differentiable Physical-Feasibility Projection for LLM-Generated 3D Scenes

## Motivation

Across both lineages of LLM/VLM-driven 3D scene synthesis — DirectLayout's staged numerical layout generation and the agentic spatially-contextualized VLM pipeline — spatial correctness is delegated to an implicit reasoner that emits structured poses in one shot, with no mechanism that ever checks the emitted layout against object penetration, support, or metric consistency. Because the model is never asked to certify feasibility and never receives a correction signal grounded in physics, geometric errors (floating objects, interpenetrating furniture, off-metric scales) propagate silently through BEV-to-3D lifting and refinement, and the pipeline cannot distinguish a plausible-looking layout from a physically realizable one. The structural root cause is that generation and feasibility verification live in disjoint modules with no closed loop between them.

## Key Insight

A numerical layout is only trustworthy when it is a fixed point of a differentiable physical-feasibility residual — i.e., the LLM's proposal is treated as an initial hypothesis that is projected onto the feasible manifold and re-certified, so correctness is defined by constraint satisfaction rather than by the reasoner's confidence.

## Method

**What it is.** FIXP (Feasibility-fixed-point Layout Projection) takes an LLM/VLM-proposed numerical layout $L_0 = \{(o_i, p_i, q_i, s_i)\}$ (object identity, translation, orientation, scale) and a differentiable physical-feasibility residual $R(L)$ composed of penetration, support, and metric-consistency terms, and returns a certified layout $L^\star$ with $R_{\text{exact}}(L^\star) \le \epsilon$ or an explicit infeasibility report. It replaces one-shot trust with closed-loop constrained projection. **Load-bearing assumption (stated explicitly):** FIXP assumes the LLM proposal $L_0$ is *near-feasible* — i.e., its exact residual $R_{\text{exact}}(L_0)$ is small enough that local gradient descent on the soft residual $R(L)$ converges to a point that the exact checker also certifies as feasible ($R_{\text{exact}}(L^\star) \le \epsilon$) within $K=50$ iterations. This assumption is not guaranteed; it is tested empirically via the calibration study in the experiment section. When it fails (grossly infeasible proposals, or soft-vs-exact disagreement), FIXP does not claim success: it emits an explicit infeasibility report and triggers a fallback discrete repair attempt (see step 5).

**How it works.**
```pseudocode
Input: text prompt T, LLM layout proposal L0, scene bounds B, tolerance eps=1e-3, max_iters K=50
1. Parse L0 into object pose tensors (p_i, q_i, s_i); freeze object identities and room shell.
2. Define residual R(L) = w_pen*Pen(L) + w_sup*Sup(L) + w_met*Met(L):
     Pen(L) = sum over pairs of soft-min signed distance of oriented bounding boxes (SDF via GJK-differentiable; implementation: differentiable GJK from the 'cbf' / 'pytorch-gjk' library, or autodiff through trimesh's proximity queries with a smooth-min temperature tau=0.01)
     Sup(L) = sum over objects of || lowest_support_point(o_i) - support_surface_below(o_i) ||^2  (gravity axis = +z)
     Met(L)  = sum over objects of || s_i - s_prior(o_i) ||^2  (metric prior from a size table keyed by object class; table built from 3D-FRONT training-split object dimensions, per-class median)
   weights: w_pen=1.0, w_sup=1.0, w_met=0.1
3. Repeat k=1..K:
     g <- dR/dL via autodiff through the SDF/support operators
     L <- L - alpha * g   with alpha from backtracking line search (Armijo, beta=0.5)
     Project scale to class-valid interval (5th-95th percentile of training-split dimensions for that class); project poses inside room bounds B (clamp).
     If R(L) < eps: break
4. Certification: recompute R_exact(L) with an EXACT (non-soft) geometric checker (implementation: trimesh's mesh-mesh intersection test for penetration, exact lowest-vertex vs. support-face distance for support, exact scale deviation for metric); if R_exact(L) <= eps, emit L* and a certificate; else proceed to step 5.
5. Fallback discrete repair (triggered only when step 4 fails): run a small physics-engine placement search (PyBullet, 200 steps, gravity on, objects dropped from current poses) restricted to the objects flagged by the exact checker as violating; re-run step 4 on the repaired layout. If still R_exact > eps, emit infeasibility report with the violating constraint set (which objects, which constraint type, exact residual value).
Output: L* (certified) or infeasibility report (with repair attempt logged)
```
The LLM proposal enters only as the initialization; the optimization, exact certification, and fallback repair are what produce the output. The infeasibility report is a first-class result, not a failure.

**Why this design.** We chose a differentiable residual + gradient projection over a discrete constraint-satisfaction solver (e.g., a physics-engine placement search) because the LLM's proposal is already near-feasible in most cases and gradient descent exploits that proximity, accepting the cost that highly infeasible proposals (e.g., a bed placed inside a wall) may need many iterations or fall into a local minimum. We chose to keep object identities and the room shell frozen and optimize only poses/scales over allowing the optimizer to add/remove objects, because identity changes would break the semantic alignment the LLM was responsible for, accepting the cost that semantic errors (wrong object choice) are not corrected — only geometric ones. We chose an exact non-soft certification pass separate from the soft residual over trusting the soft residual's own value, because soft-min SDFs can report near-zero penetration while the true geometry still interpenetrates, accepting the cost of a second geometry evaluation per sample. We chose to add a fallback discrete repair triggered only on certification failure over claiming gradient descent always succeeds, because the near-feasibility assumption is not guaranteed and a repair attempt is cheaper than a false certificate. This is not a variant of DirectLayout's CoT-Grounded Generative Layout Reward: that reward shapes the LLM's training distribution but never guarantees any individual output is feasible, whereas FIXP certifies each output against an explicit physics residual and is agnostic to which reasoner produced the proposal.

**Why it measures what we claim.** The soft penetration residual $Pen(L)$ measures physical non-interpenetration because the signed-distance formulation makes $Pen=0$ equivalent to disjoint oriented bounding boxes under the assumption that object geometry is well-approximated by its OBB; this assumption fails for concave or thin objects (chair legs, lamps), in which case $Pen$ reflects OBB overlap rather than true mesh penetration and the exact certification pass is what catches it. The support residual $Sup(L)$ measures gravitational plausibility because the distance from an object's lowest support point to the surface below is zero exactly when the object rests on a supporting surface under the assumption that the support surface is the nearest horizontal face beneath the object; this fails for wall-mounted or stacked objects, where the nearest face is not the support, and $Sup$ then reflects proximity rather than support. The metric residual $Met(L)$ measures scale consistency because the deviation from a class-conditioned size prior is zero exactly when the object matches expected real-world dimensions under the assumption that the prior table is accurate for the object class; this fails for unusual instances (a dollhouse bed), where $Met$ reflects prior mismatch rather than physical error. The certified fixed-point condition $R_{\text{exact}}(L^\star)\le\epsilon$ measures the motivation-level claim of physical feasibility because reaching a fixed point of the exact residual means no constraint violation remains under the assumption that the exact checker's zero set equals the feasible set; this fails when the exact checker and soft residual disagree, which is why certification is evaluated separately and reported. The soft-vs-exact disagreement rate is measured directly in the calibration study (experiment section) to quantify how often the soft residual's zero set diverges from the exact feasible set.

## Contribution

(1) A closed-loop layout refinement framework, FIXP, that treats an LLM/VLM numerical layout as a hypothesis and projects it onto a differentiable physical-feasibility manifold with an exact certification pass, decoupling spatial correctness from the reasoner's reliability. (2) A design principle — that generation quality should be measured by certified fixed-point feasibility rather than by the generator's one-shot output — instantiated as a differentiable penetration/support/metric residual with a separate exact checker. (3) A certification protocol and infeasibility-report format that exposes which physical constraint a proposal violates, enabling diagnosis of when LLM spatial reasoning fails geometrically versus semantically.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | 3D-FRONT text-annotated indoor scenes (test split, 200 scenes) | Paired text and layouts with real geometry |
| Primary metric | Certified feasibility rate (exact checker) | Directly tests the guarantee, not soft proxy |
| Secondary metric | Soft-vs-exact disagreement rate (false-negative rate of soft residual) | Quantifies the assumption that soft residual ≈ exact feasibility |
| Baseline | DirectLayout (CoT-grounded layout reward) | Tests whether reward shaping alone guarantees feasibility |
| Baseline | One-shot LLM layout (no projection) | Tests whether projection adds anything over raw proposal |
| Baseline | Physics-engine placement search (PyBullet, 200 steps) | Tests gradient projection vs discrete CSP on near-feasible inputs |
| Baseline | Generic penalty-based optimizer on same residual (no certification, no fallback) | Isolates gain from certification loop vs any projection |
| Ablation-of-ours | FIXP without exact certification pass | Tests whether soft residual alone suffices |
| Ablation-of-ours | FIXP without fallback discrete repair | Tests whether repair is necessary for the guarantee |

### Why this setup validates the claim

The central claim is that FIXP converts a near-feasible LLM proposal into a *certified* feasible layout, and that this certification is what distinguishes it from reward-shaped generation. 3D-FRONT gives us text prompts whose LLM proposals are typically close to feasible but not certified, which is exactly the regime where FIXP's proximity-exploiting gradient projection should help. The primary metric is the exact-checker feasibility rate, not the soft residual, because the soft residual is the thing under suspicion (soft-min SDF can hide true interpenetration); measuring on the exact checker means a win cannot be an artifact of the objective we optimize. DirectLayout tests the sub-claim that reward shaping does not guarantee per-sample feasibility; the one-shot LLM baseline tests whether projection adds anything at all; the physics-engine search tests the design choice of gradient projection over discrete CSP on near-feasible inputs. The generic penalty-based optimizer baseline isolates whether the gain comes from the certification loop and fallback repair or merely from applying any projection method to the same residual. The no-certification ablation is the sharpest test: if removing the exact pass does not change the measured feasibility rate, then the certification machinery is decorative and the soft residual was already adequate, which would undercut the core motivation. The calibration study (soft-vs-exact disagreement rate) directly measures the load-bearing assumption that the soft residual's zero set approximates the exact feasible set, and the no-fallback ablation tests whether the repair step is necessary when that assumption fails.

### Expected outcome and causal chain

**vs. DirectLayout** — On a case where the LLM proposes a sofa overlapping a coffee table by a few centimeters, DirectLayout's CoT-grounded reward pushes the training distribution toward feasible layouts but leaves this individual sample interpenetrating because the reward is a shaping signal, not a per-output constraint. Our method instead takes that same proposal as initialization and drives the exact-checked penetration to zero via gradient projection plus certification, so we expect a clear gap in certified feasibility rate concentrated on the near-miss subset, with parity on trivially feasible prompts.

**vs. One-shot LLM layout** — On a case where the proposal places a wardrobe flush against a wall with slight penetration and a floating lamp, the one-shot baseline emits it unchanged because there is no correction stage at all. Our method projects scale and pose until the exact residual is below tolerance, so we expect a large feasibility-rate gain overall, with the gain largest on prompts whose proposals are near-feasible and smaller (or replaced by explicit infeasibility reports) on grossly infeasible proposals.

**vs. Physics-engine placement search** — On a case where the proposal is already near-feasible, the discrete search spends iterations exploring a combinatorial neighborhood and may settle on a different, semantically drifted arrangement because it treats placement as search rather than local correction. Our method exploits proximity with gradient descent and freezes identities, so we expect comparable feasibility but higher semantic alignment on near-feasible prompts, and a visible slowdown or local-minimum failure for our method only on the grossly infeasible subset (where the fallback repair is triggered).

**vs. Generic penalty-based optimizer (no certification, no fallback)** — On a case where the soft residual reaches near-zero but the exact checker still reports penetration (e.g., thin chair leg vs table leg), the generic optimizer declares success because it trusts the soft residual. Our method's exact certification catches the discrepancy and either corrects further or reports infeasibility, so we expect the generic optimizer to show an inflated apparent feasibility rate that collapses when re-scored on the exact checker — a gap between soft and exact feasibility that our full method does not exhibit.

**vs. FIXP without exact certification** — On a case involving a thin chair leg whose OBB overlaps a table leg while the true meshes barely touch, the ablation reports the soft residual as near-zero and declares success because soft-min SDF under-reports true penetration. Our method's exact checker catches the discrepancy and either corrects further or reports infeasibility, so we expect the ablation to show an inflated apparent feasibility rate that collapses when re-scored on the exact checker — a gap between soft and exact feasibility that our full method does not exhibit.

**vs. FIXP without fallback discrete repair** — On a grossly infeasible proposal (e.g., bed inside a wall), gradient descent alone may stall in a local minimum and fail certification. The no-fallback ablation emits an infeasibility report without attempting repair, so we expect a lower certified feasibility rate on the grossly infeasible subset compared to full FIXP, which triggers the PyBullet repair and recovers some fraction of those cases.

### What would falsify this idea

If the certified feasibility gain over the one-shot LLM baseline were uniform across near-feasible and grossly infeasible subsets rather than concentrated where the predicted near-miss failure mode occurs, the proximity-exploitation story would be wrong. Likewise, if removing the exact certification pass left the exact-checker feasibility rate unchanged, the claim that certification is doing real work would be falsified. Additionally, if the soft-vs-exact disagreement rate were near zero across the 3D-FRONT corpus, the load-bearing assumption that soft residual approximates exact feasibility would be empirically supported and the certification pass would be less critical than claimed.

## References

1. Direct Numerical Layout Generation for 3D Indoor Scene Synthesis via Spatial Reasoning
2. Agentic 3D Scene Generation with Spatially Contextualized VLMs
