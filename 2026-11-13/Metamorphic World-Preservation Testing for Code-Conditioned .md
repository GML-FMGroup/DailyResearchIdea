# Metamorphic World-Preservation Testing for Code-Conditioned Video Generation

## Motivation

WorldDraft evaluates world preservation on 432 fixed paired cases drawn from 48 executable world families, and EditBoard assumes a comprehensive benchmark is a fixed set of test cases; both treat the test set as the unit of evaluation, so a model can score well by overfitting to the specific code instantiations (particular variable names, loop structures, gravity constants) rather than by respecting the kinematics the code prescribes. The structural root cause is that a preservation score is reported for a single representative per world family, leaving the metric's invariance to semantics-preserving code rewrites completely unestablished — a gap shared across VBench, EditBoard, and WorldDraft. Without equivalence classes, no existing benchmark can distinguish a renderer that tracks the prescribed 4D dynamics from one that memorized the canonical code surface form.

## Key Insight

If a preservation score is a genuine measure of dynamics-faithfulness rather than of surface-code familiarity, then it must be invariant across the equivalence class of programs that denote the same 4D world, so the within-class spread of scores is itself a direct, reference-free estimate of fixed-case overfitting.

## Method

**A. What it is.** MWPT (Metamorphic World-Preservation Testing) takes a WorldDraft world family's executable code and a target video model, and returns a *metamorphic preservation profile*: for each family, a set of semantics-preserving program variants (an equivalence class), the per-variant world-preservation score, and two derived statistics — class-mean preservation and class-invariance (the within-class score spread). Inputs: world-family code + camera/kinematics spec + text prompt; output: per-family (mean, spread) plus a model-level overfitting index. **Load-bearing assumption (stated explicitly):** a semantics-preserving program rewrite that leaves the executed 4D trajectory bit-exact is observationally neutral to the video model, so any within-class variation in the WorldDraft preservation score must reflect dynamics overfitting rather than the model's legitimate sensitivity to the code surface form (tokenization, variable names, control-flow structure) that conditions its rendering. This assumption is not taken for granted: we measure the model's surface-form sensitivity with a control equivalence class and subtract it from the raw spread, yielding an excess-spread estimator that isolates dynamics overfitting.

**B. How it works.**
```pseudocode
Input: world families W (from WorldDraft), model M, n_variants=8 per family
Output: per-family (mu_pres, sigma_pres, sigma_surface, sigma_excess), model overfitting index OI

for each family w in W:
    # Phase 1: build the dynamics equivalence class via semantics-preserving edits
    C_w = {w}
    C_w += rename_variables(w, seed=k)              # alpha-renaming, matched token length
    C_w += refactor_control_flow(w)                  # loop<->recursion, unroll
    C_w += reparam_physics(w, delta=0)               # equiv. constants via unit/scale maps
    C_w += reorder_independent_updates(w)            # commute non-interacting bodies
    # Phase 1b: build the surface control class (tokenization-relevant, dynamics-irrelevant)
    S_w = {w}
    S_w += insert_whitespace_and_comments(w)         # verified trajectory-identical
    S_w += rename_identifiers_matched_length(w)      # same token count, different names
    S_w += prompt_paraphrase_hold_code_fixed(w)      # same code, paraphrased prompt
    # Phase 2: verify semantic equivalence BEFORE trusting either class
    for v in C_w ∪ S_w:
        assert exec_trajectory(v) == exec_trajectory(w)  # bit-exact 4D ground truth
    # Phase 3: score preservation on each variant
    for v in C_w:
        video = M.render(v.code, v.camera, v.prompt)
        s[v]  = world_preservation(video, exec_trajectory(v))  # WorldDraft metric
    for v in S_w:
        video = M.render(v.code, v.camera, v.prompt)
        s_surf[v] = world_preservation(video, exec_trajectory(v))
    mu_pres[w]       = mean(s)
    sigma_pres[w]    = std(s)                        # raw class-invariance
    sigma_surface[w] = std(s_surf)                   # surface-sensitivity baseline
    sigma_excess[w]  = max(0, sigma_pres[w] - sigma_surface[w])  # excess spread

# model-level fixed-case overfitting index
OI(M) = mean_w( sigma_excess[w] / (mu_pres[w] + eps) )
```

**C. Why this design.** We chose to *verify* semantic equivalence by bit-exact trajectory execution rather than trusting edit templates, because an unverified "equivalent" variant silently changes the ground truth and would attribute a genuine dynamics change to model brittleness, accepting the cost that some candidate edits are rejected and the class is smaller than a template-only generator would produce. We chose to keep WorldDraft's original preservation metric unchanged and only vary the *inputs* to it, rather than designing a new metric, because this isolates the property under test (invariance) from metric-design confounds and lets the result be read as a statement about the existing benchmark; the cost is that any noise in the base metric propagates into the spread estimate, which we mitigate by averaging over n_variants=8. We chose a within-class spread statistic over a cross-family comparison because cross-family variation conflates genuine difficulty differences with overfitting, whereas within-class variation holds the world fixed by construction; the cost is that we cannot rank families by absolute difficulty, only by brittleness. We chose to add a surface control class and report excess spread rather than raw spread, because the load-bearing assumption that trajectory-identical rewrites are rendering-neutral is empirically testable and may fail: code-conditioned models are known to be sensitive to tokenization and prompt surface form, so raw sigma_pres conflates dynamics overfitting with legitimate surface conditioning. The control class measures that surface sensitivity directly, and subtracting it yields an estimator that isolates the dynamics-overfitting component. This differs from metamorphic testing as used in software engineering (Chen et al.) and from edit-based video benchmarks like EditBoard: those apply edits to *inputs* and check output *properties* of a single system, whereas here the metamorphic relation is imposed on a *benchmark metric across models*, and the quantity of interest — the excess spread of a preservation score within a verified equivalence class — is not produced by EditBoard's fixed test set or by WorldDraft's 432 fixed pairs.

**D. Why it measures what we claim.** The computational quantity `sigma_excess[w]` (within-class standard deviation of preservation scores minus the surface-control standard deviation) measures *fixed-case overfitting* because of the assumption that the surface control class captures the model's legitimate sensitivity to rendering-irrelevant code-surface changes, so the residual spread after subtraction is attributable to dynamics overfitting; this assumption fails when the surface control class does not span the same tokenization shifts as the dynamics class (e.g., alpha-renaming in C_w changes token length while the control holds length fixed), in which case `sigma_excess` under- or over-corrects and reflects residual tokenization sensitivity rather than pure dynamics overfitting. The quantity `mu_pres[w]` measures *dynamics faithfulness* because of the assumption that WorldDraft's preservation metric is monotone in trajectory/camera error; this fails when the metric saturates (all variants near-perfect or near-zero), in which case the mean reflects metric ceiling/floor rather than fidelity. The equivalence check `exec_trajectory(v) == exec_trajectory(w)` measures *semantic identity* because of the assumption that bit-exact 4D ground-truth trajectories fully determine the world's prescribed dynamics; this fails when two programs yield identical trajectories on the sampled camera path but differ off-path, in which case the class is equivalent only on the observed view. The model-level index `OI(M)` measures *benchmark-level overfitting* because of the assumption that overfitting manifests as higher excess within-class variance relative to class mean across families; this fails when a model is uniformly brittle across all families, in which case `OI` reflects general instability rather than fixed-case memorization — which is why we report `mu_pres`, `sigma_pres`, `sigma_surface`, and `sigma_excess` jointly and never `OI` alone.

## Contribution

(1) MWPT, a metamorphic testing framework that constructs verified semantics-preserving equivalence classes of executable world code (variable renaming, control-flow refactoring, equivalent physics reparameterization, commutation of independent updates) and defines the class-invariance of world-preservation scores as a reference-free overfitting diagnostic. (2) The empirical finding that world-preservation scores are not invariant across verified-equivalent code variants, with the within-class spread decomposed by edit type to identify which surface features (tokenization, control structure, constant form) drive apparent preservation gains. (3) A model-level fixed-case overfitting index and a public suite of equivalence classes extending WorldDraft's 48 world families, enabling re-scoring of any video model without new ground-truth annotation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | WorldDraft 432 paired cases (48 world families × 9 cases) | Provides executable world families with verified 4D ground truth |
| Primary metric | Excess class-invariance σ_excess = max(0, σ_pres − σ_surface) | Isolates dynamics overfitting from legitimate code-surface sensitivity |
| Secondary metric | Raw class-invariance σ_pres (within-family spread) | Quantifies total within-class variation before surface correction |
| Control metric | Surface-sensitivity σ_surface (within-control-class spread) | Measures tokenization/prompt sensitivity independent of dynamics |
| Baseline | WorldDraft fixed-pair preservation score | Tests whether static pairs miss brittleness exposed by equivalence classes |
| Baseline | EditBoard-style input-edit metamorphic test | Tests whether input-only edits conflate prompt conditioning with dynamics faithfulness |
| Baseline | Template-only variant generation (no equivalence check) | Tests whether unverified edits inflate spread via ground-truth drift |
| Ablation-of-ours | MWPT without Phase 2 equivalence verification | Isolates the contribution of bit-exact semantic verification to spread estimates |
| Ablation-of-ours | MWPT without surface control class (raw σ_pres only) | Isolates the contribution of surface-sensitivity correction to the overfitting index |

### Why this setup validates the claim

The central claim is that fixed-case overfitting in video models manifests as within-class preservation spread across *verified* semantics-preserving program variants, over and above the model's legitimate sensitivity to code surface form. The WorldDraft 432-case set supplies executable world families with bit-exact 4D trajectories, which is the only substrate that lets us construct equivalence classes without trusting edit templates. The primary metric σ_excess is chosen because it holds the world fixed by construction and subtracts the measured surface-sensitivity baseline: any residual variation in preservation score across variants cannot be attributed to dynamics differences or to tokenization shifts, so it must reflect model overfitting to the canonical code surface form. The WorldDraft fixed-pair baseline tests the sub-claim that static pairs are blind to this brittleness — if fixed-pair scores were already predictive of σ_excess, our metric would be redundant. The EditBoard-style baseline tests whether input-only metamorphic edits suffice; if they did, our program-level equivalence class would add nothing. The template-only baseline tests whether verification matters: if unverified edits produced the same spread pattern, Phase 2 would be unnecessary. The ablation without Phase 2 directly measures how much of the observed spread is genuine model brittleness versus ground-truth drift from unverified edits. The ablation without the surface control class tests whether the correction is necessary: if raw σ_pres and σ_excess yield the same model ranking and the same family-level pattern, the surface control adds no information. Together these form a falsifiable test: our claim predicts that σ_excess will be nonzero and *not* explained by fixed-pair scores, that verified edits will yield *lower* excess spread than unverified templates, that the surface control will absorb a measurable fraction of raw σ_pres, and that the ablation without Phase 2 will show inflated spread concentrated on families where unverified edits alter trajectories.

### Expected outcome and causal chain

**vs. WorldDraft fixed-pair preservation score** — On a case where a model has memorized the 432 fixed pairs (e.g., high score on canonical camera paths), the fixed-pair baseline produces a uniformly high preservation score because it only queries each world once, missing sensitivity to rendering-irrelevant code changes. Our method instead renders the same world under alpha-renamed and reordered variants, so a model that keys on surface token statistics rather than dynamics will show score drops on some variants. We expect a noticeable gap in σ_excess between models with similar fixed-pair means, with the gap concentrated on families whose variants induce large tokenization shifts.

**vs. EditBoard-style input-edit metamorphic test** — On a case where an input edit (e.g., prompt paraphrase) changes appearance but not dynamics, the EditBoard-style baseline produces a preservation score change that is attributed to model brittleness, because its metamorphic relation is on inputs and cannot separate rendering difficulty from dynamics error. Our method instead holds the executable world fixed up to verified semantic equivalence and explicitly measures the prompt-paraphrase sensitivity in the surface control class, so any score change beyond that control is attributable to the model's dynamics tracking. We expect our σ_excess to be lower than EditBoard's edit-induced variance on appearance-aligned subsets, but higher on subsets where models overfit to specific code tokenizations — a dissociation that EditBoard cannot produce.

**vs. Template-only variant generation (no equivalence check)** — On a case where a template edit (e.g., loop unrolling) silently changes floating-point accumulation order and thus the 4D trajectory, the template-only baseline produces inflated preservation spread because it attributes a genuine ground-truth change to model brittleness. Our method instead rejects that variant in Phase 2 via bit-exact trajectory comparison, so the class contains only observationally identical worlds. We expect the ablation (MWPT without Phase 2) to show systematically higher σ_excess than full MWPT, with the excess concentrated on families where unverified edits alter trajectories — a diagnostic signature that verification is doing real work.

**vs. MWPT without surface control class (raw σ_pres only)** — On a case where a model is legitimately sensitive to tokenization (e.g., variable renaming changes the prompt-conditioning distribution), the raw σ_pres includes that sensitivity and would overstate dynamics overfitting. Our method instead measures the same model's sensitivity to surface-only perturbations in the control class and subtracts it, so σ_excess isolates the dynamics-overfitting component. We expect σ_surface to be non-negligible (e.g., 20–50% of σ_pres) on families with large tokenization shifts, and we expect the ranking of models by σ_excess to differ from the ranking by raw σ_pres for at least one model pair — a diagnostic signature that the surface correction changes conclusions.

### What would falsify this idea

If σ_excess were uniformly near zero across all models and families, or if the ablation without Phase 2 produced the same spread pattern as full MWPT, or if σ_surface were so large that σ_excess was indistinguishable from zero for all models, then the central claim — that verified semantics-preserving variants expose fixed-case overfitting beyond surface sensitivity — would be wrong. Likewise, if σ_excess were perfectly predicted by fixed-pair preservation scores, the within-class excess spread would carry no information beyond the existing benchmark.

## References

1. WorldDraft: If Code Builds the World, Can Video Models Just Render It?
