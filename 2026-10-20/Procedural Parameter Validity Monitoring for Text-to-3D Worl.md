# Procedural Parameter Validity Monitoring for Text-to-3D World Generation

## Motivation

InfiniDream maps text to procedural parameters via a VLM, but this mapping is brittle under distribution shift: prompts outside the training distribution yield invalid or non-photorealistic parameters, breaking the pipeline. The root cause is structural—the VLM has no explicit representation of parameter validity, relying instead on implicit code-generation priors that do not generalize to novel prompts. This limitation is critical for open-ended text-to-world generation, where the prompt space is unbounded.

## Key Insight

Parameter validity can be directly verified by executing the generated procedural code and checking geometric and perceptual constraints, providing a reliable out-of-distribution signal without relying on LLM confidence calibration.

## Method

# PV-Monitor: Procedural Validity Monitor with Execution-Based Fallback

# Input: user prompt P, procedural algorithm library L (each algorithm a with parameter schema S_a)
# Output: validated procedural parameters θ or fallback asset

1. θ, a = VLM(P)  # VLM selects algorithm a and generates parameters θ
2. Execute a(θ) to produce mesh M
3. Compute validity score V = 1[geometry_valid(M)] * 1[photorealism_score(M) > τ_photo(a)]
   # τ_photo(a) is calibrated per-algorithm on a held-out set of 256 valid samples for algorithm a:
   # τ_photo(a) = μ_a - 2σ_a, where μ_a and σ_a are the mean and std of photorealism_score on that set.
   # photorealism_score(M) is a CLIP-based aesthetic predictor (ViT-L/14, LAION-Aesthetics v2) output scaled to [0,1].
   # This makes the gate a relative comparison to the algorithm's own valid manifold, not a global aesthetic prior.
4. If V == 1: return M
5. Else:
   a. Compute embedding e_P = CLIP(P)
   b. Retrieve nearest prompt P' from a curated library of valid (prompt, parameters) pairs using cosine similarity in CLIP space
   c. If similarity > τ_sim: return execute(a', θ') from P'
   d. Else: trigger human-in-the-loop correction: present M to user, receive corrected parameters θ_corr, add (P, θ_corr) to library, return execute(a, θ_corr)

# Hyperparameters: τ_photo(a) calibrated per algorithm as above (default τ_photo=0.7 for algorithms with no calibration data), τ_sim = 0.85 (retrieval similarity threshold)

**Why this design**
We chose execution-based validity over LLM log-probability gating because log-probs are uncalibrated for task correctness (Kadavath et al., 2022), whereas execution directly verifies geometric and perceptual constraints. We chose a retrieval fallback over a learned correction model because retrieval provides guaranteed valid parameters from the curated library, accepting the cost that the library must be maintained and may not cover all prompts. We chose human-in-the-loop as the final fallback to handle truly novel prompts, accepting the cost of user interaction, but this ensures the system never outputs invalid assets. This design avoids the 'two cooperating sub-systems glued by a controller' anti-pattern because the controller's decision rule is closed-form: execute and check constraints, then retrieve if similar enough, else ask human. Unlike Toolformer or Gorilla, which use retrieval for tool selection, our method uses retrieval only after execution-based validity fails, and the retrieval is over prompts not tools, ensuring the fallback is a last resort rather than the primary mechanism.

**Why this measures what we claim**
The validity score V measures parameter validity because executing the procedural code and checking geometric constraints (e.g., no self-intersections, watertight mesh) and photorealism (via a pre-trained CLIP-based aesthetic predictor) directly operationalizes the motivation's requirement for valid and photorealistic outputs; this assumption fails when the photorealism predictor is biased against certain styles, in which case V reflects stylistic bias rather than true photorealism. The retrieval similarity measures out-of-distributionness because CLIP embedding distance correlates with semantic novelty; this assumption fails when prompts are semantically similar but procedurally different (e.g., 'chair' vs 'stool'), in which case retrieval may return invalid parameters. The human-in-the-loop correction measures the system's ability to handle novel prompts because user corrections provide ground-truth valid parameters; this assumption fails when users are unavailable or unskilled, in which case the system defaults to the retrieved asset.

## Contribution

['A novel execution-based validity monitor that detects out-of-distribution text-to-procedural-parameter mappings by directly executing generated code and checking geometric and perceptual constraints, avoiding reliance on uncalibrated LLM confidence.', 'A fallback mechanism that combines retrieval from a curated library of valid prompt-parameter pairs with human-in-the-loop correction, ensuring valid and photorealistic outputs even for out-of-distribution prompts.', 'An empirical demonstration that execution-based validity is a more reliable OOD signal than LLM log-probabilities for procedural generation, and that retrieval fallback significantly reduces invalid outputs compared to the base InfiniDream pipeline.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | InfiniDream 100 object + 20 scene tasks | Matches prior procedural-generation benchmark scope |
| Primary metric | Valid-and-photorealistic yield rate | Jointly captures both claimed validity dimensions |
| Baseline | VLM-only procedural generation | Tests whether execution gating adds value |
| Baseline | Retrieval-first (Gorilla-style) | Tests whether execution-first ordering matters |
| Baseline | No-fallback VLM + post-hoc filter | Tests whether fallback recovers rejected cases |
| Ablation-of-ours | PV-Monitor minus retrieval fallback | Isolates retrieval's contribution to yield |
| Ablation-of-ours | PV-Monitor with fixed τ_photo=0.7 (no per-algorithm calibration) | Tests whether calibration reduces style bias |

### Why this setup validates the claim

The central claim is that execution-based validity gating plus a retrieval fallback raises the fraction of prompts producing both geometrically valid and photorealistic assets, and that the gain is concentrated precisely where VLM parameter generation fails. The VLM-only baseline tests the sub-claim that unvalidated generation yields invalid parameters at a non-trivial rate; if it does, the headroom for our method exists. The retrieval-first baseline tests the ordering sub-claim: if retrieval-first performs comparably to ours, then execution gating is doing no work and the design rationale collapses. The no-fallback baseline tests whether the fallback chain (retrieval + human) is what recovers rejected cases, separating validation from recovery. The ablation (ours minus retrieval) isolates retrieval's marginal contribution from execution checking alone. The additional ablation (fixed τ_photo=0.7 vs. per-algorithm calibration) tests whether the calibrated gate reduces the style bias that the fixed threshold introduces, as measured by the yield on stylized/low-poly prompts. The primary metric — joint valid-and-photorealistic yield — is the right detector because the claim is explicitly conjunctive: a method that improves validity but degrades photorealism (or vice versa) would not register as a win, matching the motivation's dual requirement. If our method's gain were uniform across all subsets rather than concentrated on the prompts where VLM generation fails, the causal story would be wrong even if the average rose.

### Expected outcome and causal chain

**vs. VLM-only procedural generation** — On a prompt like "a weathered wooden gazebo with irregular lattice," the VLM-only baseline emits parameters that produce self-intersecting or non-watertight meshes because the VLM has no feedback on geometric consequences of its numeric choices. Our method executes the same parameters, detects the constraint violation via the geometry check, and routes to retrieval or human correction, so we expect a clear yield gap on the complex-topology subset and near-parity on simple primitive-like prompts where VLM parameters are usually already valid.

**vs. Retrieval-first (Gorilla-style)** — On a prompt that is semantically close to a library entry but procedurally distinct ("bar stool" vs. "chair"), the retrieval-first baseline returns the nearest library parameters, which are valid but produce the wrong object class, because it never executes the generated parameters to check whether they were already adequate. Our method executes first and only falls back when validity fails, so we expect retrieval-first to show lower photorealism/semantic-match scores on the near-neighbor-but-distinct subset while our method stays close to VLM-only on that subset and only diverges where validity actually fails.

**vs. No-fallback VLM + post-hoc filter** — On a prompt where the VLM's first attempt is invalid, this baseline simply discards the sample and returns nothing (or a default), because it has no recovery mechanism. Our method converts those rejected cases into valid outputs via retrieval or human correction, so we expect our yield to exceed this baseline by roughly the invalid-rate of the VLM-only generator, with the gap widening on the harder scene tasks where first-attempt validity is lowest.

**vs. PV-Monitor minus retrieval fallback** — On a prompt where the VLM's first attempt is invalid and no similar prompt exists in the library, this ablation triggers human correction immediately, whereas our full method first attempts retrieval. We expect the ablation to match our method on prompts with high-similarity library entries but to require more human corrections on novel prompts, leading to a lower yield under a fixed human budget (e.g., 50 corrections per subset).

**vs. PV-Monitor with fixed τ_photo=0.7 (no per-algorithm calibration)** — On stylized/low-poly prompts (e.g., "low-poly tree"), the fixed threshold rejects valid but unconventional outputs because the global aesthetic prior penalizes stylization. Our calibrated gate uses τ_photo(a) = μ_a - 2σ_a, so it accepts outputs within two standard deviations of the algorithm's own valid manifold. We expect the fixed-threshold ablation to show lower yield on the stylized subset while matching on photorealistic subsets, confirming that calibration reduces style bias.

### What would falsify this idea

If our method's yield gain over the no-fallback baseline is uniform across easy and hard subsets — rather than concentrated on prompts where VLM-only generation fails — then execution gating is not doing the causal work claimed. Likewise, if retrieval-first matches or beats our method on the near-neighbor-distinct subset, the execution-first ordering rationale is falsified.

## References

1. InfiniDream: Open-Ended Photorealistic 3D Worlds Through Procedural Generation
