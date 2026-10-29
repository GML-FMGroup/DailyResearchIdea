# Interventional Failure Attribution for Vision-Language-Action Models via Counterfactual Observation Probes

## Motivation

Across both VLA policy trees (OpenVLA-OFT's parallel L1 action-chunk head) and VLM coding-agent trees (HexaShell's tool-call loop), the observation-to-action map is a single monolithic function, so when a rollout fails the only available diagnostic is an end-to-end success log that cannot localize whether the error arose in the perceptual estimate or in the reasoning/action-selection stage. Existing evaluation frameworks such as AutoEval automate the *measurement* of success but inherit this monolithicity: they can tell you a policy failed, never which subsystem caused the failure, because no internal handle exists for intervening on one stage while holding the other fixed. This is the recurring structural bottleneck the field's trajectory now exposes as policies saturate success benchmarks (OpenVLA-OFT reaches 97.1% on LIBERO), since the residual failures are precisely the ones that require subsystem-level attribution to fix.

## Key Insight

Treating the observation-to-action map as a two-stage structural causal model with a shared task-relevant latent makes perceptual and reasoning failures identifiable from a designed probe set of counterfactual observation edits that leave the latent invariant, because the two failure modes predict opposite signs of action divergence under edits that preserve versus corrupt that latent — a rank condition that end-to-end success logs structurally cannot supply.

## Method

**What it is.** FARA (Failure Attribution via Rank-conditioned counterfactuals) takes a frozen VLA policy π, a failed rollout with its observation o and instruction ℓ, and a calibrated camera/tool interface, and outputs a per-failure attribution certificate labeling the failure as perceptual or reasoning, together with the probe set that witnesses the label.

**Load-bearing assumption (stated explicitly).** FARA assumes the triangulated latent read L(o) from the calibrated camera is a faithful, edit-verifiable ground truth of the task-relevant state, such that an edit can be certified as latent-preserving or latent-corrupting independently of the policy's own perceptual encoding. This assumption is *not* taken on faith: it is verified on held-out successful rollouts before any certificate is emitted (Step 0 below), and a nuisance-sensitivity control probe family is added so that encoder-level nuisance sensitivity is separated from genuine perceptual misestimation rather than folded into the PERCEPTUAL label.

**How it works.**

```pseudocode
Input: frozen policy π, failed rollout (o, ℓ, a_exec), calibrated camera C,
       task-relevant latent spec L (object pose/identity read via triangulation),
       edit budget K=8 per family, invariance tolerance ε=0.02 (in latent units),
       divergence threshold τ=0.15 (in action units),
       sufficiency tolerance ε_suff=0.02 (in action units),
       nuisance two-sample test threshold p_min=0.05,
       calibration-validation set S_cal of 200 held-out successful rollouts
Output: certificate {label ∈ {PERCEPTUAL, REASONING}, witness probes, rank stats}

0. SUFFICIENCY CHECK (run once per policy, before any attribution):
   For each successful rollout in S_cal, construct invariant probes P_inv as in 2a
   and compute D_inv. If median_{S_cal} D_inv > ε_suff, the triangulated latent is
   NOT a sufficient statistic for π's action; FARA withholds all certificates and
   reports the measured D_inv distribution. Otherwise proceed.

1. LATENT READ: use C + triangulation tools to read L(o) = true task-relevant
   latent (object positions/identities) from the calibrated views.

2. PROBE CONSTRUCTION (three families, each size K):
   a. INVARIANT probes P_inv: counterfactual observation edits e_i that change
      nuisance factors (lighting, background texture, camera exposure, distractor
      placement) while L(e_i(o)) = L(o) within ε  (verified by re-reading L).
   b. CORRUPTING probes P_cor: edits e_j that perturb L itself (shift object pose
      by δ, swap object identity) so that ||L(e_j(o)) − L(o)|| > ε.
   c. NUISANCE-CONTROL probes P_nui: edits e_k that change nuisance factors in the
      same way as P_inv but are applied to a *known-good* successful rollout from
      S_cal, used to estimate the policy's baseline nuisance sensitivity.

3. DIVERGENCE MEASUREMENT: for each probe compute action divergence
      d(p) = ||π(e_p(o), ℓ) − π(o, ℓ)||_2 .
   Aggregate D_inv = median_{i} d(P_inv_i),  D_cor = median_{j} d(P_cor_j),
   D_nui = median_{k} d(P_nui_k).

4. RANK TEST: compute the sensitivity ratio
      R = D_cor / (D_inv + η),  η=1e-3.
   A well-formed policy should have R >> 1: it should be invariant to nuisance
   edits and sensitive to latent-corrupting edits.

5. ATTRIBUTION:
   if R < R_min (=2.0):            # policy fails the invariance/sensitivity rank condition
       if D_inv > τ and D_inv > D_nui + τ:  label = PERCEPTUAL   # action moved under latent-preserving edit beyond nuisance baseline
       else:                                label = REASONING    # action failed to move under latent-corrupting edit
   else:
       label = REASONING   # policy is locally well-formed but chose wrong action

6. WITHHOLDING CRITERION (computable):
   Before emitting a certificate, run a two-sample Kolmogorov–Smirnov test between
   the per-probe divergence distributions {d(P_inv_i)} and {d(P_cor_j)}. If the
   test fails to reject the null of equal distributions at p_min=0.05, or if the
   per-probe divergences within either family are bimodal (Hartigan dip test
   p<0.05), the certificate is WITHHELD rather than mislabeled, and the rank
   statistic R is reported as undefined.

7. CERTIFICATE: emit label, the K_inv probes that exceeded τ, the K_cor probes
   that stayed below τ, the rank statistic R, and the KS/dip-test outcomes.
```

**Why this design.** We chose a *designed* probe set over learned perturbation models because the attribution claim requires that the probe provably preserve the task-relevant latent, and only a calibrated-interface probe (triangulated pose re-read) gives that guarantee; the cost is that probe construction requires known camera calibration, so the method is inapplicable to uncalibrated in-the-wild video. We chose a *rank condition* (D_cor vs D_inv ratio) over a single divergence threshold because a monolithic policy's raw sensitivity is arbitrary and a threshold on it is unidentifiable; the ratio cancels the policy's global action scale, accepting the cost that it is undefined when D_inv ≈ 0 (handled by η). We chose a *two-family* probe design over a single edit family because perceptual and reasoning failures make opposite-sign predictions on the two families, and a one-family design cannot separate them; the cost is roughly double the probe budget. The sufficiency check (Step 0) and nuisance-control family (Step 2c) are added because the triangulated latent is a geometric pose/identity read, whereas patch-based VLAs act on dense embeddings that may be sensitive to nuisance texture and lighting a pose read declares invariant; without these, D_inv would fold encoder-level nuisance sensitivity into the PERCEPTUAL label. Crucially, this is not the retrieval-style descriptor matching of Toolformer/Gorilla: those methods select among a fixed library by similarity, whereas FARA *intervenes* on the observation and reads the policy's counterfactual response, a quantity retrieval never computes. It is also not the log-prob gating of Kadavath-style calibration: the signal is an action-space divergence under a verified latent-preserving edit, not a likelihood.

**Why it measures what we claim.** The quantity `D_inv` measures *perceptual-stage failure* because a latent-preserving edit that still moves the action implies the perceptual estimate is not a sufficient statistic for the action, under the assumption that the triangulated latent read is a faithful ground-truth of the task-relevant state; this assumption fails when calibration drifts, in which case D_inv reflects calibration error rather than perception. The quantity `D_cor` measures *reasoning-stage sensitivity* because a latent-corrupting edit that fails to move the action implies the reasoning stage ignores the corrupted latent, under the assumption that the corrupting edit is large enough to cross the policy's decision boundary; this assumption fails for near-boundary poses, in which case D_cor reflects edit magnitude rather than reasoning insensitivity. The rank statistic `R` measures *attribution identifiability* because the ratio of the two divergences is invariant to global action scale exactly when the two probe families share the same nuisance distribution, under the assumption that the invariant and corrupting edits are drawn from the same nuisance manifold; this assumption fails when the corrupting edits also change nuisance factors, in which case R conflates the two stages and the certificate is withheld rather than mislabeled. The nuisance-control divergence `D_nui` measures *encoder-level nuisance sensitivity* because it is computed on known-good successful rollouts where any action movement under a latent-preserving edit cannot be a perceptual failure of the task-relevant state; this assumption fails when the known-good rollout itself is near a decision boundary, in which case D_nui reflects boundary proximity rather than encoder nuisance sensitivity.

## Contribution

['(1) A two-stage structural causal model of the VLA observation-to-action map with a shared task-relevant latent, together with a rank condition on a designed counterfactual probe set that makes perceptual-versus-reasoning failure attribution identifiable from a frozen policy.', '(2) An interventional attribution protocol (FARA) that emits a per-failure certificate with witness probes rather than a success-log heuristic, constructible from the calibrated camera/tool interfaces that both OpenVLA-OFT-style policies and HexaShell-style coding agents already expose.', '(3) A design principle: attribution requires probes that are verified to preserve the task-relevant latent, and the ratio of corrupting-to-invariant action divergence is the scale-free statistic that separates the two failure modes.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | LIBERO failure rollouts on OpenVLA-OFT, plus 200 held-out successful rollouts for sufficiency check | Calibrated tabletop views enable triangulated latent reads |
| Dataset (generalization) | HexaShell tool-call loop failures on a calibrated coding-agent task | Tests second policy tree per reviewer |
| Primary metric | Attribution accuracy vs. simulator-injected ground-truth failure stage | Objective label source, no human labeling bottleneck |
| Secondary metric | Certificate withholding rate and calibration-error degradation curve | Tests sufficiency-check and withholding criterion |
| Baseline | Post-hoc log-prob calibration gating | Tests whether likelihood signal suffices for attribution |
| Baseline | Retrieval-style descriptor matching (Gorilla/Toolformer) | Tests whether similarity selection beats interventional probing |
| Baseline | Single-family divergence threshold | Tests whether two-family rank condition is necessary |
| Baseline | Causal-mediation stage attribution (Shapley-style) | Tests whether R is a relabeling of known attribution |
| Ablation-of-ours | FARA with invariant-only probes (drop P_cor) | Isolates contribution of the two-family rank design |
| Ablation-of-ours | FARA without sufficiency check and nuisance-control family | Isolates contribution of the load-bearing-assumption repair |

### Why this setup validates the claim

The central claim is that a rank-conditioned, two-family interventional probe separates perceptual from reasoning failures in a frozen VLA, and that this separation is not recoverable from likelihood signals or retrieval-style similarity. LIBERO failure rollouts with calibrated camera views are the right substrate because only there can triangulated latent re-reads *prove* an edit preserved or corrupted the task-relevant state — the guarantee the method's attribution certificate rests on. Attribution accuracy against simulator-injected ground-truth failure stage is the right metric because it directly scores the label the method emits with an objective label source, not a proxy; a method that merely correlates with failure severity but scrambles the perceptual/reasoning axis will score near chance even if it looks good on success-rate curves. The log-prob calibration baseline tests sub-claim (a): that action-space divergence under a verified latent-preserving edit carries information a likelihood does not. The retrieval baseline tests sub-claim (b): that intervention on the observation is categorically different from selecting among a fixed library. The single-family baseline tests sub-claim (c): that the ratio R, not a raw divergence, is what makes attribution identifiable. The causal-mediation baseline tests sub-claim (d): that R is not a relabeling of Shapley-style stage attribution. The invariant-only ablation isolates whether the corrupting family is doing real work or whether D_inv alone suffices — if the ablation matches full FARA, the two-family design is unjustified. The no-sufficiency-check ablation isolates whether the load-bearing-assumption repair (Step 0 + nuisance-control family) actually changes attribution accuracy, and the calibration-error degradation curve quantifies how attribution accuracy falls as triangulated-latent error grows.

### Expected outcome and causal chain

**vs. Post-hoc log-prob calibration gating** — On a case where the robot reaches for the wrong object because the instruction "put the bowl in the drawer" was misgrounded to a distractor, the baseline produces a low-confidence flag that fires on both perceptual and reasoning failures indiscriminately, because its mechanism gap is that a scalar likelihood conflates "the observation was misread" with "the observation was read but the plan was wrong." Our method instead constructs invariant probes that leave the bowl's triangulated pose fixed and corrupting probes that shift it, and reads the policy's action divergence under each, because the rank condition R separates the two stages by construction. We expect a clear accuracy gap on the perceptual-labeled subset (FARA near ceiling, calibration near chance) with parity on the reasoning subset, since both can flag reasoning failures but only FARA localizes the stage.

**vs. Retrieval-style descriptor matching (Gorilla/Toolformer)** — On a case where the failure arises because the policy ignored a small but decision-relevant object-pose shift (a near-boundary grasp that should have been abandoned), the baseline produces a nearest-neighbor match to a stored successful trajectory and declares the rollout "consistent with success," because its mechanism gap is that similarity selection never computes the counterfactual response of the policy to an edited observation. Our method instead corrupts the pose by δ, re-reads the latent to confirm the edit crossed ε, and measures whether the action moved; if it did not, the certificate is REASONING. We expect the retrieval baseline to systematically under-call reasoning failures (it sees similarity, not insensitivity) while FARA calls them correctly, producing a concentrated gap on the reasoning subset and near-parity on perceptual ones.

**vs. Single-family divergence threshold** — On a case where the policy is globally over-sensitive (large action scale) and fails perceptually, the baseline produces a PERCEPTUAL label for almost every failure because its raw D_inv exceeds any fixed threshold, because its mechanism gap is that a monolithic divergence threshold is not scale-invariant and cannot distinguish "sensitive to nuisance" from "globally twitchy." Our method instead divides by D_cor via R, so a globally over-sensitive policy still has R >> 1 only if it is *selectively* sensitive, and a perceptual failure shows R < R_min with D_inv > τ. We expect the single-family baseline to over-label PERCEPTUAL on high-action-scale rollouts and to collapse to chance overall, while FARA's accuracy stays high across action-scale strata — a gap that widens exactly where the scale confound is strongest.

**vs. Causal-mediation stage attribution (Shapley-style)** — On a case where the failure is a perceptual misestimation of object pose, the baseline attributes credit by ablating internal stage activations and measuring action change, because its mechanism gap is that it requires access to internal stage boundaries that a monolithic VLA does not expose, so it must approximate stages by layer groups and its attributions are sensitive to that grouping. Our method instead intervenes on the *observation* and reads the policy's counterfactual action, requiring no internal stage boundary. We expect the mediation baseline to underperform on perceptual failures where layer-group approximation blurs the perceptual/reasoning boundary, and to be unstable across grouping choices, while FARA's accuracy is stable because its intervention is external and verified.

**vs. FARA with invariant-only probes (drop P_cor)** — On a reasoning failure where the policy ignores a corrupted latent, the ablation has no corrupting family to detect the insensitivity and can only observe D_inv, because its mechanism gap is that a single edit family cannot produce the opposite-sign prediction that separates the two stages. Our method uses both families and the ratio R. We expect the ablation to collapse to near-chance on reasoning failures and to over-label PERCEPTUAL, while full FARA stays high on both subsets — the gap concentrated exactly on the reasoning subset.

**vs. FARA without sufficiency check and nuisance-control family** — On a perceptual failure caused by encoder-level nuisance sensitivity (lighting/texture) rather than task-relevant misestimation, the ablation folds D_inv into the PERCEPTUAL label without checking whether the triangulated latent is a sufficient statistic, because its mechanism gap is that it assumes the pose read is the policy's sufficient statistic. Our method runs Step 0 and subtracts D_nui. We expect the ablation to over-label PERCEPTUAL on nuisance-sensitive rollouts and to show a larger accuracy drop as calibration error grows, while full FARA withholds certificates when the sufficiency check fails and maintains accuracy on the remainder.

### What would falsify this idea

If FARA's attribution accuracy gain over the log-prob and retrieval baselines is uniform across perceptual- and reasoning-labeled subsets rather than concentrated on the subset each baseline's mechanism gap predicts it should miss, the claim that the two-family rank condition is what enables stage separation is wrong. Equivalently, if the invariant-only ablation matches full FARA's accuracy, the corrupting probe family and the rank statistic are doing no work and the central design choice is unjustified; and if the no-sufficiency-check ablation matches full FARA, the load-bearing-assumption repair is unnecessary.

## References

1. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success
2. HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents
3. π0.5: a Vision-Language-Action Model with Open-World Generalization
4. AutoEval: Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real World
5. AlphaEvolve: A coding agent for scientific and algorithmic discovery
6. RT-H: Action Hierarchies Using Language
7. PaliGemma: A versatile 3B VLM for transfer
8. Achieving Human Level Competitive Robot Table Tennis
