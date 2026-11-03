# Monotone Composition Kernel with Region-Level Shapley Attribution from Pairwise Comparisons

## Motivation

Existing work treats the composition optimization objective and perceptual evaluation as independent modules: the architectural interior design pipeline of (Automatic Image Composition Optimization and Visual Perception Simulation Evaluation) optimizes composition parameters against a simulated perception score while never enforcing that edits improving the optimizer's score also improve human-anchored perception, and PICD (Can Machines Understand Composition?) benchmarks composition understanding as a categorical label with no region-level attribution linking a judgment to the image regions that license it. Structurally, no shared sufficient statistic couples the composition score and the perceptual score, so a composition edit can raise one while degrading the other, and no identifiability condition ties a composition judgment to specific regions.

## Key Insight

A region-attributed composition score that is constrained monotone in a human-anchored perceptual score, with region weights identified as Shapley values from pairwise human comparisons, forces the optimizer's objective and the perceptual evaluation to share one region-level sufficient statistic, so improving the composition score cannot degrade simulated perception.

## Method

**What it is.** We introduce the Monotone Composition Kernel (MCK): a differentiable scoring function over an image's region decomposition that outputs a scalar composition score S(I) and per-region Shapley-style attributions φ_r, trained jointly with a perceptual head P(I) such that S is monotone in P by construction and φ_r is identifiable from human pairwise comparisons. Inputs: image I, its region decomposition R(I) (from an off-the-shelf segmentation model, fixed), and a set of pairwise human comparisons (PICD-style). Outputs: S(I), P(I), and φ_r for each region.

**Load-bearing assumption (stated explicitly).** The monotone-coupling claim — that improving S cannot degrade human-anchored perception — holds only under assumption A: the perceptual head P(I) is a sufficient statistic for human composition judgment. Failure mode F: P is trained only on PICD pairwise labels and is never anchored to an independent human perceptual measurement, so monotonicity in P is monotonicity in a learned proxy; when P is misspecified (e.g., on narrow category distributions), gamma measures coupling to the proxy rather than to human perception. We therefore (i) replace the single learned P with an ensemble of K=5 independently trained perceptual heads P_k (each a 2-layer MLP, hidden=256, GeLU, trained with a different random seed and bootstrap sample of the PICD training pairs), (ii) enforce monotonicity against the ensemble's lower confidence bound min_k P_k(I), and (iii) report the Pearson correlation between the ensemble mean P and held-out human perceptual ratings (collected as a 5-point Likert scale on a 512-image calibration set) as a gate before claiming the coupling transfers to human perception. The gamma=0 ablation cannot detect proxy misspecification because both arms share the same P; the ensemble LCB and the calibration correlation are the diagnostics that can.

**How it works.**
```pseudocode
# Phase 1: Region scoring (interaction-aware)
for each region r in R(I):
    f_r = RegionEncoder(r)                 # 128-d feature
    s_r = w^T f_r + b                       # region score, w in R^128
# pairwise interaction term (low-rank bilinear, rank d=8)
for each pair (r, r') in R(I) x R(I), r != r':
    g_{r,r'} = f_r^T U V^T f_r'            # U, V in R^{128 x 8}
S_raw = sum_r s_r + sum_{r<r'} g_{r,r'}    # raw composition score

# Phase 2: Monotone perceptual coupling (ensemble LCB)
for k in 1..K=5:
    P_k = PerceptualHead_k(I)              # scalar perceptual score, 2-layer MLP, hidden=256, GeLU
P = min_k P_k(I)                           # ensemble lower confidence bound
S = sigma(alpha * S_raw + beta) + gamma * P  # monotone in P: dS/dP = gamma > 0
# alpha, beta, gamma learned; gamma constrained positive via softplus

# Phase 3: Shapley-style region attribution via pairwise comparisons
# For a pair (I+, I-) with human preference I+ > I-:
#   phi_r = sum_{T subset R\{r}} |T|!(|R|-|T|-1)!/|R|! * [S(I+_{T+r}) - S(I+_T)]
# approximated by sampling K_shap=32 permutations per pair (Monte-Carlo Shapley)
phi_r = MCShapley(S, I, K_shap=32)

# Phase 4: Joint training loss
L_rank = max(0, margin - (S(I+) - S(I-)))        # pairwise ranking, margin=0.1
L_mono = max(0, -gamma)                          # enforce gamma > 0 (softplus already)
L_attr = sum_r (phi_r - phi_r^human)^2           # attribution supervision where available
L = L_rank + lambda_1 * L_mono + lambda_2 * L_attr
# lambda_1 = 0.01, lambda_2 = 0.5
```

**Why this design.** We chose a linear region scorer plus a low-rank bilinear interaction term over a deep region network because the interaction term makes the Shapley value of region r non-trivial and interaction-aware (a region's credit depends on which other regions are present), while the low rank d=8 keeps the pairwise term tractable and the Shapley computation still closed-form for the linear part; we accept the cost that region features must be linearly separable in composition space and that interactions are assumed low-rank. We chose to enforce monotonicity through the additive coupling term gamma * min_k P_k(I) with gamma > 0 rather than through a post-hoc calibration, because a post-hoc map would leave the optimizer and the perceptual head free to disagree during training, and the ensemble LCB prevents S from rising by exploiting one head's misspecification; we accept the cost that the perceptual heads must be trained jointly rather than frozen and that the LCB is conservative. We chose Monte-Carlo Shapley sampling with K_shap=32 permutations over an exact Shapley computation because exact computation is exponential in region count (typically 20-50 regions), accepting the cost of variance in phi_r that is controlled by lambda_2; we report MC Shapley variance vs. K_shap in the experiment to justify K_shap=32. We chose pairwise ranking loss over absolute score regression because human comparisons are the only reliable supervision available in PICD, accepting the cost that S is identified only up to a monotone transform, which is exactly what the monotonicity constraint exploits. A domain expert would NOT describe this as a variant of the perception-simulation pipeline of the architectural interior design work, because that pipeline evaluates with a frozen simulated perception model and never back-propagates perceptual disagreement into the composition score; here the perceptual ensemble and the composition kernel share the region sufficient statistic and are trained under a monotonicity constraint that the prior pipeline lacks.

**Why this measures what we claim.** The region score s_r plus interaction term g_{r,r'} measures the region's contribution to the composition judgment because the Shapley value of the linear-plus-bilinear scorer equals the region's marginal contribution under the assumption that regions interact only pairwise and with low rank; this assumption fails when regions interact strongly beyond pairwise (e.g., a subject, its background, and a third occluder jointly determine balance), in which case s_r reflects the region's standalone plus pairwise-adjusted credit rather than its full interaction-adjusted credit. The monotonicity constraint dS/dP = gamma > 0 measures the coupling between composition score and perceptual score because gamma is the derivative of S with respect to the ensemble LCB min_k P_k under the assumption that the ensemble LCB is a sufficient statistic for human perceptual judgment (assumption A); this assumption fails when the ensemble is misspecified on a narrow category distribution, in which case gamma measures coupling to the misspecified ensemble rather than to human perception, and we report the Pearson correlation between P and held-out human perceptual ratings as the gate that detects this failure. The pairwise ranking loss measures human preference consistency because the margin term enforces S(I+) > S(I-) + margin under the assumption that human comparisons are transitive and noise-free; this assumption fails when annotators disagree, in which case the loss measures agreement with the majority preference rather than with every annotator.

## Contribution

['A differentiable region-attributed composition kernel (MCK) whose composition score is monotone in a jointly-trained perceptual score by construction, with region-level Shapley-style attributions identifiable in closed form from a linear region scorer.', 'A joint training procedure that couples the composition optimization objective and the perceptual evaluation through a shared region-level sufficient statistic, trained on pairwise human comparisons with a monotonicity constraint.', 'An empirical demonstration that the monotone coupling prevents the score-perception disagreement observed when composition optimization and perception simulation are trained independently.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | PICD pairwise comparisons | Human preferences are the only reliable composition supervision. |
| Primary metric | Pairwise ranking accuracy | Directly measures agreement with held-out human preferences. |
| Baseline | PICD-trained composition embedding model | Tests whether explicit region attribution beats global embedding. |
| Baseline | Frozen perception-simulation pipeline | Tests whether back-propagated coupling beats frozen perceptual evaluation. |
| Baseline | MLLM zero-shot composition judge | Tests whether region-grounded scoring beats holistic VLM judgment. |
| Ablation-of-ours | MCK without monotone coupling (gamma=0) | Isolates contribution of the monotone perceptual constraint. |
| Ablation-of-ours | MCK with linear-only region scorer (no bilinear term) | Tests whether interaction-aware Shapley is non-degenerate. |
| Diagnostic | Pearson correlation between ensemble P and held-out human perceptual ratings (512-image calibration set, 5-point Likert) | Gates the P-sufficiency assumption A. |
| Diagnostic | MC Shapley variance vs. K_shap in {8, 16, 32, 64} | Justifies K_shap=32. |
| Transfer | Train on PICD, test on held-out category split | Tests shared-sufficient-statistic generalization. |

### Why this setup validates the claim

The central claim is that an interaction-aware region scorer with Shapley-style attributions, coupled monotonically to a jointly-trained perceptual ensemble, better predicts human composition preferences than methods that either score images holistically or evaluate perception without feeding it back. PICD pairwise comparisons are the right testbed because the ranking loss and the human attribution supervision are both defined on exactly this signal, so a held-out ranking accuracy directly measures the claim rather than a proxy. Each baseline isolates one sub-claim: the embedding baseline tests whether region decomposition adds signal over a global descriptor; the frozen-simulation baseline tests whether joint monotone coupling adds signal over post-hoc perceptual evaluation; the MLLM baseline tests whether explicit region attribution adds signal over holistic judgment. The gamma=0 ablation is the sharpest test of the coupling: if removing monotone coupling does not degrade ranking accuracy, the coupling mechanism is doing no work and the design rationale collapses. The linear-only ablation tests whether the bilinear interaction term makes Shapley attribution non-degenerate: if linear-only matches interaction-aware, the interaction term adds nothing. The P-calibration diagnostic directly tests assumption A: if the Pearson correlation between ensemble P and held-out human perceptual ratings is low (e.g., < 0.4), the coupling is to a misspecified proxy and the transfer claim is not licensed. The MC Shapley variance diagnostic justifies K_shap=32 by showing variance plateaus. The category-split transfer tests whether the shared sufficient statistic generalizes beyond the training category distribution. Ranking accuracy, not absolute score error, is correct because the method identifies S only up to a monotone transform; comparing scores across models would be meaningless, while comparing preference agreement is invariant to that transform.

### Expected outcome and causal chain

**vs. PICD-trained composition embedding model** — On a case where a strong subject sits off-center against a busy background, the embedding baseline produces a middling score because its global descriptor averages subject and background into one vector and cannot credit either region separately. Our method instead assigns s_r to subject and background regions individually plus a bilinear interaction term g_{r,r'} that captures their joint balance, and its ranking loss pushes the preferred image's region scores above the dispreferred one's, because the interaction-aware scorer makes each region's marginal contribution identifiable. We expect a noticeable ranking-accuracy gap concentrated on images with many small regions, with near-parity on single-subject clean-background images.

**vs. Frozen perception-simulation pipeline** — On a case where the simulated perception model systematically misjudges a composition (e.g., it rewards symmetry that humans find static), the frozen pipeline produces a score aligned with the simulation's bias because perceptual disagreement never back-propagates into the composition score. Our method instead trains the perceptual ensemble jointly under dS/dP = gamma > 0 against the ensemble LCB min_k P_k, so the optimizer must reconcile the perceptual signal with human pairwise preferences rather than inheriting the simulation's error. We expect the gap to appear specifically on images where the frozen simulator's ranking disagrees with human ranking, and to shrink toward parity elsewhere.

**vs. MLLM zero-shot composition judge** — On a case requiring fine-grained region credit, such as a slightly tilted horizon plus an off-balance subject, the MLLM produces a coarse holistic verdict because it has no explicit region decomposition and no calibrated preference signal. Our method instead decomposes the image, scores each region with the interaction-aware scorer, and its Shapley attributions localize the preference difference to the horizon and subject regions, because the region sufficient statistic is shared between scorer and perceptual ensemble. We expect an accuracy gap that widens with region count and composition subtlety, with parity on grossly obvious preference pairs.

**vs. MCK without monotone coupling (gamma=0)** — On a case where perceptual quality and composition quality diverge, the ablation produces a composition score that ignores the perceptual ensemble entirely because gamma is zero, so the two objectives drift apart during training. Our method instead keeps dS/dP positive, forcing the composition score to track the ensemble LCB, because the softplus-constrained gamma couples them. We expect the ablation to match us on pairs where perception and composition agree and to lose accuracy specifically on the divergent subset, which is the diagnostic signature of the coupling's contribution.

**vs. MCK with linear-only region scorer (no bilinear term)** — On a case where two regions jointly determine balance (e.g., subject and background), the linear-only ablation assigns each region a standalone score and cannot credit their interaction, so its Shapley attributions are degenerate (equal to s_r minus a constant). Our method instead includes the low-rank bilinear term g_{r,r'}, making Shapley values interaction-aware. We expect the linear-only ablation to lose ranking accuracy on images with strong region interactions and to show near-zero variance in φ_r across regions, which is the diagnostic signature of degenerate attribution.

### What would falsify this idea

If the gain over the frozen-simulation baseline is uniform across all subsets rather than concentrated where the simulator's ranking disagrees with humans, the monotone coupling is not doing the claimed work and the central mechanism is wrong. Likewise, if the gamma=0 ablation matches the full model on the divergent-perception subset, the coupling constraint is inert and the design rationale fails. Additionally, if the Pearson correlation between ensemble P and held-out human perceptual ratings is below 0.4, assumption A is violated and the coupling-to-human-perception claim is not licensed.

## References

1. Can Machines Understand Composition? Dataset and Benchmark for Photographic Image Composition Embedding and Understanding
2. Automatic Image Composition Optimization and Visual Perception Simulation Evaluation for Architectural Interior Design
3. Visual Instruction Tuning
4. Scaling Instruction-Finetuned Language Models
5. Self-Instruct: Aligning Language Models with Self-Generated Instructions
