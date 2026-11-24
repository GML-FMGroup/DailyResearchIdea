# Identifiability-Constrained Latent State Distillation for Visual Agents

## Motivation

Trajectory-selection criteria in visual-agent distillation—skill-context coverage and ordered visual-decision similarity as used by the Distilling Visual Agents from Limited Interaction Experience framework—are computed purely from observed action/observation statistics, so they establish no identifiability link between a selected context and the student's latent decision state: two distinct latent beliefs can produce identical selection scores and be treated as interchangeable training contexts. This structural gap recurs across distillation pipelines that inherit selection statistics as proxies for shared decision context, including VLM-guided representation distillation (E1/E5), where semantic supervision is assumed to transfer without any guarantee that the student's representation separates contexts the teacher treats differently. The root cause is that the selection score is a function of observable trajectories only, never of the student's belief state, so collisions in the selection metric are invisible to the objective and are silently distilled as if they were equivalent experience.

## Key Insight

Selection-equivalent contexts are only valid distillation targets when the student's latent decision state is injective with respect to the observable selection signature, and enforcing this injectivity as a contrastive separation constraint converts coverage and similarity from observed statistics into identifiable proxies.

## Method

**What it is.** ICID (Identifiability-Constrained Interaction Distillation) is an on-policy distillation procedure for visual agents that augments the OPD loss with a latent-separation regularizer. Inputs: a set of recorded interaction histories, a teacher policy, and a student encoder-policy. Outputs: student parameters whose latent decision state is separable across selection-equivalent-but-belief-distinct contexts.

**How it works.**
```pseudocode
INPUT: histories H, teacher pi_T, student pi_S (encoder f_S, policy head g_S)
HYPER: context window W=8, temperature tau=0.1, margin m=0.5, weight lambda=0.3, K=4 negatives,
       teacher entropy threshold H_max=0.5 nats, consistency filter: 3 perturbed observations per context,
       perturbation sigma=0.05 pixel-space Gaussian, consistency tolerance delta=0.1

PHASE 1 - Selection signature (frozen, from prior framework):
  for each trajectory tau in H:
    s(tau) = (skill_context_coverage(tau), ordered_visual_decision_similarity(tau))
  bucket trajectories by quantized s(tau)  -> selection-equivalent groups G

PHASE 2 - Belief-collision mining (with calibration filter):
  for each group G with |G|>=2:
    for each pair (tau_i, tau_j) in G:
      compute teacher action distributions pi_T(.|o_t) along both
      # Calibration filter: only mine pairs where teacher is confident and divergence is stable
      H_i = entropy(pi_T(.|o_i)); H_j = entropy(pi_T(.|o_j))
      if H_i > H_max or H_j > H_max: skip pair (ambiguous teacher)
      # Consistency check: divergence must be stable under small observation perturbations
      KL_base = KL(pi_T(.|o_i) || pi_T(.|o_j))
      for each of 3 perturbed observations o_i', o_j':
        KL_pert = KL(pi_T(.|o_i') || pi_T(.|o_j'))
        if |KL_pert - KL_base| > delta: skip pair (unstable)
      collision(tau_i,tau_j) = 1 if KL_base > m else 0
    retain only collision pairs as contrastive anchors

PHASE 3 - OPD + identifiability regularizer (with confidence gating):
  for each OPD step on selected task:
    roll out student -> fresh responses at fixed histories
    L_OPD = KL(pi_T(.|o) || pi_S(.|o))
    for each collision pair (a,b):
      z_a = f_S(o_a); z_b = f_S(o_b)
      # Gate lambda by teacher confidence: lambda_pair = lambda * min(1, (H_max - max(H_a,H_b))/H_max)
      L_id = -log( exp(sim(z_a,z_b)/tau) /
                    sum_{k=1..K} exp(sim(z_a,z_k)/tau) )
      where z_k are negatives drawn from OTHER selection-equivalent groups
    L = L_OPD + lambda * L_id
  update f_S, g_S
```

**Why this design.** We chose to mine collision pairs using the teacher's action-distribution divergence rather than the student's own log-probabilities because teacher distributions provide a stable, externally grounded notion of belief difference that does not depend on the student's current (possibly miscalibrated) confidence, accepting the cost that teacher queries are required during pair mining. We chose a supervised-contrastive InfoNCE form over an explicit information-bottleneck term because InfoNCE directly penalizes latent-state collisions with a well-defined negative-sampling procedure, accepting the cost that the negative pool must be curated to avoid trivial separability. We chose to draw negatives from other selection-equivalent groups rather than uniformly from the batch so that the regularizer specifically targets the failure mode (within-group collisions) rather than generic representation spread, accepting the cost that group construction depends on the frozen selection signature. A domain expert would not describe this as a variant of standard representation distillation (e.g., DGC in E1/E5) because DGC distills semantic guidance from VLM features without any constraint tying the student's latent state to the selection statistic; ICID makes the selection statistic itself a function that must be injective into latent space, which no prior distillation objective enforces.

**Why it measures what we claim.** The teacher action-distribution divergence `KL(pi_T_i || pi_T_j) > m` measures belief distinction between two contexts because the assumption is that the teacher's policy is a sufficient statistic of its latent belief; this assumption fails when the teacher is itself uncertain or when two distinct beliefs map to similar action distributions, in which case the divergence reflects action-level ambiguity rather than genuine latent difference. The InfoNCE loss over student latents `sim(z_a, z_b)` measures latent-state collision because the assumption is that the student encoder's representation is a faithful encoding of the decision state (i.e., downstream policy performance is monotone in latent separation); this assumption fails when the encoder collapses dimensions irrelevant to the current task, in which case the loss reflects generic feature similarity rather than decision-relevant belief separation. The negative pool drawn from other selection-equivalent groups measures cross-group separability because the assumption is that selection signatures partition the belief space; this assumption fails when the signature is coarse, in which case negatives may come from beliefs that are actually similar and the contrastive signal is weakened. The calibration filter (entropy threshold and perturbation consistency) measures the reliability of the teacher-KL signal because the assumption is that low-entropy, perturbation-stable divergences correspond to genuine belief differences; this assumption fails when the teacher is confidently wrong or when the perturbation check is too lax, in which case the filter admits aliased pairs and the regularizer fires on action ambiguity.

## Contribution

(1) A distillation framework, ICID, that introduces an identifiability constraint—an InfoNCE regularizer over student latents anchored by teacher action-distribution divergence—into the on-policy distillation objective for visual agents, making selection-equivalent contexts that hide distinct latent beliefs explicitly separable. (2) A belief-collision mining procedure that uses teacher action-distribution divergence to identify selection-equivalent trajectory pairs with genuinely distinct latent beliefs, providing a concrete operationalization of the identifiability condition. (3) A design principle: selection statistics for trajectory distillation are valid proxies for shared decision context only when paired with a latent-separation constraint, which we instantiate and analyze across Navigation and Sokoban-style visual agent tasks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Navigation + Sokoban interaction histories (E3) | Selection-equivalent groups with belief collisions exist here |
| Primary metric | Success rate on held-out panels | Directly reflects decision-state separation quality |
| Baseline | Matched SFT (E3) | Tests whether gains come from distillation vs plain imitation |
| Baseline | On-policy distillation (OPD) without regularizer (E3) | Isolates the identifiability term's contribution |
| Baseline | 27B teacher reference (E3) | Upper bound for gap recovery |
| Ablation-of-ours | ICID with uniform negatives instead of cross-group | Tests whether group-curated negatives drive the effect |
| Ablation-of-ours | ICID without calibration filter (no entropy threshold, no consistency check) | Tests whether the calibration filter prevents harmful collisions |
| Ablation-of-ours | ICID with calibration filter but no confidence gating (lambda fixed) | Tests whether per-pair gating adds beyond filtering |

### Why this setup validates the claim

The central claim is that OPD alone leaves belief-distinct-but-selection-equivalent contexts collapsed in the student latent, and that an identifiability regularizer keyed to teacher action-distribution divergence repairs this. The design is falsifiable only if each baseline isolates one link. Matched SFT tests whether the gain is merely more imitation signal rather than a latent-separation effect; if SFT matches ICID, the regularizer is doing nothing. Plain OPD tests whether the collision-mining and InfoNCE term add anything beyond the E3 distillation recipe; this is the primary sub-claim test. The 27B teacher anchors the recoverable gap so we can see whether ICID closes more of it than OPD. The ablation with uniform negatives tests the specific mechanism claim that cross-group negatives target within-group collisions rather than generic representation spread. The ablation without the calibration filter tests whether the entropy threshold and consistency check are necessary to avoid firing on action aliasing; if the unfiltered version matches or exceeds ICID, the filter is not the active ingredient. The ablation with fixed lambda tests whether per-pair confidence gating contributes beyond filtering. Success rate on held-out panels is the right metric because the predicted failure mode is decision error on contexts that are selection-equivalent but belief-distinct; a representation-spread metric would not distinguish this from harmless feature diversity, whereas success on panels containing such contexts directly detects whether the latent separation translated into decisions.

### Expected outcome and causal chain

**vs. Matched SFT** — On a case where two histories share the same selected skill and near-identical ordered visual-decision signatures but diverge in teacher belief (e.g., a Navigation junction that looks selection-equivalent but implies different goal-side belief), SFT produces a student that maps both to the same latent and picks the majority action because it only minimizes action imitation without any latent-separation pressure. Our method instead mines that pair as a collision and pushes their latents apart via InfoNCE, so the policy head can condition on the difference. We expect a noticeable success-rate gap on the collision-heavy subset and near-parity on collision-free subsets.

**vs. On-policy distillation (OPD) without regularizer** — On a case where OPD rolls out fresh student responses at fixed histories that are selection-equivalent but belief-distinct, the student's miscalibrated confidence makes the KL signal weak exactly where it matters, so OPD improves overall success but leaves within-group collisions unresolved because nothing in its loss references the selection statistic. Our method instead uses teacher action-distribution divergence as an external belief probe to select contrastive anchors, so the regularizer fires precisely on the collapsed pairs. We expect OPD to recover part of the teacher gap broadly, while ICID shows an additional concentrated gain on within-group collision panels with parity elsewhere.

**vs. 27B teacher reference** — On a case where the teacher itself is uncertain and two distinct beliefs map to similar action distributions, the teacher reference sets a ceiling that no distillation can exceed, and the baseline comparison here is about gap recovery rather than beating the teacher. Our method instead should recover more of the measured gap than OPD because the identifiability term targets the residual collapse that OPD leaves. We expect ICID to close a larger fraction of the teacher gap than OPD, with the largest incremental recovery on the collision-heavy subset.

**vs. ICID with uniform negatives** — On a case where within-group collisions are the dominant failure but uniform negatives are drawn from visually dissimilar contexts, the contrastive term is satisfied by trivial separability, so the ablation shows representation spread without decision-relevant separation and its success rate tracks OPD rather than ICID. Our method instead draws negatives from other selection-equivalent groups, forcing the encoder to separate beliefs that the selection signature cannot. We expect the ablation to show a clear deficit on collision-heavy panels and near-parity on easy panels, confirming the negative-pool curation is the active ingredient.

**vs. ICID without calibration filter** — On a case where the teacher is uncertain (high entropy) or the divergence is unstable under small observation perturbations, the unfiltered version mines these pairs as collisions and the InfoNCE term pushes apart latents that should be identical, actively harming performance. Our method instead skips such pairs via the entropy threshold and consistency check, so the regularizer only fires on confident, stable divergences. We expect the unfiltered ablation to underperform ICID on collision-heavy panels and possibly on all panels, confirming that the calibration filter prevents harmful collisions.

**vs. ICID with fixed lambda (no confidence gating)** — On a case where a collision pair has moderate but non-negligible teacher entropy, the fixed-lambda version applies full weight to a potentially noisy signal, while the gated version scales lambda down. We expect the gated version to show a small but consistent improvement on collision-heavy panels, with the largest effect on panels containing a mix of confident and ambiguous collisions.

### What would falsify this idea

If ICID's gain over plain OPD is uniform across collision-heavy and collision-free subsets rather than concentrated where teacher action-distribution divergence exceeds the margin, the central claim that the identifiability regularizer targets within-group latent collisions is wrong. Equivalently, if the uniform-negative ablation matches ICID, the group-curated negative pool is not the causal mechanism. Additionally, if the unfiltered version matches or exceeds ICID, the calibration filter is not necessary and the load-bearing assumption about teacher-KL sufficiency is not repaired by the filter.

## References

1. Distilling Visual Agents from Limited Interaction Experience
