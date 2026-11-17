# Composition-Invariant Cluster-Frequency Certification for Multi-Task Pretraining Mixtures

## Motivation

Clustered-importance sampling (E2) builds a specialist by reweighting a generalist corpus toward the target domain's cluster-frequency vector, but it defines this only for a single specialist and assumes that specialist mixtures combine additively into a multi-task mixture. When a fleet like K2-Horizon (E6) trains six models of different sizes and architectures (MoE, MoVA, softplus gating) on shared pretraining data, each with its own effective cluster frequencies, additivity silently breaks: a mixture that is cluster-frequency-correct for one specialist can be cluster-frequency-wrong for another, and no released artifact certifies otherwise. The root cause is that cluster-frequency composition is a constrained geometric object, not a linear one, yet existing methods treat it as linear averaging.

## Key Insight

A multi-task mixture is certified iff each specialist's realized cluster-frequency vector lies within a tolerance ball of its single-task optimum under the same shared mixture, which turns composition into a feasibility problem over a polytope whose vertices are the per-specialist cluster weights rather than an additive average.

## Method

**What it is.** CIFC (Cluster-Frequency Certification) takes as input a set of per-task specialist cluster-importance weight vectors estimated from limited domain data, the generalist corpus clustered into K clusters, and a fleet of model configurations; it outputs a single shared mixture weight vector over clusters plus a per-(task, model) certificate stating whether that mixture preserves each specialist's target cluster-frequency vector within tolerance τ. The certificate is explicitly a **marginal cluster-coverage certificate of the input mixture** (a bookkeeping guarantee), not a capability-preservation guarantee; downstream capability is measured separately in the experiment.

**How it works.**

```pseudocode
INPUT: corpus D, K clusters (K=100, k-means on 768-d mean-pooled sentence embeddings),
       T tasks with limited domain data D_t (|D_t| = 5k–50k tokens each),
       fleet configs F = {f_1..f_M} (sizes, MoE, MoVA, gating),
       tolerance τ (default 0.05 in L∞ norm), mixing budget B (max 2 passes over D)

PHASE 1 — Per-task cluster importance (extends E2):
  for each task t:
    p_t = cluster_freq(D_t)                      # target vector, sum 1
    w_t = p_t / cluster_freq(D)                  # importance weights
    normalize w_t to sum 1

PHASE 2 — Composition polytope:
  Build matrix W = [w_1 .. w_T] (K x T)
  Define feasibility set S = { w : w = W a, a in simplex_T,
                               ||w - w_t||_inf <= τ  for all t }
  # a is the task-mixing coefficient vector

PHASE 3 — Solve for certified mixture:
  w* = argmin_{w in S}  sum_t ||w - w_t||_2^2 / T   # barycenter under tolerance
  if S empty: relax τ -> τ' = min τ s.t. S nonempty (bisection, 20 iters)
  a* = coefficients recovered from w*

PHASE 4 — Fleet-level verification:
  for each config f in F:
    train on mixture w* (or replay from released checkpoints)
    measure realized cluster freq r_{f,t} on each task's held-out domain data
    cert_{f,t} = ( ||r_{f,t} - p_t||_inf <= τ )

OUTPUT: w*, a*, {cert_{f,t}}
```

**Assumption A (explicit).** The certificate `cert_{f,t}` is valid only under Assumption A: that the held-out domain-data cluster marginals `p_t` equal the input-mixture cluster marginals the model actually consumes on task t. This holds for dense models trained on the input mixture. It fails for MoE/MoVA configs because routing and value-projection experts reweight clusters away from the input mixture; in that case `r_{f,t}` reflects the input mixture's frequencies, not the model's effective data distribution. To detect this failure mode F, Phase 4 additionally computes a routing-entropy diagnostic per MoE/MoVA config: for each cluster k, the fraction of tokens routed to each expert; if the coefficient of variation of expert load across clusters exceeds 0.3, the certificate is flagged `cert_{f,t} = UNVERIFIED` rather than passed/failed.

**Why this design.** We chose a constrained barycenter over the per-specialist cluster weights rather than a simple additive average (the implicit assumption in E2) because cluster-frequency composition is non-linear: averaging two specialists' weight vectors does not preserve either specialist's target frequency, accepting the cost that the mixture may be Pareto-suboptimal for any single task. We chose an infinity-norm tolerance ball rather than an L2 ball because the failure mode we care about is any single cluster being badly under-represented for a specialist, accepting the cost of a more conservative feasible set. We chose bisection on τ rather than a hard feasibility check because real corpora rarely admit an exact shared mixture, accepting the cost that the certificate is tolerance-relative rather than absolute. A domain expert would not describe this as a variant of clustered-importance sampling (E2) because E2 estimates weights for one specialist and leaves composition unaddressed; CIFC's load-bearing object is the feasibility polytope over shared mixtures, which E2 does not construct.

**Why it measures what we claim.** The quantity ||w - w_t||_inf measures marginal cluster-frequency preservation for specialist t because the assumption is that a specialist's capability is a monotone function of its target cluster frequencies being matched within a per-cluster tolerance; this assumption fails when a specialist's capability depends on cluster co-occurrence structure rather than marginal frequencies, in which case the certificate reflects marginal coverage only. The realized frequency r_{f,t} measures fleet-level composition fidelity because the assumption is that the released K2-Horizon training logs and checkpoints let us reconstruct the effective cluster frequencies each model saw; this assumption fails when MoE routing or MoVA value-projection experts reweight clusters differently from the input mixture, in which case r_{f,t} reflects the input mixture's frequencies, not the model's effective data distribution. To bound the regime in which the monotonicity assumption holds, the experiment includes a controlled synthetic probe (see Evaluation Setup) that holds cluster marginals fixed while varying co-occurrence structure and measures whether downstream capability tracks marginal match or co-occurrence match.

## Contribution

(1) CIFC, a constrained-barycenter composition rule that replaces additive multi-task mixture averaging with a feasibility problem over per-specialist cluster-frequency targets, with a per-(task, model) tolerance certificate. (2) The empirical finding that cluster-frequency composition is non-additive across specialists and that the non-additivity is amplified in MoE/MoVA configurations relative to dense models of matched data. (3) A released certification toolkit and reconstructed cluster-frequency vectors for K2-Horizon's pretraining data and multi-domain logs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Generalist corpus (The Pile, 100 clusters) + limited per-task domain data (5k–50k tokens per task) | Mirrors specialist-from-generalist pretraining regime |
| Primary metric | Certificate pass rate at fixed τ=0.05 | Directly tests tolerance-relative frequency preservation claim |
| Secondary metric | Per-task held-out perplexity and 5-shot accuracy on specialist domains | Links certificate to downstream capability |
| Baseline | Clustered-importance sampling (E2) | Additive per-specialist weights, no shared composition |
| Baseline | Uniform generalist mixture | Ignores specialist cluster frequencies entirely |
| Baseline | Naive additive average of w_t | Tests our non-linearity claim against implicit averaging |
| Ablation-of-ours | L2-ball tolerance instead of L∞ | Isolates the infinity-norm design choice |
| Ablation-of-ours | Routing-entropy diagnostic disabled | Tests whether MoE/MoVA failure mode is detected |
| Synthetic probe | Controlled corpus with fixed marginals, varied co-occurrence | Bounds regime where monotonicity assumption holds |

### Why this setup validates the claim

The central claim is that a shared mixture can be certified to preserve each specialist's target cluster-frequency vector within an infinity-norm tolerance, and that constructing this via a constrained barycenter over the composition polytope yields certificates that additive or uniform mixtures cannot. The dataset choice is load-bearing because it reproduces the exact regime CIFC targets: abundant generalist data plus scarce per-task domain data, so that specialist frequencies must be recovered by reweighting rather than by additional data. The primary metric—certificate pass rate at a fixed τ—is the right detector because it directly operationalizes the claim: if the shared mixture genuinely preserves target frequencies, certificates pass; if it does not, they fail. The secondary metric—per-task held-out perplexity and 5-shot accuracy—links the certificate to downstream capability, so significance rests on demonstrated capability preservation rather than on a self-referential certificate metric. The clustered-importance-sampling baseline (E2) tests the sub-claim that per-specialist weight estimation alone does not yield a certified shared composition, since E2 provides weights for one specialist at a time and never constructs the feasibility polytope. The uniform-mixture baseline tests whether the generalist distribution already happens to satisfy specialist tolerances, which would trivialize the contribution. The additive-average baseline tests our explicit non-linearity argument: if averaging w_t vectors passes certificates as often as the barycenter, our core design rationale is wrong. The L2-ball ablation isolates the infinity-norm choice, testing whether the conservative feasible set is actually necessary for catching single-cluster under-representation. The routing-entropy ablation tests whether the MoE/MoVA failure mode F is detected and whether certificates are correctly flagged UNVERIFIED. The synthetic probe holds cluster marginals fixed while varying co-occurrence structure, measuring whether capability tracks marginal match or co-occurrence match, thereby bounding the regime in which the monotonicity assumption holds. Together these form a falsifiable test: the method must beat averaging specifically on tasks where one cluster is badly under-represented, not uniformly.

### Expected outcome and causal chain

**vs. Clustered-importance sampling (E2)** — On a case where two specialists demand disjoint high-weight clusters, E2 produces separate per-specialist weight vectors and no shared mixture, so any single deployed mixture must arbitrarily favor one specialist because E2 never constructs a joint feasibility set. Our method instead solves for a barycenter inside the polytope defined by both specialists' tolerance balls, because the feasibility set encodes all specialists simultaneously. We expect a large certificate pass-rate gap on multi-task subsets with conflicting cluster demands, and near-parity on single-task or aligned-demand subsets. Downstream, we expect the gap to translate into lower held-out perplexity and higher 5-shot accuracy on the conflicting-demand tasks.

**vs. Uniform generalist mixture** — On a case where a specialist's target distribution concentrates on a rare cluster, the uniform mixture produces realized frequencies dominated by high-mass generalist clusters because it ignores p_t entirely, so the rare cluster is badly under-represented and the infinity-norm certificate fails. Our method instead reweights toward that cluster via w_t and the barycenter, because the tolerance constraint forces per-cluster coverage. We expect the uniform baseline to fail certificates almost entirely on rare-cluster specialists while our method passes most, with the gap shrinking for specialists whose targets resemble the generalist distribution. Downstream, we expect the uniform baseline to show higher perplexity on rare-cluster specialists.

**vs. Naive additive average of w_t** — On a case where two specialists' weight vectors are anti-correlated across clusters, the additive average produces a vector that matches neither specialist's target because composition is non-linear and averaging cancels opposing deviations, driving the infinity norm above τ for both. Our method instead finds the constrained barycenter that stays inside each tolerance ball, because the polytope explicitly enforces per-specialist infinity-norm bounds. We expect the averaging baseline to fail certificates precisely on anti-correlated specialist pairs while our method passes, with parity on correlated pairs—a gap concentrated where the non-linearity argument predicts. Downstream, we expect the averaging baseline to show higher perplexity on anti-correlated pairs.

### What would falsify this idea

If the certificate pass-rate gain over the naive additive average is uniform across all specialist pairs rather than concentrated on anti-correlated or rare-cluster pairs, then the claimed non-linearity and single-cluster failure mode are not the operative mechanism, and the polytope construction is not doing the work we attribute to it. Equally, if the L∞-ball ablation matches the L∞ method everywhere, the infinity-norm design choice is unmotivated and the central tolerance argument collapses. Additionally, if the synthetic probe shows that capability tracks co-occurrence match rather than marginal match, then the monotonicity assumption is false and the certificate must be downgraded to a marginal-coverage bookkeeping guarantee.

## References

1. K2-Horizon: A Frontier Fleet of Open Source Language Models
