# Interventional Data Attribution for LLM Pretraining via Counterfactual Corpus Interventions

## Motivation

Existing pretraining-data attribution methods — task-gram frequency baselines (E1/E7), the MEMOED memorization framework (E3/E8), and clustered-importance sampling (E2/E10) — all infer a data-to-output link from observational co-occurrence statistics, so they cannot distinguish a document that merely correlates with an output from one that causally produces it; a high-frequency term can dominate generation regardless of relevance (E8), and a task-gram correlation can register memorization where the model would produce the same output under any corpus (E7). K2-Horizon (E6/E12) releases pretraining data, intermediate checkpoints, and training logs, but that is transparency infrastructure, not an attribution instrument: it exposes the corpus without any intervention protocol that would let a researcher perturb the corpus and re-train to observe the causal effect. The structural root cause is that no existing pipeline holds the training run fixed while intervening on a corpus cluster and re-measuring the output, so every reported attribution is a correlation that a counterfactual could overturn.

## Key Insight

A data-to-output link is causally identified only when the output-probability shift under a leave-one-cluster-out retraining run exceeds the shift predicted by the observational frequency baseline, because the residual between the two isolates the effect of the intervention from the confound of term frequency.

## Method

**What it is.** We propose CIRCA (Counterfactual Interventional Retraining for Corpus Attribution), a protocol that takes K2-Horizon's released pretraining corpus, intermediate checkpoints, and training logs as inputs and produces, for each (corpus cluster, target output) pair, a signed causal-attribution score with a frequency-baseline residual. Inputs: a corpus C partitioned into K semantically coherent clusters, a set of target prompts/outputs, and a checkpoint schedule from K2-Horizon. Outputs: per-cluster causal-attribution scores and a binary certification of causal vs. merely-correlational influence.

**Load-bearing assumption (stated explicitly).** CIRCA assumes that a continued-pretraining run from a near-converged K2-Horizon checkpoint on a perturbed corpus reaches a new optimization equilibrium within the retraining budget, so that the log-probability shift is attributable to the corpus intervention itself rather than to optimization transient, learning-rate/effective-schedule mismatch, or seed noise. If this assumption fails — i.e., if the weights are still drifting when the delta is measured — then the delta reflects how far the optimizer has moved, not the causal role of the removed or duplicated cluster. The calibration procedure in Phase 3 tests this assumption directly by measuring the null-retraining delta distribution and setting the certification threshold from that empirical null.

**How it works.**

```pseudocode
# Phase 1: cluster the corpus (K2-Horizon pretraining data)
C = K2-Horizon.pretraining_corpus
clusters = semantic_cluster(C, K=64, method="embedding+kmeans")

# Phase 2: observational baseline (task-gram / MEMOED-style)
for each target output o and prompt p:
    freq_baseline[o] = task_gram_frequency(p, o, C)   # E1/E7 baseline
    memoed_baseline[o] = MEMOED_score(o, C)           # E3/E8 baseline

# Phase 3: null-retraining calibration (tests the load-bearing assumption)
# Run identical-corpus retraining from s0 with no intervention, multiple seeds
for seed in {1,2,3,4,5}:
    model_null[seed] = continue_pretrain(
        init=K2-Horizon.checkpoint(s0),
        data=C,                    # no intervention
        steps=N_retrain=2000,
        lr=1e-4,
        seed=seed)
    for each target output o:
        null_delta[seed,o] = log model_null[seed].prob(o|p) - log p_full[o]
# Empirical null distribution per output: mean_null[o], std_null[o]
# tau[o] = 2 * std_null[o]  (replaces re-clustering std)

# Phase 4: interventional retraining runs (removal and duplication)
for each cluster c in clusters:
    # removal intervention
    C_minus_c = C \ c
    model_minus_c = continue_pretrain(
        init=K2-Horizon.checkpoint(s0),
        data=C_minus_c,
        steps=N_retrain=2000,
        lr=1e-4,
        seed=fixed)
    # duplication intervention
    C_plus_c = C with cluster c duplicated (2x weight)
    model_plus_c = continue_pretrain(
        init=K2-Horizon.checkpoint(s0),
        data=C_plus_c,
        steps=N_retrain=2000,
        lr=1e-4,
        seed=fixed)
    for each target output o:
        p_full[o]  = model_full.prob(o | p)      # full-corpus reference
        p_minus[o] = model_minus_c.prob(o | p)
        p_plus[o]  = model_plus_c.prob(o | p)
        delta_causal[c,o] = log p_full[o] - log p_minus[o]
        delta_counterfactual[c,o] = log p_plus[o] - log p_full[o]

# Phase 5: certify causal link with null-calibrated threshold
for each (c,o):
    residual[c,o] = delta_causal[c,o] - alpha * freq_baseline[o]
    # alpha fit on held-out clusters via least squares
    certified[c,o] = (|residual[c,o]| > tau[o]) and (sign(delta_causal) == sign(delta_counterfactual))
```

**Why this design.** We chose leave-one-cluster-out retraining over leave-one-document-out because document-level interventions are dominated by noise at the 0.9B–375B scale, accepting that cluster granularity may merge causally distinct documents into one unit. We chose to anchor every intervention at the same K2-Horizon intermediate checkpoint (step s0) rather than retraining from scratch, because the released checkpoint schedule (E6/E12) makes the intervention affordable and holds all pre-s0 training fixed, accepting that effects learned before s0 are invisible to the protocol. We chose a two-sided intervention (remove ∩ duplicate) over a single removal, because a frequency baseline can only be separated from a causal effect when the effect is shown to be monotone in the intervention direction; a one-sided removal cannot distinguish 'no effect' from 'saturated effect', accepting that the duplicate run doubles the compute budget. We chose a fixed 2000-step retraining horizon over full convergence because the checkpoint is already near-converged and the shift signal saturates early, accepting that weak causal effects below the 2000-step noise floor are missed. We added a null-retraining calibration (Phase 3) that runs the identical corpus with no intervention across five seeds to empirically measure the optimization-transient floor and set the certification threshold tau[o] from that null distribution, because the load-bearing assumption that 2000 steps reaches a new equilibrium must be tested rather than asserted. A domain expert would not describe CIRCA as a variant of task-gram attribution (E1/E7) or MEMOED (E3/E8) because both are pure observational frequency estimators with no retraining intervention, and would not describe it as clustered-importance sampling (E2/E10) because that method reweights the training distribution without ever measuring the counterfactual output shift. CIRCA's load-bearing operation — retraining a checkpoint on a perturbed corpus and differencing log-probs — has no counterpart in any of the three.

**Why it measures what we claim.** The quantity `delta_causal[c,o] = log p_full[o] - log p_minus_c[o]` measures causal necessity of cluster c for output o because the only thing that differs between the two models is the presence of cluster c in the training corpus, with checkpoint, seed, step count, and learning rate held fixed; this assumption fails when the retraining run at 2000 steps has not yet re-equilibrated the weights, in which case the delta reflects optimization transient rather than the corpus intervention. The quantity `delta_counterfactual[c,o]` measures causal sufficiency because duplicating cluster c is the minimal intervention that increases its causal contribution without changing the vocabulary or the other clusters; this assumption fails when duplicating a cluster shifts the effective learning-rate schedule or the tokenizer frequency in a way that is not a pure weight change, in which case the delta reflects optimization dynamics rather than corpus influence. The quantity `residual[c,o] = delta_causal - alpha * freq_baseline` measures the portion of the output shift not explained by observational frequency, because `alpha` is fit on held-out clusters where the true causal effect is assumed zero (clusters whose documents never co-occur with target o); this assumption fails when held-out clusters are themselves causally relevant through indirect pathways, in which case `alpha` is biased and the residual under- or over-states the causal effect. The certification rule `sign(delta_causal) == sign(delta_counterfactual)` measures directional consistency of the causal link because a genuinely causal cluster must both help when present and hurt when removed; this assumption fails when the model has redundant pathways (other clusters can substitute), in which case a causal cluster can show removal effect but no duplication effect and be wrongly decertified. The null-retraining calibration (Phase 3) measures the optimization-transient floor because it runs the identical corpus with no intervention across five seeds and records the log-probability shift distribution; this assumption fails when the null distribution is not representative of the intervention runs (e.g., if the intervention changes the effective learning-rate schedule), in which case tau[o] is mis-calibrated and certification is too lenient or too strict.

## Contribution

(1) CIRCA, an interventional pretraining-data attribution protocol that replaces observational frequency statistics with leave-one-cluster-out and duplicate-cluster retraining runs anchored at released intermediate checkpoints, producing signed causal-attribution scores with a frequency-baseline residual. (2) A certification criterion (residual magnitude + directional sign agreement) that separates causal from correlational data influence, with a design principle that two-sided interventions are necessary because one-sided removal cannot distinguish absence of effect from saturation. (3) A reusable partitioning-and-intervention harness over K2-Horizon's released corpus, checkpoints, and logs that makes corpus-level counterfactual retraining reproducible at 0.9B–36B scale.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | K2-Horizon released pretraining corpus + intermediate checkpoints | Only open corpus with matching checkpoint schedule |
| Primary metric | Signed residual causal-attribution score on held-out targets | Separates causal shift from frequency baseline |
| Baseline | Task-gram frequency attribution (E1/E7) | Pure observational frequency estimator, no intervention |
| Baseline | MEMOED memorization attribution (E3/E8) | Observational culture-conditioned frequency scoring |
| Baseline | Clustered-importance sampling (E2/E10) | Reweights distribution without counterfactual measurement |
| Ablation-of-ours | CIRCA with removal-only (no duplication) | Tests whether two-sided intervention is load-bearing |
| Ablation-of-ours | CIRCA with null-retraining calibration (Phase 3) vs. fixed tau from re-clustering std | Tests whether empirical null calibration is load-bearing |

### Why this setup validates the claim

The central claim is that corpus influence on a target output can be certified as causal rather than merely correlational, and only an intervention that retrains on a perturbed corpus and differences log-probs can separate the two. The task-gram and MEMOED baselines each test a distinct sub-claim: task-gram tests whether frequency alone explains the output shift, and MEMOED tests whether memorization-from-documents alone explains it, so if CIRCA's residual is non-trivial on cases where both baselines predict the shift, the observational account is insufficient. Clustered-importance sampling tests the third sub-claim: that reweighting the training distribution is not the same as measuring a counterfactual output shift, so its scores should track frequency rather than the residual. The removal-only ablation is the diagnostic key, because it isolates whether the two-sided intervention (remove and duplicate) is what produces the signed, directionally-consistent attribution — if removal-only matches CIRCA, the duplication half is decorative. The null-retraining calibration ablation tests the load-bearing assumption directly: if certification with tau from the empirical null distribution (Phase 3) does not change the set of certified pairs relative to a fixed tau from re-clustering std, then the optimization-transient floor is negligible and the original threshold was adequate; if it does change, the calibration is load-bearing and the assumption is empirically tested. The primary metric is the signed residual rather than raw delta, because raw delta is exactly what the frequency baselines already capture; the residual is the only quantity that can distinguish the causal signal from the observational one, so it is the right quantity to detect the predicted effect.

### Expected outcome and causal chain

**vs. Task-gram frequency attribution (E1/E7)** — On a case where a target factual output appears in a high-frequency cluster but is also supported by a low-frequency cluster, the task-gram baseline produces a large attribution score for the high-frequency cluster because it counts co-occurrence and cannot see that the low-frequency cluster is causally necessary. Our method instead removes each cluster and retrains, so the low-frequency cluster shows a real log-prob drop while the high-frequency one shows a smaller residual after the frequency baseline is subtracted. We expect a noticeable gap between CIRCA residual and task-gram score concentrated on low-frequency-but-causally-necessary clusters, with parity on clusters where frequency and causality coincide.

**vs. MEMOED memorization attribution (E3/E8)** — On a case where a culture-conditioned output is generated with a memorized symbol from a frequent document, the MEMOED baseline produces a high memorization score because it measures document-level frequency association and cannot distinguish memorized recall from generalized reasoning. Our method instead duplicates the cluster and observes whether the output probability rises directionally with the removal effect, so a cluster that is merely frequent but substitutable shows removal effect without duplication effect and is decertified. We expect MEMOED and CIRCA to agree on strongly memorized symbols but diverge sharply on outputs the model produces through generalization, where CIRCA certifies few clusters and MEMOED assigns broad high scores.

**vs. Clustered-importance sampling (E2/E10)** — On a case where a domain-specific target output is rare in the generalist corpus, the clustered-importance baseline produces a reweighted training distribution that boosts the relevant clusters but never measures whether the output probability actually shifts, so it cannot tell an effective reweighting from an inert one. Our method instead runs the same cluster through the removal and duplication interventions and differences log-probs, so it reports a signed causal score per cluster rather than an importance weight. We expect the importance-sampling weights and CIRCA residuals to correlate weakly, with a substantial subset of high-weight clusters showing near-zero certified causal effect because they are frequent but redundant.

**vs. CIRCA with removal-only ablation** — The removal-only ablation omits the duplication intervention and certifies based on removal effect alone. We expect it to over-certify clusters that are frequent but substitutable (showing removal effect without duplication effect) and to miss clusters whose causal role is masked by redundancy. The full two-sided CIRCA should produce a smaller certified set with higher precision on held-out ground-truth causal clusters.

**vs. CIRCA with fixed tau from re-clustering std (no null calibration)** — This ablation uses the original threshold tau = 2 * std of delta across 5 random re-clusterings instead of the empirical null distribution from Phase 3. We expect the fixed-tau version to certify more pairs, including some that fall within the optimization-transient floor measured by null retraining, especially for small clusters where the transient is large relative to the causal effect. The null-calibrated CIRCA should show a lower false-certification rate on placebo clusters (clusters with no direct or indirect pathway to the target).

### What would falsify this idea

If CIRCA's certified causal residual were uniformly distributed across clusters and targets rather than concentrated on the cases where the frequency baselines predict the shift, then the residual would not be measuring anything beyond observational frequency and the central claim would be wrong. Equally, if the removal-only ablation matched the two-sided CIRCA on certified attributions, the duplication intervention would be non-load-bearing and the causal-certification story would collapse. Additionally, if the null-retraining calibration showed that the optimization-transient floor (std_null) is comparable to or larger than the residual for most clusters at the 2000-step horizon, then the load-bearing assumption fails and the method should be dropped rather than patched.

## References

1. K2-Horizon: A Frontier Fleet of Open Source Language Models
