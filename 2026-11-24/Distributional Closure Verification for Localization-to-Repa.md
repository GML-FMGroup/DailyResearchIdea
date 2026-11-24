# Distributional Closure Verification for Localization-to-Repair Pipelines

## Motivation

Localization-to-repair pipelines such as the fixer-likelihood localizer of 'Let the Fixer Decide' treat the frozen fixer as a reliable likelihood oracle: the localizer proposes sites, the fixer ranks patches by assigned likelihood, and patch validity is inferred from that ranking rather than verified. Structurally, no invariant ties the top-scored patch to the fixer's own generation distribution, so a localization can win simply because the fixer assigns high likelihood to a plausible-looking but non-repairing patch; this is the recurring meta-level failure of inferred-likelihood ranking across such pipelines, where the score used for selection is never checked against the operator that would actually have to produce the patch. The root cause is that ranking and generation are never closed into a single verified loop, so the oracle assumption survives unchallenged.

## Key Insight

The top-scored patch is a valid repair under the fixer's own distribution exactly when its fixer-assigned likelihood is not an outlier relative to patches the frozen fixer actually samples for that localization, which turns an unverified ranking assumption into a testable closure invariant on the fixer's generation distribution.

## Method

**What it is.** We introduce DCV (Distributional Closure Verification), a post-hoc gate on a frozen localization-to-repair pipeline. Inputs: a buggy program, a localizer producing candidate localizations L = {l_1..l_k}, and a frozen fixer F that scores patches. Output: a re-ranked localization list in which any localization whose top-scored patch fails the closure invariant is demoted, and a per-localization closure statistic.

**How it works.**

```pseudocode
INPUT: buggy program p, localizations L={l_1..l_k}, frozen fixer F,
       sample budget M=32, threshold tau=0.05, temperature T=1.0,
       calibration set C of (localization, known-valid patch) pairs
OUTPUT: re-ranked localizations, closure statistics c(l)

# Pre-computation on calibration set C (held-out, disjoint from test bugs):
# For each calibration localization l_c with its known-valid patch y_valid:
#   S_lc = [ F.sample(p_c, l_c, T=1.0) for _ in range(M) ]
#   c_valid(l_c) = (1 + #{y in S_lc : F.logprob(y|p_c,l_c) >= F.logprob(y_valid|p_c,l_c)}) / (M+1)
# Fit a calibration mapping g: context_length(l) -> expected c_valid quantile
#   (e.g., bin by context length into 10 quantile bins, compute 5th percentile of c_valid per bin)

for each localization l in L:
    # 1. Top-scored patch (existing pipeline behavior)
    y_top = argmax_{y in F.propose(p,l)} F.logprob(y | p, l)

    # 2. Empirical generation distribution of the FROZEN fixer at l
    S_l = [ F.sample(p, l, T=T) for _ in range(M) ]        # M i.i.d. samples
    lp_S = [ F.logprob(y | p, l) for y in S_l ]

    # 3. Closure statistic: rank of y_top inside the fixer's own sample
    #    distribution, expressed as a right-tail mass estimate.
    c(l) = (1 + #{y in S_l : F.logprob(y|p,l) >= F.logprob(y_top|p,l)}) / (M + 1)

    # 4. Context-matched closure invariant: y_top must be no more of an outlier
    #    than a known-valid patch would be at this localization's context length.
    #    tau(l) = g(context_length(l)) from calibration; default tau=0.05 if bin empty.
    closure_pass(l) = (c(l) >= tau(l))

return re-rank(L by (closure_pass(l), F.logprob(y_top|l)))
```

The re-ranking places closure-passing localizations first, ordered by top-patch likelihood; failing localizations are demoted but retained for inspection. The statistic c(l) is a right-tail mass under the fixer's sampling distribution, estimated by rejection/importance sampling over M draws at temperature T. The threshold tau(l) is calibrated per context-length bin on a held-out set of known-valid patches, so the invariant tests whether y_top is more of an outlier than a known-valid patch would be at the same localization context length, rather than applying a fixed absolute tail cutoff.

**Why this design.** We chose to sample from the frozen fixer at inference temperature T=1.0 rather than reuse the fixer's internal top-k logits, because the closure question is about the operator's actual generation distribution, and top-k logits are a truncated surrogate that hides sampling mass; the cost is M=32 extra fixer calls per localization, which is why we cap k and share the sample budget across localizations. We chose a rank-based tail statistic c(l) over a raw logprob threshold because raw logprobs are not comparable across localizations with different context lengths, whereas the rank of y_top inside its own sample is scale-free; the trade-off is that rank discards magnitude information, so two localizations that both pass closure are still ordered by raw likelihood. We chose a context-matched threshold tau(l) calibrated on known-valid patches over a fixed absolute tau because the absolute tail mass of a valid patch varies with localization context length and patch brevity; the cost is a small calibration set and a binning procedure, which we mitigate by falling back to tau=0.05 for empty bins. We chose a hard demotion gate at tau(l) over a soft re-weighting because the motivation is to break the oracle assumption with a verified invariant rather than to blend another heuristic score; the cost is a sharp decision boundary that can misclassify borderline localizations, which we mitigate by retaining demoted localizations in the output. Unlike the fixer-likelihood localizer of 'Let the Fixer Decide', which uses the fixer's score purely as a training weight and never checks it against the fixer's generation distribution, DCV performs an explicit sampling-based closure test; a domain expert would not describe DCV as a variant of that localizer because DCV changes no localizer weights and instead audits the fixer's own operator.

**Why it measures what we claim.** The right-tail mass c(l) measures distributional validity of the top patch because it assumes that a genuine repair is a high-probability, reproducible output of the frozen fixer conditioned on the correct localization; this assumption fails when the fixer's likelihood is systematically inflated for short or repetitive patches, in which case c(l) reflects patch brevity rather than repair validity. The sample budget M=32 measures the fixer's generation distribution because it assumes i.i.d. sampling at T=1.0 is representative of the operator's output mass; this assumption fails under heavy-tailed patch distributions where 32 draws miss the tail, in which case c(l) reflects sampling noise rather than tail mass. The context-matched threshold tau(l) measures the closure invariant because it assumes that a valid repair should be no more of an outlier than a known-valid patch at the same context length; this assumption fails when the calibration set is unrepresentative of the test localizations, in which case the gate reflects calibration mismatch rather than localizer error. The linking assumption A is that the frozen fixer's T=1.0 sampling distribution is calibrated so genuine repairs are reproducible high-probability outputs; the failure mode F is that likelihood is inflated for short/repetitive patches or 32 draws miss a heavy tail, in which case c(l) reflects brevity or sampling noise rather than repair validity.

## Contribution

(1) DCV, a sampling-based closure-verification gate that converts the fixer-likelihood ranking assumption into a testable invariant on the frozen fixer's generation distribution, requiring no retraining. (2) A design principle: localization quality should be judged by whether the fixer can reproduce its own top patch under sampling, not by the magnitude of the fixer's likelihood score. (3) A per-localization closure statistic and re-ranking procedure that is drop-in compatible with any frozen fixer and any localizer.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Defects4J-style bug set with multi-candidate localizations (k=5 per bug, 200 bugs) | Realistic localization ambiguity drives closure test |
| Primary metric | Top-1 localization accuracy after re-ranking | Directly measures whether gate promotes correct localization |
| Baseline | Unmodified frozen localizer ranking | Tests whether closure gate adds signal over raw ranking |
| Baseline | Raw top-patch logprob re-ranking | Isolates sampling-based closure from likelihood heuristic |
| Baseline | Random demotion at matched rate | Controls for any gain from merely shuffling order |
| Baseline | Sample-agreement demotion (pass@k-style) | Controls for any gain from sampling consistency per se |
| Ablation-of-ours | DCV with tau=0 (no demotion) | Confirms effect comes from the closure gate itself |
| Ablation-of-ours | DCV with fixed tau=0.05 (no context-matched calibration) | Tests whether calibration removes context-length confound |
| Ablation-of-ours | DCV with length-normalized logprob in c(l) | Tests whether c(l) survives controlling for patch brevity |
| Calibration set | 100 held-out bugs with known-valid patches, disjoint from test | Provides context-length bins for tau(l) |

### Why this setup validates the claim

The central claim is that a frozen fixer's own generation distribution contains information the localizer ignores, and that auditing the top patch's typicality re-ranks localizations toward the correct one. The unmodified-ranking baseline tests the null that existing order is already optimal; the raw-logprob baseline tests whether any likelihood signal suffices, isolating the sampling-based closure statistic as the causal ingredient; the random-demotion baseline tests whether gains come from reordering per se rather than from the invariant; the sample-agreement baseline tests whether any sampling-based filter suffices, isolating the rank-tail statistic from mere consistency. The tau=0 ablation pins the effect on the gate rather than on incidental pipeline changes; the fixed-tau ablation tests whether context-matched calibration is necessary to avoid the brevity/context-length confound; the length-normalized ablation tests whether c(l) survives controlling for patch length. Top-1 localization accuracy is the right metric because the predicted effect is a promotion of the correct localization into first place; if closure statistics carry real signal, accuracy should rise specifically on cases where the correct localization was present but ranked below a spurious one, and stay flat where it was already first. This combination makes the claim falsifiable: a gain that appears equally on both subsets, or that survives random demotion or sample-agreement demotion, would refute the mechanism.

### Expected outcome and causal chain

**vs. Unmodified frozen localizer ranking** — On a case where the localizer ranks a short, repetitive region first because the fixer assigns it a high top-patch logprob, the baseline keeps that wrong localization at rank one because it never checks whether the top patch is reproducible under sampling. Our method samples the frozen fixer at that localization and finds the top patch is a tail outlier never reproduced, so c(l) falls below the context-matched tau(l) and the localization is demoted. We expect a noticeable top-1 gain concentrated on cases where the correct localization was initially ranked second or third, with parity on cases already correctly ranked first.

**vs. Raw top-patch logprob re-ranking** — On a case where two localizations both yield inflated top-patch likelihoods, the baseline cannot separate them because raw logprobs are not comparable across differing context lengths and both look confident. Our method instead converts each top patch into a scale-free rank inside its own sample distribution, so the localization whose top patch is a sampling outlier is demoted while the one whose top patch is typical survives. We expect the gap over this baseline to be largest on the subset with heterogeneous localization context lengths, and near zero where contexts are uniform.

**vs. Random demotion at matched rate** — On a case where the correct localization sits below a spurious one, the baseline demotes a localization at random and promotes the correct one only by chance, because its demotion carries no information about patch validity. Our method demotes specifically the localization whose top patch fails the closure invariant, so promotion of the correct localization is systematic rather than lucky. We expect our top-1 accuracy to exceed the random-demotion control by a clear margin that grows with the fraction of cases where the spurious localization carries an inflated top-patch likelihood.

**vs. Sample-agreement demotion (pass@k-style)** — On a case where the top patch is a high-likelihood mode but the fixer's samples are diverse and rarely agree on any single patch, the baseline demotes the localization because sample agreement is low, even though the top patch may be a valid repair. Our method instead compares the top patch's rank to the calibrated rank of known-valid patches at the same context length, so it demotes only when the top patch is more of an outlier than a valid patch would be. We expect our method to outperform sample-agreement demotion on cases where the correct localization has high fixer entropy but a valid top patch, and to be comparable elsewhere.

**vs. DCV with fixed tau=0.05 (no context-matched calibration)** — On a case where the correct localization has a long context and a valid top patch that naturally sits in a higher tail mass, the fixed-tau baseline may demote it because c(l) < 0.05, while our context-matched tau(l) accounts for the longer context and passes it. We expect the calibrated variant to improve top-1 accuracy specifically on localizations with context lengths in the top and bottom quartiles, and to be similar on median-length contexts.

**vs. DCV with length-normalized logprob in c(l)** — On a case where the top patch is short and repetitive, the unnormalized c(l) may flag it as an outlier due to brevity bias, while length-normalized logprob reduces this bias. We expect the length-normalized variant to show a smaller gap to the calibrated variant on brevity-heavy bugs, confirming that calibration and length normalization address overlapping confounds.

### What would falsify this idea

If the top-1 gain is uniform across the subset where the correct localization was already first and the subset where it was ranked below a spurious one, rather than concentrated on the latter, the closure statistic is not doing the claimed work. Likewise, if the gain over raw-logprob re-ranking, random demotion, and sample-agreement demotion vanishes once localization context lengths are matched, the mechanism is confounded by context length or sampling consistency rather than auditing fixer typicality.

## References

1. Let the Fixer Decide: Training a Code Localizer from the Fixer's Likelihood
