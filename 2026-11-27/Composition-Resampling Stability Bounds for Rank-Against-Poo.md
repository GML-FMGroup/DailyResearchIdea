# Composition-Resampling Stability Bounds for Rank-Against-Pool Forecasting Leaderboards

## Motivation

Leaderboards such as GIFT-Eval present a strict order over ~128 entries, yet the resolution-report analysis in 'What a Forecasting Leaderboard Can Establish' shows the board's smallest detectable difference (6.8 rank places) exceeds the median neighbour gap (0.91), and that pool-ranking rules reversed 43 untouched top-ten pairs across four boards — an empirical observation with no structural account of how many reversals are possible. The root cause is that rank-against-pool rules implicitly treat the dataset composition as fixed, so the order is a function of *who else submitted* and *which task subset* the board happened to include, and neither the diversity-stability decomposition nor the mean-rank branch bounds how much of the displayed order is composition noise versus genuine model difference.

## Key Insight

A rank-against-pool leaderboard's top-k order is a permutation-valued functional of a task-subset sample, so its instability is bounded by the between-task variance of pairwise score differences divided by the effective number of independent task clusters — a quantity estimable from published per-configuration scores alone, which makes the 43 reversals predictable rather than merely observable.

## Method

**What it is.** CASB (Composition-Adjusted Stability Bound) takes a leaderboard's published per-configuration score matrix (entries × configurations, with a dataset-cluster label per configuration) and outputs, for every displayed adjacent pair and for the top-k boundary, a predicted reversal probability under **pool perturbation** (adding/removing/re-scoring competing entries), plus a board-level bound on the number of top-ten pairs that can reverse. Inputs: score matrix S, cluster map c, pool-ranking rule R, and the observed pool-update deltas (the set of competitor additions/removals/re-scores between two board snapshots, or a synthetic perturbation distribution when only one snapshot is available). Outputs: pair-level reversal probabilities and a board stability certificate.

**Load-bearing assumption (stated explicitly).** The 43 observed top-ten reversals produced by pool re-ranking are driven by **pool-composition changes** — the identity and scores of the competing entries — not by resampling which tasks are on the board. CASB therefore models reversal probability under a **pool-perturbation null**: the entry's rank is recomputed as competitors are added, removed, or re-scored, and the between-cluster variance of paired score differences enters only as a secondary covariate that modulates how sensitive a pair's gap is to competitor score shifts. If the pool-perturbation model does not reproduce the 43 flips, the claim that composition variance explains pool-dependence is dropped.

**How it works.**

```pseudocode
INPUT: S[e, j]  # score of entry e on config j (lower=better error)
       c[j]     # dataset cluster of config j (e.g., GIFT-Eval: 4 datasets x 8 configs)
       R        # rank-against-pool rule (e.g., mean rank over pool)
       Delta    # observed pool-update deltas: list of (entry, config, score_change) or add/remove events
       k = 10, B = 2000

# Phase 0: pool-perturbation null
#   If Delta is observed (two snapshots), resample B times from the empirical
#   distribution of pool updates: each draw applies a random subset of Delta
#   events (bootstrap over events, with replacement) to the competitor pool.
#   If only one snapshot exists, synthesize Delta by drawing competitor score
#   perturbations from N(0, sigma_pool^2) with sigma_pool = median absolute
#   config-level score change across entries in the published matrix.

# Phase 1: cluster-level paired differences (secondary covariate)
d[e1,e2, g] = mean_{j in cluster g} (S[e1,j] - S[e2,j])   # per-cluster paired diff

# Phase 2: pool-perturbation reversal probability (primary)
for each displayed adjacent pair (e1,e2):
    # For each bootstrap draw b, recompute the pair's rank gap under perturbed pool
    gap_b = rank_R(e1 | pool_b) - rank_R(e2 | pool_b)
    P_rev_pool(e1,e2) = (1/B) * sum_b 1[sign(gap_b) != sign(gap_0)]
    # Secondary covariate: between-cluster variance of the paired diff
    mu    = mean_g d[e1,e2,g]
    sig2  = var_g d[e1,e2,g]
    n_eff = G / (1 + (G-1)*rho)          # rho = intra-cluster corr of d, estimated by ANOVA
    tau2  = sig2 / n_eff
    # Combine: pool-perturbation probability, shrunk toward cluster-model when
    # the pair's gap is dominated by a single cluster (|mu| large, tau2 small)
    P_rev(e1,e2) = P_rev_pool(e1,e2) * w + Phi(-|mu|/sqrt(tau2)) * (1-w)
    # w = tau2 / (tau2 + sigma_pool^2), i.e., weight on pool noise vs cluster noise

# Phase 3: board-level bound on top-k reversals (joint, not independence-summed)
#   Joint cluster-bootstrap of the full top-k permutation:
#   for b in 1..B: apply pool perturbation draw b, recompute full ranking,
#   record the top-k permutation pi_b.
#   N_rev_bound = (1/B) * sum_b  #{adjacent pairs in top-k whose order flips vs pi_0}
#   (exact under the pool-perturbation null; no independence assumption)

# Phase 4: validate against observed reversals
#   Pre-register the predicted P_rev vector (pair-level) before comparison.
#   Compare to the per-pair reversal table reconstructed from the four boards'
#   published score matrices (or the resolution report's per-pair table if released).
#   Report rank correlation (Spearman) between predicted P_rev and observed flip frequency.
```

The decomposition separates **within-cluster score noise** (irreducible measurement noise, estimated from config-level spread inside a cluster) from **pool-composition variance** (the part that drives pool-dependence), and the bound is computed by a joint pool-perturbation bootstrap with B=2000 draws; rho is estimated by ANOVA-style variance components and used only as the secondary covariate weight w.

**Why this design.** We chose a pool-perturbation null over a cluster-resampling null because the resolution report's 43 reversals arise from changes in the comparison set ('with neither entry rerun'), not from resampling which tasks are on the board; a cluster bootstrap models a variance source orthogonal to the one actually manipulated, so it would predict flips on the wrong pairs. We retain the cluster-level variance-component decomposition as a secondary covariate because it captures how sensitive a pair's gap is to competitor score shifts, accepting the cost that rho is estimated from G=4 clusters and is therefore noisy — which is why it enters only through the weight w, not as the primary reversal probability. We chose a joint bootstrap of the full top-k permutation over an independence-summed count because adjacent-pair reversals share dominant clusters, so the sum overstates independent reversals; the joint bootstrap yields an exact bound under the pool-perturbation null at the cost of B=2000 full re-rankings per board (feasible: 128 entries × 97 configs × 2000 draws is under 10 minutes on a single CPU). We chose to predict *pair-level* reversal probabilities rather than only a board-level count because the 43 reversals are a pairwise phenomenon and a board-level scalar would not be falsifiable against them, accepting the cost of more parameters (one P_rev per displayed pair). A domain expert would not describe this as a variant of the resolution report's pool-dependence check: that check *counts* reversals by re-ranking, whereas CASB *predicts* them from the score matrix and the observed pool-update deltas without re-ranking any submission, and the two are compared as prediction versus observation.

**Why it measures what we claim.** The per-pair quantity `P_rev(e1,e2)` measures *pool-composition instability* of the displayed order because the assumption that competitor additions/removals/re-scores are exchangeable draws from the observed pool-update distribution makes the resampled rank gap a random variable whose sign flips with that probability; this assumption fails when a pair's gap is driven by a single heavy-tailed competitor (e.g., one new entrant with an extreme score), in which case P_rev reflects that entrant's idiosyncratic score rather than board-wide pool noise. The quantity `w = tau2/(tau2 + sigma_pool^2)` measures the *relative contribution of cluster variance versus pool variance* to a pair's gap because the assumption that the two variance sources are additive and independent makes the weight the fraction of gap variance attributable to cluster structure; this assumption fails when cluster variance and pool variance are correlated (e.g., a new entrant that is also a high-variance cluster), in which case w misallocates the variance and the combined P_rev is biased toward one source. The board-level `N_rev_bound` measures the *fraction of displayed top-k order that is pool-composition noise* because the joint bootstrap of the full top-k permutation makes the expected discordant-pair count exact under the pool-perturbation null; this assumption fails when the observed pool-update deltas are not representative of future updates (e.g., a single large entrant dominates), in which case the bound should be read as conditional on the observed update distribution rather than as a universal envelope.

## Contribution

["(1) CASB, a composition-adjusted stability-bound procedure that converts a leaderboard's published per-configuration score matrix into per-pair reversal probabilities and a board-level bound on top-k rank reversals, requiring no rerun of any submission.", "(2) A diversity-stability decomposition that separates within-cluster measurement noise from between-cluster composition variance on forecasting boards, yielding an effective-cluster count n_eff that explains why boards built from 15–39 public sources are less stable than BOOM's 2,554-series board.", "(3) A falsification protocol that compares CASB's predicted reversal probabilities against the 43 observed top-ten reversals reported for pool-ranking rules, turning an empirical observation into a testable prediction."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | GIFT-Eval score matrix (97 configs, 4 dataset clusters) plus observed pool-update deltas between two board snapshots | Pool updates are the mechanism the resolution report manipulated. |
| Dataset (fallback) | BOOM score matrix (2,554 series, each as a source) | Independent board with many sources, per E10/E13. |
| Primary metric | Spearman rank correlation between predicted P_rev and observed per-pair flip frequency | Directly falsifiable against the per-pair reversal table. |
| Secondary metric | Predicted vs. observed top-ten reversal count (joint bootstrap) | Tests board-level bound without independence assumption. |
| Baseline | Flat per-configuration bootstrap | Tests whether cluster structure matters at all. |
| Baseline | Pool re-ranking count (resolution report) | Tests prediction versus observation of same phenomenon. |
| Baseline | Percentile bootstrap intervals | Tests our normal-approximation reversal probability. |
| Ablation-of-ours | CASB with w=1 (pure cluster model, no pool perturbation) | Isolates the pool-perturbation contribution. |
| Ablation-of-ours | CASB with w=0 (pure pool perturbation, no cluster covariate) | Isolates the cluster-covariate contribution. |

### Why this setup validates the claim

The central claim is that pool-composition instability of a leaderboard's displayed order can be *predicted* from the published score matrix and observed pool-update deltas without rerunning submissions, and that the prediction beats exchangeability-assuming alternatives. Each baseline tests one sub-claim. The flat per-configuration bootstrap tests whether the cluster decomposition matters: if CASB and the flat bootstrap predict the same reversal pattern, our non-exchangeability premise is empty. The resolution report's pool re-ranking count is the observation we predict against — the 43 untouched top-ten reversals — so it converts the exercise from internal consistency into prediction-versus-observation. The percentile bootstrap tests the closed-form normal reversal probability against a simulation-based alternative, isolating whether the Gaussian assumption is doing damage. The w=1 ablation removes the pool-perturbation null, so any gain concentrated on pairs that flip under pool updates is attributable to the pool model rather than to the cluster covariate. The w=0 ablation removes the cluster covariate, so any gain concentrated on high-intra-cluster-correlation pairs is attributable to n_eff. The primary metric — Spearman rank correlation between predicted P_rev and observed per-pair flip frequency — is the right detector because the claim is specifically about *which* pairs flip, so a board-level scalar alone would let a method be right on average and wrong on the pairs that matter; pair-level agreement is where the predicted pool-composition signal must show up if the mechanism is real. The BOOM fallback ensures the study does not hinge on a single board's data access.

### Expected outcome and causal chain

**vs. Flat per-configuration bootstrap** — On GIFT-Eval, where 4 datasets each contribute 8 correlated configurations, the flat bootstrap treats all 97 configurations as exchangeable and therefore draws resamples that over-represent within-cluster score noise while under-representing pool-composition shifts. Because its resampling variance omits the pool-update component, it produces reversal probabilities that are too small for pairs whose gap is sensitive to competitor score changes, and it cannot separate the two variance sources at all. Our method instead estimates the pool-perturbation distribution from observed deltas and weights it against the between-cluster variance via w = tau2/(tau2 + sigma_pool^2), so pairs whose gap is carried by pool noise get inflated P_rev. We expect a visible gap on pairs whose flip is driven by competitor changes, with near-parity on pairs whose difference is spread evenly across clusters — the signature that the pool-perturbation null, not a global rescaling, is doing the work.

**vs. Pool re-ranking count (resolution report)** — On the four boards where pool-ranking rules reversed 43 untouched top-ten pairs, the resolution report *observes* those reversals by re-ranking entries against different pools, which requires the full submission pool and cannot be computed from a single published board. Its mechanism gap is that it is a retrospective count, not a forward prediction: it tells you reversals happened but not which displayed pairs are at risk before a new submission arrives. Our method instead reads the pool-update deltas and the score matrix and emits a per-pair reversal probability with no re-ranking, so it should reproduce the *pattern* of the 43 reversals — high predicted P_rev on the pairs that actually flipped, low on the stable ones — rather than merely their count. We expect Spearman rank correlation between predicted P_rev and observed flip frequency that is clearly positive, with the strongest predicted probabilities landing on the pairs that flipped in one update in seven.

**vs. Percentile bootstrap intervals** — On pairs where the pool-perturbation rank gap is approximately normal, the percentile bootstrap and our joint bootstrap should agree closely, but where a pair's gap is driven by a single heavy-tailed competitor the bootstrap's interval is asymmetric and wide while our Gaussian tail probability is symmetric and can misstate the flip rate. The baseline's mechanism gap is that it carries no parametric form, so it cannot be summarized as a per-pair scalar without simulation; ours can, at the cost of the normality assumption. Our method handles the same instance by exposing the assumption as a diagnostic — the 2000-draw joint bootstrap flags exactly the pairs where the closed form departs from the empirical resampling distribution. We expect agreement on the bulk of pairs and a small flagged subset, concentrated on pairs involving a single dominant entrant, where the closed form and simulation diverge — which is the honest boundary of the claim rather than a uniform win.

### What would falsify this idea

If predicted P_rev fails to concentrate on the pairs that actually reversed — i.e., the per-pair reversal table shows flips distributed roughly independently of our predicted probabilities — then pool-composition variance is not the driver of pool-dependence and the central claim is wrong. Equivalently, if the w=1 ablation (pure cluster model) matches full CASB on every pair, the pool-perturbation null adds nothing and the cluster structure is the mechanism.

## References

1. What a Forecasting Leaderboard Can Establish: Resolution Reporting for Zero-Shot Time-Series Models
