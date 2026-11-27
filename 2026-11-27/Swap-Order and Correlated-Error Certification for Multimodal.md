# Swap-Order and Correlated-Error Certification for Multimodal Safety Review Audits

## Motivation

Across multimodal safety alignment, guard-style interventions such as JudiAct (from 'Guarding with Judgment') issue a diagnostic label and handling action after generating and checking review items, but the soundness of that decision rests on an unverified assumption: that review items are conditionally independent and that the sequential single-item swap kernel is order-insensitive. Because the audit corpus is finite and review items share the same source evidence, correlated errors and swap-order effects can flip the diagnostic label without any change in the underlying safety judgment, so the guard's consistency guarantee is assumed rather than certified. This recurs across the field: SPA-VL and Safe RLHF-V likewise rely on preference/guard pipelines whose labels are trusted without a closure test on the intervention kernel.

## Key Insight

A diagnostic label is audit-sound only if it is a fixed point of the review-item intervention group — invariant under every permutation of joint swaps and stable under correlated-error resampling — and this invariance can be estimated directly from the swap kernel without retraining the guard.

## Method

**What it is.** Swap-Closure Certifier (SCC) takes a trained JudiAct-style guard, a multimodal audit instance (query, image, source evidence), and the guard's generated review items, and outputs either a certified diagnostic label plus handling action or an abstention flag with the violating swap permutation and correlated-error stratum. It does not modify the guard's weights; it certifies the decision post hoc.

**Load-bearing assumption (stated explicitly).** SCC assumes the diagnostic label produced by the guard is a deterministic, exchangeable function of the review-item set, so that a label flip under a joint two-item swap or a correlated-error resample is attributable to genuine order/correlation instability in the safety judgment rather than to the guard's own sensitivity to semantically non-exchangeable item content or to decoding/context effects. This assumption is verified per instance by the semantic-equivalence check and the per-item relevance-score check in step 2; instances that fail these checks are routed to a separate non-exchangeable stratum and are not counted as order violations.

**How it works.**

```pseudocode
Input: guard G, audit instance x=(q,img,src), review items R=[r1..rk] (k=3), swap budget B, resample count M, tolerance tau, null-permutation count N_null, independent reader H
1. Baseline: y0, a0, rel0 = G(q,img,src,R)          # diagnostic label, action, per-item relevance scores rel0[i]
2. Swap-order sensitivity matrix S in R^{k x k}:
   for each ordered pair (i,j), i != j:
       R_ij = swap(R, i, j)                       # joint two-item swap with explicit item-identity tags
       # semantic-equivalence check: H judges R_ij equivalent to R (yes/no)
       eq_ij = H.equivalent(R, R_ij)
       y_ij, rel_ij = G(q,img,src,R_ij)
       # order-attributable flip: label changes AND per-item relevance scores unchanged
       S[i,j] = 1[y_ij != y0 AND rel_ij == rel0 AND eq_ij]
       if not eq_ij or rel_ij != rel0:
           route instance to non-exchangeable stratum; record (i,j)
   for each full permutation pi in Perm(k) sampled up to B (B=24 exhaustive for k=3):
       R_pi = apply_perm(R, pi)
       record y_pi, rel_pi
       if rel_pi != rel0: mark non-exchangeable
3. Correlated-error correlation matrix C in R^{k x k}:
   # rho is estimated from held-out labeled audit instances (see calibration below), not auditor-set
   for m in 1..M (M=64):
       e_m ~ CorrelatedResample(R; rho_hat)        # block-resample shared-evidence strata
       y_m = G(q,img,src, e_m)
       C[i,j] += 1[e_m[i] and e_m[j] both replaced]
   C = C / M
   # null calibration: N_null=128 benign same-span permutations (no shared-span replacement)
   # compute flip rate under null; set tau = 95th percentile of null flip rate
4. Certify iff:
   (a) all sampled y_pi equal y0 (permutation invariance) AND no non-exchangeable flag
   (b) max_{i,j} S[i,j] * C[i,j] <= tau (tau calibrated as above) over correlated strata
   (c) abstain if any y_pi disagrees with y0 and flag argmax S[i,j]
5. Output: (certified y0, a0) or (ABSTAIN, violating pair (i*,j*), stratum)
```

**Calibration of rho_hat and C.** From a held-out set of 512 labeled audit instances (disjoint from test), we compute empirical co-error rates: for each pair of review items sharing a source span, we measure the fraction of instances where both items are replaced in a block-resample and the guard's label flips. We fit rho_hat as the maximum-likelihood intra-class correlation of these binary co-error indicators under a beta-binomial model, yielding a single scalar per source-span type (e.g., image-region vs. text-span). We report calibration of C against observed co-flips via a reliability diagram (predicted C[i,j] vs. empirical co-flip frequency) and the Brier score. The null calibration in step 3 sets tau to the 95th percentile of flip rates under benign same-span permutations, so generic OOD/uncertainty flips are not misread as correlated-error exposure.

**Why this design.** We chose a joint two-item swap over JudiAct's sequential single-item swap because a sequential kernel can only detect first-order order effects, while correlated errors between two review items require a joint intervention to become identifiable; the cost is that the swap budget grows as O(k^2) rather than O(k), which we accept because k=3 in JudiAct makes exhaustive enumeration cheap (B=24). We chose an explicit correlated-resample estimator over a learned independence head because the correlation structure is a property of the shared source evidence, not of the guard's parameters, and a learned head would re-import the coupling it is meant to test (the Anti-Pattern-1 failure); the cost is that rho must be estimated, which we handle by fitting rho_hat from held-out labeled instances and reporting a sensitivity sweep over rho in {0.25, 0.5, 0.75}. We chose a hard invariance certificate with abstention over a soft consistency score because a guard that reports a label without a fixed-point guarantee is exactly the failure mode JudiAct's audit feedback was meant to close; the cost is a nonzero abstention rate that we report as a first-class metric. A domain expert would not describe SCC as a variant of JudiAct's sequential-swap refinement because SCC intervenes on the decision surface (the label as a function of the review set) rather than on the review generator, and it certifies rather than trains.

**Why it measures what we claim.** The swap-order sensitivity matrix S[i,j] measures swap-order sensitivity of the diagnostic label because it assumes the joint swap (i,j) is a valid intervention on the review set that leaves all other items fixed — i.e., that review items are exchangeable except through the guard's reading of them; this assumption fails when item i's content is only interpretable in light of item j (a genuinely non-exchangeable pair), in which case S reflects semantic dependence rather than order instability. We therefore add the semantic-equivalence check (H) and the per-item relevance-score check (rel_ij == rel0) to gate S, and route failing instances to a non-exchangeable stratum. The correlation matrix C[i,j] measures correlated-error exposure because block-resampling shared-evidence strata assumes that items drawn from the same source span share their error mode; this assumption fails when a shared span is benign for one item and adversarial for another, in which case C overstates the correlation. We validate C against empirical co-flip rates on held-out labeled instances and report a reliability diagram. The permutation-invariance condition (a) measures audit soundness because it assumes the diagnostic label is a deterministic function of the review set — the guard's decoding temperature is fixed at 0 — and this assumption fails under stochastic decoding, in which case disagreement reflects sampling noise rather than non-invariance; we therefore fix greedy decoding and report the assumption explicitly. The null calibration of tau ensures that flips attributable to generic OOD/uncertainty (per JOOD) are not counted as correlated-error exposure.

## Contribution

['A post-hoc Swap-Closure Certifier (SCC) that converts JudiAct-style review-item audits into decisions with an explicit invariance certificate, via a joint two-item swap sensitivity matrix and a correlated-error resampling estimator.', 'A design principle: audit soundness for multimodal safety guards is a fixed-point property of the review-intervention group, and it can be estimated from the intervention kernel without retraining the guard.', 'An abstention-and-violation reporting protocol that identifies the specific review-item pair and correlated-error stratum responsible for a non-certified label.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | JudiBench audit instances (test split, 1,000 instances) | Native to the guard; labeled request/answer targets |
| Primary metric | Abstention rate on correlated-error strata + certified-label accuracy | Separates genuine soundness from forced labeling |
| Baseline | JudiAct 4B backbone (unmodified) | Tests whether post-hoc certification adds soundness |
| Baseline | Sequential single-item swap audit | Tests whether joint swaps are necessary |
| Baseline | Learned independence-head consistency score | Tests whether explicit correlation beats learned coupling |
| Ablation-of-ours | SCC with C forced to identity (rho=0) | Isolates the correlated-error estimator's contribution |
| Ablation-of-ours | SCC with tau set to 0 (no null calibration) | Tests whether null calibration prevents OOD false positives |
| Ablation-of-ours | SCC with semantic-equivalence check disabled | Tests whether non-exchangeable routing matters |

### Why this setup validates the claim

SCC's central claim is that a guard's diagnostic label can be certified post hoc only if it is invariant to joint two-item swaps and robust to correlated-error strata, and that failing either condition predicts exactly where the guard is unsound. JudiBench is the right substrate because the JudiAct guard already produces review items there, so the swap interventions are defined on real review sets rather than synthetic ones. The unmodified backbone baseline tests the sub-claim that certification exposes unsoundness the guard hides; the sequential-swap baseline tests the sub-claim that first-order interventions cannot see correlated errors; the learned-head baseline tests the sub-claim that correlation must be estimated from shared source evidence, not re-imported through parameters. The rho=0 ablation is the load-bearing control: if SCC's advantage survives with C forced to identity, then the correlated-error estimator is decorative and the joint-swap machinery alone explains the result. The tau=0 ablation tests whether null calibration is necessary to avoid misreading generic OOD flips as correlated-error exposure. The semantic-equivalence ablation tests whether the non-exchangeable routing is doing work. The primary metric is abstention rate on correlated strata paired with certified-label accuracy, because a certifier that abstains everywhere or certifies everything is trivially uninformative; only the joint pattern falsifies the soundness claim. We also report calibration of C against empirical co-flip rates on a held-out set of 512 labeled instances via a reliability diagram and Brier score.

### Expected outcome and causal chain

**vs. JudiAct 4B backbone (unmodified)** — On a case where two review items draw on the same source span and the guard assigns a confident diagnostic label, the baseline produces a label with no soundness signal because its sequential refinement never intervenes on the joint decision surface. Our method instead enumerates the 24 permutations and the joint swap matrix, so it detects that the label flips under a two-item swap and abstains. We expect the baseline to show near-zero abstention with a subset of confidently wrong labels, while SCC abstains specifically on the shared-span subset and matches the baseline on independent-evidence cases.

**vs. Sequential single-item swap audit** — On a case where swapping item i alone leaves the label stable but swapping i and j together flips it (a genuine second-order correlated error), the sequential baseline produces a certified label because its O(k) kernel only sees first-order effects. Our method instead applies the joint swap and reads the flip directly from S[i,j], so it flags the violating pair. We expect a noticeable gap concentrated on instances where the two flagged items share a source span, with parity on instances where items are drawn from disjoint spans.

**vs. Learned independence-head consistency score** — On a case where the guard's own parameters couple the two items' errors, the learned head produces a high consistency score because it re-imports the coupling it is meant to test, and the guard is certified despite correlated failure. Our method instead block-resamples shared-evidence strata with an empirically estimated rho_hat, so the correlation estimate is exogenous to the guard. We expect the learned head to certify a subset of cases SCC abstains on, with the gap tracking the rho sensitivity sweep rather than the guard's confidence.

**vs. SCC with C forced to identity (rho=0)** — On a case where two review items share a source span and their errors are correlated, the rho=0 ablation sets C[i,j]=0 for all pairs, so the correlated-error condition (b) is trivially satisfied and the instance is certified despite correlated failure. Our method uses the estimated C to detect the correlation and abstains. We expect the rho=0 ablation to reproduce the full method's certified-label accuracy on independent-evidence cases but to over-certify on shared-span cases, with the gap concentrated on instances where empirical co-flip rates are high.

**vs. SCC with tau set to 0 (no null calibration)** — On a case where a benign same-span permutation causes a label flip due to generic OOD/uncertainty (per JOOD), the tau=0 ablation treats any flip as a violation and abstains, inflating abstention on benign cases. Our method uses the null-calibrated tau (95th percentile of null flip rate) to avoid this. We expect the tau=0 ablation to show higher abstention on independent-evidence cases with no gain in certified-label accuracy.

**vs. SCC with semantic-equivalence check disabled** — On a case where two review items are genuinely non-exchangeable (item i only interpretable in light of item j), disabling the equivalence check causes S to count a semantic-dependence flip as an order violation, leading to a false abstention. Our method routes such instances to a non-exchangeable stratum. We expect the disabled version to show higher abstention on non-exchangeable pairs without improving soundness on exchangeable pairs.

### What would falsify this idea

If SCC's abstentions were uniform across correlated and independent strata, or if the rho=0 ablation reproduced the full method's certified-label accuracy, then the correlated-error estimator would not be doing the work the claim assigns it and the central mechanism would be wrong. Likewise, if the sequential-swap baseline matched SCC's abstention pattern on shared-span instances, the joint-swap necessity claim would be falsified.

## References

1. Guarding with Judgment: Learning to Intervene for Multimodal Safety
