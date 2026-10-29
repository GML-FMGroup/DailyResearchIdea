# Sufficient-Statistic Expansion Certificates for Open-World Amortized Bayesian Experimental Design

## Motivation

Amortized EIG policies such as the LLM designer of 'Amortizing Sequential Experimental Design in Reasoning LLMs' optimize question-selection against a hypothesis space that is enumerated and frozen before training, so the policy never observes how its own EIG ranking over questions changes when the hypothesis space grows at deployment. The same structural gap recurs in Step-DAD (test-time policy updates still score designs against the training-time generative model) and in TNDP and PASOA, where expected information gain is computed over a closed candidate set fixed a priori. Consequently a question that is information-maximizing on the closed set can silently cease to be so under a superset of hypotheses, and no existing method exposes a per-question diagnostic that would reveal this.

## Key Insight

When each question's likelihood depends on the hypothesis only through a finite-dimensional sufficient statistic, the EIG difference between any two questions under a hypothesis superset is bounded by a per-question expansion-sensitivity term computable from the statistic's posterior-predictive mass alone, so the closed-set ranking is certified invariant without re-enumerating the superset.

## Method

**What it is.** EXPAND-CERT (Expansion-Sensitivity Certified Amortization) is a wrapper around any amortized EIG policy that (i) re-expresses each question's likelihood in sufficient-statistic form, (ii) computes a per-question expansion-sensitivity certificate from the posterior predictive over that statistic, and (iii) at deployment filters or re-ranks the policy's candidate questions using the certificate. Inputs: a trained policy π(a|h), a likelihood model p(o|a,θ), and a deployed hypothesis superset H⁺ given only as an unenumerated sampler. Output: a certified question ranking and a per-question certificate value.

**Load-bearing assumption (stated explicitly).** The certificate is valid only if the sufficient statistic T_a that makes p(o|a,θ)=g_a(T_a(o);θ) hold under the closed training hypothesis set H0 remains sufficient for every hypothesis in the deployed superset H⁺, so that the entire EIG change under expansion is mediated by the posterior predictive q_a(t|h) over T_a and eps_a + entropy_gap is a valid (monotone, non-under-reporting) bound on the true EIG gap. Sufficiency is a property relative to a model class: a statistic sufficient for θ∈H0 is generically insufficient once H⁺ adds coordinates that enter the likelihood outside T_a. This assumption is tested at deployment (see SUFFICIENCY-TEST below) and, when violated, the certificate is inflated by a residual term and the question is either down-weighted or re-scored by a coarse direct superset EIG estimate.

**How it works.**

```pseudocode
# ---------- TRAIN TIME (closed set H0) ----------
for each question a in candidate bank A0:
    identify sufficient statistic T_a(o) s.t. p(o|a,θ)=g_a(T_a(o); θ)   # from model structure
    fit predictive  q_a(t | h) = ∫ p(T_a=t|a,θ) p(θ|h) dθ              # via amortized posterior / particles
    store (T_a, q_a, g_a)

train π(a|h) with EIG reward  EIG_a(h) = E_{θ~p(θ|h)} E_{o~p(o|a,θ)} log[p(o|a,θ)/p(o|a,h)]

# ---------- SUFFICIENCY TEST (per question, per history) ----------
function SUFFICIENCY-TEST(a, h, H_plus_sampler, M=512, B=200):
    # Draw M superset samples θ_i ~ H_plus_sampler(h) with importance weights w_i = p(θ_i|h)/proposal(θ_i)
    # For each θ_i, draw B observations o_{i,b} ~ p(o|a,θ_i)
    # Compute T_a(o_{i,b}) and the residual likelihood ratio r_{i,b} = p(o_{i,b}|a,θ_i) / g_a(T_a(o_{i,b}); θ_i)
    # Test whether r_{i,b} depends on θ_i beyond T_a:
    #   - Bin observations by T_a value (or use nearest-neighbor in T_a space)
    #   - Within each bin, compute the dispersion of r_{i,b} across θ_i (e.g., variance of log r)
    #   - Aggregate to a test statistic D_a = weighted average within-bin variance of log r
    #   - Under sufficiency, D_a should be ~0; set threshold via permutation of θ labels within bins (B permutations) at level delta=0.05
    # Return D_a, p-value, and residual bound R_a = sup_{θ∈H⁺} sup_{o} |log r(o,θ)| (estimated as max over samples)
    return D_a, p_value, R_a

# ---------- CERTIFICATE COMPUTATION (per question, per history) ----------
function CERT(a, h, H_plus_sampler, delta=0.05, N=2048):
    S = {θ_1..θ_N} ~ H_plus_sampler(h)                  # superset samples, NOT enumerated
    w_i = p(θ_i|h) / proposal(θ_i)                       # importance weights
    # ESS check: if (Σ w_i)^2 / Σ w_i^2 < 0.1 * N, resample with defensive mixture proposal
    for t in support(T_a):                               # low-dim grid / quadrature
        q_plus(t) = Σ_i w_i p(T_a=t|a,θ_i) / Σ_i w_i     # superset predictive
        q_0(t)    = q_a(t|h)                             # closed-set predictive
    # per-question expansion sensitivity = sup over t of |log q_plus - log q_0| weighted by q_0
    eps_a = Σ_t q_0(t) * |log q_plus(t) - log q_0(t)|
    # EIG gap bound: |EIG_plus(a) - EIG_0(a)| <= eps_a + H(q_0) - H(q_plus)
    entropy_gap = H(q_0) - H(q_plus)
    # sufficiency test and residual inflation
    D_a, p_val, R_a = SUFFICIENCY-TEST(a, h, H_plus_sampler)
    if p_val < delta:   # sufficiency violated
        gap_bound = eps_a + entropy_gap + R_a
    else:
        gap_bound = eps_a + entropy_gap
    return eps_a, gap_bound, D_a, p_val, R_a

# ---------- DEPLOYMENT ----------
function SELECT(h, A_cand, H_plus_sampler, tau=0.02):
    scores = {a: EIG_0(a,h) for a in A_cand}             # policy's closed-set EIG
    certs  = {a: CERT(a,h,H_plus_sampler) for a in A_cand}
    # keep questions whose certificate gap is below tau; among them pick max EIG_0
    feasible = [a for a in A_cand if certs[a].gap_bound <= tau]
    if feasible: return argmax_{a in feasible} scores[a]
    else:       return argmin_{a in A_cand} certs[a].gap_bound    # fall back to safest question
```

Hyperparameters sit inline: `delta` (confidence level for the importance-weight concentration and sufficiency test, default 0.05), `N` (superset sample budget, default 2048), `M` (sufficiency test sample budget, default 512), `B` (permutation replicates, default 200), `tau` (tolerated EIG-gap threshold, default 0.02 nats). The sufficient statistic `T_a` is derived once per question from the generative model's factorization (e.g., for a Bernoulli answer to a categorical question, `T_a` is the answer itself; for a Gaussian observation, `T_a` is the scalar projection onto the question's design vector). The quadrature grid for `support(T_a)` is constructed as follows: for discrete `T_a` (e.g., categorical answers), use the exact support; for continuous `T_a` (e.g., scalar projection), use a uniform grid of 256 points spanning the 0.001–0.999 quantile range of `q_0`, with trapezoidal weights. The importance-weight ESS check triggers a defensive mixture resampling when ESS < 0.1*N: mix the original proposal with a broad proposal (e.g., prior) at 50% weight and redraw.

**Why this design.** We chose to certify via the low-dimensional posterior predictive over `T_a` rather than re-estimating EIG directly on the superset because the latter requires enumerating or densely sampling the superset at every decision, which is exactly the cost amortization was meant to remove—accepting the cost that the certificate is only as tight as the sufficient-statistic dimensionality. We chose an importance-weighted superset sampler over a parametric approximation to the deployed hypothesis space because the deployed superset is typically specified only as a generative sampler (as in the Twenty-Questions and Medi-Q settings of the amortized-LLM-BED work), accepting the cost of weight degeneracy when the proposal is poor (mitigated by `N`, the ESS check, and defensive mixture resampling). We chose a hard feasibility filter with a deterministic fallback rather than blending certificate values into the EIG score because a blended score has no interpretation as an information-maximizing ranking, accepting the cost that questions near the `tau` boundary are discarded rather than down-weighted. We added the sufficiency test and residual inflation because the certificate's validity hinges on T_a remaining sufficient under H⁺; when the test detects violation, we inflate the bound by the residual R_a and, if the inflated bound exceeds tau, the question is excluded from the feasible set, preventing silent certification of a wrong ranking. A domain expert would not describe this as a variant of Step-DAD: Step-DAD updates the policy's parameters on observed data but still scores designs against the training generative model, whereas EXPAND-CERT leaves the policy frozen and adds an external, per-question certificate that is computed from sufficient statistics and is independent of the policy's parameters.

**Why it measures what we claim.** The quantity `eps_a = Σ_t q_0(t)|log q_plus(t) − log q_0(t)|` measures the expansion sensitivity of question `a`'s EIG because the EIG of a question is a functional of its posterior predictive over `T_a` alone (by the sufficiency of `T_a` for `θ`), so any change in EIG under hypothesis-set expansion must pass through a change in `q_a`; this assumption fails when the deployed superset contains hypotheses under which `T_a` is no longer sufficient (e.g., a question whose answer distribution depends on a hypothesis coordinate not captured by `T_a`), in which case `eps_a` reflects only the sufficiency-preserving part of the expansion and under-reports the true EIG gap. The quantity `entropy_gap(q_0, q_plus)` measures the residual predictive-entropy change induced by expansion because the EIG decomposes into expected log-likelihood minus predictive entropy, and the expected-log-likelihood term is pinned by `T_a`'s sufficiency; this assumption fails when the superset changes the likelihood's normalization in a statistic-dependent way, in which case the entropy gap is a lower bound rather than an exact accounting. The sufficiency test statistic `D_a` measures the violation of T_a's sufficiency under H⁺ because it quantifies the within-T_a dispersion of the likelihood ratio r(o,θ) across superset samples; this fails to detect violation when the violation is confined to a region of T_a space with low posterior predictive mass, in which case the residual R_a provides a conservative upper bound. The feasibility filter with threshold `tau` measures the claim that the closed-set ranking is preserved because the EIG gap bound `eps_a + entropy_gap + R_a·1[p_val<delta]` is monotone in the true gap under the sufficiency assumption; this fails when two questions have certificates within `tau` of each other but the bound is loose, in which case the filter certifies a ranking that the true superset EIG would not endorse, which is why the reported certificate value is always surfaced alongside the selected question.

## Contribution

(1) A per-question expansion-sensitivity certificate, computed from a question's sufficient-statistic posterior predictive rather than from an enumerated hypothesis set, that bounds how much a closed-set EIG ranking can shift under a hypothesis superset. (2) A deployment-time selection rule that filters a frozen amortized EIG policy's candidate questions by certificate gap, converting closed-set amortization into open-world amortization with a checkable invariant. (3) A diagnostic interface (certificate value + effective sample size per question) that surfaces when the sufficiency assumption underlying the certificate is violated, giving practitioners a signal that existing amortized BED methods do not provide.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Twenty-Questions with held-out category superset | Closed-set train, expanded sampler at deploy |
| Dataset | Gaussian linear design with held-out design vectors | Structurally different domain to test sufficiency assumption |
| Primary metric | EIG gap vs. superset oracle on filtered questions | Directly measures certified ranking fidelity |
| Secondary metric | Sufficiency test pass rate (fraction of questions with p_val ≥ 0.05) | Reports how often the load-bearing assumption holds |
| Baseline | Amortized EIG policy (frozen, no certificate) | Tests whether certification alone fixes expansion |
| Baseline | Step-DAD (semi-amortized update) | Tests external certificate vs. parameter updates |
| Baseline | Inference-time superset EIG re-estimation | Tests amortization cost vs. accuracy tradeoff |
| Ablation-of-ours | EXPAND-CERT with eps_a only (drop entropy gap and sufficiency test) | Isolates the entropy-gap term's contribution |
| Ablation-of-ours | EXPAND-CERT with eps_a + entropy_gap but no sufficiency test | Isolates the sufficiency test's contribution |

### Why this setup validates the claim

The central claim is that a per-question, sufficient-statistic certificate computed from an unenumerated superset sampler can protect a frozen amortized EIG policy from expansion-induced ranking errors at amortization cost, provided the sufficiency assumption is tested and violations are handled. Twenty-Questions with a held-out category superset is the right testbed because the training categories are closed-set while deployment introduces unseen hypotheses, exactly the regime where a frozen policy's closed-set EIG diverges from superset EIG. The Gaussian linear design provides a second, structurally different domain where the sufficient statistic is a scalar projection and sufficiency violations can be induced by adding design vectors that load on unmodeled coordinates, testing the breadth of the certificate. The frozen amortized policy baseline tests the sub-claim that uncertified amortization degrades under expansion; Step-DAD tests the sub-claim that our external certificate is a viable alternative to parameter updates; inference-time superset re-estimation tests the sub-claim that we retain most of the accuracy at a fraction of the cost. The eps-only ablation isolates whether the entropy-gap term is load-bearing or decorative, and the no-sufficiency-test ablation isolates whether the sufficiency test and residual inflation are necessary to avoid under-reporting the EIG gap. EIG gap versus a superset oracle is the correct primary metric because it directly measures whether the questions we select are the ones the true expanded objective would endorse — not a proxy like task accuracy that confounds many factors. The sufficiency test pass rate reports how often the load-bearing assumption holds in practice, establishing the applicability of the certificate.

### Expected outcome and causal chain

**vs. Amortized EIG policy (frozen, no certificate)** — On a case where deployment introduces an unseen category whose answer distribution for a high-closed-set-EIG question is nearly uniform, the frozen policy keeps selecting that question because its score was learned against the training generative model and never sees the expanded likelihood. Our method instead computes q_plus over the sufficient statistic, detects a large eps_a for that question, and filters it out in favor of a question whose predictive is stable under expansion. We expect a visible EIG-gap reduction concentrated on the subset of decisions where an unseen category is probed, with near-parity on decisions where closed and expanded predictives agree.

**vs. Step-DAD (semi-amortized update)** — On a case where the deployed superset is specified only as a sampler and the new hypotheses are sparsely represented, Step-DAD's parameter update is driven by observed data from a handful of trajectories and cannot see the unobserved expansion, so it re-scores designs against the original generative model and drifts toward the same closed-set ranking. Our method instead uses the sampler directly through importance weights and certifies each candidate against the expanded predictive without touching policy parameters. We expect comparable EIG gap early in a trajectory but our method to hold its gap as the history grows, while Step-DAD's gap widens once the superset's mass concentrates away from training categories.

**vs. Inference-time superset EIG re-estimation** — On a case with a large candidate bank and a moderately sized superset sampler, re-estimation must draw nested samples per candidate per step, so it either blows the compute budget or uses too few samples and yields noisy rankings that flip between adjacent questions. Our method instead pays the sufficient-statistic quadrature once per candidate and reuses the frozen policy's score, so its ranking is stable and its cost scales with support size of T_a rather than with superset size times candidates. We expect near-identical EIG gap to the oracle on the filtered set but a large wall-clock and sample-budget advantage, with the gap between the two methods shrinking as N grows.

**vs. EXPAND-CERT with eps_a only** — On decisions where the sufficiency test flags a violation (p_val < 0.05), the eps-only variant under-reports the EIG gap because it ignores the residual likelihood variation R_a, leading to selection of questions that are not information-maximizing under the superset. We expect the full certificate to achieve lower EIG gap on the subset of decisions with sufficiency violations, while matching the eps-only variant on decisions where sufficiency holds.

**vs. EXPAND-CERT without sufficiency test** — On decisions where T_a is insufficient under H⁺, the no-test variant certifies a ranking that the true superset EIG would not endorse, because it uses the bound eps_a + entropy_gap without the residual inflation. We expect the full certificate to avoid these silent failures, showing a lower EIG gap on the violation subset, and to report a higher sufficiency test pass rate as evidence of the assumption's validity.

### What would falsify this idea

If the EIG-gap reduction from EXPAND-CERT is uniform across all decisions rather than concentrated on the subset where the certificate flags large eps_a or sufficiency violations, the mechanism claim is wrong and the gain is coming from something other than expansion sensitivity. Likewise, if the eps-only ablation matches the full certificate everywhere, the entropy-gap term is not doing the work we claim and the decomposition is misattributed; if the no-sufficiency-test ablation matches the full certificate even on decisions where the sufficiency test flags violations, the residual inflation is unnecessary and the load-bearing assumption is not actually load-bearing in these domains.

## References

1. Amortizing Sequential Experimental Design in Reasoning LLMs
2. Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People
3. Step-DAD: Semi-Amortized Policy-Based Bayesian Experimental Design
4. Amortized Bayesian Experimental Design for Decision-Making
5. PASOA- PArticle baSed Bayesian Optimal Adaptive design
