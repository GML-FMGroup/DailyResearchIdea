# Proper Scoring of Proactive Intervention Confidence for Calibration-Aware Agent Evaluation

## Motivation

HALO (and the broader proactive-agent evaluation literature) reduces the decision of whether to intervene to a categorical need-type label, discarding the agent's graded uncertainty. This structural choice prevents distinguishing an over-eager agent that is confidently wrong from a cautious agent that is appropriately uncertain, even though HALO's own diagnosis identifies over-proactivity as the dominant failure mode. Because the benchmark's labels are categorical, no existing proactive-agent metric can penalize overconfident false interventions more than hesitant false interventions, so calibration — the core property of a proactive assistant — remains unmeasured.

## Key Insight

Eliciting a probability of intervention and scoring it with a strictly proper rule makes the agent's reported confidence incentive-compatible with its true belief, so the resulting score is a joint measure of discrimination and calibration that cannot be gamed by threshold-shifting, unlike categorical need-type accuracy.

## Method

**What it is.** We introduce CALIPER (Calibrated Intervention Probability Evaluation Rule), an evaluation protocol that replaces HALO's categorical need-type judgment with a scalar probability $p \in [0,1]$ that the agent emits alongside its intervention decision. CALIPER takes as input (i) a HALO-Arena scenario with retrieved evidence, (ii) the agent's emitted intervention probability $p$, and (iii) the human-grounded binary intervention label $y \in \{0,1\}$ from HALO-8K annotations, and outputs a proper score $S(p,y)$ plus a decomposition into calibration and discrimination components.

**Load-bearing assumption (stated explicitly).** CALIPER assumes that an LLM agent's explicitly elicited scalar probability $p$ is a faithful report of its internal intervention belief, so that a strictly proper scoring rule applied to $p$ measures the agent's genuine calibration rather than the elicitation prompt's idiosyncratic mapping from belief to number. This assumption is load-bearing because the incentive-compatibility guarantee of proper scoring rules holds only for a Bayesian agent whose reported number is its belief; for LLMs the number is a prompt-conditioned generation, and verbalized probabilities are known to be systematically distorted and prompt-sensitive (Xiong et al. 2024; Tian et al. 2023). CALIPER does not correct this distortion; it measures it, and it separates elicitation artifact from genuine miscalibration via the prompt-sensitivity diagnostic below before any cross-agent comparison.

**How it works.**

```pseudocode
Input: scenario s from HALO-Arena; agent A; human label y in {0,1}
Hyperparameters: temperature T=1.0 (no recalibration), Brier weight w_B=0.5, log weight w_L=0.5,
                 number of paraphrased elicitation prompts M=5, number of repeated draws R=3,
                 bin count K=10 (swept 5..20 in sensitivity analysis)

1. Evidence retrieval gate (unchanged from HALO):
   A retrieves evidence from structured DB / unstructured docs for s.

2. Probability elicitation prompt (fixed template, pre-registered):
   "Given the evidence, state the probability in [0,1] that proactive
    intervention is warranted. Output ONLY the number."
   Elicit M=5 paraphrased versions of this template (paraphrases pre-registered;
   e.g., "How likely is it that you should intervene?", "Give the chance in [0,1]
   that intervention is needed.", etc.) and R=3 independent draws per paraphrase
   at temperature T=1.0, yielding M*R=15 elicited probabilities {p_hat_i}.

3. Parse each p_hat_i from A's response; if unparseable, record p_hat_i = 0.5
   and flag parse_failure (reported separately, not scored).
   Reconciled probability: p_hat = mean_i(p_hat_i).
   Prompt-sensitivity variance: Var_prompt = variance of the M paraphrase-level means
   (each mean over R draws). Report Var_prompt as a first-class diagnostic.

4. Compute proper scores against human label y using the reconciled p_hat:
   Brier:  S_B = (p_hat - y)^2
   Log:    S_L = -(y log p_hat + (1-y) log(1-p_hat))
   Combined: S = w_B * S_B + w_L * S_L

5. Calibration decomposition (Murphy decomposition on held-out
   scenario split, K=10 equal-frequency bins):
   Reliability  = (1/N) * sum_k n_k (mean(p_hat in bin k) - mean(y in bin k))^2
   Resolution   = (1/N) * sum_k n_k (mean(y in bin k) - mean(y))^2
   Uncertainty  = mean(y) * (1 - mean(y))
   Brier = Reliability - Resolution + Uncertainty

6. Report per-agent: S_B, S_L, Reliability, Resolution, Var_prompt, and the
   over-eagerness gap Delta = mean(p_hat | y=0) - mean(p_hat | y=1).
   Delta > 0 indicates the agent assigns higher intervention
   probability to non-intervention cases (over-eager miscalibration).
   Also report Delta stratified by annotator agreement (see block D).
```

**Why this design.** We chose a fixed probability-elicitation prompt with a single scalar output over asking the agent for a verbal confidence category because categorical self-reports are not comparable across models and cannot be plugged into a proper scoring rule; the cost is that some models may produce poorly calibrated raw probabilities, which is precisely what CALIPER measures rather than corrects. We chose the Brier and log scores over threshold-based accuracy because proper scoring rules are strictly incentive-compatible — an agent maximizes expected score only by reporting its true belief — whereas accuracy under a fixed threshold rewards threshold-shifting; the cost is that log score is unbounded and sensitive to overconfident errors on the wrong side, which we mitigate by reporting both scores with a fixed weight $w_B=w_L=0.5$. We chose Murphy's reliability-resolution decomposition over a single scalar because it separates miscalibration (reliability) from genuine discriminative signal (resolution), which matters when comparing an over-eager agent (high resolution, poor reliability) to a well-calibrated one (both good); the cost is that the decomposition requires binning and a held-out split, reducing statistical power on small scenario counts. Unlike HALO's categorical need-type accuracy or the log-prob gating in Kadavath et al. (2022), CALIPER does not treat the model's raw log-probability as a calibrated signal — it requires an explicit probability elicitation and scores it against human labels, so the calibration question is answered by the scoring rule rather than assumed. To address the load-bearing assumption that elicited $p$ faithfully reports belief, we elicit $M=5$ paraphrased prompts with $R=3$ draws each and score the reconciled mean $p_hat$; we report the prompt-sensitivity variance $Var_prompt$ as a first-class diagnostic so that reliability error attributable to elicitation artifact is separated from genuine agent miscalibration before any cross-agent comparison is made. The cost is increased inference compute (15 elicitations per scenario instead of 1), which we accept because it makes the elicitation-artifact confound measurable rather than hidden.

**Why it measures what we claim.** The Brier score $S_B = (p - y)^2$ measures joint calibration and discrimination of the agent's intervention belief because the Murphy decomposition guarantees that in expectation over the label distribution, $S_B$ equals Reliability − Resolution + Uncertainty; this assumption (that the held-out scenario distribution matches the evaluation distribution) fails when HALO-Arena scenarios are not exchangeable with deployment scenarios, in which case $S_B$ reflects in-distribution calibration only. The reliability term measures over-eagerness because it is the bin-wise squared gap between mean predicted probability and empirical intervention rate; this assumes bins are large enough for stable means, and fails when a bin has fewer than ~30 scenarios, in which case reliability reflects binning noise. The over-eagerness gap $\Delta = \mathbb{E}[p \mid y=0] - \mathbb{E}[p \mid y=1]$ measures the specific failure mode HALO diagnoses because a well-calibrated agent must assign lower probability to non-intervention cases; this assumes the human label $y$ is itself calibrated to the true need for intervention (i.e., annotator error is mean-zero and independent of scenario ambiguity), and fails when annotators disagree systematically (e.g., culture-specific norms), in which case $\Delta$ reflects annotator bias rather than agent miscalibration. To test this assumption, we stratify $\Delta$ by annotator agreement: we compute $\Delta$ separately on the high-agreement subset (scenarios where all annotators agree on $y$) and the low-agreement subset (scenarios with any disagreement), and we report the difference $\Delta_{high} - \Delta_{low}$; if the over-eagerness signal is concentrated in low-agreement scenarios, $\Delta$ is confounded by annotator bias. The log score $S_L$ measures tail sensitivity to confident errors because it diverges as $p \to 0$ when $y=1$; this assumes the agent can meaningfully report extreme probabilities, and fails when the elicitation prompt induces probability clipping (e.g., models rarely output below 0.05), in which case $S_L$ underestimates overconfidence; we report the empirical distribution of elicited $p_hat$ (histogram with bins [0,0.05), [0.05,0.1), ..., [0.95,1.0]) to quantify clipping. The prompt-sensitivity variance $Var_prompt$ measures the elicitation artifact because it is the variance of the paraphrase-level mean probabilities; this assumes paraphrases span the space of prompt framings that affect the agent, and fails when paraphrases are too similar, in which case $Var_prompt$ underestimates framing sensitivity.

## Contribution

(1) CALIPER, a proper-scoring-rule evaluation protocol for proactive agents that replaces HALO's categorical need-type labels with an elicited intervention probability scored by Brier and log rules, with a Murphy reliability-resolution decomposition and an over-eagerness gap statistic. (2) The design principle that calibration-aware proactive-agent evaluation requires strictly proper scoring to prevent threshold-shifting, demonstrated by the decomposition separating over-eager agents (high resolution, poor reliability) from well-calibrated ones. (3) A reusable elicitation template and scoring harness that plugs into HALO-Arena/HALO-8K without re-annotation, enabling calibration comparison across models on existing human-grounded labels.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HALO-Arena scenarios with HALO-8K binary intervention labels | Human-grounded labels enable proper scoring against ground truth |
| Primary metric | Murphy-decomposed Brier score (Reliability, Resolution) plus over-eagerness gap Δ and prompt-sensitivity variance Var_prompt | Separates miscalibration from discrimination and diagnoses over-eagerness and elicitation artifact |
| Baseline | HALO categorical need-type accuracy | Tests whether scalar probability beats categorical judgment |
| Baseline | Kadavath-style raw log-prob gating | Tests whether explicit elicitation beats assumed log-prob calibration |
| Baseline | Verbal confidence category elicitation | Tests whether scalar output beats categorical self-report |
| Ablation-of-ours | CALIPER with reliability term only (drop resolution, log score) | Isolates whether decomposition adds diagnostic value beyond Brier |
| Ablation-of-ours | CALIPER with single fixed prompt (M=1, R=1) | Isolates whether multi-prompt reconciliation reduces Var_prompt and improves reliability |
| Diagnostic | Annotator-agreement-stratified Δ (high vs. low agreement subsets) | Tests whether Δ reflects agent miscalibration or annotator bias |
| Diagnostic | Empirical distribution of elicited p̂ (histogram) | Quantifies probability clipping and reachability of extremes |
| Diagnostic | Bin-size sensitivity sweep K ∈ {5,10,15,20} | Tests stability of reliability estimates to binning |
| Diagnostic | Inter-annotator agreement on HALO-8K binary labels (e.g., Cohen's κ) | Provides upper bound on achievable reliability |

### Why this setup validates the claim

The design forms a falsifiable test because each baseline and diagnostic probes a distinct sub-claim of CALIPER. HALO's categorical need-type accuracy tests whether replacing a discrete judgment with a scalar proper-scored probability actually recovers calibration information that categories discard — if categorical accuracy already correlates with the reliability term, the scalar adds nothing. The Kadavath-style raw log-prob gating tests the claim that explicit probability elicitation beats treating the model's implicit log-probability as calibrated; if raw log-probs match elicited probabilities on reliability, the elicitation step is redundant. The verbal-confidence baseline tests whether the fixed scalar prompt is necessary or whether any self-report suffices. The reliability-only ablation tests whether the Murphy decomposition earns its keep: if reliability alone predicts downstream over-eagerness as well as the full decomposition, the resolution and log-score components are decorative. The single-prompt ablation tests whether multi-prompt reconciliation (M=5, R=3) reduces Var_prompt and improves reliability relative to a single elicitation, directly addressing the load-bearing assumption that elicited p faithfully reports belief. The annotator-agreement-stratified Δ tests the assumption that human labels y are calibrated to true intervention need: if Δ is concentrated in low-agreement scenarios, the over-eagerness signal is confounded by annotator bias. The p̂ histogram quantifies probability clipping, which would cause the log score to underestimate overconfidence. The bin-size sweep tests the stability of reliability estimates to the choice of K. The inter-annotator agreement (Cohen's κ) provides an upper bound on achievable reliability, so any agent reliability below this bound is attributable to the agent rather than label noise. The Brier-based primary metric is the right detector because it is strictly proper, so any systematic probability distortion — exactly the over-eagerness HALO diagnoses — must show up as elevated reliability and a positive Δ, rather than being masked by threshold-shifting that accuracy would reward.

### Expected outcome and causal chain

**vs. HALO categorical need-type accuracy** — On a case where an agent must decide whether to intervene on an ambiguous scenario, HALO's categorical judgment collapses to a single need-type label, producing an accuracy score that cannot distinguish an agent that is right for the wrong confidence reasons from one that is genuinely calibrated. Our method instead elicits a scalar probability and scores it with the Brier decomposition, so we expect the categorical baseline to show high accuracy but poor reliability and a large positive Δ on the non-intervention subset, while CALIPER exposes that gap as concentrated reliability error rather than hiding it behind a threshold.

**vs. Kadavath-style raw log-prob gating** — On a case where the agent's internal log-probability is systematically inflated by prompt framing, the raw log-prob baseline treats that number as calibrated and produces over-eager gating, because it assumes the model's implicit probability is already meaningful. Our method instead forces an explicit probability elicitation scored against human labels, so we expect the raw log-prob baseline to show a noticeably larger over-eagerness gap Δ and worse reliability than CALIPER, with the gap concentrated on scenarios where framing-induced inflation is strongest. Additionally, we expect the raw log-prob baseline to exhibit higher Var_prompt than CALIPER's reconciled p̂, because raw log-probs are not reconciled across paraphrases.

**vs. Verbal confidence category elicitation** — On a case where the agent reports "moderately confident," the categorical baseline produces a non-comparable label that cannot be plugged into a proper scoring rule, because categories are not commensurable across models and discard magnitude. Our method instead maps the response to a scalar in [0,1] and scores it, so we expect the categorical baseline to show near-chance resolution and unreliable reliability estimates, while CALIPER yields a stable reliability-resolution split and a Δ that tracks the known over-eagerness pattern.

**vs. CALIPER single-prompt ablation (M=1, R=1)** — On a case where the agent's elicited probability is sensitive to prompt phrasing, the single-prompt ablation produces a p̂ that conflates belief with a particular framing, because it does not average over paraphrases. Our method instead reconciles M=5 paraphrases with R=3 draws each, so we expect the single-prompt ablation to show higher Var_prompt and worse reliability than full CALIPER, with the gap concentrated on scenarios where paraphrase-level means diverge most.

**Annotator-agreement stratification** — We expect Δ to be larger on the high-agreement subset than on the low-agreement subset if Δ reflects genuine agent miscalibration; if the opposite holds, Δ is confounded by annotator bias and the diagnostic claim is weakened.

### What would falsify this idea

If CALIPER's reliability term and over-eagerness gap Δ show no advantage over the categorical and raw-log-prob baselines on the subsets where those baselines are predicted to fail — i.e., if the gain is uniform across subsets rather than concentrated where the predicted miscalibration occurs — then the central claim that scalar proper scoring exposes over-eagerness is wrong. Equivalently, if the reliability-only ablation matches the full decomposition on every diagnostic, or if the single-prompt ablation matches full CALIPER on Var_prompt and reliability, the decomposition or the multi-prompt reconciliation adds no measurable value and the idea is falsified.

## References

1. HALO: Human-Grounded Granular Evaluation of Proactive Agents
