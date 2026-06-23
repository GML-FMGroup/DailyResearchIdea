# PARCA: Paraphrase-Aware Reward Calibration for LLM Planning in Text-Based Games

## Motivation

LLM-based planners like MC-DML (Monte Carlo Planning with Large Language Model for Text-Based Game Agents) assume that LLM action evaluations are well-calibrated, but this assumption fails in out-of-distribution states encountered during search, leading to unreliable value estimates. Existing work lacks a training-free mechanism to detect and correct these miscalibrations without fine-tuning or additional data.

## Key Insight

Paraphrase variance of LLM evaluations reveals epistemic uncertainty because unreliable evaluations are more sensitive to input phrasing, enabling variance-weighted aggregation to produce calibrated scores without fine-tuning.

## Method

(A) **What it is**: PARCA (Paraphrase-Aware Reward Calibration) is a training-free calibration technique that takes a state-action pair and an LLM evaluator, and outputs a calibrated evaluation score by aggregating multiple paraphrased trials using a Bayesian shrinkage estimator. Its input is a game state description and candidate action; its output is a score in [0,1] that corrects for miscalibration.

(B) **How it works** (pseudocode):
```python
def parca_evaluate(state, action, llm_evaluator, paraphraser, num_paraphrases=5, prior_mu=0.5, prior_var=0.25):
    # Generate paraphrases of the state
    paraphrases = [paraphraser.paraphrase(state) for _ in range(num_paraphrases)]
    # Get LLM evaluations for each paraphrase (score between 0 and 1)
    scores = [llm_evaluator.evaluate(para, action) for para in paraphrases]
    # Compute sample mean and variance
    mean_score = np.mean(scores)
    variance = np.var(scores) + 1e-10  # avoid division by zero
    N = num_paraphrases
    # Bayesian shrinkage: posterior mean given Gaussian prior
    posterior_mean = ((mean_score / (variance / N)) + (prior_mu / prior_var)) / \
                     (1 / (variance / N) + 1 / prior_var)
    return posterior_mean
```
During MCTS planning (e.g., in MC-DML), leaf nodes are evaluated using `parca_evaluate` instead of a single LLM call. The calibrated score is backpropagated up the tree.

(C) **Why this design**: We chose Bayesian shrinkage over direct mean-variance tradeoff because it provides principled uncertainty quantification with a neutral prior, avoiding arbitrary thresholding. The use of paraphrase variance as a proxy for epistemic uncertainty leverages the observation that LLM miscalibration manifests as sensitivity to surface form, which is cheap to measure without extra training. Three design decisions: (1) generating multiple paraphrases via a separate paraphraser rather than using dropout or ensembling within the LLM—this is because paraphrasing directly probes input sensitivity, whereas dropout requires access to model internals; the cost is additional inference time. (2) Using a Gaussian prior centered at 0.5—this assumes no prior knowledge, but may bias towards neutral when variance is high; an alternative would be a prior based on in-distribution statistics, but that would require a calibration set, violating the zero-shot requirement. (3) Applying the calibration only to leaf nodes in MCTS, not to all nodes—because paraphrasing is expensive and internal nodes can be averaged via many rollouts; we accept the cost of potentially slower convergence. This design is not simply MC-DML with paraphrases added; the core innovation is the Bayesian shrinkage estimator that operationalizes paraphrase variance as a calibration signal.

(D) **Why it measures what we claim**: The posterior mean from Bayesian shrinkage measures calibrated evaluation because it downweights high-variance paraphrases, assuming that paraphrase variance correlates with evaluation unreliability; this assumption fails if the paraphraser introduces systematic bias (e.g., all paraphrases shift the meaning), in which case the posterior mean is biased toward the prior. The noise variance (sample variance of paraphrased scores) measures epistemic uncertainty under the assumption that the true evaluation is invariant to semantically equivalent paraphrases; this assumption fails when the LLM is consistently wrong but invariant, in which case variance is low and the calibration is overconfident. The number of paraphrases N controls the precision of the variance estimate; we choose N=5 as a trade-off between accuracy and cost, accepting that with small N the variance estimate is noisy and the calibration may be suboptimal. Overall, the method operationalizes the motivation-level concept of calibration by using paraphrase invariance as a proxy for correctness: low variance implies the evaluation is robust to surface form, which is necessary (but not sufficient) for good calibration. The failure mode is that robustness to paraphrases does not guarantee correctness against the true environment reward, but it reduces the risk of relying on spuriously sensitive evaluations.

## Contribution

(1) A training-free calibration method for LLM evaluators that uses paraphrase variance to compute a Bayesian shrinkage estimate, compatible with any LLM planner. (2) The design principle that paraphrase sensitivity is a cheap signal for LLM miscalibration, validated in text-based game planning. (3) Integration with MCTS (MC-DML) to improve planning robustness without additional training.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | Jericho text games | Provides ground-truth reward. |
| Primary metric | Expected Calibration Error | Directly measures calibration quality. |
| Baseline 1 | Single LLM evaluation | Baseline without calibration. |
| Baseline 2 | Average over paraphrases | No shrinkage, naive aggregation. |
| Ablation-of-ours | PARCA without shrinkage | Isolates effect of Bayesian shrinkage. |

### Why this setup validates the claim
We use Jericho games because they provide true rewards for every state-action pair, enabling direct calibration measurement. ECE is the natural metric since PARCA aims to produce well-calibrated scores. Comparing against Single LLM tests whether any paraphrasing helps; comparing against Average paraphrases tests whether shrinkage adds value; the ablation (no shrinkage) isolates the shrinkage component. If PARCA works, it should reduce ECE, and the improvement should be largest on instances where paraphrase variance is high – exactly where the Bayesian prior pulls uncertain estimates toward neutral. This setup ensures we can attribute any gains to the intended mechanism.

### Expected outcome and causal chain

**vs. Single LLM** — On a case where the state description uses atypical wording (e.g., "You see a frobnicated widget"), single LLM may produce an extreme score due to sensitivity. Because it has no calibration mechanism, it will be overconfident and wrong. Our method generates multiple paraphrases (e.g., "You see a strange device"), averaging out idiosyncrasies, and the shrinkage further reduces influence of high-variance scores. We expect a noticeable gap in ECE on instances with high paraphrasing variance (e.g., unusual language), but parity on simple, unambiguous states.

**vs. Average over paraphrases** — On a case where one paraphrase accidentally triggers a different meaning (e.g., "Open the door" vs "Unlock the door"), the average may be pulled toward an incorrect score. Our shrinkage pulls the estimate toward the neutral prior (0.5), dampening the outlier's effect. We expect lower ECE on high-variance instances because shrinkage reduces the impact of conflicting paraphrases.

**vs. Ablation (PARCA without shrinkage)** — On a case where paraphrase variance is high but the mean is close to the true reward, the ablation behaves similarly. However, when variance is high and the mean is off (e.g., due to systematic misinterpretation), shrinkage corrects toward prior. We expect the full method to outperform the ablation specifically on high-variance instances, demonstrating that shrinkage calibrates via uncertainty.

### What would falsify this idea
If PARCA's ECE improvement over Average paraphrases is uniform across all instances rather than concentrated on high-variance ones, then the claimed mechanism (shrinkage exploiting paraphrase variance) is false. Also, if PARCA performs worse than Single LLM on low-variance instances, the prior may be harming calibration.

## References

1. Monte Carlo Planning with Large Language Model for Text-Based Game Agents
2. Plan-Seq-Learn: Language Model Guided RL for Solving Long Horizon Robotics Tasks
3. How Can LLM Guide RL? A Value-Based Approach
4. Large Language Models as Commonsense Knowledge for Large-Scale Task Planning
5. Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory
6. Reason for Future, Act for Now: A Principled Framework for Autonomous LLM Agents with Provable Sample Efficiency
7. Bootstrap Your Own Skills: Learning to Solve New Tasks with Large Language Model Guidance
8. Language to Rewards for Robotic Skill Synthesis
