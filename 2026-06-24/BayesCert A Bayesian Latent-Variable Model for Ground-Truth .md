# BayesCert: A Bayesian Latent-Variable Model for Ground-Truth Confidence in Coding Agent Benchmarks

## Motivation

Existing coding agent benchmarks, such as NatureBench, assume published scientific results are perfectly accurate ground truth, but they provide no mechanism to quantify the reliability of those results. This assumption, inherited from prior work like OpenScholar and MLE-bench, risks misleading evaluation if a task's published result is erroneous or irreproducible. Our model directly addresses this meta-gap by inferring a confidence measure for each ground-truth label without requiring external verification, using multiple agent outputs and paper metadata.

## Key Insight

The posterior probability of a published result's correctness is identifiable because multiple agent outputs are conditionally independent given the true effect, and metadata features predict systematic reproducibility failures, acting as instrumental variables.

## Method

(A) **What it is:** BayesCert is a hierarchical Bayesian latent-variable model that takes as input: for each of N tasks, M coding agent performance scores (e.g., g>0.1 indicator), and K metadata features (e.g., journal impact factor, sample size, code availability flag). It outputs the posterior probability p(z=1 | data) that the published ground-truth label is correct. (B) **How it works:** \
\
```pseudocode
for each task t in 1..N:
  z_t ~ Bernoulli(sigmoid(β^T x_t))                     # prior from metadata x_t
  for each agent i in 1..M:
    if z_t == 1:
      y_{ti} ~ Normal(μ_i, σ_i)                         # agent i's typical performance on valid tasks
    else:
      y_{ti} ~ Normal(μ_0, σ_0)                         # baseline performance on invalid tasks
# Hyperparameters:
# β ~ Normal(0, 2)            (coefficients for K metadata features)
# μ_i ~ Normal(0, 1)          (agent-specific mean for correct tasks)
# σ_i ~ Gamma(2, 1)           (agent-specific noise)
# μ_0 ~ Normal(0, 1)          (baseline mean for incorrect tasks)
# σ_0 ~ Gamma(2, 1)           (baseline noise)
# Inference: Hamiltonian Monte Carlo (e.g., via Stan) over all hyperparameters jointly.
# Posterior p(z_t=1 | data) computed from MCMC samples.
```
(C) **Why this design:** We chose a Bayesian latent-variable model over simple aggregation (e.g., averaging agent scores) because aggregation ignores metadata that systematically predict reproducibility (e.g., small-sample studies are more likely false), and it does not produce calibrated confidence. The cost is added computational overhead (MCMC) and prior specification. We chose conditional independence of agents given z over a joint correlation model because it simplifies inference and is justified if agents are independently run; the cost is that correlated errors (e.g., agents sharing architectures) violate this assumption, leading to overconfident posteriors. We chose agent-specific means μ_i over a shared mean because agents have different baseline capabilities (as seen in NatureBench rankings); the trade-off is increased parameters, requiring sufficient tasks per agent for reliable estimation. We chose HMC over variational inference for accuracy given the small scale (N=90 tasks) and the need for well-calibrated uncertainty. (D) **Why it measures what we claim:** The posterior probability p(z=1|data) measures ground-truth confidence because it integrates the consistency of agent outputs with the metadata-informed prior. Specifically, the computational quantity p(z=1|data) = p(data|z=1) * p(z=1) / p(data) where p(data|z=1) assumes agent outputs are drawn from their typical distributions; this quantity measures confidence because under the assumption that agent outputs are conditionally independent given z and that metadata features predict prior odds of correctness, the posterior is a calibrated probability of correctness in the frequentist sense (i.e., if we threshold at 0.9, 90% of tasks with posterior >0.9 are truly correct). This assumption fails when agents share systematic biases (e.g., all agents fail on the same type of task), in which case the posterior may over- or under-estimate confidence. Additionally, the metadata term p(z=1) measures baseline reproducibility risk via features like sample size; this assumes that these features are predictive in the population of published results, but fails if the dataset is not representative (e.g., Nature-family papers may have different reproducibility rates).

## Contribution

(1) A Bayesian latent-variable framework that jointly models multiple coding agent outputs and paper metadata to infer a calibrated confidence score for each benchmark ground-truth label, without external verification. (2) A set of design principles for operationalizing reproducibility indicators (e.g., journal impact, sample size, code availability) into a prior that captures systematic biases in published results. (3) An open-source implementation of the inference pipeline using Hamiltonian Monte Carlo, enabling easy integration into existing benchmarks.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | NatureBench (90 tasks) | Published ground truth, realistic diversity. |
| Primary metric | Expected Calibration Error (ECE) | Measures probability calibration directly. |
| Baseline 1 | Majority vote of agents | Simple aggregation baseline. |
| Baseline 2 | Average agent scores thresholded | Common baseline for consensus. |
| Baseline 3 | Logistic regression on metadata only | Isolates metadata contribution. |
| Ablation of ours | BayesCert without metadata | Tests importance of metadata prior. |

### Why this setup validates the claim
This design directly tests whether BayesCert produces better-calibrated confidence estimates than simpler alternatives. NatureBench provides a compact set of tasks with known ground-truth labels and rich metadata, enabling a controlled test. Majority vote and average scores represent the status quo of ignoring metadata, while logistic regression on metadata alone tests the added value of agent performance signals. The ablation removes the metadata prior, isolating its effect. ECE is chosen because the central claim is about calibrated probability of correctness—not just binary accuracy. If BayesCert yields lower ECE than all baselines, especially on subsets where metadata strongly correlates with reproducibility (e.g., small-sample tasks), it validates the claim that integrating metadata via a hierarchical Bayesian model improves confidence estimation. The small N (90 tasks) is sufficient for HMC and allows close inspection of failure modes.

### Expected outcome and causal chain
**vs. Majority vote** — On a task with small sample size (n=20) where all agents happen to score high due to a fluke, majority vote confidently labels it correct because it counts agent agreement. BayesCert instead downweights the posterior via the metadata prior (small n increases false-positive risk) and by checking if agent scores are typical (if μ_i is low, high scores are unlikely given z=1). Thus, we expect BayesCert to have much lower ECE on small-sample tasks, while parity on large-sample tasks.

**vs. Average agent scores** — On a task where one agent is systematically overconfident but others are accurate, averaging may yield a middling score that, when thresholded, misclassifies the task. BayesCert uses agent-specific means μ_i, so that agent's deviation is penalized via the likelihood: if y_{ti} is far from μ_i, it reduces p(z=1). Therefore, we expect BayesCert to achieve better calibration on tasks where agents have heterogeneous biases, observable as a noticeable gap in ECE on tasks with identified outlier agents.

**vs. Logistic regression on metadata only** — On a task with strong metadata signals (e.g., high impact factor, large sample) but where agents collectively fail due to a subtle reasoning error, logistic regression would still assign high prior confidence (p(z=1) high) because metadata looks good. BayesCert, by also conditioning on agent scores (which are low), can override the metadata prior. Thus, we expect BayesCert to have lower ECE on tasks where agent scores contradict metadata—e.g., low agent scores on high-reputation journals—while matching logistic regression on tasks where agent scores align.

### What would falsify this idea
If BayesCert's ECE is not significantly lower than majority vote or average scores on the subset of tasks small in sample size (or with known reproducibility issues), or if the ablation without metadata performs equally well, then the hierarchical Bayesian model provides no calibration benefit, falsifying the claim that integrating metadata and agent-specific distributions improves confidence estimation.

## References

1. NatureBench: Can Coding Agents Match the Published SOTA of Nature-Family Papers?
2. OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs
3. Autonomous chemical research with large language models
4. MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering
5. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
6. MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation
7. Closed-loop optimization of general reaction conditions for heteroaryl Suzuki-Miyaura coupling
8. CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis
