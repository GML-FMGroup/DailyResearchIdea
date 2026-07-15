# MetaProto: Meta-Learning Verification Protocol Synthesis from Task Descriptions

## Motivation

The AMID framework [Towards Autonomous and Auditable Medical Imaging Model Development] demonstrates state-of-the-art performance via verification-guided optimization, but it requires predefined validation protocols, metric definitions, and prediction artifacts. This bottleneck prevents autonomous deployment on novel tasks where no such protocol is available a priori. Existing retrieval or LLM-based baselines cannot guarantee that generated protocols are sufficient to guide the search toward correct models because they lack a mechanism to simulate and validate the search dynamics.

## Key Insight

Protocol sufficiency can be guaranteed by training a generator to maximize the predicted outcome of a differentiable surrogate of the verification-driven search, because the surrogate internalizes the structural dependencies between protocol choices and search success.

## Method

# MetaProto: Meta-Learning Verification Protocol Synthesis

## (A) What it is
MetaProto is a meta-learning framework that synthesizes verification protocols from task descriptions by training a protocol generator using a learned sufficiency predictor as a reward signal. Input: task description T (structured as text or 512-dim embedding of dataset characteristics, modality, prediction type). Output: protocol P = (validation scheme V, metric M, prediction artifacts A).

## (B) How it works

```pseudocode
# Hyperparameters: generator hidden size 512, temperature tau=1.0, learning rate lr=1e-4, reward model epochs N=100, reward model hidden size 256, reward model layers=3, reward model activation=GeLU, uncertainty head output dim=2 (mean and variance), rejection threshold epsilon=0.1

# Phase 1: Train sufficiency predictor R (neural network with uncertainty head)
# Data: D = {(T_i, P_i, s_i)} from prior AMID runs; s_i = final performance after verification-guided search
Train R(T, P) to minimize negative log-likelihood loss assuming Gaussian output: L = (s_i - mu)^2 / (2*sigma^2) + 0.5*log(sigma^2), where (mu, sigma^2) = R(T,P).

# Phase 2: Train protocol generator G (transformer, e.g., GPT-2 small: 12 layers, 768 hidden, 12 heads, 124M params)
Initialize G randomly.
for each task T in training set (unlabeled tasks):
    Sample protocol P ~ G(T) via autoregressive decoding with temperature tau.
    Compute (mu, sigma^2) = R(T, P), reward r = mu.
    If sigma > epsilon: fall back to baseline protocol (default LLM-generated). else use r.
    Update G parameters via REINFORCE with baseline (mean reward over batch).
    # Optionally, add a diversity penalty -lambda * H(G) to encourage exploration; lambda=0.01.
end for

# Inference: Given new task T_new, generate P* = argmax_P G(T_new) (greedy decoding) as the verification protocol.
```

## (C) Why this design
We chose a transformer-based generator over an LSTM because it captures long-range dependencies in protocol structure (e.g., interaction between validation scheme and metric choice), accepting higher computational cost. We used a learned neural reward model instead of a rule-based sufficiency check because rules would oversimplify the complex search dynamics (e.g., interaction with data distribution), at the cost of requiring offline training data from prior AMID runs. We adopted REINFORCE over direct backpropagation through the reward model because the generator's sampling step is discrete; the trade-off is higher variance in gradient estimates, which we mitigate by using a baseline and temperature annealing. These decisions collectively enable the generator to learn protocol patterns that are empirically sufficient in the verification-driven search. We added an uncertainty estimation head to R (outputting mean and variance) and a rejection rule (fallback to baseline if uncertainty exceeds threshold epsilon=0.1) to handle distribution shift, as suggested by the adversarial alert.

## (D) Why it measures what we claim
**Computational quantity R(T,P)** measures **predicted sufficiency** because, by construction, it is trained to approximate the final performance achieved by AMID's verification-guided optimization given T and P; the assumption is that the optimization dynamics (search space, verification checks) are stationary across tasks and well-covered by the training data. This assumption fails when a new task exhibits a fundamentally different search landscape (e.g., unseen modality), in which case R reflects the most similar seen domain rather than true sufficiency. **The reward r** in REINFORCE operationalizes **protocol quality** because it is the primary signal that drives G toward high-sufficiency protocols; the assumption is that R's predictions correlate monotonically with true sufficiency. This assumption fails when R is poorly calibrated (e.g., due to distribution shift), in which case G may optimize a spurious correlate (e.g., protocol length). **G's autoregressive decoding** with temperature tau operationalizes **exploration** because it allows sampling diverse protocols; the assumption is that temperature-controlled sampling covers the sufficiency landscape. This assumption fails when tau is too low (mode collapse) or too high (noise overwhelms signal). **The uncertainty head and rejection rule** measure **predictive uncertainty** and provide a safeguard when assumptions are violated.

## Contribution

(1) A meta-learning framework MetaProto that synthesizes verification protocols from task descriptions without human input, using a learned sufficiency predictor as a reward model. (2) A design principle that protocol sufficiency can be guaranteed by training against a surrogate of the verification-driven search dynamics, enabling generalization to new tasks where no protocol is available a priori.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | 20 medical imaging challenge tasks: 10 radiology (CT, MRI, X-ray) and 10 pathology (whole-slide, histology) | Diverse modalities and prediction types, two distinct domains |
| Primary metric | Average final AMID performance (e.g., AUC or Dice) on held-out test set of each task | Directly measures protocol sufficiency |
| Baseline 1 | General-purpose MLE (fixed random split, accuracy) | No verification protocol synthesis |
| Baseline 2 | Standard LLM agent (no prior; GPT-4 with few-shot examples) | No meta-learned protocol generation |
| Baseline 3 | Bayesian optimization with Gaussian process (meta-learned hyperparameter optimizer) | Isolates contribution of protocol-specific structure |
| Ablation 1 | MetaProto w/o reward model (generator trained with MLE on prior protocols) | Isolates effect of learned sufficiency predictor |
| Ablation 2 | MetaProto with LSTM generator (2-layer LSTM, hidden=256) | Tests simpler architecture as proof-of-concept |
| Synthetic experiment | Simulated search with known ground-truth sufficiency (random quadratic function of protocol features) | Verifies R learns correct proxy |

### Why this setup validates the claim

This combination tests the central claim that meta-learning verification protocols improves sufficiency. The dataset spans two domains (radiology and pathology), challenging the generator to adapt across modalities and enabling zero-shot transfer evaluation. The primary metric captures the end-to-end impact of protocol quality. Comparing against baselines without meta-learned generation isolates the benefit of learning from prior tasks. The Bayesian optimization baseline controls for the effect of meta-learning itself, isolating the protocol-specific structure. The ablation without reward model tests whether the learned sufficiency predictor provides a critical reward signal; the LSTM ablation specifies the generator choice. The synthetic experiment provides a controlled environment where ground-truth sufficiency is known, allowing direct verification that R learns a correct proxy. If MetaProto outperforms baselines and ablations, it confirms both generator and reward model contribute to better protocols, providing a falsifiable test. Compute budget: approximately 2000 GPU hours on a single NVIDIA A100 for training the reward model and generator (estimated 1000 hours for Phase 1, 1000 hours for Phase 2).

### Expected outcome and causal chain

**vs. General-purpose MLE** — On a case with an unusual modality (e.g., pathology whole-slide), the baseline uses a fixed validation scheme (e.g., random split) and generic metric (e.g., accuracy), leading to poor generalization because it ignores task-specific verification needs. Our method generates a protocol tailored to the modality (e.g., stratified cross-validation with Dice score), leveraging the learned reward model that captures sufficiency from similar tasks. We expect a noticeable gap (e.g., 10-20% relative improvement) on unusual-modality tasks but parity on standard tasks where default protocols work well.

**vs. Standard LLM agent (no prior)** — On a case requiring complex prediction type (e.g., multi-class segmentation), the baseline samples protocols via trial-and-error without learned priors, causing inefficient exploration and low-quality protocols. Our method uses meta-learned generator to directly produce high-sufficiency protocols, guided by the reward model. We expect a consistent gap (e.g., 5-15% relative improvement) across all tasks, with larger differences on tasks requiring complex protocols.

**vs. Bayesian optimization with GP** — On a task with interdependent protocol choices (e.g., validation scheme and metric interact), the baseline treats protocol components as independent hyperparameters, missing structure. Our generator captures dependencies via transformer. We expect a 5-10% relative improvement on tasks with strong interactions.

### What would falsify this idea

If our method's performance is not significantly better than the ablation (MetaProto w/o reward model) across most tasks, then the learned sufficiency predictor is not driving improvements and the core claim fails. Additionally, if gains are uniform across all tasks rather than concentrated on tasks with unusual modalities or complex protocols, then the generator is merely memorizing common patterns rather than adapting to task-specific verification needs. In the synthetic experiment, if R's predicted sufficiency does not correlate well (Spearman ρ<0.5) with true sufficiency, the reward model is inadequate.

## References

1. Towards Autonomous and Auditable Medical Imaging Model Development
2. AutoMLGen: Navigating Fine-Grained Optimization for Coding Agents
3. An AI system to help scientists write expert-level empirical software
4. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
5. DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning
6. ExpeL: LLM Agents Are Experiential Learners
7. Automatic Chain of Thought Prompting in Large Language Models
