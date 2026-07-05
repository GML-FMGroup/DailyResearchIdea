# Online Goal Belief Abstention: Dynamic Abstention via Bayesian Goal Inference

## Motivation

Existing abstention mechanisms such as those evaluated in Agentic Abstention assume static, well-defined goals, leading to one-shot abstention decisions that fail when objectives are underspecified or shift during interaction. Agentic Abstention's analysis reveals that timing of abstention is a key challenge, but no method incorporates goal uncertainty reduction. This structural limitation prevents agents from strategically continuing interaction to resolve ambiguity rather than prematurely or belatedly abstaining.

## Key Insight

The expected reduction in goal uncertainty after an action provides a principled, online criterion for abstention that separates cases where further interaction is informative from those where it is not.

## Method

**Online Goal Belief Abstention (OGBA)**

(A) **What it is**: OGBA maintains a learned belief distribution over possible goals, updated after each agent action and observation, and decides to abstain when the expected reduction in goal entropy (ERGE) from the best next action falls below a threshold τ.

(B) **How it works**:
```python
# Pseudocode for OGBA
class OGBA:
    def __init__(self, goal_encoder, erg_predictor, threshold=0.1):  # τ=0.1 nats
        self.goal_encoder = goal_encoder  # 2-layer LSTM, hidden=128, output 128-d belief encoding; trained via cross-entropy on ground-truth goals
        self.erg_predictor = erg_predictor  # 2-layer MLP (hidden=64, ReLU): (128-d belief_encoding, action) → scalar ERGE; trained with MSE loss; calibrated via temperature scaling on held-out validation set of 512 episodes
        self.threshold = threshold
        self.history = []

    def update_belief(self, action, observation):
        self.history.append((action, observation))
        belief = self.goal_encoder(self.history)  # 128-d vector representing belief over goals
        return belief

    def expected_info_gain(self, action, belief):
        return self.erg_predictor(belief, action)

    def decide(self, state):
        belief = self.update_belief(None, state)
        while not done:
            action_ergs = {a: self.expected_info_gain(a, belief) for a in actions}
            best_action = max(action_ergs, key=action_ergs.get)
            if action_ergs[best_action] < self.threshold:
                return 'abstain'
            else:
                observation = environment.execute(best_action)
                belief = self.update_belief(best_action, observation)
```
Hyperparameters: threshold τ=0.1 nats, learning rate for ERGE predictor=1e-3, belief encoding size=128, ERGE MLP hidden size=64.

(C) **Why this design**: We chose to train a separate ERGE predictor rather than computing expected reduction analytically because the state dynamics and observation model may be complex or unknown; a learned predictor can approximate the information gain without explicit models. We chose a small neural network (2-layer MLP with 64 hidden units) over a larger model to keep inference fast; the cost is that training data must cover diverse belief states and may generalize poorly. To mitigate miscalibration, we apply temperature scaling to the ERGE predictor's outputs using a held-out validation set of 512 episodes. We set a fixed threshold τ rather than learning it because τ is a design choice for risk tolerance; learning it could lead to undesirable behavior in out-of-distribution scenarios. We use a belief encoder trained with supervised learning on full trajectories (using cross-entropy loss against ground-truth goals) rather than online Bayes because maintaining exact posteriors over complex goal spaces is intractable; the trade-off is that the encoder may not capture true uncertainty if the goal space is misspecified or the encoder is overconfident. Finally, we use a deterministic action selection based on max ERGE rather than a stochastic policy because abstention decisions should be deterministic for reproducibility and safety.

(D) **Why it measures what we claim**: The entropy H(B_t) measures goal uncertainty because, under the assumption that the goal space is exhaustive and mutually exclusive, entropy captures the residual ambiguity. The expected reduction in entropy ERGE measures the potential to resolve uncertainty via one more action, because it quantifies how much the posterior is expected to concentrate on a single goal. This assumption fails when the observation model is not informative (e.g., sensor noise), in which case ERGE is near zero even if uncertainty is high, indicating that abstention is appropriate. The threshold τ operationalizes the decision boundary for abstention: setting τ low means the agent continues until virtually no uncertainty reduction expected, aligning with conservative abstention. The belief encoder's output distribution is assumed to be a calibrated belief over goals; this assumption fails when the encoder is miscalibrated (e.g., overconfident), in which case ERGE may be underestimated, causing premature abstention. The ERGE predictor assumes that the expected entropy after one action can be approximated by averaging over possible observations; this fails when observations are multimodal and the predictor averages over them, possibly underestimating the reduction if some observations are very informative. The assumption that the learned ERGE predictor accurately approximates the true expected information gain is load-bearing; we address miscalibration via temperature scaling on a held-out validation set.

## Contribution

(1) A novel framework for dynamic agent abstention that treats goal inference as an online process, enabling abstention decisions based on expected reduction in goal uncertainty. (2) A concrete instantiation using a learned belief encoder and ERGE predictor, demonstrating that the expected information gain criterion separates informative from uninformative actions. (3) A design principle that abstention should be a continuous, context-sensitive decision rather than a static threshold.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|---|---|---|
| Dataset | Goal-Oriented Gridworld | Goals unknown, partial observations, clear abstention threshold. |
| Primary metric | Optimality Rate vs oracle | Directly tests goal uncertainty resolution. |
| Baseline 1 | No abstention (always act) | Agents never stop early; baseline for necessity. |
| Baseline 2 | Confidence threshold | Simple uncertainty heuristic; baseline for info-gain edge. |
| Baseline 3 | Random abstention | Random stopping; baseline for informed vs arbitrary. |
| Ablation | OGBA with heuristic entropy threshold | Tests value of learned ERGE predictor. |

### Why this setup validates the claim
This setup directly tests OGBA's central claim: that predicting expected reduction in goal entropy (ERGE) enables superior abstention decisions over simple heuristics. The Goal-Oriented Gridworld provides a controlled environment with known ground-truth goals, allowing us to compute an oracle optimal abstention decision for each episode (stop when no action reduces entropy). The baselines isolate key failure modes: no abstention shows the need for any stopping; confidence threshold tests whether posterior uncertainty alone suffices; random abstention tests the value of informed decisions. The ablation removes the learned ERGE predictor to isolate its contribution. The optimality rate metric linearly penalizes mismatches with the oracle, directly measuring alignment with information-theoretic abstention. We also evaluate calibration of the ERGE predictor via expected calibration error (ECE) on held-out episodes. If OGBA's decisions match the oracle more often than baselines, the claim is supported; if not, the core mechanism is flawed.

### Expected outcome and causal chain
**vs. No abstention (always act)** — On an episode where the agent is equally uncertain between two goals but no further observations can disambiguate (e.g., both goals lead to identical sensors), the baseline continues acting indefinitely, eventually failing to identify the goal. OGBA detects that all actions have low ERGE (below threshold τ) and abstains, yielding an optimal stop. We expect OGBA to achieve near-perfect optimality on such uninformative episodes while the baseline fails completely, producing a large gap in overall optimality rate.

**vs. Confidence threshold** — Consider an episode where the agent's belief encoder overconfidently assigns 90% probability to a wrong goal (due to biased training data). The confidence-threshold baseline (e.g., abstain if max confidence < 0.9) would act on the wrong goal, while OGBA's ERGE predictor correctly estimates low expected info gain because further actions cannot resolve the bias, so OGBA abstains. Conversely, on an episode where all goals are equally likely but one action could disambiguate, confidence threshold abstains needlessly (low confidence triggers early stop), whereas OGBA's high ERGE for the informative action prompts it to act. We expect OGBA to outperform both on low-true-confidence and high-entropy-but-informative cases, with a noticeable gap in optimality rate on these subsets.

**vs. Random abstention** — In an episode where the true goal is identifiable after two steps, random abstention with 20% stop probability may stop prematurely after the first step (uncertain) or continue past the point where abstention would be optimal (e.g., after full identification). OGBA's deterministic ERGE-based decision will stop only when no further info gain is expected, so it will always act until the goal is known or uninformative. We expect OGBA to achieve high and consistent optimality rates, while random abstention yields a lower average optimality (around 80% if correct stop probability is low) and higher variance across episodes.

**vs. Ablation: heuristic entropy threshold** — Consider an episode where current posterior entropy is low (e.g., 0.3 nats) but one action could reduce it to near zero (ERGE = 0.2 nats). The heuristic baseline abstains because entropy < τ (assume τ = 0.4), missing the chance to confirm, while OGBA's ERGE predictor (threshold τ on ERGE) sees 0.2 > 0.1 and acts. Conversely, if entropy is high (0.8) but no action reduces it (ERGE ≈ 0), the heuristic acts (entropy > τ) while OGBA abstains. We expect OGBA to outperform the heuristic on episodes where current entropy is moderate but further info is available, and where entropy is high but info gain is low, leading to a clear gap in optimality rate.

### What would falsify this idea
If OGBA's optimality rate is not significantly higher than the heuristic entropy threshold baseline on episodes where posterior entropy is high but the oracle says abstain (i.e., uninformative observations), then the ERGE predictor is not capturing the lack of info gain, and the core claim fails. Additionally, if the ERGE predictor exhibits high expected calibration error (ECE >0.1) on held-out episodes, the load-bearing assumption of accurate prediction is violated, undermining the method's reliability.

## References

1. Agentic Abstention: Do Agents Know When to Stop Instead of Act?
2. Check Yourself Before You Wreck Yourself: Selectively Quitting Improves LLM Agent Safety
3. AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions
4. Identifying the Risks of LM Agents with an LM-Emulated Sandbox
5. When Not to Answer: Evaluating Prompts on GPT Models for Effective Abstention in Unanswerable Math Word Problems
6. From Blind Solvers to Logical Thinkers: Benchmarking LLMs' Logical Integrity on Faulty Mathematical Problems
7. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents
8. UNK-VQA: A Dataset and a Probe Into the Abstention Ability of Multi-Modal Large Models
