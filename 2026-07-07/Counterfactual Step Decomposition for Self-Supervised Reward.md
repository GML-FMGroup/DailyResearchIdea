# Counterfactual Step Decomposition for Self-Supervised Reward in Interleaved Multimodal Generation

## Motivation

Existing work like Bridging Interleaved Multi-Modal Reasoning as a Unified Decision Process relies on a separate VLM judge to assign dense intermediate rewards, introducing computational overhead and potential bias. This fails to establish self-supervised reward compositionality where intermediate rewards are derived from the trajectory itself. The root cause is that step-level advantage propagation assumes external reward signals are necessary for guiding intermediate steps, ignoring the causal structure inherent in sequential generation.

## Key Insight

The causal contribution of a single step to the final outcome can be isolated by comparing the factual trajectory to a counterfactual where that step is minimally perturbed, exploiting the Markov property of the generation process.

## Method

**(A) What it is:** Counterfactual Step Decomposition (CSD) is a self-supervised method that derives step-level rewards for interleaved multimodal generation by contrasting the factual trajectory with the same trajectory but with a single step perturbed. Input: a trajectory of alternating text tokens and image denoising steps; output: a dense reward for each step.

**(B) How it works:** For each step t in the trajectory (text token generation or image diffusion denoising step):
1. Create a counterfactual trajectory by replacing the action at step t with an alternative. For text tokens: replace the sampled token with the second most likely token from the policy. For image denoising steps: add Gaussian noise (σ=0.01) to the current latent at that step.
2. **Assumption and calibration:** We assume that later steps do not compensate for the perturbation. To verify this, we compute an alternative counterfactual where all subsequent actions (steps > t) are frozen to the factual trajectory's actions. If the two estimates differ by more than a threshold (e.g., 0.1 in outcome score), we flag the step as having strong compensation and use the frozen-action estimate instead. Otherwise, we use the standard rollout.
3. Roll out the remaining steps from the perturbed state using the same policy to obtain a complete counterfactual trajectory (or use frozen actions if calibration triggered).
4. Compute the outcome score for both factual and counterfactual trajectories using a self-supervised metric: CLIP similarity between the final text and image (for text+image tasks) or the log-likelihood of the trajectory under the model (for single-modality steps). The reward for step t is the difference: r_t = score(factual) - score(counterfactual_t).
5. Average over K=5 counterfactual samples per step to reduce variance.
6. Use the collected rewards to update the policy via PPO (learning rate 3e-5, clipping ε=0.2, value function coefficient 0.5, entropy coefficient 0.01), with separate advantage estimation for text and image modalities as in BRAID.

**(C) Why this design:** We chose single-step perturbation over multi-step to isolate causal effects per step, enabling precise credit assignment. We selected the second most likely token (rather than random or least likely) as the alternative because it is a minimal, realistic perturbation that remains in-distribution, avoiding degenerate counterfactuals. For image steps, additive Gaussian noise (σ=0.01) is used instead of full-step replacement (e.g., a different denoising direction) to ensure the perturbed state is near the original, maintaining Markov consistency. We used CLIP score as the outcome metric because it is task-relevant and requires no training, but it may not capture fine-grained consistency; we accept this limitation for self-sufficiency. Rolling out with the current policy introduces variance from later steps, but we trade off computational cost for unbiased estimates. The choice of PPO over DPO (which requires paired preferences) is because our rewards are continuous and contrastive, fitting the RL framework. This design avoids any external judge, making the pipeline fully self-supervised, at the cost of increased compute for counterfactual rollouts.

**(D) Why it measures what we claim:** The quantity score(factual) - score(counterfactual_t) measures the **causal necessity** of step t to the final outcome, because the difference reflects the change in outcome when only step t is altered under the assumption that generation is Markov and the perturbation is minimal. This assumption fails when steps have strong interactions (e.g., a later step compensates for an earlier perturbation), in which case the difference captures a **local effect confounded by mediation**. The computational quantity 'score' (CLIP similarity) measures **task success** under the assumption that CLIP aligns with human judgement of multimodal coherence; this assumption fails when the generation contains subtle semantic mismatches that CLIP does not penalize, in which case the reward reflects **surface-level alignment**. The counterfactual rollout procedure assumes that the policy is invariant to the perturbation, which holds for small perturbations but fails for large ones; by using minimal perturbations, we approximate invariance. Finally, averaging multiple counterfactuals measures **expected causal effect** under the assumption that perturbations are independent; this assumption fails if the policy's stochasticity creates dependencies, in which case the average reflects a **noisy estimate** of direct effect. **Additional remark on reward hacking:** The reward measures causal necessity under the assumption that the CLIP score is a faithful proxy for task success; this assumption fails if the model learns to exploit biases in CLIP (reward hacking), in which case the reward reflects spurious correlations. A robustness check is to evaluate with a different metric (e.g., human judgment) on a held-out validation set; we include this check in the experiment.

## Contribution

(1) Counterfactual Step Decomposition (CSD), a self-supervised reward decomposition method that derives step-level rewards from counterfactual perturbations without an external judge. (2) The design principle that minimal single-step perturbations suffice for causal credit assignment in sequential multimodal generation, exploiting the Markov structure. (3) A fully self-supervised RL pipeline for interleaved text-image generation that removes the need for any separate reward model or human annotation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | Visual Storytelling (VIST) | Multi-turn multimodal coherence evaluation |
| Primary metric | CLIP score | Aligned with reward and task goal |
| Baseline 1 | UMM SFT | Tests necessity of RL |
| Baseline 2 | BRAID (separate advantages) | Tests role of step-level rewards |
| Baseline 3 | COMA (counterfactual multi-agent) | Tests against multi-agent credit assignment |
| Baseline 4 | Shapley value approximation | Tests against cooperative game-theoretic credit |
| Ablation-of-ours (1) | CSD with random token perturbation | Tests minimal perturbation design |
| Ablation-of-ours (2) | CSD with K=1 counterfactual sample | Tests compute-performance tradeoff |

### Why this setup validates the claim
This combination directly probes the central claim that CSD enables precise credit assignment for interleaved multimodal generation. Comparing against SFT isolates the benefit of RL over supervised learning, while BRAID controls for the effect of separate modality advantage estimation—if CSD outperforms BRAID, the improvement must come from per-step counterfactual rewards rather than modular separation. Adding COMA and Shapley baselines tests whether CSD's counterfactual approach is more effective than alternative credit assignment methods. The ablation with random token perturbation tests whether the specific choice of second-most-likely token is critical for stable counterfactual estimation. The K=1 ablation tests whether multiple counterfactual samples are necessary for variance reduction. CLIP score is the appropriate metric because it mirrors the outcome signal used in CSD's reward, ensuring that the optimization objective aligns with evaluation. As a robustness check against reward hacking, we also evaluate with human judgment on a 200-example held-out set. Together, these components create a falsifiable test: if the method works, it should show gains precisely where credit assignment is most challenging (e.g., long chains of interdependent steps), and the ablations should underperform on variance-sensitive tasks.

### Expected outcome and causal chain

**vs. UMM SFT** — On a case where a single early text token (e.g., "not") reverses the meaning of a later caption, SFT cannot backpropagate reward to that token because it uses uniform step-level loss. Our method assigns high reward to that token because perturbing it drastically lowers CLIP score. We expect a noticeable gap on lengthy generations with semantic dependencies, but parity on short or simple sequences where credit assignment is trivial.

**vs. BRAID** — On a case where a text token and image denoising step are tightly coupled (e.g., generating a "cat" and the corresponding latent must align), BRAID's separate advantage estimates may over-credit the image step and under-credit the text token. Our method directly measures the causal effect of each step via counterfactual contrast, so it allocates reward more accurately. We expect CSD to outperform BRAID on cross-modal coordination subsets, especially when steps are interleaved.

**vs. COMA** — COMA uses a counterfactual baseline that averages over other agents' actions but relies on a learned critic. Our method uses the actual outcome difference with minimal perturbation, which may be more accurate when the critic is misspecified. We expect CSD to be more sample-efficient and robust to critic bias.

**vs. Shapley value** — Shapley value requires marginalizing over subsets of steps, which is intractable for long trajectories. Our single-step perturbation is a first-order approximation; if Shapley provides better credit but at higher cost, CSD offers a favorable trade-off. We expect CSD to match Shapley performance on short trajectories but outperform on longer ones due to feasibility.

**vs. CSD with random token perturbation** — On a case where a token has a narrow distribution (e.g., generating "the" in a specific context), a random perturbation may jump to an unlikely token, producing an invalid counterfactual and noisy reward. Our method uses the second most likely token, which remains in-distribution. We expect our method to have lower reward variance and faster convergence, observable as a smaller gap between training and validation rewards.

**vs. CSD with K=1 counterfactual sample** — Using only one counterfactual sample increases variance but halves compute. We expect K=1 to still provide useful rewards but with slower convergence and higher variance, especially on early steps where multiple perturbations are needed for stable estimates.

### What would falsify this idea
If CSD's gains over BRAID are uniform across all step types (text vs image) rather than concentrated on interleaved steps where causal effects are hard to isolate, or if the ablation with random perturbation matches CSD's performance, then the central claim that minimal perturbation is critical would be falsified. Additionally, if human judgment metrics diverge from CLIP score trends, the assumption of no reward hacking is violated.

## References

1. Bridging Interleaved Multi-Modal Reasoning as a Unified Decision Process
2. X-Omni: Reinforcement Learning Makes Discrete Autoregressive Image Generative Models Great Again
3. DiffusionNFT: Online Diffusion Reinforcement with Forward Process
4. LMFusion: Adapting Pretrained Language Models for Multimodal Generation
5. TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation
6. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
7. Toward Guidance-Free AR Visual Generation via Condition Contrastive Alignment
8. Using Human Feedback to Fine-tune Diffusion Models without Any Reward Model
