# Interventional Token Credit for Diffusion Language Model Reinforcement Learning

## Motivation

Existing token-credit estimators for diffusion-LM RL—confidence improvement in DTCA, ELBO surrogates in GDPO, and importance-weighted process rewards—are observational proxies that never define or measure the causal effect of committing a specific token on the final outcome. These proxies converge to the true interventional credit only under an unstated trajectory-ignorability condition (that the committed token is independent of the remaining denoising path given the current state), which fails whenever token commitments influence later denoising steps, as they do in masked diffusion. Without an explicit do()-intervention and a checkable identifiability condition, these estimators can assign credit to tokens that merely correlate with good outcomes rather than cause them.

## Key Insight

In diffusion-LM denoising, the do()-intervention on a token commitment is physically realizable by resampling that token while holding all other commitments and the noise schedule fixed, and the resulting interventional contrast is identified by a shared-randomness coupling that cancels the trajectory-level noise common to factual and counterfactual paths.

## Method

**What it is**: ITC (Interventional Token Credit) is a credit assignment method for reinforcement learning of diffusion language models. Given a denoising trajectory that commits tokens sequentially, ITC produces for each committed token an estimate of the total causal effect of that token's commitment on the final reward by comparing the factual trajectory to a counterfactual trajectory in which that token's commitment is resampled (do()-intervention) and the suffix is re-denoised under the model's conditional, while the noise schedule is held fixed via shared random seeds. The estimand is the total effect of the intervention (direct effect plus induced suffix cascade), not a local single-token effect; we state this explicitly and measure the cascade.

**How it works**:
```pseudocode
Input: diffusion LM p_theta, reward function R, denoising schedule {alpha_t}, number of tokens N, counterfactual samples per token K=5
For each rollout:
  1. Sample a factual denoising trajectory tau = (x_0, ..., x_T) using seeds s_1..s_T, obtaining committed tokens z_1..z_N, their commit steps t_1..t_N, and final output y.
  2. Compute factual reward r = R(y).
  3. For each committed token i (i=1..N):
     a. Identify the denoising step t_i where token i was committed.
     b. For k=1..K: construct a counterfactual trajectory tau^(i,k) by replaying the same seeds s_1..s_T, but at step t_i, replace the committed token with a sample z_i' from the model's conditional distribution given the same context (i.e., do(z_i := z_i')), and then continue denoising with the same seeds. The suffix is re-denoised under the model's conditional; this is the defined intervention, not an assumption of locality.
     c. Compute counterfactual reward r^(i,k) = R(y^(i,k)).
     d. Interventional credit c_i = r - (1/K) sum_k r^(i,k).
     e. Record the commit-schedule divergence d_i = 1 if the set or order of later committed positions differs from factual, else 0.
  4. Use credits {c_i} as token-level advantages in a policy gradient update (REINFORCE with baseline b = average reward across rollouts in the batch).
```
The shared-randomness coupling ensures that the only difference between factual and counterfactual trajectories is the intervened token and the suffix cascade it induces; all other stochasticity (noise schedule) is held fixed. This yields an estimate of the token's total interventional effect under the assumption that the noise schedule is exogenous (independent of token commitments).

**Why this design**: We chose a do()-intervention with shared randomness over observational proxies like confidence improvement (DTCA) or ELBO (GDPO) because only the interventional contrast isolates the causal effect of a token commitment from correlated but non-causal factors (e.g., a token may be committed with high confidence simply because the context is easy, not because it causes the final answer). The cost is that we must replay each trajectory N times to compute per-token credits, increasing computational cost by a factor of N; however, this is feasible because diffusion-LM denoising is re-playable and we can batch counterfactual replays. We chose to resample only the intervened token while holding all other commitments fixed (rather than resampling the entire suffix) because in diffusion-LMs the denoising path is a sequence of token commitments, and changing one commitment does not force a cascade of changes as in autoregressive models; this makes the intervention local and the credit attributable. We chose to use the raw reward difference as the credit rather than a learned value function because it directly targets the interventional contrast and avoids introducing approximation error from a critic; the trade-off is higher variance, which we mitigate by averaging over multiple counterfactual samples per token (K=5) and using a baseline (average reward across rollouts). We chose to use REINFORCE with these credits rather than PPO because the credits are already token-level advantages and we avoid the complexity of clipping and multiple epochs; the trade-off is potentially slower convergence, but the unbiasedness of the credits compensates.

**Why it measures what we claim**: The computational quantity c_i = r - (1/K) sum_k r^(i,k) measures the total interventional causal effect of committing token i on the final reward because we enforce the do()-operation by resampling only token i and re-denoising the suffix under the model's conditional while holding the noise schedule fixed via shared random seeds; this is equivalent to the contrast E[R | do(z_i = a)] - E[R | do(z_i = b)] under the assumption that the noise schedule is exogenous (i.e., the random seeds used for denoising are independent of the token commitments). This assumption fails when the model's policy for choosing token commitments depends on the noise schedule (e.g., if the model uses the noise level to decide which token to commit), in which case the shared-randomness coupling does not fully isolate the token's effect and c_i may reflect some correlation between the token and the noise schedule. However, in standard masked diffusion models, the noise schedule is fixed and independent of the model's token choices, so the assumption holds. The use of multiple counterfactual samples (K=5) reduces the variance of the estimate without introducing bias, as each sample is an estimate of the total interventional effect. We additionally record the commit-schedule divergence d_i to empirically test the locality assumption: if d_i is frequently 1, the intervention induces a suffix cascade, and c_i reflects the total effect including that cascade, not a local token effect.

**Assumption and failure mode (D sub-block)**: The equivalence we rely on is: shared-seed single-token resampling measures the total interventional effect E[R | do(z_i = a)] - E[R | do(z_i = b)], where the intervention includes re-denoising the suffix under the model's conditional. Assumption A: the denoising noise and the commitment policy are exogenous to the intervened token (no backdoor path from z_i to later commitments via the shared seed). Failure mode F: the model's commit policy conditions on the noise level, or resampling z_i changes which positions are eligible for later commitment (commit-schedule divergence d_i = 1), in which case c_i conflates the direct effect with the induced suffix cascade. We do not claim locality; we measure d_i and report the fraction of counterfactual replays where the commit schedule diverges from factual.

## Contribution

['A novel interventional credit assignment method (ITC) for diffusion-LM RL that defines token credit as a do()-style contrast between factual and counterfactual denoising trajectories with shared randomness, providing an unbiased estimator of causal token effect.', 'A practical algorithm that implements the interventional contrast via trajectory replay with shared seeds, including a variance-reduction scheme using multiple counterfactual samples and a baseline, making it feasible for training diffusion-LMs at scale.', 'An empirical demonstration on mathematical reasoning tasks that ITC yields more accurate token-level credit than observational proxies (confidence improvement, ELBO) and leads to improved RL fine-tuning performance.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | GSM8K + MATH reasoning splits | Multi-step reasoning exposes token-credit attribution |
| Primary metric | Pass@1 accuracy | Outcome reward directly reflects credit quality |
| Baseline | DTCA (confidence-improvement credit) | Observational proxy: tests causal-isolation claim |
| Baseline | GDPO / ELBO-based credit | Likelihood proxy: tests interventional-contrast claim |
| Baseline | Outcome-only REINFORCE | No token credit: tests value of any credit |
| Ablation-of-ours | ITC with suffix-resampling counterfactual | Tests locality of single-token do()-intervention |
| Diagnostic | Commit-schedule divergence rate d_i | Tests Assumption A (locality of denoising path) |

### Why this setup validates the claim

The central claim is that an interventional, single-token do()-contrast yields more faithful token-level advantages than observational proxies (confidence improvement, ELBO) or no credit at all, and that this faithfulness translates into better reasoning. GSM8K/MATH are the right testbed because reasoning correctness depends on a few pivotal committed tokens (operator choices, intermediate quantities) whose causal effect is large relative to correlated-but-inert tokens (easy-context commitments), so a credit method that isolates causal effect should shine exactly here. The three baselines each test a distinct sub-claim: outcome-only REINFORCE tests whether token credit matters at all; DTCA tests whether confidence improvement is a sufficient observational proxy; GDPO/ELBO tests whether likelihood-based credit suffices. The ablation (suffix resampling instead of single-token resampling) tests the locality assumption that justifies our design choice. Pass@1 is the correct metric because the reward function IS correctness, so any credit misattribution should surface as accuracy loss, not merely as a reward-model artifact. The commit-schedule divergence diagnostic directly measures whether Assumption A holds: if d_i is frequently 1, the single-token intervention induces a suffix cascade and the reported credit is a total effect, not a local one.

### Expected outcome and causal chain

**vs. Outcome-only REINFORCE** — On a multi-step problem where most committed tokens are filler and one token fixes a wrong intermediate value, outcome-only REINFORCE assigns the same scalar advantage to every token, so it reinforces the filler tokens as strongly as the pivotal one and the policy drifts toward verbose-but-wrong trajectories. Our method assigns near-zero credit to inert tokens and a large negative credit to the pivotal wrong token, concentrating the update on the causal decision. We expect a clear accuracy gap on multi-step subsets (MATH) with rough parity on single-step GSM8K items.

**vs. DTCA (confidence-improvement credit)** — On a case where the model is highly confident about a token simply because the context is easy, DTCA gives that token a large positive credit even though it has no causal effect on the final answer, so the policy is pushed to reproduce easy-context tokens and wastes capacity. Our method resamples that token under shared randomness and observes ~zero reward change, assigning it ~zero credit, so the update ignores it. We expect DTCA to over-credit high-confidence tokens and show a measurable deficit on problems requiring a low-confidence pivotal commitment, where our method retains an edge.

**vs. GDPO / ELBO-based credit** — On a case where a token has high likelihood under the current model but is causally irrelevant to correctness (e.g., a plausible but unused restatement), ELBO-based credit rewards it, inflating likelihood on non-causal tokens and yielding a policy that is fluent but wrong. Our interventional contrast gives it near-zero credit because resampling it does not move the reward. We expect GDPO to show higher training reward with smaller Pass@1 gains, i.e., a reward-accuracy decoupling that our method avoids.

### What would falsify this idea

If ITC's accuracy gain over DTCA and GDPO is uniform across easy-context and pivotal-token subsets rather than concentrated where the predicted credit-misattribution failure occurs, the causal-isolation story is wrong. Likewise, if the single-token ablation matches ITC exactly, the locality assumption underlying the do()-intervention is not doing the work we claim. Additionally, if the commit-schedule divergence rate d_i is high (e.g., >50% of counterfactual replays), then Assumption A fails and the reported credit is a total effect including suffix cascade, not a local token effect.

## References

1. DTCA: Denoising Token Credit Assignment for Reinforcement Learning in Diffusion Language Models
2. Advancing Reasoning in Diffusion Language Models with Denoising Process Rewards
3. Improving Reasoning for Diffusion Language Models via Group Diffusion Policy Optimization
4. Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations
5. Simple and Effective Masked Diffusion Language Models
6. Simplified and Generalized Masked Diffusion for Discrete Data
7. Likelihood-Based Diffusion Language Models
8. Bayesian Flow Networks
