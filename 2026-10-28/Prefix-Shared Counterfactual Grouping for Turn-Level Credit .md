# Prefix-Shared Counterfactual Grouping for Turn-Level Credit in Multi-Turn LLM Agents

## Motivation

Existing multi-turn RL methods for LLM agents either learn an expensive critic (SWEET-RL) or collapse trajectories into independent single-turn episodes (single-turn RL reformulation), and neither provides a reliable per-turn credit signal: SWEET-RL's critic is biased by function approximation and requires extra training-time information, while single-turn decomposition destroys the shared-prefix structure that makes turn-level effects identifiable. Across all three branches (SWEET-RL, GiGPO-style anchor grouping, single-turn RL), the scalar outcome reward over the whole plan is assumed sufficient, so no mechanism localizes reward to the specific turn where a planning decision diverged. This meta-gap is structural: without exploiting prefix sharing, outcome differences cannot be attributed to a turn without a learned critic.

## Key Insight

Within a matched block of rollouts that share an identical prefix up to turn k, the prefix is a sufficient statistic for the state at the divergence point, so the within-block variation in final outcome is an unbiased estimate of turn k's marginal causal contribution—a property that a learned critic (biased by approximation) and single-turn decomposition (destroys the prefix) cannot deliver.

## Method

**What it is** — Prefix-Shared Counterfactual Grouping (PSCG) is a turn-level credit assignment mechanism for multi-turn LLM agent RL. Inputs: a set of N grouped rollouts (trajectories of turns) sharing a common initial task, with binary outcome rewards. Output: a per-turn advantage estimate for each turn in every rollout, used to form a policy-gradient update without a learned multi-turn critic.

**Load-bearing assumption (stated explicitly)** — Within a block of rollouts sharing an exact turn-sequence prefix up to turn k-1, the residual outcome variation is attributable to the diverging turn k's action. This requires (A) the exact-prefix match is a sufficient statistic for the environment state, and (B) no unobserved within-block confounders (stochastic environment outcomes, hidden state, or sampling temperature effects) correlate with the turn choice. When (B) fails — e.g., a random tool failure occurs independently of the action — the raw within-block difference `r_i - mean_block(r)` conflates noise with causal effect. PSCG therefore uses a variance-controlled, randomization-aware estimator (step 2b below) that assigns localized credit only when between-action outcome variance exceeds the estimated environment-noise floor; otherwise it falls back to the global baseline.

**How it works** —
```pseudocode
# Given: N rollouts, each a list of turns; binary outcome r in {0,1}
# Hyperparameter: min_block_size = 4, max_prefix_depth = 6
# Hyperparameter: min_repeats_per_action = 2 (for variance floor estimation)
# Hyperparameter: shrinkage_lambda = 0.5 (between/within variance shrinkage)

1. Build prefix trie over all rollouts (turn-level tokens/actions).
2. For each depth k = 1..max_prefix_depth:
     For each trie node at depth k with >= min_block_size descendant rollouts:
         block = rollouts passing through node
         # matched block: identical prefix up to turn k-1
         # 2a. Group rollouts in block by the diverging action a at turn k
         action_groups = {a: [r_i for rollouts in block with turn k = a]}
         # 2b. Variance-controlled estimator:
         #     between-action variance vs within-action (noise) variance
         for each action a with >= min_repeats_per_action samples:
             mean_a = mean(r_i for i in action_groups[a])
             var_within_a = var(r_i for i in action_groups[a])
         # pooled within-action variance (environment-noise floor)
         sigma2_noise = mean(var_within_a over actions with >= min_repeats_per_action)
         # between-action variance (signal)
         sigma2_signal = var(mean_a over actions with >= min_repeats_per_action)
         # shrinkage: credit only if signal exceeds noise floor
         if sigma2_signal > sigma2_noise:
             # assign per-action advantage with shrinkage toward block mean
             for each rollout i in block:
                 a_i_k = (r_i - mean_{j in block} r_j) * (sigma2_signal / (sigma2_signal + sigma2_noise))
                 assign a_i_k to turn k of rollout i
         else:
             # signal not above noise: fall back to global baseline for this block
             for each rollout i in block:
                 a_i_k = r_i - mean_all r
                 assign a_i_k to turn k of rollout i
3. For turns not covered (shallow blocks or blocks failing the variance test),
   fall back to global outcome baseline:
     a_i_k = r_i - mean_all r
4. Policy gradient: for each rollout i, weight each turn's token log-probs by a_i_k
     (sum over turns, or per-turn loss with a_i_k as coefficient).
```

**Why this design** — We chose a prefix trie over clustering by state embeddings because trie nodes exactly encode the shared-prefix condition (identical turn sequence), avoiding the false matches that embedding similarity would introduce; the cost is that only exactly shared prefixes form blocks. We chose within-block outcome differences over a learned critic because the prefix is a sufficient statistic for the divergence state, making the difference an unbiased estimate of marginal effect; the cost is higher variance when block size is small, mitigated by min_block_size=4. We chose a global fallback for shallow blocks over discarding them because early turns still need signal; the cost is that early-turn advantages are less localized. We chose binary outcome differences over reward shaping because the outcome is the only verifiable signal in multi-turn planning; the cost is that intermediate turns receive no credit unless they lie on a diverging prefix. Unlike SWEET-RL, which trains a critic with extra information, PSCG requires no critic model and no additional training-time information—its signal comes from the experimental design of grouped rollouts, not from function approximation. The variance-controlled estimator (step 2b) was chosen over the raw difference because it makes the causal claim conditional on demonstrated signal-above-noise rather than assumed; the cost is that blocks with fewer than min_repeats_per_action samples per action cannot estimate the noise floor and fall back to the global baseline.

**Why it measures what we claim** — The computational quantity `a_i_k = (r_i - mean_{j in block} r_j) * (sigma2_signal / (sigma2_signal + sigma2_noise))` measures the marginal causal contribution of turn k because the matched block holds the prefix fixed, so any outcome difference is attributable to the diverging turn under the assumption that the prefix is a sufficient statistic for the state and that no unobserved confounders vary within the block; this assumption fails when the environment is stochastic in a way correlated with the turn choice (e.g., a random tool failure), in which case `a_i_k` reflects noise rather than causal effect. The shrinkage factor `sigma2_signal / (sigma2_signal + sigma2_noise)` measures the reliability of the within-block estimate because it down-weights the raw difference when between-action variance does not exceed the estimated environment-noise floor; this fails when the noise floor is underestimated (e.g., too few repeats per action), in which case the estimator over-credits noisy turns. The prefix trie node measures 'shared-prefix identity' because exact turn-sequence equality is equivalent to identical state under the Markov assumption; this fails when the environment has hidden state not captured by the turn sequence, in which case the block is not truly matched. The min_block_size threshold measures 'reliability of the within-block estimate' because larger blocks reduce variance of the mean outcome; this fails when blocks are small, in which case the advantage is noisy. The global fallback measures 'outcome baseline' because it uses all rollouts; this fails when the task has high outcome variance, in which case early-turn advantages are weakly localized.

## Contribution

['A turn-level credit assignment mechanism (PSCG) that converts prefix-sharing structure in grouped rollouts into unbiased per-turn advantage estimates without a learned critic, using a prefix trie and within-block outcome differences.', "A design principle: exact prefix matching via trie yields identifiability of marginal turn effects under the Markov assumption, whereas embedding-based grouping introduces confounding; this principle is validated by the mechanism's construction.", 'An integration recipe for using PSCG advantages in policy-gradient training of multi-turn LLM agents, with a global fallback for shallow blocks and a min_block_size hyperparameter controlling the bias-variance trade-off.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | ColBench multi-turn tasks | Shared-prefix rollouts with verifiable outcomes |
| Primary metric | Success rate | Directly reflects outcome-reward optimization |
| Baseline | SWEET-RL (critic-based) | Tests claim: no critic needed |
| Baseline | StarPO (outcome-only RL) | Tests turn-level credit gap |
| Baseline | GPT4-o reference | Upper-bound sanity anchor |
| Ablation-of-ours | PSCG w/o prefix blocks | Isolates grouping contribution |
| Ablation-of-ours | PSCG w/o variance control (raw within-block difference) | Isolates the variance-controlled estimator's contribution |
| Diagnostic metric | Fraction of turns receiving localized (non-fallback) credit | Quantifies practical coverage of the matched-block estimator |
| Diagnostic metric | Within-block confounding test: correlation between turn choice and tool-failure indicator | Tests assumption (B) empirically |

### Why this setup validates the claim

The central claim is that exact-prefix grouping converts binary outcome differences into localized turn-level advantages without a learned critic. ColBench is the right testbed because its collaborative multi-turn structure produces many rollouts sharing long identical prefixes yet diverging at a small number of decision turns—exactly the condition under which the matched-block estimator should be sharp. SWEET-RL tests the no-critic claim: if PSCG matches a critic trained with privileged information, the critic is unnecessary. StarPO tests the credit-assignment claim: outcome-only RL spreads advantage uniformly, so any gain concentrated on diverging turns is attributable to our mechanism. The ablation removes prefix blocks and reverts to global baselines, so any gap between it and full PSCG isolates the value of matched grouping rather than of grouping-in-general or of the RL loop itself. The variance-control ablation (raw within-block difference) tests whether the shrinkage estimator is necessary to avoid confounding by environment noise: if the raw-difference variant underperforms full PSCG on tasks with high tool-failure rates, the variance control is doing its job. The diagnostic metrics quantify coverage (what fraction of turns get localized credit) and test assumption (B) directly by measuring whether turn choice correlates with tool-failure indicators within blocks. Success rate is the right primary metric because the method optimizes outcome reward directly; a turn-level metric would confound the estimator's quality with the reward's definition.

### Expected outcome and causal chain

**vs. SWEET-RL (critic-based)** — On a case where the collaborator's turn reveals a constraint late in the trajectory, SWEET-RL's critic must generalize from privileged training-time information to unseen divergence patterns and produces noisy step rewards on rare prefix branches because its function approximation smooths over them. Our method instead computes the advantage directly from the empirical outcome difference within the exact matching block, so we expect parity on common prefixes but a visible edge on rare-branch subsets where the critic's generalization error dominates.

**vs. StarPO (outcome-only RL)** — On a case where a rollout succeeds despite a poor early turn and a later rollout fails despite a good early turn, StarPO assigns the same outcome-derived weight to every turn because it has no mechanism to localize credit, so it reinforces the bad early turn and suppresses the good one. Our method instead zeroes out the advantage on matched non-diverging turns and assigns signal only where prefixes diverge, so we expect a larger success-rate improvement on tasks whose trajectories contain many shared early turns and few divergence points.

**vs. PSCG w/o prefix blocks (ablation)** — On a case where many rollouts share a long common prefix but a small subset diverges early, the ablated variant falls back to the global mean and assigns a near-uniform advantage to all turns because it lacks matched blocks, so it cannot distinguish the diverging turn from its shared predecessors. Our method instead isolates the divergence using the block mean, so we expect the full method to outperform the ablation specifically on tasks with long shared prefixes, with the gap shrinking as prefix diversity increases.

**vs. PSCG w/o variance control (raw within-block difference)** — On a case where tool failures occur randomly and independently of the turn choice, the raw-difference variant assigns large advantages to turns whose outcomes were determined by noise, because it lacks the shrinkage factor that down-weights blocks where between-action variance does not exceed the noise floor. Our method instead assigns credit only when signal exceeds noise, so we expect the full method to outperform the raw-difference variant specifically on tasks with high tool-failure rates, with the gap shrinking as tool reliability increases.

### What would falsify this idea

If PSCG's gain over StarPO is uniform across all tasks rather than concentrated on tasks with long shared prefixes and few divergence points, the prefix-grouping mechanism is not what produces the improvement. Likewise, if PSCG fails to beat the no-block ablation on exactly the long-shared-prefix subset, the matched-block estimator is not localizing credit as claimed. If PSCG fails to beat the raw-difference variant on high-tool-failure tasks, the variance control is not mitigating confounding as claimed.

## References

1. SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks
2. RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning
3. Free Process Rewards without Process Labels
4. Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning
5. Regressing the Relative Future: Efficient Policy Optimization for Multi-turn RLHF
6. Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models
7. Large Language Models Cannot Self-Correct Reasoning Yet
