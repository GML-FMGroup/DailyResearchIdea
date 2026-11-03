# Value-Calibrated History Triggers for Coding-Agent Trajectory Collection

## Motivation

History-guided collectors such as SWE-Trail flag a state for extra collection effort using a trigger score derived from the history encoder, but that trigger score and the MCTS backed-up value that actually estimates a state's downstream utility are computed by independent heads and never forced to agree. Consequently the collector can spend its rollout budget on states that look historically promising yet have low search value, while high-value states are skipped — a persistent gap that survives because both ancestors treat the trigger score as a standalone sufficient ranking criterion rather than a calibrated estimate of search value.

## Key Insight

If the history encoder is trained so that its scalar trigger score is a monotone transform of the MCTS backed-up value on the states the collector actually visits, then the trigger's ranking is forced into agreement with search value on the collection distribution, making 'high trigger score' and 'high backed-up value' the same set rather than two independent rankings.

## Method

**What it is.** VCHT (Value-Calibrated History Trigger) is a training procedure for the history encoder inside a SWE-Trail-style collector. Inputs: per-issue history of prior trajectories, the current rollout's state, and the MCTS tree built at flagged states. Outputs: a scalar trigger score `s(h)` whose ranking on visited states is consistent with the backed-up value `V(s)` returned by MCTS, plus a collection policy that flags states by `s(h)`.

**Load-bearing assumption (stated explicitly).** The method assumes that the MCTS backed-up value `V(s)` computed with a fixed visit-count threshold (see below) is an *ordinal sufficient statistic* for a state's downstream collection utility: that is, if `V(s1) > V(s2)` with disjoint confidence intervals, then collecting extra trajectories at `s1` contributes more to the post-trained agent's resolve rate than collecting at `s2`. This assumption is what licenses training `L_rank` on `V_hat` and expecting the resulting trigger ranking to transfer to collection-worthiness on the full visited-state distribution. It fails when the backup is dominated by visit-count allocation rather than value, or when the calibration set `C` is not representative of the visited-state distribution.

**How it works.**

```pseudocode
# One collection iteration for issue i
H = history_buffer[i]                      # prior trajectories, fixed length K=8
rollout = agent.rollout(issue_i)           # agent sees NO history (test-time condition)
for each timestep t in rollout:
    h_t = HistoryEncoder(H, state_t)       # 128-d embedding (2-layer MLP, hidden=256, GeLU)
    s_t = TriggerHead(h_t)                 # scalar trigger score, sigmoid output (linear layer 128->1)
    if s_t > tau:                          # tau = 0.5, flag for collection
        # Variance-aware MCTS target: run to fixed visit-count threshold per flagged state
        tree = MCTS(state_t, visit_threshold=256, c=1.4)   # 256 visits, not 64 simulations
        V_t  = tree.root.backup_value      # visit-averaged backed-up value at root
        # Confidence interval from visit-count variance across root children
        ci_t = tree.root.backup_ci(alpha=0.05)   # 95% CI half-width
        store (h_t, s_t, V_t, ci_t) in calibration_set C
    continue rollout from state_t without history in context

# Decouple calibration set from trigger: sample uniform-random states for C
# (in addition to flagged states) to avoid self-selection bias
for each rollout, also sample 10% of timesteps uniformly at random (regardless of s_t):
    h_t = HistoryEncoder(H, state_t)
    s_t = TriggerHead(h_t)
    tree = MCTS(state_t, visit_threshold=256, c=1.4)
    V_t  = tree.root.backup_value
    ci_t = tree.root.backup_ci(alpha=0.05)
    store (h_t, s_t, V_t, ci_t) in C

# Value-consistency regularizer (trained jointly with trigger BCE loss)
for (h, s, V, ci) in C:
    V_hat = ValueHead(h)                   # scalar value predictor, linear on h (128->1)
    L_value = (V_hat - V)^2                # fit value head to MCTS backup
    # monotone consistency: rank(s) must match rank(V_hat) on C,
    # but only on pairs whose backup confidence intervals are disjoint
    L_rank  = pairwise_hinge(s, V_hat, margin=0.1, sample_pairs=256,
                             mask=disjoint_ci(ci_i, ci_j))   # only ordinal-reliable pairs
    # anchor: trigger score must be a monotone transform of predicted value
    L_mono  = (s - sigmoid(alpha * V_hat + beta))^2   # alpha,beta learned (scalar params)
    L_cons  = L_value + lambda_rank * L_rank + lambda_mono * L_mono

update HistoryEncoder, TriggerHead, ValueHead, alpha, beta by L_trigger + L_cons
# lambda_rank = 0.5, lambda_mono = 0.5; L_trigger = BCE(s, 1[s > tau]) with soft labels from V_t
# L_trigger uses soft labels: target = sigmoid((V_t - median(V_C)) / std(V_C)) to avoid hard threshold
```

At inference the collector uses only `s(h)`; the value head and monotone map are training-time scaffolding discarded after training.

**Why this design.** We chose a shared history encoder plus a separate linear value head over a single end-to-end scalar because the trigger head must remain cheap at collection time while the value head needs to fit MCTS backups that are noisy on a small `C`; decoupling them lets the value head absorb backup variance without destabilizing the trigger, accepting the cost that the two heads can drift if `C` is tiny. We chose pairwise ranking loss over direct regression of `s` onto `V` because MCTS backups are only ordinal-reliable across states (visit counts differ), so forcing cardinal agreement would inject scale noise into the trigger; the trade-off is that the trigger is calibrated only up to a monotone map, not in absolute units. We chose a learned sigmoid monotone map `sigmoid(alpha V_hat + beta)` over a fixed identity link because the trigger head's output range is bounded and the backup value's scale varies per issue; the cost is two extra parameters that must be fit jointly. This is not a variant of VinePPO or process-reward-model step labeling: those assign per-step rewards from Monte-Carlo returns and never couple a history-conditioned trigger to tree backups; here the coupling itself is the trained object, and the trigger is the only signal used at collection time.

**Why it measures what we claim.** The pairwise hinge `L_rank` measures *value-consistency* between the trigger score and search value because it assumes that on the calibration set `C` the MCTS backed-up value (computed with a 256-visit threshold and filtered to pairs with disjoint confidence intervals) is an ordinal sufficient statistic for a state's downstream collection utility; this assumption fails when the MCTS budget is too small relative to branching factor or when the confidence intervals are not disjoint, in which case `V_hat` reflects visit-count noise rather than utility. The monotone anchor `L_mono` measures *calibration of the trigger as a proxy for search value* because it assumes the relationship between the two is a single monotone sigmoid; this fails when the true relation is non-monotone (e.g., very high-value states are rare and under-sampled), in which case the anchor pulls `s` toward the mean. The value-head regression `L_value` measures *faithfulness of the value head to tree search* because it assumes the 256-visit backup is an unbiased estimate of state value; this fails on high-variance states, in which case the head learns the average backup rather than the true value. The trigger threshold `tau` measures *collection-worthiness* because it assumes flagged states are exactly those where extra rollouts pay off; this fails when the history encoder cannot distinguish states that look similar under `H`, in which case `s` reflects history similarity rather than value. The uniform-random sampling of 10% of timesteps for `C` measures *representativeness of the calibration set* because it assumes that including unflagged states prevents `C` from over-sampling trigger-selected mid-trajectory states; this fails when the random sample is too small to cover the state distribution, in which case `L_rank` is fit on a biased subset.

## Contribution

['A value-consistency regularizer that jointly trains a history-conditioned trigger head and a linear value head against MCTS backed-up values via a pairwise ranking loss plus a learned monotone anchor, making the trigger score a calibrated proxy for search value on the collection distribution.', 'A design principle for history-guided collectors: the trigger signal used at collection time should be trained to agree with the search signal used to estimate state utility, rather than being an independent ranking head, which removes the budget-waste failure mode of SWE-Trail-style collectors.', "A training-time-only value head and monotone map that are discarded at inference, so the collector's test-time cost and the agent's observation remain unchanged."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Stratified subset of 180 SWE-bench issues with multi-attempt histories (≥3 prior attempts) and high branching (≥5 candidate edits per state) | Provides per-issue history and hard states needing search; pilot on 30 issues first |
| Primary metric | Downstream SWE-bench resolve rate of post-trained agent | Directly measures whether collected trajectories train a better agent |
| Baseline | Independent multi-trajectory sampling + curation | Tests whether history-conditioned triggering beats blind sampling |
| Baseline | Random-state triggering (same budget) | Tests whether value-consistency beats flagging arbitrary states |
| Baseline | Single-head end-to-end trigger regressed onto MCTS value | Tests whether decoupled value head + ranking matters |
| Baseline | PUCT-style prior trained from MCTS visit distributions (same budget) | Tests whether VCHT's decoupled ranking+anchor is more than a reparameterization |
| Baseline | VinePPO-style value-relabeled collector (same budget) | Tests against an existing search-value-coupled collector |
| Ablation-of-ours | VCHT without L_rank/L_mono (trigger BCE only) | Tests whether the value-consistency coupling is doing the work |
| Ablation-of-ours | VCHT with 64-simulation backups (original budget) vs. 256-visit threshold | Tests whether variance-aware target is necessary |
| Ablation-of-ours | VCHT without uniform-random 10% calibration sampling | Tests whether decoupling C from trigger matters |
| Diagnostic metric | Rank correlation (Spearman) between 64-simulation and 512-simulation backups on 50 held-out states | Grounds the ordinal sufficiency assumption empirically |
| Diagnostic metric | Precision at matched flag count (fraction of flagged states with top-quartile 512-simulation backup) | Measures flag quality directly |
| Diagnostic metric | KL divergence between state distribution in C and full rollout distribution | Checks representativeness of calibration set |

### Why this setup validates the claim

The central claim is that coupling a history-conditioned trigger to MCTS backed-up value, via ranking plus a monotone anchor, makes flagged states more collection-worthy than what blind or arbitrary collection yields, and that this advantage shows up as a better post-trained coding agent. The dataset matters because the claim is specifically about states whose utility depends on prior trajectories: SWE-bench issues with multi-attempt histories give the encoder something to condition on and give MCTS non-trivial branching. The primary metric, downstream resolve rate, is the right detector because the trigger is only a means; if flagged states are genuinely more useful, the agent trained on them should resolve more issues, and if the trigger is merely correlated with value without improving collection, resolve rate will not move. Independent sampling tests the history-conditioning sub-claim, random triggering tests the value-consistency sub-claim, and the single-head baseline tests the decoupling sub-claim. The PUCT-style prior and VinePPO-style collector baselines test whether VCHT's specific coupling mechanism (decoupled ranking+anchor) is mechanistically distinct from existing search-value-coupled collectors, not just a reparameterization. The no-consistency ablation isolates whether the ranking and monotone terms, rather than the trigger BCE alone, produce the gain; if the ablation matches full VCHT, the coupling is decorative and the claim is undermined. The 64-simulation vs. 256-visit ablation and the Spearman diagnostic directly test the load-bearing assumption that the backup is an ordinal sufficient statistic: if 64-simulation backups have low rank correlation with 512-simulation backups, the variance-aware target is necessary. The uniform-random calibration sampling ablation and the KL diagnostic test whether decoupling C from the trigger prevents self-selection bias.

### Expected outcome and causal chain

**vs. Independent multi-trajectory sampling + curation** — On an issue where early attempts fail for a subtle reason (e.g., a wrong assumption about an API's return type) and later attempts only succeed if the agent avoids repeating that assumption, independent sampling produces trajectories that cluster around the same failure because no attempt sees prior history. Our method instead flags the state where the history-conditioned encoder predicts high downstream value, triggering MCTS there and collecting a trajectory that branches away from the repeated mistake. We expect a noticeable resolve-rate gap concentrated on issues with multi-attempt histories and near parity on single-attempt-easy issues.

**vs. Random-state triggering (same budget)** — On an issue with a long, mostly routine edit sequence, random triggering spends its MCTS budget on states where the value is flat and the branching factor is uninformative, so collected trajectories are near-duplicates of the rollout. Our method instead concentrates flags where the trigger score ranks states by predicted backed-up value, so the budget lands on the few states where extra search changes the outcome. We expect a moderate but consistent gap that widens as issue length grows, since random triggering's waste scales with trajectory length while ours does not.

**vs. Single-head end-to-end trigger regressed onto MCTS value** — On an issue where MCTS backups are noisy because visit counts are uneven across sibling states, the single head absorbs that scale noise directly into its trigger output, so flags become jittery and sometimes fire on high-variance-but-low-value states. Our method instead lets the separate value head absorb backup variance while the trigger is trained only through ranking and a monotone anchor, so flags stay stable under the same noise. We expect our method to show fewer wasted flags (higher precision at matched flag count) and a downstream resolve-rate edge that is largest on high-branching issues.

**vs. PUCT-style prior trained from MCTS visit distributions (same budget)** — On an issue where the visit distribution is diffuse because many candidate edits look similar under the policy prior, the PUCT prior assigns non-negligible probability to many low-value states, so collection effort is spread thinly. Our method instead uses a scalar trigger trained on ordinal value consistency, which concentrates flags on the top-ranked states. We expect VCHT to show higher precision at matched flag count on high-branching issues, and a resolve-rate edge that is largest where the visit distribution is most diffuse.

**vs. VinePPO-style value-relabeled collector (same budget)** — On an issue where Monte-Carlo returns are high-variance because trajectory length varies, VinePPO's per-step value labels are noisy, so its collection policy flags states with high return variance rather than high expected value. Our method instead uses MCTS backed-up value with confidence-interval filtering, which is less sensitive to trajectory length. We expect VCHT to show a resolve-rate edge on issues with long trajectories and high return variance.

**vs. VCHT without L_rank/L_mono (trigger BCE only)** — On an issue where the trigger BCE alone can fit the flag labels without learning a value-consistent ranking, the no-consistency ablation flags states that are historically similar to past successes but have low search value. Our method instead forces the trigger ranking to agree with the value head on ordinal-reliable pairs, so flags align with search value. We expect the full VCHT to outperform the ablation by a margin that is largest on issues where history similarity and search value diverge (e.g., issues with misleading prior attempts).

**vs. VCHT with 64-simulation backups** — On an issue where 64-simulation backups are dominated by visit-count noise, the ranking loss is trained on unreliable pairs, so the trigger ranking is noisy. Our method instead uses a 256-visit threshold and filters to disjoint confidence intervals, so the ranking loss is defined on ordinal-reliable pairs. We expect the 256-visit variant to show higher precision at matched flag count and a resolve-rate edge that is largest on high-branching issues where 64-simulation backups are least reliable.

**vs. VCHT without uniform-random 10% calibration sampling** — On an issue where the trigger-selected calibration set over-samples mid-trajectory states, the ranking loss is fit on a biased subset, so the trigger ranking does not transfer to unflagged states. Our method instead includes a uniform-random 10% sample of timesteps in C, which reduces the KL divergence between C and the rollout distribution. We expect the full VCHT to show a resolve-rate edge that is largest on issues where the trigger-selected C has high KL divergence from the rollout distribution.

### What would falsify this idea

If the full VCHT matches the no-consistency ablation (trigger BCE only) on downstream resolve rate, or if its gain over random triggering is uniform across issue subsets rather than concentrated where history-dependence and branching are high, then the value-consistency coupling is not doing the causal work and the central claim is wrong. Additionally, if the Spearman rank correlation between 64-simulation and 512-simulation backups is below 0.5 on the diagnostic set, the load-bearing ordinal sufficiency assumption is empirically false and the method's calibration target is unreliable.

## References

1. SWE-Trail: History-Guided Trajectory Collection for Coding Agent Training
