# Closure-Validated Memory Updates for Continual Learning Agents

## Motivation

Existing memory-centric continual learning frameworks for AI agents, such as MCCL (Memory-Centric Continual Learning for AI Agents), assume that memory entries are independently valid: adding a new entry does not require re-validating or re-interpreting previously stored entries. This assumption fails structurally under distributional shift, because new experiences can alter the semantics of old entries (e.g., a tool's behavior changes, a fact becomes outdated, a policy becomes unsafe), yet the memory update operator remains closed only under the old distribution. Consequently, agents retain stale or misinterpreted knowledge that silently degrades retention, adaptation, and transfer.

## Key Insight

A memory update operator that is closed under distributional shift must treat each new experience as a potential semantic intervention on all existing entries, triggering re-validation and re-interpretation exactly when the new experience's likelihood ratio relative to the old distribution exceeds a threshold derived from the entries' evidential support.

## Method

**What it is** — We introduce CLOVER (CLOsure-VErified memory Update Rule), a memory update operator for continual learning agents. Its inputs are the current memory M (a set of entries, each with content, a context descriptor, and a support count), a new experience e, and a dual-loop controller state. Its output is an updated memory M' that is closed under the distributional shift induced by e: entries whose semantics are altered by e are re-validated or re-interpreted, and new entries are added only after passing a closure check.

**Load-bearing assumption (stated explicitly)** — CLOVER's trigger chain rests on the assumption that the context descriptor x (a 16-d learned embedding) is a sufficient statistic for an experience's semantics, so that kernel similarity in context space is monotone in semantic overlap and the KDE likelihood ratio on x detects the semantic shifts that actually invalidate stored entries. This assumption fails under concept drift, where the conditional p(r,o,a | x) changes while the marginal p(x) does not; in that case Lambda(e) collapses to context novelty and the closure operator does not fire on the shifts that matter. To remove this assumption from the trigger, we replace the context-only shift detector with a conditional-shift detector: we model p(r,o | x) using the agent's existing transition/reward model (a 2-layer MLP with hidden size 128, trained online on the last 512 experiences with a Gaussian likelihood head, learning rate 1e-3, minibatch 32) and compute Lambda(e) = p_old(r_e,o_e | x_e) / p_new(r_e,o_e | x_e), where p_new is p_old updated with e. The influence score I_i(e) is gated on conditional rather than marginal kernel similarity: we replace the context kernel with a conditional kernel k_cond(i,e) = exp( -||g(x_i) - g(x_e)||^2 / (2 h^2) ), where g(x) = [x, mu_old(x), sigma_old(x)] concatenates the context descriptor with the predicted mean and standard deviation of (r,o) under the old model, and h is the same bandwidth as before. This keeps the closure operator and the dual-loop controller intact while removing the sufficiency-of-context assumption that the entire trigger chain rests on.

**How it works** — The operator proceeds in four phases, integrated with the dual-loop controller (inner loop: per-experience memory update; outer loop: policy refinement using task feedback):

```pseudocode
Input: memory M = {(c_i, x_i, s_i)}  # content, context descriptor, support count
       new experience e = (o_e, a_e, r_e, x_e)  # observation, action, reward, context
       dual-loop state: inner-loop threshold tau_in, outer-loop feedback buffer F
Output: updated memory M'

# Phase 1: Shift detection via conditional likelihood ratio
1. Fit a conditional model p_old(r,o | x) from the last 512 experiences using a 2-layer MLP (hidden=128, GeLU, Gaussian likelihood head, Adam lr=1e-3, minibatch=32, 100 gradient steps per update).
2. Compute shift score for e:  Lambda(e) = p_old(r_e,o_e | x_e) / p_new(r_e,o_e | x_e), where p_new is p_old updated with e (one gradient step on e).
3. If Lambda(e) < tau_in (default 1.5), treat e as in-distribution: add entry (o_e, a_e, r_e, x_e, s=1) to M and return M.

# Phase 2: Closure trigger — identify entries whose semantics may be altered
4. For each existing entry i, compute a semantic influence score:
     I_i(e) = exp( -||g(x_i) - g(x_e)||^2 / (2 h^2) ) * (1 + s_i)^{-1/2}
   where g(x) = [x, mu_old(x), sigma_old(x)] is the conditional feature map and h is the KDE bandwidth (default 0.5).
5. Select the closure set C = { i : I_i(e) > tau_c } with tau_c = 0.3 (default).

# Phase 3: Re-validation and re-interpretation
6. For each i in C:
     a. Re-validate: check whether the stored action a_i is still consistent with e under the current conditional model. Consistency score:
          V_i = 1[ r_e >= r_i - delta ] * 1[ a_i is in the support of the updated policy pi_new ]
        with delta = 0.1 (reward tolerance).
     b. If V_i = 0, mark entry i as stale; re-interpret by replacing its content with the nearest consistent entry from e's trajectory (using the same conditional kernel similarity) and reset s_i = 1.
     c. If V_i = 1, increment s_i <- s_i + 1 (support consolidation).

# Phase 4: Closure check and dual-loop feedback
7. Verify closure: for every entry i in M, ensure that its context descriptor x_i is within the support of p_new (i.e., p_new(r,o | x_i) > epsilon for the stored (r_i,o_i), epsilon = 1e-3). If not, flag entry for outer-loop review.
8. Push the tuple (Lambda(e), |C|, number of stale entries) to the outer-loop feedback buffer F.
9. Outer loop: every K=10 inner-loop updates, refine tau_in and tau_c by minimizing a retention-adaptation loss on F (grid search over {0.5, 1.0, 1.5, 2.0} for tau_in and {0.1, 0.3, 0.5} for tau_c).
10. Return M'.
```

**Why this design** — We chose a conditional density model p(r,o | x) over a context-only KDE because the method's motivating shifts are changes in the observation-action-reward mapping, not in the context marginal; a conditional model provides a likelihood ratio that is sensitive to concept drift, accepting the cost that it requires online training (100 gradient steps per update, ~2 ms on CPU) and may be noisy early in training. We chose a support-weighted influence score I_i(e) over a uniform similarity threshold because entries with high support should be more resistant to re-validation (they have accumulated evidence), accepting the cost that new but correct entries may be re-validated too often. We chose a binary consistency check V_i over a continuous re-validation score because the dual-loop controller needs a discrete signal to decide whether to trigger re-interpretation, accepting the cost that borderline cases are forced into one category. We chose to integrate with the dual-loop controller via a feedback buffer rather than end-to-end training because the closure operator must remain interpretable and auditable, accepting the cost that the outer-loop refinement is slower. A domain expert would NOT describe CLOVER as a variant of MCCL: MCCL's memory update is a passive append/retrieve operation with no closure check, whereas CLOVER's core mechanism is the likelihood-ratio-triggered re-validation of existing entries, which MCCL does not perform.

**Why it measures what we claim** — The shift score Lambda(e) measures distributional shift because the assumption that the conditional model p(r,o | x) captures the experience distribution makes the likelihood ratio equivalent to the probability that e comes from a different conditional distribution; this assumption fails when the conditional model is misspecified (e.g., the true p(r,o | x) is multimodal and the Gaussian head cannot represent it), in which case Lambda(e) reflects model error rather than semantic shift. The influence score I_i(e) measures semantic influence because the assumption that conditional kernel similarity in the feature space g(x) is monotone in semantic overlap makes high similarity equivalent to potential semantic alteration; this assumption fails when two entries have similar conditional predictions but unrelated semantics (e.g., same tool, different task), in which case I_i(e) over-selects entries for re-validation. The consistency score V_i measures re-validation because the assumption that reward consistency and policy support are necessary conditions for an entry's continued validity makes V_i=0 equivalent to staleness; this assumption fails when the reward signal is noisy, in which case V_i reflects noise rather than staleness. The closure check measures closure because the assumption that the support of p_new covers all valid contexts makes out-of-support entries equivalent to entries that are no longer interpretable under the new distribution; this assumption fails when p_new is over-smoothed, in which case the check misses entries that should be flagged.

## Contribution

['A novel memory update operator, CLOVER, that enforces closure under distributional shift by triggering re-validation and re-interpretation of existing memory entries when new experiences alter their semantics, integrated with a dual-loop controller for scheduling closure-preserving updates.', 'A concrete design principle: closure-preserving memory updates improve retention under interference and adaptation to change compared to passive append-only memory, because they prevent stale entries from silently degrading agent behavior.', 'An evaluation protocol that measures closure violations (stale entries, out-of-support entries) alongside retention, adaptation, and transfer, enabling direct attribution of behavioral gains to closure-preserving updates.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Sequential tool-use agent tasks with mid-stream distribution shifts (synthetic tool-use simulator with 50 tasks, 10 tools, shift injected at episode 25 by changing tool reward functions) | Provides interference and genuine semantic change |
| Primary metric | Retention-adaptation tradeoff (post-shift accuracy vs. pre-shift retention) | Directly captures the two competing demands |
| Baseline | Passive retrieval-cache memory (MCCL-style) | Tests whether closure check beats passive append/retrieve |
| Baseline | Uniform-similarity re-validation (no support weighting) | Tests whether support weighting of influence matters |
| Baseline | Continuous-score re-validation (no binary consistency) | Tests whether discrete V_i drives the gain |
| Baseline | Retrieval-augmented memory with learned re-ranking (neural-memory baseline) | Tests against a strong neural-memory baseline |
| Ablation-of-ours | CLOVER with closure check disabled (append-only) | Isolates the closure mechanism from the rest |
| Calibration experiment | Ground-truth semantic-staleness labels from the simulator (binary labels for each entry after shift) | Measures precision/recall of Lambda, I_i, and V_i against labels |

### Why this setup validates the claim

The central claim is that closure-verified memory updates prevent stale entries from corrupting behavior under distributional shift while preserving retention of still-valid knowledge. The passive retrieval-cache baseline tests the sub-claim that closure is necessary at all: if it retains as well and adapts as well, the closure operator is doing nothing. The uniform-similarity baseline tests the sub-claim that support-weighted influence is what prevents over-re-validation of well-supported entries. The continuous-score baseline tests the sub-claim that the discrete consistency signal is what lets the dual-loop controller act decisively. The neural-memory baseline tests whether closure adds value beyond a strong learned re-ranker. The ablation (closure disabled) is the sharpest test: it removes exactly the mechanism we credit, so any gain that survives its removal is not attributable to closure. The retention-adaptation tradeoff metric is the right detector because the predicted effect is asymmetric — better adaptation at preserved retention — which a single accuracy number would mask and a pure adaptation metric would confound with forgetting. The calibration experiment directly tests the linking assumptions: it measures how often the failure modes (similar contexts, different rewards) occur and reports precision/recall of Lambda, I_i, and V_i against ground-truth staleness labels, so the proxy-to-concept gap is empirically quantified rather than hand-waved. Together these form a falsifiable test: the claim predicts a specific interaction, not uniform improvement.

### Expected outcome and causal chain

**vs. Passive retrieval-cache memory (MCCL-style)** — On a case where the environment shifts mid-stream so that a previously correct tool-selection entry becomes wrong, the baseline produces repeated stale actions because its memory is append-only and never re-validates existing entries against the new distribution. Our method instead flags that entry via the conditional likelihood-ratio trigger and binary consistency check, re-interpreting it before it corrupts the policy, because closure forces re-validation of semantically altered entries. We expect a noticeable post-shift adaptation gap concentrated on shifted-task subsets, with near-parity on stationary subsets.

**vs. Uniform-similarity re-validation (no support weighting)** — On a case where a well-supported, still-correct entry shares context with a new but anomalous experience, the baseline produces over-re-validation, churning correct high-support entries and eroding retention. Our method instead resists re-validating high-support entries because the inverse-square-root support factor in I_i(e) lowers their influence score, so consolidation protects accumulated evidence. We expect a retention gap on high-support subsets while adaptation stays comparable, i.e. the gain is concentrated where support weighting bites.

**vs. Continuous-score re-validation (no binary consistency)** — On a borderline case where reward noise makes consistency ambiguous, the baseline produces hesitant, threshold-sensitive updates because a continuous score gives the controller no clean trigger, delaying re-interpretation. Our method instead forces a discrete decision via V_i, so the dual-loop controller acts decisively and re-interprets stale entries promptly. We expect faster adaptation after shift (fewer stale entries lingering) with a modest retention cost concentrated on borderline cases, revealing the discrete signal's role.

**vs. Retrieval-augmented memory with learned re-ranking (neural-memory baseline)** — On a case where the shift changes the conditional p(r,o | x) but not the context marginal, the neural baseline may retrieve stale entries because its re-ranker is trained on pre-shift data and does not re-validate under the new conditional. Our method instead triggers closure via the conditional likelihood ratio and re-validates entries, so it should adapt faster on shifted subsets while maintaining retention on stationary subsets. We expect a smaller but consistent gap than against the passive baseline, concentrated on shifted subsets.

### What would falsify this idea

If the gain over the passive baseline is uniform across shifted and stationary subsets rather than concentrated on the shifted subsets where stale-entry corruption is predicted, the closure mechanism is not the cause. If disabling closure leaves retention and adaptation unchanged, the central claim that closure verification matters is wrong. If the calibration experiment shows that Lambda, I_i, and V_i have low precision/recall against ground-truth staleness labels, the linking assumptions are invalid and the proxies do not measure what they claim.

## References

1. Memory-Centric Continual Learning for AI Agents
