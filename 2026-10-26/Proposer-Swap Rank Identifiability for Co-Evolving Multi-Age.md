# Proposer-Swap Rank Identifiability for Co-Evolving Multi-Agent Script Proposers and Evaluators

## Motivation

Across self-improving proposer-evaluator trees — Self-HPO's harness-proposal/selection loop, Promptbreeder's mutation-prompt/task-prompt co-evolution, and MovieAgent's multi-agent CoT planner — the fitness that selects a candidate is computed by the same generative process that emitted it, so the ranking is not identifiable as a function of candidate quality. Self-HPO concretely exhibits this: the policy that proposes harness candidates also supplies the failure analysis and shares parameters with the evaluator that scores them, so a harness can win selection by matching proposer-specific idiosyncrasies rather than by being genuinely better. This meta-gap recurs because no existing loop performs a controlled intervention on the proposer's own context before admitting a candidate, leaving proposer-induced bias and genuine quality statistically confounded inside every selection signal.

## Key Insight

If a candidate's rank is invariant to a matched-budget swap of the proposer's own context, then the rank is a function of candidate quality rather than of the emitting proposer, giving a per-candidate identifiability certificate that separates the two confounded sources without any external verifier.

## Method

**What it is.** PSRI (Proposer-Swap Rank Identifiability) is a selection gate inserted into a co-evolving multi-agent script-generation loop. Inputs: a population of script candidates, the proposer context $c_p$ that emitted each, and an evaluator $E$ that scores candidates on a held-out script-validation set. Output: an admitted population in which every retained candidate carries a rank-identifiability certificate, plus a constrained re-proposal distribution for rejected ones.

**How it works.**

```pseudocode
PSRI loop (one generation, budget B=8 evaluator calls per candidate):
1. PROPOSE: proposer P with context c_p emits candidate set C = {c_1..c_k}
   (k=6 per generation).
2. SCORE-BASE: evaluator E ranks C on validation set V -> rank r_base(c_i)
   using B=8 rollouts per candidate.
3. PROPOSER-SWAP INTERVENTION: construct a matched-budget alternative
   proposer context c_p' by (a) re-instantiating P with a different role
   assignment from the agent roster and (b) resampling the failure-analysis
   trace at equal token budget; c_p' emits a shadow candidate set C' of the
   same size k under the same budget B.
   [Assumption A: c_p' changes only proposer identity while holding task
   difficulty fixed; this is checked by the difficulty-drift control in step 5b.]
4. RANK-UNDER-SWAP: score C ∪ C' with E -> rank r_swap(c_i) restricted to
   candidates that survive both contexts' validity filter.
5. IDENTIFIABILITY TEST: for each c_i compute
       kappa(c_i) = 1[ |r_base(c_i) - r_swap(c_i)| <= tau ]   (tau=1)
   kappa=1 -> identifiable, admit; kappa=0 -> non-identifiable.
   5b. DIFFICULTY-DRIFT CONTROL: construct a control context c_p'' that
       resamples the failure-analysis trace at equal token budget but keeps
       the role assignment fixed (i.e., swaps only non-identity context).
       Score C ∪ C'' with E -> rank r_ctrl(c_i). Compute
       kappa_ctrl(c_i) = 1[ |r_base(c_i) - r_ctrl(c_i)| <= tau ].
       Report the difference in kappa distributions (kappa vs kappa_ctrl)
       and the Pearson correlation between kappa and an independent
       difficulty proxy (mean script length in tokens, computed per candidate).
6. CONSTRAINED RE-PROPOSAL: non-identifiable candidates are not selected;
   their proposal distribution is re-weighted by a penalty
       w(c) <- w(c) * exp(-lambda * |r_base - r_swap|),  lambda=0.5
   and re-proposed for up to R=2 rounds.
7. ADMIT: population <- identifiable candidates sorted by r_base.
```

**Why this design.** We chose a *proposer-swap intervention on the context* over a learned disentanglement head because the proposer and evaluator are operationally coupled through shared parameters in Self-HPO and Promptbreeder, and no learned head can manufacture independence the data does not contain; the intervention instead creates the independence by construction, accepting the cost of one extra proposer pass per generation. We chose *rank invariance* over absolute-score invariance because rank is what the selection mechanism actually consumes, so testing the quantity that drives selection avoids a proxy gap, accepting that rank is coarser and may mask small score shifts. We chose a *matched-budget* shadow proposer (same k, same B, same token budget) over an unconstrained alternative because budget mismatch would let the swap change candidate quality for reasons unrelated to proposer identity, accepting that matched budgeting requires careful token accounting. We chose *constrained re-proposal* over outright rejection because a non-identifiable candidate may still be salvageable under a different proposal distribution, accepting extra proposal rounds. A domain expert would not describe this as a variant of Self-HPO or Promptbreeder: those loops select on a single fitness computed inside the emitting process, whereas PSRI admits a candidate only after an intervention that no prior loop performs.

**Why it measures what we claim.** The quantity $\kappa(c_i)=1[|r_{base}-r_{swap}|\le\tau]$ measures *rank identifiability* because the assumption that the shadow proposer context $c_p'$ is a valid matched-budget intervention — i.e., it changes only proposer identity while holding candidate-quality-relevant resources fixed — makes rank stability equivalent to rank being a function of candidate quality; this assumption fails when the shadow context changes the *task difficulty* of the candidate (e.g., a different role assignment elicits an easier script), in which case $\kappa$ reflects difficulty drift rather than proposer bias. The penalty weight $\exp(-\lambda|r_{base}-r_{swap}|)$ measures *proposer-induced bias magnitude* because the assumption that rank displacement under swap is monotone in the bias the proposer injects makes the exponential a valid bias score; this assumption fails when the evaluator's own noise exceeds the bias signal, in which case the penalty reflects evaluator variance. The matched budget $B$ measures *evaluation reliability* because the assumption that $B$ rollouts suffice to stabilize rank makes $r_{base}$ and $r_{swap}$ comparable; this assumption fails on high-variance script tasks, in which case $\kappa$ reflects sampling noise.

## Contribution

(1) A proposer-swap rank-identifiability gate (PSRI) that inserts a matched-budget context intervention into co-evolving proposer-evaluator loops and admits only candidates whose rank is invariant under the swap. (2) A constrained re-proposal distribution that re-weights non-identifiable candidates by their rank displacement, converting the identifiability test into a usable selection signal rather than a binary filter. (3) A design principle — rank invariance under proposer intervention as the minimal structural condition separating proposer bias from genuine quality — demonstrated inside multi-agent script-generation loops.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Multi-agent script-generation benchmark with held-out script-validation split | Provides proposer/evaluator coupling and rank-based selection |
| Primary metric | Rank-identifiability admission rate (fraction of selected candidates with κ=1) | Directly measures the quantity the gate claims to certify |
| Secondary metric | Downstream script quality (human or automated score on validation set) | Tests whether admission translates to better scripts |
| Diagnostic metric | Pearson correlation between κ and independent difficulty proxy (mean script length in tokens) | Checks that κ is not merely tracking difficulty drift |
| Baseline | Self-HPO-style co-evolving loop (single fitness) | Tests whether intervention beats in-process selection |
| Baseline | Promptbreeder-style proposer swapping without matched budget | Tests whether matched budgeting drives the gain |
| Baseline | Unconstrained re-proposal (no rank-identifiability gate) | Tests whether the gate, not extra rounds, causes improvement |
| Ablation-of-ours | PSRI with τ→∞ (admit all) | Isolates the identifiability test's contribution |
| Ablation-of-ours | PSRI with difficulty-drift control (κ_ctrl) replacing κ | Tests whether the control intervention changes admission decisions |

### Why this setup validates the claim

The central claim is that admitting a candidate only after a matched-budget proposer-swap intervention yields a population whose selection is driven by candidate quality rather than proposer identity. The dataset must therefore expose co-evolving proposer/evaluator coupling with a held-out validation split, since that is where the proxy gap the method targets actually appears. The Self-HPO-style baseline tests the sub-claim that in-process single-fitness selection cannot separate proposer bias from quality; the Promptbreeder-style unmatched-budget baseline tests the sub-claim that matched budgeting, not merely swapping, is what makes the rank comparison valid; the unconstrained re-proposal baseline tests that the identifiability gate itself, rather than additional proposal rounds, produces the effect. The τ→∞ ablation is decisive: if admitting everything performs as well as PSRI, the gate is inert. The difficulty-drift control ablation directly tests the load-bearing assumption A: if κ and κ_ctrl produce similar admission decisions, then rank displacement is attributable to proposer identity rather than task-difficulty drift. Rank-identifiability admission rate is the right primary metric because it is the exact quantity κ certifies, so a gain in it is not a proxy for the mechanism — it is the mechanism's output.

### Expected outcome and causal chain

**vs. Self-HPO-style co-evolving loop** — On a case where the proposer and evaluator share parameters and a high-fitness candidate is emitted from a context that also made the task easier (e.g., a role assignment eliciting a simpler script), the baseline admits it on raw fitness because its selection signal is computed inside the emitting process and cannot distinguish quality from proposer-induced ease. Our method re-scores that candidate under a matched-budget shadow context; its rank shifts, κ=0, and it is not admitted. We expect a visible admission-rate gap concentrated on candidates whose base rank is high but swap rank is displaced, with parity on candidates whose rank is stable. The difficulty-drift control is expected to show that κ_ctrl remains high (stable rank) for these candidates, confirming that the displacement is due to proposer identity, not difficulty drift.

**vs. Promptbreeder-style unmatched-budget swap** — On a case where the shadow proposer receives a different token budget, the baseline's rank displacement reflects the budget change rather than proposer identity, so it either rejects good candidates or admits biased ones. Our matched-budget shadow holds k, B, and tokens fixed, so displacement is attributable to proposer identity. We expect the unmatched variant to show inflated κ=0 rates on candidates that are actually high-quality, i.e., a false-rejection pattern our method avoids, with the gap largest where token accounting diverges most. The difficulty-drift control will show that κ_ctrl is not inflated in the same way, isolating the budget effect.

**vs. Unconstrained re-proposal** — On a case where a non-identifiable candidate is re-proposed without the exponential penalty, the baseline's re-proposal distribution stays near the original biased context and resamples the same failure mode. Our method re-weights by exp(-λ|r_base - r_swap|), pushing mass away from the displaced region. We expect the unconstrained baseline to recover few additional identifiable candidates per round while ours recovers more, with the difference concentrated in the first re-proposal round and shrinking by round two. The difficulty-drift control will confirm that recovered candidates have low κ_ctrl displacement, indicating that the penalty targets proposer bias rather than difficulty.

### What would falsify this idea

If the τ→∞ ablation matches PSRI on rank-identifiability admission rate and downstream script quality, the gate is inert and the central claim fails. Likewise, if gains are uniform across subsets rather than concentrated where proposer-swap rank displacement is predicted to occur, or if κ and κ_ctrl produce indistinguishable admission decisions, the intervention is not the causal driver.

## References

1. Self-Evolving Coding Agents via Harness-Policy Optimization
2. Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution
3. Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models
4. Large Language Models Are Human-Level Prompt Engineers
5. Eliminating Meta Optimization Through Self-Referential Meta Learning
6. Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models
7. STaR: Bootstrapping Reasoning With Reasoning
8. Automated Movie Generation via Multi-Agent CoT Planning
