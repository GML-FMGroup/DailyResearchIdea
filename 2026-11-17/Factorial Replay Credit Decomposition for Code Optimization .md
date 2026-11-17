# Factorial Replay Credit Decomposition for Code Optimization Agents

## Motivation

Existing optimization-agent evaluations, such as Accelerate Bench, collapse whole-program speedup into a single scalar Score@1 and, in controlled replays, show that identical source code varies between 0.29x and 2.12x due only to build and runtime settings; the benchmark therefore cannot attribute performance to the specific edits an agent made. This structural failure recurs across agent-search work like Meta Agent Search, whose archive fitness is likewise a scalar that cannot credit individual agent-authored changes, so neither benchmarks nor search loops can provide the per-edit reward signal required for learned optimization policies.

## Key Insight

A two-way factorial replay of each edit against a fixed grid of build/runtime configurations yields an edit main effect that is identifiable and additive under a stated no-interaction assumption, separating edit-attributable speedup from environment-induced variance that scalar scores conflate.

## Method

**What it is.** FACTOR-CREDIT takes an agent's final program, its ordered sequence of accepted edits, and the task's build/runtime configuration space, and outputs a per-edit credit vector c_i (estimated marginal log-speedup of edit i), a per-configuration effect vector, and an interaction residual, alongside the conventional Score@1.

**Load-bearing assumption (stated explicitly).** The credit signal c[i] is valid only if edit effects are separable from configuration effects (no edit-by-configuration interaction) and the 4x3 grid G is representative of the verifier's actual build/runtime setting. Accelerate Bench's replays show that a single code change can be worth 1.01x or 6.4x depending on settings, so this assumption is not guaranteed; we therefore (a) include the verifier's exact build/runtime setting as one cell of G, (b) report the interaction residual r and the per-edit missingness rate, and (c) validate c[i] against leave-one-edit-out retiming at the verifier's configuration (see step 8). When the assumption fails, c[i] reflects the grid-averaged effect rather than the edit's value at the scored setting, and the residual flags this.

**How it works.**
```pseudocode
Input: base program P0, edit sequence e_1..e_m, config grid G = B x R (|B|=4, |R|=3),
       correctness oracle, timing protocol (median of 5 runs, warmup 2),
       verifier configuration g* (included in G)
Output: credit vector c[1..m], config effects g[1..|G|], interaction residual r,
        missingness rate per edit, leave-one-edit-out validation correlation, Score@1

1. Reconstruct prefix programs P_1..P_m by applying e_1..e_m in order to P0.
2. For each prefix P_i and each config g in G:
     if not correct(P_i, g): mark missing; else t[i,g] = median_time(P_i, g)
3. Fit additive model on log-speedup y[i,g] = log(t[0,g] / t[i,g]):
     y[i,g] = mu + a_i + b_g + r[i,g]
   with a_0 = 0, b_g centered; solve by least squares on observed cells.
4. c[i] = a_i - a_{i-1}   (marginal log-speedup of edit i)
5. g[g] = b_g; r = residual norm; report missingness rate per edit.
6. Score@1 computed by the standard verifier on the final program only.
7. Emit (c, g, r, missingness, Score@1) to the meta-agent archive.
8. Validation: for each edit i, compute leave-one-edit-out retiming at g*:
     t_loo[i] = median_time(P_m without e_i, g*)
     c_loo[i] = log(t_loo[i] / t[m,g*])
   Report Pearson correlation between c[i] and c_loo[i] across edits and tasks.
```

**Why this design.** We chose a two-way additive factorial model over a one-factor-at-a-time sweep because Accelerate Bench's own replays already show that build and runtime settings produce a 0.29x–2.12x swing on identical code, so a one-factor sweep would confound edit effects with the configuration at which they were tested; the cost is that we must assume no edit-by-configuration interaction, which we explicitly report as the residual r rather than hide. We chose prefix reconstruction over independent single-edit trials because optimization edits are order-dependent and later edits may rely on earlier ones, accepting that credit is defined along the agent's actual trajectory rather than as a counterfactual over all edit subsets. We chose to feed the credit vector into the existing Meta Agent Search archive as a multi-objective fitness (edit main effect plus correctness) rather than replacing its scalar fitness, so the search loop remains comparable to prior work; the cost is that the archive now stores vectors and requires a Pareto or weighted aggregation rule that we fix to a weighted sum with weight 1.0 on log-speedup and 10.0 on correctness. This is not a variant of Meta Agent Search or of Accelerate Bench's scoring: those methods produce a single scalar per program, whereas FACTOR-CREDIT produces an identified per-edit decomposition whose additivity is checked by the reported residual, a quantity neither prior method computes.

**Why it measures what we claim.** The quantity c[i] = a_i - a_{i-1} measures the marginal speedup attributable to edit i because the additive factorial model assumes edit and configuration effects are separable, making the edit main effect a consistent estimate of the edit's contribution averaged over the configuration grid; this assumption fails when an edit only helps under a specific build flag, in which case c[i] reflects the grid-averaged effect rather than the edit's best-case or worst-case value. The configuration effect b_g measures environment-induced variance because it is estimated after marginalizing over edits, assuming the grid is representative of the deployment settings; this fails when the grid excludes the setting the verifier actually uses, in which case b_g reflects grid-internal variance only. The residual r measures unmodeled edit-configuration interaction because it is the least-squares leftover after both main effects are removed; this fails when missing cells force an unbalanced fit, in which case r also absorbs missingness and should be read alongside the reported missingness rate. The validation correlation between c[i] and c_loo[i] measures whether prefix-order marginal credit agrees with leave-one-edit-out counterfactual credit at the verifier's configuration; this agreement is assumed to hold, and it fails when order-dependent edits make prefix credit diverge from counterfactual credit, in which case the correlation is low and c[i] should be interpreted as trajectory-conditional rather than counterfactual.

## Contribution

['A factorial-replay credit decomposition procedure that converts a whole-program optimization trajectory into an identified per-edit credit vector, a configuration-effect vector, and an interaction residual, using the controlled-replay infrastructure of Accelerate Bench.', 'A design principle for optimization-agent search: scalar archive fitness should be replaced by a multi-objective fitness that includes the per-edit credit vector, because scalar fitness cannot distinguish an edit that genuinely sped up the program from one that merely landed on a favorable build/runtime configuration.', 'An open decomposition report format (credit vector, config effects, interaction residual, missingness) that can be attached to any optimization-agent submission for audit and reuse in meta-agent search.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Accelerate Bench v1.1 (32 CPU tasks) | Application-level tasks with config-sensitive replays |
| Primary metric | Score@1 with per-edit credit fidelity | Score@1 alone cannot test decomposition claim |
| Baseline | Scalar-fitness Meta Agent Search | Tests whether credit vector beats scalar archive |
| Baseline | Accelerate Bench top agent (harnessed model) | Tests credit decomposition against best raw optimizer |
| Baseline | Single-config OFAT credit attribution | Tests whether factorial design beats one-factor sweep |
| Ablation-of-ours | FACTOR-CREDIT with residual discarded | Tests whether interaction residual carries search signal |
| Ablation-of-ours | FACTOR-CREDIT with verifier config excluded from G | Tests grid-representativeness assumption |

### Why this setup validates the claim

The central claim is that FACTOR-CREDIT produces an identified per-edit decomposition that improves search over scalar-fitness archives, and that its additivity assumption is empirically checkable via the residual. The scalar-fitness Meta Agent Search baseline isolates the sub-claim that the credit vector itself adds value: if archive fitness is scalar, the search cannot distinguish edits that help broadly from edits that help only under one build flag, so any gain we see over it must come from the decomposition. The Accelerate Bench top-agent baseline isolates the sub-claim that decomposition beats raw optimization capability, since that agent already achieves the best Score@1 without attribution. The OFAT baseline isolates the factorial-design sub-claim by holding the credit machinery constant but removing the configuration grid, so a gap there is attributable specifically to confounding. The residual-discarded ablation tests whether the interaction term is informative or noise: if dropping it changes nothing, the residual is decorative and the additivity assumption is doing no work. The verifier-config-excluded ablation directly tests the grid-representativeness assumption: if excluding the verifier's exact build/runtime setting from G changes the credit vector or Score@1, then the grid-averaged main effect is not a valid credit signal for the scored setting. Score@1 is the right primary metric because it is the benchmark's own verifier-gated measure, so any improvement is not an artifact of our scoring; we pair it with credit fidelity (agreement between c[i] and leave-one-edit-out retiming at the verifier's configuration) because Score@1 alone cannot falsify the decomposition claim.

### Expected outcome and causal chain

**vs. Scalar-fitness Meta Agent Search** — On a task where one edit helps only under a specific build flag (the Accelerate Bench replay shows identical code swinging 0.29x–2.12x), the baseline produces a scalar fitness that averages the edit's benefit across whatever config the harness happened to run, so the archive cannot tell a broadly useful edit from a flag-specific one and tends to retain the latter. Our method instead estimates the edit main effect after marginalizing over the 4x3 grid and reports the interaction residual, so flag-specific edits show a large residual and a modest main effect, and the weighted archive deprioritizes them. We expect a noticeable Score@1 gap concentrated on tasks with high config sensitivity, with parity on config-insensitive tasks.

**vs. Accelerate Bench top agent** — On a multi-edit task where a late edit depends on an earlier one (order-dependent optimization), the baseline produces a final program whose speedup is real but whose per-edit contribution is opaque, so iterative development cannot tell which of its accepted edits to keep or revert and it stalls after a few rounds. Our method instead reconstructs prefixes P_1..P_m and reports c[i] = a_i - a_{i-1}, giving the harness a ranked list of which edits actually paid off along its own trajectory. We expect the top agent to match us on final Score@1 for single-edit tasks but to fall behind on tasks requiring four or more accepted edits, where credit guidance compounds.

**vs. Single-config OFAT credit attribution** — On a task where the build flag interacts with an edit, the baseline produces credit estimates that attribute the flag's 2x swing to whichever edit was tested under it, because it varies one factor at a time and confounds edit with configuration. Our method instead fits the two-way additive model jointly over the grid, so the configuration main effect absorbs the flag swing and c[i] reflects the edit's grid-averaged contribution. We expect our credit vector to agree with leave-one-edit-out retiming substantially more often than OFAT's, with the gap largest on tasks whose replay variance is high.

**vs. FACTOR-CREDIT with residual discarded** — On a task where an edit's benefit is configuration-dependent, discarding the residual removes the only signal that flags the edit as interaction-dominated, so the archive treats its grid-averaged main effect as reliable. We expect the residual-discarded variant to show lower credit fidelity (correlation with leave-one-edit-out retiming) and slightly lower Score@1 on high-interaction tasks, with no difference on low-interaction tasks.

**vs. FACTOR-CREDIT with verifier config excluded from G** — On a task where the verifier's build/runtime setting is an outlier in the grid, excluding it from G means the grid-averaged main effect is estimated over settings that do not match the scored one. We expect the excluded-grid variant to show lower credit fidelity and a larger discrepancy between c[i] and c_loo[i], confirming that grid representativeness is load-bearing.

### What would falsify this idea

If our Score@1 gain over scalar-fitness Meta Agent Search is uniform across config-sensitive and config-insensitive tasks rather than concentrated where the 0.29x–2.12x replay swing predicts, then the decomposition is not what drives the improvement and the central claim is wrong. Likewise, if discarding the interaction residual leaves credit fidelity and Score@1 unchanged, the additivity assumption is untested by our own diagnostic and the residual should not be reported as evidence.

## References

1. Accelerate Bench: Evaluating Agents for Application-Level Program Optimization
