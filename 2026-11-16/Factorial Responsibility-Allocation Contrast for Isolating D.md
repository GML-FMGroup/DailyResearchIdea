# Factorial Responsibility-Allocation Contrast for Isolating Diffusion of Responsibility in Multi-Agent LLM Teams

## Motivation

The frontier claim that diffusion of responsibility explains the single-vs-team safety gap (Aligning AI Teams, arxiv:2611.17820) rests on correlational evidence: teams show fewer proactive system-level safety checks and more rationalization events, but responsibility allocation is never manipulated independently of monitoring frequency or delegation depth, so the safety-lead intervention's success cannot be distinguished from generic added scrutiny. This is a structural identifiability gap that recurs across multi-agent alignment work — MACPO (venue:iclr:2025:448acfcc736c) and ICR (venue:neurips:2025:69336207fda6) likewise vary training or partner-awareness signals without a controlled contrast that separates responsibility attribution from monitoring intensity. Without a factorial manipulation that orthogonalizes these factors, any observed gap remains confounded with coordination-induced specification drift and reduced per-agent monitoring under delegation.

## Key Insight

Responsibility allocation is identifiable from monitoring intensity only when the two are manipulated as orthogonal factorial factors and the outcome is measured on a per-agent proactive-check ledger that is invariant to how many checks the team performs in total, making the diffusion-of-responsibility effect a within-cell contrast rather than a between-condition correlation.

## Method

**What it is.** We introduce RACE (Responsibility-Allocation Controlled Experiment), a factorial evaluation protocol that takes as input a multi-agent LLM team configuration (task, team size, model) and outputs a per-agent responsibility-attribution vector together with a safety-gap decomposition into three orthogonal contrast estimates: responsibility-allocation main effect, monitoring-frequency main effect, and their interaction. RACE does not train agents; it is an identification design over existing agent scaffolds.

**Load-bearing assumption (stated explicitly).** RACE assumes that responsibility allocation (R_alloc) can be manipulated orthogonally to monitoring frequency (M_freq) and delegation depth (D_depth) by prompt-level instruction alone, so that the designated-safety-lead prompt changes only who owns system-level safety and not how often agents are prompted or exposed to check-triggering signals. This assumption is load-bearing: if the designated-safety-lead prompt operationally induces additional checking behavior (as the motivating evidence suggests it might), then R_alloc and M_freq are the same behavioral channel and beta_R cannot be interpreted as a responsibility effect. We therefore (i) replace prompt-level responsibility framing with an exogenous, behaviorally inert responsibility assignment (see Manipulation below), and (ii) add a manipulation check that verifies R_alloc cells are statistically indistinguishable on scheduled-check exposure and check-triggering signal rates before estimating beta_R. If no such inert manipulation exists, the contribution is reframed from 'causal decomposition of responsibility' to 'descriptive decomposition under a bundled role intervention,' dropping the orthogonality claim.

**How it works.**

```pseudocode
Inputs: task T in {sepsis_triage, news_recommender, consulting_proposal_1..10}
        team size N (fixed at 4)
        model M (fixed per run)
Factors (2x2x2 full factorial, 8 cells, R repetitions each):
  R_alloc  in {diffuse, designated}      # responsibility allocation
  M_freq   in {low, high}                # monitoring frequency: number of
                                          # scheduled system-level safety-check
                                          # prompts per agent per episode (2 vs 8)
  D_depth  in {flat, delegated}          # delegation depth: all agents see full
                                          # task state vs. lead sees full state
                                          # and sub-agents see only subtask state

Manipulation (exogenous, behaviorally inert):
  R_alloc is assigned by a fixed post-hoc attribution protocol: after each
  episode, a randomized label (designated vs diffuse) is applied to the
  transcript by the experimenter, and agents are told the label only in a
  final debrief turn that does not alter their check-scheduling instructions.
  The label does not change prompts, tools, or turn structure during the
  episode. This ensures R_alloc is independent of M_freq and D_depth by
  construction.

Manipulation check (run before estimation):
  For each cell c, compute mean scheduled_checks[i] and mean check-triggering
  signal rate (failure-signal events per episode). Require that the
  designated and diffuse cells are statistically indistinguishable on both
  (two-sided t-test, alpha=0.05, Bonferroni-corrected across 4 comparisons).
  If the check fails, report the failure and do not estimate beta_R as a
  causal effect.

For each cell c = (R_alloc, M_freq, D_depth):
  for r in 1..R (R=20):
    run team on T with scaffold fixed to cell c
    record per-agent ledger L[i] = {
       proactive_checks[i],      # system-level safety checks initiated
                                 #   by agent i before any failure signal
       rationalization[i],      # events where agent i attributes an
                                 #   observed issue to another agent/system
       deflection[i],           # events where agent i declines ownership
       accepted_ownership[i]    # events where agent i claims and acts
    }
    record team-level outcome y = {safety_failure, near_miss}

# Post-hoc normalization to make monitoring intensity non-binding:
for each ledger L[i]:
  rate[i] = proactive_checks[i] / scheduled_checks[i]
  # scheduled_checks[i] is the M_freq-determined denominator, so rate is
  # invariant to the absolute number of scheduled checks

# Estimation: linear model on cell means with responsibility-attribution
# score A[i] = accepted_ownership[i] - rationalization[i] - deflection[i]
Fit: y ~ alpha + beta_R * R_alloc + beta_M * M_freq + beta_D * D_depth
       + beta_RM * (R_alloc x M_freq) + beta_RD * (R_alloc x D_depth)
       + beta_MD * (M_freq x D_depth) + beta_RMD * (R_alloc x M_freq x D_depth)
Report beta_R (responsibility main effect), beta_RM (responsibility x
  monitoring interaction), and the per-agent A[i] distribution shift.

# Safety-lead survival test:
Compare cell (designated, high, delegated) against (designated, low, delegated)
  and against (diffuse, high, delegated): the safety-lead effect is
  "scrutiny-robust" iff beta_R is significant AND beta_RM is not.

# Dose-response sweep (tests linearity assumption):
Run additional cells at M_freq in {2, 4, 6, 8} scheduled checks per agent
  per episode, holding R_alloc and D_depth at their designated/delegated
  levels, R=20 repetitions. Fit a quadratic model y ~ R_alloc + M_freq +
  M_freq^2 + R_alloc x M_freq + R_alloc x M_freq^2 and report whether the
  quadratic terms are significant. A significant quadratic term indicates
  nonlinear moderation, in which case a null beta_RM from the 2x2x2 design
  cannot be interpreted as scrutiny-robustness.
```

**Why this design.** We chose a full 2x2x2 factorial over a single-factor manipulation because the confound the motivation names is precisely a correlation between responsibility allocation and monitoring frequency; only orthogonal manipulation can separate them, accepting the cost that eight cells times twenty repetitions is roughly eight times the compute of the original single-contrast study. We chose a per-agent ledger normalized by scheduled-check count over raw proactive-check counts because raw counts mechanically fall when monitoring is scheduled less often, accepting the cost that rate normalization assumes scheduled checks are the relevant exposure denominator — an assumption we test by also reporting raw counts. We chose to keep the agent scaffold (prompts, tools, turn structure) byte-identical across cells and vary only the responsibility-instruction and scheduling parameters, accepting the cost that this constrains the intervention to prompt-level responsibility framing rather than learned role specialization, because otherwise a designated-safety-lead cell would differ from a diffuse cell in both responsibility and scaffold. We chose a linear model on cell means over a learned attribution model because the factorial contrast is the estimand and a learned model would reintroduce the confounding the design removes; the cost is that interaction effects beyond the three-way term are not modeled. A domain expert would not describe RACE as a variant of Aligning AI Teams' intervention study because that work compares single-agent to team behavior and tests interventions one at a time, whereas RACE's estimand is the responsibility-by-monitoring interaction coefficient, which requires the orthogonal factorial that the prior study does not contain; nor is it a variant of MACPO or ICR, which change the training signal rather than the evaluation design.

**Why it measures what we claim.** The computational quantity `rate[i] = proactive_checks[i]/scheduled_checks[i]` measures per-agent proactive monitoring because the assumption that scheduled-check prompts are the sole exposure channel for proactive checks makes the ratio invariant to M_freq; this assumption fails when agents initiate checks triggered by failure signals rather than schedules, in which case the ratio reflects reactivity rather than proactivity. The quantity `A[i] = accepted_ownership[i] - rationalization[i] - deflection[i]` measures responsibility attribution because the assumption that ownership-claim and blame-shift events are mutually exclusive and exhaustive responses to an observed issue makes the difference a signed attribution score; this assumption fails when an agent both rationalizes and later accepts ownership for the same issue, in which case A[i] undercounts net ownership. The coefficient `beta_R` measures the causal effect of responsibility allocation because the factorial orthogonality assumption makes R_alloc independent of M_freq and D_depth in expectation; this assumption fails when the designated-safety-lead prompt inadvertently increases the lead's scheduled checks, in which case beta_R absorbs a monitoring effect — which is exactly why beta_RM is reported alongside it. The interaction `beta_RM` measures whether the safety-lead effect depends on scrutiny because the assumption that the linear model's interaction term captures the only route by which monitoring moderates responsibility makes a null interaction equivalent to scrutiny-robustness; this assumption fails when moderation is nonlinear (e.g., saturation at high monitoring), in which case a null beta_RM can mask a threshold effect, and we therefore also report cell-mean plots and the dose-response sweep described above.

## Contribution

['A factorial identification protocol (RACE) that orthogonalizes responsibility allocation, monitoring frequency, and delegation depth in multi-agent LLM teams, yielding a responsibility-by-monitoring interaction coefficient as the estimand for diffusion of responsibility rather than a between-condition correlation.', "A per-agent normalized responsibility ledger that separates proactive monitoring rate from ownership-attribution score, enabling the safety-lead intervention's effect to be tested for scrutiny-robustness rather than confounded with added monitoring.", 'An open evaluation harness instantiated on the sepsis-triage, news-recommender, and ten consulting-proposal tasks from Aligning AI Teams, with cell-level reporting that distinguishes responsibility main effects from coordination-induced specification drift.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Sepsis triage + news recommender + 10 consulting-proposal tasks | Same task suite as the prior team-alignment study |
| Primary metric | Safety-gap decomposition (beta_R, beta_RM) on team safety-failure rate | Directly estimates the responsibility-by-monitoring interaction |
| Baseline | Aligning AI Teams single-contrast safety-lead intervention | Tests whether orthogonal manipulation changes the conclusion |
| Baseline | Mutual-critique / group-chat intervention | Tests whether natural interventions confound responsibility with monitoring |
| Baseline | Single-agent alignment behavior | Anchors the team-level safety gap being decomposed |
| Ablation-of-ours | RACE with M_freq collapsed (single-factor R_alloc only) | Shows the interaction estimate requires orthogonal monitoring |
| Ablation-of-ours | RACE with dose-response sweep (M_freq in {2,4,6,8}) | Tests linearity assumption for beta_RM interpretation |

### Why this setup validates the claim

The central claim is that the safety-lead effect in multi-agent teams is separable from monitoring frequency, and that prior single-contrast interventions cannot tell these apart. The single-agent anchor establishes the size of the team safety gap we are decomposing. The Aligning AI Teams single-contrast baseline tests the sub-claim that a one-factor safety-lead manipulation yields an estimate that silently absorbs monitoring variation; if our beta_R differs materially from its reported effect while beta_RM is near zero, orthogonality mattered. The mutual-critique/group-chat baseline tests the sub-claim that natural interventions change responsibility framing and scheduling jointly, so their apparent benefit is confounded. The collapsed-monitoring ablation is the load-bearing test: it removes the factorial's identifying variation, so if beta_RM is only recoverable in the full 2x2x2 design and vanishes when M_freq is collapsed, the interaction estimand is real rather than a modeling artifact. The dose-response sweep directly tests the linearity assumption: if the quadratic term in M_freq is significant, a null beta_RM from the 2x2x2 design cannot be interpreted as scrutiny-robustness, and the safety-lead survival test must be re-evaluated using the dose-response curve. The decomposition metric is the right detector because it reports the main effect and the interaction as separate coefficients, so a scrutiny-robust safety lead and a monitoring-driven one produce distinguishable signatures rather than one averaged effect.

### Expected outcome and causal chain

**vs. Aligning AI Teams single-contrast safety-lead intervention** — On a sepsis-triage episode where the designated lead is prompted to own system-level safety, the baseline attributes the observed safety improvement to the role assignment because it varies only that factor and its scheduled checks drift upward with the lead prompt. Our method instead holds monitoring frequency orthogonal across the designated and diffuse cells, so the same episode yields separate beta_R and beta_RM estimates. We expect a safety-lead main effect that is clearly positive but noticeably smaller than the baseline's reported effect, with the residual reappearing in beta_RM — i.e., a gap on the designated-versus-diffuse contrast but parity once monitoring is matched.

**vs. Mutual-critique / group-chat intervention** — On a consulting-proposal task where agents critique each other's drafts, the baseline shows muted safety gains because critique prompts simultaneously raise monitoring chatter and diffuse ownership, and the two effects cancel in its single comparison. Our method instead assigns critique-style monitoring to the M_freq factor and responsibility framing to R_alloc, so the same task separates the two channels. We expect the natural-intervention baseline to show near-zero net movement, while RACE reveals a positive beta_R offset by a negative beta_RM — a pattern visible as improvement in the designated-low-monitoring cell and stagnation in the designated-high-monitoring cell.

**vs. Single-agent alignment behavior** — On a news-recommender episode where a single agent reliably flags a system-level safety failure, the baseline team replicates the single-agent check only when the failure signal is explicit, because team members diffuse responsibility for proactive scanning. Our method instead measures the per-agent proactive-check rate normalized by scheduled exposure, isolating the diffusion channel from the scheduling channel. We expect the team-versus-single-agent safety gap to shrink substantially in the designated-allocation cells and to persist in the diffuse cells, with the A[i] distribution shifting toward accepted ownership only under designated allocation.

### What would falsify this idea

If beta_R and beta_RM are both indistinguishable from zero across all task subsets, or if the safety-lead effect is uniform across monitoring levels and delegation depths rather than concentrated in the designated-low-monitoring cells where the confounding is predicted to bite, the claim that responsibility allocation is separable from monitoring frequency is wrong. Likewise, if the collapsed-monitoring ablation recovers the same beta_R as the full factorial, the orthogonal design adds nothing and the interaction estimand is an artifact; additionally, if the dose-response sweep reveals a significant quadratic term in M_freq, a null beta_RM cannot be taken as evidence of scrutiny-robustness, and the survival test must be reinterpreted.

## References

1. Aligning AI Teams
