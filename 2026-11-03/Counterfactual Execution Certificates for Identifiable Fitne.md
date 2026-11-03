# Counterfactual Execution Certificates for Identifiable Fitness in Self-Evolving Code Agents

## Motivation

Self-evolving code agents such as Self-Evolving Coding Agents via Harness-Policy Optimization and SWE-agent-style repair loops accept candidate harnesses or patches based on verdicts produced by the same model that generated the candidate, so the fitness label and the optimized object share a generator and fitness is not identifiable from the signal alone. This self-referential loop means a candidate can be 'accepted' for reasons that do not track any external correctness quantity — the model's failure narrative is both the proposal and the evidence. The missing ingredient is not new tooling (failure narratives and static-analysis traces already exist) but a commitment structure that forces each candidate to stake its acceptance on a falsifiable behavioral prediction checked by an independent executor.

## Key Insight

Requiring each candidate to commit, before acceptance, to a minimal input pair whose behavioral divergence is predicted by the candidate's own failure narrative and verified by a non-LLM differential oracle converts the acceptance signal from a self-judged verdict into an externally reproducible quantity, breaking the shared-generator loop that makes fitness unidentifiable.

## Method

**What it is.** CEC (Counterfactual Execution Certificate) is an acceptance protocol wrapped around any harness/patch proposer. Its inputs are the incumbent program $P_{inc}$, a candidate $P_{cand}$ with its natural-language failure narrative $N$, and a held-out behavioral spec $\mathcal{S}$ (a set of input generators and observable-output projections). Its output is a scalar certificate-fitness $\phi \in [0,1]$ plus an accept/reject decision, where $\phi$ counts only behavioral deltas that an independent executor reproduces.

**Load-bearing assumption (stated explicitly).** CEC assumes that the candidate's natural-language failure narrative $N$ can be mechanically compiled by a non-LLM pattern-matcher into a differential probe (input $x_i$ drawn from $\mathcal{S}$'s grammar conditioned on the claimed condition $c_i$, plus observable projection $o_i$ named by the claimed delta $d_i$) that faithfully instantiates the narrative's claimed condition and delta, so that certified probes measure the claimed behavioral change rather than an artifact of the compilation step. This assumption fails when the narrative names a condition outside the grammar (e.g., a rare race) or uses ambiguous observable naming; in that case the delta is dropped and $\phi$ reflects only the compilable subset, replacing the shared-generator confound with a compiler-coverage confound. We therefore (i) require the proposer to emit the probe as executable code alongside the narrative, (ii) certify the probe by checking that it actually exercises the claimed branch via coverage/instrumentation and that $P_{inc}$ fails it, and (iii) report $\phi$ conditioned on compiler coverage so the metric cannot be inflated by dropping hard-to-compile deltas.

**How it works.**

```pseudocode
Input: P_inc, (P_cand, N), spec S, budget k=8 probes
Output: phi, coverage, accept/reject

1. NARRATIVE-TO-PREDICTION: parse N into claimed deltas D = {(condition c_i, expected_change d_i)}   # e.g. "on empty input it now returns [] instead of raising"
2. PROBE COMPILATION: for each d_i, mechanically compile a differential probe
     probe_i = (input x_i drawn from S conditioned on c_i, observable projection o_i)
     using a non-LLM compiler: pattern-match c_i against S's generator grammar, emit x_i; o_i = the output field d_i names.
   If a delta cannot be compiled to a probe, it is dropped (not credited).
   Additionally, the proposer must emit probe_i as executable code (input generator + projection function) alongside N; this code is used as a fallback if the pattern-matcher fails.
3. PROBE VALIDATION (independent reachability oracle): for each compiled probe_i, run P_inc and P_cand under instrumentation (e.g., line coverage or branch coverage) and verify that:
     (a) the divergent line/branch claimed by d_i is reached in at least one of P_inc or P_cand;
     (b) P_inc fails the probe (i.e., o_inc does not match d_i's expected sign or raises).
   If either check fails, the probe is marked invalid and dropped from the numerator but retained in the denominator for coverage reporting.
4. EXECUTION: run P_inc(x_i) and P_cand(x_i) under the sandbox; record (o_inc, o_cand).
5. CERTIFICATE CHECK: probe_i certifies iff o_inc != o_cand AND the sign of the difference matches d_i AND probe_i passed validation in step 3.
     A minimal pair is retained only if no strictly smaller input in S's grammar (by AST size, searched to depth 3) also certifies.
6. phi = (# certified probes) / (# compiled probes), with # compiled probes >= 1 required.
   coverage = (# compiled probes) / (# claimed deltas in D).
7. ACCEPT iff phi >= tau (tau=0.5) AND at least one minimal pair exists AND coverage >= 0.5; else reject and return the uncertified deltas as feedback to the proposer.
```

**Why this design.** We chose to compile probes from the candidate's *own* failure narrative rather than from a fixed test suite because a fixed suite lets the proposer overfit to known tests, accepting the cost that deltas the narrative cannot articulate are simply not credited (a conservative bias). We chose a minimal-pair requirement over single-input divergence because single-input divergence can be produced by incidental side effects (e.g., a changed log line), accepting the cost of an AST-size search bounded to depth 3. We chose a fraction-of-claimed-deltas fitness over a binary pass/fail because it gives a graded signal for RL, accepting the cost that a candidate claiming one easy delta can score 1.0 while a candidate claiming five hard deltas scores 0.4 — we mitigate this by requiring $\geq 1$ minimal pair and reporting claim count alongside $\phi$. We chose a non-LLM oracle (sandboxed interpreter plus spec-defined projections) over an LLM critic because the entire motivation is that LLM verdicts share a generator with candidates; an expert would not describe this as a variant of Self-HPO or SWE-agent because those accept the proposer's verdict, whereas CEC makes acceptance contingent on an execution trace the proposer cannot author. We added probe validation via an independent reachability oracle and coverage reporting to address the load-bearing assumption: the proposer-emitted executable probe and the coverage check ensure that certified probes exercise the claimed branch, and coverage conditions $\phi$ on compiler success so that dropping hard-to-compile deltas cannot inflate the metric.

**Why it measures what we claim.** The quantity $\phi$ = certified-probes / compiled-probes measures *reproducible behavioral delta* because we assume the spec's generator grammar covers the input distribution on which the narrative's condition $c_i$ is defined; this assumption fails when the narrative names a condition outside the grammar (e.g., a rare race), in which case the delta is dropped and $\phi$ reflects only the compilable subset. The minimal-pair search measures *causal specificity* of the delta because we assume AST-size ordering is a valid intervention-cost proxy; this fails when a smaller input triggers an unrelated code path, in which case the pair reflects incidental divergence rather than the claimed mechanism. The sign-match check measures *narrative faithfulness* because we assume the observable projection $o_i$ named in $d_i$ is the field the narrative actually refers to; this fails under ambiguous naming, in which case the check reflects surface string agreement rather than semantic correctness. The probe validation step measures *branch reachability* because we assume the instrumentation correctly identifies whether the divergent line is executed; this fails when instrumentation is coarse (e.g., line-level only) and the claimed branch shares a line with unrelated code, in which case validation reflects line-level coverage rather than branch-level reachability. The coverage metric measures *compiler coverage* because we assume the set of claimed deltas $D$ is a faithful enumeration of the narrative's claims; this fails when the narrative omits deltas or when the proposer strategically claims only easy deltas, in which case coverage reflects claim selection rather than narrative completeness.

## Contribution

(1) A counterfactual execution certificate protocol that converts candidate acceptance in self-evolving code agents from an LLM verdict into a fraction of claimed behavioral deltas reproducible by a non-LLM differential oracle, with a mechanical narrative-to-probe compiler. (2) A design principle — commitment-before-acceptance — showing that fitness identifiability in self-referential loops is obtained by forcing the proposer to stake acceptance on a falsifiable prediction, and an analysis of when narrative-compilability bounds the achievable certificate fitness. (3) An open protocol and probe-compiler implementation integrable with existing harness-optimization and repair pipelines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | SWE-Bench Pro held-out tasks | Long-horizon failures expose narrative-vs-behavior gaps |
| Primary metric | Certified-accept precision (accepted patches that pass held-out spec) | Directly tests whether acceptance tracks real behavioral delta |
| Secondary metric | Compiler coverage (fraction of claimed deltas that compile to a probe) | Tests load-bearing assumption; conditions phi on compiler success |
| Baseline | Self-HPO (harness+policy alternation) | Tests claim that proposer-authored verdicts inflate acceptance |
| Baseline | SWE-agent-style LLM critic acceptance | Tests claim that shared-generator critics cannot discriminate |
| Baseline | Fixed-suite test-pass acceptance | Tests claim that fixed suites invite overfitting |
| Ablation-of-ours | CEC without minimal-pair search (single-input divergence) | Isolates causal-specificity contribution |
| Ablation-of-ours | CEC without probe validation (no reachability oracle) | Isolates probe-validation contribution |
| Ablation-of-ours | CEC with free-form narratives only (no executable probe fallback) | Isolates compiler-coverage confound |

### Why this setup validates the claim

The central claim is that acceptance contingent on an independently reproduced behavioral delta, compiled from the candidate's own narrative, yields higher precision than any acceptance rule that trusts the proposer or a fixed suite. SWE-Bench Pro is the right substrate because its long-horizon tasks generate exactly the failure mode the method targets: candidates that claim a fix while changing behavior only incidentally. The three baselines each attack a distinct sub-claim: Self-HPO tests whether proposer-verdict acceptance inflates false accepts; the LLM-critic baseline tests whether a shared-generator judge can distinguish real from narrated deltas; the fixed-suite baseline tests whether known tests are overfit. The ablation isolates the minimal-pair requirement, so if precision gains survive only with minimal pairs, the causal-specificity mechanism is doing the work. Certified-accept precision is the correct metric because it directly measures the quantity the protocol claims to control — the fraction of accepted patches that truly change behavior — rather than proxy pass rates that both baselines can inflate. The secondary metric, compiler coverage, directly tests the load-bearing assumption: if coverage is low (e.g., <50%), then $\phi$ is dominated by which deltas happen to be compilable, and we report precision conditioned on coverage to ensure the metric is not inflated by dropping hard-to-compile deltas. The additional ablations isolate the probe-validation and executable-probe-fallback components, confirming that the reachability oracle and proposer-emitted probes are necessary for the certificate to measure claimed behavioral change rather than compiler artifacts.

### Expected outcome and causal chain

**vs. Self-HPO** — On a case where a candidate patch renames a variable and rewrites a log line while leaving the buggy branch untouched, Self-HPO accepts because its harness-policy loop treats the proposer's own success narrative as the verdict, and the narrative is fluent. Our method instead compiles the narrative's claimed delta into a differential probe, executes both programs, and finds identical observable outputs, so the delta fails certification and the patch is rejected with feedback. We expect a noticeable precision gap concentrated on tasks whose failures are cosmetic or narrative-only, with parity on tasks where the patch genuinely alters the buggy path.

**vs. LLM-critic acceptance** — On a case where the candidate claims "empty input now returns [] instead of raising," an LLM critic drawn from the same generator family agrees because the narrative and the diff cohere stylistically, even though the candidate still raises on empty input. Our method instead runs the compiled probe under a sandboxed interpreter and observes the exception on both programs, so the sign-match check fails and phi drops. We expect the gap to concentrate on ambiguous-naming and fluent-but-wrong cases, with the critic matching us on unambiguous crashes.

**vs. Fixed-suite test-pass acceptance** — On a case where the candidate special-cases the exact inputs in the visible suite, the fixed-suite baseline accepts because all known tests pass, while the held-out spec's generators produce inputs the candidate never saw. Our method instead draws probes from the spec grammar conditioned on the narrative's claimed condition, so the special-case does not certify the claimed general delta. We expect a precision gap that grows with the size of the visible suite relative to the spec, and near-parity when the visible suite already covers the narrative's condition.

**vs. CEC without minimal-pair search (ablation)** — On a case where a smaller input triggers an unrelated early-return path, single-input divergence accepts because some input differs, even though the claimed mechanism is untouched. Our method's AST-bounded minimal-pair search rejects the pair as incidental, so phi reflects only causally specific deltas. We expect the ablation to show inflated acceptance on tasks with broad input grammars, confirming that minimal pairs, not mere divergence, drive the precision gain.

**vs. CEC without probe validation (ablation)** — On a case where the narrative claims a delta on a branch that is never reached by the compiled probe (e.g., due to a mis-parsed condition), the unvalidated CEC credits the probe if outputs differ incidentally, while our method's reachability oracle detects that the claimed branch is not exercised and drops the probe. We expect the unvalidated ablation to show inflated acceptance on tasks with complex control flow, confirming that probe validation is necessary to prevent compiler artifacts from inflating phi.

**vs. CEC with free-form narratives only (ablation)** — On a case where the narrative uses a condition outside the spec grammar (e.g., "when the cache is cold"), the free-form-only CEC drops the delta and coverage falls below 0.5, triggering rejection even if the patch is correct. Our method's executable-probe fallback allows the proposer to supply a probe that compiles, raising coverage and enabling acceptance. We expect the free-form-only ablation to show lower recall on tasks with non-stereotyped conditions, confirming that the executable-probe fallback mitigates the compiler-coverage confound.

### What would falsify this idea

If certified-accept precision is uniform across task subsets rather than concentrated on narrative-only, ambiguous-naming, and overfit-prone cases where the predicted failure modes live, then the mechanism is not doing what we claim. Equally falsifying: if the minimal-pair ablation matches full CEC everywhere, the causal-specificity component is inert and the central claim collapses to ordinary differential testing; if compiler coverage is below 50% on real SWE-Bench Pro narratives and precision does not improve with the executable-probe fallback, the load-bearing assumption is false and the method reduces to a compiler-coverage confound.

## References

1. Self-Evolving Coding Agents via Harness-Policy Optimization
2. SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?
3. Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution
4. OpenHands: An Open Platform for AI Software Developers as Generalist Agents
5. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
6. BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions
7. Large Language Models Can Self-Improve
8. Eliminating Meta Optimization Through Self-Referential Meta Learning
