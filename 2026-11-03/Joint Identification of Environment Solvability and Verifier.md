# Joint Identification of Environment Solvability and Verifier Correctness via Exclusion-Restricted Calibration

## Motivation

Across the computer-use-agent literature, two distinct noisy signals are both treated as correctness oracles: the synthesized environment's solvability gate (Agent World Model in the Wild filters tasks via heterogeneous rollouts and cross-judge consistency) and the skill-store success verifier (ASI programmatically verifies induced skills before reuse; PAE scores rollouts to produce learning signal). Both are noisy measurements of the same latent task-correctness variable, yet no procedure identifies them jointly against human ground truth, so correlated verifier-environment error is silently absorbed into the skill store's correctness invariant. The structural problem is that without an identifiability condition, an environment that renders an interface ambiguously and a verifier whose judge model is idiosyncratically strict can produce the same observed success rate while implying opposite downstream skill-store contents — and neither tree can tell which error source is active.

## Key Insight

An exclusion restriction — interface-rendering noise perturbs solvability but is invisible to human-judged correctness, while judge-model idiosyncrasy perturbs the verifier's label but cannot change whether the task is physically solvable — supplies the moment equations that separate the two error variances against a small human-annotated anchor, making the latent correctness variable identifiable without annotating every task.

## Method

## JIVE: Joint Identifiability of Verifier and Environment

**(A) What it is.** JIVE takes as input (i) a bulk of synthesized CUA tasks each carrying a solvability-gate outcome `S ∈ {0,1}` from the Agent World Model rollout gate and a verifier success label `V ∈ {0,1}` from the skill-store judge, (ii) a small human-annotated calibration set `H` of `n_h = 512` tasks with human-judged correctness `C* ∈ {0,1}` collected on a **rendering-normalized re-rendered interface** (each task re-rendered at fixed resolution 1280×720 with MCP/CLI timeouts retried up to 3× and the cleanest successful render shown to the annotator; if no clean render is obtainable the task is excluded from `H`), and (iii) per-task interface-rendering-noise features `z_env` (screenshot-render variance measured as pixel-wise std over 5 re-renders, MCP/CLI error-rate proxies = fraction of tool calls returning non-zero exit or timeout) and per-task judge-idiosyncrasy features `z_judge` (judge-model identity one-hot, prompt-template hash one-hot, cross-judge disagreement count over 3 judges). Both `z_env` and `z_judge` are **residualized on measured shared confounders** `w` = (task length in tokens, number of rollout steps, rollout-quality score = mean reward of the 5 rollouts) by OLS before entering the moment equations, so the instruments used below are `z̃_env = z_env − Proj(z_env | w)` and `z̃_judge = z_judge − Proj(z_judge | w)`. It outputs a calibrated posterior `P(C=1 | S, V, z̃_env, z̃_judge)` for every task, which replaces the raw verifier label as the skill store's correctness invariant.

**(B) How it works.**

```pseudocode
# Phase 0: Instrument construction (shared-driver residualization)
# w_i = (task_length_i, n_rollout_steps_i, rollout_quality_i)
z_env_tilde_i   = z_env_i   - OLS_fit(z_env   ~ w)_predict(w_i)
z_judge_tilde_i = z_judge_i - OLS_fit(z_judge ~ w)_predict(w_i)

# Phase 1: Latent-variable model with exclusion restrictions
# Latent task correctness C ~ Bernoulli(sigmoid(a))
# Solvability gate:  S = 1[ f_env(C, z_env_tilde) + eps_env > tau ]   # z_env_tilde enters ONLY here
# Verifier label:    V = 1[ f_judge(C, z_judge_tilde) + eps_judge > tau ]  # z_judge_tilde enters ONLY here
# Exclusion: z_env_tilde excluded from V-equation; z_judge_tilde excluded from S-equation.
# f_env and f_judge are linear: f_env = alpha_env*C + beta_env*z_env_tilde, f_judge = alpha_judge*C + beta_judge*z_judge_tilde
# eps_env ~ N(0, sigma_env^2), eps_judge ~ N(0, sigma_judge^2)

# Phase 2: Moment construction from calibration set H (n_h = 512)
for each task i in H:
    m1_i = S_i - P(S=1 | C*_i, z_env_tilde_i)          # env residual
    m2_i = V_i - P(V=1 | C*_i, z_judge_tilde_i)        # judge residual
    m3_i = (S_i - V_i) * z_env_tilde_i                  # exclusion moment (env)
    m4_i = (S_i - V_i) * z_judge_tilde_i                # exclusion moment (judge)
# Stack moments; estimate params theta = (a, sigma_env, sigma_judge, beta_env, beta_judge)
# by generalized method of moments (GMM) with weight matrix = inverse of moment covariance.
theta_hat = GMM(moments=[m1,m2,m3,m4], data=H, weight='two-step')
# Over-identification check: Hansen J statistic on the 4 moment conditions vs 5 parameters
# (1 over-identifying restriction); reject instrument validity if p < 0.05.

# Phase 3: Out-of-sample calibration on bulk
for each unlabeled task j:
    C_hat_j = posterior_mean(C_j | S_j, V_j, z_env_tilde_j, z_judge_tilde_j, theta_hat)
    # C_hat is the calibrated success label; it is a continuous score in [0,1].

# Phase 4: Transfer to skill store
# Replace verifier label V_j with C_hat_j in the skill store's correctness invariant.
# Skills whose mean C_hat over supporting tasks falls below threshold gamma=0.5 are demoted.
```

**(C) Why this design.** We chose a GMM estimator with explicit exclusion moments over a jointly-trained neural disentanglement head because the latter (the shape of a learned independence assertion) cannot manufacture separability that the data does not contain — the exclusion restriction is a structural assumption about *where* each noise source enters, not a statistical independence claim the model can learn. We chose a small human-annotated anchor set (`n_h = 512`) over full annotation because the moment equations are over-identified once `n_h` exceeds the parameter count (5 parameters; 512 gives >100× over-identification), so the unlabeled bulk supplies the variance reduction while the anchor supplies identification; the cost is that the anchor must be representative of the bulk's task distribution, which we mitigate by stratified sampling over environment type (GUI/MCP/CLI) with proportional allocation. We chose to demote skills by posterior mean rather than by hard relabeling because hard relabeling would discard the calibrated uncertainty and reintroduce a threshold artifact; the cost is that the skill store's invariant becomes a soft score, requiring downstream consumers to accept graded correctness. A domain expert would not describe JIVE as a variant of PAE or ASI, because those systems *use* the evaluator/verifier as an oracle whereas JIVE *identifies* the oracle's error against a separate environment-error channel — the object of inference is the error decomposition, not the skill.

**(D) Why it measures what we claim.** The GMM residual `m1_i` measures environment-gate error because the exclusion restriction assumes `z̃_env` is the only channel through which interface-rendering noise enters the solvability equation, so conditioning on `C*` and `z̃_env` leaves only the gate's own noise; this assumption fails when rendering noise also degrades the human annotator's judgment (e.g., a task that looks unsolvable to a human because the screenshot is blurry), in which case `m1` reflects annotator confusion rather than gate error — which is why `C*` is collected on a rendering-normalized re-render (Phase 0) and tasks without a clean render are excluded from `H`. The residual `m2_i` measures verifier error because judge-model identity and template hash are assumed to affect only the judge's label, not physical solvability; this fails when a judge model's idiosyncrasy correlates with task difficulty (a strict judge on hard tasks), in which case `m2` conflates judge bias with difficulty — mitigated by residualizing `z_judge` on task length and rollout quality. The exclusion moments `m3_i` and `m4_i` measure the *separability* of the two error sources because under the exclusion restriction each moment is zero in expectation only if the corresponding cross-channel is absent; this fails when `z̃_env` and `z̃_judge` are themselves correlated (e.g., both driven by task length), in which case the moments cannot distinguish which channel is active and the posterior `C_hat` collapses toward the raw verifier label — the Hansen J test on the 4 moments vs 5 parameters is the empirical check of this assumption, and a rejection (p < 0.05) flags that the instruments are invalid and the decomposition should not be trusted. The calibrated posterior `C_hat_j` measures latent task correctness because the generative model assumes `C` is the common cause of both `S` and `V`; this fails when the environment gate and verifier share a *non-causal* common driver (both invoked on the same rollout), in which case `C_hat` reflects rollout quality rather than task correctness — the residualization of both instruments on `rollout_quality` in Phase 0 is the concrete guard against this failure mode.

## Contribution

(1) A joint identifiability procedure (JIVE) that co-calibrates the synthesized-environment solvability gate and the skill-store success verifier against a small human-annotated anchor using exclusion-restricted moment equations, producing a calibrated per-task correctness posterior that replaces the raw verifier label. (2) A design principle: the environment's interface-rendering noise and the verifier's judge-model idiosyncrasy are separable error channels, and the minimal structural assumption enabling separation is an exclusion restriction on observable proxy features rather than full annotation. (3) A transferable calibration artifact — a `theta_hat` parameter set estimated once on an anchor and applied to any new synthesized-environment or skill-store corpus that exposes the same proxy features, enabling cross-tree label transfer without re-annotation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | CUA-World with synthesized Agent World Model tasks | Mixed GUI/MCP/CLI, heterogeneous rollouts, known gate outcomes |
| Dataset (secondary) | VisualWebArena (GUI subset) with a second verifier family | Tests protocol reuse across benchmark and verifier |
| Primary metric | Skill-store success rate on held-out CUA tasks | Downstream consumer of correctness invariant, not label accuracy |
| Secondary metric | Parameter recovery error on sigma_env, sigma_judge (simulation) | Checks identifiability claim directly |
| Baseline | Raw verifier label (V as-is) | Tests whether calibration beyond raw judge helps |
| Baseline | Cross-judge majority vote | Tests whether naive ensembling substitutes for error decomposition |
| Baseline | PAE-style evaluator as oracle | Tests whether treating verifier as oracle is the gap |
| Baseline | Hui-Walter latent-class model (no exclusion moments) | Tests whether exclusion moments add identification |
| Baseline | Dawid-Skene multi-annotator model | Tests whether standard aggregation substitutes for decomposition |
| Ablation-of-ours | JIVE without exclusion moments (m3,m4 dropped) | Tests whether separation of error channels drives the gain |
| Ablation-of-ours | JIVE without Phase-0 residualization | Tests whether shared-driver residualization is load-bearing |
| Ablation-of-ours | JIVE with C* on raw (non-normalized) renders | Tests whether rendering-normalized anchor is load-bearing |

### Why this setup validates the claim

The central claim is that JIVE's gain comes from *separating* environment-gate error from verifier error via exclusion restrictions, and that this separation produces a better skill-store correctness invariant than either the raw verifier or naive ensembling. The raw-verifier baseline tests the sub-claim that calibration beyond the judge is needed at all: if JIVE does not beat raw V, the decomposition is pointless. The cross-judge majority vote tests the sub-claim that the gain is not merely variance reduction from aggregation — a majority vote also reduces judge noise but cannot separate it from environment noise. The PAE-style oracle baseline tests the sub-claim that *identification* rather than *use* of the verifier is the operative move. The Hui-Walter and Dawid-Skene baselines test whether the exclusion moments add identification beyond standard latent-class and multi-annotator estimators; if JIVE matches them, the exclusion restriction is not doing the claimed work. The ablation dropping exclusion moments is the load-bearing test: if the gain survives without m3/m4, the exclusion restriction is not doing the work, and the mechanism story in (D) is wrong. The Phase-0 residualization ablation and the raw-render anchor ablation test the two concrete guards against the shared-driver failure mode flagged in (D). The secondary VisualWebArena experiment with a second verifier family tests whether the identified error variances are stable across benchmarks and verifier families, establishing the protocol as reusable rather than single-dataset. The simulation study with known ground-truth C, sigma_env, and sigma_judge tests parameter recovery and bias under exclusion-restriction violation, backing the identifiability claim with demonstrated recovery rather than assertion. The primary metric is skill-store downstream success rather than label accuracy because the claim is about the invariant's usefulness to consumers, and a calibrated soft score that improves label accuracy but not downstream success would falsify the transfer step in Phase 4.

### Expected outcome and causal chain

**vs. Raw verifier label** — On a case where a strict judge model marks a genuinely solvable task as failed because its prompt template over-penalizes partial GUI state, the raw verifier propagates V=0 into the skill store, demoting a correct skill. Our method instead attributes the V=0 to judge idiosyncrasy via z_judge and the exclusion moment m4, recovering a high posterior C_hat because the environment gate S=1 provides independent evidence of solvability. We expect a noticeable skill-store success gap on the subset where S and V disagree, with near-parity where they agree — the disagreement subset is exactly where decomposition pays.

**vs. Cross-judge majority vote** — On a case where rendering noise (blurry screenshot, MCP timeout) causes the environment gate to mark a solvable task unsolvable, majority vote across judges still sees V=1 from consistent judges but cannot explain the S=0, so it either trusts V and ignores the gate or averages toward 0.5. Our method uses z_env in the S-equation and the m3 exclusion moment to recognize that the gate failure is environmental, not task-level, and keeps C_hat high. We expect the gap to concentrate on tasks with high screenshot-render variance or elevated MCP error rates, not on clean-interface tasks.

**vs. PAE-style evaluator as oracle** — On a case where the verifier and the environment gate are both invoked on the same rollout and share a non-causal common driver (rollout quality), the PAE-style oracle treats V as ground truth and inherits the shared driver as if it were task correctness, so skills supporting tasks that happened to get good rollouts are over-promoted. Our method's Phase-0 residualization of both instruments on rollout_quality, together with the generative model's residual structure, shrinks the calibrated posterior toward the prior on tasks where S and V are jointly driven. We expect PAE-oracle to over-promote on high-rollout-quality subsets and JIVE to show flatter promotion rates across rollout-quality strata.

**vs. Hui-Walter latent-class model (no exclusion moments)** — The Hui-Walter model estimates two conditionally independent noisy measurements of C but cannot attribute error to environment vs. judge because it has no instruments. On tasks where S and V disagree, Hui-Walter splits the difference symmetrically, whereas JIVE uses z̃_env and z̃_judge to assign the disagreement to the correct channel. We expect JIVE to beat Hui-Walter specifically on the S/V-disagreement subset and to match it on the agreement subset; if JIVE matches Hui-Walter everywhere, the exclusion moments add no identification.

**vs. Dawid-Skene multi-annotator model** — Dawid-Skene aggregates multiple judge labels but treats the environment gate as just another annotator, so a rendering-noise-induced S=0 is absorbed as a low-reliability vote rather than attributed to the environment channel. On high-z_env tasks we expect Dawid-Skene to under-promote correct skills relative to JIVE, with parity on low-z_env tasks.

### What would falsify this idea

If the ablation without exclusion moments matches JIVE's skill-store success within noise across all subsets, the exclusion restriction is not doing the claimed work and the error-decomposition story in (D) is wrong. Likewise, if JIVE's gain over the raw verifier is uniform across the S/V-agreement and S/V-disagreement subsets rather than concentrated on the disagreement subset, the gain is not coming from separating the two error channels. Additionally, if the Hansen J test rejects (p < 0.05) on the bulk and the Phase-0 residualization ablation shows no degradation when removed, the instrument-validity assumption is empirically false and the joint-identification claim should be dropped in favor of single-channel calibration.

## References

1. Agent World Model in the Wild: Computer-Use Agent Environment Synthesis
2. VISUALSKILL: Multimodal Skills for Computer-Use Agents
3. Inducing Programmatic Skills for Agentic Tasks
4. Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents
5. Agent Workflow Memory
6. Proposer-Agent-Evaluator(PAE): Autonomous Skill Discovery For Foundation Model Internet Agents
7. Agent S: An Open Agentic Framework that Uses Computers Like a Human
8. AgentStore: Scalable Integration of Heterogeneous Agents As Specialized Generalist Computer Assistant
