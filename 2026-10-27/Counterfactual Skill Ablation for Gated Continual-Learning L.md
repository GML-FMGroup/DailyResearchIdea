# Counterfactual Skill Ablation for Gated Continual-Learning Ladders

## Motivation

Gated ladder benchmarks such as CLE (2026) and StuLife/ELL treat level-to-level score gains as evidence of experiential learning, but they never intervene on the agent's stored skills, so a rising ladder score cannot distinguish genuine cross-level skill reuse from per-level task-specific heuristics, prompt luck, or an evaluation-agent scoring bias. This is a meta-level failure recurring across every ladder-style benchmark: the field's shared assumption that task-completion scores suffice as evidence of learning structurally forbids measuring the causal contribution of any specific learned skill, because no benchmark currently performs an intervention on the persistent memory/skill store between levels.

## Key Insight

If a skill learned at level k is genuinely reused at level k+1, then selectively ablating that skill's memory entry must produce a measurable, skill-specific drop in paired-trial success at k+1 relative to a no-ablation control; this paired difference is an identifiable causal quantity that aggregate ladder scores cannot express.

## Method

**What it is.** CSA-Ladder (Counterfactual Skill-Ablation Ladder) is an evaluation protocol, not a new agent. Its inputs are a gated ladder task, an agent with a persistent skill-memory store, and a level-k checkpoint; its outputs are a per-skill Ablation Effect (AE) vector and a Skill-Reuse Identifiability (SRI) score at each gated transition k→k+1.

**Load-bearing assumption (stated explicitly).** Deleting a single skill-memory entry m from the persistent store is a modular, skill-specific intervention whose only causal effect on level k+1 success is the removal of m's reusable content, so that the paired success-rate difference AE[m] identifies the causal necessity of that skill rather than the store's distributed, non-modular dependence on the entry. This assumption is not taken on faith: Phase 3a below verifies modularity per entry and restricts AE computation to entries that pass the check; SRI is reported as a lower bound over the modular subset of S_k.

**How it works.**

```pseudocode
INPUT: ladder task T, agent A with skill store M = {m_1..m_n},
       levels L_1..L_5, trials per pair R = 8, ablation mask = full-entry delete,
       modularity check: retrieval-score perturbation threshold eps = 0.05 (cosine),
       power-analysis-derived thresholds tau, delta (see Phase 5)

# Phase 1: learn to level k
for level in L_1..L_k:
    A.solve(level); A.commit_new_skills_to(M)   # store snapshot M_k

# Phase 2: enumerate candidate skills
S_k = {m in M_k : m.created_during_levels(1..k)}   # skills attributable to learning

# Phase 3a: modularity verification (NEW, gates which entries get AE)
for m in S_k:
    # measure retrieval-score change of all OTHER entries when m is deleted
    d_ret[m] = max_{m' in M_k \ {m}} |retrieval_score(m', query_set) - retrieval_score(m', query_set | m deleted)|
    modular[m] = (d_ret[m] <= eps)   # entry passes if deleting m barely perturbs others
S_k_mod = {m in S_k : modular[m] == True}

# Phase 3b: paired counterfactual trials at level k+1 (only on modular subset)
for m in S_k_mod:
    for r in 1..R:
        seed_r = fixed_rng(r)                     # shared seed across arms
        y_ctrl[r]  = A.solve(L_{k+1}, seed=seed_r, store=M_k)          # no ablation
        y_abl[r]   = A.solve(L_{k+1}, seed=seed_r, store=M_k \ {m})    # ablate m
    AE[m] = mean_r(y_ctrl[r]) - mean_r(y_abl[r])   # paired success-rate difference

# Phase 3c: soft-mask arm (NEW, bounds bundled-context component)
for m in S_k_mod:
    for r in 1..R:
        y_soft[r] = A.solve(L_{k+1}, seed=seed_r, store=M_k with m's retrieval weight set to 0)
    AE_soft[m] = mean_r(y_ctrl[r]) - mean_r(y_soft[r])

# Phase 4: placebo / specificity controls
for m in S_k_mod:
    m' = random_skill_from(M_k \ S_k)              # matched-length distractor
    AE_placebo[m] = mean_r(y_ctrl[r]) - mean_r(A.solve(L_{k+1},seed_r,M_k\{m'}))

# Phase 5: aggregate with power-analysis-derived thresholds
# Power analysis on paired binomial difference at R=8: minimum detectable effect
# at 80% power, alpha=0.05, two-sided is ~0.18 success-rate units; we set
# tau = 0.18 (MDE) and delta = 0.10 (half of MDE, to require placebo separation
# beyond noise). Per-skill 95% CI on AE[m] - AE_placebo[m] via paired bootstrap
# (1000 resamples over the R paired trials).
for m in S_k_mod:
    ci_low[m] = 2.5th percentile of bootstrap(AE[m] - AE_placebo[m])
SRI(k) = fraction of m in S_k_mod with ci_low[m] > 0
       # lower bound over modular subset; non-modular entries reported separately
return {AE[m]}, {AE_soft[m]}, {AE_placebo[m]}, {ci_low[m]}, SRI(k), |S_k_mod|/|S_k|
```

**Budget and subsampling.** Total rollouts per transition = 2·R·|S_k_mod| (paired) + R·|S_k_mod| (soft-mask) + R·|S_k_mod| (placebo) = 4·R·|S_k_mod|. At R=8 this is 32·|S_k_mod| rollouts. When |S_k| > 40, stratify S_k_mod by entry length into 4 quartiles and sample a fixed 10 entries per quartile (40 total), reporting SRI over the stratified sample with inverse-probability weights.

**Why this design.** We chose full-entry deletion over soft masking as the primary intervention because deletion is the only intervention whose counterfactual is unambiguous (the skill is *absent*, not *attenuated*), accepting the cost that deletion can also remove incidental context bundled in the entry; we mitigate this by (i) a modularity check that excludes entries whose deletion perturbs other entries' retrieval beyond eps=0.05, (ii) a length-matched placebo arm, and (iii) a soft-mask arm reported alongside the delete AE to bound the bundled-context component. We chose paired trials with a shared RNG seed over independent samples because pairing removes task-instance variance from the difference, accepting the cost that the paired estimate is only valid for the specific seed distribution we fix. We chose an explicit placebo distractor arm over relying on the control alone because a raw ablation drop could reflect generic store corruption rather than skill-specific reuse, accepting the cost of doubling trial budget. We chose power-analysis-derived thresholds (tau=0.18, delta=0.10) and a per-skill bootstrap CI over a learned calibration because the protocol must be auditable and reproducible across harnesses, accepting the cost that borderline skills near the threshold are reported as ambiguous rather than resolved. This is not a variant of standard leave-one-out feature attribution (e.g., SHAP-style ablations on model inputs): those methods ablate *static input features* of a fixed model, whereas CSA-Ladder ablates *entries in a persistent, agent-authored skill store that was itself produced by prior learning*, and it explicitly separates skill-specific reuse from generic store-corruption via the modularity check, matched placebo arm, and soft-mask bound — distinctions that input-attribution methods do not need and do not provide.

**Why it measures what we claim.** The paired success-rate difference AE[m] measures *causal necessity of skill m for level k+1 success* because the shared-seed paired design assumes the only difference between arms is the presence of m; this assumption fails when the agent's execution is seed-sensitive in a way correlated with m's content (e.g., m encodes a seed-specific shortcut), in which case AE[m] reflects seed-luck rather than skill necessity. The modularity check (Phase 3a) measures *whether deletion of m is a skill-specific intervention* because it assumes that an entry whose deletion leaves other entries' retrieval scores within eps=0.05 is modular; this assumption fails when entries are densely cross-referenced through embedding similarity, in which case non-modular entries are excluded and SRI is a lower bound over the modular subset. The placebo difference AE[m] - AE_placebo[m] measures *skill-specificity* because it assumes a randomly chosen distractor is exchangeable with m on all properties except relevance; this assumption fails when the distractor is much shorter than m, in which case the placebo arm reflects entry-length effects rather than relevance. The soft-mask AE_soft[m] measures *the bundled-context component* because it assumes zeroing m's retrieval weight removes m's reusable content while holding the rest of the retrieved context fixed; this assumption fails when the agent's composition is non-linear in retrieval weights, in which case AE_soft bounds rather than equals the bundled-context component. SRI(k) measures *cross-level skill-reuse identifiability* because it assumes that a skill with placebo-corrected AE CI lower bound above zero is one whose reuse is detectable and non-spurious; this assumption fails when two skills are redundant (either alone suffices), in which case SRI under-counts reuse and should be read as a lower bound.

## Contribution

(1) A counterfactual skill-ablation protocol (CSA-Ladder) that converts gated ladder benchmarks from score-only evaluations into causal-reuse evaluations, with paired shared-seed trials, a matched placebo arm, and a per-skill Ablation Effect vector. (2) A design principle — Skill-Reuse Identifiability (SRI) — that separates genuine cross-level skill reuse from generic store corruption and from redundant-skill under-counting, yielding an auditable lower bound on reuse. (3) An open protocol specification with fixed hyperparameters (R=8, tau=0.10, delta=0.05) that can be layered onto CLE-style and StuLife-style ladders without modifying their task content.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Five-level gated ladder (Continual Learning Exam) | Gated transitions give clean k→k+1 reuse points |
| Dataset (generalization) | Task-incremental sequence with replay buffer (e.g., 10-task split, 200 examples/task) | Tests whether SRI tracks known ground-truth skill reuse outside the CLE/StuLife family |
| Primary metric | SRI(k) with paired AE vector, per-skill bootstrap CI (1000 resamples), and |S_k_mod|/|S_k| modularity rate | Detects skill-specific reuse vs. store corruption; CI replaces hard threshold |
| Baseline | Single-submission harness | Tests whether reuse is detectable without revision |
| Baseline | Feedback-and-revision harness | Tests whether reuse signal survives iterative editing |
| Baseline | Raw leave-one-out ablation (no placebo) | Tests whether placebo arm is load-bearing |
| Baseline | SHAP-style input-feature ablation on the same gated transitions | Contrasts intervention target (input features vs. skill-store entries) |
| Baseline | Store-level soft-masking / retrieval-score intervention (no deletion) | Contrasts intervention type (attenuation vs. deletion) |
| Ablation-of-ours | CSA-Ladder without placebo arm | Isolates contribution of specificity control |
| Ablation-of-ours | CSA-Ladder without modularity check (all S_k) | Isolates contribution of modularity verification |
| Ablation-of-ours | CSA-Ladder without soft-mask arm | Isolates contribution of bundled-context bound |

### Why this setup validates the claim

The central claim is that CSA-Ladder separates *skill-specific reuse* from *generic store corruption* at gated transitions, and that this separation is what makes cross-level reuse identifiable. Each baseline probes a distinct sub-claim. The single-submission harness tests whether reuse is even measurable when the agent cannot revise — if SRI is near zero there, the protocol is not just picking up revision artifacts. The feedback-and-revision harness tests whether reuse signal survives the noisier, iteratively-edited skill store that realistic agents produce; this is where generic corruption is most likely to masquerade as reuse. The raw leave-one-out baseline tests whether the placebo arm is load-bearing: if SRI computed without placebo matches SRI with placebo, the specificity claim is vacuous. The SHAP-style input-feature ablation baseline tests whether the intervention *target* matters: if input-feature ablations yield the same SRI ranking as store-entry ablations, the store-level intervention is not the novel contribution. The store-level soft-masking baseline tests whether the intervention *type* matters: if soft-mask AE and delete AE diverge systematically on bundled entries, the delete intervention is doing work that attenuation cannot. The ablation-of-ours (drop placebo, drop modularity check, drop soft-mask) isolates exactly which design choice carries the argument. SRI is the right primary metric because it is a *fraction of modular skills whose placebo-corrected AE lower confidence bound exceeds zero*, which is precisely the quantity the claim is about — not aggregate task success, which would conflate reuse with general competence. The task-incremental replay-buffer dataset provides a setting where ground-truth skill reuse is known by construction (a skill authored on task i is reused on task j>i if and only if the tasks share a sub-routine), so SRI can be checked against a known label rather than only against baselines. A falsifiable test needs the metric to move in the predicted direction *on the subset where the failure mode should bite*, and this combination of datasets + baselines + metric makes that subset identifiable in advance.

### Expected outcome and causal chain

**vs. Single-submission harness** — On a case where a level-k skill genuinely encodes a reusable sub-routine for level k+1, the single-submission baseline produces a near-flat AE vector because the agent never had a revision pass to consolidate the skill into a stable store entry, so ablating it removes something that was never cleanly present. Our method instead snapshots the store after level k, verifies modularity (Phase 3a), and ablates *committed, modular* entries, because the gated ladder forces commitment before the transition. We expect a noticeable SRI gap on transitions where the skill was authored during level k, but parity on transitions where the skill predates the ladder — the signal should be concentrated, not uniform. On the task-incremental replay-buffer dataset, we expect SRI to be high precisely on task pairs with known shared sub-routines and near zero on pairs without, confirming the metric tracks ground-truth reuse.

**vs. Feedback-and-revision harness** — On a case where iterative editing bundles incidental context into a skill entry, the feedback baseline produces a large raw AE for that entry because deletion removes the bundled context too, and the baseline has no way to distinguish this from genuine skill necessity. Our method instead compares against a length-matched placebo and reports the soft-mask AE alongside the delete AE, because the placebo arm absorbs the generic-corruption component and the soft-mask arm bounds the bundled-context component. We expect the placebo-corrected AE to shrink sharply for bundled entries while remaining stable for clean entries, and we expect AE_soft to be systematically smaller than AE_delete on bundled entries — the diagnostic signature of bundled context. SRI under feedback-revision should be *lower* than under single-submission on bundled-entry-heavy transitions but comparable elsewhere.

**vs. Raw leave-one-out ablation (no placebo)** — On a case where the store is large and deletion perturbs unrelated lookup paths, the raw-ablation baseline produces inflated AE values across most skills because every deletion carries a generic corruption cost. Our method instead subtracts the placebo AE and restricts to entries passing the modularity check (d_ret[m] ≤ 0.05), because the placebo is drawn from the same store and pays the same corruption cost, and non-modular entries are excluded by construction. We expect raw AE to be uniformly elevated relative to placebo-corrected AE, with the *difference* between them concentrated on skills whose entries are long or heavily cross-referenced — the diagnostic signature of corruption rather than reuse — and we expect the modularity rate |S_k_mod|/|S_k| to drop on transitions where the store is dense.

**vs. SHAP-style input-feature ablation** — On the same gated transitions, input-feature ablations operate on the task prompt rather than the skill store, so they cannot express the counterfactual "skill m absent from memory." We expect SHAP-style SRI to be near zero or uncorrelated with store-entry SRI, demonstrating that the intervention target — not the estimator — is the novel contribution.

**vs. Store-level soft-masking (no deletion)** — Soft-masking zeroes m's retrieval weight while holding the rest of the retrieved context fixed, so it cannot remove bundled context. We expect AE_soft ≤ AE_delete on bundled entries and AE_soft ≈ AE_delete on clean entries, with the gap between them serving as the bundled-context bound.

### What would falsify this idea

If SRI computed *without* the placebo arm matches SRI *with* the placebo arm across all gated transitions, the specificity claim is wrong and the protocol reduces to standard leave-one-out ablation. If the SRI gap versus the single-submission baseline is uniform across transitions rather than concentrated on transitions where skills were authored during the immediately preceding level, or if SRI on the task-incremental replay-buffer dataset does not track known ground-truth reuse, the causal chain from gated commitment to identifiable reuse is not supported.

## References

1. Continual Learning Exam: Benchmarking Long-Horizon RSI and Experiential Learning in LLM Agents
2. Building Self-Evolving Agents via Experience-Driven Lifelong Learning: A Framework and Benchmark
