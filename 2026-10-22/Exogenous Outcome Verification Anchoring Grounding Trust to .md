# Exogenous Outcome Verification: Anchoring Grounding Trust to Post-Action State Transitions

## Motivation

Across the computer-use agent literature, grounding and feasibility outputs are validated by the very module that produced them, creating a circular trust path: VISUALSKILL's load_topic retrieval and Agent S2's Mixture-of-Grounding both let the agent's own grounding capability certify the correctness of its grounding, and Agent S's experience-augmented planning scores feasibility using the same internal representation that generated the plan. Because the verifier and the checked module share the same conditioning information, no independent trust signal exists and grounding errors that are internally self-consistent (e.g., a confidently wrong element localization) pass validation unchanged. The field therefore lacks a verifier whose correctness judgment is statistically identifiable from the grounding module it audits.

## Key Insight

Post-action environment state transitions are exogenous to the grounding module's computation — the grounding module never conditions on them — so a verifier trained to predict outcome divergence under a randomized grounding perturbation learns a decision boundary that is identifiable from the checked module rather than merely consistent with it.

## Method

**What it is.** We introduce **XOV (Exogenous Outcome Verifier)**, a grounding verifier whose training signal is the *causal effect of a grounding output on the post-action environment state*, not the grounding module's self-consistency. Inputs: an agent step's grounding output `g` (a structured skill/feasibility tuple: target element, action type, feasibility flag), a pre-action observation `o_t`, and the post-action observation `o_{t+1}` produced by executing `g`. Output: a scalar correctness score `v(g) ∈ [0,1]`.

**Load-bearing assumption (stated explicitly).** The quantity `delta` (defined below) is attributable to the grounding perturbation `g_pert` alone — i.e., replaying the same pre-action observation with a perturbed grounding yields a transition whose difference from the logged transition isolates grounding causality rather than environment stochasticity or replay infidelity. This assumption is load-bearing for the entire method: `delta` is both the contrastive target and the label weight. We make it explicit here and operationalize it via variance control and null-perturbation subtraction (Phase 1), and we bound its validity empirically via a replay-fidelity audit and a same-role-ambiguity stress test (experiment).

**How it works.**

```pseudocode
# Phase 1: counterfactual perturbation generation with variance control (no retraining of base agent)
# Pre-collected replay corpus: N logged steps (o_t, g, o_{t+1}) from OSWorld + WebArena.
# For each logged step:

# 1a. Null-perturbation baseline (replay-intrinsic stochasticity estimate)
for k in 1..K_null:                      # K_null = 5
    o_null_k = replay(o_t, g)            # execute the IDENTICAL grounding K_null times
null_dist = {o_null_1, ..., o_null_K_null}

# 1b. Perturbed grounding generation (restricted to semantically distinct perturbations)
g_pert = perturb(g)                      # randomized skill/feasibility perturbation:
                                         #   - swap target element with a random same-role candidate
                                         #   - flip feasibility flag with prob 0.5
                                         #   - replace skill id with nearest-neighbor in skill index
# Semantic-distinctness filter: accept g_pert only if the perturbed target element
# differs from g's target in at least one of {role, accessible-name, DOM-path} by an
# independent oracle (rule-based attribute comparator); otherwise resample.

# 1c. Perturbed replay distribution
for k in 1..K_pert:                      # K_pert = 5
    o_pert_k = replay(o_t, g_pert)
pert_dist = {o_pert_1, ..., o_pert_K_pert}

# 1d. Variance-controlled outcome divergence
# Two-sample statistic between null_dist and pert_dist on structured state features
# (element attributes, focus, URL, visible-text hash):
#   delta_raw = D(null_dist, pert_dist)   # e.g., total-variation distance on hashed state features
delta = max(0, delta_raw - D_null)       # subtract replay-intrinsic stochasticity
# where D_null = D(null_dist, null_dist') is estimated from a held-out split of null replays
# (or from the within-null_dist pairwise distances), so delta is an unbiased estimate of
# grounding-caused divergence rather than a mixture of causality, uniqueness, and replay noise.

# Phase 2: contrastive verifier training
train XOV (frozen CLIP-ViT visual encoder + 2-layer MLP head, 256-d, GeLU activation)
  on pairs ((o_t, g), (o_t, g_pert)) with contrastive loss:
    L = -log[ exp(s(g)/tau) / (exp(s(g)/tau) + exp(s(g_pert)/tau) ) ]
  where s(g) = XOV(o_t, g),  tau = 0.07
  label weighting w = sigmoid(delta)       # larger variance-controlled outcome divergence → stronger contrast
  # Only pairs with delta > 0 (i.e., divergence exceeds replay noise) contribute to the loss;
  # pairs with delta = 0 are down-weighted to zero, preventing the verifier from being trained
  # to penalize correct alternative groundings.

# Phase 3: inference-time gating
given a fresh grounding output g*:
    v(g*) = XOV(o_t, g*)
    accept g* if v(g*) > 0.5, else re-ground via VISUALSKILL load_topic
```

**Why this design.** We chose *replay-based counterfactual execution* over *self-critique scoring* because self-critique reuses the same grounding representation and therefore reintroduces the circular trust path that VISUALSKILL and Agent S2 suffer from, accepting the cost that replay requires a sandboxed environment and logged transitions. We chose a *contrastive objective over a regression objective* because the contrastive pair isolates the effect of the grounding perturbation while holding `o_t` fixed, accepting that we need paired data and cannot train on unpaired steps. We chose a *frozen visual encoder with a shallow MLP head* over end-to-end fine-tuning because the verifier must not absorb the grounding module's representation (which would re-couple the two), accepting reduced capacity. We chose *outcome divergence as a soft label weight* over a hard binary label because environment transitions are noisy and a soft weight lets the verifier down-weight ambiguous replays, accepting that the label is a proxy for correctness rather than correctness itself. We chose *variance-controlled divergence with null-perturbation subtraction* over a single-replay delta because replay nondeterminism (loading spinners, async responses, session state) would otherwise inject noise directly into the verifier's decision boundary; this accepts K× replay cost (K=5) in exchange for an unbiased estimate of grounding causality. A domain expert would not describe XOV as a variant of Agent Q's self-critique or ASI's execution-based verification: Agent Q critiques trajectories using the agent's own policy, and ASI verifies induced programs by executing *the program itself*; XOV instead executes a *counterfactual perturbation* and trains on the *divergence between two outcome distributions*, which neither prior method constructs.

**Why it measures what we claim.** The quantity `delta = max(0, D(null_dist, pert_dist) - D_null)` measures *causal necessity of the grounding output for the observed outcome* because the only difference between the two replay distributions is the grounding perturbation, and the null-perturbation baseline subtracts replay-intrinsic stochasticity; this assumption fails when the environment is stochastic in a way not captured by the null replays (e.g., a loading spinner resolves differently across replays), in which case `delta` reflects residual environment noise rather than grounding causality. The quantity `s(g) = XOV(o_t, g)` measures *grounding correctness* because the contrastive loss forces `s` to rank `g` above `g_pert` whenever `g_pert` produces a divergent outcome (delta > 0), making `s` monotone in causal necessity; this assumption fails when a perturbed grounding is *also* correct (e.g., two valid elements of the same role), in which case `s` reflects *uniqueness* of the grounding rather than its correctness — we mitigate this by filtering perturbations to be semantically distinct and by zero-weighting pairs with delta = 0. The gating threshold `v(g*) > 0.5` measures *trustworthiness of the grounding output* because the verifier never conditions on the grounding module's internal representation, so its score is identifiable from that module; this assumption fails when the replay distribution drifts from the live environment (e.g., a new UI version), in which case `v` reflects *replay fidelity* rather than grounding correctness.

**Missing equivalence (D sub-block).** The claim that `s(g)` measures grounding correctness rests on assumption A: *outcome divergence is monotone in grounding incorrectness* — i.e., every incorrect grounding produces a divergent outcome (delta > 0) and every correct one does not (delta = 0). Failure mode F: two valid same-role targets yield divergence despite correct grounding, so delta > 0 for a correct `g`, and the contrastive loss would push `s(g)` below `s(g_pert)`. We bound F empirically via a same-role-ambiguity-filtered evaluation subset (experiment) and mitigate it in training via the semantic-distinctness filter and zero-weighting of delta = 0 pairs.

## Contribution

(1) XOV, a grounding verifier trained on a contrastive objective over counterfactual grounding perturbations whose label is the exogenous post-action outcome divergence, yielding a correctness signal statistically identifiable from the checked grounding module. (2) The design principle that replay-based counterfactual execution of a grounding perturbation — rather than self-critique or direct execution verification — is what breaks the self-validation circularity, with the soft outcome-divergence weighting as the mechanism that tolerates noisy environment transitions. (3) A perturbation operator over structured skill/feasibility representations (same-role element swap, feasibility flip, nearest-neighbor skill substitution) that generates contrastive pairs without retraining the base agent.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | OSWorld + WebArena live replay logs (N=5,000 steps) | Sandboxed replay enables counterfactual perturbation execution |
| Dataset (stress) | Synthetic gridworld with known grounding correctness | Ground-truth correctness for rank-correlation analysis |
| Primary metric | Success-rate delta on perturbable steps | Isolates grounding-caused failures from policy failures |
| Secondary metric | Spearman rank correlation between XOV score and true correctness | Quantifies monotonicity assumption A |
| Baseline | VISUALSKILL (self-consistent grounding) | Tests whether self-critique shares our causal signal |
| Baseline | ASI execution-based verification | Tests whether program-execution verification transfers to grounding |
| Baseline | Agent S2 unverified grounding | Tests whether gating adds value over strong generalist |
| Ablation-of-ours | XOV without outcome-divergence weighting (w=1) | Tests whether soft causal label drives the gain |
| Ablation-of-ours | XOV with single-replay delta (no variance control) | Tests whether null-perturbation subtraction matters |
| Ablation-of-ours | XOV with unrestricted perturbations (no semantic filter) | Tests whether same-role-ambiguity mitigation matters |

### Why this setup validates the claim

The central claim is that training a verifier on the *causal effect of a grounding output on the post-action state* yields a correctness signal that self-consistency and program-execution baselines cannot construct. VISUALSKILL tests the sub-claim that self-consistency is circular: if our verifier beats it, the gain must come from an exogenous signal the self-critique path cannot access. ASI tests whether executing *the program itself* is a sufficient substitute for executing a *counterfactual perturbation*; if XOV outperforms ASI, the divergence-between-two-outcomes construction is doing real work beyond mere execution. Agent S2 as an unverified strong generalist tests whether gating adds anything atop a competent grounding module, ruling out the possibility that our numbers are just a better base policy. The ablation — removing outcome-divergence weighting — is the sharpest test: it keeps the contrastive architecture but discards the causal label, so any gap between full XOV and the ablation is attributable specifically to the claim that *delta* measures causal necessity. The variance-control ablation tests whether null-perturbation subtraction is necessary to make delta an unbiased estimate of grounding causality rather than a mixture of causality and replay noise. The semantic-filter ablation tests whether the same-role-ambiguity mitigation is necessary to prevent the verifier from being trained to penalize correct alternative groundings. The primary metric, success-rate delta restricted to steps where a valid perturbation exists, is the right detector because it concentrates measurement exactly where the predicted mechanism fires; a uniform gain across all steps would be consistent with generic verifier capacity rather than the causal-signal story. The synthetic gridworld with known ground-truth correctness provides a controlled setting to compute Spearman rank correlation between XOV score and true correctness separately for deterministic vs. stochastic environments and for unique vs. same-role-ambiguous targets, directly quantifying where the monotonicity assumption A holds and where it breaks (failure mode F).

### Expected outcome and causal chain

**vs. VISUALSKILL** — On a case where the grounding module confidently selects a same-role but wrong element (e.g., a "Submit" button adjacent to a "Cancel" in a modal), VISUALSKILL accepts it because its self-critique reuses the same representation that produced the error and finds it internally consistent. Our method instead rejects it because the counterfactual replay of the swapped element produces a divergent post-action state (modal stays open), and the contrastive score was trained to rank the divergent perturbation below the logged grounding. We expect a noticeable gap concentrated on same-role-confusion steps, with parity on unambiguous single-candidate steps.

**vs. ASI** — On a case where the grounding output is syntactically executable but semantically mis-targeted (a valid skill id applied to the wrong element), ASI's execution-based verification passes it because the induced program runs without error — execution success is not outcome correctness. Our method instead flags it because the perturbation replay shows the environment state diverging from the logged transition, which ASI never constructs. We expect XOV to lead on mis-targeted-but-executable steps while matching ASI on genuinely broken executions, producing a characteristic subset-shaped gap rather than a uniform lift.

**vs. Agent S2** — On a case where the generalist grounds correctly but a downstream step compounds an earlier feasibility-flag error, Agent S2 propagates the mistake because it has no per-step trust signal. Our method instead gates each grounding at inference, catching the flipped feasibility flag when its replay divergence is high, so the error does not compound. We expect the gap to widen with episode length — modest at 15 steps, pronounced at 50 — which is the signature of per-step gating rather than a stronger base policy.

**vs. Ablation (no outcome-divergence weighting)** — On a case where the environment is stochastic (loading spinner resolves differently across replays), the ablation treats all contrastive pairs equally, so replay noise directly corrupts the decision boundary. Our method down-weights pairs with low variance-controlled delta, so the verifier learns only from grounding-caused divergence. We expect the ablation to underperform on stochastic-environment subsets, confirming that the soft causal label is doing real work.

**vs. Ablation (single-replay delta)** — On a case where replay nondeterminism is high (async network responses, session tokens), the single-replay delta conflates grounding causality with replay noise. Our method subtracts the null-perturbation baseline, so delta isolates grounding causality. We expect the single-replay ablation to show higher variance in verifier scores and lower success-rate delta on high-stochasticity steps.

**vs. Ablation (unrestricted perturbations)** — On a case where two valid same-role elements exist, the unrestricted ablation trains the verifier to penalize a correct alternative grounding because the perturbation produces divergence. Our method filters perturbations to be semantically distinct and zero-weights delta=0 pairs, so the verifier is not trained to be wrong on same-role-ambiguity steps. We expect the unrestricted ablation to underperform specifically on the same-role-ambiguity subset, confirming the necessity of the semantic filter.

### What would falsify this idea

If XOV's gain over VISUALSKILL and ASI is uniform across all steps rather than concentrated on same-role-confusion and mis-targeted-but-executable subsets, the causal-divergence mechanism is not what drives the improvement. Equally fatal: if the outcome-divergence ablation matches full XOV, then *delta* is not carrying the causal signal we claim, and the contrastive architecture alone explains the result. Additionally, if the variance-control ablation matches full XOV, then null-perturbation subtraction is unnecessary and the load-bearing assumption about replay nondeterminism is not actually binding.

## References

1. VISUALSKILL: Multimodal Skills for Computer-Use Agents
2. Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents
3. Inducing Programmatic Skills for Agentic Tasks
4. Agent S: An Open Agentic Framework that Uses Computers Like a Human
5. AgentStore: Scalable Integration of Heterogeneous Agents As Specialized Generalist Computer Assistant
6. Proposer-Agent-Evaluator(PAE): Autonomous Skill Discovery For Foundation Model Internet Agents
7. Agent Workflow Memory
8. Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents
