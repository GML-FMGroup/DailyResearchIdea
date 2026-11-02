# Identifiable Goal Routing for Asynchronous Brain-Cerebellum Control

## Motivation

Existing routed embodied runtimes such as REA dispatch on latent goals inferred from partial observations without checking whether those goals are identifiable, so the router may commit to a goal that is indistinguishable from another goal under the current observation history; this is a structural failure of the dispatch criterion, not a data-scarcity issue, because the mapping from observation histories to latent goals is many-to-one on the quotient by observationally equivalent states. The same residual gap recurs across the Do-as-I-can and GR00T N1 branches: each assumes the brain's inferred goal is a sufficient dispatch statistic, but partial observability makes that assumption false whenever two distinct goals generate identical observation histories.

## Key Insight

A latent goal is safe to dispatch on exactly when the observation-to-goal map is injective on the quotient of observationally equivalent states, so requiring the router to wait for an observation history that separates the goal from its observationally equivalent alternatives is a structural condition on the belief space rather than a confidence heuristic.

## Method

**What it is.** IG-REA (Identifiable-Goal REA) augments the REA runtime with a belief-space identifiability gate: before the router dispatches a policy for a latent goal inferred by the deliberative brain, IG-REA computes the quotient belief set of goals consistent with the current observation history and dispatches only if that set is a singleton. Inputs: observation history $h_t$, brain's goal posterior $p(g \mid h_t)$, policy success table $S$, and an empirical observation-likelihood table $\hat{p}(o \mid g)$ accumulated from executed episodes (or a held-out calibration set of 512 episodes). Output: a dispatch decision (goal $g$, policy $\pi$) or a deferral to the brain for an information-gathering action.

**How it works.**

```pseudocode
# Per control step t, after brain produces posterior over latent goals
# G = {g_1..g_K} finite goal vocabulary from route memory
# H = observation history buffer (last W=8 steps, fixed)
# Empirical likelihood table: for each g in G, counts n_g(o) over a discretized
# observation grid (256 bins per sensor channel), updated online from executed
# episodes; total count N_g = sum_o n_g(o). Minimum N_g = 64 before a goal is
# considered 'data-supported'.

1. Compute belief b_t(g) = p(g | h_t) from brain (normalized over G)
2. Compute equivalence classes on G under the empirical observation kernel:
   for each pair (g_i, g_j):
       if min(N_gi, N_gj) < 64:
           state_ij = UNKNOWN   # insufficient data to separate
       else:
           # empirical TV with sample-size-aware confidence bound
           rho_ij = TV( \hat{p}(. | g_i), \hat{p}(. | g_j) )
           # 95% TV confidence half-width from sample size (Weissman et al. bound):
           delta_ij = sqrt( (1/(2*N_gi)) * log(2/0.05) ) + sqrt( (1/(2*N_gj)) * log(2/0.05) )
           if rho_ij + delta_ij <= eps_obs:      state_ij = EQUIVALENT
           elif rho_ij - delta_ij > eps_obs:     state_ij = DISTINCT
           else:                                 state_ij = UNKNOWN
   # eps_obs = 0.05 (maximum tolerable observational aliasing)
   # Equivalence classes are the connected components of the EQUIVALENT graph;
   # any pair in UNKNOWN state is treated as potentially equivalent (conservative).
3. Identifiability test on quotient G/~ :
   let [g*] = equivalence class of argmax_g b_t(g)
   identifiable  iff  |[g*]| == 1  AND  no UNKNOWN pair involving g*  AND  b_t(g*) >= tau_bel   (tau_bel = 0.6)
4. If identifiable:
       dispatch pi = argmax_{pi} S[g*, pi]   # route memory lookup
   else:
       select info-gathering action a* = argmax_a  E[ H(b_{t+1}) - H(b_t) | a ]
       # one-step entropy reduction, closed form over G using the brain's forward model
       execute a* with fast policy, append observation, update empirical counts, goto 1
5. Boundary-triggered judge gate unchanged from REA: at execution boundaries,
   if route memory success for (g*, pi) < 0.5, consult brain for re-plan.
```

The equivalence-class computation uses the empirical observation-likelihood table $\hat{p}(o \mid g)$ accumulated from executed episodes (or a held-out calibration set of 512 episodes), not the brain's forward model. The TV confidence half-width $\delta_{ij}$ is computed from the sample sizes $N_{g_i}, N_{g_j}$ using the Weissman et al. (2003) bound at 95% confidence; this replaces the fixed $\epsilon_{obs}$ threshold with a sample-size-aware criterion that shrinks as data accumulates. The three-state classification (EQUIVALENT / DISTINCT / UNKNOWN) treats 'insufficient data to separate' as a third state distinct from 'equivalent' and 'distinct' that triggers information-gathering rather than dispatch. The information-gathering action in step 4 is selected from a fixed discrete set of exploratory primitives (look-left, look-right, move-forward-one-cell, grasp-probe) whose one-step entropy reduction is computed in closed form under the brain's forward model.

**Why this design.** We chose a quotient-space injectivity test over a learned confidence head because the latter (as in standard gating) cannot distinguish a goal that is merely low-probability from a goal that is observationally indistinguishable from the argmax — the former should still dispatch, the latter must not; the quotient test makes this distinction structural, accepting the cost that it requires the brain to expose a forward observation model. We chose empirical TV-equivalence with a sample-size-aware confidence bound over a learned similarity metric because TV is a proper metric on distributions and its confidence bound has a direct interpretation as the maximum tolerable observational aliasing given the available data, accepting the cost that the bound requires sufficient per-goal samples and that the observation grid must be discretized. We chose a one-step entropy-reduction information-gathering action over a multi-step POMDP planner because the dispatch decision is made at every control step and multi-step planning would violate REA's golden rule that no running policy waits on the brain; the cost is that one-step lookahead can be myopic when the goal posterior is multimodal. This design is not a variant of the standard REA router-plus-brain composition (the closest prior method) because REA's router dispatches on the brain's posterior directly; IG-REA inserts a quotient-space injectivity condition that REA does not have and that cannot be recovered from the posterior alone.

**Why it measures what we claim.** The quotient class size $|[g^*]|$ measures the identifiability of the latent goal because the assumption that the empirical observation-likelihood table $\hat{p}(o \mid g)$ is estimated from enough episodes (minimum $N_g = 64$ per goal) and that the TV confidence bound $\delta_{ij}$ correctly accounts for sampling error makes empirical TV-equivalence coincide with true observational equivalence; this assumption fails when a goal has fewer than 64 samples or when the observation grid is too coarse, in which case $|[g^*]|$ reflects sampling noise or discretization error rather than true aliasing. The belief threshold $\tau_{bel}$ measures the commitment of the brain to a goal because the assumption that the brain's posterior is normalized and non-degenerate over the goal vocabulary makes $b_t(g^*)$ a sufficient statistic for goal commitment; this assumption fails when the brain is overconfident on a single goal regardless of evidence, in which case $\tau_{bel}$ reflects prior strength rather than evidence. The one-step entropy reduction $E[H(b_{t+1}) - H(b_t) \mid a]$ measures the information value of an exploratory action because the assumption that the brain's forward model is the true data-generating process makes expected entropy reduction equal to expected information gain; this assumption fails when the forward model is biased, in which case the selected action reflects model bias rather than true information.

**(D) Assumption-and-failure-mode sub-block for information-gathering.** The one-step expected entropy reduction $E[H(b_{t+1}) - H(b_t) \mid a]$ measures true information gain $I(g; o \mid a, h)$ because the assumption A holds: the brain's forward model $p(o \mid g, h)$ equals the true observation kernel, so expected entropy reduction equals mutual information. This assumption fails when the forward model is biased (failure mode F), in which case the selected primitive can be anti-informative — it raises entropy under the true kernel — and the realized information gain is negative. We report realized (post-hoc) information gain of the chosen primitive versus the model-predicted gain on the multimodal subset to expose F.

## Contribution

['A belief-space identifiability gate for routed embodied runtimes that computes the quotient of the latent goal space by observational equivalence and dispatches only on singleton classes, replacing the unidentifiability-blind dispatch of REA.', 'A closed-form one-step entropy-reduction information-gathering action selector that lets the runtime actively disambiguate observationally equivalent goals instead of deferring indefinitely.', "A design principle: dispatch safety in asynchronous brain-cerebellum control is a property of the quotient belief space, not of the brain's posterior confidence, and can be enforced with a fixed threshold on total variation between goal-conditioned observation kernels."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Aliased-Goal Navigation Suite (paired goal-aliasing scenes) | Contains observationally ambiguous goal pairs to stress identifiability. |
| Dataset | Habitat/ProcTHOR navigation task with genuine perceptual aliasing (held-out calibration set of 512 episodes) | Tests aliasing failure mode on realistic benchmark. |
| Primary metric | Task success rate on aliased-goal episodes | Directly measures dispatch correctness under observational aliasing. |
| Secondary metric | Fraction of control steps where posterior-argmax dispatch is provably wrong (true goal ≠ argmax and TV ≤ eps_obs) | Quantifies prevalence of aliasing failure mode. |
| Secondary metric | Per-control-step wall-clock cost of quotient computation (ms) vs K and M | Verifies runtime compatibility with asynchronous control. |
| Secondary metric | Realized vs predicted information gain of chosen primitive on multimodal subset | Exposes forward-model bias (failure mode F). |
| Baseline | Standard REA router (posterior-argmax dispatch) | Tests whether posterior alone suffices without quotient gate. |
| Baseline | Learned confidence-head gating | Tests whether confidence gating can separate low-prob from aliased goals. |
| Baseline | Oracle-deferral upper bound (brain consulted every ambiguous step) | Establishes ceiling for information-gathering benefit. |
| Ablation-of-ours | IG-REA minus quotient test (tau_bel only) | Isolates contribution of equivalence-class gate from belief threshold. |
| Ablation-of-ours | IG-REA with fixed eps_obs=0.05 and M=32 Monte-Carlo TV (original design) | Tests whether sample-size-aware empirical TV improves over model-internal TV. |
| Ablation-of-ours | IG-REA with forward-model noise injection (sweep σ from 0 to 0.5) | Measures degradation of |[g*]| and success rate vs miscalibration. |

### Why this setup validates the claim

The central claim is that a quotient-space injectivity test catches a failure mode posterior-based dispatch cannot: goals that are observationally indistinguishable from the argmax but not low-probability. The Aliased-Goal Navigation Suite is constructed with paired goals whose observation kernels are near-identical at specific histories, so the failure mode is present by design rather than incidental; this makes the test falsifiable rather than a general benchmark sweep. The Habitat/ProcTHOR task with genuine perceptual aliasing tests whether the failure mode occurs outside the constructed suite, and the fraction-of-wrong-steps metric quantifies its prevalence. Standard REA tests sub-claim one (posterior sufficiency fails), the confidence-head baseline tests sub-claim two (a learned scalar cannot separate aliasing from low probability), and the oracle-deferral bound tells us whether any residual gap is due to the one-step myopia of our information-gathering rather than the gate itself. The ablation without the quotient test isolates the gate from the belief threshold, so a gain attributable to tau_bel alone would not be mistaken for the quotient mechanism. The ablation with fixed eps_obs and M=32 Monte-Carlo TV tests whether the sample-size-aware empirical TV (with confidence bound) is necessary to avoid Monte-Carlo noise dominating the equivalence relation. The forward-model noise injection sweep measures how |[g*]| and success rate degrade as the brain's forward model becomes miscalibrated, grounding the identifiability claim in a measured quantity. The realized-vs-predicted information gain metric on the multimodal subset exposes failure mode F (anti-informative primitives due to forward-model bias). Success rate on aliased episodes is the right primary metric because it is exactly the quantity the gate is supposed to protect: dispatching the wrong policy when goals are aliased. A method that improves overall success without concentrating gains on aliased episodes would not support the claim.

### Expected outcome and causal chain

**vs. Standard REA router (posterior-argmax dispatch)** — On a case where two goals share near-identical observation kernels at the current history and the brain's posterior splits roughly 0.55/0.45, the baseline dispatches the argmax goal's policy because it reads the posterior directly and has no notion of observational equivalence, so it commits to a goal the sensors cannot confirm and executes a policy that is wrong under the true goal. Our method instead computes the quotient class of the argmax using the empirical observation-likelihood table with sample-size-aware confidence bounds, finds a second goal within TV ≤ eps_obs (or UNKNOWN state), declares the class non-singleton, and defers to an information-gathering primitive that separates the pair before dispatch. We expect a pronounced success-rate gap concentrated on aliased episodes (large, e.g. tens of points) with near-parity on unaliased episodes. On the Habitat/ProcTHOR task, we expect the fraction of control steps where posterior-argmax dispatch is provably wrong to be non-negligible (e.g., >5%), confirming the failure mode is not an artifact of the constructed suite.

**vs. Learned confidence-head gating** — On a case where the true goal is genuinely low-probability (0.3) but observationally distinct from the argmax, the baseline suppresses dispatch because its confidence head conflates low posterior with unidentifiability, so it stalls or over-explores on a goal that was actually resolvable. Our method instead finds the argmax's equivalence class to be a singleton (using empirical TV with confidence bound) and, provided tau_bel is met, dispatches immediately, because the quotient test is structural and does not penalize low probability per se. We expect the confidence-head baseline to show excess deferral latency and lower success on the low-probability-but-distinct subset, while our method matches the oracle-deferral bound there and beats the baseline on aliased episodes.

**vs. Oracle-deferral upper bound (brain consulted every ambiguous step)** — On a case where the goal posterior is multimodal across three or more aliased goals, the oracle bound resolves it by consulting the brain at every step, so it pays no dispatch error but incurs maximal brain-load and latency. Our method instead uses one-step entropy reduction over a fixed primitive set, which can be myopic when the separating observation requires two steps, so we expect our method to sit below the oracle bound on the multimodal subset but close to it on the two-goal aliased subset, with a clear brain-consultation-count advantage that grows with episode length. On the multimodal subset, we report realized vs predicted information gain of the chosen primitive; if forward-model bias causes anti-informative primitives, realized gain will be negative while predicted gain is positive, exposing failure mode F.

**vs. IG-REA with fixed eps_obs=0.05 and M=32 Monte-Carlo TV (original design)** — On a case where the brain's forward model is misspecified for rare goals, the original design's Monte-Carlo TV estimate has sampling error ~0.25, five times the eps_obs threshold, so the equivalence relation is dominated by noise and the gate either over-defers or under-defers. Our method instead uses empirical counts with a sample-size-aware confidence bound, which shrinks as data accumulates and treats insufficient data as UNKNOWN (triggering information-gathering). We expect the original design to show higher variance in |[g*]| and lower success on aliased episodes, while our method is more stable and closer to the oracle bound.

**vs. IG-REA with forward-model noise injection (sweep σ)** — As we inject controlled noise into the brain's forward model (σ from 0 to 0.5), we expect |[g*]| to become noisier and success rate to degrade, with a steeper drop for the original Monte-Carlo TV design than for our empirical TV design. This measures the calibration gap and grounds the identifiability claim in a measured quantity.

### What would falsify this idea

If IG-REA's success-rate gain over standard REA is uniform across aliased and unaliased episodes rather than concentrated on the aliased subset, the quotient gate is not doing the work we claim and the central mechanism is wrong. Likewise, if the ablation without the quotient test matches full IG-REA on aliased episodes, the equivalence-class computation is inert and the effect is attributable to the belief threshold alone. Additionally, if the realized information gain of the chosen primitive is consistently negative on the multimodal subset (failure mode F dominates), the one-step entropy reduction is anti-informative and the information-gathering mechanism is flawed.

## References

1. REA: A Routed Embodied Agent Runtime for Asynchronous Brain-Cerebellum Control
