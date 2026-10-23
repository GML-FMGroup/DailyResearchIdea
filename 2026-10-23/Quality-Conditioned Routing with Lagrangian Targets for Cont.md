# Quality-Conditioned Routing with Lagrangian Targets for Controllable Cost-Quality Frontiers

## Motivation

Existing learned routers such as RouteWeaver's COMET-GRPO optimize routing policies purely from outcome rewards, so the resulting policy exposes a single operating point and offers no inference-time knob for the cost-quality trade-off; the structural reason is that outcome-only RL collapses the entire Pareto frontier into one reward-weighted policy, making monotone consistency under a quality target unestablishable. Hybrid LLM shows that a tunable quality level is achievable for two-model difficulty routing, but this controllability has never been transferred to a multi-mode, multi-model RL router, where the mode-selection and execution decisions interact and cannot be thresholded by a single difficulty score. The field therefore lacks a router whose quality target is an explicit input to the policy rather than an emergent byproduct of reward shaping.

## Key Insight

Conditioning the routing policy on a scalar quality target and training it with a Lagrangian-penalized constrained reward makes the target a monotone sufficient statistic of the policy's cost-quality operating point, so sweeping it at inference traces the frontier by construction rather than by post-hoc calibration.

## Method

**What it is.** QCR (Quality-Conditioned Router) is a unified LLM routing policy that takes a query and a continuous quality target τ∈[0,1] as inputs and outputs a routing decision (mode ∈ {single, multi, agentic} and model within mode), trained with a target-conditioned COMET-GRPO objective whose reward penalizes cost subject to a Lagrangian constraint that enforces the target quality. The load-bearing assumption is stated explicitly: **Assumption A (τ-sufficiency):** a single scalar τ, fed as a policy input and enforced by a learned monotone multiplier head λ_θ(τ) via dual ascent, is a sufficient statistic that monotonically orders the policy's realized (quality, cost) operating points across all queries and modes — i.e., the constrained optimum is a monotone function of τ and the dual variable tracks it. This assumption is operationalized and tested in the experiment section; it can fail when the quality-cost trade-off is non-convex (discrete mode jumps), when the constraint is inactive (low τ) or infeasible (high τ, ceiling saturation), or when dual ascent oscillates.

**How it works.**
```pseudocode
Input: query q, target τ ~ U(0,1) sampled per rollout
Policy π_θ(mode, model | q, τ)  # τ concatenated as a scalar token embedding

# Rollout and outcome
 traj = rollout(π_θ, q, τ)            # mode, model calls, final answer
 quality = judge_score(traj.answer)   # held-out verifier, in [0,1]
 cost    = normalized_token_cost(traj)

# Target-conditioned constrained reward
 r = -cost + λ_θ(τ) · (quality - τ)   # λ_θ(τ) is a learned multiplier head

# COMET-GRPO decomposition (mode vs. execution credit)
 A_mode, A_exec = decompose_advantage(r, traj)   # per RouteWeaver
 update π_θ with GRPO on A_mode, A_exec

# Lagrangian multiplier update (dual ascent, per τ-bin)
 λ_θ(τ) ← λ_θ(τ) + η · (τ - E[quality | τ])      # η = 0.05, 8 τ-bins

# Inference: sweep τ, no retraining
for τ in grid: decision = argmax π_θ(· | q, τ)
```
The multiplier head λ_θ(τ) is a 2-layer MLP with hidden size 64, GeLU activation, and softplus output, constrained to be monotone in τ by using positive weights (e.g., weight normalization with non-negative scaling) so higher targets automatically command higher quality enforcement; the dual update runs every 50 policy steps on 8 τ-bins (bins: [0,0.125), [0.125,0.25), …, [0.875,1.0]) to keep the constraint tight without destabilizing GRPO. The policy π_θ is a 2-layer MLP with hidden size 256 and GeLU activation that takes as input the concatenation of a frozen query embedding (e.g., from a sentence encoder) and the scalar τ (embedded via a linear layer to 16 dimensions), and outputs a categorical distribution over the 6 routing actions (2 modes × 3 models). Training uses GRPO with group size 8, learning rate 1e-4, and 10k rollouts. The judge_score is a held-out verifier (e.g., a strong LLM) that outputs a scalar in [0,1]; its mode-unbiasedness is assumed and tested in the experiment.

**Why this design.** We chose an explicit continuous τ conditioning token over Hybrid-LLM-style post-hoc threshold tuning because a threshold can only reorder a fixed difficulty score, whereas τ inside the policy lets mode selection (single vs. agentic) itself respond to the target, accepting the cost that the policy must see a distribution of τ during training. We chose a learned monotone multiplier head λ_θ(τ) over a single global Lagrange multiplier because a global λ produces one operating point and cannot trace a frontier, accepting the cost of a slightly larger dual optimization. We chose to keep RouteWeaver's COMET-GRPO advantage decomposition rather than replace it with vanilla PPO because mode-vs-execution credit assignment is orthogonal to target conditioning and removing it would confound the controllability claim, accepting the cost of a more complex training loop. A domain expert would not describe this as 'Hybrid LLM + RouteWeaver': Hybrid LLM tunes a threshold on a difficulty classifier and RouteWeaver has no target input at all, whereas QCR makes τ a policy input with a dual-ascent constraint that changes the mode-selection distribution, which neither prior method does.

**Why it measures what we claim.** The computational quantity `E[quality | τ]` measured by the held-out judge over τ-conditioned rollouts measures the policy's realized operating point at target τ because the judge score is assumed to be a monotone proxy for true answer quality; this assumption fails when the judge is systematically lenient on a mode (e.g., rewarding verbose agentic traces), in which case `E[quality | τ]` reflects judge bias rather than true quality and the frontier shifts. The multiplier `λ_θ(τ)` measures the marginal cost of quality at target τ because dual ascent on the constraint `quality ≥ τ` yields the shadow price under the assumption that the constraint is active at the optimum; this fails when τ exceeds the policy's achievable quality ceiling, in which case λ saturates and no longer reflects a marginal trade-off. The monotonicity of the τ-sweep measures controllability because conditioning on τ with a monotone multiplier head is assumed to order policies by target; this fails when mode-selection credit `A_mode` dominates and the policy ignores τ, which we detect by checking rank correlation between τ and realized quality.

## Contribution

(1) QCR, a quality-conditioned routing policy that augments RouteWeaver's action space with a continuous target τ and a monotone Lagrangian multiplier head trained by dual ascent inside COMET-GRPO. (2) A design principle: target conditioning plus a per-τ shadow price yields a monotone quality-cost frontier at inference without retraining, whereas outcome-only RL collapses to a single operating point. (3) A τ-sweep evaluation protocol that measures frontier monotonicity (rank correlation between τ and realized quality/cost) as a controllability metric for routers.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Mixed suite: math (GSM8K), coding (HumanEval), knowledge (MMLU), reasoning (BBH), recall (TriviaQA) | Matches RouteWeaver coverage; spans difficulty regimes |
| Primary metric | Quality-cost frontier AUC over τ-sweep (τ grid: 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0) | Captures controllability, not single operating point |
| Secondary metric | Spearman rank correlation between τ and realized quality (per query and aggregate) | Directly tests monotonicity (Assumption A) |
| Secondary metric | Expected utility under user-specified cost-quality utility U = quality - w·cost for w ∈ {0.1, 0.5, 1.0} | Tests deployment value beyond frontier AUC |
| Baseline | Hybrid LLM (threshold router) | Tests target-conditioning vs post-hoc threshold |
| Baseline | RouteWeaver (mode routing, no τ) | Tests whether τ input adds controllability |
| Baseline | Always-large-model | Tests cost savings at matched quality |
| Ablation-of-ours | QCR with global λ (no τ head) | Isolates learned monotone multiplier contribution |
| Ablation-of-ours | QCR with τ input but fixed λ schedule (no dual ascent) | Tests whether dual ascent is necessary for monotonicity |
| Diagnostic | Judge-bias audit: 200 samples stratified by mode, human or stronger-model audit | Tests Assumption A (judge mode-unbiasedness) |
| Diagnostic | τ-token gradient/attention attribution: average gradient magnitude of τ embedding w.r.t. mode logits | Tests whether A_mode dominates τ signal |

### Why this setup validates the claim

The central claim is that making τ a policy input with a learned monotone multiplier head traces a controllable quality-cost frontier that neither a post-hoc threshold (Hybrid LLM) nor a target-free mode router (RouteWeaver) can produce. The frontier-AUC metric is the right detector because it rewards both placement (high quality at high τ, low cost at low τ) and ordering (monotone quality in τ); a single operating-point metric would be blind to the controllability claim. Hybrid LLM tests whether thresholding a fixed difficulty score suffices: if it matched QCR's frontier, τ-inside-policy would be unnecessary. RouteWeaver tests whether mode selection itself must respond to τ: if its frontier is flat in τ, the target-conditioning sub-claim is unsupported. The global-λ ablation isolates the multiplier head: if it traces the frontier equally well, the monotone head is decorative. The fixed-λ ablation tests whether dual ascent is necessary for monotonicity, addressing the non-convex dual concern. The judge-bias audit directly tests the assumption that the judge is mode-unbiased, which is load-bearing for interpreting E[quality|τ] as true operating point. The τ-attribution diagnostic measures whether the policy actually uses τ or ignores it in favor of mode-selection credit. Together these contrasts form a falsifiable test — each baseline removes exactly one mechanism, so a gain that survives all three is attributable to the claimed design.

### Expected outcome and causal chain

**vs. Hybrid LLM** — On a query where the difficulty classifier is ambiguous but a high τ demands extra verification, Hybrid LLM produces a single-model answer with no verification because its threshold reorders a fixed score and cannot promote single to agentic. Our method instead escalates to multi/agentic mode because τ is a policy input that shifts mode-selection mass upward. We expect a noticeable quality gap at high τ on hard subsets, with parity at low τ where the threshold already picks the cheap model.

**vs. RouteWeaver** — On a query where the router's difficulty score is mid-range and the user wants either cheap or thorough, RouteWeaver produces the same routing regardless of stated need because it has no target input. Our method instead splits its decisions across τ, sending the same query to single at low τ and agentic at high τ. We expect RouteWeaver's frontier to be nearly flat in τ (low rank correlation) while QCR's realized quality rises monotonically with τ.

**vs. Always-large-model** — On an easy query, always-large produces a correct but expensive answer because it ignores cost entirely. Our method instead routes to the small model at low τ because the cost penalty dominates when the quality constraint is slack. We expect comparable quality at high τ but substantially lower token cost at low τ, yielding a frontier that dominates the large-model point.

**vs. QCR with global λ** — On a τ-sweep, the global-λ ablation produces one operating point repeated across τ because a single multiplier enforces one constraint level. Our method instead moves along the frontier because λ_θ(τ) varies monotonically per τ-bin. We expect the ablation to show a collapsed frontier with high variance in realized quality across τ, while QCR shows a tight monotone curve.

**vs. QCR with fixed λ schedule** — On a τ-sweep, the fixed-λ ablation may produce a non-monotone frontier if the schedule is not well-calibrated to the query distribution, because without dual ascent the constraint may be inactive or infeasible. Our method instead adapts λ per τ-bin via dual ascent, which we expect to yield tighter monotonicity (higher Spearman correlation) and higher frontier AUC, especially at high τ where the constraint is more likely to bind.

### What would falsify this idea

If QCR's realized quality is not monotone in τ (Spearman rank correlation near zero or negative) or its frontier AUC is no better than Hybrid LLM and the global-λ ablation, the controllability claim is wrong. A gain that is uniform across all τ-bins rather than concentrated at high τ and on hard subsets would also indicate the effect is not coming from τ-conditioning. Additionally, if the judge-bias audit reveals systematic mode-dependent leniency that explains the frontier shift, or if the τ-attribution diagnostic shows near-zero gradient of τ on mode logits, the sufficient-statistic assumption is unsupported.

## References

1. RouteWeaver: Weaving Mode Selection and Execution into Unified LLM Routing
2. Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing
3. ReTool: Reinforcement Learning for Strategic Tool Use in LLMs
