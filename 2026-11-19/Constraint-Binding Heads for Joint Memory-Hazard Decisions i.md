# Constraint-Binding Heads for Joint Memory-Hazard Decisions in Personalized Vision-Language Assistants

## Motivation

Personalized vision-language assistants must combine retrieved personal memory with visual hazard recognition, yet PReMind's two-phase RL optimizes memory-grounding and hazard-response through separate reward channels, leaving joint success on interdependency cases at only 28.6%. The root cause is structural: the policy has no shared internal variable that both memory tokens and hazard representations must agree on, so the model can satisfy one requirement while failing the other without any penalty for the conflict. This compositional-consistency gap recurs across the PVIT lineage, where personalized instruction tuning (E1) and retrieval-augmented personalization (E7) similarly add personalization capability without tying it to safety-relevant visual decisions.

## Key Insight

A single differentiable constraint-binding variable that both the memory-token set and the hazard representation must jointly determine makes memory-grounding and hazard-response adequacy two projections of the same internal state, so conflicting projections become detectable and penalizable rather than invisible to separate reward channels.

## Method

```pseudocode
# Constraint-Binding PReMind (CB-PReMind)
Inputs: image I, query q, retrieved memory token set M = {m_1..m_k}, base VLM policy pi_theta

1. Encode: h_vis = VisionEncoder(I); h_mem = MemoryEncoder(M); h_q = TextEncoder(q)
2. Hazard head: z_haz = MLP_haz([h_vis; h_q])            # 2-layer MLP, hidden=256, GeLU, output dim d=64
3. Memory-grounding head: z_mem = MLP_mem([h_mem; h_q])   # 2-layer MLP, hidden=256, GeLU, output dim d=64
4. Constraint-binding head (CBH):
     b = sigma( W_b [z_haz ; z_mem ; z_haz ⊙ z_mem] + c )   # W_b: (3d x d), d=64; b in [0,1]^d
5. Response generation: y ~ pi_theta(· | I, q, M, b)      # b is prepended as a soft prompt (projected to 1 token via linear layer)
6. Consistency loss: L_cons = || proj_haz(b) - z_haz ||^2 + || proj_mem(b) - z_mem ||^2
                       + lambda * max(0, margin - cos(proj_haz(b), proj_mem(b)))   # lambda=0.5, margin=0.3
     where proj_haz and proj_mem are linear maps from R^d to R^d (d=64), initialized as identity.
     Rank regularizer: L_rank = - (logdet(proj_haz^T proj_haz + eps*I) + logdet(proj_mem^T proj_mem + eps*I)) / d, eps=1e-4, added to L_cons with weight gamma=0.01.
7. RL phase (two-phase as in PReMind): on-demand and forced-call rollouts;
     reward r = r_task + beta * (-L_cons - gamma * L_rank)   # beta=0.1, r_task = PReMind joint-success reward
8. Update theta and CBH parameters by PPO with reward r.
```

**Why this design.** We chose a single binding variable `b` over two separate heads (as in PReMind's separate reward channels) because the interdependency task requires a shared sufficient statistic: two independent heads can each be locally correct while disagreeing, which is exactly the 28.6% joint-success failure. We chose a projection-based consistency loss over a KL-divergence between heads because projections are differentiable end-to-end and impose a geometric constraint (both heads must lie in the same low-dimensional subspace) rather than merely matching distributions, accepting the cost that the projection matrices add parameters and can collapse if not regularized; the rank regularizer `L_rank` (weight gamma=0.01) explicitly prevents this collapse by penalizing rank-deficient `proj_haz` and `proj_mem`. We chose to inject `b` as a soft prompt into the policy rather than as a hard gate because a hard gate would make the policy non-differentiable through the binding step, accepting the cost that the policy can partially ignore `b`; the consistency loss counteracts this by penalizing internal disagreement even when the final answer is correct. A domain expert would not describe this as a variant of PReMind because PReMind's two-phase RL has no mechanism that forces memory and hazard representations to share a variable; the binding head is a new architectural invariant, not a reward-shaping tweak. Nor is it a variant of standard multi-task heads (e.g., shared-trunk models) because the consistency loss explicitly penalizes cross-head conflict rather than merely sharing parameters.

**Why it measures what we claim.** The consistency loss `L_cons` measures compositional consistency between memory-grounding and hazard-response because it assumes that if both heads are projections of the same binding variable `b`, then their disagreement in the projected subspace is equivalent to a conflict in the joint decision; this assumption fails when the projection matrices `proj_haz` and `proj_mem` are rank-deficient, in which case `L_cons` reflects projection collapse rather than true head conflict. The binding variable `b` measures the joint decision state because it is the only variable both heads must reconstruct, under the assumption that a single low-dimensional variable (d=64) can capture the interdependency between personal memory and hazard; this assumption fails when the interdependency is high-rank (e.g., many-to-many memory-hazard combinations), in which case `b` under-represents the joint state and the loss penalizes benign disagreements. The reward term `-beta * L_cons` measures joint-success pressure because it assumes that reducing internal conflict during RL rollouts translates to higher joint success on the interdependency set; this assumption fails when the task reward `r_task` is sparse, in which case the consistency term dominates and the policy optimizes internal agreement without improving external correctness. To test this assumption directly, we add a rank regularizer `L_rank` (gamma=0.01) that keeps `proj_haz` and `proj_mem` full-rank, and we report joint success stratified by interdependency rank (see experiment).

## Contribution

['A constraint-binding head (CBH) architecture that produces a single differentiable joint decision variable from memory-token and hazard representations, with a projection-based consistency loss that penalizes cross-head disagreement during RL.', 'A design principle: compositional consistency in personalized vision-language assistants can be enforced as an architectural invariant (shared binding variable) rather than as an additional reward term, converting separately-optimized objectives into a single policy surface.', 'An integration of CBH into the PReMind two-phase RL framework, yielding a training procedure that jointly optimizes memory-grounding, hazard-response, and their consistency without additional data collection.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | PReMind-Bench interdependency set | Directly targets memory-hazard joint success |
| Primary metric | Joint success rate on interdependency cases | Single number capturing both requirements together |
| Baseline | PReMind 4B (two-phase RL) | Tests whether binding beats separate reward channels |
| Baseline | Base VLM without personalization | Tests whether memory grounding is needed at all |
| Baseline | Shared-trunk multi-task heads | Tests whether consistency loss beats shared parameters |
| Ablation-of-ours | CB-PReMind without L_cons | Isolates the consistency loss contribution |
| Ablation-of-ours | CB-PReMind without L_rank | Tests whether rank regularizer prevents projection collapse |
| Diagnostic | Effective rank and condition number of proj_haz, proj_mem | Rules out projection collapse and benign-disagreement penalization |
| Stratification | Joint success by interdependency rank (low/medium/high) | Tests sufficiency of b for high-rank interdependencies |

### Why this setup validates the claim

The central claim is that a single binding variable shared by hazard and memory heads, reinforced by a projection-consistency loss, raises joint success on cases where personal memory and visible hazard interact. PReMind-Bench is the right dataset because it already separates memory access, grounding, abstention, and hazard response, and reports joint success on interdependency cases — exactly the regime where two independent heads can each be locally correct yet disagree. The primary metric, joint success, is the only metric that cannot be gamed by solving one requirement alone; a method that improves hazard response but loses memory grounding will show flat joint success. PReMind 4B tests the sub-claim that binding beats separate reward channels: if our gain over PReMind is concentrated on interdependency cases and absent on single-requirement subsets, the binding mechanism is doing the work. The shared-trunk multi-task baseline tests whether the consistency loss adds anything beyond parameter sharing, and the no-consistency ablation tests whether the loss, rather than the extra parameters, produces the effect. The no-rank-regularizer ablation and the effective-rank diagnostic directly test the load-bearing assumption that `proj_haz` and `proj_mem` remain full-rank and that `b` is a sufficient statistic; if the rank regularizer is necessary for the gain, or if joint success degrades on high-rank interdependency cases, the sufficiency assumption is violated. Together these form a falsifiable test: the predicted pattern is a joint-success gain over PReMind that vanishes when L_cons is removed, that does not appear on non-interdependent subsets, and that is maintained across interdependency ranks only when L_rank is active.

### Expected outcome and causal chain

**vs. PReMind 4B (two-phase RL)** — On a case where a user with a peanut allergy is shown a dish containing peanuts, PReMind's separate memory and hazard reward channels can each fire correctly while the policy still recommends the dish, because nothing forces the two channels to agree before generation. Our binding head produces a single variable b that both heads must reconstruct, so disagreement is penalized internally before the response is emitted. We expect a clear joint-success gain on interdependency cases, roughly doubling the base rate, with near-parity on single-requirement subsets where no binding is needed. The gain should be largest on low- and medium-rank interdependency cases and may attenuate on high-rank cases, which we will report explicitly.

**vs. Base VLM without personalization** — On a case where a user's mobility constraint is stored in memory but the scene shows a step the base VLM would describe neutrally, the base model produces generic scene description because it never retrieves or grounds the memory token. Our method grounds memory into z_mem and binds it with the hazard representation, so the response conditions on both. We expect a large gap on memory-dependent hazard cases and abstention behavior, but little difference on pure hazard-recognition items that need no personal context.

**vs. Shared-trunk multi-task heads** — On a case where memory says "allergic to shellfish" and the hazard head flags a shrimp dish while the memory head flags a different attribute, shared parameters let both heads drift without explicit conflict penalty, so the policy can output a locally coherent but jointly wrong answer. Our consistency loss explicitly penalizes cross-head conflict in the projected subspace, forcing alignment. We expect our method to outperform shared-trunk heads specifically on the interdependency subset, with parity elsewhere — the diagnostic signature that the consistency constraint, not parameter sharing, drives the gain.

**vs. CB-PReMind without L_cons** — Removing the consistency loss leaves the binding head untrained to align projections, so z_haz and z_mem can diverge even when b is present. We expect joint success to drop to near PReMind levels on interdependency cases, confirming that L_cons is the active ingredient.

**vs. CB-PReMind without L_rank** — Without the rank regularizer, proj_haz and proj_mem may become rank-deficient, allowing L_cons to be minimized by degenerate agreement (both projections collapsing to a constant). We expect the effective rank of the projection matrices to drop below 10 (out of 64) and joint success on high-rank interdependency cases to fall, demonstrating that the sufficiency assumption fails without regularization.

### What would falsify this idea

If CB-PReMind's joint-success gain over PReMind is uniform across interdependency and non-interdependency subsets, or if removing L_cons leaves joint success unchanged, then the binding variable and consistency loss are not the causal mechanism and the central claim is wrong. Additionally, if the effective rank of proj_haz and proj_mem collapses without L_rank and joint success on high-rank interdependency cases does not degrade, then the sufficiency assumption is not load-bearing.

## References

1. PReMind: When Personal Memory Meets Visual Hazards
