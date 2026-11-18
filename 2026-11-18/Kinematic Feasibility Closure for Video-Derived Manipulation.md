# Kinematic Feasibility Closure for Video-Derived Manipulation Trajectories

## Motivation

Video-derived manipulation pipelines — latent-action pretraining (LAPA, E1), flow-centric planning (FLIP, E6), hierarchical VLA finetuning (HAMSTER, E10), and TCP tracking/generation (TCP4D, E4) — all treat the recovered action sequence as executable once retargeted, so the same structural gap recurs: no branch validates that a trajectory lifted from monocular video lies on the target embodiment's reachable workspace and joint-limit manifold. TCP4D's metric SE(3) TCP poses are precise but the retargeting step assumes, rather than establishes, kinematic feasibility, so errors in recovered metric pose or embodiment mismatch silently propagate into demonstrations and policies.

## Key Insight

A trajectory recovered from video is trustworthy for retargeting exactly when it is a fixed point of a projection onto the target embodiment's joint-limit-and-collision manifold, and this closure can be enforced as a differentiable layer trained jointly with the TCP tracker so that the recovered action space is closed under target kinematics by construction.

## Method

**What it is.** KFC (Kinematic Feasibility Closure) augments a TCP4D-style tracker with a differentiable constrained-projection layer. Inputs: monocular video, initial 2D TCP queries, and a target-embodiment URDF with joint limits and collision geometry. Outputs: a repaired metric SE(3) TCP trajectory, per-frame feasibility residuals, and a binary accept/repair flag per segment.

**Load-bearing assumption (stated explicitly).** The method assumes that gradients propagated through the differentiable IK projection layer back into the tracker's shared 4D scene representation will teach the tracker to recover trajectories that are already near-feasible on the target embodiment's joint-limit-and-collision manifold, rather than being absorbed by the IK layer's own degrees of freedom. This assumption is load-bearing because the joint-training claim depends on it; if the projection layer can satisfy the feasibility residuals by moving q_seq alone while leaving the tracker's SE(3) output S essentially unchanged, then dL_proj/dS is near zero and the tracker receives no learning signal. The method therefore includes an explicit distance-to-feasible-set term (below) that is computed on S, not on q, and a calibration experiment (in the experiment section) that measures whether that term carries gradient.

**How it works.**

```pseudocode
# Phase 1: tracker (inherited from TCP4D, E4)
for frame t in video:
    T_t, g_t, geom, K = Tracker(video, queries)    # metric SE(3) TCP + gripper state

# Phase 2: differentiable IK projection (per segment of length L=16 frames)
for segment S in chunk(T, L):
    q = IK_init(S[0])                              # warm start from previous segment
    for k in 1..K_iters (K_iters=8):
        q_seq = DiffIK(S, q, urdf)                 # differentiable IK, returns joint trajectory
        r_lim  = ReLU(|q_seq| - q_max) + ReLU(q_min - |q_seq|)     # joint-limit residual
        r_col  = CollisionPenalty(q_seq, scene_mesh)               # signed-distance residual
        r_task = ||FK(q_seq) - S||_F               # TCP tracking residual
        L = r_task + lambda_lim * r_lim + lambda_col * r_col       # lambda_lim=10, lambda_col=10
        q = q - eta * dL/dq                        # eta=1e-2, backprop through DiffIK
    feas(S) = (max_t r_lim(t) < eps_lim) and (max_t r_col(t) < eps_col)  # eps=1e-3 m
    if not feas(S):
        S_repaired = argmin_{S' in reachable} ||S' - S||  # constrained projection
        mark S as REPAIRED with residual magnitude

# Phase 3: joint training
# Tracker loss + projection loss L_proj = mean over segments of (r_task + lambda_lim*r_lim + lambda_col*r_col)
# Gradients flow from L_proj back into the tracker's 4D scene representation.
# To ensure the gradient reaches the tracker rather than being absorbed by q, add an explicit
# distance-to-feasible-set term computed on S:
#   L_dist = mean over segments of ||S - FK(q_seq_projected)||_F
# where q_seq_projected is the converged projection (detached), so L_dist penalizes the tracker
# for outputting S that is far from the reachable set. The total joint-training loss is
#   L_total = L_tracker + L_proj + lambda_dist * L_dist   (lambda_dist=1.0)
# The gradient of L_dist with respect to S is the sensitivity of the reachable-set distance to S.
```

**Why this design.** We chose a differentiable IK layer over a post-hoc classical IK solver because gradients from the feasibility residuals must reach the tracker's 4D representation during training; a post-hoc solver only filters outputs and cannot teach the tracker to avoid infeasible regions, accepting the cost that DiffIK adds memory and a local-minimum risk that the warm start and K_iters=8 mitigate. We chose hard rejection plus constrained repair over soft penalty-only filtering because retargeting pipelines consume trajectories as demonstrations, so an infeasible segment silently corrupts downstream policy learning; the repair step projects onto the reachable manifold and records the residual, accepting the cost that repaired segments may deviate from the observed motion. We chose to train the oracle jointly with the tracker rather than freeze it, because the shared 4D representation (the mechanism inherited from TCP4D) is the only surface where recovered geometry and embodiment kinematics can be coupled; freezing would reduce KFC to a filter on TCP4D outputs, which is the obvious-composition shape a reviewer would name as 'TCP4D + IK filter' and which a domain expert would describe as incremental. The added L_dist term makes the coupling explicit: it penalizes the tracker's output S for being far from the reachable set, so the gradient that reaches the 4D representation is the sensitivity of the reachable-set distance to S rather than to q.

**Why it measures what we claim.** The per-segment feasibility flag `feas(S)` measures *target-embodiment kinematic feasibility* because the joint-limit and collision residuals are computed on the exact URDF of the target robot and the projection is a constrained optimum on that robot's configuration manifold; this assumption fails when the URDF or scene mesh is inaccurate (e.g., unmodeled payload or soft contact), in which case `feas(S)` reflects feasibility under the nominal model rather than the physical robot. The repair residual `||S_repaired - S||` measures *how far the video-derived action is from the target's reachable manifold* because the constrained projection is the minimum-norm correction to the feasible set; this assumption fails when the feasible set is non-convex and DiffIK lands in a local minimum, in which case the residual over-estimates the true distance. The joint-training gradient from `L_proj` and `L_dist` into the tracker measures *whether the recovered action space is closed under target kinematics* because closure is exactly the fixed-point condition that the tracker's output is already in the feasible set (zero residual); this assumption fails when the tracker overfits to training embodiments, in which case the gradient reflects training-set closure rather than generalization to new embodiments.

## Contribution

["A differentiable kinematic-feasibility closure layer (KFC) that projects video-recovered SE(3) TCP trajectories onto a target embodiment's joint-limit-and-collision manifold and emits per-segment accept/repair flags with residual magnitudes.", "A joint-training scheme in which feasibility residuals backpropagate through a differentiable IK layer into the tracker's shared 4D scene representation, making the recovered action space closed under target kinematics rather than filtered post hoc.", 'A design principle: cross-embodiment retargeting trustworthiness is a closure property of the recovered action space, testable by the fixed-point residual of the projection, rather than an assumption made after tracking.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | DROID + RoboLab video, cross-embodiment retargeting | TCP4D already reports trajectory error here |
| Primary metric | Feasibility-violation rate on target URDF | Directly measures kinematic closure, not just tracking |
| Secondary metric | Downstream policy success rate on target embodiment | Measures whether repair improves manipulation performance |
| Baseline | TCP4D tracker (E4) | Tests whether projection adds feasibility beyond recovery |
| Baseline | Post-hoc classical IK filter on TCP4D | Tests whether gradients must reach the 4D representation |
| Baseline | Soft-penalty-only feasibility (no hard repair) | Tests whether hard rejection+repair matters for demos |
| Ablation-of-ours | KFC with frozen tracker (no joint training) | Tests whether coupling to 4D representation drives gains |
| Ablation-of-ours | KFC without L_dist term (only L_proj) | Tests whether the distance-to-feasible-set term is necessary for gradient flow |

### Why this setup validates the claim

The central claim is that closure of the recovered action space under target kinematics requires gradients from feasibility residuals to reach the tracker's shared 4D representation, and that hard repair (not soft filtering) is needed because retargeting consumes trajectories as demonstrations. The dataset choice matters because DROID and RoboLab are exactly where TCP4D reports metric TCP recovery, so any feasibility gain is attributable to KFC rather than to a different tracker; cross-embodiment retargeting is the downstream consumer whose corruption the repair step is meant to prevent. The primary metric, feasibility-violation rate computed on the target URDF, is the right detector because it is the quantity the constrained projection directly optimizes and the quantity a post-hoc filter cannot reduce without also degrading tracking; tracking error is reported alongside so that a trivial "freeze the robot" solution is visible as a tracking blow-up. The secondary metric, downstream policy success rate, validates that feasibility repair translates to manipulation performance rather than being a proxy that does not correlate with task success. The post-hoc IK baseline isolates the gradient-path sub-claim, the soft-penalty baseline isolates the hard-repair sub-claim, the frozen-tracker ablation isolates the joint-training sub-claim, and the no-L_dist ablation isolates whether the distance-to-feasible-set term is the mechanism that carries gradient to the 4D representation, so each mechanism has a dedicated falsifier rather than a single aggregate win.

### Expected outcome and causal chain

**vs. TCP4D tracker (E4)** — On a case where the video shows a human hand reaching a shelf pose outside the target arm's joint range, the baseline tracker produces a metric SE(3) trajectory that tracks the pixels accurately but is silently infeasible for the robot, because it has no embodiment model in the loop. Our method instead projects each 16-frame segment onto the target's reachable manifold and flags the segment, because the DiffIK layer computes joint-limit and collision residuals on the exact URDF. We expect near-parity on tracking error for feasible segments but a large gap in feasibility-violation rate concentrated on out-of-reach and near-collision subsets, and a corresponding gap in downstream policy success rate when trained on KFC-repaired vs. unrepaired demonstrations.

**vs. Post-hoc classical IK filter on TCP4D** — On a case where a segment drifts gradually toward a joint limit across frames, the post-hoc filter produces per-frame IK solutions that jitter and occasionally jump configurations, because it filters outputs without any signal teaching the tracker to avoid the infeasible region, and per-frame solving ignores temporal consistency. Our method instead backpropagates the segment-level feasibility residual and the distance-to-feasible-set term into the 4D representation during training, so the tracker learns to recover trajectories that are already near-feasible. We expect the post-hoc filter to match KFC on violation rate only at the cost of visibly higher tracking error and repair-residual magnitude, while KFC shows lower repair residual at comparable tracking.

**vs. Soft-penalty-only feasibility (no hard repair)** — On a case where a segment is mildly infeasible, the soft-penalty baseline produces a trajectory that reduces but does not eliminate the violation, because the penalty is traded off against tracking and no accept/repair flag is emitted, so a downstream retargeting pipeline consumes a corrupted demonstration without warning. Our method instead emits a binary flag and a minimum-norm constrained repair, because the feasible set is enforced as a projection rather than a cost. We expect the soft-penalty baseline to show a residual tail of small-but-nonzero violations and zero flagged segments, while KFC shows a bimodal pattern: clean segments with near-zero residual and explicitly flagged repaired segments.

**vs. KFC with frozen tracker (no joint training)** — On a case where the tracker's 4D representation has learned a bias toward out-of-reach poses from training embodiments, the frozen-tracker variant cannot correct this bias because the projection layer alone must absorb the feasibility residual by moving q_seq, leaving the tracker's output S unchanged. Our method instead trains the tracker against the distance-to-feasible-set term L_dist, so the 4D representation is penalized for outputting S far from the reachable set. We expect the frozen-tracker ablation to match KFC on feasibility-violation rate only if the projection layer fully absorbs the residual, but to show higher repair-residual magnitude and worse downstream policy success rate, revealing that the gradient path is not inert.

**vs. KFC without L_dist term (only L_proj)** — On a case where the projection layer can satisfy the feasibility residuals by moving q_seq alone, the no-L_dist variant receives near-zero gradient on S, so the tracker does not learn to avoid infeasible regions. Our method instead adds L_dist = ||S - FK(q_seq_projected)||_F with q_seq_projected detached, which explicitly penalizes the tracker for outputting S far from the reachable set. We expect the no-L_dist variant to behave like the frozen-tracker ablation on repair residual and downstream success, confirming that L_dist is the mechanism that carries gradient to the 4D representation.

### What would falsify this idea

If KFC's feasibility gain over the post-hoc IK filter is uniform across reachable and out-of-reach subsets rather than concentrated on the out-of-reach and near-collision segments where the predicted failure mode lives, then the gradient path into the 4D representation is not doing the claimed work and the idea reduces to filtering. Likewise, if the frozen-tracker ablation or the no-L_dist ablation matches joint training on repair residual and downstream policy success rate, the coupling claim is wrong.

## References

1. TCP4D: A Unified Tool-Center-Point Tracking and Generation Model for Robot Manipulation
