# Relative-Motion Constraint Graph for Compositionally Consistent 3D Trajectory Repair

## Motivation

Dense 3D tracking methods such as ReLift repair each object's trajectory independently, so repaired trajectories can be individually plausible yet mutually inconsistent—contact points drift, articulated joints dislocate, and relative motion between interacting objects breaks. The root cause is structural: trajectories are modeled as independent sequences coupled only by soft across-trajectory attention, which cannot enforce hard relative-motion invariants during repair. This failure recurs across any trajectory-repair pipeline that treats objects as separate sequences, making compositional consistency an open meta-level bottleneck rather than a local limitation of one corrector.

## Key Insight

Interacting objects' 3D trajectories are not independent sequences but sections of a single rigid/articulated motion field, so relative-motion constraints between them are exactly the invariants that a repair must preserve—making them hard, differentiable constraints rather than soft attention biases.

## Method

**What it is.** We introduce **RMG-Repair** (Relative-Motion Graph Repair), a trajectory corrector that takes lifted 3D trajectories for all tracked objects plus a per-frame object interaction graph, and outputs repaired trajectories that satisfy hard relative-motion constraints at articulated joints and contact points.

**Load-bearing assumption (stated explicitly).** The hard projection is only meaningful if the interaction graph is correct: edge existence (are these two objects actually in contact / articulated?) and edge type (contact vs. articulated joint) must be right, and the objects must be rigid over the constraint window. We do NOT assume the graph can be recovered by thresholding the noisy lifted trajectories alone. Instead, edge existence and type are produced by a learned, calibrated edge classifier (below), trained with supervision on interaction labels, and the hard projection is gated by that classifier's confidence. The constraint-residual metric is therefore reported conditioned on edge-classification precision/recall, so the manifold-satisfaction guarantee is stated relative to graph correctness rather than assumed.

**How it works.**

```pseudocode
Input: lifted trajectories T = {T_o in R^{F x 3}}_{o in objects}, per-object lifting confidence sigma_o in R^F
Output: repaired trajectories T_hat

# Phase 0: learned, calibrated edge inference (replaces distance-threshold graph)
# Candidate edges = all object pairs within a generous 0.5 m gate (recall-oriented), then classified.
for each candidate pair (i,j):
    e_ij = EdgeClassifier([H_i, H_j, R_ij, ||R_ij||, sigma_i, sigma_j])   # 2-layer MLP, hidden=128, GeLU
    p_ij = sigmoid(e_ij)                       # edge-existence confidence
    type_ij = argmax softmax(e_ij_type)        # {contact, articulated_joint, none}
    # calibrated on a 512-edge held-out split via temperature scaling (single scalar T_cal)
    p_ij_cal = calibrate(p_ij; T_cal)
E = {(i,j,type_ij) : p_ij_cal > tau_hard}      # tau_hard = 0.7 (calibrated)
E_soft = {(i,j,type_ij) : tau_soft < p_ij_cal <= tau_hard}   # tau_soft = 0.3

# Phase 1: per-object temporal encoding
for each object o:
    H_o = TemporalAttention(T_o)            # along-trajectory attention, window=full sequence

# Phase 2: relative-motion graph message passing
G = build_graph(objects, E union E_soft)     # nodes = objects, edges = inferred interaction edges
for l in 1..L (L=3):
    for each edge (i,j,type) in E union E_soft:
        R_ij[f] = T_i[f] - T_j[f]              # relative translation
        if type == articulated_joint:
            r_ij[f] = ||R_ij[f]|| - d_ij      # joint distance should be constant d_ij
        if type == contact:
            r_ij[f] = distance(T_i[f], T_j[f]) - 0  # contact => zero distance
        m_ij = MLP_type([H_i, H_j, R_ij, r_ij, p_ij_cal])   # 2-layer MLP, hidden=128, GeLU
    H_o = H_o + sum_{j in N(o)} m_oj        # aggregate messages

# Phase 3: differentiable constraint layer (hard projection, confidence-gated)
for each edge (i,j,type) in E union E_soft:
    delta_ij = project_to_manifold(R_ij, type, d_ij)   # closed-form: scale to constant distance / zero distance
    # confidence gate: hard projection only above tau_hard; soft penalty below
    w_ij = 1.0 if p_ij_cal > tau_hard else p_ij_cal / tau_hard   # in [0,1]
    # uncertainty-weighted split (replaces fixed 0.5): weight by inverse lifting variance
    a_i = (1/sigma_i^2) / (1/sigma_i^2 + 1/sigma_j^2)
    T_i <- T_i - w_ij * a_i       * delta_ij
    T_j <- T_j + w_ij * (1 - a_i) * delta_ij

# Phase 4: residual refinement
T_hat_o = T_o + ConstraintAwareHead(H_o)     # small MLP, conditioned on constraint residuals r_ij
return T_hat
```

Hyperparameters: L=3 message-passing layers, MLP hidden size 128, GeLU activation, constraint layer applied after each of 2 refinement iterations, edge-classifier gate tau_hard=0.7 / tau_soft=0.3 (calibrated via temperature scaling on 512 held-out edges), per-object split weight a_i from inverse lifting variance sigma_i^2 (falls back to 0.5 when sigma_i is unavailable).

**Why this design.** We chose a hard differentiable projection layer over soft penalty losses because soft penalties only encourage consistency and can be traded off against repair accuracy, whereas the projection guarantees the constraint manifold is reached at every forward pass, accepting the cost that the projection may slightly perturb already-correct observations (mitigated by the residual refinement head). We chose an explicit interaction graph with typed edges (contact vs. articulated joint) over a fully connected across-trajectory attention because different physical relations impose different invariants (zero distance vs. constant distance), accepting the cost of requiring an interaction graph at inference, which we obtain from a learned calibrated edge classifier trained on interaction labels rather than from distance thresholds on the noisy lifted trajectories. We chose confidence-gated projection (hard above tau_hard, soft below) over unconditional projection because a wrong edge enforced exactly would rigidly weld non-contacting objects, accepting the cost that low-confidence true edges receive only a soft penalty. We chose uncertainty-weighted correction splitting over fixed equal 0.5 splitting because inverse-variance weighting preserves the constraint while down-weighting the more occluded object, accepting the cost that lifting confidence must be supplied by the backbone. A domain expert would not describe this as ReLift with an added GNN because the constraint projection changes the output space—repaired trajectories are guaranteed to lie on the relative-motion manifold for edges whose classifier confidence exceeds tau_hard—whereas ReLift's soft attention only biases the output distribution.

**Why this measures what we claim.** The constraint residual `r_ij[f] = ||R_ij[f]|| - d_ij` measures **relative-motion consistency** because the assumption that interacting objects maintain constant joint distance or zero contact distance makes the residual equal to the physical inconsistency; this assumption fails when objects deform, lose contact, or the edge classifier is wrong, in which case the residual reflects apparent geometric deviation or a false invariant rather than true violation. The projection magnitude `||delta_ij||` measures **compositional repair effort** because the assumption that the lifted trajectories are locally Euclidean makes the minimum-norm projection equal to the smallest correction restoring the invariant; this assumption fails under large depth errors, in which case the projection reflects a local linearization rather than the true minimal repair. The graph message `m_ij` measures **inter-object motion coupling** because the assumption that relative motion is informative about which object's trajectory is corrupted makes the message a sufficient statistic for distributing correction; this assumption fails when both objects are jointly corrupted in a correlated way, in which case the message reflects shared error rather than one object's inconsistency. The edge-classifier confidence `p_ij_cal` measures **graph correctness** because the assumption that the classifier is calibrated on held-out edges makes p_ij_cal equal to the probability the edge is real and correctly typed; this assumption fails under distribution shift (unseen interaction types), in which case the gate reflects classifier overconfidence rather than true edge reliability.

## Contribution

['A relative-motion constraint graph with typed edges and a differentiable projection layer that enforces hard compositional consistency across interacting objects during 3D trajectory repair.', 'A design principle showing that hard constraint projection, rather than soft cross-attention penalties, is necessary to preserve relative-motion invariants when repairing trajectories of interacting objects.', 'An interaction-graph construction procedure from lifted trajectories alone, requiring no external contact or articulation annotations.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | An existing articulated-object / hand-object benchmark with ground-truth contact and articulation annotations (e.g., an articulated-object or hand-object dataset that ships per-frame contact/articulation labels) | Provides the interaction labels needed to train and calibrate the edge classifier and to score residual conditioned on graph correctness; the first experiment derives the graph from annotations, deferring the threshold-based graph to an ablation |
| Primary metric | Relative-motion constraint residual on held-out edges, reported conditioned on edge-classification precision/recall (binned by p_ij_cal) | Directly measures manifold satisfaction and states the guarantee relative to graph correctness rather than assuming it |
| Secondary metric | Edge-classification precision/recall and calibration error (ECE) on held-out edges | Quantifies the load-bearing assumption so residual gains can be attributed to correct edges |
| Baseline | ReLift (soft-attention repair) | Tests whether hard projection beats soft bias |
| Baseline | Dedicated 3D tracker | Tests whether repair beats end-to-end tracking |
| Baseline | Lifted trajectories (no repair) | Tests whether repair helps at all |
| Ablation-of-ours | RMG-Repair w/ soft penalty instead of projection | Isolates the hard-projection mechanism |
| Ablation-of-ours | RMG-Repair w/ distance-threshold graph instead of learned classifier | Tests the load-bearing graph-inference assumption |
| Ablation-of-ours | RMG-Repair w/ fixed 0.5 split instead of inverse-variance split | Tests the uncertainty-weighted correction |
| Ablation-of-ours | RMG-Repair w/ injected graph errors (false/missed edges at 10%/20%/30%) | Sensitivity analysis of residual degradation to graph errors |

### Why this setup validates the claim

The central claim is that a hard differentiable projection onto the relative-motion manifold produces trajectories that are guaranteed constraint-consistent, and that this guarantee buys accuracy that soft penalties cannot—provided the interaction graph is correct. The metric choice is load-bearing: relative-motion constraint residual on held-out interaction edges is precisely the quantity the projection claims to drive to zero, so it is the right instrument to detect the predicted effect rather than a proxy, and reporting it conditioned on edge-classification precision/recall makes the guarantee honest rather than assumed. The no-repair lifted baseline establishes the headroom—how inconsistent lifted trajectories actually are—so any gain is attributable to repair rather than to the tracker. ReLift tests the specific sub-claim that hard projection outperforms soft attention biasing, since both consume the same lifted input and differ only in whether the output is constrained to the manifold. The dedicated 3D tracker tests whether repair of lifted trajectories can rival end-to-end geometry, which is the practical claim. The soft-penalty ablation is the decisive control: it holds architecture, capacity, and data fixed and changes only the projection mechanism, so if the projection is doing the work, this ablation should lose the constraint-residual advantage while retaining the same inputs. The threshold-graph ablation and the injected-graph-error sweep directly test the load-bearing assumption: if residual gains collapse when the graph is wrong, the guarantee is conditional on graph correctness as stated. The fixed-0.5-split ablation isolates the uncertainty-weighted correction. Together these form a falsifiable test because each baseline isolates one link in the causal chain from mechanism to outcome.

### Expected outcome and causal chain

**vs. Lifted trajectories (no repair)** — On a case where a hand manipulates an object, the lifted trajectories drift apart at the contact point because independent per-object lifting has no mechanism to enforce zero contact distance, so the residual grows with depth error. Our method instead projects the relative motion onto the zero-distance manifold at every forward pass because the typed contact edge supplies the invariant, so we expect a large residual reduction concentrated on contact and articulated-joint edges, with near-parity on isolated objects that have no interaction edges.

**vs. ReLift (soft-attention repair)** — On a case where an articulated joint's distance is violated by a depth outlier on one link, ReLift's soft attention only biases the output toward consistency and can trade that bias against reconstruction accuracy, leaving residual on the manifold because nothing forces the constraint. Our method instead applies the closed-form projection that restores constant joint distance exactly, then refines, so we expect a noticeably lower constraint residual on high-violation edges while both methods are comparable on already-consistent edges—the gap should scale with initial violation magnitude.

**vs. Dedicated 3D tracker** — On a case where an object is briefly occluded, the dedicated tracker commits to a smooth prior and loses the contact relationship because it tracks geometry without an explicit relative-motion invariant, producing drift at re-emergence. Our method instead repairs the lifted trajectory against the interaction graph, which carries the joint constraint across the occlusion gap, so we expect a smaller residual at occlusion boundaries and comparable accuracy on fully visible frames, with the advantage concentrated where occlusion breaks tracking.

**vs. RMG-Repair w/ distance-threshold graph** — On a case where a true contact has already drifted apart in depth, the threshold graph misses the edge (false negative) and the projection never fires, so residual stays high; our learned calibrated classifier recovers the edge from motion and confidence cues, so we expect lower residual on drifted-contact edges and higher edge recall, with the gap widening as initial drift grows.

**vs. RMG-Repair w/ fixed 0.5 split** — On a case where one object is heavily occluded and has high lifting variance, the fixed split moves the well-observed object as much as the occluded one, injecting error into the correct trajectory; inverse-variance weighting instead moves the occluded object more, so we expect lower residual on the well-observed object's trajectory and comparable residual on the constraint itself.

**vs. RMG-Repair w/ injected graph errors** — As false/missed edge rate rises from 10% to 30%, we expect residual to degrade monotonically and the hard-projection advantage over the soft-penalty ablation to shrink, quantifying how much of the guarantee depends on graph correctness.

### What would falsify this idea

If the soft-penalty ablation matches our hard-projection variant on the constraint residual across all edge types, the claim that projection guarantees manifold satisfaction is wrong. If our gain over ReLift is uniform across interaction and non-interaction edges rather than concentrated where typed constraints are violated, or if residual gains vanish once edges are conditioned on classifier precision/recall, the mechanism is not the projection and the central claim fails.

## References

1. ReLift: Dense 3D Tracking as Lifted Trajectory Repair
