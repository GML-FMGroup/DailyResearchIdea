# Occlusion-Order Constrained Trajectory Repair via Differentiable Visibility Polytopes

## Motivation

ReLift repairs lifted 3D trajectories with alternating attention but has no invariant tying its corrections to scene geometry, so a repaired point can drift behind the surface that actually occluded it — the depth inherited from the occluder gets smoothed rather than rejected, producing trajectories that are geometrically plausible yet occlusion-order-violating. This is the local instance of a broader structural gap: across dense 3D tracking pipelines, scene reconstruction (e.g., OmniRe) and trajectory estimation share a representation but never share a constraint system, so geometric reasoning cannot veto trajectory hypotheses. The root cause is that occlusion is treated as a per-frame visibility label rather than an order relation over the whole trajectory, leaving the corrector free to interpolate through occluders.

## Key Insight

Occlusion is a transitive order relation on depth along each camera ray, so enforcing it as a projection onto the intersection of per-ray order cones makes repaired trajectories provably consistent with reconstructed scene geometry rather than merely smooth.

## Method

**What it is.** OCCUR (Occlusion-Constrained Unified Repair) takes lifted 3D trajectories $\mathbf{P}\in\mathbb{R}^{T\times N\times 3}$ plus the OmniRe dynamic scene representation (per-frame meshes $\mathcal{M}_t$ with tracked dynamic nodes) and outputs repaired trajectories whose per-frame depths respect the occlusion order induced by $\mathcal{M}_t$. Internally it wraps ReLift's alternating-attention corrector and interleaves a differentiable occlusion-order projection layer (OOP) after each attention block. **Load-bearing assumption (stated explicitly):** the OmniRe mesh provides a per-ray occluder depth $z_\text{occ}$ that is a faithful depth oracle for the true occlusion order, so that the sign of $z_\text{occ}-P_\text{hat}.z$ is equivalent to a genuine occlusion-order violation rather than an artifact of mesh holes, thin structures, or reconstruction error. This assumption is not asserted silently: it is bounded empirically by a mesh-fidelity audit (see experiment) and guarded inside the method by a per-ray confidence $c[t,n]\in[0,1]$ that gates the hard projection, so a low-confidence $z_\text{occ}$ falls back to a soft hinge with slack rather than a hard clip.

**How it works.**
```pseudocode
Input: P (T x N x 3), OmniRe meshes M_t, camera intrinsics K, extrinsics E_t
Hyper: n_blocks=6, tau_order=1e-2, lambda_vis=0.5, iters_proj=3,
       tau_conf=0.5, sigma_slack=0.05, detach_cast=False

1. Ray casting (differentiable, once per forward):
     for each (t, n): cast ray from camera center through pixel of P[t,n]
     z_occ[t,n] = depth of nearest mesh intersection with M_t   # occluder depth
     vis[t,n]   = 1 if |P[t,n].z - z_occ[t,n]| < tau_order else 0
     # per-ray confidence on z_occ: c=1 if exactly one intersection and
     # local mesh thickness > 2*tau_order; c=0 if zero intersections (hole)
     # or >2 intersections within 0.1m (thin structure / spurious fragment);
     # else c = clip(1 - |n_intersect-1|/2, 0, 1)
     c[t,n] = confidence(z_occ[t,n], M_t, tau_order)
     if detach_cast: z_occ, c = detach(z_occ), detach(c)

2. Trajectory tokens X = embed(P)  # (T x N x d)

3. for block b in 1..n_blocks:
     X = AlongTrajectoryAttention(X)      # temporal aggregation (ReLift)
     X = AcrossTrajectoryAttention(X)     # neighbor sharing (ReLift)
     P_hat = decode(X)                    # (T x N x 3) candidate repair
     # --- OOP: constrained projection ---
     for k in 1..iters_proj:
        for each (t,n) with vis[t,n]==0:
            if c[t,n] >= tau_conf:
                # confident occluder: hard clip BEHIND the occluder
                P_hat[t,n] = project_behind(P_hat[t,n], z_occ[t,n], K, E_t)
            else:
                # unconfident occluder: soft hinge with slack sigma_slack
                P_hat[t,n] = project_behind_soft(P_hat[t,n], z_occ[t,n],
                                                 sigma_slack, K, E_t)
        # temporal order invariant: along a trajectory, depth must not
        # cross an occluder between two visible frames
        P_hat = enforce_order_along_tracks(P_hat, z_occ, vis, c)
     X = re_embed(P_hat)                  # feed projected geometry back

4. Loss = L_repair(P_hat, P_gt) + lambda_vis * L_order(P_hat, z_occ, vis, c)
   L_order = c * hinge(max(0, z_occ - P_hat.z)) for occluded pts  # behind constraint
```
The projection is a closed-form clip along the viewing ray: $P_\text{proj} = c_\text{cam} + \frac{z_\text{occ}+\epsilon}{\hat z}(P_\text{hat}-c_\text{cam})$ for occluded points with $c[t,n]\ge\tau_\text{conf}$, with $\epsilon$ a small margin; for $c[t,n]<\tau_\text{conf}$ the clip is replaced by a soft hinge $P_\text{proj} = P_\text{hat} + \frac{\max(0, z_\text{occ}+\epsilon-\hat z)}{\hat z}(P_\text{hat}-c_\text{cam})\cdot\sigma_\text{slack}$ so an unverified occluder cannot forcibly push a correctly-tracked point. `enforce_order_along_tracks` solves a per-track isotonic regression of depth against occluder depth, which is a PAVA pass (monotone, $O(T)$), weighted by $c[t,n]$ so low-confidence occluders contribute proportionally less to the ordering. Gradients flow through the ray-casting via the mesh SDF surrogate, so the corrector learns to predict already-consistent repairs; `detach_cast` is an ablation switch that detaches $z_\text{occ}$ and $c$ from the graph if SDF-surrogate gradients prove unstable.

**Why this design.** We chose a hard projection layer over a soft occlusion penalty because ReLift's failure is a *violation* of an order invariant, not a gradual drift; a penalty tolerates residual violations while a projection onto the order cone cannot, accepting the cost that projection is non-smooth at the cone boundary and requires the extra PAVA pass. We chose differentiable ray casting against the OmniRe mesh over a learned visibility classifier because the mesh gives an explicit occluder depth $z_\text{occ}$ that anchors the constraint to actual geometry rather than to a learned proxy, accepting the cost that mesh quality (thin structures, holes) directly bounds constraint accuracy — which is why the per-ray confidence $c[t,n]$ gates the hard clip and falls back to a soft hinge on unverified occluders. We chose to interleave OOP inside the attention stack rather than apply it once at the output because early projection lets downstream attention blocks reason over already-consistent geometry, accepting the cost of $n_\text{blocks}\times\text{iters}_\text{proj}$ ray-cast calls. Unlike ReLift, which is a pure attention corrector with no geometric invariant, and unlike soft-penalty 3D trackers, OCCUR's corrector is trained *through* the projection, so an expert would not describe it as ReLift-plus-a-loss-term: the projection changes what the attention sees at every block.

**Why it measures what we claim.** The quantity $z_\text{occ}[t,n]-P_\text{hat}[t,n].z$ measures *occlusion-order compliance* because the assumption that the OmniRe mesh is a faithful depth oracle along the ray makes the sign of this difference equivalent to the point being behind vs. in front of the true occluder; this assumption fails when the mesh has holes or thin structures (e.g., a pole), in which case the difference reflects mesh sparsity rather than true occlusion order — which is why the per-ray confidence $c[t,n]$ is reported alongside the violation rate and the mesh-fidelity audit quantifies $z_\text{occ}$ error on thin structures and holes. The isotonic PAVA residual measures *temporal order consistency* because the assumption that depth along a track is monotone between two visible frames (no occluder entry/exit within the gap) makes the isotonic fit equivalent to a valid occlusion ordering; this assumption fails at occlusion boundaries where an object enters between frames, in which case the residual reflects legitimate depth discontinuity rather than a repair error. The visibility flag $\text{vis}[t,n]$ measures *whether a point is observable* because the assumption that lifted depth within $\tau_\text{order}$ of the mesh is trustworthy makes the threshold equivalent to a visibility test; this assumption fails under correspondence error that shifts the pixel, in which case the flag reflects tracker error rather than true visibility.

## Contribution

(1) A differentiable occlusion-order projection layer that casts occlusion as a per-ray order-cone constraint and enforces it via closed-form clipping plus per-track isotonic regression, interleaved inside a trajectory-repair attention stack so the corrector is trained through the constraint. (2) The design principle that scene reconstruction and trajectory repair should share a constraint system (occlusion order) rather than only a representation, with the projection acting as a hard invariant the corrector cannot violate. (3) An analysis of when mesh-derived occluder depth is a valid constraint oracle, identifying thin-structure and occlusion-boundary failure modes and their effect on repair.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Dense 3D tracking benchmark with OmniRe meshes | Provides explicit occluder depth for order constraints |
| Primary metric | Occlusion-order violation rate on occluded subset | Directly measures the invariant OCCUR enforces |
| Secondary metric | Full metric suite (trajectory error, visibility accuracy) on all tracks | Shows gains not confined to occluded subset |
| Baseline | ReLift corrector | Tests whether projection beats pure attention repair |
| Baseline | Soft occlusion-penalty tracker | Tests whether hard projection beats penalty tolerance |
| Baseline | Lifted trajectories (no repair) | Tests whether repair is needed at all |
| Ablation-of-ours | OCCUR without OOP (attention only) | Isolates the projection layer's contribution |
| Ablation-of-ours | OCCUR with hard clip on all rays (no confidence gate) | Tests whether confidence gating protects against bad z_occ |
| Ablation-of-ours | OCCUR with detached ray casting | Tests stability of SDF-surrogate gradients |
| Audit | Mesh-fidelity audit: z_occ error vs. held-out GT depth on thin structures/holes | Bounds the 'faithful depth oracle' assumption |

### Why this setup validates the claim

The central claim is that occlusion-order violations are a *hard invariant* failure that soft penalties tolerate but projection eliminates, and that interleaving projection inside the attention stack changes what downstream blocks see. The dataset choice matters because OmniRe meshes supply the explicit occluder depth $z_\text{occ}$ that makes the violation metric well-defined rather than proxy-based. The primary metric — violation rate restricted to the occluded subset — is the right detector because the predicted effect is a *concentration* of gains precisely where the order invariant is violated; a generic trajectory-error metric would dilute this signal across visible points where all methods should be near parity, which is why we also report the full metric suite on all tracks as a secondary check. ReLift tests the sub-claim that attention alone cannot enforce a hard invariant; the soft-penalty tracker tests the sub-claim that tolerance leaves residual violations; lifted trajectories establish the floor. The no-OOP ablation is the decisive control: if OCCUR's gain survives removing projection, the mechanism claim is false and the gain is attributable to architecture or training rather than the order constraint. The hard-clip-without-confidence ablation tests the load-bearing assumption directly: if removing the confidence gate does not degrade violation rate, the gate is unnecessary; if it degrades, the gate is doing the work the assumption requires. The mesh-fidelity audit bounds the assumption empirically by reporting $z_\text{occ}$ error against held-out ground-truth depth on thin structures and holes, and by plotting violation-rate gain as a function of mesh error, so the 'faithful depth oracle' claim is measured rather than asserted. Together these form a falsifiable test because each baseline and ablation isolates one link in the causal chain from mechanism to metric.

### Expected outcome and causal chain

**vs. ReLift corrector** — On a case where a tracked point passes behind a thin foreground object, ReLift produces a repaired depth that drifts in front of the occluder because its attention corrector has no geometric invariant anchoring depth to the mesh, only learned smoothness priors. Our method instead projects that point behind $z_\text{occ}$ via the closed-form ray clip (when $c[t,n]\ge\tau_\text{conf}$) and feeds the projected geometry back into subsequent attention blocks, so downstream blocks reason over already-consistent depth. We expect a pronounced gap in violation rate on the occluded subset, with near-parity on fully visible tracks where no order constraint binds.

**vs. Soft occlusion-penalty tracker** — On a case where an occluded point sits marginally in front of the occluder, the soft-penalty tracker produces a residual violation because its hinge penalty is traded off against the repair loss and tolerates small positive margins. Our method instead projects onto the order cone so the violating component is clipped to zero by construction when the occluder is confident, accepting non-smoothness at the boundary. We expect OCCUR to show a lower violation rate concentrated in the near-boundary band where penalties leave slack, while both methods match on grossly occluded points.

**vs. Lifted trajectories (no repair)** — On a case where 2D correspondence error corrupts depth under occlusion, lifted trajectories produce large order violations because no mechanism reconciles lifted depth with mesh geometry. Our method instead repairs through alternating attention plus interleaved projection, so violations are corrected block by block. We expect a large uniform reduction in violation rate relative to lifting, establishing that repair is necessary, with the OCCUR-vs-ReLift gap then isolating the projection's specific contribution.

**vs. OCCUR without OOP (attention only)** — On a case where a tracked point passes behind a thin foreground object, the no-OOP variant produces the same drift as ReLift because the attention stack alone has no geometric invariant. Our method instead interleaves the projection after each attention block, so the projected geometry is what downstream blocks see. We expect the no-OOP ablation to match ReLift's violation rate, confirming that the projection layer, not the architecture, carries the gain.

**vs. OCCUR with hard clip on all rays (no confidence gate)** — On a case where the mesh has a hole or a thin structure reconstructed as a blob, the ungated variant forcibly pushes a correctly-tracked point behind a spurious surface, and the PAVA pass propagates the corrupted ordering along the track. Our method instead gates the hard clip by $c[t,n]$ and falls back to a soft hinge with slack $\sigma_\text{slack}$ on unverified occluders. We expect the ungated variant to show *higher* violation rate than full OCCUR precisely on low-confidence rays, with parity on high-confidence rays, bounding the load-bearing assumption empirically.

**vs. OCCUR with detached ray casting** — On a case where SDF-surrogate gradients are unstable, the detached variant removes gradient flow through $z_\text{occ}$ and $c$. Our method instead trains through the differentiable cast by default. We expect the detached variant to be a stability control: if it matches full OCCUR, the differentiable cast is not contributing and can be detached for compute savings; if it underperforms, the gradient through the cast is doing work.

### What would falsify this idea

If OCCUR's gain over the soft-penalty tracker and ReLift is uniform across occluded and visible subsets rather than concentrated on occluded tracks and near-boundary points, the claim that hard order projection is the causal mechanism would be wrong. Likewise, if the no-OOP ablation matches full OCCUR, the projection layer contributes nothing and the design rationale collapses; and if the ungated hard-clip variant matches full OCCUR on low-confidence rays, the confidence gate is unnecessary and the load-bearing assumption is not actually load-bearing.

## References

1. ReLift: Dense 3D Tracking as Lifted Trajectory Repair
