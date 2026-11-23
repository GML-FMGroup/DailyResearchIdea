# Reachability-Certified Repair Sampling for Aerial Vision-Language Navigation

## Motivation

AeroRSI's recursive self-improvement treats rule-based repair proposals as valid whenever continuation from the repaired state succeeds in simulation, but it never checks whether the repaired action sequence lies on the continuous flight-control manifold that realistic UAV platforms such as OpenUAV actually expose; conversely, OpenUAV discretizes realistic flight control for VLN evaluation without auditing whether discretized repairs are dynamically executable. The structural root cause is that repair validity is judged in the closed loop of the policy's own simulator rather than against the reachable set induced by the platform's continuous dynamics, so a repair that 'works' in the AeroRSI loop can be dynamically infeasible the moment it is flown under OpenUAV's continuous control, and neither paper supplies a coverage certificate linking the two.

## Key Insight

A repair is only a valid self-improvement signal if its discrete action word is the image of at least one dynamically feasible continuous control trajectory under the platform's reachable-set relaxation, so the repair sampler must be certified against a reachable-set over-approximation rather than against simulator continuation alone.

## Method

**FARA (Feasibility-Audited Repair Acceptance)** takes as input (i) a candidate repair action word produced by AeroRSI's rule-based sampler, (ii) the UAV state at the failure point, and (iii) the OpenUAV continuous flight-control model; it outputs ACCEPT / REJECT plus, on ACCEPT, a witness continuous control trajectory that realizes the repair.

**Load-bearing assumption (stated explicitly).** FARA's acceptance signal is only meaningful if the reachable-set over-approximation used in step 2 is *sound* for the nonlinear OpenUAV quadrotor dynamics under the action envelope C(a_i) at dt = 0.05 s over horizon H = 2.0 s — i.e., the true reachable tube is contained in the propagated set R_H. A convex hull of f(R_{t-1}, C(a_i)) is not sound by construction for a nonlinear f, and the shortest-path witness in step 5 can leave the true reachable tube. FARA therefore does not rely on the raw hull: step 2 uses a sound interval/Taylor-model propagation, and step 5 verifies the extracted witness by forward integration under the true OpenUAV dynamics, with a CEGAR refinement loop (steps 2b–5b) that splits the control envelope and re-propagates until the witness is verified or the repair is rejected. The certificate is thus sound-by-construction; the hull is used only as an inner relaxation to seed witness search.

```pseudocode
Input: repair word w = (a_1..a_k), state s0, OpenUAV dynamics f, horizon H=2.0s, dt=0.05s
Output: decision d in {ACCEPT, REJECT}, witness tau or None

1. Discretization lift: for each a_i, enumerate the discrete-to-continuous
   control envelope C(a_i) = {u : ||u - u_center(a_i)||_inf <= r_a}  (r_a from OpenUAV action set)
2. Sound reachable-set propagation (Taylor-model / interval arithmetic, CORA-style):
   R_0 = {s0}
   for t in 1..H/dt:
       R_t = TM_flow( R_{t-1}, C(a_{ceil(t/dt)}), f, dt )   # sound over-approx of true reachable set
       if R_t is empty: return REJECT, None
2b. Inner convex relaxation (for witness seeding only):
   H_t = ConvexHull( f(R_{t-1}, C(a_{ceil(t/dt)})) )  for t in 1..H/dt
3. Coverage certificate: set C_cert = R_H (sound); compute margin
   m(w) = min_{s in R_H} dist(s, ObstacleSet)  (signed distance)
4. Acceptance rule (preliminary): d = ACCEPT iff m(w) >= m_min (0.3 m) AND
   R_H intersects the goal-region induced by the instruction
5. Witness extraction: solve a shortest-path problem over the inner convex
   relaxation H_t to recover tau = (u_1..u_H) inside C_cert
5b. Witness verification (CEGAR): forward-integrate tau under true f from s0;
   if the resulting trajectory violates m_min or leaves the goal-region intersection,
   split the offending C(a_i) into two half-envelopes and goto step 2 with the refined envelope;
   repeat up to K_max = 4 splits; if no verified witness after K_max, return REJECT, None
6. Return (ACCEPT, tau) only if step 5b verified tau; else (REJECT, None)
```

Repairs rejected at step 4 or step 5b are returned to AeroRSI's sampler with the constraint set (C_cert, m_min) so the next proposal is drawn from the certified envelope; accepted repairs are distilled into the policy together with their verified witness trajectories.

**Why this design.** We chose a sound Taylor-model/interval reachable-set propagation (with a convex-hull inner relaxation used only to seed witness search) over a raw convex-hull over-approximation because exact nonlinear reachability for the OpenUAV quadrotor model is intractable at the 0.05 s control rate and a raw hull is not sound for nonlinear f, accepting the cost that the certificate is conservative and will reject some dynamically feasible repairs. We chose to return the certificate back to the sampler as a constraint set rather than only filtering repairs because filtering alone leaves the sampler's proposal distribution unchanged, accepting the cost of an extra projection step per round. We chose a signed-distance margin threshold m_min = 0.3 m rather than a binary obstacle test because binary tests are brittle to the over-approximation's slack, accepting the cost of one tunable hyperparameter. Unlike AeroRSI, which validates repairs only by continuation in its own loop, FARA's certificate is computed against OpenUAV's continuous dynamics and its witness is verified by forward integration under the true dynamics, so a domain expert would not describe FARA as a variant of AeroRSI's continuation verification: the acceptance signal is geometric (reachable-set membership and clearance) plus a verified executable witness, rather than behavioral (does the rollout succeed).

**Why it measures what we claim.** The reachable-set margin m(w) measures dynamic feasibility of a discrete repair because the Taylor-model propagation in step 2 soundly over-approximates the true reachable set of the OpenUAV dynamics under the action envelope C(a_i); this assumption fails when the sound propagation is loose enough that a state in R_H is not actually reachable, in which case m(w) reflects geometric clearance of a superset rather than of a flyable trajectory — which is why step 5b verifies the witness by forward integration and rejects repairs whose witness cannot be realized. The witness trajectory tau measures executability of the repair because it is a concrete control sequence that has been forward-integrated under the true OpenUAV dynamics and confirmed to satisfy m_min; this assumption fails when the forward integration is numerically inaccurate (dt too coarse or integrator tolerance too loose), in which case tau is a numerical artifact rather than a flyable control — we use RK4 integration with tolerance 1e-6 and report the integration residual. The goal-region intersection at step 4 measures task-relevance of the repair because the instruction-induced goal region is defined independently of the policy; this assumption fails when the goal region is multi-modal and the relaxation intersects the wrong mode, in which case the repair is certified as relevant to a goal the instruction did not mean.

## Contribution

['A reachability-coverage certificate, FARA, that maps each discrete repair from a rule-based sampler to a certified continuous control envelope under realistic UAV dynamics, with a witness trajectory for every accepted repair and a constraint set returned to the sampler for every rejected one.', 'A design principle for recursive self-improvement in embodied navigation: repair acceptance must be judged against a platform-induced reachable set, not only against simulator continuation, because continuation success and dynamic feasibility can diverge under continuous flight control.', 'An integration of the certificate into the recursive self-improvement loop of AeroRSI on OpenUAV, turning the repair sampler from an unconstrained proposer into a reachability-constrained one.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | TravelUAV (OpenUAV-simulated trajectories) | Matches AeroRSI baseline and continuous-flight platform domain. |
| Primary metric | Success rate on collision-prone repair subset | Directly probes dynamic feasibility of accepted repairs. |
| Secondary metric | False-accept / false-reject rate vs. Monte-Carlo reachability oracle | Quantifies conservatism of the certificate. |
| Secondary metric | Rejection rate per self-improvement round; rounds to target SR | Probes sampler-projection contribution. |
| Secondary metric | Wall-clock per repair round (ms) | Bounds the extra projection cost claim. |
| Baseline | AeroRSI continuation verification | Tests whether geometric certificate beats behavioral rollout check. |
| Baseline | AeroRSI + fixed clearance filter | Tests whether certificate feedback to sampler adds value. |
| Baseline | Direct policy distillation (no audit) | Tests whether auditing prevents infeasible repair absorption. |
| Baseline | Monte-Carlo reachability oracle (N=200 rollouts per repair) | Exact-but-sampled reference for soundness audit. |
| Ablation-of-ours | FARA without sampler constraint feedback | Isolates contribution of returned certified envelope. |
| Ablation-of-ours | FARA with raw convex-hull propagation (no Taylor-model, no CEGAR) | Isolates contribution of sound propagation + witness verification. |

### Why this setup validates the claim

The central claim is that acceptance of a discrete repair should be decided by geometric reachability under OpenUAV's continuous dynamics rather than by behavioral continuation, and that feeding the resulting certificate back to the sampler improves the proposal distribution. TravelUAV is the right dataset because AeroRSI reports its 46.4%→67.5% self-improvement gain there, giving us a known reference point where repairs are proposed by exactly the rule-based sampler FARA consumes. The primary metric — success rate restricted to collision-prone repairs — is chosen because the predicted effect is not a uniform uplift but a concentration of gains where the baseline's mechanism fails: AeroRSI's continuation check validates a repair only if its own rollout happens to succeed, so it admits repairs whose success was luck and rejects repairs whose failure was sampled noise. The continuation-verification baseline therefore tests the sub-claim that geometric certification is a better acceptance signal; the fixed-clearance baseline tests whether the certificate's returned constraint set (not just filtering) matters; the no-audit baseline tests whether auditing at all is necessary; and the no-feedback ablation isolates the sampler-projection contribution. The Monte-Carlo oracle baseline and the raw-hull ablation directly test the load-bearing soundness assumption: if the raw hull admits infeasible repairs that the oracle rejects, the sound Taylor-model + CEGAR design is doing the causal work, and the false-accept rate quantifies the conservatism we trade for soundness. If FARA's advantage is real, it must appear as a gap on the collision-prone subset with rough parity elsewhere — a pattern none of these baselines can produce by construction.

### Expected outcome and causal chain

**vs. AeroRSI continuation verification** — On a case where a repair word sequence is sampled that would clear an obstacle only if the quadrotor flies a tight, dynamically demanding arc, the baseline accepts or rejects based on whether its own single rollout happens to succeed, so it can admit a repair whose success was a lucky sample and reject a feasible repair whose rollout drifted. FARA instead propagates a sound Taylor-model reachable set and verifies the extracted witness by forward integration, so acceptance depends on whether some control in the envelope clears the obstacle by the signed-distance margin and the witness is confirmed flyable, independent of sampling luck. We expect a noticeable success-rate gap concentrated on collision-prone repairs, with rough parity on open-corridor repairs where continuation checks rarely fail.

**vs. AeroRSI + fixed clearance filter** — On a case where the sampler repeatedly proposes repairs from a region that is geometrically infeasible, the baseline filter rejects each proposal but leaves the proposal distribution unchanged, so the sampler keeps redrawing from the same infeasible region and burns rounds without progress. FARA returns the certified envelope (C_cert, m_min) to the sampler, so the next proposal is projected into the feasible set rather than resampled blindly. We expect the gap to show up as fewer self-improvement rounds needed to reach a given success level and a lower rejection rate per round, with the largest divergence on scenes where the initial policy's failures cluster in one geometric region.

**vs. Direct policy distillation (no audit)** — On a case where a repair is proposed that is textually plausible but dynamically infeasible — for example a turn that exceeds the quadrotor's yaw authority at the current speed — the baseline distills the repair into the policy with no feasibility check, so the policy learns an action it cannot execute and degrades on nearby states. FARA rejects the repair at step 4 or step 5b, so distillation only absorbs flyable behavior. We expect the baseline to show a success-rate drop or stagnation after the first self-improvement round on collision-heavy scenes, while FARA's success rate rises monotonically across rounds.

**vs. Monte-Carlo reachability oracle (N=200 rollouts per repair)** — On a small scene subset (50 repairs), we compare FARA's ACCEPT/REJECT decisions to the oracle's empirical reachability. We expect FARA's false-accept rate to be near zero (soundness) and its false-reject rate to be non-zero but bounded (conservatism), and we expect the raw-hull ablation to show a substantially higher false-accept rate, confirming that the sound propagation and witness verification are necessary for the certificate's meaning.

**vs. FARA without sampler constraint feedback** — On scenes where the initial policy's failures cluster in one geometric region, the no-feedback ablation rejects infeasible proposals but resamples from the unchanged distribution, so it burns rounds like the fixed-clearance baseline. We expect full FARA to reach a given success level in fewer rounds and with a lower rejection rate per round, isolating the sampler-projection contribution.

**vs. FARA with raw convex-hull propagation (no Taylor-model, no CEGAR)** — On repairs whose witness lies inside the hull but outside the true reachable tube, the raw-hull ablation accepts them and distills an infeasible witness, so its success rate on the collision-prone subset should be lower than full FARA's and its false-accept rate against the oracle should be higher. This isolates the contribution of sound propagation and witness verification to the certificate's validity.

### What would falsify this idea

If FARA's gain over AeroRSI continuation verification is uniform across collision-prone and open-corridor repair subsets rather than concentrated where continuation checks are expected to fail, then the geometric certificate is not doing the causal work claimed and the improvement comes from something else. Likewise, if the no-feedback ablation matches full FARA on rejection rate per round, the sampler-projection mechanism is inert and the design's second stated choice is unsupported; and if the raw-hull ablation matches full FARA on false-accept rate against the oracle, the sound-propagation and witness-verification machinery is unnecessary and the load-bearing assumption is not actually load-bearing.

## References

1. AeroRSI: Recursive Self-Improvement for Aerial Vision-Language Navigation
