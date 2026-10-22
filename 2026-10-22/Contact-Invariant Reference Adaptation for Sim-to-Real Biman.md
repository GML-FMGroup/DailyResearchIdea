# Contact-Invariant Reference Adaptation for Sim-to-Real Bimanual Retargeting

## Motivation

Retargeting pipelines such as ManipTrans and DONIC both treat simulation-reconstructed human motion as a sufficient reference for policy training, so their reference trajectories carry no explicit guarantee that contact points, contact timing, and force profiles remain feasible once the sim-to-real gap (friction, compliance, actuator latency, sensing noise) is applied. This is a shared structural gap rather than a per-paper bug: the reference is validated only inside the same simulator in which the policy is later trained, so any reference whose contact structure is fragile under real dynamics is silently accepted and its brittleness is absorbed — or hidden — by downstream residual/RL correction. DONIC in particular recycles successful simulated executions as demonstrations without any test of whether the reference's contact schedule would survive real-world contact and dynamics constraints.

## Key Insight

The structural property that makes a retargeted bimanual reference transferable is invariance of its discrete contact schedule and per-contact force-direction cone under the sim-to-real parameter gap, and this invariance can be certified before policy training by requiring the reference to remain dynamically feasible for every adversary in a learned, physically-parameterized gap set rather than for a single nominal simulator.

## Method

**What it is.** CIRRA (Contact-Invariant Reference Repair and Adaptation) is a pre-training layer that takes a retargeted bimanual reference trajectory $\tau = \{(q_t^{L}, q_t^{R}, o_t)\}$ (left/right hand joints, object pose) plus a nominal simulator, and returns a repaired reference $\tau'$ whose contact schedule and force profile are certified invariant over a learned sim-to-real gap set $\mathcal{G}$. The policy is then trained on $\tau'$ with the existing DONIC tracker, unchanged.

**Load-bearing assumption (stated explicitly).** CIRRA's certificate is a guarantee *only over the explicitly enumerated, hardware-calibrated gap family* $\mathcal{G}$ defined by the bounded ranges below. It is **not** a guarantee of invariance on the real robot for gaps outside $\mathcal{G}$ (e.g., unmodeled tendon compliance, soft-contact departure from Coulomb friction). The certificate $C$ therefore reports both (i) the worst-case residual over $\mathcal{G}$ and (ii) the residual on a held-out set of hardware-measured gaps $\mathcal{H}$ that the adversary never sees, including at least one out-of-family perturbation (tendon-compliance proxy, soft-contact proxy). A clip is certified only if both residuals are below $\epsilon$; if the held-out residual exceeds $\epsilon$, the clip is flagged rather than certified.

**How it works.**

```pseudocode
Input: retargeted reference tau (T steps), nominal sim S0, gap-set learner G_phi,
       hardware-calibrated gap family G (bounded ranges below),
       held-out hardware gap set H (disjoint from G, includes >=1 out-of-family perturbation),
       invariance threshold eps=0.02 m / 3 N, repair budget K=4 iterations
Output: repaired reference tau' with certificate C = {max_{g in G} r(tau',g), max_{h in H} r(tau',h), contact schedule, force margins}

Gap family G (hardware-calibrated bounds):
  friction mu_s, mu_k in [0.4, 1.2]
  actuator delay tau_act in [0, 40] ms
  contact stiffness k_c in [200, 5000] N/m
Held-out set H: 20 hardware-measured gap samples (friction, delay, stiffness) from the target robot,
  plus 4 out-of-family perturbations: 2 tendon-compliance proxies (series-elastic element with
  stiffness 100-300 N/m inserted at the actuator) and 2 soft-contact proxies (Hunt-Crossley
  contact model with damping 0.1-0.5 replacing rigid Coulomb).

PHASE 1 - Gap-set learning (adversarial, per reference clip)
  Initialize phi so G_phi outputs gap samples g = (mu_s, mu_k, tau_act, k_c) drawn
    from the bounded ranges above.
  for epoch in 1..N_adv (N_adv=200):
      g ~ G_phi                       # adversary proposes a gap
      rollout tau in S(g)             # simulator with gap g applied
      compute contact residual r(tau,g):
        - contact-point drift: max_t ||p_c^ref(t) - p_c^roll(t)||
        - force-cone violation: max_t max(0, angle(f(t), n(t)) - atan(mu_s))
        - schedule slip: # of contact on/off flips not in reference
      update phi by REINFORCE on r(tau,g)  # adversary maximizes residual
      if r(tau,g) > eps for all recent g: break   # tau is fragile

PHASE 2 - Reference repair (only if Phase 1 flagged fragility)
  for k in 1..K (K=4):
      Sample worst-case gap set G* = top-M gaps by r(tau,g) (M=10)
      Solve per-contact-point adjustment:
        - shift contact onset/offset by delta_t in [-2,2] frames
        - adjust wrist approach offset by delta_p in [-1,1] cm
        - re-solve finger joint targets by IK to new contact point
      Minimize  sum_{g in G*} r(tau_k, g)  +  lambda * ||tau_k - tau_0||^2
        with lambda=0.1 (stay close to human reference)
      tau_{k+1} = argmin
  Return tau' = tau_K if certified (r < eps for all g in G* AND r < eps for all h in H), else flag clip.

PHASE 3 - Certified reference set
  Emit tau' with certificate C = {max_{g in G} r(tau',g), max_{h in H} r(tau',h), contact schedule, force margins}.
  Downstream DONIC PPO+BC trains only on certified clips.
```

**Why this design.** We chose a *learned adversarial gap set* over uniform domain randomization because uniform sampling over friction/delay/stiffness spends most budget on gaps that no real robot will ever exhibit, whereas an adversary concentrates samples on the boundary where the reference's contact schedule actually breaks; the cost is that the learned gap set can overfit to the current reference clip, which we mitigate by re-initializing $\phi$ per clip and bounding gap ranges with hardware-measured priors. We chose *reference repair in the retargeting space* over letting the downstream residual policy absorb the gap (as ManipTrans and DexH2R do) because a residual policy trained in nominal sim will itself be miscalibrated for real contact forces — repairing the reference removes the fragility at its source; the cost is that aggressive repair can pull the trajectory away from the human demonstration, hence the $\lambda$ anchor. We chose *discrete contact-schedule edits* (onset/offset shifts, wrist offsets) over continuous whole-trajectory re-optimization because contact on/off structure is the quantity that must be invariant, and editing it discretely keeps the human motion prior intact; the cost is a coarser repair space that may fail on clips needing large re-planning, which is why we emit an explicit failure flag rather than silently accepting. A domain expert would *not* describe this as "domain randomization + retargeting," because standard DR randomizes the *policy's* training environment, whereas CIRRA randomizes the *reference's* validation environment and edits the reference itself before any policy exists — the adversary's output is a certificate on the reference, not a training distribution.

**Why it measures what we claim.** The contact-point drift $\max_t \|p_c^{ref}(t)-p_c^{roll}(t)\|$ measures *sim-to-real contact invariance* because it is computed under the worst-case gap in $\mathcal{G}$, and the named assumption is that the learned gap set's worst case upper-bounds the true hardware gap; this assumption fails when the real robot exhibits a gap outside the bounded ranges (e.g., unmodeled tendon compliance), in which case the metric reflects only invariance over the modeled gap family. The force-cone violation $\max_t \max(0, \angle(f,n)-\arctan\mu_s)$ measures *force-profile transferability* because a contact whose required force direction exits the friction cone under reduced $\mu_s$ cannot be sustained on real surfaces, and the assumption is that Coulomb friction with the sampled $\mu_s$ is an adequate model of real contact; this assumption fails on soft or compliant objects, where the metric instead reflects conservative friction margins. The schedule-slip count measures *contact-timing robustness* because on/off flips under actuator delay $\tau_{act}$ indicate that the reference's contact events are not temporally separated by more than the real latency, and the assumption is that delay is the dominant timing perturbation; this assumption fails under high-frequency force feedback, where the metric reflects only open-loop timing margin. The repair objective's anchor term $\lambda\|\tau_k-\tau_0\|^2$ measures *fidelity to the human demonstration* because it penalizes deviation from the original retargeted reference, and the assumption is that Euclidean joint-space distance is a valid proxy for motion naturalness; this assumption fails for kinematically redundant poses, where the metric instead reflects joint-space proximity rather than perceptual similarity. The held-out residual $\max_{h\in\mathcal{H}} r(\tau',h)$ measures *out-of-family robustness* because it is evaluated on hardware-measured gaps and out-of-family perturbations never seen by the adversary, and the assumption is that $\mathcal{H}$ is representative of the target robot's true gap; this assumption fails if the hardware gap drifts after calibration, in which case the metric reflects only the calibrated snapshot.

## Contribution

(1) A pre-training reference-certification layer (CIRRA) that learns a physically-bounded adversarial sim-to-real gap set per retargeted bimanual clip and repairs the reference's discrete contact schedule and wrist offsets until contact-point drift, force-cone violation, and schedule slip fall below hardware-derived thresholds. (2) The design principle that sim-to-real invariance of a retargeted reference is a property of the reference's contact schedule under worst-case gap, not of the downstream policy's training distribution — separating reference certification from policy training. (3) A certified-reference dataset interface that emits per-clip certificates (worst-case residual, contact schedule, force margins) usable by any downstream bimanual tracker such as DONIC.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | BiHOI bimanual sequences (229 objects), stratified subset of 40 clips selected by a cheap fragility proxy (contact-onset spacing < 40 ms or force-cone margin < 5°) | Matches DONIC's native tracking benchmark; subset makes hardware trials feasible within 3-6 months |
| Primary metric | Certified contact-schedule survival rate on hardware (contact on/off events detected via fingertip F/T thresholds: >2 N normal force for onset, <0.5 N for offset) | Directly tests invariance claim, not proxy |
| Baseline | DONIC tracker on unrepaired retargeted reference | Isolates effect of reference repair |
| Baseline | ManipTrans residual-learning transfer | Tests whether residual policy absorbs fragility |
| Baseline | Uniform domain-randomization-trained tracker (same gap ranges as G, 10k samples) | Tests learned adversary vs naive randomization |
| Ablation-of-ours | CIRRA without Phase 2 repair (certificate only) | Separates gap-set detection from repair benefit |
| Ablation-of-ours | CIRRA with held-out set H disabled (certificate over G only) | Tests whether out-of-family flagging changes hardware survival |
| Calibration | Measure hardware gap on target robot: friction via sliding tests on 5 surfaces, latency via step-response, stiffness via indentation; report coverage of G against measured values | Grounds the upper-bound assumption empirically |

### Why this setup validates the claim

The central claim is that reference-level contact fragility, not policy-level robustness, is what breaks bimanual sim-to-real transfer. That claim is only falsifiable if we can separate the reference's contribution from the policy's. DONIC-on-unrepaired-reference isolates exactly that: same tracker, same training budget, only the reference differs, so any gain must come from repair. ManipTrans tests the competing hypothesis directly — if a residual policy trained in nominal sim absorbs the gap as well as a repaired reference does, our premise that repair must happen upstream is wrong. Uniform-DR tests whether the learned adversary is doing real work or just re-deriving what naive randomization already covers. The ablation (detection without repair) checks that the certificate alone, with no edits, does not already fix hardware survival — if it did, repair is unnecessary. The held-out-set ablation checks whether the out-of-family flagging step (the explicit acknowledgment that the certificate is only over the calibrated family) changes hardware survival, which is the direct test of the load-bearing assumption. The primary metric is contact-schedule survival on hardware rather than tracking error, because the idea's own causal story says contact on/off structure is the invariant that must hold; a tracker can have low joint error while flipping contacts, and that is precisely the failure CIRRA predicts and claims to prevent. The calibration experiment reports the measured hardware gap and the coverage of G, so the reader can see whether the bounded family is plausible for the target robot.

### Expected outcome and causal chain

**vs. DONIC on unrepaired reference** — On clips where a contact onset sits within the actuator-delay window (e.g., a regrasp that assumes sub-20 ms latency), the baseline tracker produces contact flips and object slip on hardware because its training never saw the reference's schedule break under delay. Our method shifts the onset out of the delay window and re-solves IK before training, so we expect a large survival-rate gap concentrated on delay-fragile clips and near-parity on clips already temporally separated.

**vs. ManipTrans residual learning** — On a case where reduced friction pushes the required force direction outside the cone, the baseline's residual policy corrects in nominal sim and stays miscalibrated for real friction because its correction was learned against the wrong contact model. Our method detects the cone violation under the worst-case gap and edits the wrist approach to restore margin, so we expect a friction-margin gap on low-mu clips while both methods agree on high-friction clips.

**vs. Uniform-DR-trained tracker** — On a clip whose fragility lives at a narrow corner of the gap space (high stiffness plus moderate delay), the baseline's uniform samples rarely land there and the policy remains fragile because its training distribution underweights that corner. Our adversary concentrates samples at that boundary and flags or repairs the clip, so we expect our gain to be concentrated on the small subset of corner-fragile clips rather than spread uniformly.

**vs. CIRRA without held-out set H** — On clips whose fragility is driven by out-of-family perturbations (tendon compliance or soft contact), the ablation certifies them because its residual over G is below eps, but hardware survival is low because the real gap lies outside G. Full CIRRA flags these clips via the held-out residual and excludes them from training, so we expect full CIRRA to have higher survival on the flagged subset while both methods agree on in-family clips.

### What would falsify this idea

If the gain over DONIC-on-unrepaired-reference is uniform across all clips rather than concentrated on the clips Phase 1 flags as fragile, the repair is not doing what we claim and the certificate is not identifying the true failure mode. Equally, if the ablation (certificate without repair) matches full CIRRA on hardware survival, then detection alone suffices and reference repair is unnecessary; and if the held-out-set ablation matches full CIRRA, then out-of-family flagging is unnecessary and the certificate over G alone is sufficient.

## References

1. DONIC: Generalist Dexterous Bimanual Tracking Control from Human Demonstrations
2. ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning
3. Object-Centric Dexterous Manipulation from Human Motion Data
4. DexH2R: Task-Oriented Dexterous Manipulation From Human to Robots
