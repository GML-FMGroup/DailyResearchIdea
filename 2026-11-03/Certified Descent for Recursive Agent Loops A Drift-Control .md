# Certified Descent for Recursive Agent Loops: A Drift-Control Criterion via Distribution-Shift-Bounded Potentials

## Motivation

Recursive self-referential optimization loops in LLM code-generation agents — the alternating harness-search/policy-update rounds of Self-HPO (Self-Evolving Coding Agents via Harness-Policy Optimization), the prompt-mutation loop of Promptbreeder, and the ReAct repair iterations of SWE-agent — all run an iteration operator with no monotone or bounded quantity governing it, so the loop can drift, silently reorder its candidate rankings, or keep spending compute after the marginal improvement has vanished. This is a structural gap across these trees, not a local bug: none of them attaches an external grounding signal to the iteration itself, so 'the loop is still improving' is asserted by the heuristic rather than certified by any observable. The root cause is that improvement is measured against a moving belief (the policy's own failure analysis, or a refined candidate set), and with no distribution-shift term the measured improvement is not comparable across iterations.

## Key Insight

A Lyapunov-type potential over the joint (policy, harness/candidate-set) state, whose per-iteration decrease is certified by a computable distribution-shift bound between consecutive belief states, makes the loop's improvement observable and comparable across iterations — and when the certified decrease exceeds the actually observed improvement, the loop is provably not making progress and can be damped or halted, giving a monotone-consistency invariant for candidate ranking that no existing recursive agent loop possesses.

## Method

**What it is.** We introduce **CDL (Certified Descent Loop)**, a drift-control wrapper around any recursive self-referential optimization loop (harness-policy co-evolution as in Self-HPO, or execution-free repair iteration as in SWE-agent's ReAct loop). Inputs: the loop operator (a black-box function mapping a joint state to a successor state), a candidate set per round, and an evaluation set of held-out tasks. Outputs: a per-round decision in {accept, damp, halt} plus a certified lower bound on the potential decrease, and a monotone-consistent ranking of the candidates in the current set.

**Load-bearing assumption (stated explicitly).** The potential `Phi(s) = L_eval(pi, H) + lambda * D(s, s_ref)` is assumed Lipschitz in the harness/belief state with a constant `beta` that is stable across rounds, so that `beta * KL_estimate` is a valid upper bound on the loss change attributable to distribution shift for every subsequent round and every candidate. This assumption is load-bearing: if it fails (e.g., a single harness token flips many tasks, making the loss non-Lipschitz), the certificate is not a bound but an unvalidated extrapolation. To keep the certificate honest under non-Lipschitz jumps, we replace the single global `beta` with a per-round, per-candidate local continuity modulus and a conservative high-confidence KL upper bound, as specified below.

**How it works.**

```pseudocode
# Joint state s_t = (pi_t, H_t, C_t): policy, selected harness, candidate set
# Potential Phi(s) = L_eval(pi, H) + lambda * D(s, s_ref)
#   L_eval  = held-out task loss (execution pass-rate deficit)
#   D       = distribution shift between consecutive belief states
#   s_ref   = anchor state at last accepted round
#   lambda  = 1.0 (fixed; not tuned per benchmark)

# Initialization (once, on a dev split of 128 tasks, disjoint from probe set P):
#   - Measure empirical Lipschitz constant beta_0 of L_eval under harness perturbation:
#     perturb harness by up to 5 token edits, record max |dL_eval| / KL, set beta_0 = 95th percentile.
#   - Set eta = 0.01 (margin), alpha = 0.5 (damping mix), n_rollouts = 512 (probe rollouts per round).

for t in rounds:
    # 1. Operator step: run H-step then P-step (or one ReAct repair pass)
    s_next = Operator(s_t)                 # produces pi_{t+1}, H_{t+1}, C_{t+1}

    # 2. Measure observed potential change on a FIXED held-out probe set P (|P| = 256 tasks)
    dPhi_obs = Phi(s_next; P) - Phi(s_t; P)

    # 3. Compute the distribution-shift certificate
    #    KL between the failure-trajectory distributions of s_t and s_next,
    #    estimated on P via a two-sample estimator with n=512 probe rollouts.
    #    Use a conservative high-confidence upper bound KL_upper = KL_hat + z * SE(KL_hat),
    #    where z = 2.33 (99% one-sided) and SE is the bootstrap standard error over 100 resamples.
    kl_hat, kl_upper = KL_estimate_upper(traj_dist(s_t), traj_dist(s_next), P, n=512, z=2.33)

    #    Per-round, per-candidate local continuity modulus beta_local:
    #    For the actual harness perturbation applied this round (H_t -> H_{t+1}),
    #    compute worst-case loss change over a small adversarial perturbation set
    #    around s_t: sample 16 harness variants within KL <= kl_upper of s_t,
    #    evaluate L_eval on P, and set beta_local = max |dL_eval| / kl_upper.
    #    If kl_upper == 0, set beta_local = beta_0.
    beta_local = local_modulus(s_t, s_next, P, kl_upper, n_variants=16, fallback=beta_0)

    #    Certified decrease bound: how much Phi could have improved if the
    #    shift were entirely responsible for the apparent change
    dPhi_cert = dPhi_obs - beta_local * kl_upper

    # 4. Decision rule (closed-form, no learned controller)
    if dPhi_cert < -eta:      # certified descent beyond margin eta=0.01
        accept(s_next); s_ref = s_next
    elif dPhi_cert < 0:       # observed improvement not certified
        damp(s_next); s_t = mix(s_t, s_next, alpha=0.5)
    else:                     # certified bound exceeds observed improvement
        halt()

    # 5. Monotone-consistent candidate ranking
    #    Rank candidates by certified potential, not raw score; a candidate
    #    that only wins under a shifted belief is demoted to the anchor's rank.
    #    For each candidate c, compute beta_local(c) using the same local modulus procedure
    #    with the perturbation from s_ref to c.
    C_ranked = sort(C_{t+1}, key=lambda c: Phi(c; P) + beta_local(c) * KL_upper(c, s_ref))
```

The potential's shift term `D(s, s_ref)` is the KL between the failure-trajectory distribution of the current state and that of the last accepted anchor, estimated by a two-sample test on the fixed probe set. `beta` is replaced by the per-round, per-candidate `beta_local` computed as above; the initialization `beta_0` is used only as a fallback when `kl_upper == 0`.

**Why this design.** We chose a fixed held-out probe set over re-sampling the training set each round because re-sampling makes the potential non-comparable across rounds (the measurement surface moves with the belief), accepting the cost that the probe set can be overfit after many rounds — which we mitigate by holding out a disjoint probe set for the halt decision. We chose a closed-form decision rule over a learned controller because the whole point is to certify progress against an external signal, and a learned controller would reintroduce the self-referential unreliability the method is meant to remove; the cost is that the threshold `eta` and the local modulus procedure must be calibrated per benchmark. We chose to demote candidates under a shifted belief rather than re-rank by raw score because the meta-gap is precisely that belief refinement can silently reorder candidates; the cost is that genuinely improved candidates from a large harness change may be temporarily suppressed until the shift term decays. This is not a variant of Promptbreeder or Self-HPO: those evolve a population or alternate H/P steps with no external certificate, whereas CDL attaches a computable distribution-shift bound to every iteration and halts on its violation — the certificate, not the evolution, is the contribution.

**Why it measures what we claim.** The potential `Phi` measures the motivation-level concept *progress of the recursive loop* because the held-out probe loss is a fixed external surface and the KL term is the only channel through which belief refinement can inflate apparent improvement; this assumption fails when the probe set is unrepresentative of the task distribution, in which case `Phi` reflects probe-set fit rather than true agent progress. The distribution-shift certificate `beta_local * kl_upper` measures *how much of the observed improvement could be an artifact of belief drift* because `beta_local` is the per-round local continuity modulus of the probe loss under the actual harness perturbation, making the product an upper bound on the loss change attributable to the shift alone; this assumption fails when the loss is non-Lipschitz in the harness (e.g., a single harness token flips many tasks), in which case the certificate is conservative and the loop damps more than necessary. The monotone-consistent ranking measures *whether a candidate's apparent win survives belief refinement* because ranking by `Phi(c) + beta_local(c) * KL_upper(c, s_ref)` charges each candidate the shift cost of the belief under which it was evaluated; this assumption fails when two candidates are evaluated under genuinely different task subsets, in which case the ranking reflects subset difficulty rather than candidate quality.

## Contribution

(1) CDL, a drift-control wrapper for recursive self-referential agent loops that computes a distribution-shift-bounded Lyapunov potential over the joint (policy, harness, candidate-set) state and emits a closed-form accept/damp/halt decision per iteration. (2) A monotone-consistency invariant for candidate ranking under belief refinement, implemented by charging each candidate the shift cost of the belief under which it was evaluated, which prevents the silent reordering that plagues unguarded recursive loops. (3) An empirical protocol for calibrating the Lipschitz constant `beta` and margin `eta` from a single harness-perturbation sweep, applicable to any loop operator without retraining.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | SWE-Bench Pro (held-out + commercial splits) | Long-horizon tasks where belief drift compounds across rounds |
| Primary metric | Held-out pass-rate at fixed round budget | External surface that drift cannot inflate |
| Baseline | Self-HPO (alternating H/P, no certificate) | Tests whether certification beats uncertified co-evolution |
| Baseline | Harness-only optimization | Tests whether policy co-evolution adds value beyond harness search |
| Baseline | SWE-agent ReAct repair loop (no halt) | Tests whether halting on drift beats indefinite repair |
| Ablation-of-ours | CDL without KL certificate (accept on raw dPhi) | Isolates the certificate's contribution from the loop wrapper |
| Ablation-of-ours | CDL with global beta (single initialization estimate) | Tests whether per-round local modulus is necessary |
| Ablation-of-ours | CDL with point-estimate KL (no upper bound) | Tests whether conservative KL upper bound matters |

### Why this setup validates the claim

The central claim is that a computable distribution-shift certificate, attached to every iteration of a self-referential loop, converts silent belief-driven reordering into a detectable and correctable signal, yielding better held-out pass-rate at a fixed round budget. SWE-Bench Pro is the right surface because its long-horizon tasks force many repair rounds, which is exactly where belief drift accumulates; a short-horizon benchmark would let every method converge before drift matters. The Self-HPO baseline tests the sub-claim that certification beats uncertified co-evolution — if CDL's gain over Self-HPO vanishes, the certificate is decorative. Harness-only optimization tests whether the policy side of the loop contributes anything the certificate can protect, and the no-halt ReAct baseline tests whether halting itself (rather than damping alone) is doing the work. The ablation — CDL with the KL term removed but the accept/damp/halt wrapper retained — is the decisive control: it isolates the certificate from the surrounding machinery, so any gain shared between CDL and the ablation cannot be attributed to the certificate. The additional ablations (global beta, point-estimate KL) test whether the per-round local modulus and conservative upper bound are necessary for the certificate to be honest under non-Lipschitz jumps. Held-out pass-rate at a fixed round budget is the correct primary metric because it is measured on a surface the loop never optimizes against; raw training-set or probe-set scores would be exactly the quantity the method claims can be inflated, so they cannot adjudicate the claim. A method that improves only probe-set fit while held-out pass-rate stays flat would be the predicted failure of our own approach, which is why the metric must be external.

### Expected outcome and causal chain

**vs. Self-HPO** — On a case where the harness is revised mid-run to a form that happens to fit the current probe set but shifts the failure-trajectory distribution, Self-HPO accepts the revision on raw score and continues optimizing against a moved belief, so later rounds compound the drift and held-out pass-rate plateaus or regresses. Our method instead computes the KL between pre- and post-revision failure trajectories on the fixed probe set, charges the observed gain `beta_local * kl_upper`, and damps the step when the certified decrease is not negative, keeping the anchor stable. We expect a visible gap concentrated on the subset of tasks where harness revisions are large relative to policy revisions, with near-parity on tasks whose revisions are small.

**vs. Harness-only optimization** — On a case where the harness is already near-optimal but the policy's repair behavior is the bottleneck, harness-only search keeps perturbing harness tokens with no channel to fix the policy, so it burns rounds on harness variants that cannot recover the failing tasks. Our method's joint operator moves both policy and harness, and the certificate's shift term specifically flags rounds where the harness moved without a corresponding policy improvement, redirecting the loop. We expect harness-only to saturate early at a lower held-out pass-rate while CDL keeps improving for several more rounds, with the gap widening as round budget grows.

**vs. SWE-agent ReAct repair loop (no halt)** — On a case where the repair loop's own belief about the bug has drifted after several failed attempts, the no-halt loop keeps issuing repairs conditioned on the drifted belief, producing confident but wrong patches that never trigger a stop condition. Our method's halt rule fires when the certified decrease bound is non-negative, ending the loop at the anchor state rather than submitting a drifted patch. We expect CDL to show fewer wasted rounds on unsolvable instances and a higher pass-rate on the solvable subset where drift would otherwise have consumed the round budget.

**vs. CDL without KL certificate** — This ablation accepts on raw `dPhi_obs` with no shift penalty, so it should behave like Self-HPO on drift-heavy subsets. We expect CDL to outperform it specifically on tasks where harness revisions are large, confirming the certificate's role.

**vs. CDL with global beta** — Using a single initialization `beta_0` may under- or over-penalize later rounds with larger harness revisions. We expect the per-round local modulus to yield higher held-out pass-rate on rounds with large revisions, and to reduce the violation rate of the certified bound (measured as fraction of rounds where `dPhi_obs < -beta_local * kl_upper` but held-out pass-rate decreases).

**vs. CDL with point-estimate KL** — Using `kl_hat` instead of `kl_upper` may under-penalize due to estimator noise, leading to more false accepts. We expect the conservative upper bound to reduce false accepts and improve held-out pass-rate, at the cost of slightly more damping (fewer accepted rounds).

### What would falsify this idea

If CDL's gain over the no-certificate ablation is uniform across all task subsets rather than concentrated on the subsets where harness revisions are large relative to policy revisions, the certificate is not the causal mechanism and the central claim is wrong. Likewise, if the halt rule fires predominantly on easy instances where the baseline would have succeeded anyway, the certificate is measuring noise rather than drift.

## References

1. Self-Evolving Coding Agents via Harness-Policy Optimization
2. SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?
3. Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution
4. OpenHands: An Open Platform for AI Software Developers as Generalist Agents
5. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
6. BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions
7. Large Language Models Can Self-Improve
8. Eliminating Meta Optimization Through Self-Referential Meta Learning
