# Damped Cross-Level Fixed-Point Optimization for Two-Level Agent Pipelines

## Motivation

Two-level alternating procedures — harness/policy in Self-HPO (Self-Evolving Coding Agents via Harness-Policy Optimization), and plan/generate in multi-agent script generation — have no fixed-point guarantee: each level's update shifts the other's effective objective, so the iteration converges to a sequence of per-level optima rather than a joint equilibrium. Self-HPO's closed loop of failure analysis, harness selection, and RL fine-tuning assumes the harness search set remains meaningful after the policy moves, but the policy update invalidates the very failure distribution the harness was selected against. The structural problem is not model capacity but the absence of any stability certificate for coupled alternating optimization.

## Key Insight

If the cross-level Jacobian of each level's objective with respect to the other level's state is estimated online and kept contractive by a damping schedule, the joint two-level map becomes a contraction with an identifiable fixed point, so alternation converges to a single equilibrium instead of oscillating between incompatible per-level optima.

## Method

**What it is.** CLFPO (Cross-Level Fixed-Point Optimization) is a wrapper around any two-level alternating pipeline (harness/policy, plan/generate). It takes as input two update operators U0 (level-0, e.g. harness selection) and U1 (level-1, e.g. policy/plan update) together with their loss evaluations, and outputs a damped update sequence whose joint fixed point is certified online by an estimated cross-level Lipschitz constant L_cross. **Load-bearing assumption (stated explicitly):** the finite-difference probe L_cross is a valid contraction constant for the joint two-level map only if (A) each level's objective is locally Lipschitz in the other level's state and epsilon-perturbations are representative of the state manifold, and (B) the nominal update operators U0, U1 are themselves Lipschitz, so that the damped joint map has Lipschitz constant bounded by (1-eta_t) + eta_t*L_cross. Assumption (A) fails when the other level's state space is discrete or high-dimensional enough that epsilon-perturbations are not representative (e.g., argmax harness selection over a discrete search set, discrete token/plan choices), in which case L_cross reflects local probe noise rather than a global contraction constant. Assumption (B) fails when the nominal updates are not Lipschitz (e.g., argmax harness selection), in which case eta_t reflects a smoothed surrogate rather than the true update. To keep the certificate meaningful under these failures, CLFPO operates on a continuous relaxation of the discrete levels and reports the relaxation gap as a measured quantity (see step 2b and step 5 below).

**How it works.**
```pseudocode
Input: update operators U0, U1; damping base eta0=0.5; probe budget K=3;
       safety margin rho=0.9; window W=5; relaxation temperature tau=1.0 (annealed to 0.1);
       stationarity tolerance tol_stat=1e-3; decrease tolerance tol_dec=1e-4
Initialize states s0, s1; residual history R = []
for round t = 1..T:
    # 0. Continuous relaxation of discrete levels (softmax harness selection, temperature-annealed policy)
    U0_tau = softmax_relax(U0, tau_t)   # tau_t = max(0.1, 1.0 * 0.95^t)
    U1_tau = softmax_relax(U1, tau_t)
    # 1. Nominal per-level updates on the relaxed operators
    s0_nom = U0_tau(s0, s1)
    s1_nom = U1_tau(s1, s0)
    # 2. Cross-level Lipschitz probe on the relaxed operators
    #    Perturbation delta is sampled in the continuous embedding of the discrete state:
    #    for harness (level-0), delta = eps * unit_vector in the harness embedding space (dim d_h=64);
    #    for policy (level-1), delta = eps * unit_vector in the policy parameter space (dim d_p=256).
    #    eps = 1e-3 * ||s||_2 (relative perturbation).
    for k in 1..K:
        delta = eps * unit_vector(sample)
        L0_hat[k] = || loss0(U0_tau(s0, s1+delta)) - loss0(U0_tau(s0, s1)) || / ||delta||
        L1_hat[k] = || loss1(U1_tau(s1, s0+delta)) - loss1(U1_tau(s1, s0)) || / ||delta||
    L_cross = max(median(L0_hat), median(L1_hat))
    R.append(L_cross); L_cross = median(R[-W:])   # smooth over window
    # 2b. Relaxation gap: measure ||U0_tau - U0_discrete||_1 + ||U1_tau - U1_discrete||_1
    #     where U0_discrete = argmax harness selection, U1_discrete = argmax plan/token choice.
    gap_t = ||U0_tau(s0,s1) - U0_discrete(s0,s1)||_1 + ||U1_tau(s1,s0) - U1_discrete(s1,s0)||_1
    # 3. Curvature-aware damping schedule
    eta_t = eta0 / (1 + L_cross)          # contraction requires eta_t * L_cross < rho
    if eta_t * L_cross >= rho:
        eta_t = rho / (L_cross + 1e-8)    # hard clamp to preserve contraction
    # 4. Damped joint update
    s0 = (1 - eta_t) * s0 + eta_t * s0_nom
    s1 = (1 - eta_t) * s1 + eta_t * s1_nom
    # 5. Two-sided certificate (replaces residual-only stall test)
    res = ||s0 - s0_prev|| + ||s1 - s1_prev||
    # (a) decrease condition on the joint objective
    dec_ok = (loss0(s0,s1) + loss1(s1,s0)) <= (loss0(s0_prev,s1_prev) + loss1(s1_prev,s0_prev)) - tol_dec * res
    # (b) cross-level stationarity: gradient of each level's loss w.r.t. the other level's state
    stat0 = ||grad_{s1} loss0(s0,s1)||_2
    stat1 = ||grad_{s0} loss1(s1,s0)||_2
    stat_ok = (stat0 < tol_stat) and (stat1 < tol_stat)
    if res < tol and eta_t * L_cross < rho and dec_ok and stat_ok:
        emit "joint fixed point certified at round t (relaxation gap = gap_t)"
        break
return s0, s1, certificate(round, L_cross, eta_t, gap_t, stat0, stat1)
```

**Why this design.** We chose an online finite-difference probe over a learned Jacobian estimator because the probe requires no extra model capacity and directly measures the quantity the contraction condition depends on, accepting the cost of K extra loss evaluations per round. We chose a median-over-window smoothing of L_cross over a single-round estimate because single-round probes are noisy when the level update is stochastic, accepting a slower reaction to genuine curvature changes. We chose a hard clamp eta_t = rho/L_cross over a fixed learning rate because the contraction condition eta_t * L_cross < rho must hold at every round for the fixed-point argument to apply, accepting that early rounds may move very slowly when L_cross is large. We chose to damp both levels symmetrically over asymmetric damping because the cross-level coupling is bidirectional in harness/policy and plan/generate alike. We chose to run the probe and the damped update on a softmax relaxation of the discrete levels (temperature-annealed from tau=1.0 to 0.1) and to report the relaxation gap gap_t alongside the certificate, because the nominal argmax harness selection and discrete plan/token choices are not Lipschitz and the convex-combination bound (1-eta_t)+eta_t*L_cross does not apply to them; the relaxation makes the bound applicable to the map actually iterated, and gap_t makes explicit how far that map is from the discrete pipeline. We chose a two-sided certificate (joint-objective decrease plus cross-level stationarity) over a residual-norm stall test because a small residual can also occur at a saddle or a discrete jump, and the two-sided test cannot misreport those as a certified joint fixed point. A domain expert would not describe this as a variant of Self-HPO, Promptbreeder, or SPIN: those methods prescribe what each level optimizes but never estimate or control the cross-level coupling that determines whether alternation converges; CLFPO's load-bearing quantity is L_cross, which none of those methods computes.

**Why it measures what we claim.** The finite-difference probe L_cross measures the cross-level coupling strength because it assumes the level objective is locally Lipschitz in the other level's state with a constant that the probe estimates to first order; this assumption fails when the other level's state space is discrete or high-dimensional enough that epsilon-perturbations are not representative, in which case L_cross reflects local probe noise rather than a global contraction constant. The damping factor eta_t measures the contraction margin because it assumes the joint update is a convex combination of the nominal per-level maps, making the joint map's Lipschitz constant bounded by (1-eta_t) + eta_t * L_cross; this assumption fails when the nominal updates are not Lipschitz (e.g., argmax harness selection), in which case eta_t reflects a smoothed surrogate rather than the true update. The residual res measures proximity to the joint fixed point because it assumes the damped iteration is monotone in residual norm under contraction; this assumption fails near a saddle, in which case res reflects a local stall rather than a certified equilibrium. The relaxation gap gap_t measures the distance between the relaxed map actually iterated and the discrete pipeline because it assumes the softmax relaxation is a faithful surrogate at temperature tau_t; this assumption fails when tau_t is too high (relaxation far from argmax) or too low (gradients vanish), in which case gap_t reflects the annealing schedule rather than a meaningful surrogate error. The cross-level stationarity statistics stat0, stat1 measure proximity to a joint stationary point because they assume the level losses are differentiable in the other level's state; this assumption fails when the other level's state is discrete and has no gradient, in which case stat0, stat1 are computed on the relaxed state and reflect surrogate stationarity rather than true discrete stationarity.

## Contribution

(1) CLFPO, a damping wrapper for two-level alternating pipelines that estimates a cross-level Lipschitz constant online from perturbation probes and uses it to set a curvature-aware damping schedule with a residual-based fixed-point certificate. (2) The design principle that the cross-level Jacobian, not per-level learning rates, is the load-bearing quantity for stable alternation, demonstrated by comparing certified vs. uncertified rounds on a two-level script-generation pipeline. (3) A probe protocol (K perturbation samples, median-over-window smoothing, hard contraction clamp) that is model-agnostic and requires no additional trainable parameters.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multi-agent script generation benchmark (MovieAgent-style), 200 held-out scripts | Two-level alternation with genuine cross-level coupling |
| Primary metric | Joint fixed-point residual at convergence round | Directly measures the quantity CLFPO certifies |
| Secondary metric | Relaxation gap gap_t at convergence round | Measures surrogate-vs-discrete distance of the certificate |
| Secondary metric | Downstream script quality: MovieAgent coherence and coverage scores | Tests whether residual reduction is non-vacuous |
| Baseline | Undamped alternating optimization | Tests whether damping/contraction control is load-bearing |
| Baseline | Fixed learning-rate alternation (Self-HPO-style, eta=0.1) | Tests whether online L_cross estimate beats a static rate |
| Baseline | Harness-only / policy-only (single-level) | Tests whether cross-level coupling is the real bottleneck |
| Baseline | Bilevel hypergradient method (TTSA-style) | Tests whether CLFPO certifies contraction where bilevel fails |
| Ablation-of-ours | CLFPO without probe (L_cross frozen at init) | Isolates contribution of online coupling estimation |
| Ablation-of-ours | CLFPO with probe budget K=1 and K=5 | Tests cost-accuracy tradeoff of probe |
| Ablation-of-ours | CLFPO with residual-only certificate (no two-sided test) | Tests whether stationarity check prevents false certificates |

### Why this setup validates the claim

The central claim is that alternation between harness/plan and policy/generate converges only when the cross-level coupling is estimated and controlled, so the design must isolate that coupling as the causal variable. The undamped baseline tests whether contraction control matters at all: if joint residual is already small without damping, L_cross is not load-bearing. The fixed-rate baseline tests whether a static damping rate suffices, which would undercut the need for an online probe. The single-level baselines test whether the two-level structure itself, rather than coupling control, drives gains. The bilevel hypergradient baseline (TTSA-style) tests whether CLFPO certifies contraction on instances where bilevel methods fail to converge, directly addressing the novelty comparison. The no-probe ablation is the sharpest test: it preserves the damping machinery but removes online estimation, so any gain attributable to CLFPO must vanish when L_cross is frozen. The K=1 and K=5 probe-budget ablations test whether the cost-accuracy tradeoff is manageable. The residual-only certificate ablation tests whether the two-sided stationarity check is load-bearing for avoiding false certificates. Joint fixed-point residual is the right primary metric because it is exactly the quantity the certificate claims to bound; a method that improves script quality but not residual would fail to validate the mechanism, and a method that reduces residual but not script quality would reveal the certificate is vacuous. The relaxation gap gap_t is reported alongside residual so that a small residual on the relaxed map is not mistaken for a small residual on the discrete pipeline. Downstream coherence and coverage scores are reported for all baselines, and a scatter of residual vs. quality is plotted to establish that the certificate is non-vacuous and practically meaningful.

### Expected outcome and causal chain

**vs. Undamped alternating optimization** — On a case where the harness update sharply reshapes the task distribution (e.g., harness search set derived from previous round's hard tasks), the undamped baseline oscillates: the policy chases a moving harness target and residual stays flat or grows across rounds because the joint map's Lipschitz constant exceeds one. Our method instead damps both levels by eta_t = rho/L_cross, keeping the joint contraction below rho, so residual decays monotonically. We expect a clear residual gap concentrated in rounds immediately after harness-set shifts, with near-parity on rounds where L_cross is small. We also expect the undamped baseline to show larger relaxation gap gap_t at its stall rounds, because it does not control the surrogate-vs-discrete distance.

**vs. Fixed learning-rate alternation (Self-HPO-style)** — On a case where cross-level coupling spikes transiently (a plan update that dramatically changes the downstream generation objective), a fixed rate either overshoots into divergence or is so small that convergence stalls for many rounds. The baseline produces either residual spikes or a long plateau because its rate is blind to the instantaneous coupling. Our method instead re-estimates L_cross each round via the probe and clamps eta_t, so it tracks the transient. We expect fewer rounds-to-certificate and a smaller residual at matched round count, with the gap widening on instances exhibiting curvature transients.

**vs. Harness-only / policy-only (single-level)** — On a case requiring joint adaptation (a script whose narrative coherence demands both a new planning decomposition and a new generation policy), the single-level baseline optimizes one level against a frozen partner and plateaus at a residual floor set by the unoptimized level's mismatch. Our method instead moves both levels symmetrically toward a joint fixed point, so it can reach residuals below that floor. We expect the single-level baselines to plateau early and our method to continue decaying past their floor, with the effect most visible on instances where both levels are initially misaligned.

**vs. Bilevel hypergradient method (TTSA-style)** — On instances where the cross-level coupling is strong (L_cross > 1), the bilevel hypergradient baseline requires differentiating through the inner level, which is ill-conditioned when the inner problem is nonconvex and the outer update is discrete; it is expected to either diverge or converge to a per-level optimum without a joint certificate. Our method instead certifies contraction on the relaxed map and reports the relaxation gap, so it can certify a joint fixed point where the bilevel baseline cannot. We expect the bilevel baseline to fail to produce a certificate on these instances, while CLFPO certifies with a measurable gap_t.

**vs. CLFPO without probe (L_cross frozen at init)** — If online coupling estimation is load-bearing, freezing L_cross at its initial value should cause the damping schedule to be either too permissive (when true L_cross grows) or too conservative (when true L_cross shrinks), leading to residual spikes or slow convergence. We expect the no-probe ablation to match full CLFPO only on instances where L_cross is stationary, and to diverge on instances with curvature transients, with the gap concentrated where L_cross varies most across rounds.

**vs. CLFPO with probe budget K=1 and K=5** — We expect K=1 to produce noisier L_cross estimates and slightly worse residual, while K=5 yields marginal improvement over K=3 at higher probe cost. The cost-accuracy tradeoff is reported as wall-clock and probe-evaluation overhead per round, with the expectation that K=3 is near-optimal.

**vs. CLFPO with residual-only certificate** — We expect the residual-only certificate to occasionally declare a fixed point at a saddle or a discrete jump, which the two-sided stationarity check rejects. The ablation should show false certificates (certificates that do not correspond to a joint stationary point) on a small fraction of instances, demonstrating that the two-sided test is load-bearing for certificate validity.

### What would falsify this idea

If the no-probe ablation (frozen L_cross) matches full CLFPO on joint residual across all subsets, then online coupling estimation is not load-bearing and the central claim is wrong. Likewise, if CLFPO's residual gain is uniform across instances rather than concentrated where L_cross is large and transient, the mechanism is not the cause of the improvement. Additionally, if the relaxation gap gap_t is large at certified rounds (e.g., > 0.1 in L1 norm), the certificate on the relaxed map does not transfer to the discrete pipeline and the contribution reduces to an empirical damping heuristic.

## References

1. Self-Evolving Coding Agents via Harness-Policy Optimization
2. Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution
3. Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models
4. Large Language Models Are Human-Level Prompt Engineers
5. Eliminating Meta Optimization Through Self-Referential Meta Learning
6. Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models
7. STaR: Bootstrapping Reasoning With Reasoning
8. Automated Movie Generation via Multi-Agent CoT Planning
