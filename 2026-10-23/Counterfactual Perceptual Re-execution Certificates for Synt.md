# Counterfactual Perceptual Re-execution Certificates for Synthesized Procedural Skills

## Motivation

Across both instruction-following constraint synthesis (AgentIF, CFBench, FollowBench) and exploration-based skill discovery (EXIF), a synthesized skill's correctness is attributed to a monolithic generator, so a skill that 'works' may merely encode the explorer's idiosyncratic observation-to-action mapping rather than a procedure that is genuinely executable. EXIF in particular trains Bob on Alice's retrospectively generated skills without any test that the procedural content survives a change of the perceiving agent, which means executability and correctness cannot be jointly certified and a downstream failure cannot be attributed to either perception or procedure. The structural gap is that perception and procedure are operationally coupled in every single trajectory, so no statistical estimator computed on the generator's own data can separate them.

## Key Insight

Holding the procedural content fixed while intervening on the observation-to-action mapping at certification time turns perceptual competence into a varied nuisance factor, so invariance of the re-executed action sequence up to a declared equivalence class is a structural identifiability condition that no single-trajectory statistical estimator could supply.

## Method

**What it is.** CFX-Cert (CounterFactual eXecution Certificate) takes as input a synthesized procedural skill `s = (π_proc, C)` — a procedural policy skeleton plus a declared equivalence class `C` over action sequences — and a candidate set of independently perturbed perceptual renderers `R = {r_0, r_1, ..., r_k}` (r_0 is the generator's own renderer). It outputs a certificate `cert(s) ∈ {PASS, FAIL, UNDECIDED}` with an attribution label per failure: PERCEPTION-BOUND or PROCEDURE-BUGGY.

**Load-bearing assumption (stated explicitly).** The certification is valid only if swapping the renderer `r` while holding `π_proc` fixed is a *state-matched* intervention: at each step the policy is evaluated on the *same* environment state under `r_0` and `r_i`, so the only quantity that varies is the observation-to-action mapping. Without state matching, a closed-loop agent's actions change the state trajectory, and divergence would conflate perception dependence with dynamics shift. CFX-Cert therefore enforces state matching by snapshot/restore (or branching from a recorded state) at every step, and it decorrelates the five perturbation families so that each touches a distinct observation channel. The assumption fails if state matching is not enforced (e.g., latency injection that advances dynamics) or if the perturbation families are correlated (e.g., all five touch the same sensor); in those cases the attribution label reflects a dynamics or correlation artifact rather than perception dependence.

**How it works.**

```pseudocode
INPUT: skill s = (π_proc, C), renderer family R = {r_0..r_k}, env E, budget B
# Phase 1: reference execution under generator's renderer, with state snapshots
tau_0 = execute(π_proc, r_0, E, B)          # action sequence, length <= B
snapshots_0 = [snapshot(E, t) for t in 0..len(tau_0)]   # env state at each step
a_0   = project(tau_0, C)                   # canonical form under equivalence class C
# Phase 2: state-matched counterfactual re-execution under perturbed renderers
for r_i in R[1:]:
    tau_i = []
    for t in 0..len(tau_0)-1:
        restore(E, snapshots_0[t])          # do(state) intervention: same state
        obs_i = r_i(E)                      # only the observation mapping changes
        act_i = π_proc(obs_i)
        tau_i.append(act_i)
    a_i = project(tau_i, C)
    if |tau_i| < B and not terminal_ok(tau_i): return FAIL(label=PROCEDURE-BUGGY, i)
# Phase 3: invariance test
if all(a_i == a_0 for i in 1..k):
    return PASS
else:
    # Phase 4: attribution by renderer-subset analysis
    divergent = { i : a_i != a_0 }
    if len(divergent) >= ceil(k/2): return FAIL(label=PERCEPTION-BOUND, divergent)
    else:                           return FAIL(label=PROCEDURE-BUGGY, divergent)
```

Renderer perturbations are drawn from a fixed, declared family of five, each targeting a distinct observation channel to satisfy the independence assumption: (i) observation-channel dropout (mask one sensor/field), (ii) coordinate-frame rotation of spatial observations, (iii) synonym-substitution on textual observations with an entity-preserving lexicon, (iv) latency injection that shifts observation timestamps by δ — implemented as a *state-matched* timestamp shift (the environment state is restored before the shifted observation is rendered, so dynamics do not advance), and (v) a held-out observer prompt that re-describes the same environment state. `k = 5`, `B = 2×` the reference trajectory length. The equivalence class `C` is declared per skill as one of {exact action sequence, action multiset, goal-state predicate}, chosen by the skill author and stored with the skill.

**Why this design.** We chose *structural intervention on the renderer at certification time* over a learned disentanglement head (the obvious alternative) because perception and procedure are operationally coupled in any single trajectory, so a head trained on the generator's data would have no signal to separate them; the cost we accept is that certification requires `k` extra rollouts per skill, which we amortize by certifying in batch and by reusing rollouts across skills that share a renderer. We chose a *declared equivalence class* per skill over a fixed exact-match criterion because exact action-sequence invariance is too strict for skills whose procedure legitimately admits multiple orderings (e.g., parallel tool calls), accepting the cost that a maliciously loose `C` could pass a bad skill — which is why `C` is stored with the skill and auditable. We chose a *subset-majority attribution rule* over a per-renderer error message because a single divergent renderer is more likely a perturbation artifact than a perception dependence, accepting the cost that a skill depending on exactly one of five renderers is mislabeled PROCEDURE-BUGGY. We chose *five fixed perturbation families* over an adversarially optimized renderer because an optimizer would find the generator's specific blind spot and fail every skill, destroying discrimination; the cost is that a skill depending on a perturbation family we did not include will pass. This is not a variant of EXIF's iterative-feedback loop: EXIF refines *which skills* are proposed based on Bob's performance, whereas CFX-Cert holds the skill's procedural content fixed and intervenes on the *perceptual mapping* to produce an identifiability certificate — the quantity being varied (the renderer) is not a quantity EXIF varies at all.

**Why it measures what we claim.** `project(tau, C)` measures *procedural equivalence* because the declared equivalence class `C` is assumed to be a sufficient statistic for the skill's intended effect on the environment; this assumption fails when `C` is declared too loosely (e.g., goal-state predicate that any random walk satisfies), in which case the projection reflects goal-reaching rather than procedure. The state-matched invariance test `all(a_i == a_0)` measures *attributability of correctness to procedure* because the only quantity varied between `tau_0` and `tau_i` is the observation-to-action mapping `r` at each restored state, so any divergence must flow through that mapping; this assumption fails when state matching is not enforced (e.g., latency injection that advances dynamics) or when a renderer perturbation also changes the *environment dynamics*, in which case divergence reflects dynamics shift rather than perception dependence. The majority rule `len(divergent) >= ceil(k/2)` measures *perception-bound failure* because a genuinely perception-dependent skill will diverge under most independent renderers while a procedure bug will diverge under few; this assumption fails when the renderer family is correlated (e.g., all five perturbations touch the same sensor), in which case the majority reflects a single perturbation's artifact.

## Contribution

(1) A certification procedure, CFX-Cert, that produces a PASS/FAIL/UNDECIDED verdict plus a PERCEPTION-BOUND vs PROCEDURE-BUGGY attribution label for any synthesized procedural skill by intervening on the observation-to-action mapping while holding procedural content fixed. (2) A design principle — declared equivalence classes plus a fixed, non-adversarial perturbation family — under which perceptual invariance becomes an auditable property of a skill rather than a property of the generator that produced it, with a concrete failure-mode analysis of when each component's assumption breaks. (3) An empirical finding on skills synthesized by an EXIF-style explorer showing what fraction of 'working' skills are perception-bound, i.e., pass under the generator's renderer but fail under a majority of perturbed renderers.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Synthesized agent skills from ALFWorld-style procedural tasks | Skills have declared action-orderings and goal predicates |
| Primary metric | Attribution accuracy of FAIL labels (PERCEPTION-BOUND vs PROCEDURE-BUGGY) | Directly tests the identifiability claim |
| Baseline | EXIF iterative-feedback skill set | Tests claim that renderer intervention beats feedback refinement |
| Baseline | Uncertified skill pool (no renderer perturbation) | Tests claim that certification adds discrimination over raw synthesis |
| Baseline | Disentanglement-head classifier | Tests claim that structural intervention beats learned separation |
| Ablation-of-ours | CFX-Cert with single-renderer divergence rule | Tests whether the majority rule is doing real work |
| Ablation-of-ours | CFX-Cert with correlated perturbation family (all five touch same sensor) | Tests independence assumption of majority rule |
| Ablation-of-ours | CFX-Cert with state matching disabled (full trajectory replay) | Tests whether state matching is necessary for attribution |
| Downstream metric | Transfer-agent task success rate on CFX-Cert PASS-filtered pool vs uncertified pool | Ties certificate to outcome, not just attribution |

### Why this setup validates the claim

The central claim is that swapping only the renderer while holding the procedural policy fixed yields a certificate that correctly attributes failures to perception versus procedure. That claim is falsifiable only if we have ground-truth labels for which skills are genuinely perception-dependent and which contain procedure bugs, so we plant both kinds of skill deliberately during synthesis and score attribution accuracy against those planted labels. The planted-label generation procedure is pre-registered: for each skill we record whether its policy reads a single spatial field (perception-dependent) or contains a latent action-order bug (procedure-buggy), and we hold out these labels from the disentanglement-head baseline's training data to avoid label leakage. The EXIF baseline tests the sub-claim that iterative feedback on skill *proposal* cannot substitute for renderer intervention, because EXIF never varies the observation-to-action mapping and therefore has no signal separating the two failure sources. The uncertified pool tests whether certification adds discrimination at all beyond raw synthesis pass rates. The disentanglement-head baseline tests the design choice we explicitly rejected: if a head trained on generator data matched our attribution accuracy, our structural-intervention argument would be undermined. The single-renderer ablation isolates the subset-majority attribution rule so we can see whether the rule, rather than the perturbation family itself, produces the attribution. The correlated-perturbation ablation directly stress-tests the independence assumption: if attribution accuracy collapses when all five perturbations touch the same sensor, the majority rule is reading a correlation artifact, as predicted. The state-matching-disabled ablation tests the load-bearing assumption that state matching is necessary: if full-trajectory replay matches state-matched attribution, the dynamics confound is not operative in this environment. The downstream transfer-agent metric ties the certificate to an outcome: a PASS-filtered pool should yield higher task success for a held-out observer prompt than the uncertified pool. Attribution accuracy is the right primary metric because a method that flags failures but mislabels their cause is useless for the auditing purpose CFX-Cert exists to serve; raw PASS/FAIL rates would not detect the predicted effect.

### Expected outcome and causal chain

**vs. EXIF iterative-feedback skill set** — On a case where a skill's procedure is correct but its policy reads a single spatial field that a coordinate-frame rotation perturbs, EXIF's feedback loop sees Bob fail and refines *which skills are proposed*, never touching the renderer, so it re-proposes variants of the same perception-dependent skill and its attribution is essentially uninformative. Our method instead re-executes the identical `π_proc` under the rotated renderer at each restored state, observes divergence, and because most of the five perturbations touch perception, labels PERCEPTION-BOUND. We expect a large attribution-accuracy gap concentrated on the perception-dependent subset and near-parity on the procedure-bug subset.

**vs. Uncertified skill pool** — On a case where a skill contains a latent procedure bug that happens to succeed under the generator's renderer by luck, the uncertified pool passes it through with no signal, because it never executes under any renderer but `r_0`. Our method re-executes under perturbed renderers with state matching; the buggy skill diverges under few renderers and is labeled PROCEDURE-BUGGY with the divergent indices attached. We expect the uncertified pool to show inflated apparent pass rates and near-zero attribution accuracy, while our method's FAIL set is enriched for genuinely buggy skills.

**vs. Disentanglement-head classifier** — On a case where perception and procedure are coupled in every generator trajectory (the head's only training signal), the disentanglement head cannot separate them and defaults to whatever marginal label dominates, so it mislabels perception-dependent skills as procedure bugs at roughly the base rate. Our method sidesteps this because the state-matched renderer swap *is* the intervention that creates the separating signal, not a latent factor to be inferred. We expect our attribution accuracy to exceed the head's by a clear margin specifically on the coupled subset, with parity on skills where coupling is absent.

**vs. CFX-Cert with single-renderer divergence rule (ablation)** — On a skill that diverges under exactly one perturbation due to a perturbation artifact, the single-renderer rule labels it PERCEPTION-BOUND, while the majority rule labels it PROCEDURE-BUGGY. We expect the majority rule to outperform the single-renderer rule on the planted-label set, with the gap concentrated on skills whose divergence is driven by one perturbation family.

**vs. CFX-Cert with correlated perturbation family (ablation)** — When all five perturbations touch the same sensor, the majority rule's independence assumption is violated. We expect attribution accuracy to drop toward the single-renderer baseline on the perception-dependent subset, confirming that the majority rule requires decorrelated perturbations to reflect independent evidence.

**vs. CFX-Cert with state matching disabled (ablation)** — When full-trajectory replay is used instead of state-matched replay, the dynamics confound is active. We expect attribution accuracy to degrade on skills where the policy's actions change the state trajectory, with the degradation concentrated on the perception-dependent subset, confirming that state matching is necessary for the claimed attribution.

**vs. Downstream transfer-agent task success** — A held-out agent with a different observer prompt is evaluated on the CFX-Cert PASS-filtered pool versus the uncertified pool. We expect the PASS-filtered pool to yield a higher task success rate, with the gain concentrated on skills that were certified PASS despite being perception-dependent under the generator's renderer.

### What would falsify this idea

If our attribution gain over the disentanglement head is uniform across the coupled and uncoupled subsets rather than concentrated on the coupled subset, then the renderer intervention is not creating the separating signal we claim and the central identifiability argument is wrong. Likewise, if the single-renderer ablation matches the majority rule's attribution accuracy, the subset-majority mechanism is not doing the work we attribute to it; if the correlated-perturbation ablation does not degrade attribution, the independence assumption is not load-bearing; and if the state-matching-disabled ablation matches state-matched attribution, the dynamics confound is not operative and state matching is unnecessary.

## References

1. Automated Skill Discovery for Language Agents through Exploration and Iterative Feedback
2. AGENTIF: Benchmarking Instruction Following of Large Language Models in Agentic Scenarios
3. SysBench: Can Large Language Models Follow System Messages?
4. CFBench: A Comprehensive Constraints-Following Benchmark for LLMs
5. Instruction-Following Evaluation for Large Language Models
6. FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models
7. Can LLMs Follow Simple Rules?
8. Can Large Language Models Understand Real-World Complex Instructions?
