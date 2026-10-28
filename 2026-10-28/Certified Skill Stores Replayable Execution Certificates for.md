# Certified Skill Stores: Replayable Execution Certificates for Monotone Multimodal Agent Self-Improvement

## Motivation

Across the agent literature, skill and reward artifacts are authored or judged offline and never closed under execution feedback: VISUALSKILL builds hierarchical multimodal skills per application with a two-stage authoring pipeline but never verifies that a stored skill's recorded trace actually reproduces its claimed effect when re-executed; Agent S retrieves experience from memory but never re-validates retrieved experiences against the current environment; AgentStore integrates third-party agents but has no mechanism to retire a specialist whose competence has silently decayed. The structural root cause is that the field treats artifact construction and artifact consumption as independent phases, so no stored skill carries a binding between (a) the trace that justifies it and (b) the environment parameters under which that justification is valid. Without such a binding, online update cannot be a closure operation, and transfer to unseen environments is a hope rather than a checkable property.

## Key Insight

A skill store is a set of hypotheses closed under replayable execution certificates that are parametric in observable environment features, so admission, refinement, and retirement become a monotone closure operator whose fixed points are exactly the skills whose post-conditions hold on the observed environment-parameter manifold.

## Method

**What it is.** CERTSTORE takes as input a stream of agent interaction traces (each trace = sequence of multimodal observations, actions, and a final post-condition predicate evaluated by an environment checker) plus the observable environment-parameter vector φ(e) for each episode, and outputs a monotonically updated skill store in which every skill s carries a certificate C(s) = ⟨τ_s, ψ_s, Φ_s⟩: the recorded trace τ_s, the pre/post-condition pair ψ_s = (pre_s, post_s) expressed as functions of φ(e), and the set Φ_s of environment-parameter vectors on which the certificate has been replayed successfully. The store is queried at inference by the agent's existing load_topic tool, which now returns only skills whose certificate covers the current φ(e).

**Load-bearing assumption (stated explicitly).** CERTSTORE assumes φ(e) is a *sufficient statistic* for the environment's effect on a skill: that replay success on resampled φ drawn from a Gaussian neighborhood N(φ(e), σ²) of an observed episode is equivalent to post-condition success on genuinely unseen environments. This assumption fails when φ omits a latent environment factor (e.g., UI theme, locale, font scaling, window-manager state, timing/race conditions, hidden modal dialogs), in which case a Gaussian neighborhood in φ-space is not a neighborhood in the space of environment behaviors, and replay success reflects robustness to the observed parameters only. The method therefore does not claim that Φ_s membership implies transfer to arbitrary unseen environments; it claims that Φ_s membership implies post-condition holding on the observed φ-manifold, and the experiment measures the divergence between replay success and true post-condition holding under a deliberately injected latent factor.

**How it works.**

```pseudocode
CERTSTORE(stream of episodes, replay budget B, coverage threshold δ,
          latent-probe set L, probe count K_L)
  Store S ← ∅                              # each entry: (skill text/figures, C(s))
  for each episode e with parameters φ(e):
    τ ← agent.run(e)                       # produces trace + terminal post-condition
    if env.check(τ.post) is False: continue
    # 1. PROPOSE: abstract a candidate skill from the successful trace
    s_new ← ABSTRACT(τ, φ(e))              # pre/post as functions of φ
    # 2. CERTIFY: replay under K fresh seeds with resampled φ near φ(e)
    Φ_cand ← { φ' : φ' ~ N(φ(e), σ²) ∩ observed-manifold, |Φ_cand| < B }
    certified ← TRUE
    for φ' in Φ_cand:
        e' ← env.reset(seed=fresh, params=φ')
        τ' ← REPLAY(s_new, e')             # re-execute recorded actions, allow re-grounding
        if env.check(τ'.post) is False: certified ← FALSE; break
    # 2b. LATENT-PROBE: replay under K_L perturbations of unobserved nuisance factors
    #     (theme, locale, font scaling, window geometry, timing jitter) drawn from L
    for ℓ in L:                            # |L| = K_L, fixed probe set
        e' ← env.reset(seed=fresh, params=φ(e), nuisance=ℓ)
        τ' ← REPLAY(s_new, e')
        if env.check(τ'.post) is False: certified ← FALSE; break
    # 3. CLOSURE RULE (monotone, idempotent, order-preserving)
    if certified:
        if ∃ s ∈ S with SAME_EFFECT(s, s_new):
            S ← S \ {s} ∪ {REFINE(s, s_new)}   # pre/post := intersection over all certified traces
            C(REFINE) ← C(s) ∪ C(s_new)        # Φ grows monotonically
        else:
            S ← S ∪ {(s_new, C(s_new))}
    else:
        # certificate failed: retire the closest existing skill if its coverage no longer holds
        for s ∈ S with OVERLAP(s, s_new):
            if not REVALIDATE(s, φ(e)): S ← S \ {s}
  return S

QUERY(S, φ(e)) = { s ∈ S : φ(e) ∈ Φ_s and pre_s(φ(e)) holds }

# Concrete predicate definitions (all computable):
# SAME_EFFECT(s, s') := post_s ≡ post_s' on a probe set P of 32 (φ, observation) pairs
#   drawn from the observed manifold; equivalence = identical boolean output on all 32.
# OVERLAP(s, s') := |Φ_s ∩ Φ_s'| / min(|Φ_s|, |Φ_s'|) ≥ 0.5.
# REVALIDATE(s, φ) := replay s under φ with 2 fresh seeds; returns TRUE iff both
#   replays satisfy post_s; two consecutive FALSE results trigger retirement.
# REPLAY re-grounding: perception module = OSWorld's accessibility-tree parser
#   (pyatspi) for element localization; action re-binding = match recorded action's
#   target element by (role, name) tuple in the fresh accessibility tree, falling
#   back to bounding-box IoU ≥ 0.5 against the recorded coordinates.
# Hyperparameters: B = 8, K = 3 fresh seeds per φ', σ = 0.1 (normalized φ units),
#   K_L = 4 latent probes (theme=dark, locale=de_DE, font_scale=1.25, timing_jitter=200ms),
#   δ = 0.75 (minimum |Φ_s|/|Φ_cand| for admission).
```

**Why this design.** We chose a *replay-based* certificate over a learned verifier because replay is a deterministic intervention on the environment that yields a ground-truth post-condition, whereas a learned verifier would reintroduce the offline-judgment problem that VISUALSKILL and Agent S suffer from; the cost is that replay requires a resettable environment, which excludes purely conversational settings. We chose *environment-parametric* pre/post-conditions (functions of φ) over instance-specific predicates because transfer to unseen environments is then a membership check φ(e) ∈ Φ_s rather than a hope, accepting the cost that φ must be observable at inference time. We chose a *monotone closure rule* (refine by intersection, never widen pre-conditions without new certificates) over ad-hoc fine-tuning because monotonicity gives idempotence and order-preservation, so the store's fixed point is well-defined and repeated updates cannot oscillate; the cost is that a skill whose certificate is later invalidated must be *retired*, not patched, which can temporarily shrink coverage. We chose *intersection refinement* over weighted averaging of pre/post-conditions because intersection is the unique operation that preserves the invariant 'every admitted skill has a certificate covering its pre-condition', accepting the cost that over many refinements pre-conditions can become too narrow to fire, which the retirement rule then handles. We added the *latent-probe* step (2b) because the Gaussian-in-φ resampling in step 2 samples only the observed manifold and therefore cannot, by construction, certify a region the agent has not already visited; the probe set L varies unobserved nuisance factors and requires the certificate to hold across them, so Φ_s is defined over a space that spans a known subset of the shift the evaluation tests. A domain expert would not describe this as a variant of ToolLLM's offline solution-path annotation or Agent S's experience retrieval, because neither carries a replayable certificate nor a monotone closure operator — both are one-shot artifact construction.

**Why it measures what we claim.** The certificate's replay success rate |Φ_s| / |Φ_cand| measures *transferability* because the assumption that φ(e) is a sufficient statistic for the environment's effect on the skill makes replay success on resampled φ equivalent to post-condition holding on unseen environments; this assumption fails when φ omits a latent environment factor (e.g., a UI theme not captured by φ), in which case the metric reflects robustness to the observed parameters only — which is why step 2b adds latent probes and why the experiment includes a diagnostic that injects a known latent factor to measure the divergence. The intersection operation REFINE measures *monotone refinement of admissibility* because the assumption that post-conditions are closed under intersection of certified traces makes the intersected pre-condition exactly the set of parameters on which all certified traces succeeded; this assumption fails when two traces succeed for different causal reasons (e.g., one via keyboard shortcut, one via menu), in which case the intersection reflects a conjunction of sufficient conditions rather than a necessary one. The retirement rule measures *soundness of the store* because the assumption that a failed revalidation under a new φ indicates the skill's certificate no longer covers the manifold makes retirement equivalent to restoring the invariant; this assumption fails when the failure is due to a transient environment glitch, in which case retirement reflects noise rather than decay — mitigated by requiring two consecutive failures before retirement.

## Contribution

(1) A certified skill store in which every stored skill carries a replayable execution certificate parametric in observable environment features, and store update is a monotone closure operator with admission, intersection-refinement, and retirement rules. (2) A design principle — environment-parametric certificates as the binding object between offline skill construction and online execution feedback — that converts transfer to unseen environments from an untestable hope into a membership check φ(e) ∈ Φ_s. (3) An open-source replay harness that wraps OSWorld-style resettable environments to produce fresh-seed traces for certification, with a protocol for measuring certificate coverage versus task success on held-out environment parameters.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | OSWorld (369 tasks, resettable envs) | Provides resettable environment with observable parameters |
| Dataset | WebArena (812 tasks, resettable envs) | Second heterogeneous resettable substrate for scope check |
| Primary metric | Success rate on held-out φ regions | Directly tests transfer beyond certified coverage |
| Secondary metric | Replay-success vs. true-post-condition divergence under injected latent factor | Quantifies φ-sufficiency assumption |
| Baseline | No-skill agent | Lower bound isolating skill-store contribution |
| Baseline | VISUALSKILL (matched text-only skill) | Tests value of replay certificates vs one-shot skills |
| Baseline | Agent S2 experience retrieval | Tests monotone closure vs ad-hoc retrieval |
| Ablation-of-ours | CERTSTORE w/o certification (propose-only) | Isolates the replay-certificate mechanism |
| Ablation-of-ours | CERTSTORE w/o latent-probe step 2b | Isolates the latent-probe contribution |
| Ablation-of-ours | CERTSTORE w/ φ omitting one known factor (theme) | Measures divergence of replay success from true post-condition |

### Why this setup validates the claim

The central claim is that replay-certified, environment-parametric skills transfer to unseen environments where one-shot skill artifacts fail. OSWorld is the right substrate because it is resettable and exposes environment-parameter vectors, which is exactly the precondition CERTSTORE assumes; a non-resettable benchmark would test nothing about the certificate mechanism. WebArena is added as a second heterogeneous resettable substrate to check that the mechanism is not specific to OSWorld's parameterization. The primary metric — success rate on held-out φ regions deliberately outside the certified Φ_s — is the only metric that can detect the predicted effect, since in-distribution success would be achieved by any skill store. The no-skill baseline establishes the floor, VISUALSKILL tests whether replay certificates beat one-shot text skills at transfer, and Agent S2 tests whether monotone closure beats ad-hoc retrieval when the environment shifts. The propose-only ablation is decisive: if removing certification does not hurt held-out transfer, then the certificate is doing no causal work and the mechanism is decorative. The latent-probe ablation and the φ-omission ablation are the direct tests of the load-bearing sufficiency assumption: if replay success under the omitted factor diverges from true post-condition holding, the assumption is quantified rather than asserted. Together these form a falsifiable test because the predicted gain is localized to the held-out-φ subset, not uniform.

### Expected outcome and causal chain

**vs. No-skill agent** — On a case where the task requires a multi-step UI procedure under an unseen window layout, the no-skill agent produces exploratory flailing because it has no recorded procedure to ground actions. Our method instead retrieves a certified skill whose Φ_s covers the current φ(e) and replays its recorded actions with re-grounding, because the certificate guarantees the post-condition held across resampled parameters and latent probes. We expect a large gap on multi-step procedural tasks and near-parity on single-action tasks, since the latter need no skill.

**vs. VISUALSKILL (text-only skill)** — On a case where the environment shifts (e.g., a dialog position or default setting differs from skill-construction time), the text-only skill produces a plausible but mis-grounded action sequence because its artifact was built once and never replayed under variation. Our method instead only admits the skill if replay succeeded across the resampled φ neighborhood and the latent probes, so the pre-condition narrows to parameters where it actually fires. We expect VISUALSKILL to match us in-distribution but degrade sharply on the held-out-φ subset, producing a visible gap concentrated there.

**vs. Agent S2 experience retrieval** — On a case where a retrieved experience partially overlaps the current environment, Agent S2 produces a blended or stale action plan because retrieval has no admissibility check against φ(e). Our method instead returns only skills whose certificate covers φ(e) and whose pre_s holds, and refines by intersection rather than blending. We expect Agent S2 to show higher variance and occasional confident failures on shifted environments, while our method shows lower variance and a cleaner in/out-of-coverage success separation.

**vs. CERTSTORE w/o latent-probe step 2b** — On a case where the environment shifts along an unobserved nuisance factor (theme, locale, font scaling), the no-probe variant admits skills whose certificates were built only on the observed φ-manifold, so it fails at replay time when the nuisance factor differs. Our method instead requires the certificate to hold across the K_L = 4 latent probes, so the admitted skill's Φ_s spans the probed nuisance directions. We expect the no-probe variant to match ours on the observed manifold but to drop on the held-out-φ subset that varies the probed factors, isolating the latent-probe contribution.

**vs. CERTSTORE w/ φ omitting one known factor (theme)** — On a case where theme is omitted from φ, the certificate's replay success on resampled φ overestimates true post-condition holding because theme is a latent confounder. Our method's diagnostic measures the divergence directly: we report the gap between |Φ_s|/|Φ_cand| and the true post-condition success rate on a held-out set that varies theme, quantifying the sufficiency assumption's failure mode rather than asserting it.

### What would falsify this idea

If the propose-only ablation (no certification) matches CERTSTORE's held-out-φ success rate, then replay certification is doing no causal work and the central claim is wrong. Likewise, if our gain over VISUALSKILL is uniform across in-distribution and held-out-φ subsets rather than concentrated on the held-out subset, the predicted transfer mechanism is not what is producing the improvement. Finally, if the φ-omission diagnostic shows replay success tracks true post-condition holding even when a known latent factor is omitted, the sufficiency assumption is empirically supported; if it diverges, the transfer claim is bounded to the observed manifold.

## References

1. VISUALSKILL: Multimodal Skills for Computer-Use Agents
2. Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents
3. OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments
4. Agent S: An Open Agentic Framework that Uses Computers Like a Human
5. AgentStore: Scalable Integration of Heterogeneous Agents As Specialized Generalist Computer Assistant
6. OS-ATLAS: A Foundation Action Model for Generalist GUI Agents
7. OSCAR: Operating System Control via State-Aware Reasoning and Re-Planning
8. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
