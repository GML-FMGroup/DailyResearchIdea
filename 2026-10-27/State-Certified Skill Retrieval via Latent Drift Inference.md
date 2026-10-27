# State-Certified Skill Retrieval via Latent Drift Inference

## Motivation

Across skill-learning and tool-use lines of work, procedural knowledge is stored and retrieved as discrete items indexed by surface-context similarity: AutoGuide mines conditional guidelines and matches them to the agent's current context, ExpeL recalls past experiences by task resemblance, and standard tool-selection systems (Gorilla/RAG-style k-NN) pick the nearest descriptor. Δτ-Bench concretely exposes the structural failure of this paradigm: even the strongest harness reaches only 42.3% task success, and gains from existing skill-learning methods do not consistently transfer as the ecosystem evolves through capability-preserving transformations, because a skill mined under one ecosystem state silently breaks under another while still looking contextually similar. The root cause is that retrieval never certifies that a skill's preconditions still hold under the inferred ecosystem state — validity is undefined once tool semantics drift, so similarity-matching returns skills that are plausibly relevant but no longer executable.

## Key Insight

Because capability-preserving transformations change surface semantics while moving the ecosystem along a low-dimensional drift manifold, the latent drift state is identifiable from observable tool-behavior probes (I/O contracts, error signatures), which turns skill retrieval from a similarity-matching problem into a state-identification problem whose posterior can certify a skill's preconditions as a testable invariant rather than a confidence score.

## Method

**What it is.** DRIFT-CERT is a skill store whose entries are indexed by a posterior over latent ecosystem-state (drift) coordinates rather than by surface context. Given a task and a live tool ecosystem, it runs behavior probes, infers a posterior over the drift state, and returns a skill only when a state-conditioned validity certificate — derived from a generative model of tool-semantics drift — certifies that the skill's preconditions still hold; otherwise it abstains and triggers re-mining.

**Load-bearing assumption (stated explicitly).** The certificate is computed from probe-observable validity. It equals true execution validity only under the assumption that the probe-outcome likelihood p(o|z) is a sufficient statistic for ecosystem state — i.e., two ecosystems with identical probe signatures (I/O contracts, error signatures) have identical precondition-satisfaction profiles for every stored skill. This assumption fails when a transformation changes a tool's precondition without altering any probed I/O contract or error signature (silent drift: side-effect ordering, statefulness, rate limits, idempotency, hidden argument semantics). To make the certificate honest about what it certifies, we replace the point certificate with a lower confidence bound that marginalizes over a calibrated unprobed semantic residual u, and we add an explicit per-skill 'unobservable precondition' flag that forces abstention when a skill's preconditions are not fully covered by the probe set.

**How it works.**

```pseudocode
# Offline: build store indexed by drift coordinates, not context
for each skill s mined in ecosystem state tau_s:
    precond[s]  = precondition set P_s (tool names, I/O contract predicates, error codes)
    anchor[s]   = probe-outcome signature recorded when s was validated  # vector of probe results
    covered[s]  = 1 if every predicate in P_s is in the probe-observable set O, else 0
    store.append(entry=(P_s, anchor[s], covered[s], s))

# Generative model of drift: probe outcome o_i = f_i(z) + noise
# z in R^d is the latent drift coordinate; f_i learned per probe from
# labeled (state, probe-outcome) pairs collected across the 6 Delta-Tau states.
fit  p(o | z) = N( f(z), Sigma )   # f: R^d -> R^m, d=4, m=#probes=12
# Calibrate unprobed semantic residual u ~ N(0, sigma_u^2 I) using
# leave-one-state-out residuals of precondition satisfaction not explained by z.
u(z) ~ N(0, I)                    # drift-coordinate prior

# Online: state identification, then certificate-gated retrieval
def retrieve(task, ecosystem):
    obs = run_probes(ecosystem, probes=P)        # P = 12 contract/error probes
    post = Laplace_approx_posterior(obs, p(o|z), u(z))   # mean z_hat, cov Sigma_z
    candidates = [s for s in store if task_compatible(s, task)]
    certified = []
    for s in candidates:
        if covered[s] == 0:
            continue  # unobservable precondition -> forced abstention
        # certificate: lower confidence bound on P(preconditions hold | z, u)
        # P_s satisfied = g(z, u); u ~ N(0, sigma_u^2 I) calibrated offline
        cert = E_{z~post}[ min_{u in support} 1{ g(z,u) satisfied } ]
        # operationalized as the 5th percentile of the posterior predictive
        # over (z, u) samples: draw 1000 z ~ post, 1000 u ~ N(0, sigma_u^2 I),
        # compute fraction satisfying all preconditions, take 5th percentile.
        if cert >= gamma:                        # gamma = 0.9
            certified.append((s, cert))
    if certified: return argmax_cert(certified)
    else: return ABSTAIN -> trigger_skill_remine(task, z_hat)
```

The certificate `cert` is computed by pushing each posterior sample of `z` and each draw of the unprobed residual `u` through the generative model to predict whether each precondition predicate (tool presence, argument-contract compatibility, error-signature class) is satisfied; the 5th percentile of the fraction of (z,u) samples satisfying all preconditions is the certificate value. Abstention is a first-class output: when no skill clears `gamma`, or when `covered[s]=0`, the agent re-mines rather than executing a stale skill.

**Why this design.** We chose to index the store by latent drift coordinates rather than surface context because surface context is exactly what capability-preserving transformations perturb, so context-indexed retrieval (as in AutoGuide's guideline matching and ExpeL's experience recall) cannot distinguish a skill that is still valid from one whose tool contract has silently changed, accepting the cost that we must fit a generative probe model offline. We chose a Laplace-approximated posterior over a 4-dimensional drift coordinate rather than a learned neural classifier because the drift manifold is low-dimensional by construction (transformations are capability-preserving), so a small, inspectable posterior is sufficient and avoids the sample-inefficiency of a deep retrieval head, accepting the cost that the Gaussian-probe likelihood is an approximation. We chose a hard certificate threshold (`gamma=0.9`) with explicit abstention rather than a soft relevance score because the downstream action is executing a procedure, so a false-positive retrieval is far costlier than an abstention, accepting the cost of lower recall and more re-mining. We chose behavior probes over documentation parsing because Δτ-Bench's noisy documentation is precisely the surface signal that drifts; probes observe input/output contracts and error signatures that are invariant under capability-preserving transformation. A domain expert would not describe this as a variant of Gorilla/RAG-style nearest-neighbor tool selection: those methods return the closest descriptor and have no notion of a state posterior or a validity certificate, whereas here retrieval is gated by a posterior probability that a precondition set still holds.

**Why it measures what we claim.** The lower-confidence-bound certificate `E_{z~post}[ min_{u} 1{g(z,u) satisfied} ]` measures the motivation-level concept of *skill-transfer validity* because the named assumption is that the probe-outcome likelihood `p(o|z)` is a sufficient statistic for ecosystem state — i.e., two ecosystems with the same probe signature have the same precondition-satisfaction profile; this assumption fails when a transformation changes a tool's precondition without altering any probed I/O contract or error signature, in which case the certificate reflects probe-observable validity rather than true validity and can over-certify. The probe set `P` measures *drift-coordinate identifiability* because the assumption is that the probe Jacobian `df/dz` has full column rank at the operating state, so distinct drift coordinates produce distinct probe signatures; this fails near rank-deficient regions (e.g., two transformations that cancel on all probes), in which case the posterior is broad and the certificate correctly drops toward abstention. The abstention threshold `gamma` measures *execution risk* because the assumption is that the certificate is monotone in the probability of successful execution under the true state; this fails when precondition satisfaction is necessary but not sufficient for skill success (e.g., the skill's control flow is also stale), in which case `gamma` controls contract-validity rather than end-to-end success. The per-skill `covered[s]` flag measures *probe coverage* because the assumption is that every precondition predicate in `P_s` is in the probe-observable set `O`; this fails when a skill depends on unprobed semantics (side-effect ordering, statefulness), in which case the flag forces abstention and the certificate is not computed.

## Contribution

(1) A skill-retrieval framework, DRIFT-CERT, that indexes procedural knowledge by a posterior over latent drift coordinates and gates retrieval with a state-conditioned validity certificate derived from a generative probe model, converting retrieval into state identification. (2) A design principle — certificate-gated retrieval with first-class abstention and re-mining — that makes skill-transfer validity a testable invariant rather than a similarity score, and an analysis of when probe signatures are sufficient statistics for drift state. (3) A probe-design methodology (input/output contract and error-signature probes) and an offline generative drift model that together make ecosystem state identifiable across capability-preserving transformations.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Δτ-Bench six-state ecosystem | Only benchmark with capability-preserving drift across states |
| Primary metric | Task success on drifted states | Directly measures transfer validity under drift |
| Secondary metric | Certified-execution rate vs. task success rate gap | Quantifies how often the contract-validity-to-execution-success assumption (A) breaks |
| Secondary metric | Abstention rate vs. probe-Jacobian rank deficiency | Tests whether the posterior tracks drift-coordinate identifiability |
| Baseline | Strongest model+harness (no skills) | Tests whether skill store adds anything |
| Baseline | AutoGuide guideline matching | Tests context-indexed retrieval under drift |
| Baseline | ExpeL experience recall | Tests experience-indexed retrieval under drift |
| Baseline | Learned neural state classifier (same 12 probe features, 2-layer MLP hidden=256, GeLU) | Tests whether the certificate's calibration advantage is reproducible by a simpler learned alternative |
| Baseline | Conformal-prediction gating (same probe features, split-conformal on precondition satisfaction) | Tests whether the certificate's calibration advantage is reproducible by a distribution-free alternative |
| Ablation-of-ours | DRIFT-CERT minus certificate (retrieve top posterior-match) | Isolates the certificate gate's contribution |
| Ablation-of-ours | DRIFT-CERT minus unprobed-residual marginalization (point certificate E_z[1{P_s satisfied}]) | Isolates the lower-confidence-bound repair's contribution |
| Ablation-of-ours | DRIFT-CERT minus covered[s] flag (force certificate even when preconditions unprobed) | Isolates the probe-coverage abstention's contribution |

### Why this setup validates the claim

The central claim is that gating retrieval on a state-conditioned validity certificate, rather than on surface-context similarity, is what preserves skill-transfer validity as the ecosystem drifts. Δτ-Bench is the right dataset because it holds task semantics fixed while perturbing tool contracts across six states, so any success difference is attributable to drift handling, not task difficulty. The strongest-model-without-skills baseline establishes the ceiling a skill store must beat; if DRIFT-CERT does not exceed it on drifted states, the store adds nothing. AutoGuide and ExpeL each test a distinct sub-claim: that context-indexed and experience-indexed retrieval respectively cannot distinguish a still-valid skill from one whose contract silently changed. The learned neural state classifier and conformal-prediction gating baselines test whether the certificate's calibration advantage is a repackaging of simpler alternatives on the same probe features. The no-certificate ablation is the load-bearing control — it uses the same probe posterior and the same store, differing only in whether the certificate gate is applied, so any gap between it and full DRIFT-CERT isolates the certificate as the causal ingredient rather than the probe model or the drift-coordinate indexing. The residual-marginalization and coverage-flag ablations isolate the two components of the load-bearing-assumption repair. Task success on drifted states is the right primary metric because the claim is specifically about execution validity after transformation; a relevance or retrieval-accuracy metric would not detect the false-positive executions that motivate abstention. The certified-execution-rate vs. task-success-rate gap is the right secondary metric because it directly quantifies how often the named assumption A (precondition satisfaction is near-sufficient for skill success) breaks, and the abstention-rate vs. rank-deficiency metric tests the identifiability claim.

### Expected outcome and causal chain

**vs. Strongest model+harness (no skills)** — On a case where a tool's argument contract has shifted between ecosystem states, the no-skills agent rediscovers the procedure from scratch each episode and succeeds only when in-context exploration happens to recover the new contract, so it fails on the harder multi-turn tasks that require composing several drifted tools. Our method instead retrieves a certified skill whose preconditions are checked against the inferred drift state, so it starts from a valid plan and spends turns executing rather than rediscovering. We expect a clear success gap concentrated on drifted states with composed tool use, with near-parity on early, undrifted states.

**vs. AutoGuide guideline matching** — On a case where a guideline was mined against an earlier tool contract and the tool's error signature has since changed class, AutoGuide matches the guideline to the task by surface context, returns it as still applicable, and the agent executes a stale call sequence that now errors out mid-trajectory. Our method instead observes the changed error signature through probes, infers a shifted drift coordinate, and the certificate for that skill drops below gamma, triggering abstention and re-mining. We expect AutoGuide to show confident-but-wrong retrievals that manifest as mid-trajectory failures, while DRIFT-CERT converts many of those into abstentions followed by successful re-mined executions, yielding a success gap that widens with the number of intervening drift states.

**vs. ExpeL experience recall** — On a case where an experience was stored under a tool whose I/O contract later drifted, ExpeL recalls it by task similarity and replays the stored action sequence, since its recall mechanism has no representation of ecosystem state and cannot represent that the tool changed. Our method instead conditions retrieval on the posterior over drift coordinates, so the same experience is only returned when its anchor probe signature is consistent with the current state. We expect ExpeL's failures to cluster on the specific transformations that alter probed contracts, while DRIFT-CERT shows parity on transformations that leave probed contracts intact — the diagnostic signature that the gain comes from state-conditioned gating rather than generic skill reuse.

**vs. Learned neural state classifier** — On a case where the probe signature is near a decision boundary between two drift states, the neural classifier returns a hard label with no calibrated uncertainty, so it confidently retrieves a skill whose preconditions hold in the wrong state. Our method instead returns a posterior over z and a lower confidence bound that accounts for the unprobed residual, so it abstains when the posterior mass is diffuse. We expect the classifier to match DRIFT-CERT on well-separated states but to show higher false-positive execution rates on boundary states, and we expect DRIFT-CERT's risk-coverage curve to dominate the classifier's.

**vs. Conformal-prediction gating** — On a case where the calibration set does not cover a silent-drift transformation (unprobed precondition change), the conformal predictor's coverage guarantee is marginal over the calibration distribution and does not certify the specific skill's preconditions under the current state, so it can return a skill whose contract is valid but whose unprobed semantics have drifted. Our method instead explicitly marginalizes over the calibrated unprobed residual and forces abstention when a skill's preconditions are not probe-covered. We expect conformal gating to match DRIFT-CERT on probe-covered skills but to over-certify on skills with unobservable preconditions, and we expect the coverage-flag ablation to reproduce that over-certification.

**vs. DRIFT-CERT minus certificate** — On a case where two drift states are distinguishable by probes but a candidate skill's preconditions only hold in one, the no-certificate variant retrieves the top posterior-match skill unconditionally and executes it in the wrong state, producing a false-positive execution. Our method computes the posterior mass satisfying all preconditions and abstains when it falls below gamma, converting that false positive into a re-mine. We expect the ablation to match DRIFT-CERT on undrifted states and on states where all candidates are valid, but to show a marked excess of confident wrong executions on the drifted subsets, isolating the certificate as the causal mechanism.

**vs. DRIFT-CERT minus unprobed-residual marginalization** — On a case where a skill's preconditions are probe-covered but a silent-drift transformation changes an unprobed semantic (e.g., side-effect ordering), the point certificate E_z[1{P_s satisfied}] over-certifies because it assumes the residual is zero, while the lower-confidence-bound certificate drops below gamma. We expect the point-certificate variant to show a higher certified-execution rate but a larger gap to task success rate on silent-drift subsets, isolating the residual marginalization as the mechanism that makes the certificate honest about what it certifies.

**vs. DRIFT-CERT minus covered[s] flag** — On a case where a skill depends on an unprobed precondition (statefulness), the no-flag variant computes a certificate from probe-observable predicates only and certifies the skill, while the full method forces abstention. We expect the no-flag variant to show excess false-positive executions on skills with unobservable preconditions, and the gap between certified-execution rate and task success rate to be concentrated there, isolating the coverage flag as the mechanism that prevents over-certification on silent drift.

### What would falsify this idea

If DRIFT-CERT's gain over the no-certificate ablation and over AutoGuide/ExpeL were uniform across drifted and undrifted subsets rather than concentrated on transformations that alter probed contracts, the certificate would not be doing the causal work claimed. Likewise, if abstention rates stayed flat as probe-Jacobian rank deficiency increased, the posterior would not be tracking drift-coordinate identifiability as asserted, and if the gap between certified-execution rate and task success rate did not shrink when the unprobed-residual marginalization and coverage flag were added, the load-bearing-assumption repair would not be doing the work claimed.

## References

1. Δτ-Bench: Pushing Agents Toward a Moving Capability
2. Towards General Agentic Intelligence via Environment Scaling
3. T-Eval: Evaluating the Tool Utilization Capability of Large Language Models Step by Step
4. AutoGuide: Automated Generation and Selection of Context-Aware Guidelines for Large Language Model Agents
5. ReAct: Synergizing Reasoning and Acting in Language Models
6. xLAM: A Family of Large Action Models to Empower AI Agent Systems
7. ExpeL: LLM Agents Are Experiential Learners
8. AgentTuning: Enabling Generalized Agent Abilities for LLMs
