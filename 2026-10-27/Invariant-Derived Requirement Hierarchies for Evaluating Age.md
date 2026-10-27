# Invariant-Derived Requirement Hierarchies for Evaluating Agentic Work under Partial Observability

## Motivation

Evaluation of agentic work currently depends on a human annotation oracle: Agent-as-a-Judge requires humans to pre-write hierarchical requirements, and the Work Agents framework explicitly offers no substitute for that oracle, so any task whose intermediate steps are unverifiable (a long-horizon coding or data-analysis run in a stateful workspace) simply cannot be assessed. The root cause is structural rather than a matter of annotation effort: existing benchmarks such as ScienceAgentBench and SWE-agent verify only the terminal program output, so when a run partially completes or its trajectory is opaque, there is no requirement object to check against — the requirement set and the verifiable signal live on different surfaces.

## Key Insight

A workspace's state-transition relation plus its task-declared terminal goal uniquely determines a lattice of reachability obligations, so evaluation requirements can be read off the workspace itself as the pre-/post-condition pairs that every goal-reaching trajectory must satisfy, without any human writing them.

## Method

**What it is.** IDRA (Invariant-Derived Requirement Algebra) takes as input a workspace state $s_0$, a task specification $T=(G, C)$ with terminal goal predicate $G$ and declared constraints $C$, and the workspace's executable transition operator $\delta$ (the same command interface SWE-agent exposes), and outputs a requirement hierarchy $\mathcal{H}$: a DAG of nodes, each a triple $(\text{precondition}, \text{action-class}, \text{postcondition})$ annotated with a *necessity certificate* and a *checkability tier*. It evaluates an agent trajectory $\tau$ by assigning each node a three-valued status (satisfied / violated / unobservable) and rolling these up the DAG.

**Load-bearing assumption (stated explicitly).** IDRA assumes that $\delta$ is a sound and complete effect-axiom model of the real workspace: that the four action-classes (file-edit, exec, nav, query) capture every state change relevant to $G$, and that the modeled effects actually occur. This is a closed-world assumption; it fails when the workspace has unmodeled side effects (background daemons, environment variables, package installs, non-deterministic test flakiness). Under this assumption, weakest-precondition regression over $\delta$ yields preconditions that are genuinely necessary for reaching $G$ in the actual environment. When the assumption fails, the derived preconditions over-approximate necessity and the score is a conservative lower bound on progress. To make the assumption testable rather than assumed, IDRA includes a model-repair loop (Phase 2b) that audits $\delta$ against observed rollout outcomes and reports a measured soundness rate.

**How it works.**

```pseudocode
INPUT: s0, T=(G,C), transition operator delta, probe budget B, audit sample size K
OUTPUT: requirement DAG H, trajectory verdict V(tau), model-fidelity report

PHASE 1 - Goal regression (backward)
  frontier <- {G};  H <- {};  depth <- 0
  while frontier nonempty and depth < D_max:
      post <- pop(frontier)
      for each action-class a in delta.action_classes (file-edit, exec, nav, query):
          pre <- weakest_precondition(post, a, s0)   # symbolic, via delta's effect axioms
          if pre is not TRUE and pre not implied by C:
              node <- (pre, a, post); node.cert <- regression_witness(pre,a,post)
              H.add(node); frontier.push(pre)
      depth += 1

PHASE 2 - Necessity certification (forward, counterfactual)
  for each node n in H:
      s_pre <- sample_state_satisfying(n.pre, s0, delta)   # reachable state
      tau_neg <- delta.rollout(s_pre, skip=n.a, target=n.post, budget=B)
      if G(tau_neg.final) is FALSE:
          n.necessity <- CERTIFIED        # n.a is causally required for G
      else:
          n.necessity <- REDUNDANT; prune(n)

PHASE 2b - Model-repair loop (audit delta against observed outcomes)
  # Each Phase 2 skip-rollout is also a test of the effect axiom for action-class a.
  # For a sample of K nodes (K=32 by default), record the observed outcome of tau_neg.
  for each audited node n in sample(H, K):
      observed <- outcome(tau_neg)   # did postcondition hold? did G hold?
      predicted <- delta.predict(s_pre, skip=n.a)   # model's predicted outcome
      if observed contradicts predicted:
          retract_or_update(effect_axiom(n.a))   # e.g., add unmodeled side-effect clause
          re-run PHASE 1 for the affected sub-DAG (nodes whose pre/post mention n.a)
  soundness_rate <- (# audited nodes where observed == predicted) / K
  # Report soundness_rate alongside every necessity certificate.

PHASE 3 - Checkability tiering
  for each surviving node n:
      if n.post is decidable from observable state alone:  n.tier <- T1
      elif n.post is decidable from state + declared constraint C:  n.tier <- T2
      else:  n.tier <- T3   # unobservable intermediate; kept, flagged

PHASE 4 - Trajectory evaluation
  for each node n in H (topological order):
      status[n] <- check(n, tau)   # satisfied / violated / unobservable
      if status[n]==violated and all ancestors satisfied: mark n as blocking
  V(tau) <- (fraction of CERTIFIED T1/T2 nodes satisfied,
             set of blocking nodes, set of T3 nodes reached,
             soundness_rate from Phase 2b)
```

Hyperparameters: regression depth $D_max=6$; probe budget $B=8$ rollouts per node (adaptive: start with $B=2$, increase to $B=8$ only for nodes whose skip-rollout outcome is ambiguous, i.e., $G$ is reached in some but not all probe rollouts); audit sample size $K=32$ nodes for Phase 2b; action-class set fixed to the four ACI verb families, with the initial implementation restricted to file-edit and exec.

**Why this design.** We chose *symbolic goal regression over the workspace's own transition operator* rather than learning requirements from demonstrations, because the transition operator is the only object that is both task-agnostic and causally tied to goal reachability, accepting the cost that the method is limited to workspaces with an explicit or executable $\delta$. We chose *counterfactual necessity certification* over simply keeping every regressed precondition, because naive regression produces many redundant obligations (e.g., "file was read" when the edit could have been blind), accepting the cost of $B$ extra rollouts per node. We chose *three-valued status with a T3 tier* over forcing a binary verdict, because under partial observability a binary verdict would silently convert unobservability into failure, accepting the cost that the final score is a pair rather than a scalar. We added the *model-repair loop (Phase 2b)* because the soundness of $\delta$ is the load-bearing assumption: without auditing it, a necessity certificate certifies model-necessity, not world-necessity. The loop treats each skip-rollout as a test of the effect axiom, retracts contradicted axioms, and reports a measured soundness rate so the score is explicitly a lower bound under a stated model-fidelity budget. This is not a variant of Agent-as-a-Judge or of process-reward-model step scoring: those consume a human-written or learned requirement/step-value set, whereas IDRA *derives* the requirement set from workspace structure and certifies each element by intervention — a domain expert would describe it as program-analysis-style precondition inference, not as judging.

**Why it measures what we claim.** The fraction of certified T1/T2 nodes satisfied measures *requirement-identifiability-adjusted task progress* because goal regression over a sound transition operator yields preconditions that are logically necessary for $G$, under the assumption that the workspace's effect axioms are sound (every modeled action effect actually occurs); this assumption fails when the workspace has side effects outside the modeled action classes (e.g., background daemons mutating files), in which case the derived preconditions over-approximate necessity and the score reflects a conservative lower bound on progress. The counterfactual necessity flag measures *causal requirement status* because skipping action $a$ from a state satisfying its precondition and failing to reach the postcondition is evidence that $a$ is on every goal-reaching path, under the assumption that the probe rollout's failure is attributable to the skip rather than to probe-policy weakness; this assumption fails when the probe policy is too weak to reach the postcondition even without the skip, in which case the node is falsely certified and the hierarchy contains a spurious obligation. The T3 tier measures *residual unverifiability* because a node is T3 exactly when its postcondition is not decidable from observable state plus declared constraints, under the assumption that the observable-state projection is fixed and complete for T1/T2; this assumption fails when an observable channel is noisy, in which case some T2 nodes are mis-tiered as T3 and the unobservability count overstates the true gap. The Phase 2b soundness rate measures *model fidelity* because it is the fraction of audited skip-rollouts whose observed outcome matches the model's prediction, under the assumption that the audit sample is representative of the nodes in $\mathcal{H}$; this assumption fails when the sample is biased toward easy nodes, in which case the reported soundness rate overstates the fidelity of $\delta$ on the hard nodes that actually determine the score.

## Contribution

(1) IDRA, a requirement-derivation mechanism that constructs a necessity-certified, checkability-tiered requirement DAG directly from a workspace's transition operator and task goal, replacing the human annotation oracle assumed by Agent-as-a-Judge. (2) A counterfactual necessity-certification procedure that prunes redundant regressed obligations by intervening on the workspace, together with a three-valued trajectory verdict that separates violation from unobservability instead of collapsing the two. (3) A design principle — that partial-observability evaluation should report a (satisfied-fraction, blocking-set, unobservable-set) triple rather than a scalar — supported by the tiering analysis.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | SWE-bench verified subset (500 instances) | Real repo issues with executable workspace and ground-truth patches |
| Dataset | Data-analysis sandbox (SQL/file ops, 100 tasks) | Tests generality across a different transition operator |
| Primary metric | Requirement-identifiability-adjusted progress (fraction of certified T1/T2 nodes satisfied) | Directly operationalizes the claimed construct; penalizes unverifiable progress |
| Secondary metric | Model soundness rate (Phase 2b, K=32) | Quantifies the load-bearing assumption empirically |
| Baseline | SWE-agent pass@1 | Tests whether requirement-level scoring adds signal beyond end-task success |
| Baseline | Process-reward-model step scoring | Tests whether derived+certified requirements beat learned step values |
| Baseline | Naive goal regression without necessity certification | Tests whether counterfactual pruning is what fixes false obligations |
| Baseline | Standard invariant detector (Daikon-style) on command log | Tests whether domain-specific regression plus certification is necessary |
| Ablation-of-ours | IDRA without T3 tier (binary collapse) | Tests whether three-valued status prevents unobservability-as-failure |
| Ablation-of-ours | IDRA without Phase 2b model-repair loop | Tests whether auditing δ changes the blocking set or score ordering |
| Ablation-of-ours | IDRA with fixed B=2 vs adaptive B | Reports trade-off between probe budget and certification accuracy |

### Why this setup validates the claim

The central claim is that IDRA yields a requirement hierarchy that is both causally necessary (certified by intervention) and correctly tiered by checkability, and that this hierarchy produces a progress signal that is more diagnostic than end-task pass/fail or learned step scores. SWE-bench verified gives us a workspace with an explicit executable transition operator (file edit, exec, nav, query), which is exactly the precondition for Phase 1 regression to be sound; without an executable delta the method has no input. The data-analysis sandbox provides a second workspace with a different transition operator (SQL/file operations) to test whether the method generalizes beyond the four-verb ACI model. SWE-agent pass@1 tests the *incremental* sub-claim: if IDRA's score merely re-predicts pass@1, then requirement-level evaluation adds nothing, so we need to see that IDRA separates resolved from unresolved trajectories even when pass@1 is tied. The PRM baseline tests the *derivation* sub-claim: if a learned step-value model matches IDRA's blocking-node set, then the symbolic regression and counterfactual certification were unnecessary. The naive-regression baseline tests the *certification* sub-claim: if pruning redundant nodes does not change the blocking set or the score ordering, then Phase 2 is doing no work. The Daikon-style invariant detector baseline tests the *domain-specificity* sub-claim: if a standard invariant detector on the command log produces the same blocking set, then IDRA's regression over the workspace's transition operator adds nothing beyond generic invariant mining. The T3-collapse ablation tests the *tiering* sub-claim: if binary collapse produces the same verdicts as the three-valued rollup, then unobservability is not a real phenomenon in this workspace. The Phase 2b ablation tests the *model-fidelity* sub-claim: if auditing δ does not change the blocking set or score ordering, then the load-bearing assumption is not load-bearing in practice. The primary metric is the right one because it is the quantity the method claims to estimate — requirement-identifiability-adjusted progress — and because it is a pair (fraction satisfied, blocking set), which lets us detect the predicted effect (blocking nodes concentrated on violated T1/T2 requirements) rather than a single scalar that could be right for the wrong reason. The model soundness rate is reported alongside every score so that the necessity certificates are explicitly conditional on a measured model-fidelity budget.

### Expected outcome and causal chain

**vs. SWE-agent pass@1** — On a case where the agent eventually passes the hidden tests by a lucky late edit after an earlier blind edit that violated a certified precondition (e.g., editing a file without reading its import structure), pass@1 reports success and cannot distinguish this from a clean trajectory, because its only signal is the terminal test outcome. IDRA instead flags the earlier node as a blocking violation (the precondition "file read before edit" is T1-checkable from the command log) and records the trajectory's progress as partial with a named blocking node. We expect a noticeable divergence on the subset of resolved trajectories that contain at least one certified-T1 violation, and parity on trajectories with no such violations — i.e., IDRA's score should separate clean from lucky resolutions, which pass@1 cannot.

**vs. Process-reward-model step scoring** — On a case where the agent performs a plausible-looking sequence of exploratory reads and greps that a PRM would score as high-quality process, but skips the one certified action (e.g., running the failing test to observe the actual error before editing), the PRM produces a high step score because its learned prior rewards exploration, while the trajectory is actually missing a causally necessary action. IDRA instead marks the skipped node as a blocking violation because Phase 2 certified it by counterfactual skip-rollout, so the score drops and the blocking set names the omission. We expect IDRA's blocking set to contain requirements that the PRM scores as satisfied or neutral, concentrated on the certified-necessity nodes that naive regression would have kept but that a learned model would not have learned to weight.

**vs. Naive goal regression without necessity certification** — On a case where the goal requires an edit to a config file and naive regression adds the obligation "config file was read first," the naive hierarchy contains a redundant node that the agent may never satisfy (it edits blind), so the naive score under-reports progress and its blocking set contains a false obligation. IDRA's Phase 2 skip-rollout shows the edit can reach the postcondition without the read, certifies the node as REDUNDANT, and prunes it, so the blocking set contains only causally required nodes. We expect the naive baseline to produce larger blocking sets and lower scores on trajectories that use blind or batched edits, with the gap concentrated on trajectories whose only violations are pruned-redundant nodes.

**vs. Standard invariant detector (Daikon-style) on command log** — On a case where the agent's command log exhibits a stable pattern (e.g., always runs `grep` before `edit`) that Daikon mines as an invariant, the detector reports that invariant as a requirement, but it is not causally necessary for the goal (the edit could succeed without the grep). IDRA's Phase 2 skip-rollout certifies the grep node as REDUNDANT and prunes it, so the blocking set contains only causally required nodes. We expect the Daikon baseline to produce larger blocking sets and lower scores on trajectories that deviate from the mined pattern without violating causal necessity, with the gap concentrated on trajectories whose only violations are pattern-based but not causally necessary.

**vs. IDRA without Phase 2b model-repair loop** — On a case where the workspace has an unmodeled side effect (e.g., a background daemon that mutates a file after an edit), the un-audited δ predicts that the edit's postcondition holds, so the node is certified as satisfied, but the actual state violates the postcondition. IDRA with Phase 2b detects the contradiction during the audit, retracts the effect axiom, re-runs regression for the affected sub-DAG, and reports a lower soundness rate, so the blocking set correctly includes the violated node. We expect the un-audited variant to produce smaller blocking sets and higher scores on trajectories affected by unmodeled side effects, with the gap concentrated on the audited nodes where the soundness rate is below 1.0.

### What would falsify this idea

If IDRA's blocking-node set is uncorrelated with the actual failure point identified by post-hoc human or test-based diagnosis (i.e., the named blocking node is not the action whose repair flips the outcome), or if the T3-collapse ablation produces identical verdicts to the three-valued rollup on trajectories with partial observability, or if the Phase 2b soundness rate is below 0.8 on the audited sample and the blocking set does not change when the model-repair loop is enabled, then the central claim that derived-and-certified requirements measure requirement-identifiability-adjusted progress would be wrong.

## References

1. Work Agents: Systems, Capabilities, and Optimization
2. ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery
3. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
