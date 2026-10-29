# Belief-Conditioned User Oracles: Measuring Interactive Adaptivity via Mutual Information with Agent Belief Trajectories

## Motivation

Across interactive agent benchmarks, the evaluation signal is a function of a pre-authored artifact rather than a closed-loop function of the agent's internal state: RealCLI's REALCLI-COLLAB mode scripts a user that delivers progressive information, corrections, and authorizations from a fixed template, so its response distribution does not condition on what the agent currently believes. Because the oracle never queries the agent's evolving belief, interactive adaptivity is neither established nor measurable — RealCLI's reported collaboration scores (peak J_collab 37.6%) reflect how well an agent matches a script, not whether the user simulation actually adapts to agent uncertainty. This meta-gap recurs whenever the user oracle is a template rather than a probabilistic program whose latent variable is the agent's belief state.

## Key Insight

If the oracle's next utterance is sampled from an explicit posterior over the agent's inferred belief, then the mutual information between oracle turns and the agent's belief trajectory becomes an estimable quantity that certifies adaptivity, because a template oracle — regardless of its richness — has zero such mutual information by construction.

## Method

**What it is.** We introduce BCO (Belief-Conditioned Oracle), a user simulator that replaces the pre-authored script in interactive agent benchmarks. Inputs: a structured task ontology (entities, constraints, authorization states, hidden facts), the agent's observable actions and utterances, and a held-out probe battery. Output: the next user utterance, sampled from a posterior over the agent's inferred belief state, plus a per-episode adaptivity certificate (a lower bound on mutual information between oracle turns and the agent's belief trajectory).

**Load-bearing assumption (stated explicitly).** The certificate MI(U; B_t) is claimed to measure adaptivity to the agent's *true* belief Z only under assumption A: that B_t is a calibrated posterior over Z, i.e. the probe likelihood P(a|q,Z) is correct and the conditional-independence assumption holds (agent answers depend only on the probed fact given Z). When A fails — e.g. the agent's answer is driven by a correlated unprobed fact it inferred indirectly — B_t is a mixture over probed and unprobed beliefs, and MI(U; B_t) certifies adaptivity to a probe-elicited belief report rather than to Z. We therefore (i) replace the single-probe Bayes-factor update with an explicit measurement model P(a|q,Z) whose calibration is validated on a synthetic set where Z is known by construction, and (ii) report the certificate alongside a calibration bound (ECE of B_t against ground-truth Z on the synthetic set) so the reader can see how faithful B_t is to Z.

**How it works.**

```pseudocode
# Phase 0: Measurement-model calibration (offline, synthetic, Z known)
# Synthetic tasks: 200 episodes per ontology slot, Z ground-truth logged.
# Fit measurement model P(a|q,Z) as a 2-layer MLP (hidden=128, GeLU)
#   over features [probe_id, answer_id, Z_onehot]; trained with cross-entropy
#   on 80% of synthetic episodes, calibrated on held-out 20%.
# Report ECE(B_t, Z) on held-out synthetic set; certificate is reported
#   only for ontologies with ECE < 0.10 (else flagged 'uncalibrated').

# Phase 1: Belief probe construction (offline, per task ontology)
for each task T with ontology O:
    probes[T] = {contrastive query pairs (q, q') that differ only in
                 one hidden fact or constraint; answerable by agent}

# Phase 2: Belief inference from probe responses (per episode)
B_0 = uniform over latent belief states Z (structured: which facts agent knows,
      which constraints it has registered, which authorizations it holds)
for turn t = 1..N:
    if t in scheduled_probe_turns (every k=3 agent turns):
        emit contrastive probe q_t from probes[T]
        observe agent answer a_t
        B_t = posterior_update(B_{t-1}, q_t, a_t)   # Bayes over Z via
            # calibrated P(a|q,Z) from Phase 0 (not a raw Bayes factor)
    else:
        B_t = B_{t-1}  # belief carried forward from last probe

# Phase 3: Belief-conditioned utterance sampling
    z_hat ~ B_t                                  # sample a belief hypothesis
    u_t ~ P(u | z_hat, O, dialogue_history)      # probabilistic-program oracle
    emit u_t to agent
    record (u_t, B_t) pair

# Phase 4: Adaptivity certificate
MI_lower = estimate_MI_lower_bound({(u_t, B_t)})
# Info-NCE critic: 2-layer MLP (hidden=256, GeLU), temperature tau_nce=0.07,
# trained 200 steps Adam lr=1e-3 on (u_t, B_t) positives vs.
# (u_t, B_{t'}) shuffled negatives (t' uniform over other episodes);
# gives variational lower bound on I(U; B).
certify episode if MI_lower > tau (tau = 0.05 nats, set by null-template calibration)
# Certificate is reported only when ECE(B_t, Z) < 0.10 on the synthetic
# calibration set for that ontology; otherwise episode is flagged
# 'uncalibrated' and excluded from pass-rate aggregation.
```

The oracle `P(u | z_hat, O, history)` is a small probabilistic program over the ontology: it selects which hidden fact or constraint to reveal based on `z_hat` (e.g., if the sampled belief says the agent has not registered constraint c, the oracle emits a clarification about c), with a fixed emission grammar per ontology slot. Probe turns are held out from scoring so that probing does not contaminate task outcome.

**Why this design.** We chose a structured probabilistic program over a neural user simulator because the latent variable must be *interpretable as belief* for the posterior to be well-defined — a neural simulator would require an additional identifiability argument we cannot supply, accepting the cost that the oracle's utterance diversity is bounded by the ontology's slot grammar. We chose contrastive held-out probes over free-form belief elicitation because contrastive pairs give a likelihood ratio that is a proper Bayes factor for the belief update, accepting the cost of scheduled probe turns that slightly perturb the dialogue. We chose a variational Info-NCE lower bound over a plug-in MI estimator because the bound is valid under finite samples without distributional assumptions on U, accepting the cost that the bound is loose (it certifies adaptivity only when MI is above the null-template floor). We chose to schedule probes every k=3 turns rather than every turn because dense probing dominates the dialogue, accepting the cost that belief estimates are stale between probes. We added the Phase-0 measurement-model calibration because the raw Bayes-factor update assumes probe answers are faithful readouts of Z, which the method's own failure analysis shows can be false; the calibrated P(a|q,Z) and the ECE gate make the faithfulness assumption testable per ontology rather than asserted. A domain expert would not describe BCO as a variant of RealCLI's scripted user or of SWE-bench's fixed issue description: those are pre-authored artifacts with no latent variable, whereas BCO's emission distribution is a function of a posterior over agent belief, which is what makes the adaptivity certificate non-vacuous.

**Why it measures what we claim.** The computational quantity `B_t` (posterior over Z after probe responses, using the calibrated P(a|q,Z)) measures *the agent's inferred belief state* because the probe likelihood ratio is a Bayes factor for the hidden-fact hypothesis under the assumption that agent answers are conditionally independent of other hidden facts given the probed fact and that P(a|q,Z) is calibrated; this assumption fails when the agent's answer is driven by a correlated fact it inferred indirectly or when the measurement model is miscalibrated, in which case `B_t` reflects a mixture of probed and unprobed beliefs (and the ECE gate flags the ontology as uncalibrated). The quantity `MI_lower` (Info-NCE bound between `u_t` and `B_t`) measures *oracle adaptivity to agent belief* because the variational lower bound equals the true mutual information under the assumption that the critic class contains the true density ratio; this assumption fails when the critic underfits rare belief states, in which case `MI_lower` reflects adaptivity only over well-represented beliefs. The quantity `tau` (null-template floor) measures *the adaptivity threshold attributable to chance* because it is calibrated by running the same MI estimator on a template oracle whose emissions are independent of `B_t`; this fails when the template oracle happens to correlate with belief through task structure, in which case `tau` is conservative and the certificate is harder to pass — an acceptable direction of error. The quantity `ECE(B_t, Z)` on the synthetic set measures *the faithfulness of B_t to the true belief Z* because it compares the posterior's predicted belief distribution to the ground-truth Z logged by construction; this fails when the synthetic tasks do not cover the ontology's hidden-fact combinations, in which case ECE reflects calibration only on covered combinations and the gate is optimistic — we therefore stratify the synthetic set by ontology slot and report per-slot ECE.

## Contribution

["BCO, a belief-conditioned user oracle implemented as a probabilistic program over a structured task ontology, whose emission distribution is a function of a posterior over the agent's inferred belief state rather than a pre-authored script.", 'A contrastive-probe belief inference procedure and an Info-NCE adaptivity certificate that turns interactive adaptivity into an estimable mutual-information lower bound with a null-template-calibrated threshold.', "A design principle — 'the evaluation signal must be a measurable function of agent belief' — instantiated as a drop-in replacement for scripted users in interactive benchmarks, with probe scheduling that keeps probing out of the scored outcome."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | REALCLI-COLLAB interactive tasks, fixed subset of 20 ontologies with a single shared probe grammar | Scripted user is the artifact BCO replaces; bounded scope |
| Dataset (calibration) | Synthetic tasks with ground-truth Z logged (200 episodes per ontology slot) | Validates B_t against known belief |
| Primary metric | Adaptivity certificate pass rate (episodes with ECE < 0.10 only) | Directly measures belief-conditioned oracle turns |
| Secondary metric | ECE(B_t, Z) on synthetic set, per ontology slot | Quantifies faithfulness of B_t to true belief |
| Baseline | REALCLI scripted user | No latent belief; fixed script baseline |
| Baseline | Neural user simulator (LLM persona) | Adaptive-looking but no belief posterior |
| Baseline | Template oracle (belief-independent) | Null floor for chance adaptivity |
| Ablation-of-ours | BCO without probe turns | Tests whether posterior needs probes |
| Ablation-of-ours | BCO with raw Bayes-factor update (no Phase-0 calibration) | Tests whether calibrated P(a|q,Z) matters |

### Why this setup validates the claim

The central claim is that replacing a pre-authored user script with a belief-conditioned oracle produces measurable, non-vacuous adaptivity to the agent's inferred belief state, and that this adaptivity is what scripted benchmarks cannot supply. REALCLI-COLLAB is the right substrate because it already pairs a scripted interactive user with task execution, so the only variable we change is the emission mechanism of the user, holding task ontology and scoring fixed. The scripted-user baseline tests the sub-claim that a fixed script yields near-zero mutual information with agent belief; the neural-simulator baseline tests the sub-claim that adaptivity without an interpretable posterior cannot be certified, since its Info-NCE bound should sit near the null floor despite fluent utterances; the template oracle calibrates the null floor itself, so a certificate passing above tau is not attributable to task structure. The ablation (BCO minus probes) isolates whether the posterior update from contrastive probes is doing the work, or whether belief-conditioned sampling alone suffices. The second ablation (BCO with raw Bayes-factor update) isolates whether the Phase-0 calibrated measurement model is doing the work, directly testing the load-bearing assumption A: if the raw-Bayes-factor variant's certificate pass rate diverges from the calibrated variant's on ontologies with correlated hidden facts, then B_t is not a faithful belief estimate and the certificate measures probe-elicited belief reports rather than true belief. The synthetic calibration set with ground-truth Z makes ECE(B_t, Z) computable, so the faithfulness assumption is measured rather than asserted, and the ECE < 0.10 gate ensures the reported pass rate is over ontologies where B_t is validated. The certificate pass rate is the correct primary metric because it is a per-episode, thresholded quantity derived from a variational lower bound; it directly operationalizes the claim rather than proxying it with task success, which conflates oracle adaptivity with agent competence.

### Expected outcome and causal chain

**vs. REALCLI scripted user** — On a case where the agent silently misregisters an authorization constraint and proceeds to attempt a forbidden action, the scripted user produces its next pre-authored clarification about an unrelated slot because its emission distribution is fixed at authoring time and cannot condition on what the agent has registered. Our method instead samples a belief hypothesis in which the constraint is unregistered and emits a clarification targeting that constraint because the posterior over Z shifts after the scheduled probe, so we expect certificate pass rates near the null floor for the scripted baseline and a clear separation for BCO concentrated on episodes containing hidden constraints, with parity on episodes whose ontology has no hidden facts.

**vs. Neural user simulator (LLM persona)** — On a case where the agent reveals through a probe answer that it has inferred a hidden fact indirectly, the neural simulator emits a plausible but belief-agnostic turn because its emission is conditioned on dialogue history, not on an explicit posterior, so its Info-NCE bound against B_t stays near the shuffled-negative floor. Our method instead updates B_t via the calibrated P(a|q,Z) from the contrastive probe and samples from the corresponding ontology slot because the posterior is the sampling variable, so we expect BCO to clear tau on episodes where the simulator's utterances are fluent yet uncorrelated with B_t, i.e. a gap in certificate pass rate despite comparable utterance naturalness.

**vs. Template oracle (belief-independent)** — On a case where task structure happens to align slot order with belief order, the template oracle emits turns that correlate with B_t by coincidence, inflating raw MI estimates; this is precisely why tau is calibrated on it. Our method instead must exceed that floor because its emissions are a function of the sampled z_hat, so we expect the template oracle to sit at or just below tau by construction and BCO to exceed it on the subset where probes actually move the posterior, with the gap shrinking where belief is stale between probes.

**vs. BCO with raw Bayes-factor update (no Phase-0 calibration)** — On a case where the agent's probe answer is driven by a correlated unprobed fact, the raw Bayes-factor update treats the answer as a direct readout of the probed fact, so B_t becomes a mixture and the certificate may pass on a probe-elicited artifact. Our calibrated variant instead uses P(a|q,Z) fitted on synthetic episodes with known Z, so we expect the raw variant to show higher ECE(B_t, Z) on ontologies with correlated hidden facts and a certificate pass rate that diverges from the calibrated variant's on exactly those ontologies, while the two variants agree on ontologies with independent hidden facts.

### What would falsify this idea

If BCO's certificate pass-rate gain over the neural simulator is uniform across episodes rather than concentrated on episodes with hidden facts and authorization states, or if the probe-free ablation matches full BCO, or if the raw-Bayes-factor variant matches the calibrated variant on ontologies with correlated hidden facts (i.e. ECE(B_t, Z) is low without calibration), then the adaptivity is not coming from a faithful belief posterior and the central claim is wrong.

## References

1. RealCLI: Toward a Unified Evaluation of Human-Agent Collaboration for CLI Agents
2. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
