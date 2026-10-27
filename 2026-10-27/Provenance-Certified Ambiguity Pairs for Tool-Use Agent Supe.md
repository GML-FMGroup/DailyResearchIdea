# Provenance-Certified Ambiguity Pairs for Tool-Use Agent Supervision

## Motivation

Synthesis pipelines are becoming the sole supervision source for tool-use agents (Trajectory2Gym, DreamGym, ReTool), but a learner trained on synthesized interactions cannot be distinguished from one that has memorized the generator's idiosyncratic resolution of under-specified instructions. Trajectory2Gym's structural-support selection and parent-normalized SFT treat each synthesized interaction as grounded supervision while leaving every generator-side latent decision (which tool schema was assumed, which ambiguity slot was resolved which way) unobserved and untested, so generator bias is an unverifiable assumption rather than a measurable property of the interaction. The field therefore lacks a synthesis-time criterion certifying that learner decisions are invariant to non-recoverable generator variables and that ambiguous-instruction decision states are actually covered.

## Key Insight

If two synthesized interactions are constructed to share an identical learner-visible history and differ only in a generator-side latent decision that the history cannot recover, then any learner decision that diverges across the pair is attributable to generator bias, making invariance a directly computable property of the synthesized supervision rather than an assumption about it.

## Method

**What it is.** PROCA (Provenance-Certified Ambiguity Pairs) is a synthesis wrapper around a trajectory generator (instantiated on Trajectory2Gym's pipeline) that emits, for every synthesized tool-use interaction, a provenance certificate recording the minimal set of generator-side latent decisions, and that additionally emits paired interactions differing only in one under-specified instruction slot. Its output is a training set where each interaction carries a certificate and a paired twin, plus a per-example invariance weight used in supervised fine-tuning.

**How it works.**

```pseudocode
# Input: generator G (Trajectory2Gym), base task distribution D, learner pi, budget N
# Fixed hyperparameters: k = 3 matched executions per instruction; divergence threshold tau = 0.1
# (fraction of tool-call trace positions that may differ); K = 5 grounded decisions per slot value
# for the multi-validity check; recoverability threshold r_l > 0.5; collapse weight lambda = 0.3.
for each synthesized interaction h = (instruction I, history H, decision a) from G:
    # Phase 1: provenance certificate
    L = G.latent_log()            # ordered latent decisions used to build h
    L_min = minimal_subset(L, criterion=recoverable(H, l) == False)  # keep only latents NOT recoverable from H
    cert(h) = { l : l in L_min, value: v_l, recoverability_score: r_l }

    # Phase 2: ambiguity pair via single-slot intervention
    slot = argmax_l over L_min of ambiguity_score(l)   # slot with highest generator discretion
    I' = resample_instruction_slot(I, slot)            # same history H, different slot value

    # Phase 2b: matched-execution validity gate (nondeterminism control)
    # Execute I and I' each k=3 times from the same H; collect visible histories H_i and H'_i.
    # Require the two visible-history distributions to be statistically indistinguishable:
    #   (i) identical tool-call trace (sequence of tool names and arguments) across all k runs of I,
    #   (ii) identical tool-call trace across all k runs of I',
    #   (iii) trace divergence between I and I' below tau = 0.1 (fraction of differing positions).
    # If any condition fails, mark pair as ungrounded (coverage = -1) and discard from training.
    if not matched_execution(I, I', H, k=3, tau=0.1):
        cert(h).coverage = -1
        continue

    # Phase 2c: multi-validity check (legitimate-ambiguity control)
    # Sample K=5 grounded decisions per slot value from the environment; require the decision sets
    # for the two slot values to be disjoint. If they overlap, the task admits multiple equally valid
    # decisions and collapse cannot be attributed to generator bias; mark pair as ungrounded.
    if not disjoint_decision_sets(I, I', H, K=5):
        cert(h).coverage = -1
        continue

    h' = G.execute(I', H)        # executable twin, same visible history (verified above)
    pair(h, h') = (a, a')         # learner-visible decisions

    # Phase 3: certificate check (synthesis-time, learner-dependent)
    if grounded(a, H) and grounded(a', H) and a != a':
        cert(h).coverage = 1     # ambiguity state covered: learner distinguishes the pair
    elif grounded(a,H) and grounded(a',H) and a == a':
        cert(h).coverage = 0     # learner collapses the pair -> generator bias suspected
    else:
        cert(h).coverage = -1    # ungrounded; discard from training

    # Phase 4: invariance-weighted SFT
    w(h) = 1.0 if cert(h).coverage == 1 else 0.3 if cert(h).coverage == 0 else 0.0
    add (h, w(h)) to training set

train pi with parent-normalized SFT weight w(h) * parent_weight(h)  # lambda = 0.3 for collapsed pairs
```

Hyperparameters: recoverability threshold on r_l set to 0.5 (a latent is declared non-recoverable if a held-out probe cannot predict its value from H above chance); collapse weight lambda = 0.3; pair budget one twin per interaction; matched-execution repetitions k = 3; trace-divergence threshold tau = 0.1; multi-validity samples K = 5.

**Why this design.** We chose to record only the *minimal non-recoverable* latent subset rather than the full latent log because a full log makes every interaction trivially 'certified' (nothing is invariant to a variable that determines the output), accepting the cost that minimality requires a recoverability probe per latent. We chose a *single-slot resampling* intervention over full re-synthesis of the twin because re-synthesis changes many latents at once and makes divergence unattributable, accepting the cost that we cannot certify invariance to latent interactions. We chose to *down-weight* (lambda = 0.3) rather than discard collapsed pairs because collapsed pairs are exactly the generator-bias signal we want the learner to be exposed to under a corrective weight, accepting the cost that lambda is a tuned hyperparameter. We added the matched-execution gate (k=3, tau=0.1) because the single-slot intervention is only a controlled intervention if the twin's visible history H' is verified equal to H; without this gate, environment nondeterminism would be silently absorbed as 'ambiguity-state coverage.' We added the multi-validity check (K=5, disjoint decision sets) because collapse is only attributable to generator bias if the two slot values demand distinct grounded decisions; otherwise collapse reflects legitimate ambiguity. A domain expert would not describe this as a variant of Trajectory2Gym or DreamGym: those methods *generate* grounded interactions, whereas PROCA *audits* the generator by constructing a counterfactual twin whose divergence is a measurable certificate, which is closer in spirit to a randomized controlled trial than to experience synthesis.

**Why it measures what we claim.** The recoverability score `r_l` measures *non-recoverability of a generator latent from learner-visible history* because we assume a held-out probe trained on H is a sufficient test of whether H determines l; this assumption fails when the probe underfits (l is recoverable but the probe misses it), in which case r_l reflects probe capacity rather than true non-recoverability. The single-slot pair divergence `a != a'` measures *ambiguity-state coverage* because we assume resampling exactly one instruction slot while holding H fixed is a valid intervention that isolates the slot's causal effect on the decision; this assumption fails when the executed twin's visible history H' differs from H due to environment nondeterminism, in which case divergence reflects environment noise rather than slot sensitivity — the matched-execution gate (k=3, tau=0.1) is the concrete control that detects this failure and marks the pair ungrounded (coverage = -1) rather than attributing divergence to the slot. The certificate coverage flag measures *absence of generator bias* because we assume a learner that has learned the task (not the generator) will resolve a genuinely under-specified slot distinctly when the two slot values demand distinct grounded decisions; this assumption fails when the task admits multiple equally valid decisions, in which case collapse reflects legitimate ambiguity rather than bias — the multi-validity check (K=5, disjoint decision sets) is the concrete control that detects this failure and marks the pair ungrounded.

## Contribution

(1) A provenance-certificate schema and minimal-non-recoverable-latent extraction procedure that turns each synthesized tool-use interaction into an auditable object with an explicit generator-bias surface. (2) A single-slot ambiguity-pair construction and certificate-coverage criterion that operationalizes ambiguous-instruction decision-state coverage as a computable synthesis-time quantity, yielding invariance-weighted parent-normalized SFT. (3) An analysis of which generator latents are non-recoverable in Trajectory2Gym-style synthesis and how collapsed pairs distribute across structural-support categories.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Tau1 + Tau2 + BFCL held-out | Compositional tool-use tasks with under-specified slots |
| Primary metric | Accuracy on ambiguity-pair twins (a≠a' rate) | Directly measures invariance to under-specified slots |
| Baseline | Trajectory2Gym SFT | Tests whether naive grounded synthesis suffices |
| Baseline | Full-latent-log certification | Tests whether minimal non-recoverable subset matters |
| Baseline | Collapse-discard (λ=0) | Tests whether down-weighting beats discarding bias signal |
| Baseline | Single-execution twin (no matched-execution gate) | Tests whether nondeterminism control matters |
| Ablation-of-ours | PROCA without provenance certificate (pairs only) | Isolates certificate's contribution from pairing |
| Ablation-of-ours | PROCA without multi-validity check | Isolates legitimate-ambiguity control from bias signal |

### Why this setup validates the claim

The central claim is that auditing a generator with a provenance-certified counterfactual twin produces training signal that a raw trajectory generator cannot, and that this signal is what transfers to under-specified tool-use decisions. Trajectory2Gym SFT is the right baseline because it is the strongest instance of the "just generate grounded interactions" hypothesis: if PROCA's gain over it is real, the gain must come from the certificate and pairing, not from grounding. The full-latent-log baseline tests the minimality sub-claim specifically — if minimality is doing no work, this baseline should match PROCA, and any gap between them isolates the recoverability filter as causal. The collapse-discard baseline tests the down-weighting sub-claim: if collapsed pairs carry corrective signal, discarding them should underperform λ=0.3 exactly on the subsets where generator bias is concentrated. The single-execution-twin baseline tests the matched-execution gate: if environment nondeterminism is not controlled, the certificate coverage flag will absorb noise as coverage, and this baseline should show inflated coverage but no accuracy gain on the ambiguity-pair metric. The pairs-only ablation strips the certificate while keeping the twin, so any remaining gain over Trajectory2Gym is attributable to pairing alone, and the residual gap to full PROCA is the certificate's contribution. The no-multi-validity ablation strips the disjoint-decision-set check, so any gain over full PROCA on the ambiguity-pair metric isolates the legitimate-ambiguity control. The primary metric — accuracy on held-out ambiguity-pair twins — is the right detector because it is the only metric that requires the learner to have internalized that a slot is under-specified and to resolve it consistently with the instruction, which is precisely the behavior the certificate is supposed to certify. Standard task accuracy would be confounded by memorization of generator conventions and would not distinguish a learner that has learned the task from one that has learned the generator. We also report the discard rate from the matched-execution gate (fraction of pairs failing k=3, tau=0.1) and from the multi-validity check (fraction failing K=5 disjointness) as diagnostics.

### Expected outcome and causal chain

**vs. Trajectory2Gym SFT** — On a case where the instruction under-specifies a slot (e.g., which of two equally reachable files to open) and the generator's latent convention silently picks one, Trajectory2Gym SFT produces the generator's convention as if it were the task, because it supervises on single grounded trajectories with no counterfactual. Our method instead exposes the learner to the twin where the same history demands the other slot value, and weights it so the learner must condition on the instruction rather than the history. We expect a noticeable gap on the under-specified-slot subset of Tau1/Tau2 and near-parity on fully-specified tasks, since the certificate only fires where a non-recoverable latent exists.

**vs. Full-latent-log certification** — On a case where the generator's latent log contains many decisions but only one is unrecoverable from history, full-log certification marks the interaction as invariant to every latent, so the twin is resampled on a slot the history already determines. The baseline produces a twin whose divergence is zero or noise, because there is nothing to intervene on, and the certificate coverage flag becomes uninformative. Our method instead restricts to the minimal non-recoverable subset, so the resampled slot is one the history genuinely cannot pin down, and divergence is attributable. We expect the full-log baseline to show inflated certificate coverage but no accuracy gain on the ambiguity-pair metric, while PROCA shows coverage that tracks accuracy.

**vs. Collapse-discard (λ=0)** — On a case where the learner collapses the pair (a == a'), the collapse-discard baseline removes the example entirely, so the learner never sees the generator-bias signal. The baseline thus produces a learner that is clean on well-specified tasks but retains the generator's convention on exactly the under-specified slots where bias lives, because the corrective gradient was deleted. Our method instead keeps the collapsed pair at weight 0.3, so the learner receives a weak but present signal that the two instructions should have produced different decisions. We expect collapse-discard to match PROCA on well-specified subsets but to lag on the under-specified subset, with the gap concentrated where collapse rates were highest during synthesis.

**vs. Single-execution twin (no matched-execution gate)** — On a case where the environment is nondeterministic (e.g., a web-search tool with rate limits or an LLM-backed tool), executing I' once from H produces a visible history H' that differs from H due to environment noise, not slot sensitivity. The single-execution baseline attributes the resulting divergence a != a' to the slot's causal effect, so its certificate coverage flag is inflated by noise. Our method instead executes both I and I' k=3 times and requires identical tool-call traces within each instruction and trace divergence below tau=0.1 between instructions before admitting the pair; pairs failing this gate are marked ungrounded (coverage = -1) and discarded. We expect the single-execution baseline to show higher certificate coverage but lower accuracy on the ambiguity-pair metric than PROCA, with the gap concentrated on tasks whose tools are nondeterministic, and we report the discard rate from the matched-execution gate as a diagnostic.

**vs. PROCA without multi-validity check** — On a case where the task admits multiple equally valid decisions (e.g., two files are both acceptable to open), the no-multi-validity ablation admits the pair and treats collapse (a == a') as generator bias, even though collapse reflects legitimate ambiguity. The baseline thus down-weights (λ=0.3) examples where the learner's collapse is correct, injecting a spurious corrective signal. Our method instead samples K=5 grounded decisions per slot value and requires the decision sets to be disjoint; pairs failing this check are marked ungrounded (coverage = -1) and discarded. We expect the no-multi-validity ablation to underperform PROCA on the ambiguity-pair metric on subsets where legitimate ambiguity is common, and we report the discard rate from the multi-validity check as a diagnostic.

### What would falsify this idea

If PROCA's gain over Trajectory2Gym SFT is uniform across fully-specified and under-specified subsets rather than concentrated on the under-specified subset, then the certificate is not doing what it claims and the gain is a generic regularization effect. Likewise, if the pairs-only ablation matches full PROCA on the ambiguity-pair metric, the provenance certificate contributes nothing beyond pairing and the central claim is wrong; and if the single-execution-twin baseline matches PROCA on the ambiguity-pair metric, the matched-execution gate is not doing what it claims and environment nondeterminism is not a confound in this setup.

## References

1. Trajectory2Gym: Decision-Grounded Compositional Interaction Synthesis for Tool-Use Agents
2. Scaling Agent Learning via Experience Synthesis
3. ReTool: Reinforcement Learning for Strategic Tool Use in LLMs
4. Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents
5. WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning
