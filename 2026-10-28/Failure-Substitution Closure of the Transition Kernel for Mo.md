# Failure-Substitution Closure of the Transition Kernel for Model-Based Agent Environments

## Motivation

Environment models in agent training pipelines are built from a human-authored action decomposition and silently assume that decomposition is action-complete: CoworkEnv defines its transition kernel and programmatic validators only over the authored workflow action set A_spec, and DreamGym distills a reasoning-based experience model over the same authored step granularity, so both leave T(s'|s,a) undefined precisely on the recovery actions a' that become available only after an authored action a fails. Long-horizon agents spend most of their recovery budget on these out-of-spec actions, which means the environment model is undefined exactly where the agent's behavior matters most, and no per-workflow validator can fix this because the missing actions are not enumerable at authoring time.

## Key Insight

Failure substitution is an equivalence relation on action instances — two actions are substitutable exactly when they share the same precondition-violation signature at the same state — so the recovery action class is recoverable as the closure of A_spec under that equivalence, identifiable from failure traces alone without ever enumerating recovery actions at authoring time.

## Method

**What it is.** FSC (Failure-Substitution Closure) takes as input a teacher execution corpus E (the same traces CoworkEnv uses to define A_spec), the authored action set A_spec, and a base transition model T_0, and outputs an augmented transition kernel T* defined on the substitution closure cl(A_spec) = A_spec ∪ {a' : a' substitutable for some a ∈ A_spec}. The only supervision it adds is the failure signature attached to each attempted action in E.

**Load-bearing assumption (stated explicitly).** The violated-precondition id extracted from failure observations is treated as a sound and complete signature of action substitutability: two action instances that fail on the same authored precondition clause with the same argument-slot grounding are assumed interchangeable at that state, so the equivalence class R_k recovered from teacher traces is assumed to be exactly the set of recovery actions available and valid there. This assumption is the load-bearing one for the closure; it is not verified by construction, and the method below includes a concrete calibration step (Phase 2b) that measures and prunes violations of it before the fixed-point closure runs.

**How it works.**

```pseudocode
INPUT: teacher traces E = {(s_t, a_t, o_t, s_{t+1})}, authored set A_spec, base model T_0
OUTPUT: closed kernel T*, recovery action vocabulary R

# Phase 1: extract failure signatures from teacher traces
for each transition (s_t, a_t, o_t, s_{t+1}) in E:
    if o_t is a failure observation (precondition violated / error return):
        sig(a_t, s_t) <- (violated_precondition_id, argslot_vector(s_t, a_t))
        # violated_precondition_id: which precondition of a_t was false at s_t
        # argslot_vector: the grounded argument slots of a_t at s_t

# Phase 2: build the substitution equivalence class
# two action instances are substitutable iff they fail on the SAME precondition
# with the SAME argument-slot grounding pattern
for each failure signature k = (p, v):
    R_k <- { a' observed in E succeeding at some s with sig(a', s) sharing p and v }
    # R_k is the recovery class for precondition p under grounding v

# Phase 2b: effect-conditioned calibration and held-out substitutability gate
# (prunes spurious merges from shared error codes before the closure runs)
for each k = (p, v), for each candidate a' in R_k:
    # effect match: compare observed post-state effect distribution of a' at s
    # to the failing action a's intended effect, using a learned effect embedding
    # (2-layer MLP, hidden=256, GeLU, trained on (s, a, s') triples in E to predict
    #  the post-state delta; embedding dimension 128; cosine similarity threshold 0.7)
    if cos_sim(effect_embed(a', s), effect_embed(a, s)) < 0.7:
        R_k <- R_k \ {a'}   # spurious merge from shared error code
    # held-out substitutability gate: does swapping a for a' preserve task progress?
    # evaluated on a held-out 20% split of E (stratified by precondition id);
    # a' is retained only if swap-progress rate >= 0.8 on that split
    if swap_progress_rate(a', k; heldout_E) < 0.8:
        R_k <- R_k \ {a'}
R <- union_k R_k

# Phase 3: closure operator on the action set
cl(A_spec) <- A_spec
repeat:
    for a in cl(A_spec), for k in signatures(a):
        cl(A_spec) <- cl(A_spec) ∪ R_k
until fixed point   # monotone, terminates because R is finite

# Phase 4: fit the closed kernel
T* <- fit_transition_model( E ∪ {(s, a', s') : a' in R, s' from teacher success traces} )
# T* is defined on cl(A_spec); T*(.|s,a) = T_0(.|s,a) for a in A_spec
return T*, R
```

Hyperparameters: the precondition-id granularity (one id per authored precondition clause, taken directly from the CoworkEnv workflow requirement), the grounding-vector tolerance (exact match; no fuzzy matching), the effect-embedding cosine threshold (0.7), and the held-out swap-progress threshold (0.8 on a 20% stratified held-out split of E).

**Why this design.** We chose a signature-equivalence closure over a learned recovery-action classifier because the classifier would need labeled recovery actions that no authoring pipeline produces, accepting the cost that the closure only recovers recovery actions whose failure signatures were actually observed in teacher traces. We chose to key substitutability on the violated-precondition id rather than on surface action similarity because two actions that fail on the same precondition are interchangeable at that state by construction of the precondition, accepting the cost that actions sharing a precondition but differing in side effects are merged into one class; Phase 2b's effect-conditioned calibration is the concrete step that measures and prunes this specific cost before the closure runs. We chose a monotone fixed-point closure over a single-pass expansion because substitutability composes — a recovery action can itself fail and open a further class — and the fixed point is the smallest set closed under the relation, accepting the cost that the closure can grow if the teacher corpus is noisy. We did not adopt a per-workflow validator as in CoworkEnv because a validator can only check authored actions; the closure is derived from the failure-conditional equivalence class and therefore extends beyond A_spec, which is exactly the property a per-workflow validator cannot provide and which distinguishes FSC from validator-augmented CoworkEnv or from the reasoning-based experience model of DreamGym.

**Why it measures what we claim.** The violated-precondition id sig(a_t, s_t) measures substitutability between a and a' because two actions are substitutable exactly when the same precondition clause is the one that failed at the same grounded state, under the assumption that authored preconditions are the complete failure surface of each action; this assumption fails when an action fails for a reason not captured by any authored precondition clause (e.g., an external service outage), in which case the signature reflects an environmental fault rather than an action-level substitutability relation and the recovered class R_k is spurious. The fixed-point closure cl(A_spec) measures action-completeness of the environment model because closure under the substitutability relation is equivalent to T* being defined on every action reachable by failure substitution from A_spec, under the assumption that teacher traces exercise every precondition at least once; this assumption fails when a precondition is never violated in E, in which case cl(A_spec) under-covers that precondition's recovery class and T* remains undefined on those recovery actions. The success-trace filter on R_k measures that a candidate recovery action is genuinely available at the failing state because it succeeded at a state with the same grounding vector, under the assumption (A) that argument-slot grounding determines availability and hidden state is irrelevant; this assumption fails (failure mode F) when availability depends on hidden state not in the grounding vector, in which case R_k includes actions that are not actually available and T* over-covers. Phase 2b's held-out swap-progress gate and effect-embedding filter are the concrete diagnostics that measure the rate at which these assumptions are violated on the teacher corpus.

## Contribution

(1) A failure-substitution closure operator on the authored action set that derives the recovery action vocabulary from failure signatures in teacher traces, making the transition kernel defined on recovery actions without any per-workflow validator. (2) The design principle that failure signatures (violated precondition id plus grounded argument slots) form an equivalence relation whose closure is the minimal action set on which the environment model must be defined. (3) An augmented transition kernel T* fitted on the closed action set, usable as the environment model inside CoworkEnv-style or DreamGym-style training pipelines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | CoworkEnv teacher traces + Claw-Eval | Provides failure signatures and held-out long-horizon tasks |
| Primary metric | Success rate on tasks requiring recovery after precondition failure | Directly probes closure beyond authored actions |
| Baseline | CoworkEnv validator-augmented agent | Tests whether closure beats authored-action-only checking |
| Baseline | DreamGym reasoning-based experience model | Tests whether signature closure beats learned recovery reasoning |
| Baseline | SFT-only PocketWorker | Tests whether FSC's kernel adds value over imitation |
| Baseline | Learned recovery-action classifier trained on the same traces | Tests whether the closure is merely a re-derivation of a known classifier |
| Ablation-of-ours | FSC without fixed-point closure (single-pass expansion) | Isolates compositional substitutability contribution |
| Ablation-of-ours | FSC without Phase 2b calibration (exact-match precondition-id only) | Isolates the effect-conditioned and held-out substitutability gate contribution |
| Diagnostic | Hidden-state perturbation probe on R_k | Measures false-positive rate of the grounding-vector availability assumption (A/F) |

### Why this setup validates the claim

The central claim is that keying substitutability on the violated-precondition id yields a transition kernel defined on recovery actions that authored-action-only methods cannot represent. The validator-augmented CoworkEnv baseline tests exactly this boundary: it can only validate actions in A_spec, so on tasks where the teacher's authored action fails and a substitutable recovery action is needed, it has no model for that action. DreamGym tests the alternative mechanism — a learned reasoning model of recovery — which should generalize differently but lacks the failure-signature grounding that makes FSC's closure sound. The SFT-only baseline tests whether any gain is merely from more imitation data rather than the closed kernel. The learned recovery-action classifier baseline tests whether the closure is merely a re-derivation of a known classifier trained on the same traces. The single-pass ablation is the sharpest test of compositional substitutability: if fixed-point composition matters, removing it should hurt specifically on instances where a recovery action itself fails and opens a further class, while leaving single-failure instances intact. The no-Phase-2b ablation isolates the contribution of the effect-conditioned calibration and held-out substitutability gate, and the hidden-state perturbation probe directly measures the false-positive rate of the grounding-vector availability assumption. Success rate on recovery-requiring tasks is the right primary metric because it is the only observable that directly reflects whether T* is defined and accurate on actions outside A_spec; a metric like overall task success would be diluted by instances that never trigger a failure.

### Expected outcome and causal chain

**vs. CoworkEnv validator-augmented agent** — On a case where the authored action's precondition is violated mid-trajectory and the correct continuation requires an action never in A_spec, the validator baseline produces a stall or a re-try of the same failing authored action because its validator can only accept or reject authored actions and has no transition model for the substitute. Our method instead looks up the failure signature, retrieves the recovery class R_k whose members passed the Phase 2b effect-match and held-out swap-progress gates, and continues with T* defined on that action because the closure explicitly extends the kernel to cl(A_spec). We expect a noticeable success-rate gap concentrated on tasks whose teacher traces contain at least one precondition violation, with near-parity on tasks that never trigger a failure.

**vs. DreamGym reasoning-based experience model** — On a case where the failure is a precondition violation with a specific argument-slot grounding, the reasoning model produces a plausible-sounding but ungrounded recovery action because it has no explicit signature-to-class mapping and must infer substitutability from surface semantics, which merges actions that share a precondition but differ in side effects or misses the grounding constraint. Our method instead restricts R_k to actions that succeeded under the same violated-precondition id and grounding vector and that passed the Phase 2b effect-embedding and swap-progress gates, so the recovered action is available by construction. We expect the gap to be largest on instances with fine-grained grounding distinctions (same precondition, different argument slots) and to shrink where failures are coarse and unambiguous.

**vs. SFT-only PocketWorker** — On a case where the demonstration corpus contains a failure followed by a recovery, the SFT-only model produces the recovery only if that exact state-action pair was memorized, and degrades when the failing state's grounding differs from any seen demonstration because it has no explicit equivalence class to generalize over. Our method instead generalizes across groundings within the same signature class via R_k, so it recovers on unseen grounding combinations. We expect SFT-only to match FSC on in-distribution failures but to fall behind on held-out failure instances whose grounding vector was not observed with the specific action, giving a gap that grows with the diversity of grounding patterns in the evaluation set.

**vs. Learned recovery-action classifier trained on the same traces** — On a case where the failure signature is shared by structurally different causes (e.g., the same error code returned for auth failure vs. rate limit), the classifier can learn to separate them from the effect features it sees, whereas the exact-match precondition-id closure merges them. We expect the no-Phase-2b ablation to fall behind the classifier on these spurious-merge instances, and full FSC (with Phase 2b) to match or exceed the classifier by pruning the same merges via the effect-embedding and swap-progress gates; if full FSC does not match the classifier, the closure is not adding value over the classifier and the claim is weakened.

**vs. FSC without Phase 2b calibration** — On instances where the same error code is returned for structurally different causes, the no-Phase-2b variant includes spurious members in R_k and T* over-covers, producing invalid recovery actions. Full FSC prunes these via the effect-embedding cosine threshold (0.7) and the held-out swap-progress threshold (0.8), so we expect the gap to be concentrated on instances with shared error codes and to be near-zero on instances with unique precondition ids.

**Hidden-state perturbation diagnostic** — We perturb hidden state (state components not in the grounding vector) at fixed (s, a') pairs drawn from R_k and measure the fraction of pairs whose availability flips; this false-positive rate directly quantifies how often assumption (A) fails and R_k over-covers, and we report it alongside the primary metric so the effectiveness claim is conditioned on a measured quantity rather than an unstated sufficiency assumption.

### What would falsify this idea

If FSC's gain over the validator baseline were uniform across all task subsets rather than concentrated on tasks containing precondition violations in the teacher traces, the claim that the closure is what produces the improvement would be wrong. Likewise, if the single-pass ablation matched full FSC on instances where a recovery action itself fails, the compositional fixed-point argument would be falsified; and if the hidden-state perturbation diagnostic showed a high false-positive rate in R_k that the Phase 2b gates did not reduce, the grounding-vector availability assumption would be falsified.

## References

1. CoworkEnv: Synthesizing Training Environments for Long-Horizon Cowork Agents
2. Scaling Agent Learning via Experience Synthesis
3. AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials
4. WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning
5. AgentTuning: Enabling Generalized Agent Abilities for LLMs
6. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
7. Android in the Wild: A Large-Scale Dataset for Android Device Control
8. WebArena: A Realistic Web Environment for Building Autonomous Agents
