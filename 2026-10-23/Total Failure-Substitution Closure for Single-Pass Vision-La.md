# Total Failure-Substitution Closure for Single-Pass Vision-Language-Action Policies

## Motivation

Across the VLA literature — π0.5, π0, RT-2 — the policy is trained to emit one committed action chunk per forward pass and has no representation of failure inside its output space, so the action set is not closed under substitution: when the committed chunk is found to be non-executable there is no operator that maps (observation, committed chunk, failure) to a substitute guaranteed executable by the same low-level controller. The root cause is structural rather than a data or capacity deficit: failure is treated as an external event (a fallback, a re-query, a human takeover) rather than an output-space element, so no training signal ever forces the substitute to be controller-feasible. π0.5's co-training and semantic-subtask prediction improve generalization but do not change this — the model still commits once and cannot certify that any replacement chunk lies in the controller's executable set.

## Key Insight

Executability by a fixed low-level controller is a decidable property of a chunk (it is the controller's own feasibility predicate), so a substitute generator can be trained as a total operator whose codomain is provably inside that predicate — closure under substitution becomes a structural invariant enforced by construction rather than a confidence-gated fallback.

## Method

**What it is.** FSC (Failure-Substitution Closure) augments a single-forward-pass VLA policy with two coupled heads trained jointly: a failure predicate `f_θ(obs, chunk) ∈ {0,1}` over the committed chunk, and a substitute generator `S_θ(obs, chunk, f)` that emits a replacement chunk. The pair is trained so that for every reachable `(obs, chunk)` the substitute lands in the controller's executable set `E`, making `S_θ` a total operator on the action set. **Load-bearing assumption (stated explicitly):** the frozen low-level controller's `is_executable` predicate is treated as the ground-truth definition of executability; the closure claim is conditional on this predicate being complete for the task, and we measure its completeness bound empirically (see block D and the experiment's predicate-completeness reporting).

**How it works.**

```pseudocode
# ---- Training phase (per batch) ----
for (obs, chunk_gt, controller) in batch:
    chunk_hat = policy(obs)                      # committed chunk, single forward pass
    f_hat     = f_theta(obs, chunk_hat)          # failure predicate (2-layer MLP, hidden=256, GeLU)
    sub       = S_theta(obs, chunk_hat, f_hat)   # substitute generator (flow-matching head, 4 Euler steps)

    # (1) executability certificate: run the REAL controller's feasibility check
    cert_sub  = controller.is_executable(sub)    # binary, from the frozen low-level controller
    cert_gt   = controller.is_executable(chunk_gt)

    # (2) contrastive executability loss: pull substitutes into E, push non-executable away
    #     d_phi: 2-layer MLP, hidden=256, GeLU, sigmoid output; trained with BCE on cert_sub
    L_exec = -[ cert_sub * log sigma(d_phi(sub)) + (1-cert_sub) * log(1 - sigma(d_phi(sub))) ]

    # (3) failure-predicate supervision from controller rejection
    L_fail = BCE(f_hat, 1 - cert_gt)

    # (4) closure loss: substitute must be executable AND task-progressing
    L_closure = || sub - chunk_gt ||^2  *  cert_sub   # masked by certificate

    L = L_exec + lambda_f * L_fail + lambda_c * L_closure   # lambda_f=1.0, lambda_c=0.5

# ---- Inference (single forward pass + rejection-sampling substitution) ----
obs -> policy -> chunk_hat
if f_theta(obs, chunk_hat) == 1:
    # generate K=8 candidate substitutes, filter by the REAL controller predicate,
    # return the first that passes; if none pass, return the candidate with highest d_phi score
    # and flag the state as outside the closure guarantee (see block D)
    candidates = [S_theta(obs, chunk_hat, 1) for _ in range(K)]
    for cand in candidates:
        if controller.is_executable(cand):
            chunk_hat = cand
            break
    else:
        chunk_hat = argmax_{cand} d_phi(cand)
        flag_outside_guarantee(obs)
return chunk_hat
```

The controller feasibility predicate `is_executable` is the frozen low-level controller's own check (joint limits, velocity bounds, collision-free IK). `d_phi` is a learned executability scorer (2-layer MLP, hidden=256, GeLU, sigmoid output) trained with the certificate as label. The substitute generator is a small flow-matching head (same architecture family as π0's action head, 4 Euler steps) conditioned on `(obs, chunk_hat, f_hat)`.

**Why this design.** We chose to make the controller's feasibility predicate the training label rather than a learned proxy because the whole claim rests on the substitute being *actually* executable, and a learned proxy would reintroduce the very gap we are closing — accepting the cost that training requires a differentiable or at least callable controller check at every step. We chose a contrastive executability loss over a plain regression-to-ground-truth loss because regression pulls substitutes toward the demonstrated chunk (which may itself be non-executable in the current state) whereas the contrastive form explicitly separates the executable and non-executable manifolds — accepting the cost of needing both positive and negative certificates per batch. We chose to condition the substitute on the failed committed chunk rather than on `obs` alone because the substitute must *repair* a specific failure, not re-sample from scratch; this is the structural difference from a standard fallback policy (e.g., a re-query of π0.5), which discards the failed chunk and has no guarantee about the replacement. A domain expert would not describe this as a variant of π0.5 because π0.5 never represents failure in its output space and has no executability certificate; the closure operator is a new output-space element, not a better re-query. It is also not a retrieval scheme (anti-pattern 2): the substitute is generated by a flow head conditioned on the failure, not selected from a library, and the certificate is a property of the generated chunk, not a nearest-neighbor match.

**Why it measures what we claim.** `cert_sub = controller.is_executable(sub)` measures *controller-feasibility of the substitute* because the assumption that the controller's predicate is the ground-truth definition of executability holds by construction (the controller is the thing that executes); this assumption fails when the controller's predicate is itself incomplete (e.g., a contact-rich task where the predicate misses dynamic infeasibility), in which case the certificate reflects kinematic feasibility only. `L_fail` measures *failure-predicate correctness* because the assumption that controller rejection is a sufficient label for failure holds when the controller is the sole executor; it fails when the policy fails for reasons the controller accepts (semantic failure), in which case `f_hat` under-predicts failure. `L_closure` masked by `cert_sub` measures *task-progressing closure* because the assumption that the demonstrated chunk is a valid repair target holds when demonstrations are executable in the current state; it fails in distribution shift where the demonstrated chunk is itself non-executable, in which case the closure loss is zeroed by the mask and only `L_exec` drives learning. The totality claim is therefore: for every `(obs, chunk)` where the controller predicate is complete and the demonstrated chunk is executable, `S_θ` is trained to land in `E`; outside those conditions the certificate still reports feasibility but the task-progress guarantee is not claimed. **Equivalence assumption A (d_phi-to-controller calibration):** `d_phi`'s contrastive score measures controller-feasibility because the assumption that `d_phi` is calibrated to the controller predicate on the substitute distribution (not just the training batch) holds when the substitute distribution stays within the training manifold; this assumption fails when substitutes drift off the training manifold and `d_phi` false-accepts, in which case `d_phi` reflects a learned proxy rather than the controller predicate. We report `d_phi`'s false-accept rate against the real controller on held-out rejection states to bound this gap.

## Contribution

['A failure-substitution closure operator for single-forward-pass VLA policies, comprising a failure predicate over the committed action chunk and a substitute generator trained with a controller-feasibility certificate so that the operator is total on the executable set.', "A contrastive executability objective that uses the frozen low-level controller's own feasibility check as the training label, turning controller-feasibility into a differentiable training signal without a learned proxy.", "A design principle: representing failure inside the policy's output space (rather than as an external fallback event) is what makes the action set closed under substitution, and this closure is trainable and checkable at inference time with one extra substitution step."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Long-horizon dexterous manipulation suites (kitchen/bedroom cleaning, unseen homes) | Matches π0.5 open-world regime where controller rejections occur |
| Primary metric | Executable-chunk rate under controller check on failure-triggering states | Directly measures the closure claim, not proxy success |
| Secondary metric | `d_phi` false-accept rate vs. real controller on held-out rejection states | Bounds the d_phi-to-controller equivalence assumption A |
| Secondary metric | Predicate-completeness bound: fraction of inference-time states flagged outside the closure guarantee (K=8 rejection-sampling failures) | Reports the measured completeness bound of the controller predicate |
| Baseline | π0.5 re-query fallback | Tests whether re-sampling from obs beats failure-conditioned repair |
| Baseline | π0 single-forward-pass policy | Tests whether closure head adds value over base VLA |
| Baseline | Learned executability-filtered resampler | Tests whether certificate-grounded generation beats post-hoc filtering |
| Ablation-of-ours | FSC without certificate in L_closure (regression-only substitute) | Isolates the certificate's contribution to closure |
| Ablation-of-ours | FSC without rejection-sampling at inference (single substitute, no K=8 filter) | Isolates the contribution of the inference-time rejection-sampling guarantee |

### Why this setup validates the claim

The central claim is that FSC makes the substitute a total operator into the controller's executable set, and that this closure—not better re-sampling—drives gains. The π0.5 re-query baseline tests the sub-claim that failure-conditioned repair beats discarding the failed chunk: if re-query matched FSC, the structural argument collapses. The π0 baseline tests whether any single-forward-pass VLA without an explicit failure output space can achieve the same executable-chunk rate, isolating the closure operator's contribution. The filtered-resampler baseline tests whether the certificate must be inside the generator's training signal rather than applied post-hoc. The ablation removes the certificate mask from L_closure, so if the certificate is doing the work, the ablation should regress toward the regression-to-ground-truth failure mode. The primary metric is executable-chunk rate on states where the committed chunk is controller-rejected, because that is precisely the population where the closure claim is non-vacuous—on executable states all methods trivially agree. The secondary metrics directly bound the load-bearing assumption: the `d_phi` false-accept rate on held-out rejection states measures whether the learned scorer is calibrated to the real controller predicate (assumption A), and the predicate-completeness bound reports the fraction of inference-time states where the K=8 rejection-sampling loop fails to find an executable substitute, making the conditional nature of the closure claim explicit rather than assumed away.

### Expected outcome and causal chain

**vs. π0.5 re-query fallback** — On a case where the committed chunk violates joint limits near a cluttered counter, the baseline re-queries from `obs` alone and re-samples a fresh chunk from the same policy distribution that just produced the infeasible one, so it has no mechanism to avoid re-entering the non-executable manifold and often emits a second infeasible chunk. Our method instead conditions the substitute on the specific failed chunk and is trained with the controller certificate as label, so it repairs the violation rather than re-sampling; at inference, the K=8 rejection-sampling loop returns only a chunk that passes the real controller predicate. We expect a noticeable executable-chunk-rate gap concentrated on rejection-triggering states, with near-parity on already-executable states.

**vs. π0 single-forward-pass policy** — On a case where the policy's committed chunk is rejected for collision-free IK failure, π0 has no failure representation in its output space and returns the infeasible chunk unchanged, so the controller rejects it and the episode stalls or the low-level loop faults. Our method detects the rejection via `f_θ` and applies one substitution step into the executable set. We expect π0 to show near-zero recovery on rejection states and FSC to show a large recovery rate, with the gap vanishing on states where π0's chunk was already executable.

**vs. learned executability-filtered resampler** — On a case where a candidate chunk is marginally non-executable, the baseline draws several samples and keeps the first that passes a learned post-hoc filter, which can accept a chunk the real controller rejects because the filter is a proxy. Our method trains the generator against the real controller certificate and additionally filters at inference with the real controller predicate (K=8 rejection sampling), so accepted substitutes are actually executable. We expect the gap to be modest in aggregate but concentrated on the marginal-infeasibility subset, where the proxy filter's false-accept rate is highest; we report `d_phi`'s false-accept rate on held-out rejection states to quantify this gap.

### What would falsify this idea

If FSC's executable-chunk-rate gain over π0.5 re-query and π0 were uniform across both rejection-triggering and already-executable states, rather than concentrated on the rejection states, the closure mechanism would not be the cause and the central claim would be wrong. Equivalently, if the certificate-free ablation matched FSC on rejection states, the certificate would be doing no work; and if the `d_phi` false-accept rate on held-out rejection states were high (indicating assumption A fails), the equivalence between the learned scorer and the controller predicate would be unsupported.

## References

1. π0.5: a Vision-Language-Action Model with Open-World Generalization
2. PaliGemma: A versatile 3B VLM for transfer
3. π0: A Vision-Language-Action Flow Model for General Robot Control
4. Visual Instruction Tuning
5. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
6. RT-1: Robotics Transformer for Real-World Control at Scale
7. Self-Instruct: Aligning Language Models with Self-Generated Instructions
8. Scaling Instruction-Finetuned Language Models
