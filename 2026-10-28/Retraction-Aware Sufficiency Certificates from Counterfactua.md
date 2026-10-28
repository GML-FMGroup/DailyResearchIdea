# Retraction-Aware Sufficiency Certificates from Counterfactual Evidence Trajectories for Agentic RAG

## Motivation

Agentic RAG benchmarks such as InfoDeepSeek and CausalRAG score only determinate queries with authoritative grounding, so the joint event 'the answer is unsupported by retrieved evidence AND an already-committed retrieval/tracing decision would be reversed under further search' never enters the operational space and no calibrated signal exists for it. Existing sufficiency signals (log-probs, post-hoc verifiers, causal-graph continuity scores) observe only the forward trajectory the agent actually took, so they cannot detect a commitment that later evidence would have contradicted; the two decisions of abstaining and retracting are consequently treated as independent or absent. The structural gap is that sufficiency estimation and decision retraction are never coupled through a quantity that observes the reversal itself.

## Key Insight

A counterfactual evidence trajectory — the retrieval path the agent would have taken after a mid-stream evidence perturbation — is the only observable that carries information about commitment reversal, and calibrating a single scalar against held-out reversal-probe pairs makes that scalar monotone in the joint event of insufficiency and wrong early commitment by construction, which no forward-pass signal can be.

## Method

**What it is.** We introduce RASC (Retraction-Aware Sufficiency Certificate): for each query and each committed retrieval/tracing decision, RASC outputs a scalar c in [0,1] computed from a counterfactual evidence trajectory, calibrated on held-out reversal probes, and used in one monotone rule to gate both abstention and retraction.

**How it works.**

```pseudocode
INPUT: query q, agent policy pi, retriever R, commitment log C = [(t_i, d_i)] (decision, evidence)
HYPERPARAMS: perturbation budget k=3, divergence threshold tau=0.15, calibration quantile alpha=0.9

PHASE 1 - forward commitment trace
  T_fwd = run pi(q, R)  # yields final answer a and ordered commitments C

PHASE 2 - counterfactual evidence trajectory (per commitment)
  for each commitment (t_i, d_i) in C:
    for j in 1..k:
      q' = q  # query held fixed
      context' = context with d_i removed (all other retrieved chunks and prior commitments unchanged)
      seed_j = fixed random seed j
      T_cf[j] = run pi(q', context', R, seed=seed_j) restricted to steps >= i  # re-search from the commitment point
    divergence_i = mean_j KL( commit_dist(T_cf[j]) || commit_dist(T_fwd) )
    reversal_i   = 1[ divergence_i > tau ]   # would the commitment be reversed?

PHASE 3 - sufficiency + retraction coupling
  support_score = evidence_coverage(a, T_fwd)      # fraction of answer spans entailed by retrieved chunks
  reversal_score = max_i reversal_i * divergence_i  # worst reversed commitment
  raw_c = 1 - (1 - support_score) * (1 - reversal_score)   # joint-event complement

PHASE 4 - calibration on reversal probes
  D_probe = held-out pairs (T, T') differing only in one mid-stream evidence perturbation
  fit isotonic regression g: raw_c -> P(insufficient AND reversed) on D_probe
  certificate c = g(raw_c); store g, tau, alpha

PHASE 5 - monotone decision rule
  if c > alpha:  ABSTAIN (do not answer)
  elif reversal_score > tau and c <= alpha:  RETRACT commitment argmax_i divergence_i and re-run from step i
  else:  COMMIT answer a
OUTPUT: certificate c, decision in {ABSTAIN, RETRACT, COMMIT}
```

**Why this design.** We chose a counterfactual re-search trajectory over a post-hoc verifier because a verifier only sees the evidence the agent already retrieved and therefore cannot observe whether a commitment would have been reversed; the cost is k extra retrieval passes per commitment, which we bound by restricting re-search to steps at or after the commitment point. We chose isotonic regression on held-out reversal probes over log-prob calibration because log-probs are not monotone in task correctness (Kadavath et al. show this directly) while the probe pairs are constructed so that the joint event is ordered by construction; the cost is that calibration requires a reversal-probe set that must be built per corpus. We chose a single scalar c gating both abstention and retraction over two independent heads because the joint event is a conjunction and a single monotone rule preserves the coupling; the cost is coarser control than two thresholds would give. A domain expert would not describe RASC as a variant of CausalRAG or InfoDeepSeek because neither computes a counterfactual trajectory nor calibrates against reversal probes — CausalRAG scores forward causal continuity and InfoDeepSeek scores forward accuracy/utility/compactness, both of which are blind to the reversal event.

**Why it measures what we claim.** `divergence_i` measures commitment reversibility because the assumption that perturbing the inducing evidence chunk is a valid intervention on the commitment — i.e., the chunk is the sufficient cause of the decision — makes divergence equivalent to reversal; this assumption fails when the commitment was driven by query phrasing rather than the chunk, in which case divergence reflects query sensitivity instead of reversal. `support_score` measures evidence sufficiency because the assumption that answer spans entailed by retrieved chunks are exactly the supported spans makes coverage equivalent to sufficiency; this assumption fails on multi-hop answers whose support is distributed across chunks not jointly retrieved, in which case coverage under-counts support. `c = g(raw_c)` measures the joint event probability because the assumption that the reversal-probe distribution matches deployment queries makes isotonic calibration transfer; this assumption fails under distribution shift in query difficulty, in which case c reflects probe-set difficulty rather than deployment sufficiency.

**(D) Assumption A and failure mode F.** The load-bearing assumption A is: the inducing chunk d_i is the sufficient cause of commitment t_i, so removing d_i while holding the query and all other context fixed is a valid intervention on t_i. Under A, divergence_i measures commitment reversal. Failure mode F: the commitment is driven by query phrasing rather than d_i, so divergence_i reflects query sensitivity instead of reversal. To empirically separate A from F, we add a diagnostic probe that varies query phrasing while holding d_i fixed (paraphrase q -> q_para, keep d_i and all other context unchanged) and computes divergence_i^query = KL(commit_dist(T_cf^query) || commit_dist(T_fwd)). We report the fraction of commitments where divergence_i^query > tau (query-sensitivity rate) and the fraction where divergence_i > tau but divergence_i^query <= tau (reversal-specific rate).

## Contribution

(1) RASC, a per-query per-commitment sufficiency certificate computed from a counterfactual evidence trajectory and calibrated on held-out reversal probes, with a single monotone rule that jointly gates abstention and retraction. (2) A reversal-probe construction procedure — paired trajectories differing only in one mid-stream evidence perturbation — that makes the certificate monotone in the joint insufficiency-and-reversal event by construction, and an analysis of when the perturbation-is-a-valid-intervention assumption breaks. (3) A retraction mechanism that re-runs the agent from the reversed commitment point rather than restarting, and an empirical characterization of the abstention/retraction trade-off as the calibration quantile alpha varies.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | InfoDeepSeek dynamic agentic queries | Stresses mid-stream commitment reversal under re-search |
| Dataset | Multi-hop QA with injected contradictions (e.g., HotpotQA + synthetic contradiction set) | Tests generalization of reversal phenomenon beyond InfoDeepSeek |
| Dataset | Tool-use agent trace set (e.g., ToolBench) | Tests generalization to tool-call commitments |
| Primary metric | Selective accuracy at matched abstention rate | Couples coverage and correctness, mirrors gated decision |
| Secondary metric | Reversal-specific detection rate (fraction of true reversals flagged) | Directly measures the joint event |
| Baseline | CausalRAG (forward causal continuity) | Tests sub-claim: forward scores miss reversals |
| Baseline | InfoDeepSeek utility/compactness scorer | Tests sub-claim: forward utility blind to reversal |
| Baseline | Standard RAG + logprob abstention | Tests sub-claim: logprobs not monotone in correctness |
| Baseline | Leave-one-chunk-out influence scoring (counterfactual attribution) | Tests whether re-search trajectory adds beyond chunk ablation |
| Ablation-of-ours | RASC without counterfactual trajectory (support only) | Isolates reversal term's contribution |
| Ablation-of-ours | RASC with query-phrasing perturbation instead of chunk removal | Tests assumption A vs failure mode F |
| Diagnostic | Query-sensitivity rate and reversal-specific rate (see method (D)) | Bounds proxy validity |
| Cost sweep | Perturbation budget k ∈ {1,2,3,5} and probe-set size ∈ {100, 500, 1000} | Identifies minimal viable configuration |

### Why this setup validates the claim

The central claim is that a calibrated counterfactual-reversal certificate detects insufficient evidence that forward-only scores cannot see, enabling better abstain/retract/commit decisions. InfoDeepSeek provides dynamic queries where evidence arrives mid-stream and can be invalidated, so the reversal event actually occurs and is measurable. The two additional datasets (multi-hop QA with injected contradictions and tool-use traces) test whether the reversal phenomenon and the gain generalize beyond a single benchmark. Selective accuracy at a matched abstention rate is the right metric because RASC's value is precisely in trading coverage for correctness via gating; a metric that ignores abstention (raw accuracy) would be blind to the mechanism. CausalRAG tests whether forward causal continuity captures reversals; InfoDeepSeek's utility scorer tests whether forward utility captures them; logprob abstention tests the calibration sub-claim. The leave-one-chunk-out influence baseline tests whether the re-search trajectory carries information beyond simple chunk-ablation influence, directly addressing the novelty concern. The support-only ablation is decisive: if removing the counterfactual trajectory does not degrade selective accuracy at the same abstention rate, the reversal term is doing no work and the design collapses to a coverage heuristic. The query-phrasing ablation and the diagnostic rates directly measure assumption A versus failure mode F: if divergence under query paraphrase is as high as under chunk removal, the intervention is not valid and the certificate reflects query sensitivity. The cost sweep shows the compute/annotation burden is tractable. Together these form a falsifiable test because each baseline isolates one sub-claim and the ablation isolates the core mechanism.

### Expected outcome and causal chain

**vs. CausalRAG** — On a case where a later retrieved chunk contradicts an earlier commitment, CausalRAG scores forward causal continuity and sees a coherent chain, so it commits to the stale answer because its graph continuity is computed only over retrieved evidence. Our method removes the inducing chunk, re-searches with the query held fixed, and observes high divergence at that commitment, so it retracts and re-runs from that step. We expect a noticeable selective-accuracy gap concentrated on contradiction-bearing subsets and parity on single-hop subsets where no reversal occurs.

**vs. InfoDeepSeek utility/compactness scorer** — On a case where the agent retrieves compact, high-utility chunks that nonetheless omit a disambiguating fact, the utility scorer rates the trajectory highly and commits because compactness and utility are forward properties of retrieved content. Our method's counterfactual trajectory reveals that removing the inducing chunk reverses the commitment, driving reversal_score up and triggering abstention. We expect higher selective accuracy on the omission subset and parity on subsets where evidence is complete, with the gap scaling with reversal frequency.

**vs. Standard RAG + logprob abstention** — On a case where the model is confidently wrong after a mid-stream evidence swap, logprob abstention stays below threshold and answers because log-probs are not monotone in correctness. Our isotonic-calibrated certificate maps the joint insufficient-and-reversed event to a probability, so confidence is replaced by a probe-calibrated signal. We expect the abstention curve to separate correct from incorrect answers where logprob abstention's does not, visible as a larger area under the selective-accuracy curve on reversal-heavy queries.

**vs. Leave-one-chunk-out influence scoring** — On a case where removing the inducing chunk changes the commitment, leave-one-chunk-out influence scoring also detects the change, but it cannot distinguish whether the change is due to reversal or query sensitivity because it does not re-search with the query held fixed. Our method's re-search trajectory with fixed query and seed isolates reversal from query drift. We expect RASC to outperform leave-one-chunk-out on reversal-bearing subsets, with the gap largest on queries where the commitment is driven by the chunk rather than phrasing (high reversal-specific rate).

**vs. RASC with query-phrasing perturbation** — This ablation replaces chunk removal with query paraphrase while holding the chunk fixed. If assumption A holds, chunk removal should produce higher divergence on true reversals than query paraphrase, and the certificate should degrade when using query paraphrase. We expect the query-phrasing variant to have lower selective accuracy and a higher false-positive rate on non-reversal queries, confirming that the chunk-removal intervention is necessary.

### What would falsify this idea

If RASC's selective-accuracy gain over the support-only ablation is uniform across subsets rather than concentrated on reversal-bearing queries, the counterfactual trajectory is not the causal driver and the central claim is wrong. Equivalently, if the query-sensitivity rate is high (divergence under query paraphrase as large as under chunk removal) and certificate error correlates with it, the chunk-as-sufficient-cause assumption fails and the joint-event interpretation collapses.

## References

1. InfoDeepSeek: Benchmarking Agentic Information Seeking for Retrieval-Augmented Generation
2. Benchmarking Large Language Models in Retrieval-Augmented Generation
3. Large Language Models with Controllable Working Memory
4. Few-shot Learning with Retrieval Augmented Language Models
5. CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation
6. Retrieval-Generation Synergy Augmented Large Language Models
