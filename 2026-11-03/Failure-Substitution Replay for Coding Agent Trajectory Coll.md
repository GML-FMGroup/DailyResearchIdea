# Failure-Substitution Replay for Coding Agent Trajectory Collection

## Motivation

Trajectory collectors such as SWE-Trail guide collection toward promising states but leave the action-selection process at those states unchanged, so a re-collected rollout can re-enter the same flagged state and replay the identical failure mode, producing redundant trajectories that add no new learning signal. This is a structural consequence of treating each rollout as an independent sample: the collector inherits no memory of which actions already failed at a state, and therefore has no mechanism to force behavioral divergence on re-entry. The bottleneck recurs across any history-guided collector whose only guidance is where to collect, not what not to repeat.

## Key Insight

A per-state failure memory combined with a substitution rule that provably excludes every previously failed action at re-entry makes the collector's action distribution at flagged states disjoint from prior failures, so re-collection is guaranteed to explore a behaviorally distinct branch rather than replay a known dead end.

## Method

**What it is.** Failure-Substitution Replay (FSR) is a drop-in replacement for the action-selection step of a history-guided collector. Its inputs are the current rollout state `s`, the per-issue failure memory `M` (a map from state fingerprints to sets of failed action signatures), and the agent policy `π`. Its output is a single action `a` that is guaranteed not to match any action signature recorded as failed at `s`.

**Load-bearing assumption (stated explicitly).** FSR assumes that the state fingerprint `fingerprint(s)` reliably identifies genuine state re-entry: that a failed action signature recorded at fingerprint `fp` corresponds to the same dead end when the agent later lands on a state with the same `fp`. This assumption is load-bearing because the exclusion invariant is a guarantee about a hash bucket, not about behavior; if fingerprints do not track the failure-relevant state, the divergence guarantee evaporates. We make this assumption explicit and test it directly via the calibration study in the experiment section (agreement between signature-level distinctness and post-action environment-effect distinctness).

**How it works.**

```pseudocode
# State fingerprint: hash of (file, line-range, normalized AST diff, last tool-call type)
def fingerprint(s): return hash(s.file, s.line_range, norm(s.ast_diff), s.last_tool_type)

# Action signature: (tool_name, normalized-argument skeleton) — e.g., (edit, "<file>:<line>:<diff-op>")
def signature(a): return (a.tool, norm_args(a.args))

def collect(issue, policy, M, K=8, tau=3):
    traj = []
    s = env.reset(issue)
    for t in range(T_max):
        fp = fingerprint(s)
        failed = M.get(fp, set())
        if len(failed) >= tau:                       # flagged: repeated failure at this state
            a = substitute(policy, s, failed, K)
        else:
            a = policy.sample(s)
        s_next, r, done = env.step(a)
        if r < 0:                                    # failure observed
            M.setdefault(fp, set()).add(signature(a))
        traj.append((s, a, r))
        s = s_next
        if done: break
    return traj

def substitute(policy, s, failed, K):
    # Sample K candidates from π, rank by behavioral distance to failed set,
    # pick the highest-distance candidate whose signature is not in `failed`.
    cands = [policy.sample(s) for _ in range(K)]
    cands = [c for c in cands if signature(c) not in failed]
    if not cands:                                    # constrained fallback: negate effect
        return negate_failed_effect(s, failed)
    return max(cands, key=lambda c: min_dist(signature(c), failed))

def min_dist(sig, failed):
    # Behavioral distance: 1 if tool differs, else Jaccard distance over argument tokens
    return min((0.0 if sig[0] != f[0] else jaccard(sig[1], f[1])) for f in failed)
```

A state becomes *flagged* once `tau=3` distinct failed action signatures have accumulated at its fingerprint; from then on every re-entry routes through `substitute`, which draws `K=8` candidates, discards any whose signature is in the failed set, and selects the candidate maximizing minimum behavioral distance to the failed set. When all `K` candidates are excluded, the fallback negates the effect of the most recent failed action (e.g., for an `edit` that deleted a line, propose an `edit` that inserts at the same location; for a `run_test` that timed out, propose a narrower test invocation). The negate-effect fallback is scoped to two tools only: `edit` and `run_test`; for any other tool, the fallback returns the highest-distance candidate from the original `K` samples regardless of exclusion (and we log this as an empty-pool event).

**Why this design.** We chose a *hard exclusion* of failed signatures over a soft penalty on re-sampling because a penalty still admits the same failure with nonzero probability, and the whole point is a guarantee of divergence; the cost is that the candidate pool can empty, which is why the negate-effect fallback exists rather than a re-sample. We chose a *behavioral-distance ranking* over uniform sampling from the remaining pool because uniform sampling wastes the `K` budget on near-duplicate candidates that are technically distinct but replay the same failure mode; the cost is that distance is a heuristic over argument tokens and can over-rank superficially different actions. We chose a *per-state fingerprint with AST normalization* over raw text hashing because coding-agent states differ in irrelevant whitespace and line offsets, so raw hashing would fragment the failure memory and never reach the flagging threshold; the cost is that two genuinely different bugs sharing an AST skeleton collapse into one memory slot. We chose `tau=3` over a single-failure trigger because one failure is often noise (transient test flake), and we want the flag to indicate a *repeated* mode; the cost is that a genuinely deterministic single-failure bug is only caught on its second re-entry. A domain expert would not describe FSR as a variant of SWE-Trail: SWE-Trail only decides *where* to collect and leaves action selection untouched, whereas FSR intervenes on *what action is taken* and carries a per-state exclusion invariant that SWE-Trail has no representation for.

**Why it measures what we claim.** The state fingerprint `fingerprint(s)` measures *state re-entry* because it is an AST-normalized hash that is invariant to whitespace and line-offset noise but sensitive to the semantic diff, under the assumption that semantically equivalent coding states share an AST skeleton; this assumption fails when two distinct bugs share a skeleton (e.g., same function edited at different offsets), in which case the fingerprint reflects a coarser state than the agent actually occupies and the memory conflates their failures. The action signature `signature(a)` measures *behavioral equivalence of actions* because it pairs the tool with a normalized argument skeleton, under the assumption that two actions with the same tool and normalized arguments have the same effect on the environment; this assumption fails for actions whose effect depends on hidden environment state (e.g., an `edit` whose target line shifted), in which case the signature reflects surface syntax rather than effect. The flagging threshold `len(failed) >= tau` measures *repeated failure mode* because counting distinct failed signatures at a fingerprint counts distinct dead ends, under the assumption that distinct signatures correspond to distinct failure causes; this assumption fails when one cause produces several signatures (e.g., a flaky test yielding different stack traces), in which case the count over-triggers and FSR substitutes earlier than necessary. The exclusion `signature(c) not in failed` measures *guaranteed behavioral divergence* because set membership is exact over the normalized signature space, under the assumption that the signature space captures the failure-relevant behavior; this assumption fails when an action is behaviorally identical to a failed one but differs in a token the normalizer ignores, in which case the exclusion admits a replay. The distance ranking `max min_dist` measures *novelty relative to prior failures* because maximizing the minimum distance pushes the chosen action away from every recorded dead end, under the assumption that Jaccard distance over argument tokens is monotone in behavioral difference; this assumption fails for arguments where a single token flip changes the effect entirely (e.g., a comparison operator), in which case the ranking reflects token overlap rather than behavioral novelty. The primary metric *distinct dead-end re-entry per issue* measures *true repeated failure modes* because it counts re-entry events at flagged fingerprints, under the assumption A that signature-level distinctness corresponds to environment-effect distinctness (validated by the post-action state-diff agreement in the calibration study); this assumption fails under failure mode F where a token-level signature difference masks an identical effect (e.g., comparison-operator flips), in which case the metric counts a re-entry that is behaviorally identical but signature-distinct.

## Contribution

We introduce Failure-Substitution Replay, a collector operator that maintains a per-state failure memory keyed by AST-normalized state fingerprints and enforces a hard exclusion of previously failed action signatures at flagged states, with a negate-effect fallback when the candidate pool empties. We establish the design principle that history-guided collection must intervene on action selection, not only on state selection, because otherwise re-collection is not guaranteed to explore a distinct branch. We release the failure-memory and substitution module as a drop-in component compatible with any history-guided trajectory collector.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | SWE-bench-verified issues with reproducible failure states | Real repo states where re-entry failure modes occur |
| Primary metric | Rate of distinct dead-end re-entry per issue | Directly detects whether failed actions recur |
| Baseline | Independent multi-trajectory sampling | Tests whether selection alone suffices without action intervention |
| Baseline | SWE-Trail (history-guided collection) | Tests whether where-to-collect guidance alone fixes re-entry |
| Baseline | Soft-penalty re-sampling | Tests hard exclusion vs probabilistic discouragement |
| Baseline | Tabu search (explicit tabu list of action signatures per state) | Tests whether FSR is just tabu search applied to coding agents |
| Baseline | Novelty search (archive of action signatures, select max novelty) | Tests whether per-state exclusion is the right abstraction |
| Ablation-of-ours | FSR with uniform candidate selection (no distance ranking) | Isolates contribution of behavioral-distance ranking |
| Calibration study | 512 sampled states with post-action state-diff comparison | Validates signature-vs-effect equivalence assumption |
| Downstream evaluation | Fine-tune a 7B code model on collected trajectories, report pass@1 on held-out SWE-bench issues | Ties significance to learning signal, not just re-entry |

### Why this setup validates the claim

The central claim is that a per-state hard exclusion invariant on failed action signatures reduces repeated dead-end re-entry during collection, and that this reduction is caused by the exclusion-plus-distance mechanism rather than by more sampling or better issue selection. Independent multi-trajectory sampling tests the sub-claim that selection/curation alone cannot fix re-entry, because it has no memory and will re-draw the same failed action with the same probability on re-entry. SWE-Trail tests the sub-claim that deciding *where* to collect is insufficient, since it leaves action selection untouched and thus inherits the same re-entry distribution. Soft-penalty re-sampling tests the sub-claim that a guarantee of divergence matters over mere discouragement: with a penalty, the failed signature retains nonzero probability and re-entry recurs at a measurable rate. The tabu-search baseline tests whether FSR is merely a known technique applied to a new domain: tabu search also maintains a per-state tabu list, so if FSR matches tabu search on re-entry rate, the contribution is not the exclusion itself but the coding-agent-specific signature normalization and distance ranking. The novelty-search baseline tests whether per-state exclusion is the right abstraction versus a global novelty archive: if novelty search matches FSR, the per-state memory is unnecessary. The uniform-selection ablation isolates the distance-ranking sub-claim: if ranking contributes nothing, uniform selection from the non-failed pool should match FSR. The calibration study directly tests the load-bearing assumption that signature-level distinctness corresponds to environment-effect distinctness: on 512 sampled states, we compute the false-exclusion rate (actions with distinct signatures but identical post-action state diff) and false-admission rate (actions with identical signatures but distinct post-action state diff), reporting both. The downstream fine-tuning evaluation ties the re-entry reduction to learning signal: if FSR-collected trajectories yield higher pass@1 on held-out issues than baseline-collected trajectories, the reduction in re-entry translates to better training data. The primary metric — distinct dead-end re-entry per issue — is the right detector because it counts exactly the event the mechanism is designed to prevent, and it is computable from trajectories without confounding with final solve rate, so a gain concentrated in re-entry reduction (rather than uniform across all metrics) is the predicted signature.

### Expected outcome and causal chain

**vs. Independent multi-trajectory sampling** — On an issue where the agent first attempts an `edit` that deletes a needed line and the test fails, the baseline re-enters the same post-edit state on a later trajectory and, having no memory, re-samples the same deletion with substantial probability because its action distribution is unchanged at that state. Our method instead flags the fingerprint after three distinct failed signatures and routes through `substitute`, which excludes the deletion signature and picks a distant candidate, so we expect a noticeable drop in re-entry count on issues with repeated identical edits but parity on issues whose first attempt already succeeds.

**vs. SWE-Trail (history-guided collection)** — On a case where SWE-Trail correctly routes collection back to a promising earlier state, it still lets the policy sample the same failing `run_test` invocation at that state, because SWE-Trail only decides where to collect and has no representation of failed action signatures. Our method intervenes on what action is taken at that same state, excluding the timed-out invocation and proposing a narrower test via distance ranking, so we expect a gap concentrated on issues where re-entry states are reachable but their failures recur, and near-parity on issues where collection never revisits a state.

**vs. Soft-penalty re-sampling** — On a case where a single argument token (a comparison operator) caused the failure, the baseline down-weights that signature but still samples it with nonzero probability, so a fraction of re-entries replay the identical failure and the flag never fully suppresses it. Our method hard-excludes the signature and, when the candidate pool empties, negates the failed effect, so we expect a smaller residual re-entry rate for FSR than for the penalty baseline, with the gap widening on issues where the failed action is otherwise high-probability under the policy.

**vs. Tabu search** — Tabu search maintains a per-state tabu list of action signatures and forbids their selection, which is structurally similar to FSR's hard exclusion. However, tabu search typically uses raw action strings without AST-normalized state fingerprints or behavioral-distance ranking, so it may over- or under-trigger on coding-agent states. We expect FSR to outperform tabu search on re-entry rate specifically on issues where state fingerprints are stable (AST-normalized) and where distance ranking avoids near-duplicate candidates, and to match tabu search on issues with simple, stable states. If FSR matches tabu search everywhere, the contribution reduces to a known technique.

**vs. Novelty search** — Novelty search maintains a global archive of action signatures and selects actions maximizing novelty relative to the archive, without per-state memory. On coding-agent states, a global archive may select actions that are novel globally but still replay a local dead end (e.g., a different file but same failure mode). We expect FSR to achieve lower re-entry rate than novelty search on issues where the same dead end recurs across surface-distinct states, because FSR's per-state exclusion directly targets the local failure. If novelty search matches FSR, the per-state abstraction is unnecessary.

**vs. FSR with uniform candidate selection** — This ablation removes the distance-ranking step and selects uniformly from the non-failed candidate pool. If distance ranking contributes to avoiding near-duplicate failures, uniform selection will show higher re-entry rate on issues where the policy's candidate distribution is concentrated around the failed action. We expect a measurable gap in re-entry rate, with the gap widening on issues where the failed action is high-probability and the candidate pool contains many near-duplicates.

**Calibration study** — On 512 sampled states, we compute the false-exclusion rate (fraction of action pairs with distinct signatures but identical post-action state diff) and false-admission rate (fraction with identical signatures but distinct post-action state diff). We expect false-exclusion to be low (signature normalizer is conservative) and false-admission to be moderate (hidden environment state causes some identical signatures to have different effects). If false-admission is high, the exclusion invariant is weaker than claimed, and we will report this as a limitation.

**Downstream fine-tuning** — We fine-tune a 7B code model on trajectories collected by FSR and by each baseline, then evaluate pass@1 on held-out SWE-bench issues. We expect FSR-collected trajectories to yield higher pass@1 than baseline-collected trajectories, with the largest gains on issues where re-entry reduction was largest. If pass@1 is uniform across collectors, the re-entry reduction does not translate to learning signal.

### What would falsify this idea

If FSR's reduction in dead-end re-entry is uniform across all issue subsets rather than concentrated on issues whose trajectories actually re-enter flagged fingerprints, the exclusion invariant is not the cause and the central claim is wrong. Likewise, if the uniform-selection ablation matches FSR on re-entry rate, the distance-ranking sub-claim fails. If the calibration study shows high false-admission (signature-level distinctness does not correspond to environment-effect distinctness), the load-bearing assumption is false and the divergence guarantee is about a hash bucket, not behavior.

## References

1. SWE-Trail: History-Guided Trajectory Collection for Coding Agent Training
