# Reachability-Constrained Monotone Commitment for Speculative Video World Models

## Motivation

Speculative streaming world models such as ShowNow prepare multiple action-conditioned futures during playback but commit only a single latent per decision step, with no invariant tying the committed latent to the trajectory the model will actually realize once the next observation arrives. This structural gap lets prepared speculative futures silently diverge from the realized future: ShowNow reports latency and restart-empty-tick gains but never enforces that a committed prefix remains reachable from subsequent observations, so a fast response can be a physically or dynamically inconsistent one. The same absence of a commitment invariant recurs across the video-world-model tree, where long-horizon consistency is handled by memory mechanisms (Video World Models with Long-term Spatial Memory, Learning 3D Persistent Embodied World Models) rather than by constraining what gets committed during streaming.

## Key Insight

A commitment rule that only advances the maximal prefix of speculative futures lying inside a learned reachability tube around the next observation makes the committed trajectory a closure of the model's own transition kernel, so speculative divergence becomes a detectable violation at commit time rather than a silent error that compounds downstream.

## Method

**What it is.** RCMC (Reachability-Constrained Monotone Commitment) is a drop-in replacement for ShowNow's single-latent commitment operator. Inputs: the current committed latent state $s_t$, the set of $K\in[1,8]$ prepared action-conditioned futures $\{f^{(k)}_{t+1:t+H}\}_{k=1}^{K}$ produced by ShowNow's shared-history tree execution, and the next observation $o_{t+1}$ that arrives after the user's action. Output: a committed latent prefix $s_{t+1:t+m}$ of length $m\ge 1$ and a re-prepared speculative set conditioned on that prefix.

**How it works.**

```pseudocode
# RCMC commitment operator, called at every decision step t
# reachability tube: learned encoder phi, adaptive radius rho_t
# K prepared futures from ShowNow; H horizon per future

input: s_t, {f^(k)_{t+1:t+H}} for k=1..K, observation o_{t+1}

1.  z_obs <- phi(o_{t+1})                      # observation embedding (phi: 2-layer MLP, hidden=256, GeLU, output dim=128)
2.  for k in 1..K:
3.      d_k <- max_{j=1..H} || phi(f^(k)_{t+j}) - z_obs ||_2   # tube distance per future
4.  rho_t <- quantile_{0.95}({d_k}_{k=1..K} over most recent window of W=50 realized transitions)  # adaptive radius
5.  R <- { k : d_k <= rho_t }                     # reachable future set
6.  if R is empty:
7.      m <- 1; s_{t+1} <- decode(phi^{-1}(z_obs))   # fallback: commit observation-anchored latent
8.      record invariant_violation <- True
9.  else:
10.     m <- max over k in R of (longest common prefix length of f^(k) with any other f^(k') in R)
11.     s_{t+1:t+m} <- that maximal common reachable prefix
12. re-prepare: condition ShowNow's tree on s_{t+1:t+m}, regenerate K futures for t+m+1..t+m+H
13. return s_{t+1:t+m}, re-prepared futures, invariant_violation
```

Hyperparameters: $K=8$ (matching ShowNow's reported configuration), $H$ inherited from ShowNow, tube radius $\rho_t$ set adaptively as the 95th percentile of $d_k$ over the most recent $W=50$ realized transitions (initialized from a held-out calibration set of 512 matched multi-future pairs). The encoder $\phi$ is a 2-layer MLP (hidden=256, GeLU, output dim=128) trained contrastively on (prepared future, realized continuation) pairs: InfoNCE loss with temperature $\tau=0.07$, positive pair = prepared future that actually matched the realized continuation (within a tolerance of 0.1 in latent L2), negative pairs = other prepared futures from the same state. Training uses 10k such pairs from ViZDoom, batch size 256, AdamW lr=1e-4, 20 epochs.

**Why this design.** We chose a monotone maximal-prefix commitment over ShowNow's single-latent rule because a single latent gives no way to distinguish a speculative future that is consistent with the observation from one that is not, accepting the cost that commitment length $m$ can shrink to 1 under distribution shift and throughput drops toward the native-block baseline. We chose a learned reachability tube $\phi,\rho_t$ over a hand-specified physical simulator because ViZDoom-style environments do not expose exact dynamics, accepting the cost that the tube is only as calibrated as its held-out set and online window. We chose to re-prepare the remaining speculative futures conditioned on the committed prefix over reusing stale futures because stale futures were generated under a different conditioning state and would reintroduce the divergence the operator is meant to remove, accepting the extra tree-execution cost that ShowNow's shared-history execution is designed to absorb. A domain expert would not describe RCMC as a variant of ShowNow: ShowNow's contribution is the speculative preparation and fused commitment machinery, whereas RCMC changes the commitment rule itself from a point decision to a closure-constrained prefix decision, and it would still be meaningful applied to any speculative video world model.

**Why this measures what we claim.** The tube distance $d_k$ measures dynamic reachability of a prepared future from the realized observation because the assumption is that $\phi$ embeds latent states such that Euclidean distance in embedding space is monotone in transition-kernel likelihood under the model's own dynamics; this assumption fails when $\phi$ collapses distinct dynamics into nearby embeddings, in which case $d_k$ reflects visual similarity rather than reachability. We make this assumption explicit and test it via rank correlation between $d_k$ and the realized continuation's log-likelihood under the model's own dynamics on held-out pairs. The committed prefix length $m$ measures how much of the speculative tree survives the closure invariant because the assumption is that a common prefix across reachable futures is exactly the part of the trajectory on which all reachable continuations agree; this assumption fails when two reachable futures agree on a prefix but diverge immediately after, in which case $m$ overstates committed consistency. The re-preparation step measures conditioning fidelity of the remaining speculative set because the assumption is that regenerating from the committed prefix restores the same conditional distribution as the original preparation; this assumption fails when the re-prepared futures are systematically shorter-horizon than the originals, in which case downstream latency gains are overstated.

## Contribution

(1) A monotone reachability-constrained commitment operator that replaces point commitment with maximal-prefix commitment under a learned reachability tube, with a fallback that guarantees at least one committed latent per step. (2) The closure property as a design principle for speculative video world models: committed latents are constrained to be reachable from the next observation, turning speculative divergence into a detectable commit-time violation. (3) A calibration procedure for the reachability tube on matched multi-future data, and a re-preparation protocol that conditions ShowNow's tree on the committed prefix.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | ViZDoom matched multi-future pairs | Matches ShowNow evaluation; enables reachability calibration |
| Primary metric | Median input-to-display latency | Directly measures responsiveness ShowNow optimizes |
| Secondary metric | Restart empty tick rate | Measures consistency of committed futures |
| Secondary metric | Revisit consistency (PSNR on revisited frames) | Measures long-horizon spatial consistency |
| Baseline | Native-block SolarWM-FT | Tests whether commitment rule alone improves latency |
| Baseline | ShowNow single-latent commitment | Tests whether closure-constrained prefix beats point commitment |
| Baseline | ShowNow with stale-future reuse | Tests whether re-preparation is necessary |
| Ablation-of-ours | RCMC with fixed m=1 commitment | Isolates prefix-length effect from reachability filtering |
| Ablation-of-ours | RCMC with fixed global rho (no adaptive) | Isolates adaptive radius effect |
| Diagnostic | Rank correlation between d_k and realized log-likelihood | Tests monotonicity assumption of phi |

### Why this setup validates the claim

The central claim is that replacing ShowNow's single-latent commitment with a reachability-constrained monotone prefix commitment reduces input-to-display latency without sacrificing consistency, because the operator commits only to futures that are dynamically reachable from the realized observation. The native-block SolarWM-FT baseline tests the sub-claim that any speculative commitment helps at all; ShowNow's single-latent commitment tests the sub-claim that closure-constrained prefix commitment beats point commitment; stale-future reuse tests the sub-claim that re-preparation is necessary to avoid reintroducing divergence. The RCMC fixed-m=1 ablation isolates whether gains come from the reachability filter or merely from committing longer prefixes. The fixed-global-rho ablation tests whether the adaptive radius is necessary for the tube to remain sound and complete. The rank-correlation diagnostic directly tests the load-bearing assumption that $\phi$ is monotone in transition-kernel likelihood: if the Spearman correlation between $d_k$ and the realized continuation's log-likelihood is near zero, the tube distance reflects visual similarity rather than reachability, and any latency gain cannot be attributed to the closure invariant. Median input-to-display latency is the right primary metric because it directly captures the responsiveness that speculative streaming is meant to improve, and it is sensitive to both commitment length and re-preparation overhead, so a false gain would show up as latency parity with worse consistency.

### Expected outcome and causal chain

**vs. Native-block SolarWM-FT** — On a case where the user issues a rapid action change mid-block, the baseline produces a full native block of stale frames because it has no prepared futures to switch to. Our method instead commits a reachable prefix and re-prepares conditioned on it because the reachability tube filters inconsistent futures, so we expect a large latency gap on action-change-heavy segments but parity on steady segments.

**vs. ShowNow single-latent commitment** — On a case where the prepared future diverges from the realized observation due to compounding error, the baseline commits a single latent that is already inconsistent because it has no mechanism to detect divergence. Our method instead shrinks commitment to the maximal reachable prefix and re-prepares because the tube distance flags the inconsistent future, so we expect fewer restart empty ticks and a modest latency improvement concentrated on long-horizon segments.

**vs. ShowNow with stale-future reuse** — On a case where the committed prefix differs from the original conditioning state, the baseline reuses stale futures that were generated under a different state because it skips re-preparation. Our method instead regenerates futures from the committed prefix because conditioning fidelity requires it, so we expect better consistency on revisits and a latency cost on re-preparation that is absorbed by shared-history execution.

**vs. RCMC with fixed m=1 commitment** — On a case where multiple reachable futures share a common prefix longer than 1, the fixed-m=1 ablation commits only the first latent and discards the rest, because it ignores prefix length. Our method commits the maximal common reachable prefix, so we expect lower latency on segments with high agreement among reachable futures, with no consistency loss.

**vs. RCMC with fixed global rho** — On a case where the distribution of $d_k$ shifts due to environment changes, the fixed-global-rho ablation either rejects all futures (R empty, m=1) or admits divergent ones, because the radius is not adapted. Our method uses an online quantile over recent transitions, so we expect fewer invariant violations and more stable commitment lengths, leading to better latency-consistency trade-off.

### What would falsify this idea

If RCMC's latency gain over ShowNow single-latent commitment is uniform across action-change-heavy and steady segments rather than concentrated where reachability filtering should trigger, the central claim that closure-constrained commitment drives the improvement is wrong. If the fixed-m=1 ablation matches RCMC's latency, the prefix-length mechanism is not the cause. If the rank correlation between $d_k$ and realized log-likelihood is near zero, the tube distance does not measure reachability, and the closure invariant is unsupported.

## References

1. ShowNow: Fine-Grained Speculative Streaming for Responsive World Models
