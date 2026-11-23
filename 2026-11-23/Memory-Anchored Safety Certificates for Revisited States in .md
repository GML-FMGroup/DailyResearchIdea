# Memory-Anchored Safety Certificates for Revisited States in Video World Models

## Motivation

Video world models built on long-term spatial memory (Video World Models with Long-term Spatial Memory, NeurIPS 2025) retrieve stored 3D geometry to keep scenes consistent across revisits, but their safety decisions are made independently: NoPeWorld reads penetration risk from frozen DiT hidden states that are conditioned only on the current observation window, not on the retrieved memory state. Because the safety head and the memory store are never coupled, an action certified safe at time t can be re-evaluated against a different retrieved memory at a revisit and flip to unsafe (or vice versa) with no record of why — the field therefore has no notion of safety that is stable under its own memory mechanism, and this recurs across the memory branch (persistent embodied world models) and the interactive-generation branch (VRAG global state conditioning).

## Key Insight

If the safety head's input is the retrieved memory token set itself rather than the transient DiT hidden state, then safety becomes a function of a persistent, explicitly-updated state, which makes a revisitable safety certificate — an invariant that survives revisits until the memory is edited — a well-defined object rather than a hope.

## Method

# MemCert: Memory-Anchored Safety Certificates
# Frozen: DiT video world model V, spatial-memory store M (from long-term spatial memory framework)
# Trained: safety head h_theta (2-layer MLP, hidden=256, GeLU), memory encoder E_phi (frozen pretrained 3D-map encoder from the persistent-map branch, output dim 256), certificate ledger L

INPUT: observation window o_t, candidate action a_t, memory store M

# --- Assumption A (stated explicitly) ---
# The ledger's replay is sound only if the memory encoder E_phi is injective on safety-relevant geometry:
# two retrieved token sets with identical version vectors must induce identical penetration-relevant geometry.
# This is an assumption, not a guarantee; we measure its violation via the encoder-collision diagnostic (see experiment).

1. RETRIEVE  R_t = M.retrieve(o_t)            # set of K=64 memory tokens; R_t.version_vector = hash of token ids + positions
2. ENCODE    z_t = E_phi(R_t, o_t)            # 256-d memory-conditioned state vector
3. CERTIFY   s_t, c_t = h_theta(z_t, a_t)     # s_t in {safe, unsafe}, c_t = logit margin |logit_safe - logit_unsafe|
4. LEDGER    key = hash(R_t.version_vector, a_t)
             if key in L and L[key].margin >= tau_margin (tau_margin = 1.0):
                 s_t = L[key].decision          # replay stored certificate only if calibrated margin
             else:
                 L[key] = (s_t, c_t, R_t.version_vector)   # store margin; low-margin entries are never replayed
5. INVARIANT CHECK (at every revisit, before acting):
             for each (key, decision, margin) in L whose R.version_vector still matches AND margin >= tau_margin:
                 s_fresh, c_fresh = h_theta(E_phi(R_t, o_t), a_key)
                 if s_fresh != decision:
                     flag INVARIANT_VIOLATION -> block action, log (key, decision, s_fresh, c_fresh)
6. MEMORY UPDATE: any write to M bumps version_vector and invalidates
             only ledger entries whose retrieved token set changed.
7. ACT: if s_t == safe and no violation: execute a_t
        else: replace a_t with argmax_{a' in A_safe} intent_sim(a', a_t)  # as in NoPeWorld
OUTPUT: executed action, certificate ledger entry (with stored margin)

**Why this design.** We condition the safety head on retrieved memory tokens rather than frozen DiT hidden states (as NoPeWorld does) because the memory token set is the only quantity that is explicitly persisted and versioned, accepting the cost that the head must be retrained per memory-encoder and loses the zero-shot transfer NoPeWorld reports from raw DiT features. We add a certificate ledger keyed by (memory version, action) instead of recomputing safety from scratch at each revisit, because recomputation is exactly what makes revisit safety unstable, accepting the cost of O(|L|) replay at revisits and a bounded ledger. We invalidate ledger entries only when the retrieved token set changes rather than on any memory write, because most writes leave the local retrieval neighborhood untouched and full invalidation would discard valid certificates, accepting the risk that a subtle global memory edit escapes invalidation — which the invariant check catches by re-running the head. A domain expert would not describe this as NoPeWorld plus a memory store: NoPeWorld's safety signal is a function of transient hidden states and has no persistence semantics, whereas here the certificate is an object in the memory's version space and the invariant is enforced by replay against that version space. The ledger's soundness rests on Assumption A (encoder injectivity on safety-relevant geometry), which we state explicitly and measure via the collision-rate diagnostic; low-margin certificates (c_t < tau_margin = 1.0) are never replayed, so the invariant check tests geometry-scope agreement rather than head calibration noise.

**Why it measures what we claim.** The certificate ledger's version-keyed entries measure *memory-consistent safety* because we assume that two retrieved token sets with identical version vectors induce identical safety-relevant geometry (Assumption A), making the stored decision equivalent to a fresh evaluation; this assumption fails when the memory encoder is lossy and two different underlying geometries collapse to the same tokens, in which case the ledger reflects encoder-equivalence rather than world-equivalence — we quantify this via the encoder-collision rate diagnostic. The invariant-violation flag measures *safety stability under revisits* because we assume that a head disagreement across revisits with unchanged memory indicates a genuine instability rather than head noise, an assumption that fails when the head is under-trained and its logit margin is small, in which case the flag reflects calibration noise — we mitigate this by requiring c_t >= tau_margin = 1.0 for replay and report the logit-margin distribution on revisit pairs. The memory-conditioned state z_t measures *safety relevant to the retrieved scene* because we assume retrieved tokens carry the obstacle geometry needed for penetration judgment, which fails when obstacles lie outside the retrieved K=64 tokens, in which case z_t reflects only the visible neighborhood — we report obstacle-in-retrieval recall as a function of K.

## Contribution

(1) A memory-anchored safety certification framework (MemCert) that makes safety a function of versioned retrieved memory tokens and enforces a replay-based memory-safety consistency invariant at revisits. (2) The empirical finding, via the invariant-violation rate, of how often frozen-DiT safety heads like NoPeWorld's contradict themselves across revisits under a long-term spatial memory, isolating memory-state drift as a distinct failure source from penetration itself. (3) A version-keyed certificate ledger with selective invalidation that bounds replay cost while preserving the invariant.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | NoPeBench revisit split (scripted camera-return trajectories; 200 episodes, K=64) | Tests penetration safety under scene revisits |
| Primary metric | Violation rate on revisit episodes | Directly measures memory-consistent safety |
| Secondary metric | Decision-flip rate on repeated-revisit pairs (same memory version, perturbed observation noise σ=0.01) | Measures stability under replay vs. recompute |
| Diagnostic (a) | Encoder-collision rate: fraction of distinct 3D geometries (from persistent-map branch) mapping to identical version vectors | Directly tests Assumption A |
| Diagnostic (b) | Head logit-margin distribution on revisit pairs (histogram, 10 bins) | Separates calibration noise from genuine instability |
| Diagnostic (c) | Obstacle-in-retrieval recall vs. K ∈ {16, 32, 64, 128} | Tests whether retrieved tokens carry obstacle geometry |
| Baseline | NoPeWorld (DiT-hidden-state head) | Tests whether memory conditioning beats transient features |
| Baseline | Fresh recompute per revisit | Tests whether ledger replay stabilizes revisits |
| Baseline | Full-invalidation ledger | Tests whether targeted invalidation preserves valid certificates |
| Ablation-of-ours | MemCert without invariant check | Isolates the replay-verification contribution |
| Ablation-of-ours | MemCert with tau_margin = 0 (replay all margins) | Isolates the calibrated-margin contribution |

### Why this setup validates the claim

The central claim is that anchoring safety certificates in a versioned memory space yields stable, memory-consistent penetration avoidance across revisits, which transient-hidden-state safety (NoPeWorld) and per-revisit recomputation cannot deliver. NoPeWorld tests the sub-claim that memory conditioning, not raw DiT features, is what makes revisit decisions consistent: if MemCert only matches it, the memory-anchoring story is empty. The fresh-recompute baseline tests the sub-claim that ledger replay, not merely better features, is what removes revisit instability; if recomputation already matches MemCert, the ledger is doing nothing. The full-invalidation baseline tests the sub-claim that targeted invalidation preserves valid certificates rather than discarding them; if it matches MemCert, our invalidation rule is unmotivated. The no-invariant-check ablation isolates whether the replay check catches the global-edit escapes we predict. The tau_margin=0 ablation isolates whether the calibrated-margin threshold is what prevents head-noise-driven flips. The three diagnostics directly measure the linking assumptions: collision rate tests Assumption A (encoder injectivity), margin distribution tests whether the invariant check reflects geometry rather than calibration noise, and obstacle-in-retrieval recall tests whether z_t carries the obstacle geometry. Violation rate on revisit episodes is the right primary metric because it is exactly the quantity the mechanism claims to improve; aggregate violation rate would dilute the effect with non-revisit frames where all methods should be similar.

### Expected outcome and causal chain

**vs. NoPeWorld (DiT-hidden-state head)** — On a case where the agent leaves a room, moves, and returns to an obstacle whose geometry is only in the persisted memory, NoPeWorld's head reads transient hidden states that no longer encode the obstacle, so it certifies a penetrating action as safe. Our method instead conditions on retrieved memory tokens that still carry that obstacle geometry, so it flags the action unsafe. We expect a noticeable violation-rate gap concentrated on revisit episodes, with near-parity on first-visit frames where transient features suffice.

**vs. Fresh recompute per revisit** — On a case where the same obstacle is revisited under slightly perturbed observation noise (σ=0.01), recomputation re-runs the head each time and small logit-margin jitter flips the decision between safe and unsafe across visits. Our method replays the stored certificate for the unchanged memory version when margin >= tau_margin = 1.0, so the decision is stable by construction. We expect our method to show lower decision-flip rates and lower violation rate specifically on repeated-revisit episodes, with parity on single-visit episodes.

**vs. Full-invalidation ledger** — On a case where a distant memory write changes the global store but leaves the local retrieval neighborhood intact, full invalidation discards valid certificates and forces recomputation, reintroducing the instability above. Our method bumps the version vector but invalidates only entries whose retrieved token set changed, keeping the valid certificate. We expect our method to retain a meaningful fraction of replayed certificates and show lower violation rate on episodes with interleaved distant writes.

**Diagnostic expectations** — We expect encoder-collision rate to be low (< 5%) on the NoPeBench revisit split, confirming Assumption A holds approximately; if collision rate is high, the ledger's stability claim is weakened and we report this as a precondition failure. We expect the logit-margin distribution to be bimodal with most revisit-pair margins > 1.0, justifying tau_margin = 1.0; if margins cluster near zero, the invariant check is dominated by calibration noise. We expect obstacle-in-retrieval recall to exceed 90% at K=64 and degrade at K=16, confirming K=64 is sufficient for the benchmark.

### What would falsify this idea

If MemCert's gain over NoPeWorld is uniform across revisit and first-visit subsets rather than concentrated on revisits, or if the no-invariant-check ablation matches the full method on global-edit episodes, or if the encoder-collision rate exceeds 20% (violating Assumption A), then the memory-anchoring and invariant-check mechanisms are not producing the claimed effect and the central claim is wrong.

## References

1. Non-penetrating World Models
