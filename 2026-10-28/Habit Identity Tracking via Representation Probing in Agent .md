# Habit Identity Tracking via Representation Probing in Agent Memory

## Motivation

HABIT-Bench evaluates memory as a black-box pipeline and finds that even the best method reaches only 33.61% macro accuracy, leaving the representational stability of habit identity under evidence accumulation entirely unestablished. This mirrors a broader gap in memory-agent benchmarks such as MemoryAgentBench and LongMemEval, which measure end-task accuracy or retrieval competence but never inspect whether the memory's internal state preserves a durable behavioral pattern across sessions. The root cause is structural: without probing the memory representation itself, one cannot distinguish a system that fails to infer a habit from one that infers it but lets the representation drift or collapse after exceptions.

## Key Insight

Habit identity is preserved in a memory system if and only if the internal representation of a habit, when probed at each session, forms a trajectory whose within-habit displacement is bounded while between-habit displacement grows — a geometric invariant that end-task accuracy cannot reveal because a system can answer correctly from a single recent session even when its habit representation has been overwritten.

## Method

**What it is.** HABIT-TRACKER is a representational probing layer for HABIT-Bench that, at each session t, extracts a habit-state vector h_t from the memory system's internal state via a per-system structured probe, and computes three trajectory statistics — stability S, drift D, and post-exception recovery R — that quantify whether the habit's identity is preserved as evidence accumulates. Inputs: the memory system's per-session internal state (MemCube embeddings for MemOS, retrieved-context embeddings for RAG baselines) and the HABIT-Bench habit graph labels. Output: a per-habit identity-trajectory curve and scalar S/D/R scores aligned with the end-task accuracy of the same system.

**Load-bearing assumption (stated explicitly).** The method assumes that habit-label information is linearly decodable from each memory system's internal state z_t in a session-stable coordinate frame, so that a probe fit on that system's own TRAIN-split internal states yields a habit-state vector h_t whose cosine/L2 trajectory statistics are comparable across sessions and across heterogeneous memory systems. This assumption is not taken for granted: it is verified per system by a decodability gate (Phase 1b) and enforced by per-system probe refitting and session-frame normalization (Phase 2b). If a system fails the gate, IPS is not reported for it.

**How it works.**

```pseudocode
# Phase 1a: Anchor collection (once per domain, before any evaluation)
for each domain d in {food, finance, software, travel}:
    collect anchor set A_d = {(habit_label k, session_text s)} from TRAIN split only
    # |A_d| = all TRAIN sessions for domain d (HABIT-Bench TRAIN split)

# Phase 1b: Per-system probe construction and decodability gate
for each memory system sys in {MemOS, RAG, best-baseline, ...}:
    for each domain d:
        run TRAIN-split sessions of domain d through sys's memory pipeline
        collect Z_d^sys = {z_t^sys = sys.internal_state_after_session(t)} with habit labels
        fit linear probe P_d^sys: z -> one-hot(k)   # logistic regression, L2=1e-3
        # P_d^sys is frozen after fitting; it is the fixed measurement instrument for sys
        evaluate held-out habit-label accuracy acc(P_d^sys) on a 20% held-out slice of Z_d^sys
        if acc(P_d^sys) <= chance + 0.10:   # chance = 1/K for K habit labels
            mark sys as FAILING the decodability gate for domain d
            # IPS is not computed or reported for (sys, d)

# Phase 2a: Per-session habit-state extraction (at eval time)
for each session t = 1..T of a user history:
    z_t = memory_system.internal_state_after_session(t)   # e.g. MemCube emb.
    h_t = P_d^sys.project(z_t)                             # habit-state vector in R^K
    record (t, h_t, exception_flag_t, change_flag_t)       # flags from habit graph

# Phase 2b: Session-frame normalization (per system, per domain)
# Fit a whitening transform W_d^sys on TRAIN-split Z_d^sys (mean 0, covariance I)
# and apply it to every z_t before projection, so h_t is computed in a session-stable frame.
# This addresses global rotation / rescaling between sessions.
for each session t:
    z_t_norm = W_d^sys @ (z_t - mean(Z_d^sys))
    h_t = P_d^sys.project(z_t_norm)

# Phase 3: Trajectory statistics
S = mean over t of cos(h_t, h_{t-1})                      # stability
D = mean over t of ||h_t - h_{t-1}||_2                    # drift
for each exception event at session t*:
    R(t*) = cos(h_{t*+3}, h_{t*-1})                       # recovery after 3-session lag
    # 3-session lag chosen to match HABIT-Bench exception window

# Phase 4: Identity-preservation score
IPS = S - lambda * D + mu * mean(R)                       # lambda=0.5, mu=0.3 (default)
# Report IPS alongside macro accuracy; correlate across 8 memory methods x 2 LLMs
# Sensitivity: refit lambda, mu on held-out memory methods via rank-correlation maximization;
# report IPS-accuracy rank correlation under default and refit weights, plus +/-0.1 perturbations.
```

**Why this design.** We chose a frozen linear probe over a learned disentanglement head because the probe must be a fixed measurement instrument whose output is comparable across memory systems; a learned head would co-adapt to each system and destroy cross-system comparability, accepting the cost that the probe may under-read nonlinear habit encodings. We chose cosine similarity and L2 displacement as the trajectory statistics over KL-divergence between successive habit-state distributions because cosine/L2 are defined on the projected vector itself and do not require assuming a generative model of the memory state, accepting the cost that they are insensitive to magnitude-only drift. We chose a fixed 3-session recovery lag over a learned lag because the HABIT-Bench exception window is annotated in the habit graph, so the lag is an experimental control rather than a free parameter, accepting the cost that recovery on slower-drifting habits may be mis-timed. Unlike MemoryAgentBench, which evaluates retrieval competence through end-task questions, HABIT-TRACKER does not ask the memory system any question at all — it reads the representation directly, so a domain expert would not describe it as a variant of retrieval evaluation. The per-system probe refitting and whitening normalization are not new mechanisms; they are the concrete calibration steps that make the frozen-probe measurement instrument valid for each system's own embedding space, as required by the load-bearing assumption above.

**Why this measures what we claim.** The projected habit-state h_t measures habit identity preservation because the per-system frozen probe P_d^sys is a sufficient statistic for habit label under the assumption that habit-relevant information is linearly decodable from the memory embedding in a session-stable frame (verified by the decodability gate and whitening normalization); this assumption fails when the memory system encodes habit information in a nonlinear manifold that the linear probe cannot read, in which case h_t reflects only the linearly accessible component and IPS under-reports true identity preservation. The stability S measures within-habit representational consistency because cosine similarity between successive projections is invariant to the habit's absolute embedding location under the assumption that the probe's decision boundary is stable across sessions and the whitening transform removes global rotation; this assumption fails when the memory system's embedding space undergoes global rotation between sessions that whitening cannot remove, in which case S reflects coordinate drift rather than habit drift. The recovery R measures post-exception identity restoration because the 3-session lag matches the annotated exception window under the assumption that exceptions are resolved within the habit graph's declared duration; this assumption fails when a memory system handles an exception by permanently rewriting the habit representation, in which case R correctly reflects non-recovery rather than measurement error.

## Contribution

['HABIT-TRACKER: a representational probing layer for HABIT-Bench that extracts per-session habit-state vectors from agent memory via a frozen linear probe and computes stability, drift, and post-exception recovery statistics, producing a habit identity-preservation score (IPS) that is comparable across memory architectures.', 'A design principle: habit identity preservation, measured as the geometric invariant S - lambda*D + mu*R on probed habit-state trajectories, is a necessary condition for reliable habit induction that end-task accuracy cannot certify, because a system can answer correctly from a single recent session while its habit representation has been overwritten.', 'An analysis tool that aligns IPS curves with HABIT-Bench macro accuracy across eight memory methods and two LLMs, exposing which memory architectures preserve habit identity under evidence accumulation versus which merely retrieve recent context.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | HABIT-Bench (four domains, exception/change cases) | Provides annotated habit graphs and per-session histories |
| Primary metric | IPS–macro-accuracy rank correlation across methods | Tests whether trajectory stats track end-task identity preservation |
| Baseline | Best HABIT-Bench method (33.61% macro) | Strongest end-task competitor for correlation anchor |
| Baseline | RAG retrieval-context baseline | Probes whether linear readout works on retrieved-context embeddings |
| Baseline | MemOS / MemCube system | Probes whether probe reads structured memory state faithfully |
| Ablation-of-ours | Shuffled-label probe (P_d^sys fit on permuted habit labels) | Confirms signal comes from habit structure, not probe capacity |
| Ablation-of-ours | Nonlinear (2-layer MLP, hidden=256, GeLU) probe vs. linear probe | Tests whether linear probe under-reads nonlinear habit encodings |
| Ablation-of-ours | Learned disentanglement head (co-adapted per system) | Tests whether frozen probe is necessary for cross-system comparability |
| Diagnostic | Per-system probe decoding accuracy on held-out TRAIN | Flags when the load-bearing assumption breaks (gate: > chance + 0.10) |
| Sensitivity | IPS weights lambda, mu refit on held-out methods; +/-0.1 perturbation | Shows IPS-accuracy correlation is not an artifact of chosen constants |

### Why this setup validates the claim

The central claim is that a per-system frozen linear probe read off a memory system's internal state yields trajectory statistics (S, D, R, and their combination IPS) that co-vary with the system's end-task habit-identity accuracy. To falsify or confirm this, we need (i) a benchmark where habit identity is the ground-truth construct — HABIT-Bench provides annotated habit graphs and macro accuracy as the end-task anchor; (ii) a spread of memory mechanisms whose internal states differ qualitatively — MemOS (structured MemCubes), RAG (retrieved context), and the strongest reported baseline give three distinct representational regimes, so a single lucky pairing cannot drive the correlation; (iii) a metric that is a cross-method rank correlation rather than a per-method score, because the claim is comparative (IPS orders systems the same way accuracy does), not absolute; (iv) a shuffled-label probe ablation, which holds probe capacity fixed while destroying habit-relevant structure — if IPS–accuracy correlation survives shuffling, the probe is reading something other than habit identity and the claim collapses; (v) a per-system decodability gate that explicitly tests the load-bearing assumption — if a system's probe fails to decode habit labels above chance + 0.10 on held-out TRAIN, IPS is not reported for that system, so the correlation is computed only over systems where the measurement instrument is valid; (vi) a linear-vs-MLP-vs-learned-head probe comparison on held-out TRAIN, reporting decoding accuracy and IPS divergence, to show the linear probe is a principled measurement instrument rather than a convenience; and (vii) a weight-sensitivity analysis that refits lambda and mu on held-out methods and perturbs them by +/-0.1, so the IPS-accuracy correlation is not an artifact of hand-tuned constants. Together these form a falsifiable test: either IPS ranks the eight memory methods consistently with macro accuracy, or it does not.

### Expected outcome and causal chain

**vs. Best HABIT-Bench method (33.61% macro)** — On a case where a user's habit is contradicted once (an exception) and then re-asserted, the strongest end-task baseline recovers the habit on most questions but fails on change cases because its retrieval is keyed to recent surface text rather than a persistent habit state. Our method reads the internal state directly and, on that same instance, should show high S and a clear R rebound after the 3-session exception window, because the probe reads the residual habit signal even where retrieval ranking misleads the end-task reader. We expect IPS to rank this method near the top and to correlate with its accuracy advantage concentrated on the exception/change subsets, with parity on unaccepted-suggestion cases. The per-system probe for this baseline must pass the decodability gate (held-out accuracy > chance + 0.10) for its IPS to enter the correlation.

**vs. RAG retrieval-context baseline** — On a case where the habit is weakly evidenced across many sessions and a single salient distractor appears late, the RAG baseline's retrieved-context embedding is dominated by the distractor, so its projected habit-state h_t flips toward the wrong habit and its end-task answers degrade on that subset. Our method handles the same instance because the per-system frozen probe projects the retrieved-context embedding through a habit-label decision boundary fit on that system's own TRAIN-split retrieved-context states, exposing the flip as a large D and a low post-exception R rather than hiding it. We expect a pronounced IPS–accuracy gap on weak-evidence-plus-distractor cases, since the RAG baseline's D should spike exactly where its accuracy drops.

**vs. MemOS / MemCube system** — On a case where an exception is handled by permanently rewriting the MemCube, the MemOS system's end-task accuracy on later questions about the original habit drops, but its internal state no longer contains the original habit signal, so a naive probe would show clean stability. Our method instead shows low R at the exception session and a sustained D afterward, because the per-system frozen probe's decision boundary is fixed to the original habit label and cannot co-adapt. We expect IPS to rank MemOS below the best baseline on exception-heavy subsets while remaining high on stable-habit subsets, matching its accuracy profile.

**vs. Shuffled-label probe ablation** — With P_d^sys fit on permuted habit labels, the probe's held-out decoding accuracy should fall to chance, and IPS scores should lose their rank correlation with macro accuracy. If the shuffled-label probe still correlates with accuracy as strongly as the true probe, the probe is reading something other than habit identity and the claim collapses.

**vs. Nonlinear (MLP) probe ablation** — On systems where habit information is encoded nonlinearly, the MLP probe should achieve higher held-out decoding accuracy than the linear probe, and IPS computed from the MLP probe may diverge from linear-probe IPS. We expect the linear probe to remain competitive on MemOS and RAG (where the decodability gate passes) and to under-read only on systems flagged by the gate, confirming that the linear probe is a valid measurement instrument where the load-bearing assumption holds.

**vs. Learned disentanglement head ablation** — A head co-adapted per system should show higher in-system decoding accuracy but destroy cross-system comparability, so its IPS scores should not rank-correlate with macro accuracy across systems as strongly as the frozen per-system probe. This confirms the design choice of a frozen probe over a learned head.

### What would falsify this idea

If IPS–macro-accuracy rank correlation across the eight memory methods (restricted to systems passing the decodability gate) is near zero, or if the shuffled-label probe ablation produces IPS scores that correlate with accuracy just as strongly as the true probe, the claim that trajectory statistics measure habit identity preservation is wrong. A second falsifier: if the predicted subset concentration (gains on exception/change cases, parity on stable cases) does not appear and instead IPS tracks accuracy uniformly, the causal story about what the probe reads is unsupported.

## References

1. HABIT-Bench: Evaluating Habit Induction from Longitudinal Weak Evidence in Agent Memory
2. Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions
3. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory
