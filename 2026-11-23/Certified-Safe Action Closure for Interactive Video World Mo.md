# Certified-Safe Action Closure for Interactive Video World Models

## Motivation

Interactive video world models such as NoPeWorld (E1, E6) reduce obstacle penetration from 74% to 16% by swapping unsafe actions for safe ones, but the replacement is a heuristic that carries no invariant: the substituted action is not guaranteed to remain admissible under the frozen DiT's own safety head, and repeated substitution can walk the action off the admissible manifold. The same closure gap recurs across the interactive video generation branch (E2, E7), where compounding errors accumulate because substituted actions are not certified stable under re-evaluation, and across the persistent 3D / long-term spatial memory branches (E3, E4, E8), where revisited states drift because the action set used to probe memory is not closed under correction. The structural root cause is that safety is treated as a per-step classification rather than as a set-theoretic property of the action space.

## Key Insight

If the safe subset is defined as the fixed point of a learned retraction onto the certified-safe region of the DiT hidden-state manifold, then substitution is idempotent by construction, so the substituted action set is closed under repeated substitution rather than merely empirically low-violation.

## Method

**What it is.** CASAC (Certified-Safe Action Closure) is a wrapper around a frozen interactive video world model and a frozen DiT safety head (as in NoPeWorld) that maps any candidate action `a` to a certified-safe action `Π(a)` lying in an explicitly constructed admissible action manifold `M`, such that `Π(Π(a)) = Π(a)` and the safety head certifies `Π(a)` as safe. Inputs: candidate action `a`, current latent/observation context `s`, frozen DiT hidden states `h(s)`, frozen safety head `g`. Output: certified-safe action `a*` plus a certificate flag.

**Load-bearing assumption (stated explicitly).** CASAC's certificate is *relative to the frozen safety head `g`*: we assume `g`'s safe/unsafe label is a sound certificate of true physical admissibility. This assumption is load-bearing and is not verified by the method itself; it can fail when `g` is systematically wrong on a coherent region of action space (e.g., novel obstacle geometry), in which case `M` and `Π` are closed with respect to `g`'s biases rather than to physics. To bound this, we measure the head's false-safe rate on substituted actions against an independent oracle (ground-truth collision/penetration checks in the NoPeBench simulator) and report it as a measured quantity entering the certificate as a probabilistic safety guarantee with an explicit error budget `ε_head` (see step 11). If no independent oracle is available, the contribution is restated as 'closure relative to a given safety head' with `ε_head` propagated, not as certified physical safety.

**How it works.**

```pseudocode
# Offline: build the admissible action manifold M from the frozen model
1. Sample N=50k actions a_i ~ A around the task distribution; compute h(s_i) via frozen DiT.
2. Label y_i = g(h(s_i), a_i) in {safe, unsafe} using the frozen NoPeWorld head.
3. Fit a low-dim latent chart: train an autoencoder E: A -> R^d, D: R^d -> A with d=16
   (E: 3-layer MLP, hidden=256, GeLU; D: symmetric 3-layer MLP, hidden=256, GeLU;
    trained with MSE reconstruction loss, Adam lr=1e-3, batch=256, 100 epochs).
4. On the latent codes z_i = E(a_i), fit a one-class support model (deep SVDD) with
   radius r chosen so that 95% of certified-safe codes lie inside; define
   M = { a : ||E(a) - c|| <= r }  (c = SVDD center).
5. Define the retraction R_theta as the exact Euclidean projection onto the SVDD ball:
   R_theta(z) = c + (z - c) * min(1, r / ||z - c||).
   This map is closed-form idempotent: R_theta(R_theta(z)) = R_theta(z) for all z.
   No learned retraction network is used; only E and D are learned.

# Online: certified substitution
6. Given candidate a and context s:
7.   if g(h(s), a) == safe and a in M: return (a, certificate=SAFE)
8.   z <- E(a); z* <- R_theta(z); a* <- D(z*)
9.   if g(h(s), a*) == safe and ||E(a*)-c|| <= r:
        return (a*, certificate=SAFE_CLOSED)
10.  else: return (nearest certified-safe action from a cached bank of 1k
        pre-certified actions minimizing ||E(a)-E(a_bank)||, certificate=SAFE_BANK)
11.  Certificate error budget: report ε_head = measured false-safe rate of g on
        substituted actions against ground-truth simulator collision checks on a
        held-out NoPeBench split (n=512 episodes); the returned certificate is
        annotated with ε_head as a probabilistic bound, not a zero-error guarantee.
```

Hyperparameters: latent dim `d=16`, SVDD radius `r` at the 95th percentile of certified-safe codes, bank size `1000` (with a bank-size sweep {100, 500, 1000, 5000} reported), autoencoder training as in step 3, `ε_head` measured on n=512 held-out episodes.

**Why this design.** We chose a closed-form projection onto an explicitly fitted manifold over NoPeWorld's free-form intent-preserving replacement because NoPeWorld's replacement has no membership test, accepting the cost that our manifold may exclude some genuinely safe but out-of-distribution actions (mitigated by the bank fallback). We chose a one-class SVDD latent chart over a discriminative safe/unsafe classifier because SVDD defines a set with a boundary we can project onto, whereas a classifier gives only a scalar score that cannot support a projection operator; the cost is that SVDD is sensitive to the 5% contamination rate in the safe set. We chose the exact Euclidean projection `R_theta(z) = c + (z-c)*min(1, r/||z-c||)` rather than a learned retraction network because closure under repeated substitution is a fixed-point property, and the Euclidean projection is idempotent by construction, so `Π(Π(a)) = Π(a)` holds up to the autoencoder reconstruction error `||D(E(a))-a||`; the cost is that the composed map `D∘proj∘E` is idempotent only up to that reconstruction error, which we report as a measured residual. A domain expert would not describe this as a variant of NoPeWorld because NoPeWorld's substitution is a lookup among safe candidates with no set-membership invariant, whereas CASAC's substitution is a projection whose output is certified to lie in a fixed point set — the mechanism is set-theoretic closure, not candidate ranking.

**Why this measures what we claim.** The SVDD membership test `||E(a)-c|| <= r` measures admissibility of an action because the assumption that certified-safe actions concentrate in a low-dimensional latent ball around a center holds when the frozen DiT's safety head is locally consistent; this assumption fails when the safety head is systematically wrong on a coherent region of action space (e.g., a novel obstacle geometry), in which case the ball reflects the head's bias rather than true physical admissibility. The idempotence residual `||Π(Π(a))-Π(a)||` measures closure under repeated substitution because the assumption that the Euclidean projection is idempotent at its fixed points makes the residual zero exactly on the closed set; this assumption fails when the autoencoder reconstruction error `||D(E(a))-a||` is large, in which case the residual reflects reconstruction error rather than projection error. The certificate flag `SAFE_CLOSED` measures certified safety because the assumption that the frozen head's safe label is a sufficient certificate for the substituted action makes the flag equivalent to admissibility; this assumption fails when the head's false-negative rate on substituted actions is nonzero, in which case the flag overstates safety and residual violations remain — this is why we report `ε_head` as a measured, bounded quantity entering the certificate rather than assuming it is zero.

## Contribution

['A certified-safe action closure operator (CASAC) that composes a frozen DiT safety head with a learned retraction onto a one-class admissible action manifold, producing substituted actions that are certified safe and idempotent under repeated substitution.', 'A design principle: safety in interactive video world models is a set-closure property of the action space, not a per-step classification accuracy, and closure can be enforced by a projection-form retraction with an explicit idempotence penalty.', 'An evaluation protocol that measures residual violation rate after k rounds of repeated substitution (k=1,2,5,10), exposing the closure gap that single-step violation rate hides.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | NoPeBench obstacle-interaction episodes | Same benchmark the frozen safety head was validated on |
| Primary metric | Physical violation rate under repeated substitution (pre-registered: 10 consecutive obstacle encounters, angle perturbation N(0, 15°), stopping criterion = 10 substitutions or 1 violation) | Directly tests closure, not one-shot safety |
| Secondary metric | Empirical idempotence residual `||Π(Π(a))-Π(a)||` distribution vs. training penalty | Tests the fixed-point claim directly |
| Secondary metric | Head false-safe rate `ε_head` on substituted actions vs. ground-truth simulator collision checks (n=512 held-out episodes) | Measures the load-bearing assumption |
| Baseline | NoPeWorld free-form replacement | Tests whether manifold membership beats candidate lookup |
| Baseline | Unconstrained retraction (no idempotence penalty) | Isolates the closure claim from projection alone |
| Baseline | Safe-bank nearest-neighbor only (bank size 1000, with sweep {100, 500, 1000, 5000}) | Tests whether the manifold adds anything over fallback |
| Ablation-of-ours | CASAC without SVDD membership gate | Isolates the set-membership invariant |
| Ablation-of-ours | CASAC with learned retraction network (original formulation) vs. closed-form projection | Tests whether closed-form idempotence is necessary |

### Why this setup validates the claim

CASAC's central claim is that certified-safe substitution should be a projection onto a fixed-point set, not a one-shot candidate swap. That claim is only falsifiable if the evaluation stresses repeated substitution and distinguishes membership from ranking. NoPeWorld's free-form replacement tests the ranking hypothesis: if lookup among safe candidates already achieves closure, CASAC's manifold is unnecessary machinery. The unconstrained-retraction baseline tests whether idempotence comes from the penalty or is free; if the penalty-free variant closes just as well, the closure argument is wrong. The safe-bank-only baseline tests whether the fitted manifold `M` adds coverage beyond a cached certified set. The no-SVDD ablation tests whether the membership boundary, rather than the retraction network, is doing the work. Violation rate measured after repeated application is the right metric because it is exactly the quantity that diverges between a one-shot substitution and a projection: a non-idempotent mapper accumulates drift across substitutions, and only a repeated-application metric exposes that. Single-application violation rate would leave the central claim untested. The `ε_head` measurement against ground-truth simulator collision checks directly tests the load-bearing assumption that the frozen head's label is a sound certificate, and the idempotence-residual distribution tests the fixed-point claim against the training penalty.

### Expected outcome and causal chain

**vs. NoPeWorld free-form replacement** — On a case where the agent repeatedly approaches the same obstacle from slightly different angles, NoPeWorld swaps each unsafe action for the nearest safe candidate independently, so successive substitutions drift toward different corners of the safe set and the trajectory oscillates near the obstacle boundary. Our method instead maps every candidate through the same projection onto `M`, so repeated substitution lands on the same fixed point and the trajectory stabilizes. We expect a modest gap on first-application violation rate but a visibly larger gap under repeated substitution, with the divergence growing with the number of consecutive obstacle encounters.

**vs. Unconstrained retraction (no idempotence penalty)** — On a case where a candidate action sits just outside `M` and the retraction output itself falls marginally outside the SVDD ball, the unconstrained network maps it to a nearby but different point on re-application, producing a two-cycle. Our method's closed-form Euclidean projection collapses that cycle to a fixed point. We expect near-parity on single substitutions but a clear separation in the fraction of trajectories whose substituted actions stop changing after one application; the unconstrained variant should show residual oscillation on a small but non-trivial slice of boundary cases.

**vs. Safe-bank nearest-neighbor only** — On a case where the candidate action is in a region of action space sparsely covered by the 1k cached safe actions, the bank returns a distant safe action that preserves safety but distorts intent, and the resulting trajectory deviates from the task. Our method projects onto the continuous manifold, landing closer to the original intent while remaining certified. We expect comparable violation rates between the two, but a measurable intent-preservation gap concentrated on the sparse-coverage subset, with parity where the bank is dense.

### What would falsify this idea

If CASAC's gain over NoPeWorld is uniform across first-application and repeated-substitution regimes rather than concentrated under repeated substitution, the closure mechanism is not what produces the improvement and the central claim is wrong. Likewise, if the unconstrained-retraction baseline matches CASAC on the oscillation metric, the idempotence penalty is doing nothing and the fixed-point argument fails. If the measured `ε_head` is large (e.g., >10%) on substituted actions, the certificate overstates safety and the contribution must be restated as closure relative to the head, not certified physical safety.

## References

1. Non-penetrating World Models
