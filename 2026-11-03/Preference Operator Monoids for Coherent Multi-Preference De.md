# Preference Operator Monoids for Coherent Multi-Preference Decoding

## Motivation

Personalized decoding methods such as PAD (2025) condition the LM on one preference at a time and then reweight the decoding distribution independently per preference, so combining several user preferences is done by ad-hoc score arithmetic that ignores how preferences interact. The structural root cause is that each preference is treated as a scalar modifier of the logit vector rather than as an operator with its own algebra, so composition is neither order-invariant nor guaranteed to stay on the manifold of coherent distributions; the Avatar Intelligence agenda paper explicitly lists identity-continuous personalization as an unsolved challenge but offers no compositional mechanism for enforcing multiple simultaneous preferences. This means a user asking for 'concise AND technically rigorous AND non-condescending' has no principled way to have all three honored jointly, because the existing decoder has no object that represents their interaction.

## Key Insight

If each preference is a projection-like operator with a shared fixed point, then composing them via a commutative monoid with an idempotent normalization yields a joint distribution that is order-invariant and stays coherent, because the composition acts on the same logit surface rather than on independently reweighted copies.

## Method

**What it is.** POM (Preference Operator Monoid) is a decoding-time procedure that represents each user preference as a parameterized operator T_pi acting on a base logit vector, and produces the next-token distribution by composing the active preference operators through a commutative, associative, idempotent reduction before a single softmax. Inputs: base LM logits z in R^V, a set of active preferences {pi_1..pi_k} with per-preference operator parameters, and a coherence temperature tau. Output: next-token distribution p over the vocabulary.

**How it works.**
```pseudocode
# Offline: fit one operator per preference from contrastive pairs
for each preference pi:
    collect pairs (z, y+) where y+ satisfies pi, (z, y-) where it violates pi
    fit T_pi(z) = z + W_pi * phi(z)        # phi = frozen 64-d logit summary
    loss = softmax-CE(T_pi(z), y+) - lambda * softmax-CE(T_pi(z), y-)
    project W_pi so T_pi is 1-Lipschitz in the logit metric  # idempotent-safe

# Online: compose at each decoding step
z_base = LM(context)
for pi in active_preferences:
    z_pi = T_pi(z_base)                    # each operator sees the SAME base
z_star = mean_over_permutations( reduce_assoc(z_pi) )  # monoid reduction
z_star = z_star / (1 + tau * ||z_star - z_base||)      # coherence rescale
p = softmax(z_star)
```
The reduction is the associative, commutative operation `a ⊕ b = a + b - a⊙b` applied coordinate-wise on the normalized operator outputs, which is idempotent (a⊕a = a) and therefore order-invariant. The coherence rescale pulls the composed logits back toward the base when the composition drifts far from the base manifold, with tau controlling how strictly coherence is enforced (tau=0 means no rescale, tau=2 means strong pullback).

**Why this design.** We chose a projection-like operator family (1-Lipschitz residual on a frozen logit summary) over an unconstrained learned reweighting head because an unconstrained head can compose into a distribution that no preference would individually endorse, accepting the cost that each operator is slightly less expressive than a full fine-tune. We chose a commutative idempotent monoid reduction over sequential application because sequential application makes the result order-dependent, which is unacceptable when a user states preferences in arbitrary order, accepting the cost that the reduction discards pairwise interaction terms beyond the first-order overlap captured by `a⊙b`. We chose a single coherence rescale applied after composition over per-operator normalization because normalizing each operator independently is exactly the independent-reweighting failure mode of PAD (2025), accepting the cost that tau must be tuned per model family. A domain expert would not describe POM as a variant of PAD because PAD conditions on one preference and reweights; POM composes operators on a shared base with an algebraic law that PAD's scalar reweighting does not satisfy.

**Why this measures what we claim.** The monoid reduction `a ⊕ b = a + b - a⊙b` measures joint preference satisfaction because the idempotent law makes the composed logits equal to the union of preference-consistent directions under the assumption that each operator's residual is an indicator of its preference's satisfied subspace; this assumption fails when two preferences are mutually exclusive (e.g., 'very concise' and 'exhaustive'), in which case the reduction reflects a compromise point rather than a jointly satisfiable region. The coherence rescale `z_star / (1 + tau * ||z_star - z_base||)` measures distributional coherence because the distance from the base logit surface is a proxy for how far the composed distribution has drifted off the language manifold under the assumption that the base model's logit geometry is locally linear; this assumption fails for rare tokens far from the base mode, in which case the rescale under-penalizes drift. The per-operator contrastive loss measures preference direction fidelity because the margin between y+ and y- CE is monotone in the operator's alignment with the preference under the assumption that the contrastive pairs are representative of the preference's boundary; this assumption fails for preferences with heavy context dependence, in which case the operator captures the average direction rather than the conditional one.

**Explicit load-bearing assumption and calibration.** The method assumes that each preference operator's residual is a projection-like indicator of that preference's satisfied subspace with a shared fixed point, so that the coordinate-wise idempotent reduction `a⊕b = a+b−a⊙b` faithfully represents joint satisfaction rather than an arbitrary compromise. This assumption is not enforced by the contrastive fitting objective; it is verified and calibrated on held-out multi-preference contrastive pairs. Concretely, we hold out 512 multi-preference examples (each with two or more active preferences and a binary joint-satisfaction label from an LLM judge) and fit a small gating network g_theta over the per-preference residuals (a 2-layer MLP with hidden size 128, GeLU activation, input dimension 64*k where k is the number of active preferences, output dimension 64) that modulates the reduction: `z_star = reduce_assoc( g_theta(z_pi) ⊙ z_pi )`. The gating network is trained with a binary cross-entropy loss on the joint-satisfaction label using the composed logits, with L2 regularization weight 1e-4 and Adam optimizer (learning rate 1e-3, batch size 32, 100 epochs). For preference pairs where the gating network's output indicates conflict (i.e., the predicted joint-satisfaction probability falls below 0.5), we output a detected-infeasibility signal and fall back to a Pareto compromise: we select the operator subset that maximizes the minimum per-preference satisfaction score (computed by the same LLM judge) and apply the reduction only to that subset. This calibration step is applied after the per-operator fitting and before online decoding; it does not alter the per-operator training procedure.

## Contribution

(1) A compositional preference algebra, POM, that represents each preference as a 1-Lipschitz logit operator and composes them through a commutative idempotent monoid reduction with a coherence rescale, giving order-invariant joint decoding over multiple simultaneous preferences. (2) A design principle: idempotence of the reduction is what buys order-invariance, and the coherence rescale is what keeps the joint distribution on-manifold, so any compositional decoder must satisfy both to avoid the independent-reweighting failure of PAD. (3) A procedure for fitting per-preference operators from contrastive pairs that keeps each operator Lipschitz-bounded, enabling the monoid law to hold at inference time without retraining.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | MultiPref pairwise preference benchmark with per-utterance preference tags | Provides order-invariance and conflict cases in one corpus |
| Dataset (secondary) | Held-out set of 500 real user instructions with 3+ simultaneous constraints | Tests whether monoid advantage persists on structurally different corpus |
| Primary metric | Joint-satisfaction rate under LLM-judge on multi-preference prompts | Directly measures the composed, not per-preference, outcome |
| Secondary metric | Perplexity of composed distribution under base model | Validates coherence rescale mechanism against judge coherence |
| Baseline | PAD (2025) scalar reweighting | Tests single-preference conditioning failure |
| Baseline | Sequential operator application | Tests order-dependence failure |
| Baseline | Independent per-operator normalization | Tests independent-reweighting failure mode |
| Ablation-of-ours | POM without coherence rescale (tau=0) | Isolates coherence term's contribution |
| Ablation-of-ours | POM without gating network (fixed identity gating) | Isolates calibration contribution |

### Why this setup validates the claim

The central claim is that composing preference operators on a shared base via a commutative, idempotent monoid, followed by a single coherence rescale, yields higher joint preference satisfaction than methods that condition on one preference at a time or apply operators sequentially. The dataset must therefore contain utterances with two or more simultaneously active preferences, and must include cases where preferences are stated in varying orders and cases where they partially conflict; MultiPref supplies both. The secondary dataset of 500 real user instructions with 3+ simultaneous constraints tests whether the advantage generalizes beyond the benchmark's construction. The primary metric — joint-satisfaction rate judged on the full composed output — is the right one because per-preference satisfaction can be high while the joint distribution is incoherent, which is exactly the failure mode we predict for baselines. The secondary metric — perplexity of the composed distribution under the base model — provides a measurable distributional quantity to validate the coherence rescale: we will compute the correlation between this perplexity and the LLM-judge coherence score on a held-out set of 200 composed outputs, expecting a negative correlation (higher perplexity, lower coherence). PAD tests whether single-preference reweighting can be composed at all; sequential application tests whether order-invariance matters in practice; independent normalization tests whether per-operator scaling produces the independent-reweighting collapse. The tau=0 ablation isolates whether the coherence rescale, rather than the monoid reduction alone, is responsible for the predicted gain. The gating-network ablation isolates whether the calibration step (which verifies the load-bearing assumption) contributes to joint satisfaction. Together these form a falsifiable test: if POM's advantage is real, it must appear specifically on multi-preference and order-permuted subsets, not uniformly, and the perplexity-coherence correlation must be significant.

### Expected outcome and causal chain

**vs. PAD (2025) scalar reweighting** — On a case where a user states "be concise" and "explain the tradeoffs" together, PAD conditions on one preference and applies a scalar reweight that satisfies conciseness at the expense of tradeoff coverage, because its reweighting has no mechanism to represent two preference directions simultaneously. Our method instead fits a separate operator per preference and composes them on the shared base, so the concise direction and the tradeoff-coverage direction both contribute to z_star. We expect a noticeable joint-satisfaction gap on two-preference prompts but near parity on single-preference prompts, where PAD's mechanism is not stressed.

**vs. Sequential operator application** — On a case where the same two preferences are stated in the order (exhaustive, then concise) versus (concise, then exhaustive), sequential application yields two different distributions because the second operator sees the first operator's output rather than the base, so the result depends on statement order. Our method applies each operator to the same base and reduces with an idempotent commutative operation, so both orderings collapse to the same z_star. We expect near-zero variance across order permutations for POM and a measurable order-induced spread for sequential application, concentrated on prompts where the two preference residuals point in different directions.

**vs. Independent per-operator normalization** — On a case where a strong preference (e.g., "no jargon") and a weak preference (e.g., "prefer active voice") are both active, independent normalization rescales each operator's output to comparable magnitude, letting the weak preference dominate the composed distribution and producing output that over-applies active voice while still leaking jargon. Our method composes the raw operator residuals before a single coherence rescale, so relative preference strength is preserved. We expect a gap on mixed-strength preference pairs and parity on equal-strength pairs, with the tau=0 ablation showing that the coherence term is what keeps the strong-preference output from drifting off-manifold.

**vs. POM without gating network** — On a case where the gating network detects conflict (e.g., "concise" and "exhaustive"), the uncalibrated POM applies the idempotent reduction and produces a compromise point that may satisfy neither preference well. With the gating network, the method outputs a detected-infeasibility signal and falls back to a Pareto compromise, which should yield higher joint satisfaction on conflict cases. We expect the gating ablation to show a gap specifically on conflict-heavy prompts, with parity on non-conflict prompts.

### What would falsify this idea

If POM's joint-satisfaction gain over PAD and sequential application were uniform across single-preference, multi-preference, and order-permuted subsets — rather than concentrated on multi-preference and order-permuted cases where the predicted mechanism gap occurs — the central claim that the monoid reduction and coherence rescale are what produce the effect would be wrong. Likewise, if the tau=0 ablation matched full POM on the multi-preference subset, the coherence rescale would be doing no causal work and the design rationale would collapse. Additionally, if the perplexity of the composed distribution under the base model did not correlate with LLM-judge coherence, the coherence rescale's mechanism would be unsupported.

## References

1. Avatar Intelligence: Amplifying Human Agency Through Persistent Personal Agents
