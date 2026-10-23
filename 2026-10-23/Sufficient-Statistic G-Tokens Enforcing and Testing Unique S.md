# Sufficient-Statistic G-Tokens: Enforcing and Testing Unique Semantic Recovery in Unified Multimodal Models

## Motivation

The compression-and-retrieval bridge in GAFU (What Generation Knows) assumes that generation-side features form a stable, task-agnostic substrate whose compressed G tokens uniquely determine the fine-grained semantics they carry, and that understanding gains are causally attributable to those tokens. Neither property is enforced or tested: the bridge is post-hoc alignment of two separately-trained pathways with no unique-recovery constraint and no channel-isolated intervention, so the token's contribution is confounded with correlated post-training effects. This is the meta-gap across the GAFU, UniHetero, and BAGEL trees: repurposing generation-side representations without controlled isolation or identifiability guarantees.

## Key Insight

A compressed G token is a sufficient statistic for its claimed fine-grained semantics if and only if a probe trained only on generation-side data can recover those semantics from the token alone and a swapped-token intervention on the understanding pathway produces the exact predicted semantic change, turning a post-hoc alignment assumption into a falsifiable identity.

## Method

**What it is.** Sufficient-Statistic G-Tokens (SSG) is a post-training contract imposed on the GAFU bridge: the generation encoder E_g, the compressor C (GAFU's token compressor), and a held-out probe ensemble {P_1,...,P_K} (each trained only on generation-side data, with disjoint inductive biases) must satisfy a cycle-consistency identity across all probes, and the G-token channel is then tested by a channel-isolated swap intervention on the frozen understanding pathway U, with a matched nuisance-swap null and a Jacobian-based local-linearity check.

**Load-bearing assumption (stated explicitly).** SSG assumes that the frozen probe ensemble {P_k} is an injective, task-agnostic decoder of fine-grained semantics, so that low cycle-consistency error on g across all probes implies g contains all probe-recoverable semantic content, and that the probe delta P_k(g_orig)-P_k(g_swap) is a valid target for U's output delta. This assumption is not taken for granted: it is tested via (i) a Hewitt-Liang selectivity control for each probe, (ii) a probe-injectivity diagnostic (rank/entropy of P_k(g) across semantic classes), and (iii) a Jacobian conditioning check of U w.r.t. the G channel. If these diagnostics fail, the sufficiency identity is a property of the probe's inductive bias, not of g, and the causal_score is reinterpreted accordingly.

**How it works.**
```pseudocode
# Phase 0: probe ensemble training (held-out, no understanding data)
for k in 1..K:  # K=3 probes with disjoint inductive biases
    P_k = train_probe( E_g(D_gen), labels=semantic_labels(D_gen),
                       arch=ARCH[k] )  # ARCH = {linear, 2-layer MLP hidden=256 GeLU, 3-layer MLP hidden=128 GeLU}
    # Hewitt-Liang selectivity control: train P_k^ctrl on randomized labels
    P_k^ctrl = train_probe( E_g(D_gen), labels=shuffle(semantic_labels(D_gen)), arch=ARCH[k] )
    selectivity_k = acc(P_k) - acc(P_k^ctrl)  # require selectivity_k > 0.15
# Probe-injectivity diagnostic on held-out D_gen:
#   compute rank and entropy of {P_k(g)} across semantic classes; require
#   effective rank >= 0.8 * #classes and class-conditional entropy > 0.5 * log(#classes)

# Phase 1: cycle-consistency contract on G tokens (C updated only; E_g, {P_k} frozen)
for (x, y_sem) in D_contract:
    z   = E_g(x)                      # generation-side features
    g   = C(z)                        # compressed G token (GAFU compressor)
    y_hat_k = P_k(g) for all k        # semantic probe readouts
    L_cycle = mean_k || y_hat_k - y_sem ||^2 + lambda * || g - C(E_g(x)) ||^2   # lambda=0.1
    # enforce sufficient-statistic identity across all probes:
    L_unique = mean_k max(0, margin - ||P_k(g) - P_k(g_neg)||^2 + ||P_k(g) - P_k(g_pos)||^2)  # margin=0.5
    update C only (E_g and {P_k} frozen)

# Phase 2: channel-isolated intervention on understanding pathway U (frozen post-training)
# Local-linearity check: fit Jacobian J = dU/dg on held-out D_test (finite differences, eps=1e-3);
#   report condition number cond(J); require cond(J) < 100 for the linearity assumption to hold.
for (x, y_sem) in D_test:
    g_orig = C(E_g(x))
    g_swap = C(E_g(x'))   # x' has different fine-grained semantics
    # Channel-isolation sanity: verify all non-G channels of U are bit-identical under swap
    out_orig = U(x, g_orig)   # understanding pathway with original G channel
    out_swap = U(x, g_swap)   # understanding pathway, only G channel swapped
    Delta = out_orig - out_swap
    # Nuisance-swap null: swap a matched nuisance channel (pose/background) while holding g fixed
    out_nuis = U(x, g_orig, nuisance=x'_nuis)  # only nuisance channel swapped
    Delta_nuis = out_orig - out_nuis
    # Causal necessity: Delta must match P_k(g_orig)-P_k(g_swap) beyond the nuisance null
    causal_score = mean_k cos( Delta - Delta_nuis, P_k(g_orig)-P_k(g_swap) )
```

**Why this design.** We chose to freeze the generation encoder E_g and the semantic probe ensemble {P_k} during Phase 1 rather than jointly fine-tune them, because joint training would let the probes co-adapt to the compressor and destroy the held-out sufficiency test, accepting the cost that the compressor must fit a fixed probe geometry. We chose a contrastive uniqueness term over a pure reconstruction loss because reconstruction alone permits many-to-one compression (multiple semantics collapsing to the same g), while the contrastive margin forces distinct fine-grained semantics to occupy distinct probe-decodable regions, accepting the cost of needing negative pairs with controlled semantic distance. We chose a channel-isolated swap intervention over ablating the G channel entirely because ablation confounds token contribution with the loss of any input, whereas swapping only the G token while holding x and U fixed isolates the token's causal effect; this is the property that distinguishes SSG from GAFU's post-hoc alignment, and a domain expert would not describe SSG as a variant of GAFU because GAFU never tests unique recovery or causal necessity. We chose a probe trained only on generation-side data over a probe trained on understanding labels because the latter would leak the understanding pathway's semantics into the contract, accepting the risk that generation-side labels may be coarser. We chose a probe ensemble with disjoint inductive biases plus a selectivity control over a single frozen probe because a single probe's accuracy does not establish that g encodes semantics in a recoverable/unique form (Alain & Bengio 2016; Hewitt & Liang 2019), accepting the cost of K=3 probe trainings. We chose a nuisance-swap null over a zero baseline for causal_score because U's post-training may entangle G tokens with other channels, so the causal claim must be tested against a matched nuisance swap rather than against zero.

**Why it measures what we claim.** The cycle-consistency loss L_cycle measures sufficient-statistic sufficiency because the assumption that the frozen probe ensemble {P_k} is an injective decoder of fine-grained semantics (verified by the selectivity control and injectivity diagnostic) makes low reconstruction error across all probes equivalent to g containing all probe-recoverable semantic content; this assumption fails when probes are under-trained or semantic labels are coarse, in which case L_cycle reflects probe fit rather than token sufficiency. The contrastive uniqueness term measures unique recovery because the assumption that negative pairs differ only in fine-grained semantics makes the margin equivalent to separation of semantic equivalence classes; this assumption fails when negatives differ in incidental factors (pose, background), in which case the term reflects nuisance separation. The causal_score measures causal necessity because the assumption that the swap intervention changes only the G-token channel (verified by the bit-identical non-G channel check) and that U's output is locally linear in the semantic readout (verified by the Jacobian conditioning check, cond(J) < 100) makes the output delta minus the nuisance delta proportional to the probe delta; this assumption fails when U's post-training entangles G tokens with other channels, in which case causal_score reflects correlated channel effects rather than token-specific causation.

## Contribution

(1) A generation-side identifiability contract for compressed G tokens — a cycle-consistency identity between a frozen generation encoder, a compressor, and a held-out semantic probe — that operationalizes sufficient-statistic sufficiency and unique recovery as trainable constraints. (2) A channel-isolated swap intervention protocol that tests causal necessity of the G-token channel while holding the understanding pathway's post-training fixed, separating token contribution from correlated post-training effects. (3) An empirical finding on whether GAFU-style generation-side features are a stable, task-agnostic substrate for fine-grained semantics, and under what semantic granularity the sufficiency contract holds.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Fine-grained VQA subset with controlled semantic-difference pairs (e.g., CUB-200-2011 attribute annotations; negative pairs sampled with matched pose/background via documented rule) | Enables channel-isolated swap with known semantics |
| Primary metric | causal_score (mean_k cosine between output delta minus nuisance delta and probe delta) | Directly tests token-specific causal necessity |
| Baseline | GAFU post-hoc alignment | Tests whether SSG adds beyond alignment |
| Baseline | Joint-trained compressor+probe | Tests held-out sufficiency value |
| Baseline | Ablate-G-channel variant | Tests swap isolation vs. mere contribution |
| Baseline | Nuisance-swap-only variant | Tests whether causal_score reflects correlated-channel effects |
| Ablation-of-ours | SSG without contrastive uniqueness term | Isolates unique-recovery contribution |
| Ablation-of-ours | SSG with single probe (no ensemble) | Tests probe-ensemble necessity |
| Diagnostic | Probe selectivity (Hewitt-Liang control), probe injectivity (rank/entropy), Jacobian conditioning of U | Grounds sufficiency and linearity assumptions in measured properties |

### Why this setup validates the claim

The central claim is that SSG's G-token channel is a *sufficient and causally necessary* statistic for fine-grained semantics, not merely correlated with understanding performance. Each baseline probes a distinct sub-claim. GAFU post-hoc alignment tests whether the cycle-consistency contract adds anything beyond what alignment already achieves — if SSG matches GAFU, the contract is doing no work. The joint-trained compressor+probe baseline tests the value of the held-out probe: if joint training yields the same causal_score, freezing was unnecessary and the sufficiency test is vacuous. The ablate-G-channel variant tests whether the swap intervention is genuinely isolating the token channel or just detecting input perturbation; if ablation and swap give the same signal, the causal claim collapses to a generic contribution claim. The nuisance-swap-only variant tests whether causal_score reflects correlated-channel effects rather than token-specific causation; if it matches SSG, the causal claim is confounded. The ablation-of-ours (no uniqueness term) tests whether the contrastive margin is what forces distinct semantics into distinct probe-decodable regions. The single-probe ablation tests whether the probe ensemble is necessary for the sufficiency identity to hold across disjoint inductive biases. The primary metric — mean_k cosine between the understanding-output delta (minus nuisance delta) and the probe delta under a channel-isolated swap — is the right detector because it directly operationalizes causal necessity: a sufficient statistic should make U's output move in lockstep with P's readout when only g changes, beyond what a matched nuisance swap produces. A generic alignment or correlation method would produce high accuracy but low causal_score, which is exactly the dissociation the design is built to expose. The diagnostics (probe selectivity, probe injectivity, Jacobian conditioning) make the load-bearing assumptions explicit and measured rather than assumed.

### Expected outcome and causal chain

**vs. GAFU post-hoc alignment** — On a case where two images share coarse category but differ in fine-grained attribute (e.g., same object, different material), GAFU produces similar G tokens because its alignment objective rewards semantic overlap, not separation of fine-grained equivalence classes. Our method instead forces distinct probe-decodable regions via the contrastive uniqueness term, so the swap intervention yields a probe delta aligned with the output delta. We expect a noticeable causal_score gap on the fine-grained-difference subset but near-parity on coarse-category subsets.

**vs. Joint-trained compressor+probe** — On a case where the probe has co-adapted to the compressor's idiosyncratic geometry, the joint baseline produces high L_cycle reconstruction but the probe readout no longer tracks held-out semantic labels, because co-adaptation lets the probe absorb compressor-specific quirks. Our method instead keeps {P_k} frozen, so low L_cycle can only be achieved by g genuinely containing probe-recoverable semantics. We expect the joint baseline to show inflated reconstruction with degraded causal_score under swap, while SSG shows moderate reconstruction with high causal_score — a dissociation, not a uniform win.

**vs. Ablate-G-channel variant** — On a case where the G channel is removed entirely, the ablate baseline produces a large output delta simply because an input is missing, and this delta is uncorrelated with the probe delta because it reflects generic input loss. Our method instead swaps only g while holding x and U fixed, so the output delta is attributable to the token channel. We expect the ablate variant to show high-magnitude but low-cosine deltas, while SSG shows deltas whose direction matches P_k(g_orig)-P_k(g_swap) — the diagnostic is the cosine, not the magnitude.

**vs. Nuisance-swap-only variant** — On a case where only a matched nuisance channel (pose/background) is swapped, the nuisance variant produces an output delta that is correlated with incidental factors, not with the probe delta. Our method subtracts this nuisance delta before computing causal_score, so the remaining delta is attributable to the G-token channel. We expect the nuisance variant to show low causal_score, while SSG shows high causal_score after subtraction — confirming that the causal claim is tested against a matched null rather than zero.

**vs. SSG without contrastive uniqueness term** — On a case where distinct fine-grained semantics collapse to the same g under pure reconstruction, the no-uniqueness ablation produces high L_cycle but low separation of probe-decodable regions, so the swap intervention yields a probe delta that is not aligned with the output delta. Our method instead forces distinct semantics into distinct probe-decodable regions via the contrastive margin. We expect the no-uniqueness ablation to show degraded causal_score, while SSG shows high causal_score — isolating the uniqueness term's contribution.

**vs. SSG with single probe** — On a case where a single probe's inductive bias permits nuisance-correlated geometry, the single-probe variant satisfies the cycle-consistency identity for that probe but fails to generalize across disjoint inductive biases, so the sufficiency claim is a property of the probe's bias rather than of g. Our method instead requires the identity to hold across K=3 probes with disjoint architectures. We expect the single-probe variant to show lower causal_score and lower probe-injectivity diagnostics, while SSG shows high causal_score across all probes — confirming the ensemble's necessity.

### What would falsify this idea

If SSG's causal_score advantage over GAFU post-hoc alignment is uniform across fine-grained and coarse subsets rather than concentrated where fine-grained semantic separation is required, the claim that the contrastive contract enforces unique recovery is wrong. Likewise, if the no-uniqueness ablation matches full SSG on causal_score, the uniqueness term is doing no causal work and the sufficiency story reduces to ordinary alignment. Additionally, if the probe selectivity control fails (selectivity_k ≤ 0.15) or the Jacobian conditioning check fails (cond(J) ≥ 100), the load-bearing assumptions are not met and the causal_score cannot be interpreted as token-specific causation.

## References

1. What Generation Knows: Unlocking Fine-Grained Understanding in Unified Multimodal Models
2. UniHetero: Could Generation Enhance Understanding for Vision-Language-Model at Large Data Scale?
3. Emerging Properties in Unified Multimodal Pretraining
4. Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling
