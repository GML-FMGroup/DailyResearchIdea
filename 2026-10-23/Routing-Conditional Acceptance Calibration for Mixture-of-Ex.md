# Routing-Conditional Acceptance Calibration for Mixture-of-Experts Speculative Decoding

## Motivation

Speculative decoding in MoE systems inherits a stationary acceptance test from dense models, but the target distribution in an MoE is a mixture whose component identity shifts with the draft prefix. Specter prefetches experts predicted by a quantized draft and SpecMoE self-drafts from a reduced expert path, yet both verify accepted tokens with the dense speculative-sampling ratio p_target/p_draft, which silently assumes a fixed expert subset; when draft and target routes diverge, the acceptance ratio is computed against the wrong mixture component, the acceptance process loses its martingale property, and accepted prefixes can become internally inconsistent or be invalidated by sibling branches during tree verification. The meta-level gap is that no acceptance-calibration invariant across these paper-trees ties the accept decision to the routing state that actually generated the token.

## Key Insight

The acceptance probability must be conditioned on the joint routing state (draft expert set, target expert set) rather than the token alone, because only then does the acceptance indicator remain a martingale under the target's routing-conditional measure, which is precisely what makes accepted-prefix length monotone in routing divergence and makes sibling branches unable to invalidate an accepted branch.

## Method

**What it is.** Routing-Conditional Acceptance (RCA) is a verification rule for MoE speculative decoding. Its inputs are the draft token sequence with the draft routing sets, the target model's routing sets for each candidate token, and the target/draft token distributions restricted to those sets; its output is an accepted-prefix length and a per-token acceptance decision that is a function of the pair of expert-activation sets.

**Load-bearing assumption (stated explicitly).** RCA preserves the exactness (distribution-preservation) property of speculative sampling only under the following assumption: the routing-conditional acceptance ratio r[t] = p_t(x_t | S_t) / p_d(x_t | S_d) is used as the exact rejection-sampling ratio, and on rejection the token is resampled from the normalized positive residual max(0, p_t(x | S_t) - p_d(x | S_d)). The multiplicative factor tau[t] = 1 - delta[t] is NOT part of the exact rejection-sampling identity; applying it without a compensating residual correction strictly reduces the acceptance probability below min(1, r[t]) and biases the accepted-token distribution toward tokens whose routes agree. Therefore, in this specification, tau[t] is applied only as a proposal-side reweighting (see step 2 below), and the accept/reject step uses the exact routing-conditional ratio with the exact residual resampling. If tau were instead multiplied into the accept probability, the rule would be a heuristic and the lossless claim would be dropped.

**How it works.**

```pseudocode
# Per candidate token x_t at position t, given draft prefix
# R_draft[t] = expert set activated by draft at t
# R_target[t] = expert set activated by target at t (from one target forward pass)
# p_d(x | R_draft[t]) = draft distribution over vocab
# p_t(x | R_target[t]) = target distribution over vocab

for t in 1..T:
    S_d = R_draft[t]; S_t = R_target[t]
    # routing divergence: symmetric difference normalized by union
    delta[t] = |S_d XOR S_t| / |S_d UNION S_t|
    # routing-conditional acceptance threshold (used only for proposal reweighting)
    tau[t]  = clip( 1 - delta[t] , tau_min=0.05, tau_max=1.0 )
    # exact routing-conditional importance ratio
    r[t]    = p_t(x_t | S_t) / p_d(x_t | S_d)
    # proposal reweighting: fold tau into the draft distribution (not the accept prob)
    q_t(x)  = tau[t] * p_d(x | S_d) + (1 - tau[t]) * p_t(x | S_t)
    # exact rejection-sampling accept test against the reweighted proposal
    accept[t] = ( u_t < min(1, p_t(x_t | S_t) / q_t(x_t)) )   # u_t ~ Uniform(0,1)
    if not accept[t]:
        # resample from exact residual for the routing-conditional target
        x_t ~ normalize( max(0, p_t(x | S_t) - q_t(x)) )
        break
return accepted_prefix_length = t-1

# Tree verification: for each root-to-leaf branch, run the loop above
# independently; a branch is accepted only if every node on it passes.
# Sibling branches do not share accept state; each branch's tau uses its
# own R_target, so a sibling's expert activation cannot alter another
# branch's accept decision.
```

Hyperparameters: `tau_min=0.05` prevents total rejection under complete routing divergence; `tau_max=1.0` recovers dense behavior when routes match exactly.

**Why this design.** We chose a routing-conditional multiplicative factor `tau[t]` over a learned acceptance head because the factor is a closed-form function of the routing sets, which means it can be computed from the routing signals Specter and SpecMoE already expose without new instrumentation, accepting the cost that it cannot adapt to distributional shifts beyond routing. We chose the normalized symmetric difference as the divergence measure over KL between expert-gate distributions because set overlap is invariant to gate-logit scale and directly reflects which expert parameters are actually loaded, accepting the cost that it discards soft gate magnitudes. We chose per-branch independent acceptance over a shared tree-level acceptance state because sharing state reintroduces the coupling that lets a sibling's expert activation invalidate an accepted branch, accepting the cost of redundant computation when branches share prefixes. A domain expert would not describe RCA as a variant of Specter or SpecMoE: those systems change *which* experts are prefetched or *how* drafts are produced, whereas RCA changes the acceptance *test itself* to be a function of the routing pair, which neither system does.

**Why it measures what we claim.** The quantity `delta[t]` measures routing divergence between draft and target because the normalized symmetric difference of expert-activation sets is equivalent to the fraction of active parameters that differ under the assumption that each expert contributes independently to the output; this assumption fails when experts interact through the shared gate, in which case `delta[t]` reflects set-level parameter mismatch rather than functional output divergence. The quantity `r[t]` restricted to `p_t(x|S_t)/p_d(x|S_d)` measures the routing-conditional importance weight because conditioning both distributions on their own expert sets is equivalent to evaluating each under the mixture component that generated it under the assumption that the routing decision is exogenous to the token (the router does not depend on the sampled token within a step); this assumption fails when the router is token-dependent (e.g., top-k gate computed after the token), in which case `r[t]` reflects an unconditional ratio mislabeled as conditional. The threshold `tau[t]` measures the admissible acceptance slack because scaling the dense acceptance probability by `1-delta[t]` is equivalent to requiring the accept decision to be robust to the routing mismatch under the assumption that acceptance probability should degrade linearly with the fraction of mismatched experts; this assumption fails under highly nonlinear expert interactions, in which case `tau[t]` reflects a conservative linear penalty rather than the true robustness margin.

## Contribution

(1) A routing-conditional acceptance rule for MoE speculative decoding whose per-token threshold is a closed-form function of the draft and target expert-activation sets, replacing the stationary dense acceptance test used by Specter and SpecMoE. (2) A design principle that accepted-prefix length should be monotone non-increasing in routing divergence and that tree branches must carry independent routing-conditional acceptance state, yielding a verification procedure in which no accepted branch can be invalidated by a sibling branch's expert activation. (3) An open implementation of the acceptance rule that consumes existing routing predictions without new instrumentation, applicable to any MoE speculative-decoding runtime.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | MoE model generation traces with per-token expert activation logs | Only source exposing both draft and target routing sets |
| Primary metric | Acceptance length (accepted tokens per target forward pass) | Directly measures the acceptance test we modify |
| Baseline | SpecMoE (dense acceptance) | Tests whether routing-conditional slack beats dense min(1,r) |
| Baseline | Specter (quantized multi-step offloading) | Tests whether acceptance gain survives real offloading bandwidth limits |
| Baseline | Dense speculative decoding (no routing conditioning) | Isolates the effect of tau from speculative decoding itself |
| Ablation-of-ours | RCA with tau fixed to 1.0 (dense test, same tree) | Removes only the routing factor, holding tree and drafts fixed |
| Ablation-of-ours | RCA with exact residual resampling (no tau) | Tests whether the exact rejection-sampling correction alone (without routing reweighting) changes acceptance length |

### Why this setup validates the claim

The central claim is that making the acceptance test a function of the draft-target routing pair increases accepted-prefix length precisely where routing diverges, without hurting agreement cases. Acceptance length is the right metric because it is the exact quantity the RCA loop returns; throughput and TPOT are downstream and confounded by offloading bandwidth, so they cannot isolate the acceptance mechanism. The dense-acceptance baseline (SpecMoE-style) tests the sub-claim that routing-conditional slack beats min(1,r) on the same drafts and tree. The Specter baseline tests whether that gain survives real offloading, where extra accepted tokens must translate into fewer target passes. The no-routing dense speculative baseline isolates RCA from speculative decoding itself. The tau=1.0 ablation is the decisive control: it holds the tree, drafts, and per-branch independence fixed and removes only the routing factor, so any gap between it and full RCA is attributable to the routing-conditional term and nothing else. The exact-residual ablation isolates the effect of the rejection-sampling correction from the routing reweighting. Together these form a falsifiable test: the gain must appear where routing diverges and vanish where routes match.

### Expected outcome and causal chain

**vs. SpecMoE (dense acceptance)** — On a case where the draft and target route the same token through mostly disjoint expert sets (high symmetric difference), dense acceptance uses min(1, r) with no routing signal, so it accepts at the dense rate even though the target's active parameters differ, producing accepted tokens that later force re-verification and a shorter effective prefix on divergent steps. Our method reweights the proposal by tau = 1 - delta, so it accepts aggressively when routes match and conservatively when they diverge, concentrating acceptance where the routing pair actually agrees. We expect a noticeable acceptance-length gap on the high-divergence subset and near-parity on the low-divergence subset, giving a subset-dependent, not uniform, improvement.

**vs. Specter (quantized multi-step offloading)** — On a case where Specter's multi-step drafts hit a target step whose active experts were not prefetched, the target forward pass stalls on expert loading, and because Specter's acceptance is routing-blind it cannot anticipate that stall, so extra accepted tokens do not reduce target passes on divergent steps. Our method's tau shrinks acceptance exactly at those divergent steps, so accepted branches are the ones whose experts were already loaded, and the accepted-prefix length tracks the prefetchable set. We expect RCA's gain over Specter to be concentrated on requests with bursty routing, with parity on requests whose routes are stable across steps.

**vs. Dense speculative decoding (no routing conditioning)** — On a case where the router is token-dependent and the draft's route differs from the target's, dense speculative decoding applies an unconditional importance ratio that mislabels the conditional weight, so it accepts a token whose target component never generated it, inflating false accepts and then truncating the prefix at the next step. Our method conditions both distributions on their own sets and reweights the proposal by tau, so a mismatched route lowers the accept probability before the false accept occurs. We expect fewer false accepts per branch and a longer accepted prefix on token-dependent-router workloads, with no change on exogenous-router workloads where the conditional and unconditional ratios coincide.

**vs. RCA with exact residual resampling (no tau)** — This ablation uses the exact routing-conditional ratio without the tau reweighting. We expect it to match dense speculative decoding on low-divergence steps and to show a smaller gain than full RCA on high-divergence steps, confirming that the tau reweighting is the source of the routing-conditional improvement.

### What would falsify this idea

The idea is falsified if the acceptance-length gain over the tau=1.0 ablation is uniform across routing-divergence subsets rather than concentrated on high-divergence steps, or if the gain is equally large on cases where draft and target routes match exactly — either pattern would show tau is not acting through routing divergence and the causal story is wrong.

## References

1. Specter: Fast MoE Inference via Quantization-Enabled Multi-Step Speculative Offloading
2. SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding
