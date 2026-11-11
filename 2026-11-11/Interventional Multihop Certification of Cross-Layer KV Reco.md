# Interventional Multihop Certification of Cross-Layer KV Reconstruction

## Motivation

Cross-layer KV reconstruction methods such as DepthGraft, FusedKV, and KVSharer all justify substituting reconstructed deep-layer KV for full cross-layer KV using representational-geometry diagnostics and linear probes, which are structurally blind to nonlinear composition: a reconstructed KV pair can match the source KV's key directions and value subspace while failing to drive the same multi-head, multi-layer circuit. The field's shared assumption — that geometric/linear-probe similarity certifies substitution — is never tested against the causal property that actually matters, namely whether downstream nonlinear multihop computation remains executable under the reconstruction map, so DepthGraft's reported geometry improvements cannot distinguish a functionally substitutable reconstruction from a merely correlated one.

## Key Insight

A reconstructed KV pair is substitutable exactly when the interventional effect of ablating a downstream head's inputs is invariant under the reconstruction map along every multihop circuit edge, a property that representational similarity cannot express because it is defined on the composition of heads rather than on the KV vectors themselves.

## Method

**What it is.** Causal Multihop Certification (CMC) takes a trained transformer with DepthGraft-style hierarchical cross-layer KV reconstruction and outputs, per reconstruction edge (deep layer ℓ ← source pair (s1,s2)), a certification verdict: whether the reconstructed KV supports the same nonlinear multihop circuit as the full KV. Inputs are the model, a probe corpus, and a set of candidate multihop circuits; output is a per-edge causal-capacity gap Δ and a pass/fail certificate.

**Load-bearing assumption (stated explicitly).** CMC certifies an edge only under the assumption that the interventional path-patching effect of ablating an upstream head on a downstream head's logit-difference contribution, measured under full KV, is invariant under substitution of the reconstructed KV at layer L — i.e., that a per-edge, mean-ablation, single-head-pair interventional score computed on the composition of heads is a faithful and stable proxy for the multihop substitutability of the KV pair itself. This assumption fails under head redundancy and backup behavior (Wang et al., IOI 2023; Conmy et al., ACDC 2023; Michel et al., 2019): mean-ablating one head and patching clean activations yields redundancy-adjusted contribution rather than unique causal role, so Δ can be contaminated by redundancy structure. CMC therefore (i) replaces the single-head score with a redundancy-robust set-level estimand, (ii) requires certificate invariance across two corruption schemes and two ablation baselines, and (iii) reports a per-edge redundancy diagnostic so Δ is interpreted only where the circuit is non-degenerate.

**How it works.**

```pseudocode
# Phase 1: discover multihop circuits through the target deep layer
#   Pre-registered candidate circuit set: use published circuit analyses
#   (IOI circuit heads from Wang et al. 2023; relational-recall heads from
#   the probe suite's known circuit) as the fixed candidate upstream set,
#   rather than exhaustive upstream-head search.
for each deep layer L with reconstruction edge e = (s1, s2) -> L:
    circuits[L] = []
    for each downstream head h_d in layers L..L+K (K=4):
        # path patching: which upstream heads feed h_d via the reconstructed KV?
        for each candidate upstream head h_u in pre_registered_set(L):
            score = path_patch_effect(h_u -> h_d | clean vs corrupted prompt pair)
            if score > tau_path (tau_path = 0.05 of logit-diff scale):
                circuits[L].append((h_u, h_d, score))
    # redundancy diagnostic per edge: fraction of circuit edges whose
    # single-head score changes by >20% when the full upstream set is
    # ablated jointly (set-level vs single-head discrepancy).
    redundancy[L] = mean_over_edges( |score_single - score_set| / (score_set + eps) )

# Phase 2: define causal capacity functional on a circuit
#   C(KV) = E_x[ sum over (h_u,h_d) in circuit of
#             | logit_diff( x ; KV, do(ablate full upstream head-set) )
#             - logit_diff( x ; KV, do(ablate full upstream head-set), do(patch h_d input from clean) ) | ]
#   i.e., the interventional necessity of the full upstream head-set for h_d's
#   contribution, summed over multihop edges. Set-level joint ablation replaces
#   single-head mean-ablation to be redundancy-robust.

# Phase 3: compare full-KV vs reconstructed-KV capacity
for each edge e:
    # two corruption schemes: (a) subject/object swap, (b) random token replacement
    # two ablation baselines: (a) mean-ablation over probe set, (b) zero-ablation
    for scheme in {swap, random_replace}:
        for baseline in {mean_ablate, zero_ablate}:
            C_full[scheme,baseline]  = causal_capacity(circuits[L], KV=full_KV,          n_probe=256, scheme, baseline)
            C_recon[scheme,baseline] = causal_capacity(circuits[L], KV=reconstructed_KV, n_probe=256, scheme, baseline)
            delta_e[scheme,baseline] = |C_full - C_recon| / (C_full + eps)
    # certificate requires invariance across all four (scheme, baseline) cells
    delta_e = max(delta_e[scheme,baseline])
    certify(e) = PASS  if delta_e < delta_max (delta_max = 0.10) AND redundancy[L] < 0.20
                 FAIL  otherwise

# Phase 4: report per-edge certificate + aggregate; flag edges whose
# reconstruction preserves geometry (cos-sim > 0.9) but fails CMC;
# report redundancy[L] per edge so Δ is interpreted only where non-degenerate.
```

The functional C(KV) is computed by (i) running the clean prompt to cache all head outputs, (ii) running a corrupted prompt (subject/object swapped, or random token replacement) to obtain a counterfactual reference logit-difference, (iii) for each circuit edge, jointly ablating the full discovered upstream head-set (mean-ablation over the probe set, or zero-ablation) and measuring the drop in the target head's logit-difference contribution, both with full KV and with reconstructed KV substituted at layer L. The gap Δ isolates exactly the part of the multihop computation that the reconstruction map fails to carry. The certificate is issued only when Δ < 0.10 across all four (corruption scheme × ablation baseline) cells and the redundancy diagnostic is below 0.20; otherwise the edge is flagged as non-certifiable due to redundancy contamination.

**Why this design.** We chose interventional path patching over representational cosine similarity (as used in DepthGraft's geometry analysis and FusedKV's information-flow probing) because cosine similarity lives on KV vectors and cannot see whether two heads compose; path patching is defined on the composition itself, accepting the cost that it requires paired clean/corrupted prompts and 256 forward passes per edge rather than a single forward pass. We chose a *relative* capacity gap Δ (normalized by C_full) over an absolute threshold because absolute interventional effects scale with model size and task, so a normalized gap is comparable across DepthGraft's stride-aligned source pairs, accepting the cost that edges with near-zero C_full become noisy and must be excluded by a minimum-capacity filter. We chose to restrict circuits to depth K=4 downstream layers over full-model circuit discovery because multihop effects beyond four layers are dominated by residual-stream carryover rather than the reconstructed KV, accepting the cost that very long-range circuits are not certified. We chose set-level joint ablation over single-head mean-ablation because single-head scores are non-identifying under head redundancy (Wang et al. 2023; Conmy et al. 2023; Michel et al. 2019), accepting the cost that set-level ablation requires the full discovered upstream set and cannot attribute to individual heads. We chose to require invariance across two corruption schemes and two ablation baselines because circuit membership is threshold- and corruption-scheme-dependent (Conmy et al. 2023), accepting the cost of 4× forward passes per edge. A domain expert would not describe CMC as a variant of DepthGraft's linear-probe validation: DepthGraft probes whether the reconstructed KV *looks like* the full KV in representation space, whereas CMC intervenes on the heads that *consume* the KV and measures whether their composition is preserved — the object of measurement is the circuit, not the vector.

**Why this measures what we claim.** The normalized capacity gap Δ measures *causal substitutability* of a reconstructed KV pair because the assumption that interventional path-patching effects on downstream head composition are invariant under the reconstruction map is exactly the substitution property being certified; this assumption fails when a downstream head's output depends on KV components that are linearly recoverable but nonlinearly recombined (e.g., a key direction that is preserved in isolation but whose interaction with a second key in the same head is distorted), in which case Δ reflects local linear faithfulness rather than multihop substitutability. The set-level path-patch score measures *circuit membership* of the upstream head-set because the assumption that jointly ablating the full upstream set and patching its clean activation back isolates that set's causal contribution to the target head holds when the set is not redundantly duplicated; this assumption fails under head redundancy, in which case the score reflects the redundancy-adjusted contribution rather than the set's unique role — hence the redundancy diagnostic and the requirement that Δ be invariant across corruption schemes and ablation baselines before a certificate is issued. The clean/corrupted prompt pair measures the *task-relevant logit-difference* because the assumption that subject-object swap (or random token replacement) is a valid intervention on the multihop relation holds for the relational-recall probes used; this assumption fails for prompts where the swap changes the task itself, in which case the functional reflects a different relation than the one the circuit implements.

## Contribution

(1) A causal capacity functional and certification procedure (CMC) that replaces geometric/linear-probe validation of cross-layer KV reconstruction with an interventional multihop-computation criterion, producing per-edge pass/fail certificates. (2) A concrete empirical finding about the dissociation between representational similarity and causal substitutability: reconstruction edges that pass DepthGraft-style geometry checks can still fail CMC, identifying which stride-aligned source-pair topologies preserve nonlinear circuits. (3) An analysis tool that maps each reconstruction edge to the downstream multihop circuits it supports, enabling topology selection guided by circuit preservation rather than representational geometry.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Relational-recall probe suite with subject/object swaps and random-token-replacement corruptions | Supplies clean/corrupted pairs for path patching across two schemes |
| Primary metric | Normalized causal-capacity gap Δ per reconstruction edge, computed as max over {swap, random_replace} × {mean_ablate, zero_ablate} | Directly measures multihop substitutability, not vector similarity |
| Secondary metric | Redundancy diagnostic per edge (fraction of circuit edges with >20% single-head vs set-level score discrepancy) | Flags edges where Δ is redundancy-contaminated |
| Baseline | DepthGraft geometry validation (cosine probe) | Tests whether representation similarity suffices |
| Baseline | FusedKV information-flow probing | Tests whether flow statistics predict circuit preservation |
| Baseline | KVSharer reconstruction without certification | Tests whether sharing quality implies causal faithfulness |
| Ablation-of-ours | CMC with geometry-only gate (drop Phase 3 interventions) | Isolates value of interventional certification |
| Ablation-of-ours | CMC with single-head mean-ablation (original Phase 2) instead of set-level joint ablation | Isolates value of redundancy-robust estimand |

### Why this setup validates the claim

The central claim is that representational similarity of a reconstructed KV pair does not imply causal substitutability in downstream multihop circuits, and that interventional path patching on the consuming heads is what detects the difference. The dataset supplies paired clean/corrupted prompts under two corruption schemes (subject/object swap and random token replacement), so the metric Δ is computed on the same relation the circuit implements and is required to be invariant across schemes before a certificate is issued. DepthGraft's cosine probe tests the sub-claim that geometry is sufficient; if Δ and cosine disagree on the same edges, geometry is not sufficient. FusedKV's information-flow probing tests the sub-claim that aggregate flow statistics predict composition; if Δ flags edges that flow probing passes, flow is not sufficient. KVSharer provides a third reconstruction family to check the pattern is not specific to one map. The ablation removes the interventional step while keeping the geometric gate, so any gain attributable to certification rather than to the reconstruction map itself shows up as a gap between CMC and the ablation on the same edges. The single-head ablation isolates the value of the redundancy-robust set-level estimand: if the single-head variant produces certificates that flip under redundancy (high redundancy diagnostic), while the set-level variant is stable, the load-bearing assumption is verified. Δ is the right metric because it is normalized by C_full, making it comparable across edges whose absolute interventional effects differ, and because it is defined on head composition rather than on vectors.

### Expected outcome and causal chain

**vs. DepthGraft geometry validation (cosine probe)** — On a case where a reconstructed key direction is preserved in isolation (cos-sim above 0.9) but its interaction with a second key in the same head is distorted, the cosine probe passes the edge because it measures vector-level fidelity and cannot see the recombination. Our method instead path-patches the full upstream head-set feeding that downstream head and measures the drop in the target head's logit-difference contribution under reconstructed KV, so the distorted interaction shows up as a capacity gap. We expect a cluster of edges with high cosine but Δ above the 0.10 threshold — a clear pass/fail disagreement concentrated on edges whose downstream heads attend to multiple keys jointly.

**vs. FusedKV information-flow probing** — On a case where top-layer information flow statistics look healthy because the reconstructed KV carries the same marginal entropy and norm profile, but the specific head-to-head composition that implements the multihop relation is broken, flow probing reports no anomaly because it summarizes the KV distribution rather than the circuit consuming it. Our method intervenes on the upstream head-set and patches the downstream head's input, isolating the composition edge itself, so the broken edge produces a large Δ while flow statistics remain unremarkable. We expect flow-probe scores to be near parity between passing and failing edges, with Δ separating them.

**vs. KVSharer reconstruction without certification** — On a case where a layer-wise dissimilar sharing decision reconstructs a KV pair that is adequate for single-hop recall but loses the second hop of a two-hop relation, an uncertified sharing pipeline accepts the edge because it optimizes reconstruction error, not circuit preservation. Our method certifies per edge against the discovered multihop circuit, so the second-hop loss surfaces as a relative capacity gap even when reconstruction error is low. We expect uncertified sharing to show near-zero reconstruction error on failing edges, and CMC to flag exactly those edges, with the aggregate failure rate rising as circuit depth K increases.

**vs. CMC with single-head mean-ablation (ablation-of-ours)** — On edges with high redundancy diagnostic (above 0.20), the single-head variant is expected to produce unstable Δ values that flip across corruption schemes and ablation baselines, because mean-ablating one head and patching clean activations yields redundancy-adjusted contribution rather than unique causal role. The set-level joint ablation is expected to yield stable Δ across all four (scheme, baseline) cells on the same edges, verifying that the redundancy-robust estimand is necessary for a faithful certificate.

### What would falsify this idea

If edges that pass the geometry probe and the flow probe show the same Δ distribution as edges that fail them — i.e., if Δ does not concentrate on edges with high cosine but distorted multihop composition — then interventional certification is not measuring anything the representation-level probes miss, and the central claim is wrong. Equivalently, if CMC's pass/fail verdicts are uniform across circuit depth K rather than worsening with K, the depth-restricted circuit is not the locus of the effect; and if the set-level variant's Δ is not more stable than the single-head variant's on high-redundancy edges, the redundancy-robust estimand does not repair the load-bearing assumption.

## References

1. DepthGraft: Structural Regularization through Hierarchical Cross-Layer KV Reconstruction
2. Reconstructing KV Caches with Cross-layer Fusion For Enhanced Transformers
3. KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing
