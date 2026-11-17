# Chain-Indexed Permutation-Equivariant Residue-State Bundles for Multi-Chain Protein Folding

## Motivation

Existing protein language model (PLM) interfaces to structure predictors, exemplified by Calpha (E6, E12), are structurally limited to a single residue-state channel: the LLM's residue states are the sole information path to the folding head, and this path is validated only for monomeric single-chain folding. Because the channel carries no chain identity or interface context, the LLM's language reasoning cannot be shown to govern quaternary-structure assembly, and the folding action set is undefined for multi-chain inputs. This is not a data-scaling issue but a representational one: the interface schema itself lacks the compositional structure needed to express which residue belongs to which chain and how chains interface.

## Key Insight

A chain-indexed, permutation-equivariant residue-state bundle is the minimal sufficient statistic for multi-chain folding: if the SimpleFold head receives a concatenated bundle that is invariant to chain ordering and tagged with chain identity and interface context, then language reasoning can govern quaternary assembly without any change to the folding head's weights, because chain permutation symmetry is enforced structurally rather than learned.

## Method

**What it is.** We introduce **CAPS** (Chain-Aware Permutation-equivariant State interface), a schema and training procedure that extends the Calpha residue-state interface from a single sequence channel to a compositional, multi-chain bundle. Inputs: a multi-chain protein specification (sequences, chain identifiers, optional interface hints) and a language prompt. Outputs: per-chain residue states tagged with chain identity and interface context, and folded 3D coordinates for all chains.

**Load-bearing assumption (stated explicitly).** CAPS assumes that a frozen SimpleFold folding head — trained only on single-chain residue states from a single sequence channel — will correctly interpret a concatenated multi-chain bundle tagged with chain identity and interface context and produce accurate quaternary coordinates without any change to its weights. This is a transferability claim across an input-distribution shift the head was never trained on. We therefore treat it as a hypothesis to be verified, not a premise: before any multimeric claim is made, we run a **frozen-head compatibility calibration** (Phase 0 below) that feeds a naively stacked multi-chain bundle into the unmodified SimpleFold head and reports whether it produces non-degenerate coordinates at all. If the calibration fails, we fall back to a **lightweight input-projection adapter** on the SimpleFold head (a single linear layer mapping the bundle's D-dimensional state to the head's native input width, 256 hidden units, trained for 2k steps at lr=1e-4 while all other head weights stay frozen), and the contribution is reframed as a chain-aware interface schema plus a minimally-adapted head rather than a strictly frozen-head result. The permutation-consistency loss cannot repair this failure because it only shapes the LLM-side states, not the frozen head's interpretation of them.

**How it works.**

```pseudocode
# Phase 0: Frozen-head compatibility calibration (run FIRST, before any multimeric claim)
B_naive = stack({Qwen3.5-9B.encode(sequence_c, ESM2_features_c, prompt) for c in input.chains})  # no tags, no iface
coords_naive = SimpleFold_head(B_naive)          # frozen, unmodified
report: fraction of residues with non-degenerate coordinates (finite, within 5 Å of any ground-truth atom),
        coordinate variance across chains, and whether output dimensionality matches C chains
if calibration fails (degenerate or wrong-shape output):
    enable input_projection_adapter: Linear(D -> 256), GeLU, trained 2k steps, lr=1e-4, all other head weights frozen
    re-run calibration; if still failing, restrict claim to per-chain single-channel reuse and report as negative result

# Phase 1: Chain-tagged state emission (Qwen3.5-9B)
for each chain c in input.chains:
    h_c = Qwen3.5-9B.encode(sequence_c, ESM2_features_c, prompt)
    tag_c = chain_identity_embedding(c)          # learned, d=128
    iface_c = interface_context_pool(h_c, all_other_chains)  # cross-chain attention, 2 layers, 4 heads
    s_c = concat(h_c, tag_c, iface_c)            # per-chain residue states

# Phase 2: Permutation-equivariant bundle assembly
B = stack({s_c})                                # shape [C, L, D]
B_perm = random_chain_permutation(B)            # training-time augmentation

# Phase 3: Folding head invocation
coords = SimpleFold_head(B)                     # frozen weights (or adapter-enabled per Phase 0), receives bundle
loss = TM_score_loss(coords, ground_truth) + lambda * permutation_consistency_loss(B, B_perm)
```

The permutation-consistency loss enforces that `SimpleFold_head(B)` and `SimpleFold_head(B_perm)` produce coordinates that differ only by the same chain permutation, with lambda=0.1. The interface-context pool uses 2 cross-chain attention layers with 4 heads each. Chain-identity embeddings are learned during a short fine-tuning phase (10k steps, lr=1e-4) while SimpleFold head weights remain frozen (except the Phase 0 adapter, if enabled).

**Why this design.** We chose a chain-indexed bundle with a permutation-consistency loss over a learned chain-ordering classifier because chain order in multi-chain complexes is arbitrary and a classifier would memorize orderings rather than learn equivariance, accepting the cost that the loss adds a training-time augmentation step and requires paired forward passes. We chose to keep the SimpleFold head frozen over fine-tuning it end-to-end because the motivation is to show that language reasoning can govern assembly through the interface alone, accepting the cost that any folding errors attributable to the head cannot be corrected; the Phase 0 calibration and optional input-projection adapter are the concrete verification that this frozen-head choice is viable, and if they fail we say so rather than assert the claim. We chose cross-chain attention for interface context over concatenating all chains into one long sequence because concatenation destroys chain boundaries and scales quadratically in total length, accepting the cost that cross-chain attention adds parameters and a second forward pass. A domain expert would not describe this as a variant of Calpha because Calpha's interface is a single residue-state channel with no chain structure, whereas CAPS makes chain identity and interface context explicit components of the state schema and enforces permutation equivariance structurally.

**Why it measures what we claim.** The permutation-consistency loss measures chain-order invariance because it assumes that the folding head's output should be equivariant to chain permutation under the same physical complex; this assumption fails when two chains are structurally identical and interchangeable, in which case the loss reflects arbitrary symmetry breaking rather than true equivariance. The interface-context pool measures interface placement because it assumes that cross-chain attention captures residue-residue contacts across chains; this assumption fails when interface contacts are mediated by ligands or ions not represented in the sequence, in which case the pool reflects sequence-level co-evolution rather than physical interface geometry. The chain-identity embedding measures chain ordering because it assumes that chain identity is a sufficient tag for the head to assign coordinates to the correct chain; this assumption fails when chains have identical sequences but different structural roles, in which case the embedding reflects positional index rather than functional identity. The permutation-consistency loss also assumes that the frozen SimpleFold head treats bundle entries symmetrically (no positional bias in its input encoding); this assumption fails if the head's fixed positional encoding leaks chain index, in which case the loss is minimized by index memorization rather than equivariance, and we diagnose this by permuting chains at inference and reporting coordinate drift (mean RMSD between `SimpleFold_head(B)` and the permuted-back `SimpleFold_head(B_perm)`) on a held-out set of 256 complexes.

## Contribution

['A chain-indexed, permutation-equivariant residue-state schema (CAPS) that extends the Calpha single-channel interface to multi-chain protein complexes, with explicit chain-identity tags and cross-chain interface-context pooling.', "A permutation-consistency training objective that enforces chain-order invariance in the folding head's output without modifying the head's weights, enabling language reasoning to govern quaternary assembly through the interface alone.", 'An evaluation protocol for multi-chain folding that tests chain ordering, interface placement, and closure of the folding action set across multi-chain inputs.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Multimeric complexes with varied chain counts and homomer/heteromer mix; first milestone restricted to C=2 (one homodimer family, one heterodimer family) | Tests chain-order equivariance and identity tagging directly |
| Primary metric | Mean TM-score on multimeric assemblies | Captures whole-complex coordinate accuracy, not single-chain |
| Baseline | Calpha single-channel residue-state interface | Isolates contribution of chain-aware bundle schema |
| Baseline | Concatenated multi-chain sequence into SimpleFold | Tests whether chain boundaries matter versus flat input |
| Baseline | Fine-tuned SimpleFold head (unfrozen) | Tests frozen-head claim and language-governed interface |
| Baseline | Dedicated multimer predictor (AlphaFold-Multimer) on the same benchmark | Establishes whether language-governed interface beats purpose-built quaternary predictors |
| Ablation-of-ours | CAPS without permutation-consistency loss | Isolates equivariance term from bundle schema |
| Ablation-of-ours | CAPS with fixed one-hot chain index replacing learned chain-identity embedding | Isolates whether learned identity tag drives gain |
| Ablation-of-ours | CAPS with mean-pooled chain states replacing cross-chain attention | Isolates whether interface-context pool drives gain |
| Diagnostic | Inference-time chain permutation drift (mean RMSD between `SimpleFold_head(B)` and permuted-back `SimpleFold_head(B_perm)`) on 256 held-out complexes | Tests whether frozen head's positional encoding leaks chain index |
| Calibration | Phase 0 frozen-head compatibility: fraction of non-degenerate coordinates on naively stacked bundle | Establishes head's input contract before claiming schema sufficiency |

### Why this setup validates the claim

The central claim is that making chain identity and interface context explicit in the residue-state schema, and enforcing permutation equivariance structurally, lets a frozen folding head assemble multi-chain complexes better than interfaces that lack chain structure. Each baseline probes one sub-claim. Calpha's single-channel interface tests whether the chain-aware bundle adds anything beyond the original residue-state channel; if CAPS only matches it, the schema extension is inert. The concatenation baseline tests whether preserving chain boundaries matters versus simply feeding all sequences as one long input, which is the cheapest alternative and the one most likely to be tried. The unfrozen-head baseline tests whether the frozen-head design is a virtue or a handicap: if end-to-end fine-tuning dominates, the claim that language reasoning governs assembly through the interface alone is undermined. The dedicated multimer predictor baseline (AlphaFold-Multimer) tests whether the language-governed interface offers any advantage over purpose-built quaternary predictors, not just over Calpha. The two component ablations (fixed one-hot index, mean-pooled chain states) test whether the compositional bundle structure itself, rather than its individual components, drives any gain. The permutation-consistency ablation isolates the equivariance term from the bundle schema, so a gain cannot be attributed to equivariance if it appears without the loss. The Phase 0 calibration and inference-time permutation-drift diagnostic test the load-bearing assumption that the frozen head can consume the bundle and treats entries symmetrically. Mean TM-score on whole assemblies is the right metric because the predicted effect is about correctly placing chains relative to one another; per-chain metrics would hide exactly the inter-chain errors the method targets.

### Expected outcome and causal chain

**vs. Calpha single-channel interface** — On a heteromer where two chains share high sequence similarity but occupy different structural roles, the single-channel interface produces states that the frozen head cannot disambiguate, so it swaps or misplaces the similar chains and whole-complex TM-score drops. Our method instead tags each chain with a learned identity embedding and pools cross-chain interface context, giving the head a signal that distinguishes the two chains, so we expect a clear gap on similar-chain heteromers with near-parity on single-chain and clearly distinct-chain cases.

**vs. Concatenated multi-chain sequence** — On a complex with several short chains, concatenation destroys chain boundaries and the head must infer where one chain ends and the next begins, producing coordinate bleed across junctions and incorrect relative placement. Our method keeps chains as separate bundle entries with explicit identity tags, so the head receives unambiguous chain structure, and we expect the gap to grow with chain count and shrink toward parity on two-chain complexes.

**vs. Fine-tuned SimpleFold head** — On a complex whose interface is well described by sequence co-evolution but poorly captured by the frozen head's prior, the unfrozen baseline can absorb interface errors into its weights and recover coordinates that the frozen head cannot. Our method instead relies on the interface-context pool to supply that information through the state schema, and we expect the frozen-head CAPS to trail slightly on such cases while matching or beating the unfrozen baseline where interface information is already in the states, revealing whether the frozen-head claim holds.

**vs. Dedicated multimer predictor (AlphaFold-Multimer)** — On complexes with well-resolved homologous templates, AlphaFold-Multimer is expected to be strong; our method's advantage, if any, should appear on cases where the language prompt carries family-specific constraints not present in the template set. We expect near-parity on template-rich cases and a possible gap on prompt-conditioned cases, and we report both rather than claiming uniform superiority.

**vs. CAPS without permutation-consistency loss** — On homomers where chain order is arbitrary, the ablation memorizes a canonical ordering and produces order-dependent coordinates, so its TM-score is sensitive to input permutation. Our method's consistency loss forces the head's output to follow the same permutation, so we expect the ablation to show a large variance across permuted inputs while CAPS stays stable, with the mean gap concentrated on homomeric and symmetric complexes.

**vs. CAPS with fixed one-hot chain index** — If the learned chain-identity embedding is doing the work, replacing it with a fixed one-hot index should degrade performance on similar-chain heteromers where identity must be inferred from context rather than position. We expect a measurable drop on those cases and near-parity elsewhere, isolating the learned tag's contribution.

**vs. CAPS with mean-pooled chain states** — If cross-chain attention is doing the work, replacing it with mean-pooled chain states should degrade performance on complexes with asymmetric interfaces where per-residue cross-chain contacts matter. We expect a drop concentrated on heteromers with non-uniform interface geometry, isolating the interface-context pool's contribution.

**Phase 0 calibration and permutation-drift diagnostic** — We expect the naively stacked bundle to produce non-degenerate coordinates for at least a majority of residues if the frozen head's input contract tolerates the extra channels; if it does not, the adapter path is invoked and reported. The inference-time permutation-drift diagnostic is expected to show near-zero drift for CAPS and non-zero drift for the no-consistency-loss ablation; if CAPS also shows non-zero drift, the frozen head's positional encoding is leaking chain index and the equivariance claim is unsupported.

### What would falsify this idea

If CAPS's gain over the single-channel Calpha interface is uniform across all complex types rather than concentrated on similar-chain heteromers and high-chain-count cases where the predicted failure modes occur, then the chain-aware schema is not doing the causal work claimed and the improvement comes from something else. Likewise, if the permutation-consistency ablation shows no variance across permuted inputs, the equivariance mechanism is not being exercised and the central structural claim is unsupported; and if the Phase 0 calibration fails even with the adapter, the frozen-head claim is falsified and the contribution reduces to a per-chain single-channel reuse.

## References

1. Calpha: Toward Native Biomolecular Intelligence in a General-Purpose Language Model
