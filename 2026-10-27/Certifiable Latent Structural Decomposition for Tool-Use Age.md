# Certifiable Latent Structural Decomposition for Tool-Use Agent Supervision

## Motivation

Existing methods like Trajectory2Gym assume that tool-use competence can be partitioned into a finite, enumerable set of separable compositional observation-action structures, and they target under-supported structures without verifying that this decomposition is identifiable from learner-visible observations or that the decomposed components are jointly consistent. This assumption fails structurally because the latent structural grammar generating tool-use trajectories may only be identifiable up to a group action (e.g., permutation or scaling of components), and without a joint-consistency test, synthesized interactions can reinforce inconsistent decompositions, leading to supervision that does not transfer.

## Key Insight

A measurable invariant computed from execution traces—the residual of a joint-consistency test across synthesized interactions—certifies whether a latent structural decomposition is identifiable and jointly consistent, thereby gating admissible supervision without relying on unverifiable separability assumptions.

## Method

**What it is:** We introduce CERT-D (Certifiable Decomposition), a framework that learns a latent structural grammar from tool-use execution traces and uses a structure certificate—a measurable invariant—to decide which synthesized interactions are admissible for structural-support selection and parent-normalized SFT weighting. Inputs: a set of execution traces (observation-action sequences) from a base policy; outputs: a certified set of compositional structures and a filtered supervision dataset.

**How it works:**
```pseudocode
1. Initialize latent grammar G with K components (K=8), each component c having a continuous embedding e_c (dim=64) and a discrete operation type o_c from a fixed set (e.g., 'sequence', 'choice', 'loop').
2. For each execution trace τ, parse it into a parse tree T_τ using G via a chart parser that scores trees by the product of component likelihoods.
3. For each pair of synthesized interactions (τ_i, τ_j) that target the same structure, compute the joint-consistency residual:
   r_ij = || φ(T_τ_i) - φ(T_τ_j) ||_2, where φ extracts a vector of component usage counts from the parse tree.
   [Load-bearing assumption A: r_ij is a valid, group-action-invariant measure of structural inconsistency, so that a small max-min ratio C certifies identifiability and joint consistency of the latent grammar. This assumption fails when the group action is large (e.g., permutation of components with similar usage counts), in which case r_ij reflects permutation differences rather than true inconsistency.]
4. Compute the structure certificate C = max_{i,j} r_ij / (1 + min_{i,j} r_ij) (a normalized max-min ratio).
5. If C ≤ 0.1 (threshold), the decomposition is certified as identifiable and jointly consistent; else, reject the synthesized interactions and re-estimate G via EM with a penalty on inconsistent pairs.
6. For certified interactions, compute parent-normalized SFT weights w_τ = 1 / (1 + N_parent(τ)), where N_parent(τ) is the number of interactions sharing the same parent structure, and train the policy with weighted SFT.
```

**Why this design:** We chose a latent grammar with continuous embeddings and discrete operation types over a predefined taxonomy (like T-Eval's six sub-domains) because the latter assumes enumerability, whereas our grammar can represent an open-ended set of structures, accepting the cost of a more complex parser. We chose a joint-consistency residual based on parse-tree usage vectors over a learned independence head because usage vectors are directly computable from traces without additional assumptions, and they capture coupling between components that independence heads cannot manufacture; the trade-off is that usage vectors ignore sequential order, which we mitigate by using a chart parser that enforces order constraints. We chose a max-min ratio certificate over a simple average residual because the max-min ratio is sensitive to worst-case inconsistency, which is critical for gating supervision; this makes the certificate conservative, potentially rejecting some admissible interactions, but we prefer false negatives to false positives. Finally, we chose parent-normalized SFT weights as in Trajectory2Gym but gated by the certificate, because without certification the weights may amplify inconsistent structures; the cost is that we may discard some useful but inconsistent interactions.

**Why this measures what we claim:** The joint-consistency residual r_ij measures the structural inconsistency between two synthesized interactions because we assume that if the latent grammar is identifiable up to a group action, then parse trees of interactions targeting the same structure should have similar component usage vectors up to that group action; this assumption fails when the group action is non-trivial (e.g., component permutation), in which case r_ij reflects permutation differences rather than true inconsistency. The certificate C measures the identifiability and joint consistency of the decomposition because we assume that a small max-min ratio implies that all interactions are consistent up to a bounded group action; this assumption fails when the group action is large, in which case C may be small even for inconsistent decompositions. The parent-normalized weight w_τ measures the relative importance of an interaction because we assume that interactions with fewer siblings are more informative; this assumption fails when sibling count is correlated with difficulty, in which case w_τ reflects difficulty rather than informativeness.

## Contribution

['A certifiable latent structural decomposition framework (CERT-D) that learns a latent grammar from execution traces and computes a structure certificate—a measurable invariant—to gate synthesized interactions for supervision.', 'A joint-consistency residual based on parse-tree component usage vectors that empirically distinguishes identifiable decompositions from inconsistent ones without requiring predefined separable components.', 'A gated parent-normalized SFT weighting scheme that uses the certificate to filter supervision, which we show improves transfer on under-supported structures compared to ungated weighting.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Tau2 compositional-structure split | Under-supported structures expose decomposition gaps |
| Primary metric | Certified-structure success rate | Directly tests identifiability-gated supervision |
| Baseline | Trajectory2Gym weighted SFT | Same weighting, no certificate gating |
| Baseline | Vanilla SFT on base traces | Tests value of synthesized interactions at all |
| Baseline | Predefined-taxonomy (T-Eval) decomposition | Tests open-ended grammar vs enumerable taxonomy |
| Ablation-of-ours | CERT-D without certificate gate | Isolates certificate's contribution |
| Calibration metric | Pearson correlation between C and downstream structural error rate | Validates that C tracks structural error (assumption A) |

### Why this setup validates the claim

The central claim is that a structure certificate gating which synthesized interactions enter parent-normalized SFT prevents the amplification of inconsistent structures, so gains should concentrate on under-supported compositional structures. Trajectory2Gym weighted SFT tests the sub-claim that parent-normalized weighting alone is insufficient without certification: if our method beats it, the certificate—not the weighting—is doing the work. Vanilla SFT tests whether synthesized interactions help at all, establishing the floor. The predefined-taxonomy baseline tests whether an open-ended latent grammar beats an enumerable taxonomy on structures the taxonomy cannot name. The no-certificate ablation is the cleanest test: it shares our grammar, parser, and weighting, differing only in gating, so any gap isolates the certificate's causal contribution. Certified-structure success rate is the right metric because it measures performance exactly where the predicted failure mode (inconsistent structure amplification) should bite, rather than being diluted by easy structures where all methods saturate. Additionally, we measure the Pearson correlation between C and downstream structural error rate on a held-out set of 512 synthesized interactions to validate assumption A: if C does not correlate with structural error, the certificate is not measuring identifiability.

### Expected outcome and causal chain

**vs. Trajectory2Gym weighted SFT** — On a case where a synthesized interaction pairs a valid tool sequence with a subtly wrong loop-bound structure, Trajectory2Gym computes parent-normalized weights from sibling counts alone and upweights it because it is a rare sibling, so the policy learns the wrong structure. Our method computes a joint-consistency residual against other interactions targeting the same structure, the certificate exceeds threshold, and the interaction is rejected before weighting. We expect a noticeable gap on the under-supported-structure subset and near-parity on well-supported structures.

**vs. Vanilla SFT on base traces** — On a case requiring a compositional structure absent from base traces, vanilla SFT has no supervision signal for that structure and defaults to the closest seen structure, producing a plausible but structurally wrong plan. Our method synthesizes interactions targeting that structure, certifies them for consistency, and supplies targeted supervision. We expect vanilla SFT to fail broadly on under-supported structures while our method succeeds there, with the gap widening as structure rarity increases.

**vs. Predefined-taxonomy (T-Eval) decomposition** — On a case whose structure combines operations in a way the six-sub-domain taxonomy cannot express, the taxonomy baseline forces the trace into the nearest enumerated category, mislabeling it and supervising the wrong decomposition. Our latent grammar assigns it to a component with a continuous embedding and discrete operation type that fits the actual usage pattern. We expect the taxonomy baseline to degrade specifically on out-of-taxonomy structures while matching us on in-taxonomy ones.

### What would falsify this idea

If our gain over the no-certificate ablation is uniform across subsets rather than concentrated on under-supported structures, or if the certificate rejects interactions at a rate uncorrelated with downstream structural errors (Pearson correlation between C and structural error rate < 0.3), then the certificate is not measuring identifiability and the central claim is wrong.

## References

1. Trajectory2Gym: Decision-Grounded Compositional Interaction Synthesis for Tool-Use Agents
2. Scaling Agent Learning via Experience Synthesis
3. ReTool: Reinforcement Learning for Strategic Tool Use in LLMs
4. Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents
5. WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning
