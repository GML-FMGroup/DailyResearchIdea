# Do-Calculus Attention Masks for Counterfactual-Consistent Interactive Video Generation

## Motivation

Interactive video world models built on In-Context Conditioning, such as World-in-Context, tokenize actions, cameras, and views into a shared self-attention context and rely on FullDiT2-style correlation-based attention masks; these masks encode statistical co-occurrence between control tokens and latent video tokens, not the causal effect of intervening on a control. As a result, when a user issues a counterfactual intervention (e.g., 'what if the camera turned left instead of right at this step'), the generated frames condition on the observed action history rather than on the intervention, so the model reproduces the correlated past trajectory instead of the do(action) distribution. The structural root cause is that the mask is a learned/engineered correlation graph, and correlation-based conditioning is not invariant to the do-operator: it has no mechanism to sever incoming edges to the intervened action token and re-route its effect forward to the latent state.

## Key Insight

Replacing the correlation mask with a do-calculus invariant mask—constructed by deleting all parent edges into the intervened action token and retaining only forward edges from the intervention to the latent state—makes the attention computation compute the interventional distribution P(video | do(action)) rather than the observational P(video | action), because the mask topology directly encodes the graph surgery that the do-operator performs.

## Method

**What it is.** We introduce **DoMask**, a causal attention-mask construction that replaces FullDiT2's correlation-based mask inside an In-Context Conditioning video world model. Inputs: a token sequence of view tokens V, action tokens A, camera tokens C, and latent video tokens L, plus a per-step intervention set I specifying which action/camera tokens the user counterfactually overrides. Output: a binary attention mask M_do and the generated latent video, where M_do enforces the post-intervention causal graph.

**Load-bearing assumption (stated explicitly).** DoMask assumes that the frozen DiT backbone's learned conditional factorizes according to the hand-specified typed graph G_typed, so that restricting self-attention to the post-surgery mask M_do is equivalent to probabilistic do-surgery on the model's internal causal structure. This assumption is not verified by construction: the backbone was trained by maximum likelihood on observational data under a correlation factorization, so severing parents of an intervened token produces an activation pattern that may be off-manifold. We therefore add a calibration/verification step (block E) that measures whether the masked forward pass reproduces held-out do(action) rollouts before any interventional claim is made.

**How it works.** We build a per-token-type causal graph G with typed edges, then apply do-calculus graph surgery to produce M_do.

```pseudocode
def build_do_mask(tokens, I, G_typed, kappa=2):
    # G_typed: typed DAG over token types
    #   V -> L, C -> L, A -> L, A -> A_next, C -> C_next,
    #   V -> A, V -> C, L_{t-1} -> L_t, L_{t-1} -> A_t
    # I: set of intervened token indices (counterfactual overrides)
    M = adjacency_to_mask(G_typed, tokens)          # baseline typed causal mask
    for j in I:
        # 1. Graph surgery: sever all incoming edges to intervened node j
        M[:, j] = 0                                 # delete parents of do-node
        M[j, j] = 1                                 # self-attention retained
        # 2. Re-route intervention forward: j -> descendants within kappa hops
        for d in descendants(G_typed, j, hops=kappa):
            M[j, d] = 1                             # forward edge from do-node
    # 3. Enforce acyclicity: remove any edge that closes a cycle after surgery
    M = break_cycles(M, order=topological_order(G_typed))
    return M
```

The generated video is produced by running the frozen DiT backbone with attention restricted to M_do. The intervention set I is supplied by the interactive controller at each execution step; when I = empty, M_do reduces to the typed causal mask and the model behaves as standard In-Context Conditioning.

**Calibration and verification of the load-bearing assumption (block E).** Before reporting any interventional metric, we run two concrete checks on the frozen backbone with M_do:

1. **Edge-effect validation via activation patching.** For each typed edge e in G_typed, we measure its causal effect on the next latent state by interchange intervention: run the backbone on a base input, then patch the activation of the source token type with the activation from a counterfactual input, and record the change in the generated latent L_t (L2 distance in latent space). Edges whose measured effect is below a null threshold (95th percentile of effects from randomly permuted source tokens, computed over 512 calibration pairs) are pruned from G_typed before surgery. This yields a validated graph G_validated, and M_do is built from G_validated rather than the asserted G_typed.
2. **Masked-vs-reconditioned calibration.** On a held-out set of 512 do(action) rollouts (intervention pairs from the synthetic SCM dataset, see experiment), we compare the masked forward pass output to the output of a re-conditioned forward pass that explicitly feeds the intervened action as the only action context (the on-manifold interventional input). We report the latent L2 gap and the action-following accuracy gap. If the masked output does not match the re-conditioned output within a pre-registered tolerance (latent L2 < 0.05 and accuracy gap < 5 points), the backbone is deemed off-manifold under surgery, and we report this as a failure of the load-bearing assumption rather than as evidence for DoMask.

**Why this design.** We chose typed graph surgery over a learned attention mask because a learned mask re-introduces correlation (the very failure of FullDiT2), whereas surgery is a discrete, auditable operation whose output is invariant to the observed action history—accepting the cost that the graph must be hand-specified per token type and cannot discover unmodeled edges. We chose to retain self-attention on intervened nodes and add forward edges within kappa=2 hops rather than fully re-running the backbone, because re-running would break the IT-RoPE accumulated-transition encoding that World-in-Context relies on for temporal continuity—accepting the cost that effects propagating further than two hops are only captured through the latent-state chain L_{t-1}->L_t. We chose cycle-breaking by topological order rather than a soft mask penalty because soft penalties leave residual correlation leakage—accepting the cost that some legitimate long-range edges may be pruned. A domain expert would not describe this as a variant of a learned causal-discovery attention head (e.g., the correlation-mask line of FullDiT2) because the mask here is not estimated from data at all: it is the deterministic image of a do-operator on a fixed typed graph, and its correctness is a property of the graph surgery, not of a fitting procedure.

**Why it measures what we claim.** The mask entry M[i,j]=0 for j in I measures *severance of the observational parent set* because the do-operator is defined as deleting incoming edges to the intervened variable, under the assumption that the typed graph G_typed is causally sufficient (no unmeasured confounders between action tokens and latent states); this assumption fails when an unmodeled confounder (e.g., a scene property that jointly drives both the user's action choice and the next frame) is present, in which case M_do reflects a partially adjusted interventional distribution rather than the true do(action) effect. The forward edges M[j,d]=1 for d in descendants within kappa hops measure *propagation of the counterfactual intervention* because in a causal DAG the effect of do(X=x) on Y is exactly the flow along directed paths from X to Y, under the assumption that the typed graph's directed paths are the only causal channels; this assumption fails when the true causal path exceeds kappa hops, in which case the metric reflects a truncated intervention effect. The cycle-breaking step measures *consistency of the post-surgery graph* because a valid do-graph must remain a DAG, under the assumption that the pre-surgery typed graph is acyclic; this assumption fails when IT-RoPE's accumulated-transition edges create a back-edge, in which case the pruned mask reflects the largest acyclic subgraph rather than the full interventional graph. The calibration check in block E measures *whether the frozen backbone's masked forward pass equals the interventional conditional* because it compares the masked output to an explicitly re-conditioned on-manifold output, under the assumption that the re-conditioned pass is a faithful proxy for P(video | do(action)); this assumption fails when the re-conditioned pass itself is off-distribution, in which case the calibration gap reflects a mismatch between two off-manifold regimes rather than a validated equivalence.

## Contribution

(1) DoMask, a do-calculus invariant attention-mask constructor that performs explicit graph surgery on a typed token DAG, replacing correlation-based masking in In-Context Conditioning video world models with a deterministic interventional mask. (2) The design principle that counterfactual consistency in interactive video generation is a property of mask topology—severance of parent edges plus forward re-routing—rather than of training data or a learned causal head, yielding a training-free modification that preserves IT-RoPE temporal continuity. (3) A typed token-graph specification (view, action, camera, latent) with a kappa-hop propagation parameter that makes the intervention-effect horizon an explicit, tunable design variable rather than an emergent property.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset (core causal validation) | Synthetic SCM-generated video dataset with known ground-truth do(action) frames (512 calibration pairs + 2,048 test pairs) | Provides exact interventional targets for the load-bearing assumption. |
| Dataset (transfer check) | WBench navigation subset with user action/camera overrides | Interactive counterfactual overrides are native to this task. |
| Primary metric | Counterfactual action-following accuracy under intervention | Directly measures do(action) propagation, not correlation. |
| Secondary metric | Latent L2 gap between masked and re-conditioned outputs (block E calibration) | Tests whether surgery yields on-manifold interventional outputs. |
| Baseline | World-in-Context-5B (ICC, no surgery) | Tests whether typed surgery beats plain in-context conditioning. |
| Baseline | FullDiT2 correlation-mask variant | Tests whether discrete surgery beats learned correlation masking. |
| Baseline | 22B Ec model | Tests whether a large scale-only model closes the gap. |
| Baseline | Learned-graph causal-discovery attention head (estimates typed graph from data on same backbone) | Tests whether deterministic surgery beats a learned graph. |
| Ablation-of-ours | DoMask with kappa=0 (severance only, no forward edges) | Isolates the contribution of intervention re-routing. |
| Ablation-of-ours | DoMask with kappa sweep {0,1,2,3,full} | Locates where kappa-truncation bites. |
| Ablation-of-ours | DoMask with deliberately added/removed typed edge (graph misspecification) | Bounds sensitivity to hand-specification. |

### Why this setup validates the claim

The central claim is that a deterministic do-operator mask severs observational parents and propagates counterfactuals more faithfully than either plain in-context conditioning or a learned correlation mask. The synthetic SCM-generated video dataset is the right core testbed because it supplies exact ground-truth do(action) frames, so the load-bearing assumption—that the masked forward pass equals the interventional conditional—can be tested directly via the block E calibration (latent L2 gap and accuracy gap between masked and re-conditioned outputs) rather than assumed. WBench's navigation subset is reserved for a smaller transfer check because it supplies explicit action and camera overrides, so the intervention set I is well-defined and the ground-truth post-intervention video is observable, but it does not provide the exact interventional targets needed to validate the assumption. The primary metric—accuracy of following the overridden action/camera under intervention—is chosen because it is exactly the quantity the do-mask is supposed to control: a correlation mask can score well on observational next-frame prediction while failing here, so this metric discriminates the mechanisms rather than rewarding scale. World-in-Context-5B tests sub-claim (a): that typed surgery beats no surgery. The FullDiT2 correlation-mask variant tests sub-claim (b): that a discrete, auditable surgery beats a learned mask that re-introduces correlation. The 22B Ec baseline tests sub-claim (c): that the gain is mechanistic, not merely a scale effect. The learned-graph causal-discovery baseline tests that the mechanism is not reducible to a static sparse mask: if DoMask's deterministic surgery does not beat a learned graph on the same backbone, the novelty claim is weakened. The kappa=0 ablation isolates whether forward re-routing (not just severance) drives the predicted gain, and the kappa sweep {0,1,2,3,full} empirically locates where truncation bites, making the design falsifiable: if severance alone matches full DoMask, the propagation claim is unsupported.

### Expected outcome and causal chain

**vs. World-in-Context-5B (ICC, no surgery)** — On a case where the controller overrides the action token to turn left while the observed history implies a right turn, the baseline produces a right-turning frame because its causal mask still lets the historical action tokens attend into the latent state, so the override is diluted by observational history. Our method instead zeroes incoming edges to the intervened action token and adds forward edges within two hops, so the override propagates to the next latent state uncontaminated. We expect a noticeable gap on override-heavy navigation segments but near-parity on segments with no intervention, since M_do reduces to the typed mask when I is empty. On the synthetic SCM dataset, we additionally expect the block E calibration to show a latent L2 gap below 0.05 between masked and re-conditioned outputs; if it does not, the load-bearing assumption is reported as failed.

**vs. FullDiT2 correlation-mask variant** — On a case where a scene property (e.g., a wall) jointly drives both the user's action choice and the next frame, the baseline's learned correlation mask keeps that confounder edge active, so the generated video follows the spurious correlation rather than the do(action). Our method instead deletes all incoming edges to the intervened node by graph surgery, which is invariant to observed action history, so the confounder cannot leak through the intervened path. We expect the gap to concentrate on confounded scenes (cluttered corridors) and to shrink toward parity on unconfounded open scenes, which is the diagnostic signature of severance working.

**vs. 22B Ec model** — On a case where the user issues a counterfactual camera override mid-trajectory, the 22B baseline produces a plausible but override-ignoring trajectory because scale improves observational fidelity but does not change the conditioning mechanism, which still treats overrides as correlated context. Our method instead applies the same deterministic surgery regardless of backbone size, so the override is enforced structurally. We expect DoMask-5B to match or exceed the 22B baseline specifically on intervention segments while trailing it on non-intervention fidelity, showing the gain is mechanistic rather than scale-driven.

**vs. Learned-graph causal-discovery attention head** — On the same backbone, the learned-graph baseline estimates the typed graph from data and may recover spurious edges under confounding, whereas DoMask applies deterministic surgery on the validated graph. We expect DoMask to outperform the learned-graph baseline on the confounded subset and to be comparable on the unconfounded subset, showing that deterministic surgery yields better-calibrated interventional behavior than a learned graph.

### What would falsify this idea

If DoMask's gain over the correlation-mask baseline is uniform across confounded and unconfounded subsets rather than concentrated where the predicted confounder leakage occurs, the severance claim is wrong. Likewise, if the kappa=0 ablation matches full DoMask on long-horizon overrides, the forward-propagation claim is unsupported and the method reduces to mere edge deletion. Additionally, if the block E calibration shows a latent L2 gap above 0.05 or an accuracy gap above 5 points between masked and re-conditioned outputs on the synthetic SCM dataset, the load-bearing assumption that masking equals probabilistic do-surgery is falsified.

## References

1. World-in-Context: Interactive Video World Modeling with In-Context Conditioning
