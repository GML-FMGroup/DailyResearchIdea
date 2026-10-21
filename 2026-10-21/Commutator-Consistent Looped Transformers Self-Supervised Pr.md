# Commutator-Consistent Looped Transformers: Self-Supervised Progress via Monoid Homomorphism

## Motivation

Progress-aligned recurrence methods such as PAR require a hand-specified discrete operation decomposition to define prefix states, so they cannot be applied to state-tracking tasks whose iterative structure is not already known. This limitation recurs across the field: every progress signal in PAR, looped Transformers, and implicit-CoT training is defined relative to a human-supplied step decomposition, making alignment unavailable for tasks where the decomposition must be discovered. The structural gap is that no existing method infers the task's transition monoid from input-output pairs and ties recurrent latent updates to it.

## Key Insight

If the recurrent latent update is constrained to be a monoid homomorphism—i.e., the latent update for a composed operation equals the composition of latent updates for its parts—then the iteration index automatically aligns with the algebraic depth of the operation, yielding a progress signal without any hand-specified decomposition.

## Method

**What it is.** We introduce **MonoLoop**, a looped transformer trained with a self-supervised commutator consistency loss that forces its recurrent latent state to be a homomorphism from the task's transition monoid to a learned latent monoid. Inputs are input-output pairs of a state-tracking task (swap-S₅, graph reachability, Dyck-2 with bounded stack depth D=8); outputs are iteration-wise latent states whose composition respects the monoid structure, plus a final answer. No prefix decomposition or step count is provided during training.

**How it works.**

```python
# Hyperparameters: L=16 loop iterations, d=256 latent dim, λ_comm=1.0, λ_pred=1.0, λ_norm=0.1
# MonoidOp: 2-layer MLP, hidden=256, GeLU, residual connection: MonoidOp(a,b) = b + MLP([a;b])
# Readout: linear probe on h_k (dim 256 -> |states|)
for each episode (x, y) from task distribution:
    # x: sequence of operations; y: final state
    h_0 = embed(x[0])                    # initial latent state
    for k in 1..L:
        h_k = LoopBlock(h_{k-1}, x[k])   # shared transformer block, no step index given
    # --- self-supervised commutator consistency ---
    # Sample two random cut points a < b < L
    # Compute latent updates for segments [0:a], [a:b], [b:L]
    Δ1 = h_a - h_0
    Δ2 = h_b - h_a
    Δ3 = h_L - h_b
    # Commutator of consecutive updates: [Δ1, Δ2] = Δ1∘Δ2 - Δ2∘Δ1 (composition via learned monoid op)
    # We enforce that the latent update for the concatenation [0:b] equals the composition of Δ1 and Δ2
    comm_loss = || (h_b - h_0) - MonoidOp(Δ1, Δ2) ||^2
    # Similarly for [a:L] = [a:b] ∘ [b:L]
    comm_loss += || (h_L - h_a) - MonoidOp(Δ2, Δ3) ||^2
    # --- triple-consistency diagnostic (logged, not backpropagated) ---
    # Sample three random cut points a < b < c < L
    # Compute Δa = h_a - h_0, Δb = h_b - h_a, Δc = h_c - h_b
    # triple_loss = || MonoidOp(MonoidOp(Δa, Δb), Δc) - MonoidOp(Δa, MonoidOp(Δb, Δc)) ||^2
    # --- prediction loss ---
    pred_loss = CE(Readout(h_L), y)
    # --- norm anchoring (from PAR) ---
    norm_loss = || ||h_k|| - ||h_0|| ||^2 for all k
    total_loss = λ_pred * pred_loss + λ_comm * comm_loss + λ_norm * norm_loss
    update parameters
```

MonoidOp is a learned associative binary operation on latent updates, parameterized as a 2-layer MLP (hidden=256, GeLU) with a residual connection `MonoidOp(a,b) = b + MLP([a;b])` that is trained to be associative via the commutator loss itself (no separate associativity penalty). The loop block is a standard transformer layer with shared weights across iterations. At inference, the model runs until a learned completion readout (a linear probe on h_k) signals convergence, but the number of iterations is not externally specified.

**Why this design.** We chose a self-supervised commutator loss over PAR's prefix-aligned targets because PAR requires a known discrete decomposition to define prefix states, whereas the commutator loss only requires that the latent updates compose consistently—a property that holds for any transition monoid. We chose a learned associative MonoidOp over a fixed group operation (e.g., matrix multiplication) because state-tracking tasks include non-group monoids (e.g., Dyck languages with stack operations), and a learned op can approximate any finite monoid; the cost is that associativity must be enforced implicitly by the commutator loss rather than guaranteed by construction. We chose to anchor norms to the first iteration (following PAR) over unconstrained latent norms because looped transformers otherwise suffer from norm explosion across iterations, which would destroy the commutator signal; the trade-off is that norm anchoring may limit the expressiveness of latent updates that require large magnitude changes. We chose a learned completion readout over a fixed iteration count because the monoid homomorphism property implies that the latent state converges when the operation reaches a fixed point, but the convergence rate depends on the task, so a learned readout adapts to the monoid's structure. A domain expert would not describe MonoLoop as a variant of PAR because PAR's progress signal is defined by a human-specified prefix decomposition, while MonoLoop's progress signal is an emergent property of monoid-homomorphism consistency, requiring no decomposition.

**Why it measures what we claim.** The commutator loss `|| (h_b - h_0) - MonoidOp(Δ1, Δ2) ||^2` measures **monoid-homomorphism consistency** because the assumption that the latent update for a concatenated operation equals the composition of latent updates for its parts is exactly the definition of a monoid homomorphism; this assumption fails when the task's transition monoid is not finitely generated by the observed operations (e.g., tasks with unbounded memory like general context-free grammars), in which case the commutator loss reflects only local consistency and does not guarantee global homomorphism. The norm anchoring loss measures **state-scale stability** because the assumption that the first-iteration norm is a canonical scale for the latent state holds when the initial state is drawn from a stationary distribution; this assumption fails when the task has non-stationary initial states (e.g., graph reachability with varying source nodes), in which case norm anchoring may bias the latent representation toward the training distribution's scale. The learned completion readout measures **convergence of the latent trajectory** because the assumption that the latent state reaches a fixed point when the operation reaches a fixed point follows from the monoid homomorphism property; this assumption fails when the monoid has cycles without fixed points (e.g., permutation groups with no identity element in the observed operations), in which case the readout may signal convergence prematurely. The triple-consistency diagnostic `|| MonoidOp(MonoidOp(Δa, Δb), Δc) - MonoidOp(Δa, MonoidOp(Δb, Δc)) ||^2` measures **global associativity of MonoidOp on sampled triples** because the assumption that pairwise consistency on consecutive segments extends to all triples holds when the operation set generates the monoid; this assumption fails when MonoidOp satisfies sampled cuts but violates associativity on triples, in which case the diagnostic reveals a non-associative MonoidOp that would break the homomorphism claim.

## Contribution

["A self-supervised commutator consistency loss that trains a looped transformer's recurrent latent state to be a monoid homomorphism, eliminating the need for hand-specified discrete operation decompositions in progress-aligned recurrence.", "A learned associative MonoidOp module that composes latent updates, enabling the model to discover the task's transition monoid structure from input-output pairs alone.", 'An empirical demonstration that monoid-homomorphism consistency yields automatic progress alignment, as evidenced by length generalization on state-tracking tasks without any prefix decomposition.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | swap-S₅ and Dyck-2 (bounded stack depth D=8), train ≤32 moves | Finite monoids with known transition structure |
| Primary metric | Full-state accuracy at 16× training length | Directly tests length generalization of state tracking |
| Baseline | RLT (looped, no consistency loss) | Tests whether looped recurrence alone suffices |
| Baseline | PAR (prefix-aligned targets) | Tests whether human-specified decomposition is necessary |
| Baseline | Chain-of-thought (explicit) | Tests whether latent recurrence beats verbalized steps |
| Baseline | Fixed-group MonoidOp (matrix multiplication in GL(d)) | Isolates learned-associative-op vs. commutator loss |
| Ablation-of-ours | MonoLoop without commutator loss | Isolates monoid-homomorphism consistency from looping |
| Diagnostic | Triple-consistency loss (logged) | Checks global associativity of MonoidOp on triples |

### Why this setup validates the claim

The claim is that a self-supervised commutator loss forcing latent updates to compose as a monoid homomorphism yields length generalization without any human-specified prefix decomposition. Each baseline isolates one competing explanation. RLT uses the same looped backbone and adaptive steps but lacks the consistency signal, so any gap over RLT attributes the effect to the commutator loss rather than to looping per se. PAR attains progress signals via human-defined prefix states; if MonoLoop matches or beats PAR without that supervision, it shows the decomposition is not necessary — the central novelty. Explicit chain-of-thought tests whether the benefit is really about latent monoid structure rather than just more computation. The fixed-group MonoidOp baseline replaces the learned MLP with a fixed matrix multiplication in GL(d), keeping the commutator loss identical; if MonoLoop matches this baseline, the learned-associative-op component is not the contributor, and the commutator loss alone suffices. The ablation (MonoLoop minus commutator loss) is the sharpest test: it keeps the architecture, norm anchoring, and learned completion readout identical, so any collapse in out-of-length accuracy must be caused by the missing homomorphism constraint. The triple-consistency diagnostic is logged (not backpropagated) to verify that the sampled-cut commutator loss does not yield a non-associative MonoidOp that satisfies pairwise cuts but violates associativity on triples. Full-state accuracy at 16× training length is the right metric because the mechanism predicts accurate state tracking precisely where transitions must compose over many unseen operations; a metric like final-answer-only accuracy would confound state-tracking with readout luck. Together these form a falsifiable test: the mechanism predicts gains concentrated on long sequences and on tasks whose transition monoid is finitely generated, with parity on short sequences.

### Expected outcome and causal chain

**vs. RLT (looped, no consistency loss)** — On a swap-S₅ instance of 200 moves, RLT's latent state drifts because nothing ties the iteration-k update to a composable monoid element, so errors accumulate and full-state accuracy collapses toward chance at long lengths. Our method instead enforces that the update over any concatenation equals the MonoidOp composition of its parts, so the latent state tracks the true permutation regardless of length. We expect a large gap at 16× length (our method near-ceiling, RLT far below) but parity on short sequences, confirming the gain is a length-generalization effect rather than a capacity effect.

**vs. PAR (prefix-aligned targets)** — On a Dyck-2 instance requiring a deep stack (bounded depth D=8), PAR's progress signal is defined by a human-specified prefix decomposition that must be computed for each task; on non-group monoids like the stack monoid this decomposition is awkward or ill-defined, so PAR's progress targets misalign with the true transition structure and its long-length accuracy degrades. Our method instead derives consistency from the commutator alone, requiring no decomposition, so MonoidOp can approximate the stack monoid's composition. We expect MonoLoop to match PAR on swap-S₅ (a group, where PAR's decomposition is natural) but pull ahead on Dyck-2 at long lengths, with the gap widening as length grows.

**vs. explicit chain-of-thought** — On a 512-move swap-S₅ instance, explicit CoT must verbalize each of 512 intermediate states, and any single verbalization error (or context-length limit) derails the trajectory, so accuracy is near floor at long lengths. Our method instead carries the state in a recurrent latent that is constrained to compose consistently, so it does not pay per-step verbalization cost and its errors do not compound the same way. We expect a dramatic gap at 512 moves (our method substantially above CoT) while both are comparable at training length.

**vs. fixed-group MonoidOp (matrix multiplication in GL(d))** — On a swap-S₅ instance of 200 moves, the fixed-group baseline uses a hard-coded matrix multiplication as MonoidOp, which is exactly associative by construction but may not match the task's monoid if the task is non-group (e.g., Dyck-2 stack monoid). Our method uses a learned MLP MonoidOp that can approximate any finite monoid. We expect MonoLoop to match the fixed-group baseline on swap-S₅ (a group) but outperform it on Dyck-2 at long lengths, isolating the learned-associative-op component as the contributor for non-group tasks.

**vs. MonoLoop without commutator loss (ablation)** — On a 200-move instance, the ablated model still loops and still uses norm anchoring and the completion readout, but without the commutator loss its latent updates need not compose associatively, so the learned MonoidOp is unconstrained and the latent trajectory diverges from the true monoid action at long lengths. Our full method instead ties every segment update to MonoidOp composition, forcing a homomorphism. We expect the ablation to match our method at short lengths but fall off sharply beyond training length, isolating the commutator loss as the causal ingredient. Additionally, we log the triple-consistency loss for both models; we expect the ablated model to show high triple-consistency loss (non-associative MonoidOp) while the full model maintains low triple-consistency loss, confirming that the commutator loss enforces associativity on sampled triples.

### What would falsify this idea

If the commutator-loss ablation matches full MonoLoop at 16× length — i.e., the gain over RLT is uniform across lengths and subsets rather than concentrated on long sequences and on finitely-generated-monoid tasks — then monoid-homomorphism consistency is not doing the work, and the central claim is wrong. Likewise, if MonoLoop fails to beat PAR specifically on Dyck-2 while matching it on swap-S₅, the claim that no human decomposition is needed would be unsupported. Additionally, if the triple-consistency diagnostic shows high loss (non-associative MonoidOp) for the full model despite low commutator loss, the sampled-cut commutator loss does not imply global associativity, falsifying the homomorphism claim.

## References

1. PAR: Progress-Aligned Recurrence for Length Generalization in State Tracking
2. Training Large Language Models to Reason in a Continuous Latent Space
3. Looped Transformers for Length Generalization
4. Guiding Language Model Reasoning with Planning Tokens
5. Think before you speak: Training Language Models With Pause Tokens
6. Looped Transformers as Programmable Computers
7. Teaching Algorithmic Reasoning via In-context Learning
8. Transformers Learn Shortcuts to Automata
