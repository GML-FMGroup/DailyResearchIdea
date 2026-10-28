# Typed Operator Algebra for Composable Task-Conditioned Physical Dynamics

## Motivation

Current VLA models such as PDRVLA condition action generation on a monolithic flat set of physical tokens, so the backbone must implicitly entangle all task factors and generalize to composite tasks only if the composite appeared during training. This flat conditioning interface lacks any operator for composing or hierarchically factoring known sub-task dynamics, making zero-shot composition of known sub-dynamics structurally impossible without retraining on the composite. The meta-gap recurs across paper-trees: BC-Z likewise relies on a single task embedding with no algebraic structure over the conditioning signal, so unseen task combinations are handled only by similarity in embedding space rather than by exact composition.

## Key Insight

If the task grammar's syntactic operators are constrained to act as a homomorphism on the physical-token space—i.e., the token transformation for a composite expression equals the monoid composition of the token transformations for its sub-expressions—then composite task dynamics are exact by construction and require no retraining on the composite.

## Method

**What it is.** TOPA (Typed Operator Physical Algebra) parses a task instruction into a typed expression tree over a small operator set {SEQ, PAR, COND, REPEAT}, maps each leaf sub-task to a physical-token transformation via a learned monoid action, and executes the composite by applying the composed transformation to the physical-token conditioning vector before action generation. The load-bearing assumption, stated explicitly: the leaf transformations T_i(p) = p + W_i·phi(p) are elements of a monoid whose composition operation is function composition, and the family {I + W·phi} is closed under that composition—i.e., composing two leaf transforms yields a map of the same form with a composed W. This closure holds exactly when phi is linear (phi(p) = A p for a fixed matrix A), in which case T_i(p) = p + W_i A p = (I + W_i A) p, and composition of two such maps is (I + W_2 A)(I + W_1 A) p = (I + W_2 A + W_1 A + W_2 A W_1 A) p, which is again of the form (I + W' A) p with W' = W_2 + W_1 + W_2 A W_1. We therefore fix phi to be a linear projection A ∈ R^{d×d} (d = physical-token dimension, 256) initialized to identity and frozen after the first 1k steps, so the leaf family is closed under composition by construction. PAR and COND are NOT homomorphisms of this monoid: the convex mixture alpha·T_e1 + (1-alpha)·T_e2 and the gated sum g_c·T_e1 + (1-g_c)·T_e2 are affine combinations, not compositions, and do not satisfy T_{PAR(e1,e2)} = T_e1 ∘ T_e2. We therefore restrict the 'exact composition' claim to the SEQ/REPEAT sub-monoid (which is closed and associative by construction) and re-frame PAR and COND as approximate soft operators with an explicit error bound: for any p, ||T_PAR(p) - (T_e1 ∘ T_e2)(p)|| ≤ (1/2)||T_e1(p) - T_e2(p)|| + (1/2)||T_e1(p) - T_e2(p)|| = ||T_e1(p) - T_e2(p)||, and similarly for COND with g_c in place of alpha; these bounds are logged as law-residual diagnostics (see block D). The homomorphism loss L_hom is enforced only on SEQ/REPEAT pairs; PAR/COND pairs are trained with L_task and L_gate only, and their law residuals are reported separately.

**How it works.**

```pseudocode
# Input: instruction x, observation o, sub-task library L = {(s_i, T_i)} where T_i: P -> P
# Output: action a
# Hyperparameters: d = 256 (physical-token dim), phi(p) = A p with A ∈ R^{d×d} frozen after 1k steps

1. Parse x into typed expression tree E with operators {SEQ, PAR, COND, REPEAT}
   # leaves are sub-task symbols s_i; each internal node has a type (sequence/parallel/conditional/repeat)

2. For each leaf s_i, retrieve its learned monoid element T_i from L
   # T_i is a physical-token transformation: T_i(p) = p + W_i * phi(p), phi(p) = A p (linear, frozen)
   # W_i ∈ R^{d×d} is trained by the homomorphism loss below; closure under composition holds because
   # (I + W_2 A)(I + W_1 A) = I + (W_2 + W_1 + W_2 A W_1) A, so the composed W is W_2 + W_1 + W_2 A W_1

3. Evaluate E bottom-up using the monoid action:
   - SEQ(e1, e2): T = T_e2 o T_e1          (function composition; closed: W = W_e2 + W_e1 + W_e2 A W_e1)
   - PAR(e1, e2): T = alpha * T_e1 + (1-alpha) * T_e2   (convex mixture, alpha=0.5 fixed; NOT a homomorphism)
   - COND(c, e1, e2): T = g_c * T_e1 + (1-g_c) * T_e2  (g_c = sigmoid(MLP(phi(p), c)) learned gate; NOT a homomorphism)
   - REPEAT(e, k): T = T_e^k                (k-fold composition, k from parser; closed under SEQ composition)

4. Compute composite physical tokens p' = T_E(p) where p = PhysicalDynamicsModule(o, x)

5. Generate action a = ActionHead(p', o)

# Training losses:
# L_hom: homomorphism consistency (SEQ/REPEAT only)
#   For each composite training pair (E, E') with same denotation and both using only SEQ/REPEAT,
#   enforce T_E(p) = T_E'(p)
#   L_hom = || T_E(p) - T_{E'}(p) ||^2
# L_task: action imitation loss on composite and atomic tasks
# L_gate: binary cross-entropy on COND gate labels from subtask annotations
# Law-residual diagnostics (logged, not optimized):
#   SEQ associativity residual: ||T_{SEQ(SEQ(A,B),C)}(p) - T_{SEQ(A,SEQ(B,C))}(p)||
#   PAR commutativity residual: ||T_{PAR(A,B)}(p) - T_{PAR(B,A)}(p)||
#   PAR idempotence residual: ||T_{PAR(A,A)}(p) - T_A(p)||
#   COND saturation residual: ||g_c - 1|| when c true, ||g_c|| when c false
```

**Why this design.** We chose a fixed monoid action with function composition for SEQ over a learned recurrent composer because function composition is associative and closed by construction (given linear phi), guaranteeing that any parenthesization of the same sequence yields identical tokens, accepting the cost that the transformation family must be expressive enough to cover all sub-tasks. We chose a convex mixture for PAR with fixed alpha=0.5 over a learned attention mixer because a fixed convex combination is commutative and idempotent, which matches the semantics of parallel sub-tasks and prevents the model from learning spurious ordering, accepting the cost that equal weighting may under-represent a dominant sub-task. We chose a learned gate for COND over a hard symbolic branch because conditional sub-task selection depends on continuous observation features that a discrete parser cannot resolve, accepting the cost that the gate must be supervised with subtask annotations to avoid collapse. We chose to train the homomorphism loss on paired expressions with identical denotation over a purely end-to-end action loss because the action loss alone does not constrain the token transformations to compose correctly—it can fit composites by memorization—accepting the cost that paired expression data must be generated by a grammar rewriter. A domain expert would not describe TOPA as a variant of PDRVLA because PDRVLA conditions on a flat token set with no operator algebra, and would not describe it as a variant of BC-Z because BC-Z uses a single embedding with no composition rule; the homomorphism constraint is the structural property neither prior method establishes.

**Why this measures what we claim.** The homomorphism consistency loss L_hom measures exact composability because it enforces T_{E1 o E2}(p) = T_{E1}(T_{E2}(p)) for all paired expressions with the same denotation and both using only SEQ/REPEAT; this assumption fails when the parser produces ambiguous denotations for syntactically distinct expressions, in which case L_hom reflects parser ambiguity rather than algebraic exactness. The PAR convex mixture measures commutative parallel composition because alpha=0.5 makes the operator symmetric; this assumption fails when parallel sub-tasks have asymmetric resource contention, in which case the mixture reflects an average rather than a true parallel semantics. The COND gate g_c measures observation-conditioned branch selection because it is trained with subtask boundary labels; this assumption fails when the annotation is noisy at transition boundaries, in which case g_c reflects label noise rather than conditional necessity. The REPEAT k-fold composition measures iterative sub-task application because k is read from the parser; this assumption fails when the instruction uses vague repetition counts, in which case the composition reflects a default count rather than the intended iteration. The law-residual diagnostics measure whether the learned T_i actually satisfy the monoid laws on held-out leaves: SEQ associativity residual measures closure of the SEQ sub-monoid because it compares T_{SEQ(SEQ(A,B),C)} and T_{SEQ(A,SEQ(B,C))} on the same p; this assumption fails when the linear phi is not frozen or when W_i drift during training, in which case the residual reflects optimization noise rather than algebraic violation. PAR commutativity and idempotence residuals measure the soft-operator approximation because they compare T_{PAR(A,B)} to T_{PAR(B,A)} and T_{PAR(A,A)} to T_A; these assumptions fail when alpha is not exactly 0.5 or when the leaf transforms are not idempotent, in which case the residuals reflect operator mis-specification rather than parallel semantics. COND saturation residual measures gate calibration because it compares g_c to 0/1 under known condition labels; this assumption fails when the gate MLP is under-trained, in which case the residual reflects optimization rather than conditional necessity.

## Contribution

['A typed operator algebra (TOPA) over physical tokens with a learned monoid action constrained by a homomorphism loss, providing exact composition of sub-task dynamics without retraining on composite tasks.', 'A grammar-rewriter data augmentation scheme that generates paired expressions with identical denotation to supervise homomorphism consistency, enabling the algebra to be trained from atomic task demonstrations alone.', "An empirical design principle: fixed convex mixture for PAR and learned gating for COND yield commutative and observation-conditioned composition respectively, which we validate by ablating each operator's structural constraint."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | LIBERO + LIBERO-Plus (composite tasks) | Tests compositional generalization with paired denotations. |
| Dataset | CALVIN (long-horizon, composites from seen leaves) | Tests long-horizon composition on real benchmark. |
| Primary metric | Success rate on composite task suites | Directly measures correct composed execution. |
| Secondary metric | Law-residual diagnostics (SEQ assoc., PAR comm./idem., COND sat.) | Tests whether learned T_i satisfy monoid laws. |
| Secondary metric | Fraction of benchmark instructions parseable by grammar | Judges practical coverage of claimed significance. |
| Baseline | PDRVLA (flat physical tokens) | Tests whether operator algebra beats flat conditioning. |
| Baseline | BC-Z (single task embedding) | Tests whether composition rules beat monolithic embeddings. |
| Baseline | SEQ-recurrent composer (learned) | Tests whether fixed monoid composition beats learned recurrence. |
| Ablation-of-ours | TOPA without L_hom (end-to-end only) | Tests whether homomorphism constraint prevents memorization. |
| Ablation-of-ours | TOPA with nonlinear phi (original residual family) | Tests whether linear-phi closure is necessary for exact composition. |

### Why this setup validates the claim

This design forms a falsifiable test of TOPA's central claim: that typed operator algebra with a homomorphism-constrained monoid action yields exact, compositional generalization to novel task structures. LIBERO and LIBERO-Plus composite suites provide paired expressions with identical denotation (via grammar rewriter), so success rate on held-out composites directly probes whether the composed transformation generalizes rather than memorizes. CALVIN adds a long-horizon real benchmark where composites are built from seen leaves, testing whether the SEQ/REPEAT sub-monoid remains closed over longer horizons. PDRVLA tests the sub-claim that operator structure beats flat physical tokens: if TOPA's gain over PDRVLA is uniform across atomic and composite tasks, the algebra adds nothing; the claim predicts a concentrated gain on composites. BC-Z tests whether composition rules beat a single embedding—its failure on composites with novel operator nesting would confirm the need for explicit algebra. The learned SEQ-recurrent composer baseline tests whether fixed function composition (associative by construction) is necessary versus a learned recurrent mixer that can violate associativity. The ablation without L_hom is the critical diagnostic: if it matches TOPA on composites, the homomorphism constraint is not doing the claimed work, and the gain is due to parser or token capacity rather than algebraic exactness. The ablation with nonlinear phi tests the load-bearing closure assumption: if it matches TOPA on SEQ/REPEAT composites, the linear-phi restriction is unnecessary; if it degrades, closure is doing the work. The law-residual diagnostics directly test whether the learned T_i satisfy the monoid laws on held-out leaves, and the parseability fraction reports how much of each benchmark the grammar covers. Success rate is the right primary metric because it directly measures the downstream behavioral consequence of correct composition; token-space distances or loss values could be gamed without action-level correctness.

### Expected outcome and causal chain

**vs. PDRVLA** — On a composite instruction like “pick the block, then place it left of the cup while keeping the cup upright,” PDRVLA conditions on a flat token set with no operator algebra, so its physical tokens for the composite are a bag-of-subtasks embedding that cannot enforce sequential order or parallel constraints, producing order-violating or constraint-dropping actions. Our method parses the instruction into SEQ(place, PAR(keep-upright)) and composes the corresponding monoid elements via function composition and convex mixture, so the conditioning vector encodes the exact nesting. We expect a noticeable success-rate gap on multi-operator composites (SEQ containing PAR or COND) but near parity on atomic single-subtask instructions, where PDRVLA's flat tokens are sufficient.

**vs. BC-Z** — On a zero-shot composite task with a novel nesting such as REPEAT(3, COND(graspable, pick, push)), BC-Z uses a single task embedding trained on monolithic demonstrations, so it cannot represent the conditional inside the repetition and collapses to an average behavior (e.g., repeated pushing regardless of graspability). Our method retrieves the leaf transformations and evaluates the typed tree bottom-up, applying the learned COND gate g_c and k-fold composition, so the conditioning vector respects both the condition and the iteration count. We expect BC-Z to fail sharply on composites with nested COND/REPEAT while TOPA retains most of its single-task success, yielding a large gap concentrated on those nested structures.

**vs. SEQ-recurrent composer** — On a long sequence where the same sub-task appears twice in different orders, e.g., SEQ(A, B, A) vs. SEQ(A, A, B), a learned recurrent composer can produce different tokens for the two parenthesizations because its composition is not associative by construction, so it may mis-order the repeated A relative to B. Our method uses function composition, which is associative by construction, so both parenthesizations yield identical tokens and the action respects the true order. We expect the recurrent composer to show inconsistent success across reordered equivalents of the same task, while TOPA shows near-identical success across reorderings—a variance gap rather than a mean gap.

**vs. TOPA without L_hom** — On a held-out composite whose denotation can be reached by two syntactically distinct expressions (e.g., SEQ(A, SEQ(B, C)) vs. SEQ(SEQ(A, B), C)), the end-to-end-only variant can fit each composite by memorizing a separate transformation, so it will succeed on training composites but fail to transfer the composition rule to a novel nesting of seen leaves. Our method enforces T_E(p) = T_{E'}(p) for paired denotations, so the transformation is constrained to compose correctly and transfers to novel nestings. We expect the ablation to match TOPA on seen composites but drop noticeably on novel-nesting composites built from seen leaves—the diagnostic signature that the homomorphism constraint, not extra capacity, drives compositional generalization.

**vs. TOPA with nonlinear phi** — On SEQ/REPEAT composites, the nonlinear-phi variant composes maps p + W_2·phi(p + W_1·phi(p)) that leave the leaf family {p + W·phi(p)}, so the composed transformation is not representable as a leaf and the 'exact by construction' guarantee is vacuous. Our method fixes phi linear and frozen, so composition stays in the family with W' = W_2 + W_1 + W_2 A W_1. We expect the nonlinear-phi variant to show higher SEQ associativity residuals and lower success on novel-nesting SEQ/REPEAT composites, while matching TOPA on atomic tasks.

### What would falsify this idea

If TOPA's gain over PDRVLA and BC-Z is uniform across atomic and composite subsets rather than concentrated on multi-operator and novel-nesting composites, or if the no-L_hom ablation matches TOPA on novel-nesting composites, or if the law-residual diagnostics show SEQ associativity residuals above 1e-3 on held-out leaves, then the central claim that the homomorphism-constrained operator algebra is what enables compositional generalization would be wrong.

## References

1. PDRVLA: A Vision-Language-Action Model with Physical Dynamics Representations
2. BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning
3. Wan: Open and Advanced Large-Scale Video Generative Models
