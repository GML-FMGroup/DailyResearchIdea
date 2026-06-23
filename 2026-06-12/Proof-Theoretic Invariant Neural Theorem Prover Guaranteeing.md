# Proof-Theoretic Invariant Neural Theorem Prover: Guaranteeing Stepwise Logical Correctness via Linear Embedding Constraints

## Motivation

Current neural theorem provers (e.g., Aristotle, DeepSeek-Prover-V1.5) rely on external formal verification to catch incorrect steps, treating the neural model as a black-box generator. This decoupling means the generation process itself has no internal guarantee of logical validity, leading to wasted search in incorrect branches and reliance on expensive verifier calls. The structural root cause is that these models operate on token sequences without any embedded knowledge of inference rules; correctness is only checked post-hoc. Aristotle, for instance, uses a separate informal reasoning model that proposes steps in natural language, which are then formalized and verified in Lean; the informal model has no built-in constraint to produce logically sound proposals.

## Key Insight

By embedding premises in a vector space where each inference rule corresponds to a linear transformation, we can guarantee that the output of the network lies in the linear span of premise embeddings under rule-specific operators, making logical correctness a geometric invariant rather than a probabilistic guess.

## Method

# Proof-Theoretic Invariant Neural Theorem Prover (PINN-TP)

(A) **What it is:** PINN-TP is a neural architecture that generates proof steps constrained to be valid inferences by mapping premises to embeddings and applying rule-specific linear operators that are learned from data. Its input is a proof state (set of premises and goal) encoded as vectors; its output is a candidate conclusion embedding that must be representable as a linear combination of premise embeddings under the chosen rule's operator. **Load-bearing assumption:** The learned linear operators M_r faithfully capture the semantics of inference rules, and the GNN embedding space is semantically complete (logically equivalent formulas map to similar embeddings).

(B) **How it works:**
```pseudocode
Input: Premises P = {p1,...,pn}, goal g (each as formula tokens)
1. Encode premises and goal into d=512-dimensional vectors using a shared GNN.
   - GNN: 4-layer graph attention (GAT) with 4 heads, hidden size 256, ReLU activation.
   - Additionally, a Transformer with 6 layers, 8 attention heads, hidden size 512 processes the formula tree to produce final embeddings e(p_i), e(g) ∈ R^512.
2. Define 10 common inference rules: modus ponens, modus tollens, ∀-elimination, ∀-introduction, ∃-elimination, ∃-introduction, ∧-introduction, ∧-elimination, ∨-introduction, ∨-elimination.
   - For each rule r, learn a linear operator M_r ∈ R^{512×512} and bias b_r ∈ R^{512}. For binary rules, M_r ∈ R^{512×(2*512)} handling concatenated premise embeddings.
3. For each premise p_i, compute candidate conclusion e_cand_i = M_r e(p_i) + b_r (unary rules); for binary rules, e_cand_ij = M_r [e(p_i); e(p_j)] + b_r.
4. Generate attention weights α over rule-premise pairs via a two-layer attention: first attend over rules using rule embeddings (learned vector per rule), then over premises. Score: score(r,i) = e(p_i)^T W_r e(g) with W_r ∈ R^{512×512} learned per rule. Softmax yields α.
   - Output conclusion embedding e_c = sum_{r,i} α_{r,i} * e_cand_i (for unary) or sum over binary pairs.
5. Training: contrastive loss with temperature τ=0.1 and 32 negative examples per positive.
   - Optimizer: Adam, learning rate 1e-4, batch size 32, 50 epochs on miniF2F train+valid.
6. Regularization: orthogonal projection penalty λ * ||M_r^T M_r - I||_F^2 with λ=0.01 to encourage M_r to be orthogonal (preserving structure).
7. Hyperparameters: d=512, τ=0.1, negatives=32, λ=0.01.
```

(C) **Why this design:** We chose linear operators M_r rather than non-linear transformations because linearity is essential for the invariance property: if each rule's action is linear, then the set of conclusions reachable from premises is the linear span of premise embeddings under the operators. This structural guarantee is lost with non-linearities. We chose to learn M_r from data rather than hand-specify them because rules in formal mathematics often have subtle semantic variations (e.g., modus ponens in different logics) that are better captured empirically. This comes at the cost of requiring sufficient training data for each rule; for rare rules, we risk overfitting, which we mitigate by enforcing the orthogonal projection regularization that biases M_r toward known logical properties. We chose a contrastive loss rather than direct regression to the conclusion embedding because exact equality is too strict due to embedding approximation; contrastive learning tolerates variation while still pulling the correct conclusion closer. The trade-off is that contrastive loss may not precisely enforce the linear span constraint, so we also add the regularizer to keep e_c within the span. We chose a shared GNN for encoding premises and goal because it captures formula structure and dependencies; a simple bag-of-words would lose critical ordering information. The GNN incurs higher computational cost but is necessary for accurate embeddings. This design relies on the load-bearing assumption that learned M_r are faithful and the embedding space is semantically complete.

(D) **Why it measures what we claim:** The computational quantity α_{r,i} (the attention weight over rule-premise pairs) measures the model's confidence that applying rule r to premise i yields the correct next step; this equates to a probabilistic selection of a valid inference, assuming that the learned operators M_r are faithful to the logical rules. That assumption fails if the training data contains spurious correlations (e.g., a rule being used in a non-standard way), in which case α reflects the frequency of that pattern rather than logical validity. The embedding e_c (the generated conclusion embedding) is constrained to lie in the linear span of the candidate embeddings e_cand_i; this computational representation directly operationalizes the property of deductive closure under the rules, because any vector in that span can be written as a linear combination of rule-applied premise embeddings. Specifically, the linear span measures deductive closure under the assumption that GNN embeddings preserve logical equivalence (i.e., logically equivalent formulas have similar embeddings). If that assumption fails (e.g., syntactically different but logically equivalent formulas have distant embeddings), then being in the span reflects syntactic closeness, not logical validity. The contrastive loss L measures proximity in embedding space; it is a proxy for correctness under the assumption that the GNN encoder maps logically equivalent conclusions to nearby embeddings and inequivalent ones to distant embeddings. This assumption fails when the encoder is poorly trained, in which case L measures surface form similarity rather than logical equivalence.

## Contribution

(1) A novel neural theorem prover architecture that enforces logical correctness of each generation step via a linear embedding constraint, eliminating the need for external verifiers during inference.
(2) A design principle that inference rules can be represented as linear operators in a learned embedding space, providing an internal guarantee of step validity.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|-----------------------|
| Dataset | miniF2F | Standard high-school math competition benchmark |
| Primary metric | Proof success rate | Directly measures valid step generation |
| Baseline 1 | DeepSeek-Prover-V1 | Neural method without linear constraint |
| Baseline 2 | AlphaProof | Strong non-neural symbolic prover |
| Baseline 3 | LLMSTEP | Neural step suggestion without invariance |
| Ablation | PINN-TP without linear operators | Tests necessity of linear structure |

### Why this setup validates the claim
This combination of dataset and baselines creates a falsifiable test of our central claim that linear operators enforce deductive closure and attention weights measure inference validity. miniF2F is a controlled benchmark with ground-truth proofs, allowing precise measurement of step correctness. DeepSeek-Prover-V1 (without linear constraint) tests the necessity of linearity by comparing to a non-linear neural step predictor; if our method outperforms primarily on steps requiring strict logical entailment, the claim is supported. AlphaProof tests against a symbolic system that has perfect deductive closure but limited scalability; a win on a subset of problems would show linearity bridges neural flexibility with logical rigor. LLMSTEP tests whether simple next-step suggestion benefits from our structure. The ablation (removing linear operators) isolates the effect: if it degrades performance on deduction-heavy but not search-heavy problems, the claim is strengthened. The primary metric (proof success rate) captures end-to-end validity, directly reflecting whether generated steps are indeed valid inferences.

### Expected outcome and causal chain

**vs. DeepSeek-Prover-V1** — On a problem requiring a chain of modus ponens steps, the baseline may produce a conclusion that is syntactically plausible but not entailed because its non-linear projection can violate transitivity. Our method applies linear operators that preserve the entailment structure, so generated conclusions stay within the deductive closure. We expect a noticeable gap on problems requiring multi-step deduction (e.g., 15% absolute improvement) but parity on single-step rewrites.

**vs. AlphaProof** — On a problem with large search space but simple deduction (e.g., a combinatorial identity), AlphaProof may exhaust resources exploring irrelevant branches due to lack of learned heuristics. Our method uses learned attention to focus on the rule-premise pair that leads to the correct conclusion, drastically reducing search. We expect our method to solve a subset of problems (e.g., 20% more) where AlphaProof times out, but AlphaProof may win on problems requiring deep but narrow deduction.

**vs. LLMSTEP** — On a problem where a non-linear neural network tends to overgeneralize (e.g., confusing a ∀-elimination with an ∃-introduction), LLMSTEP might suggest a plausible but invalid step. Our linear operator is specifically regularized to align with rule semantics, reducing such semantic errors. We expect our method to have higher step validity (e.g., 10% higher success rate per step) while LLMSTEP may produce faster but less accurate suggestions.

### What would falsify this idea
If our method’s improvement over DeepSeek-Prover-V1 is uniform across all problem types (including those where deductive closure is irrelevant, like simple rewriting) rather than concentrated on multi-step deduction problems, this would indicate the linear constraint is not the source of the gain and the central claim is wrong.

## References

1. Aristotle: IMO-level Automated Theorem Proving
2. DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search
3. LLMSTEP: LLM proofstep suggestions in Lean
4. LeanDojo: Theorem Proving with Retrieval-Augmented Language Models
5. CoCoMIC: Code Completion by Jointly Modeling In-file and Cross-file Context
6. Few-shot Learning with Retrieval Augmented Language Models
