# Amortized Role-Playing via Low-Rank Behavioral Primitive Mixing

## Motivation

Existing role-playing methods (e.g., RAR from Thinking in Character) either require multiple LLM calls per scenario or a large pre-trained backbone, leading to computational cost scaling linearly with the number of scenarios or model size. The root cause is that each character or scenario is treated as a separate task requiring its own parameters or multiple inference steps, with no sharing of common behavioral patterns.

## Key Insight

Role-playing behaviors across characters and scenarios are composed of a small set of reusable behavioral primitives that lie in a low-dimensional subspace, allowing a lightweight mixing network to predict the composition coefficients cheaply, amortizing the cost of adaptation.

## Method

## Method: Factorized Role-Playing via Basis Mixing (FRBP)

### (A) What it is
FRBP is a method that learns a set of K behavioral basis vectors shared across characters and scenarios, and a lightweight mixing network that predicts their combination weights from the character description and scenario. The final behavior policy is a linear combination of these bases, which is then used as a conditioning signal for a frozen LLM, enabling sublinear cost for new characters/scenarios.

### (B) How it works
Input: Character description C (text), Scenario context S (text), Query Q (text).
Parameters:
- K behavioral basis vectors {b_i in R^d}, i=1..K, with K=256 (increased from 32 to better cover the behavioral space)
- Mixing network M: a 2-layer MLP (hidden size 128, ReLU) that takes a concatenated embedding e = [E(C); E(S)] and outputs logits l_i; softmax gives weights w_i.
- E is a fixed sentence encoder (e.g., Sentence-BERT) to produce 768-dim embeddings.
- Frozen LLM (e.g., Llama 2 7B) with prefix-tuning parameters P (learned prefix length 64) that take the behavioral vector b.
- Libraries: PyTorch 1.13, Transformers 4.30, Sentence-Transformers 2.2. Training requires ~48 GPU hours on a single A100 80GB.

Inference:
1. Compute e = [E(C); E(S)]  # 2*768 dim
2. Compute w = softmax(M(e))  # K-dim
3. Compute behavioral vector b = sum_i w_i * b_i  # weighted sum (convex combination)
4. Generate response: LLM(prefix=b; query Q)  # b is prepended as a learned prefix to the LLM's hidden states

Training (episodic):
- Dataset: (C, S, Q, target R) for various characters and scenarios.
- Loss: L = -log p(R | prefix=b, query=Q) (language modeling loss)
- Regularization: L_reg = -alpha * H(w) where H is entropy of w (encourages sparse mixing)
- Update M and {b_i} (and optionally prefix parameters P) via gradient descent. LLM weights are frozen.
- Hyperparameters: K=256, d=4096 (LLM hidden), alpha=0.1, learning rate=1e-4, batch size=16, Adam optimizer.

### (C) Why this design
We chose a fixed set of K basis vectors over per-character parameterization (e.g., Character-LLM's profile editing) because the latter's cost scales linearly with the number of characters, while the former amortizes representation across all characters. We chose a lightweight mixing network (2-layer MLP) over a full LLM-based predictor (e.g., using the LLM itself to output mixing weights) to keep the adaptation cost sublinear and avoid the very linear cost we aim to reduce. We chose entropy regularization on the mixing weights to encourage sparse, interpretable compositions rather than uniform mixing, which would dilute character-specific signals. The trade-off of this design is that the expressiveness of behavior is limited to the subspace spanned by the K basis vectors; if K is too small, subtle character nuances may be lost. However, as K increases, the cost of computing b remains O(Kd) which is sublinear relative to retraining a full model per character.

### (D) Why it measures what we claim
The mixing weights w measure the degree to which each behavioral primitive is active for a given character-scenario pair, because w is the only variable that changes across characters and scenarios in the inference pipeline. This operationalizes the claim that role-playing behaviors are compositions of shared primitives; the central load-bearing assumption is that the space of role-playing behaviors is low-rank and can be accurately represented as a convex combination of K=256 basis vectors. This assumption fails when two characters require behaviors that are linearly independent and cannot be expressed as a convex combination of the K bases; in that case, w captures an approximation error rather than the true behavior, and the generated response will be a mix of incorrect primitives. To verify this assumption, we perform PCA on the learned behavioral vectors (the weighted combinations from training episodes) and measure the cumulative variance explained by the top K components. If the top 256 components explain >90% variance, the low-rank assumption is empirically supported. The entropy regularization further ensures that each character uses a sparse combination of primitives, aligning with the hypothesis that characters are distinguished by a small subset of traits.

## Contribution

(1) Introduces Factorized Role-Playing via Basis Mixing (FRBP), a novel framework that learns a low-rank behavioral primitive space shared across characters and scenarios, with a lightweight mixing network for sublinear adaptation. (2) Demonstrates that role-playing behaviors can be efficiently approximated by linear combination of a small number of learned basis vectors, enabling cost-effective scaling to new characters and scenarios without retraining full models. (3) Provides a training procedure combining language modeling and entropy regularization to learn interpretable and sparse behavioral compositions.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RoleBench (100 chars × 10 scenarios) | Diverse characters and contexts |
| Primary metric | Character consistency accuracy | Measures role faithfulness |
| Baseline1 | Character-LLM (per-character fine-tuning) | Linear cost with characters |
| Baseline2 | Full LLM weight prediction | Linear cost via LLM calls |
| Baseline3 | Standard LLM (role description only) | No behavior adaptation |
| Ablation | FRBP (uniform mixing) | Tests sparsity benefit |
| Additional analysis | PCA variance explained on learned behavioral vectors | Validates low-rank assumption |

### Why this setup validates the claim
This setup directly tests FRBP's central claim: that role-playing behaviors are compositions of shared primitives, and that a lightweight mixing network can achieve sublinear adaptation cost without sacrificing quality. By comparing against per-character fine-tuning (Character-LLM) and full LLM-based weight prediction, we isolate the cost-efficiency advantage. The standard LLM baseline tests the necessity of any adaptation. The uniform mixing ablation tests whether entropy regularization's sparsity improves interpretability and performance. Character consistency accuracy is chosen because it directly measures the core goal: staying in character. If FRBP matches the per-character approach on consistency while being cheaper, the claim is supported. The dataset's diversity ensures coverage of the failure modes we predict. Additionally, we analyze the learned basis vectors by computing PCA variance explained across all characters to directly validate the low-rank assumption. If the top K=256 components explain >90% variance, the assumption is supported.

### Expected outcome and causal chain

**vs. Character-LLM** — On a new character not seen in training, Character-LLM requires full fine-tuning (O(1) per character) or retraining; this is costly. FRBP instead computes mixing weights from the character description alone (O(K) inference), enabling instant adaptation. We expect FRBP to achieve comparable or slightly lower consistency on the first few interactions (since bases may not capture all nuance), but the gap narrows with K=256. The observed signal: FRBP matches Character-LLM within 5% on consistency but at 10× lower cost per character. The PCA analysis is expected to show >90% variance explained, supporting the low-rank assumption.

**vs. Full LLM weight prediction** — On a complex scenario requiring multiple behavioral primitives, the full LLM method uses the LLM to output mixing weights (O(LLM inference) per request), which is expensive and linearly scales with new characters. FRBP's lightweight MLP (O(M LP) vs O(LLM)) is sublinear. Both can fail if bases are insufficient, but FRBP's fixed bases amortize cost across all characters. We expect FRBP to be slightly worse on rare primitive combinations but overall within 3% of consistency, while being 100× faster in adaptation overhead.

**vs. Standard LLM** — On a subtle character trait (e.g., a detective with a phobia), the standard LLM with only role description often ignores the trait in specific scenarios (e.g., when phobia is triggered). FRBP's behavioral vector directly conditions the LLM, making the trait active. We expect a large gap (~20% consistency) on scenario-subset where the trait is critical, and parity on generic responses.

### What would falsify this idea
If FRBP's consistency is not significantly better than standard LLM on trait-specific subsets (within 2% difference), or if its cost advantage over Character-LLM is offset by a >10% consistency drop, then the central claim that basis mixing captures behavior composition would be falsified. Additionally, if PCA variance explained by top 256 components is below 80%, the low-rank assumption is questionable.

## References

1. Thinking in Character: Advancing Role-Playing Agents with Role-Aware Reasoning
2. Role-Playing Evaluation for Large Language Models
3. Unveiling the Secrets of Engaging Conversations: Factors that Keep Users Hooked on Role-Playing Dialog Agents
4. Character-LLM: A Trainable Agent for Role-Playing
5. Dungeons and Dragons as a Dialog Challenge for Artificial Intelligence
6. IBSEN: Director-Actor Agent Collaboration for Controllable and Interactive Drama Script Generation
7. LLMs + Persona-Plug = Personalized LLMs
8. Identity-Driven Hierarchical Role-Playing Agents
