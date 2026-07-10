# Unsupervised Grammar Induction for Low-Resource Code Completion via Perplexity-Driven Parsing

## Motivation

Existing approaches to code completion for low-resource languages rely on tedious manual data curation or transfer from high-resource languages (e.g., Teaching LLMs a Low-Resource Language). These methods scale poorly because each new language requires anew annotation effort. Unsupervised grammar induction from unlabeled code could bypass this bottleneck, but prior parsing objectives (e.g., language modeling likelihood) do not directly optimize for completion quality, leaving a gap between induced syntax and downstream utility.

## Key Insight

Completion perplexity of a held-out token conditioned on a parse tree forms a natural, differentiable reward that is maximized only when the tree captures the true syntactic structure, because only then does the completion model see the minimal predictive uncertainty.

## Method

(A) **What it is**: Unsupervised Parser-Completion Co-training (UPC2). It takes as input unlabeled code snippets and jointly trains a neural chart parser and a tree-conditioned completion Transformer. The parser outputs a distribution over constituency trees; the completion model predicts the next token given the prefix and the parse tree (via tree-adjusted attention). Training uses a perplexity-based ranking objective that encourages the parser to assign higher probability to trees that yield lower completion perplexity on held-out positions.

(B) **How it works** (pseudocode):
```python
# Hyperparameters
K = 5  # number of sampled parses
m = 1.0  # margin for ranking loss
learning_rate = 1e-4

for each iteration:
    sample minibatch of code snippets x
    for each x in minibatch:
        # Generate candidate parses via neural CKY
        compute PCFG parameters from P_θ encoding of x
        use CKY to compute top-K most probable parse trees: τ_1,...,τ_K
        
        # Compute completion perplexity for each tree
        for each τ_k:
            for each position t (sampled subset or all):
                PP_k(t) = exp( -log C_φ(x_t | x_{<t}, τ_k) )
            PP_k = mean over t of PP_k(t)
        
        # Ranking loss for parser
        PP_best = min_k PP_k
        for each k:
            loss_k = max(0, PP_k - PP_best - m)
        update P_θ to minimize sum_k loss_k via Gumbel-softmax differentiation
        
        # Language modeling loss for completion model using best parse
        τ_best = argmin_k PP_k
        for each t:
            L_LM = -log C_φ(x_t | x_{<t}, τ_best)
        update φ with gradient descent on L_LM
```

(C) **Why this design**: We chose a neural chart parser (CKY) over sequence-tagging parsers (e.g., transition-based) because CKY provides a distribution over all trees of a sentence, enabling us to sample diverse candidates for the ranking objective; the cost is cubic complexity O(n^3), but for code snippets (typically <200 tokens) it remains tractable and allows exact gradient computation via inside-outside algorithm if we relax the discrete tree sampling. We use a margin ranking loss with K=5 to avoid the expense of enumerating all trees; the margin m=1.0 provides a stable gradient signal without forcing ties. We also update C_φ only on the best parse per iteration rather than averaging over all trees, because using the best parse provides a sharper gradient that avoids overfitting to noisy trees; the trade-off is that we might miss alternative good structures, but the ranking loss on P_θ already explores diversity. Finally, we use Gumbel-softmax to make the tree selection differentiable, avoiding REINFORCE variance; this choice accepts a bias in the gradient estimate but enables end-to-end backpropagation.

(D) **Why it measures what we claim**: The completion perplexity PP_k (computed via log C_φ) measures the predictive uncertainty of the completion model conditioned on parse τ_k. The key assumption is that a parse tree that accurately captures syntactic dependencies reduces uncertainty for future tokens because it encodes long-range structure; this assumption fails when the completion model C_φ has limited capacity to exploit tree information (e.g., if attention is not effectively guided by the tree), in which case PP_k reflects the model's inefficiency rather than tree quality. To mitigate this, we design C_φ with tree-relative position encodings that explicitly inject subtree depth and sibling relations, ensuring that tree structure directly influences the attention computations. Another component: the margin ranking objective on P_θ uses PP_k as a reward to shift probability mass toward low-perplexity trees. The implicit assumption is that the ranking order of PP_k among trees is consistent with their syntactic correctness; this can be violated if the completion model is not equally well-calibrated for all trees (e.g., if it overfits to non-grammatical structures seen during training). We address this by updating C_φ only on the best parse per step, which gradually improves its capacity to distinguish good from bad trees. Thus, the full system operationalizes 'syntactic quality' as 'completion perplexity reduction' under the assumption that the completion model is a faithful evaluator of tree utility; when this assumption breaks due to model bias, the ranking loss degrades to noise, preventing further improvement.

## Contribution

(1) A novel co-training framework (UPC2) for unsupervised grammar induction in code completion that uses completion perplexity as a reward signal without any annotated data. (2) A design principle that tree-conditioned completion models, when paired with a margin ranking loss, can bootstrap syntactic knowledge from downstream task performance. (3) An empirical finding that the induced grammar improves completion accuracy on low-resource languages (e.g., Pharo, Lua) compared to unparsed baselines.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Pharo code corpus (method-level snippets) | Low-resource language with available data |
| Primary metric | Exact match accuracy on masked code completion | Measures functional correctness |
| Baseline 1 | Pre-trained LLM (CodeLlama-7B) | General code LLM without adaptation |
| Baseline 2 | Fine-tuned CodeLlama on Pharo | Supervised specialization baseline |
| Ablation | UPC2 w/o parser co-training (only completion model) | Isolates parser's contribution |

### Why this setup validates the claim
Low-resource languages lack parallel data, so our unsupervised parser-completion co-training must demonstrate that leveraging unlabeled syntax improves completion accuracy over both general large models and simple fine-tuning. Using Pharo, a truly low-resource language, ensures the difficulty is realistic. Exact match accuracy directly tests whether completed tokens are correct, avoiding noise from perplexity metrics. The ablation removes the parser, showing whether the co-training loop is necessary. If our method outperforms both baselines and the ablation, it confirms that the parser's ranking objective yields syntactic representations that boost completion performance on low-resource code, exactly when supervised data is scarce.

### Expected outcome and causal chain

**vs. Pre-trained LLM (CodeLlama-7B)** — On a case like a Pharo method with unique naming conventions (e.g., `#at:ifAbsent:`), the pre-trained LLM likely produces an irrelevant method call because it was never exposed to Pharo syntax. Our method instead parses the prefix, identifies the method's argument slots via the tree's internal nodes, and conditions the next-token prediction on those slots, yielding the correct selector. Thus, we expect a noticeable gap (e.g., 50–70% accuracy vs. 20–30%) on idiomatic Pharo constructs, but parity on generic control flow.

**vs. Fine-tuned CodeLlama on Pharo** — On a case where fine-tuning data covers the exact pattern (e.g., a common loop), the baseline may memorize and succeed. However, on an unseen syntactic variation (e.g., nested block with message cascades), fine-tuned LLM lacks structural generalization. Our method's parser abstracts the tree, guiding attention to dependencies regardless of surface form. Consequently, we expect our method to outperform on structurally complex but rare patterns (e.g., 40% vs. 10% accuracy), while performing similarly on frequent patterns.

**vs. Ablation (UPC2 w/o parser co-training)** — On a case requiring long-range dependency (e.g., matching parentheses for a block with many statements), the completion model without a parser may lose track of open structures. Our full method uses parse trees to reinforce hierarchical boundaries, reducing perplexity for matching tokens. Thus, we expect the ablation to show higher error on bracket matching (e.g., 15% more mismatches) and a lower overall accuracy on nested structures, confirming that the parser's structural cues are essential.

### What would falsify this idea
If the improvement over the ablation is uniform across all code patterns (suggesting the completion model alone can learn structure from raw text, or the parser adds no specific benefit), or if our method only improves perplexity but not exact match accuracy (indicating the ranking objective overfits to completion model artifacts), then the central claim that syntactic trees aid completion is not supported.

## References

1. Teaching LLMs a Low-Resource Language: Enhancing Code Completion in Pharo
2. Mellum: Production-Grade in-IDE Contextual Code Completion with Multi-File Project Understanding
3. Enhancing Code Generation for Low-Resource Languages: No Silver Bullet
4. Evaluating Quantized Large Language Models for Code Generation on Low-Resource Language Benchmarks
5. Qwen2.5-Coder Technical Report
6. Context Composing for Full Line Code Completion
7. Full Line Code Completion: Bringing AI to Desktop
8. Multi-line AI-Assisted Code Authoring
