# CompCAS: Compositional Semantics for Code Hallucination and Quality Attributes over Abstract Syntax Trees

## Motivation

Existing taxonomies for code generation hallucinations rely on manual annotation, leading to inconsistent categories across studies (e.g., the survey 'Hallucination by Code Generation LLMs' notes definitional inconsistency). Concurrently, benchmarks like Collu-Bench assume static suites and lack coverage guarantees—they test only properties that are easily annotated or executed. Both issues stem from a lack of formal, compositional definitions over code structure: without a mechanism to decompose code properties into checkable sub-properties, labeling is ad-hoc and benchmark construction cannot be exhaustive.

## Key Insight

A compositional semantics over abstract syntax trees yields a closure property: any valid property can be decomposed into sub-properties of sub-trees, enabling automated verification and exhaustive benchmark generation.

## Method

### (A) What it is
CompCAS (Compositional Code Anomaly Semantics) is a formal language for specifying code generation hallucinations and quality attributes as predicates over abstract syntax trees (ASTs). Its inputs are: an AST of generated code, a reference AST (optional), and a set of property definitions. Outputs are: a labeled property vector (boolean per property) and a set of benchmark cases with known ground truth.

### (B) How it works
We define a compositional semantics using a recursive function over AST nodes. Each node type has attributes (e.g., identifier names, types). Properties are first-order logic formulas over node attributes and structural relations (parent, child, sibling). The property checker evaluates formulas via structural recursion. Benchmark generation uses a grammar-based AST generator that enumerates ASTs satisfying or violating specified properties.

```python
# Pseudocode for property checker (simplified)
def check_property(ast, prop_formula):
    # prop_formula is a lambda that takes an AST node and context
    def visit(node, context):
        if prop_formula(node, context):
            return True
        for child in node.children:
            if visit(child, context | {node.role: node}):
                return True
        return False
    return visit(ast.root, {})

# Example property: identifier hallucination (identifier not in scope)
def id_hallucination(node, context):
    if node.type == 'Identifier':
        # Check if identifier is declared in surrounding scopes
        return not is_declared(node.name, context)
    return False

# Benchmark generator (conceptual)
def generate_benchmark(properties, depth=3):
    # Enumerate all ASTs up to depth satisfying property combinations
    for ast in enumerate_all_asts(depth):
        labels = {p: check_property(ast, p.formula) for p in properties}
        yield (ast, labels)
```

### (C) Why this design
We chose a compositional, first-order logic approach over monolithic rule-based classification because compositionality ensures that every property is defined in terms of sub-structures, making the semantics provably closed under structural recursion—eliminating ad-hoc categories. We fixed on first-order logic over a probabilistic model because we need deterministic ground truth for benchmark construction; probability introduces uncertainty that undermines ground truth. We use AST representation rather than token sequences because ASTs capture syntax and structure directly, avoiding tokenization ambiguities. The trade-off is our method cannot capture dynamic or runtime properties (e.g., infinite loops) without additional execution semantics. We accept this limitation, focusing on statically checkable properties. This design also enables modular addition of new properties without altering existing definitions.

### (D) Why it measures what we claim
The property checker’s evaluation of a logical formula over AST nodes measures the presence of `identifier hallucination` because the formula encodes the condition that an identifier usage has no declaration in scope under standard scoping rules; this assumption holds for lexically-scoped languages. This assumption fails in the presence of dynamic scoping or `eval`-like constructs, where the property checker would misclassify legitimate uses as hallucinations. The compositional definition of `unreachable code` measures control flow anomaly because it relies on static control flow analysis assuming structured programming (no gotos); this fails in languages with unstructured jumps, where reachability becomes undecidable. The benchmark generator’s enumeration of ASTs with known labels measures coverage of the defined property space because generation systematically iterates over node attribute combinations satisfying or violating the formula; this enumerative assumption is valid for bounded AST depth, but fails when properties require unbounded recursion (e.g., general recursion depth), where we must impose a depth limit. Each computational quantity in (B) (the recursive visit, the scope context, the enumeration loop) directly operationalizes the corresponding motivation-level concept (checking property, tracking scoping, covering property space) under explicit assumptions that we named.

## Contribution

(1) A compositional formal ontology for code generation hallucinations and quality attributes defined over ASTs, enabling automatic and consistent labeling. (2) A property checker that evaluates any defined property via structural recursion, eliminating manual annotation. (3) An automated benchmark generator that produces test cases with known ground truth covering all defined properties, enabling exhaustive evaluation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | HumanEval-X with injected hallucinations | Synthetic ground truth for hallucination types |
| Primary metric | F1 score on hallucination detection | Balances precision and recall; binary per property |
| Baseline 1 | Rule-based AST pattern matcher | No compositional recursion; flat rule list |
| Baseline 2 | LLM-based zero-shot classifier | Leverages LLM's code understanding for comparison |
| Ablation of ours | Monolithic rule version of CompCAS | Removes compositionality; keeps same property formulas |

### Why this setup validates the claim
The combination of a synthetic dataset with known ground truth allows precise measurement of detection accuracy. The rule-based baseline tests whether simple pattern matching suffices; the LLM baseline tests if a strong general model can match our structured approach. The ablation tests the specific contribution of compositionality. F1 metric captures both false positives and false negatives, essential for benchmark generation where both matter.

### Expected outcome and causal chain

**vs. Rule-based AST pattern matcher** — On a case where identifier hallucination occurs in a nested scope (e.g., a function inside another function where a variable is used but declared only in the outer scope), the rule-based baseline may fail because it lacks recursive scope tracking; it might miss the hallucination if its rule only checks immediate parent scope. Our method handles this via compositional scope context propagation, so we expect a noticeable gap on nested-scope examples (e.g., 20% F1 improvement) but parity on simple flat-scope cases.

**vs. LLM-based zero-shot classifier** — On a case of unreachable code due to dead branches in complex control flow (e.g., nested if-else with mismatched conditions), the LLM may hallucinate reasoning or rely on surface patterns, leading to inconsistent predictions. Our method uses static control flow analysis, which is deterministic and correct under structured programming assumptions. We expect our method to achieve near-perfect accuracy on structured control flow, while the LLM may show ~70% F1 on complex patterns.

### What would falsify this idea
If our method does not outperform the monolithic ablation on multi-property cases, or if its performance is uniformly better across all subsets rather than specifically on nested or compositional scenarios, then the compositionality claim is not supported.

## References

1. Hallucination by Code Generation LLMs: Taxonomy, Benchmarks, Mitigation, and Challenges
2. LLM Hallucinations in Practical Code Generation: Phenomena, Mechanism, and Mitigation
3. Collu-Bench: A Benchmark for Predicting Language Model Hallucinations in Code
4. FELM: Benchmarking Factuality Evaluation of Large Language Models
5. A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions
6. A Survey of Large Language Models for Code: Evolution, Benchmarking, and Future Trends
7. Understanding Factual Errors in Summarization: Errors, Summarizers, Datasets, Error Detectors
8. Chain of Thought Prompting Elicits Reasoning in Large Language Models
