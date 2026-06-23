# ProofCheck: Intrinsic Verification of Natural Language Proofs via Discourse-Aware Dependency Graphs

## Motivation

Existing proof verification methods require an external oracle—either human judges (Open Proof Corpus) or formal systems like Lean (Aristotle)—limiting scalability and automation. The structural root cause is the absence of a technique that assesses correctness using only the problem statement and proof text, without translation into a formal language. We address this by exploiting discourse markers that signal logical support relations, enabling graph-theoretic verification of necessary proof properties.

## Key Insight

A correct proof induces a directed acyclic graph (DAG) of logical support, where discourse markers like 'since' and 'therefore' signal the direction of inference, making the graph verifiable purely from surface text.

## Method

### Method

**ProofCheck: Intrinsic Verification of Natural Language Proofs via Discourse-Aware Dependency Graphs**

(A) **What it is**: ProofCheck is a rule-based parser that constructs a dependency graph from a natural language proof by identifying discourse markers and their spans, then checks three graph-theoretic properties—acyclicity, coverage, and consistency—to verify correctness without an external oracle. Input: a problem statement and a proof text. Output: a binary pass/fail verdict.

(B) **How it works** (pseudocode):
```
1. Preprocessing: Segment proof into sentences. Normalize text (lowercase, remove punctuation).
2. Dependency Extraction:
   For each statement S_i:
       If S_i contains a forward marker (e.g., 'therefore', 'hence', 'thus'):
           Extract the conclusion clause after marker, and hypothesis clause(s) from preceding context.
           Add edge: for each hypothesis H, add H -> S_i.
       If S_i contains a backward marker (e.g., 'since', 'because', 'as'):
           Extract the premise clause after marker, and conclusion clause (main clause).
           Add edge: S_i -> conclusion sentence (if conclusion is in separate sentence; otherwise treat S_i as node with no outgoing edge for simplicity).
       For statements with no explicit marker:
           Apply fallback heuristic: if S_i is not a premise (as per problem statement), look backward up to 3 sentences for a preceding statement that is not already used as a supporter. If found, add edge from that statement to S_i; else leave S_i isolated.
3. Graph Construction:
   Vertices V = all statements.
   Edges E from step 2.
   Remove duplicate edges.
4. Verification Checks:
   a) Acyclicity: Perform topological sort; if cycle detected, fail.
   b) Coverage: For each non-premise statement (not designated as problem premise or axiom), check it has at least one incoming edge. If any lacks support, fail.
   c) Consistency: Build set of asserted propositions (each statement's main claim). If a proposition and its negation both appear, fail; negation detection via antonym list or simple 'not-' keyword.
5. Decision: If all checks pass, return PASS; else FAIL.
```

(C) **Why this design**: We chose a rule-based discourse marker approach over a learned dependency parser because discourse markers like 'since' and 'therefore' have consistent logical function across mathematical proofs, avoiding the need for expensive training data (as in Open Proof Corpus). The trade-off is that proofs without explicit markers may yield incomplete graphs, leading to false negatives; we accept this cost for competition-level proofs which are typically verbose. We use directed edges from hypothesis to conclusion (consistent direction) rather than undirected relations because direction determines acyclicity—if flipped, cycles could be incorrectly introduced. We treat unmarked statements as unsupported, which is conservative: it ensures coverage is strict but may penalize implicit reasoning; we prefer false negatives over false positives for verification. We chose a simple negation-based consistency check over full logical entailment because entailment is undecidable in natural language; the trade-off is missing subtle inconsistencies, but the method remains tractable and captures clear contradictions. The fallback heuristic for unmarked statements assumes that nearby sentences are likely to provide support, which may sometimes introduce spurious edges; we accept this risk to improve recall.

(D) **Why it measures what we claim**: The edge extraction from discourse markers operationalizes 'logical support': each edge X -> Y means the author intends X as a reason for Y. **Coverage** measures the requirement that every non-primitive statement has such support; the assumption is that discourse markers reliably indicate intended support relations. This assumption fails when a proof uses marker-less inferences (e.g., 'Clearly'), leading to false negatives. **Acyclicity** measures that support is well-founded; the assumption that correct proofs are acyclic holds universally, as circular reasoning is invalid. **Consistency** measures non-contradiction; the assumption that explicit negation detection captures all contradictions fails when negation is indirect (e.g., 'x is not prime' vs 'x is composite' may not be exact negation), so we may miss inconsistencies. **Incoming edge from discourse marker** measures logical support under assumption A: discourse markers always indicate intended inference direction. Failure mode F: markers may be absent or misleading (e.g., sarcasm or complex reasoning), in which case the edge may not represent genuine support.

#### Load-bearing assumption stated explicitly:
Discourse markers like 'since' and 'therefore' reliably indicate the intended logical support relations in natural language mathematical proofs. This assumption is unverified and we add a fallback heuristic to mitigate its failure.

## Contribution

(1) A novel framework for intrinsic proof verification that uses only natural language text and discourse markers, eliminating the need for external oracles. (2) A concrete set of graph-theoretic checks (acyclicity, coverage, consistency) that are necessary for proof correctness and derivable from surface-level linguistic cues. (3) An empirical demonstration on a corpus of competition-level proofs that discourse-marker-based graphs reliably separate correct from incorrect proofs with high precision.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Open Proof Corpus (OPC) subset (500 proofs with human judgments) | Contains LLM-generated proofs with human judgments. |
| Primary metric | Agreement with human correctness (accuracy) | Directly measures verification validity. |
| Baseline 1 | GPT-4 judge (single prompt, no graph) | Strong LLM baseline, no graph reasoning. |
| Baseline 2 | Keyword heuristic (count markers, threshold=2) | Naive baseline, tests value of structure. |
| Baseline 3 | Random baseline (50% PASS/FAIL) | Floor performance for comparison. |
| Ablation-of-ours | No acyclicity check (skip step 4a) | Tests necessity of cycle detection. |
| Additional dataset | Textbook proofs (100 from Geometry textbook) | Assess generalizability beyond competitions. |
| Additional metric | Edge-level agreement with human annotations | Measures reliability of marker-based edges. |

### Why this setup validates the claim

This experimental design forms a falsifiable test of the claim that three graph-theoretic properties (acyclicity, coverage, consistency) are sufficient for verifying competition-level proofs. The Open Proof Corpus provides a realistic distribution of natural language proofs with known correctness, so agreement with human judgments directly measures verification fidelity. Comparing against GPT-4 tests whether our simple rule-based approach can match or exceed a powerful LLM that lacks explicit graph reasoning. The keyword heuristic isolates the value of structured dependency extraction over mere marker counting. The ablation removes acyclicity to test its causal role; if performance drops specifically on circular reasoning cases, the property is necessary. The random baseline sets a lower bound. The additional textbook dataset tests generalization to more formal, less dialogic proofs, where discourse markers may differ. The edge-level agreement metric on a pilot sample (50 proofs from OPC) quantifies how often marker-based edges align with human-annotated logical support, directly evaluating the key assumption.

### Expected outcome and causal chain

**vs. GPT-4 judge** — On a proof where a subtle circular dependency exists (e.g., statement A supports B, B supports C, C supports A), GPT-4 may issue a false PASS because it lacks explicit cycle detection and relies on surface plausibility. Our method instead identifies the cycle via topological sort failure and returns FAIL, because the acyclicity check is designed to catch such structure. Therefore, we expect a noticeable gap on proofs with implicit circularities (e.g., 30% lower false positive rate), while performance on linear, marker-rich proofs remains comparable.

**vs. Keyword heuristic** — On a proof that uses diverse discourse markers (e.g., 'therefore', 'since', 'because') but a few markers are missing or ambiguous, the keyword heuristic may incorrectly deem the proof unsupported (false negative) because it only counts markers and ignores implicit dependencies. Our method instead extracts a dependency graph from the markers present and applies a fallback heuristic for unmarked statements; even if some markers are missing, the coverage check may still pass if each non-premise statement has an incoming edge from a marker-identified supporter or from the fallback. Thus, we expect our method to have lower false negative rate (e.g., 20% higher recall) on proofs with incomplete but still recoverable discourse structure.

**vs. Ablation (no acyclicity)** — On a proof with a clear circular argument (e.g., chain of 'therefore' statements forming a loop), the ablation without acyclicity returns PASS because it ignores cycles, leading to a false positive. Our full method catches the cycle and returns FAIL. Hence, we expect a significant difference (e.g., 40% higher false positive rate for ablation) specifically on proofs containing circular reasoning, while both perform similarly on acyclic proofs.

**vs. Textbook proofs** — Our method may achieve lower accuracy on textbook proofs because they often use implicit reasoning (e.g., 'by Lemma X' without explicit marker) or have less predictable discourse structure. However, the fallback heuristic may partially compensate. We expect accuracy to drop by ~15% compared to OPC, but still outperform keyword heuristic.

### What would falsify this idea

If our method shows no systematic advantage over the keyword heuristic on proofs with incomplete discourse markers, or if the ablation without acyclicity performs identically to the full method on circular proofs, then the central claim that these graph properties are sufficient for verification is falsified. Additionally, if the pilot study reveals that marker-based edges agree with human annotations less than 50% of the time, the load-bearing assumption is unsupported and the method's foundation is weak.

## References

1. Aristotle: IMO-level Automated Theorem Proving
2. The Open Proof Corpus: A Large-Scale Study of LLM-Generated Mathematical Proofs
3. PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition
4. Autograding Mathematical Induction Proofs with Natural Language Processing
5. Omni-MATH: A Universal Olympiad Level Mathematic Benchmark For Large Language Models
6. Llama 2: Open Foundation and Fine-Tuned Chat Models
7. Solving Quantitative Reasoning Problems with Language Models
8. Have LLMs Advanced Enough? A Challenging Problem Solving Benchmark For Large Language Models
