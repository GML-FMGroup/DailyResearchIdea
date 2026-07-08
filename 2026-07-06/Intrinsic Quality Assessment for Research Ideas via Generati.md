# Intrinsic Quality Assessment for Research Ideas via Generative Stability and Descendant Diversity

## Motivation

Existing methods for evaluating research ideas, such as the approach in 'Measuring the Gap Between Human and LLM Research Ideas', treat human-authored ideas as the definitive gold standard, introducing human bias and ignoring other quality dimensions like coherence and novelty. This reliance on human references is structurally limited because it evaluates LLM ideas by how closely they match human ideas rather than by their internal consistency and generative capacity. A method that assesses ideas based on their intrinsic generative properties—invariance under paraphrase (coherence) and diversity of descendant ideas (novelty)—can overcome this meta-gap.

## Key Insight

The intrinsic quality of a research idea can be assessed by its stability under paraphrasing (coherence) and the diversity of its generated descendants (novelty), both computable solely from an LLM's own generative variability without any human reference.

## Method

### (A) What it is
**IIA-GSD (Intrinsic Idea Assessment via Generative Stability and Divergence)** is a reference-free evaluation method that takes a candidate research idea (a short text) and outputs two scores: *coherence* (ranging 0-1) and *novelty* (ranging 0-1). The method requires only an LLM (e.g., GPT-4), a semantic embedding model (e.g., Sentence-BERT), and a semantic equivalence classifier (e.g., a DeBERTa-based NLI model). No human-authored ideas are needed.

### (B) How it works

**Input:** Idea text \(I\) (string).  
**Hyperparameters:** number of paraphrases \(P=20\), temperature \(\tau=0.8\) for paraphrase generation, number of descendant ideas \(D=20\), temperature \(\tau'=0.9\) for descendant generation, embedding model \(E\) (e.g., all-MiniLM-L6-v2), semantic equivalence classifier \(C_{eq}\) (e.g., DeBERTa-large fine-tuned on MNLI, threshold \(t=0.5\) for entailment), maximum invalid ratio \(r_{max}=0.5\).

**Phase 1 – Coherence via Invariance**  
1. Generate \(P\) paraphrases of \(I\) using LLM with prompt: "Paraphrase the following research idea without changing its core meaning: {I}". Use temperature \(\tau\) and top-p=0.95.  
2. For each paraphrase \(p_i\), compute its semantic equivalence score \(s_i = C_{eq}(I, p_i)\) (probability of entailment). Keep only paraphrases with \(s_i \ge t\). If the number of kept paraphrases is less than \(\lceil P \cdot (1 - r_{max}) \rceil\), generate additional paraphrases until the quota is met.  
3. For each valid paraphrase \(p_i\), compute its embedding \(e_i = E(p_i)\).  
4. Compute pairwise cosine similarity matrix \(S\) where \(S_{ij} = \cos(e_i, e_j)\) for \(i \neq j\).  
5. Coherence score \(C = \text{mean}(S)\) (excluding diagonal).  

**Phase 2 – Novelty via Descendant Diversity**  
1. Generate \(D\) descendant ideas by prompting LLM: "Given the following research idea, propose a novel extension or variation: {I}". Use temperature \(\tau'\) and top-p=0.95.  
2. For each descendant \(d_j\), compute embedding \(f_j = E(d_j)\).  
3. Compute pairwise cosine dissimilarity matrix \(T\) where \(T_{kl} = 1 - \cos(f_k, f_l)\).  
4. Novelty score \(N = \text{mean}(T)\) (average pairwise dissimilarity).  

**Output:** \((C, N)\).

### (C) Why this design  
We chose phrase-level paraphrasing (single rephrasing instruction) over multi-prompt aggregation because a single controlled transformation isolates invariance to wording, whereas multiple prompts (e.g., "explain differently") confound style changes with semantic drift — the cost is that we may miss invariance to very different phrasings (e.g., from abstract to concrete), but those can be captured by varying the temperature instead. To mitigate the risk of semantic drift during paraphrase generation, we introduce a calibration step that filters out paraphrases that fail a semantic equivalence test (entailment score \(\geq 0.5\) from a DeBERTa-based NLI classifier), ensuring that only meaning-preserving paraphrases contribute to the coherence score. We selected cosine similarity as the invariance metric rather than exact match or BLEU because it is robust to synonym substitutions and syntactic variation, accepting that it may give high scores to paraphrases that are semantically similar but logically inconsistent (e.g., swapping cause and effect), which we further mitigate by the calibration step. For descendant diversity, we used a high temperature (0.9) to encourage diverse generations, trading off the risk of generating off-topic or invalid extensions — but we do not filter by validity because novelty should account for wild ideas; coherence can be separately assessed if needed. Finally, we chose embedding-based dissimilarity over lexical diversity (e.g., distinct n-grams) because lexical overlap can underestimate conceptual novelty when descendants use different terminology for similar concepts (a known issue in scientific text).

### (D) Why it measures what we claim  
The average pairwise cosine similarity among filtered paraphrases (C) measures **coherence** because it quantifies the invariance of the idea's meaning across paraphrasing; this operationalization assumes that a coherent idea has a single, stable core meaning that is preserved under re-expression. **Assumption: paraphrase similarity captures conceptual stability. Failure mode:** ambiguous ideas get low similarity despite being coherent, so C reflects ambiguity rather than coherence. In that case, the metric reflects conceptual clarity (or lack thereof) rather than coherence per se. The average pairwise embedding dissimilarity among descendants (N) measures **novelty** because it captures the conceptual spread of ideas that branch from the original; this operationalization assumes that a novel idea is a fertile source of diverse extensions. **Assumption: descendant diversity captures intrinsic novelty. Failure mode:** descendants are diverse but trivially related (e.g., "apply to different domains" without deep innovation), so N reflects breadth of applicability rather than intrinsic novelty. To partially address this, descendants are prompted to be "extensions or variations," which encourages creative leaps rather than simple domain shifts. Overall, the two scores provide an intrinsic, human-independent assessment that directly targets the meta-gap of reliance on human gold standards.

## Contribution

(1) A reference-free evaluation framework (IIA-GSD) that measures research idea coherence via invariance under paraphrase and novelty via diversity of generated descendants, eliminating reliance on human-authored gold standards. (2) A design principle that intrinsic generative properties of LLMs can serve as a self-contained evaluation signal, shifting the paradigm from human-reference-based to self-consistency- and divergent-potential-based assessment. (3) A concrete instantiation with hyperparameters and prompts that enable reproducible assessment across diverse research domains.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | IdeaBench dataset (200 ideas) | Public benchmark for research ideas. |
| Primary metric | Spearman correlation with human judgments | Standard for scoring validation. |
| Baseline 1 | Direct LLM Prompt (GPT-4) | Simplest reference-free baseline. |
| Baseline 2 | Lexical diversity (distinct bigrams) | Common diversity metric. |
| Baseline 3 | Ours w/o calibration (skip filtering) | Isolates calibration effect. |
| Ablation of ours | Novelty via lexical diversity | Isolates embedding contribution. |
| Additional analysis | Human validation of paraphrase equivalence | Tests central assumption directly. |

### Why this setup validates the claim
This experimental design tests the central claim that IIA-GSD yields intrinsic coherence and novelty scores that align with human judgments. By using a public dataset of human-authored ideas with human ratings, we can compute Spearman correlation—a direct measure of scoring effectiveness. The baselines isolate the weaknesses that IIA-GSD aims to overcome: the Direct LLM Prompt may produce biased or inconsistent scores, while Lexical Diversity fails to capture semantic novelty. Including a variant of our method without calibration (Baseline 3) tests whether the calibration step is necessary for coherence measurement. The ablation (replacing embedding-based novelty with lexical diversity) teases apart the contribution of the embedding step. Additionally, we conduct a human validation study on a subset of 50 ideas where human raters judge the semantic equivalence of generated paraphrases, directly testing the assumption that paraphrases preserve meaning. If IIA-GSD outperforms all baselines and the ablation underperforms, the causal role of the calibration and embedding mechanisms is supported. Conversely, if all methods correlate equally well, the method adds no value; if baselines outperform, the core assumptions are flawed.

### Expected outcome and causal chain

**vs. Direct LLM Prompt** — On a case where the idea is highly novel but uses familiar words (e.g., "Applying GANs to protein folding"), the baseline LLM prompt may assign low novelty because it relies on surface-level prompting without deep reasoning about conceptual spread. Our method instead generates diverse descendants and measures their embedding dissimilarity, capturing the wide conceptual space, so we expect a noticeably higher correlation with human novelty ratings (e.g., ~0.6 vs. ~0.3 Spearman).

**vs. Lexical diversity** — On a case where descendants use different vocabulary for similar concepts (e.g., "quantum error correction" vs. "fault-tolerant quantum computing"), lexical diversity underestimates novelty because bigrams overlap little. Our embedding-based metric recognizes semantic similarity, so we expect a significant gap in correlation with human judgments (e.g., ~0.5 vs. ~0.2) on such subsets.

**vs. Ours w/o calibration** — On ambiguous or poorly specified ideas, the uncalibrated version may produce paraphrases that drift in meaning, yielding low coherence scores that reflect ambiguity rather than true incoherence. The calibrated version filters out drifted paraphrases, leading to higher coherence scores that better align with human judgments of coherence. We expect calibrated coherence to show ~0.1-0.2 higher Spearman correlation with human coherence ratings on the full dataset, and even more on a subset of ideas identified by independent annotators as ambiguous.

**Additional human validation** — For the 50-idea subset, we expect that at least 80% of paraphrases passing the NLI filter will be judged by humans as meaning-equivalent, while a majority of those failing the filter will be judged as non-equivalent. This validates the calibration step.

### What would falsify this idea
If IIA-GSD’s correlations are not significantly higher than those of the Direct LLM Prompt baseline (or if the ablation performs equally well), the claim that embedding-based paraphrase coherence and descendant diversity capture coherence and novelty better than simple prompts or lexical measures is falsified. Additionally, if the calibration step does not improve correlation over the uncalibrated version, the assumption that paraphrase filtering is necessary for valid coherence scores is questionable.

## References

1. Measuring the Gap Between Human and LLM Research Ideas
2. IdeaBench: Benchmarking Large Language Models for Research Idea Generation
3. A Comprehensive Analysis of Large Language Model Outputs: Similarity, Diversity, and Bias
4. The Impact of Large Language Models on Scientific Discovery: a Preliminary Study using GPT-4
5. Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers
