# CFG-Transfer: Zero-Shot Code Completion for Low-Resource Languages via Language-Agnostic Control Flow Prediction

## Motivation

Existing code completion approaches for low-resource languages, such as Teaching LLMs a Low-Resource Language (Pharo study), require manual curation of language-specific training data (e.g., collecting and filtering GitHub repositories) and continued pre-training, which is costly and does not generalize to new languages. This structural bottleneck stems from the assumption that adaptation needs large-scale, language-specific data, ignoring that control flow structures are largely invariant across languages. By decoupling language-agnostic structure from language-specific surface form, we can achieve zero-shot transfer without any curated data.

## Key Insight

Control flow structures are largely invariant across programming languages, enabling a universal CFG predictor trained on high-resource languages to serve as a structural backbone, while language-specific surface forms can be learned from unlabeled monolingual code alone.

## Method

### CFG-Transfer

**A. CFG-Transfer** is a two-stage framework for zero-shot code completion. The first stage is a **CFG predictor** (Frozen Universal CFG Predictor, FUCP) that maps any code token sequence to a linearized control flow graph represented as a token sequence of node types (e.g., `IF`, `WHILE`, `BLOCK`, `FUNC_DEF`) and edge markers. The second stage is a **surface generator** (Language-Specific Surface Predictor, LSSP) that predicts the next code token conditioned on both the CFG tokens and the preceding code tokens.

**B. How it works (pseudocode)**
```pseudocode
# Phase 1: Train CFG Predictor on High-Resource Languages
# Input: High-resource code snippets (Python, Java, JavaScript) with their ground-truth CFGs (linearized as token sequences)
# Model: Transformer encoder-decoder (6 layers, 512 dim, 8 heads)
# Hyperparameters: learning rate 1e-4, batch size 64, trained on 100M tokens until convergence (5 epochs)
for each (code_tokens, cfgtokens) in high_resource_data:
    loss = cross_entropy(decoder(cfgtokens | encoder(code_tokens)), cfgtokens)
    update FUCP parameters

# Phase 2: Train Language-Specific Surface Generator on Low-Resource Language (e.g., Pharo)
# Input: Unlabeled low-resource code (tokens only)
# Model: Small decoder-only transformer (4 layers, 256 dim, 4 heads) with causal attention
# Hyperparameters: learning rate 5e-5, batch size 32, trained on 1M tokens until convergence (10 epochs)
# Frozen FUCP provides CFG for each code snippet
for each code_tokens in low_resource_data:
    cfgtokens = FUCP(code_tokens)  # freeze FUCP
    # Prepend CFG tokens to the code prefix for autoregressive training
    # Training example: input = [CLS] + cfgtokens + code_tokens[:-1], target = code_tokens[1:]
    loss = next_token_prediction(LSSP([CLS] + cfgtokens + code_tokens[:-1]), code_tokens[1:])
    update LSSP parameters

# Inference (zero-shot)
# Input: partial code prefix (tokens) in low-resource language
# Step 1: cfgtokens = FUCP(prefix)  # predict CFG for the entire code (using completed future? Actually use only prefix? We use the prefix to predict CFG; but CFG prediction may need full code. To handle partial code, we pad with a special token or use a mask. We choose to train FUCP to predict CFG from incomplete code by randomly masking tokens during training.)
# Step 2: generate next token(s) using LSSP([CLS] + cfgtokens + prefix)
```

**C. Why this design** (≥80 words, 3 design decisions with trade-offs)
We chose to linearize the CFG as a token sequence over using a graph neural network (GNN) as an intermediate representation, because a token sequence can be directly fed into a standard transformer language model, avoiding complex graph-to-sequence architectures and enabling seamless integration with existing pre-trained models. The trade-off is that linearization loses some structural information (e.g., long-range control flow dependencies may be compressed into local tokens), but we mitigate this by using attention across the entire sequence. We freeze the CFG predictor after training on high-resource languages rather than fine-tuning it on the target language, because the target language has no labeled CFG data and fine-tuning would require expensive annotation or synthetic data. The cost is that CFG predictions may be imperfect for languages with novel control flow constructs (e.g., coroutines), but since control flow is abstract, we expect robustness: the predictor learns to map diverse syntactic patterns to universal structures. We train the surface generator as a small decoder (4 layers) rather than a larger model, because we want to avoid overfitting on limited unlabeled data and keep adaptation fast; the generator only needs to learn the mapping from a fixed CFG structure to surface tokens, which is a lower-entropy task than full language modeling, so a small model suffices. The downside is that the generator may lack capacity for very complex surface patterns, but we accept this for efficiency.

**D. Why it measures what we claim** (≥60 words, naming assumptions and failure modes)
The CFG predictor’s output quality on low-resource languages is measured by the downstream completion accuracy, which operationalizes **cross-lingual transfer** because we assume control flow structures are invariant; this assumption fails if the target language introduces constructs unseen in training (e.g., pattern matching in Rust), in which case the CFG predictor may produce erroneous structures, and the metric reflects the gap between predicted and actual structure. The surface generator’s next-token prediction loss on unlabeled data measures its ability to learn **language-specific surface forms** because we assume that conditioning on the CFG reduces token entropy to only lexical and syntactic residue; this assumption fails if the CFG is too coarse (e.g., missing variable declaration patterns), in which case the generator must memorize language-agnostic token co-occurrences, and the loss may not decrease as expected. The overall completion accuracy on a benchmark measures **zero-shot adaptation effectiveness** because we assume that the decomposed pipeline can match or exceed a model trained from scratch on scarce data; this assumption fails if the CFG predictor’s errors propagate and the generator cannot correct them, in which case the accuracy is lower than a simple n-gram baseline.

**E. Load-bearing assumption and calibration**
We assume that control flow structures are largely invariant across programming languages, enabling the frozen FUCP to produce accurate CFGs for any target low-resource language. To calibrate this assumption, we will evaluate FUCP on a held-out set of 512 manually annotated CFGs from the low-resource language (or heuristically extracted CFGs with error margin). If FUCP's node-type accuracy falls below 70%, we flag a potential failure of the method; this calibration step does not alter the training but provides a diagnostic for transfer validity.

## Contribution

(1) A two-stage framework (CFG predictor + surface generator) for zero-shot code completion in any low-resource language, requiring only unlabeled code of the target language. (2) Empirical evidence on three low-resource languages (e.g., Pharo, Lua, Racket) that the approach achieves competitive exact-match accuracy compared to language-specific fine-tuning baselines that use curated data, demonstrating that language-agnostic control flow is a effective transfer medium. (3) A method for linearizing control flow graphs into token sequences suitable for conditioning standard language models, enabling simple integration without specialized architectures.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Pharo code completion benchmark | Publicly available low-resource language data. |
| Primary metric | Exact Match Accuracy at token level | Measures exact surface form correctness. |
| Baseline 1 | Pre-trained CodeLlama (zero-shot) | Shows need for low-resource adaptation. |
| Baseline 2 | CodeLlama fine-tuned on Pharo w/o CFG | Isolates effect of CFG transfer. |
| Ablation-of-ours | LSSP with random CFG tokens | Tests importance of learned CFG. |

### Why this setup validates the claim

This design forms a falsifiable test of the claim that CFG-Transfer enables zero-shot code completion by decomposing structure and surface. Comparing against a zero-shot CodeLlama baseline shows whether any adaptation is needed; if our method matches or exceeds it, the decomposition provides benefit. The fine-tuned baseline (CodeLlama without CFG) directly tests whether conditioning on a predicted CFG adds value over standard language model fine-tuning on the same data. If our method outperforms it, the CFG must be capturing transferable control flow knowledge. The ablation with random CFG tokens tests whether the CFG predictor’s output is genuinely informative; if performance drops significantly, it confirms that predicted CFGs guide generation. Exact match accuracy is the right metric because it directly reflects token-level correctness, the ultimate goal of code completion, and any improvement must come from better structural or surface predictions.

### Expected outcome and causal chain

**vs. Pre-trained CodeLlama (zero-shot)** — On a case with a complex control flow (e.g., nested if-else inside a while loop), the zero-shot model produces syntactically invalid tokens (e.g., missing `end` keywords) because it has never seen Pharo’s syntax. Our method first predicts a CFG token sequence that captures the structural skeleton, then the surface generator learns to map that skeleton to Pharo-specific tokens during training. Thus, even without Pharo in pre-training, the generator can produce valid code by following the CFG. We expect a noticeable gap on examples involving control flow (e.g., ~20-30% higher accuracy) but parity on simple straight-line code (e.g., variable assignments) where structure is trivial.

**vs. CodeLlama fine-tuned on Pharo w/o CFG** — On a rare control flow pattern (e.g., a `do-while` loop emulated with `whileTrue` in Pharo), the fine-tuned model overfits to frequent patterns and mistakes the rare syntax because it has no structural guidance. Our method uses the FUCP to predict a universal CFG (e.g., a `DO_WHILE` node) that the fine-tuned generator has seen in training from high-resource languages, so it can generate the correct surface tokens even if the pattern is rare in Pharo. We expect a larger improvement on rare constructs (e.g., ~15% higher accuracy) than on common ones (e.g., ~5% higher).

**vs. LSSP with random CFG tokens** — On any code snippet, the random CFG provides meaningless structure, so the surface generator must rely solely on the code prefix, reverting to a standard language model. Since it is a small model (4 layers), it performs poorly. Our method with a learned CFG provides a strong structural prior, leading to much higher accuracy. We expect a large drop (e.g., 30-40% absolute) in exact match accuracy when using random CFG, confirming that the CFG predictor is crucial.

### What would falsify this idea

If the ablation with random CFG tokens achieves accuracy similar to the full method, or if the fine-tuned baseline (without CFG) outperforms our method overall, the central claim of CFG-Transfer as an effective decomposition is wrong.

## References

1. Teaching LLMs a Low-Resource Language: Enhancing Code Completion in Pharo
2. Mellum: Production-Grade in-IDE Contextual Code Completion with Multi-File Project Understanding
3. Enhancing Code Generation for Low-Resource Languages: No Silver Bullet
4. Evaluating Quantized Large Language Models for Code Generation on Low-Resource Language Benchmarks
5. Qwen2.5-Coder Technical Report
6. Context Composing for Full Line Code Completion
7. Full Line Code Completion: Bringing AI to Desktop
8. Multi-line AI-Assisted Code Authoring
