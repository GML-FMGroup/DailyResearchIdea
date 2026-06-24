# Self-Supervised Skill Induction from Unlabeled Lab Notebooks via Contrastive Learning on Repeated Experiments

## Motivation

Existing methods for skill induction from scientific texts, such as Notes2Skills or fine-tuning LLMs for protocol extraction (Structured information extraction from scientific text with large language models), rely on annotated data or assume access to successful trajectories (Agent Workflow Memory). However, lab notebooks contain abundant unlabeled records with inherent repetition: multiple runs of the same protocol under varying equipment. Current approaches fail to exploit this structural redundancy, requiring expensive human annotation for each new domain. This reliance on labeled data creates a bottleneck, as annotated scientific text is scarce and costly to produce.

## Key Insight

Repeated experimental runs in unlabeled lab notebooks naturally form positive pairs under data augmentation that masks equipment-specific details, enabling contrastive learning to discover equipment-invariant procedure embeddings without any human labels.

## Method

**Method: Contrastive Skill Induction from Notebooks (CSIN)**

**(A) What it is**: CSIN takes as input a collection of unlabeled lab notebooks (timestamped entries with text) and outputs a skill encoder that maps any procedural snippet to a fixed-dimensional vector representing the underlying experimental procedure, invariant to equipment variation.

**(B) How it works**:
```pseudocode
Input: Unlabeled notebook entries E = {e_1,...,e_N}, each with timestamp t_i and text t_i
Hyperparameters: window size w (e.g., 30 min), similarity threshold τ (e.g., 0.7, calibrated as below), embedding dimension d (e.g., 128), temperature τ_c (e.g., 0.1)
Output: Encoder f: text → R^d

1. Temporal Clustering: Group entries into sessions by time gaps > threshold Δt=30 min. Each session S_j contains sequential entries.
2. Topical Clustering within Sessions: For each session, compute TF-IDF vectors for each entry, then run agglomerative clustering with cosine distance. The threshold τ is chosen via calibration on a held-out set of 100 manually annotated positive/negative pairs drawn from 5 notebooks, selecting τ that maximizes F1. Clusters within a session represent likely repeated protocol steps.
   *Assumption: TF-IDF cosine similarity with calibrated threshold τ reliably identifies entries describing the same protocol step within a session. This assumption may fail when generic phrases (e.g., 'add 5ml') appear across distinct steps, causing false positives. The calibration step mitigates this by selecting τ based on verified pairs.*
3. Positive Pair Construction: For each entry e in cluster C, pair it with another entry e' in same cluster from same session. Also augment by replacing equipment mentions (identified by a regex-based entity tagger) with a special token [MASK]. These form positive pairs (augmented(e), augmented(e')).
4. Contrastive Pretraining: Train encoder f (a small Transformer: 4 layers, 8 heads, hidden size 256) using NT-Xent loss:
   L = -log( exp(sim(z_i, z_j)/τ_c) / (exp(sim(z_i, z_j)/τ_c) + sum_{negatives} exp(sim(z_i, z_neg)/τ_c) )
   where z_i = f(augmented(e)), z_j = f(augmented(e')), and negatives are drawn from other clusters/sessions.
5. Output: Skill encoder f.
```

**(C) Why this design**: We chose temporal clustering (step 1) over manual annotation because time gaps naturally separate distinct tasks, leveraging the implicit structure of notebook organization; the trade-off is that sessions with no clear time breaks may be mismerged, but professional notebooks typically have session demarcations. We chose TF-IDF within sessions (step 2) over deep clustering because it is fast and interpretable for short entries; the cost is poorer handling of synonymy, but scientific protocols use domain-specific jargon with limited synonymy, making it acceptable. We chose equipment masking as the only augmentation (step 3) rather than more aggressive transformations like paraphrasing because we want to force invariance specifically to equipment while preserving procedure details; the trade-off is that if equipment is correlated with procedure, the encoder may falsely treat procedure variations as equipment artifacts, but repeated runs with varying equipment break that correlation. We selected NT-Xent loss (step 4) over triplet loss because it naturally handles multiple positives via batch proxies; the downside is sensitivity to batch composition, which we mitigate by using large batch sizes (256) and careful negative sampling from different sessions.

**(D) Why it measures what we claim**: The temporal clustering (step 1) operationalizes the concept of 'session' (a contiguous block of experimental activity) because we assume that time gaps exceeding 30 min indicate a shift in task; this assumption fails when multi-tasking or interruptions occur, in which case sessions may contain multiple unrelated procedures. The topical clustering (step 2) operationalizes 'same protocol step' because we assume that TF-IDF cosine similarity above the calibrated threshold captures semantic equivalence of steps; this assumption fails when different steps share vocabulary (e.g., 'add solvent' appears in many steps), resulting in false positives—then the metric reflects lexical overlap rather than functional equivalence. The equipment masking augmentation (step 3) operationalizes 'equipment invariance' because we assume that masking equipment names forces the encoder to ignore them during contrastive learning; this assumption fails when equipment is essential for distinguishing procedures (e.g., different equipment implies different protocol), in which case the encoder may discard discriminative information—the metric then measures invariance to spurious but actually relevant equipment. The NT-Xent loss (step 4) measures 'embedding similarity for positive pairs' because we assume that augmented versions of the same underlying procedure should be closer in embedding space than negatives; this assumption fails when augmentations distort the procedure semantics (e.g., masking critical equipment), in which case the loss measures similarity of corrupted representations.

## Contribution

(1) A novel self-supervised framework (CSIN) for skill induction from unlabeled lab notebooks that leverages temporal and topical clustering to identify positive pairs without human labels. (2) Empirical design principle that equipment-masking augmentation induces invariance to equipment variation, enabling the encoder to focus on procedural content. (3) An open-source implementation and demo on synthetic notebook data for reproducibility.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | Lab notebook corpus (50 protocols, 1000 sessions) + Cross-domain: Physics notebooks (10 protocols, 200 sessions) | Realistic unlabeled procedural text + test generality |
| Primary metric | Mean Reciprocal Rank (MRR) | Tests retrieval of same protocol step |
| Secondary metric | Cosine similarity between same-procedure different-equipment pairs vs. different-procedure pairs | Directly measures equipment invariance |
| Baseline 1 | TF-IDF + cosine similarity | No temporal or contrastive learning |
| Baseline 2 | Average Word2Vec embeddings | Static semantic representation |
| Ablation | CSIN without equipment masking | Isolates effect of masking |
| User study | 5 domain scientists rate top-3 retrieved steps for relevance | Validates practical utility of skills |
| Sensitivity analysis | Vary Δt ∈ {15,30,60} min, τ ∈ [0.5,0.9] step 0.1 on held-out validation set | Assess robustness of clustering |

### Why this setup validates the claim

The combination of an unlabeled notebook dataset with known protocol structure and a retrieval metric that requires distinguishing steps despite equipment variation creates a direct test of the claim. TF-IDF baseline shows whether lexical overlap suffices; Word2Vec baseline tests if static embeddings capture procedure similarity; the ablation isolates the contribution of equipment masking. If CSIN achieves higher MRR, especially on instances with equipment changes, it confirms that contrastive learning from temporal and topical clusters induces equipment-invariant skill embeddings. The cross-domain dataset (physics) tests generality beyond the original domain. The user study with domain scientists provides qualitative evidence that the induced skills are practically useful for tasks like protocol summarization or error detection. The sensitivity analysis ensures that performance is not overly dependent on precise hyperparameter choices. The targeted invariance test directly measures whether embeddings of same-procedure different-equipment pairs are closer than different-procedure pairs; a high ratio indicates true invariance.

### Expected outcome and causal chain

**vs. TF-IDF + cosine similarity** — On a case where two entries describe the same centrifugation step but one uses "centrifuge" and the other "microcentrifuge", TF-IDF similarity will be low because the terms differ. Our method masks both equipment mentions before feeding to the encoder, forcing the representation to ignore equipment identity. We expect CSIN to retrieve such pairs with MRR >0.6 while TF-IDF stays below 0.3.

**vs. Average Word2Vec embeddings** — On a case where two steps are semantically equivalent but worded differently (e.g., "add 5ml buffer" vs "pour 5ml of buffer"), Word2Vec averages may not align them if vocabulary differs. Our method first clusters entries within a session using TF-IDF (which handles some variation) to form positive pairs, then applies contrastive learning to pull them together. We expect CSIN to have MRR ~0.5 higher on such pairs.

**vs. CSIN without equipment masking** — On a case where two identical steps use different equipment (e.g., different brand centrifuge), the ablation model may treat them as different because equipment tokens are present. Our full model masks equipment, so it sees them as same. We expect CSIN to outperform ablation by at least 0.2 MRR on steps with equipment variation, but perform similarly on steps without.

**Targeted invariance test** — For pairs of entries with identical procedure but different equipment (e.g., "centrifuge at 5000g" vs "microcentrifuge at 5000g"), we compute cosine similarity of their embeddings. We expect CSIN to yield similarity >0.7, while ablation yields <0.5. For pairs with different procedures and same equipment, similarity should be <0.3 for both.

### What would falsify this idea

If CSIN's MRR improvement over the ablation is uniform across all test instances rather than concentrated on equipment-varying ones, or if CSIN fails to beat the TF-IDF baseline, then the claim that it learns equipment-invariant skill embeddings from notebooks would be unsupported. Additionally, if the targeted invariance test shows high similarity for both same-procedure different-equipment and different-procedure pairs, or if the cross-domain experiment fails to show significant improvement, the generality claim would be weakened.

## References

1. Notes2Skills: From Lab Notebooks to Certainty-Aware Scientific Agent Skills
2. Structured information extraction from scientific text with large language models
3. ProtoCode: Leveraging large language models (LLMs) for automated generation of machine-readable PCR protocols from scientific publications.
4. Agent Workflow Memory
5. ChatGPT Chemistry Assistant for Text Mining and the Prediction of MOF Synthesis
6. Llama 2: Open Foundation and Fine-Tuned Chat Models
7. The Laboratory Automation Protocol (LAP) Format and Repository: a platform for enhancing workflow efficiency in synthetic biology
8. Evaluating the Usefulness of a Large Language Model as a Wholesome Tool for De Novo Polymerase Chain Reaction (PCR) Primer Design
