# Document-Ablation Invariance Certificates for Culture-Conditioned Generation

## Motivation

Existing attribution of culture-conditioned generations to pretraining corpora is frequency-based: MEMOED (Attributing Culture-Conditioned Generations to Pretraining Corpora) infers that an output is memorized whenever its symbols are frequent in the pretraining corpus, and it can only observe correlations in a fixed, already-trained model. This conflates frequency with causal dependence, because a frequently repeated cultural symbol may be produced by a generalizable cultural schema rather than by any specific document, and MEMOED has no way to tell these apart since it never perturbs the corpus. The field's trajectory — K2-Horizon now releases full pretraining data and intermediate checkpoints — makes causal intervention feasible, yet no method specifies which documents to remove, how to hold everything else fixed, or how to convert an output change into a per-document attribution verdict rather than a single aggregate frequency statistic.

## Key Insight

If a culture-conditioned output is causally attributable to a specific document, then removing exactly that document from the pretraining corpus must change the output, while removing a matched control document of identical frequency and length must not — so the difference between the two perturbation effects is an invariance-based attribution signal that frequency alone cannot produce.

## Method

**What it is.** We introduce **DAIC** (Document-Ablation Invariance Certificates), a procedure that takes a culture-conditioned prompt set, a released K2-Horizon checkpoint plus its pretraining corpus, and a target document set, and returns for each (culture, document) pair a signed *ablation effect* together with an invariance certificate stating whether the culture-conditioned output is attributable to that document or to a generalizable schema.

**Load-bearing assumption (stated explicitly).** DAIC assumes that continue-training the released checkpoint for S=200 steps at lr=1e-5 on the corpus-minus-one-document yields an output change that is a valid, monotone proxy for the document's true causal contribution to the culture-conditioned generation. This is a *partial-erasure* proxy: it is a lower bound on true causal dependence, not an exact counterfactual, and it is informative only when the effect exceeds the stochastic churn of continue-training itself. The matched-control subtraction and multi-seed averaging are designed to absorb that churn; the certificate logic treats the residual as a lower bound. This assumption fails when (i) a culture's documents are near-duplicates, so removing one leaves siblings to regenerate the same symbol and the effect collapses to zero even though the culture is document-driven, or (ii) the single document's marginal contribution is smaller than the trajectory noise of 200 continue-training steps, in which case the 2-std threshold is a noise-calibration claim rather than a causal-necessity claim. We therefore report, alongside every certificate, the near-duplicate density of the target document's cluster and an explicit non-identifiability flag for high-redundancy cultures.

**How it works.**

```pseudocode
Input: checkpoint C (K2-Horizon 0.9B dense), corpus D, culture prompts P = {(p, c)} for 110 cultures,
       target docs T_c subset of D containing culture-c symbols, N_control = 5 matched controls per doc
Hyperparams: steps S = 200, lr = 1e-5, batch = 64, seed set Sigma = {1,2,3}, effect threshold tau = 0.05,
             redundancy threshold rho = 0.8 (MinHash Jaccard for near-duplicate cluster membership)

1. BASELINE: for each prompt (p,c) in P, generate G0(p,c) from C; record symbol set Sym0(p,c)
   and per-token log-prob l0(p,c).

2. DOCUMENT SELECTION: for each target doc d in T_c, select N_control control docs from D\T_c
   matched to d on (token length +/-10%, corpus frequency +/-5%, language, topic cluster id).
   Controls are the frequency-matched counterfactual: they carry the same frequency mass but
   are not the hypothesized source. Additionally, compute the near-duplicate cluster of d via
   MinHash (128 permutations, Jaccard >= rho) and record its size |cluster(d)| and density.

3. ABLATION (per doc x in {d} U controls, per seed s in Sigma):
   a. continue-train C for S steps on D \ {x} with seed s  ->  C_{-x,s}
   b. for each prompt (p,c): generate G_{-x,s}(p,c); record Sym_{-x,s}(p,c) and l_{-x,s}(p,c)

4. EFFECT: ablation_effect(d, p, c) = mean_s [ Jaccard(Sym0, Sym_{-d,s}) complement ]
   minus mean over controls x of mean_s [ Jaccard(Sym0, Sym_{-x,s}) complement ]
   (i.e., target-document removal effect minus matched-control removal effect).

5. CERTIFICATE: doc d is CERTIFIED as a causal source for (p,c) iff
   effect(d,p,c) > tau AND effect(d,p,c) > 2 * std over controls of their effects
   AND |cluster(d)| == 1 (no near-duplicate siblings).
   Otherwise d is DECLARED non-attributable (schema-driven or redundant-document-driven) for that prompt,
   with the reason recorded (low-effect, high-variance, or high-redundancy).

6. AGGREGATE: per culture c, report fraction of certified docs and the certified-effect distribution;
   compare to MEMOED's frequency rank for the same culture. Report certified-fraction separately for
   high-redundancy (|cluster(d)| > 1) and low-redundancy (|cluster(d)| == 1) subsets.
```

**Why this design.** We chose *continue-training from the released checkpoint on the corpus-minus-one-document* over full retraining from scratch because full retraining would confound the document removal with a fresh optimization trajectory, making the output change uninterpretable; the cost we accept is that continue-training only partially erases a document's influence, so our effects are a lower bound on true causal dependence rather than an exact counterfactual. We chose *frequency- and length-matched control documents* over random controls because the whole point is to separate causal dependence from frequency, and a random control would reintroduce exactly the frequency confound MEMOED already suffers from; the cost is that matching shrinks the eligible control pool for rare cultures, which we handle by reporting an explicit insufficient-controls flag rather than silently dropping those cultures. We chose *symbol-set Jaccard complement on generated outputs* over log-probability differences because symbol-set change is the same observable MEMOED uses, keeping the comparison to MEMOED on the same surface; the cost is that it discards graded probability shifts, which we recover only as a secondary reported quantity. We chose *a multi-seed mean with a control-std threshold* over a single-seed difference because continue-training is stochastic and a single seed cannot distinguish a real ablation effect from optimization noise; the cost is a 3x compute multiplier. A domain expert would not describe DAIC as a variant of MEMOED: MEMOED is a passive frequency-correlation statistic computed on one fixed model, whereas DAIC is an interventional procedure whose output is a per-document certificate that can contradict MEMOED's frequency ranking, and the matched-control subtraction is the operation MEMOED structurally cannot perform.

**Why it measures what we claim.** The quantity `ablation_effect(d,p,c)` measures *causal attribution of the culture-conditioned output to document d* because the assumption that continue-training on corpus-minus-d isolates d's contribution holds only when the matched controls absorb all other frequency-correlated variation; this assumption fails when a culture's documents are near-duplicates, in which case removing one leaves the others to regenerate the same symbol and the effect collapses to zero even though the culture is document-driven. The quantity `Jaccard(Sym0, Sym_{-d,s}) complement` measures *output dependence on the removed document* because it assumes generated symbol sets are a sufficient statistic of the culture-conditioned behavior; this assumption fails when the model produces the same symbols via paraphrase, in which case the metric under-reports dependence and reflects surface-form stability instead. The quantity `effect(d,p,c) > 2 * std over controls` measures *certified causal necessity rather than noise* because it assumes the control-effect distribution is a valid null for stochastic continue-training; this assumption fails when controls are systematically shorter or rarer than targets despite matching, in which case the threshold certifies length artifacts as causation, which is why matching tolerance is reported alongside every certificate. The quantity `|cluster(d)| == 1` in the certificate condition measures *non-redundancy of the target document* because it assumes MinHash Jaccard >= rho identifies near-duplicate siblings that can regenerate the same symbol; this assumption fails when paraphrased duplicates fall below the MinHash threshold, in which case a redundant document may be certified as a unique causal source.

## Contribution

(1) DAIC, a document-ablation invariance procedure with frequency/length-matched controls and a multi-seed certificate criterion that converts corpus perturbation into per-document causal-attribution verdicts for culture-conditioned generation. (2) The empirical design principle that frequency-matched control subtraction is necessary to separate memorization from schema generalization, operationalized as a certificate that can disagree with frequency-based ranking. (3) An open release of the matched-control document sets, perturbation seeds, and per-culture certificate tables built on K2-Horizon's released corpus and checkpoints.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | K2-Horizon pretraining corpus + 110-culture prompt set, stratified sample of 30 target documents (15 high-redundancy, 15 low-redundancy; balanced high/low frequency) | Released corpus enables true corpus-minus-one ablation; stratified sample cuts compute 10x |
| Primary metric | Certified-document fraction per culture, reported separately for high- and low-redundancy subsets | Directly measures causal attribution, not correlation; redundancy-conditioned |
| Baseline | MEMOED frequency rank | Passive frequency statistic; the confound to beat |
| Baseline | Random-control ablation | Tests whether matching controls actually matter |
| Baseline | No-ablation checkpoint stability | Tests whether effects exceed continue-training noise |
| Ablation-of-ours | DAIC without matched controls | Isolates the frequency-confound removal step |
| Ablation-of-ours | DAIC without redundancy flag (certify all docs regardless of |cluster(d)|) | Isolates the near-duplicate collapse failure mode |

### Why this setup validates the claim

The central claim is that culture-conditioned outputs can be causally attributed to specific pretraining documents, and that this attribution diverges from what frequency-correlation statistics like MEMOED would predict. The design is falsifiable only if each baseline attacks a distinct sub-claim. MEMOED tests whether our certificates merely reproduce its frequency ranking: if certified-document fraction tracks MEMOED's frequency rank monotonically with no dissociations, the interventional machinery adds nothing. The random-control ablation tests whether frequency/length matching is doing real work: if random controls yield the same effect distribution as matched controls, our matched-control subtraction is decorative. The no-ablation stability baseline tests whether continue-training noise alone produces symbol-set churn comparable to our effects, which is the null our 2-std threshold presumes. The ablation-of-ours (DAIC minus matched controls) is the load-bearing step: it should degrade toward the random-control baseline if matching is what separates causation from frequency. The redundancy-flag ablation tests whether the near-duplicate collapse is empirically bounded: if removing the flag changes certified-fraction only on the high-redundancy subset, the flag is doing its job. Symbol-set Jaccard complement is the right primary metric because it is the exact observable MEMOED uses, so any divergence is attributable to the interventional design rather than a metric change; graded log-prob shifts are secondary, not the falsification surface. The stratified sample of 30 documents (15 high-redundancy, 15 low-redundancy, balanced high/low frequency) is pre-registered and sized to test the dissociation claim at 10x lower compute than the full 110-culture sweep.

### Expected outcome and causal chain

**vs. MEMOED frequency rank** — On a high-frequency culture whose symbols are concentrated in a few near-duplicate documents, MEMOED ranks it as strongly memorized because output frequency correlates with corpus frequency. Our method instead removes one document at a time and finds the effect collapses to zero because sibling near-duplicates regenerate the same symbols, so we expect certified-fraction to be low precisely where MEMOED is high, producing a visible dissociation on the high-frequency, high-redundancy subset but agreement on the low-redundancy subset. The redundancy-conditioned reporting makes this prediction directly testable: certified-fraction on the low-redundancy subset should track MEMOED, while on the high-redundancy subset it should not.

**vs. Random-control ablation** — On a rare culture with few eligible documents, random controls are drawn from a very different frequency and length regime, so removing a random document perturbs the model as much as removing the target. The baseline therefore reports large but meaningless effects everywhere. Our matched controls carry the same frequency mass, so target removal should stand out above the control distribution; we expect a clear separation between target and control effects under matching that vanishes under random controls, concentrated in the rare-culture subset.

**vs. No-ablation checkpoint stability** — On a culture-conditioned prompt where continue-training is stochastic, repeated continue-training without removing any document still shifts generated symbol sets, so a single-seed difference cannot distinguish signal from optimization noise. Our multi-seed mean with a control-std threshold treats this churn as the null, so we expect certified documents to be a strict subset of those flagged by a naive single-seed difference, with the discarded ones concentrated among low-effect, high-variance prompts.

**vs. DAIC without redundancy flag** — On the high-redundancy subset, certifying all documents regardless of cluster size should inflate certified-fraction relative to the flagged version, because near-duplicate siblings regenerate the same symbols and the effect collapses to zero. We expect the gap between flagged and unflagged certified-fraction to be concentrated entirely in the high-redundancy subset, with near-zero gap on the low-redundancy subset, confirming that the redundancy flag bounds the collapse failure mode rather than suppressing real signal.

### What would falsify this idea

If certified-document fraction correlates near-perfectly with MEMOED's frequency rank across all 110 cultures with no dissociation on the high-redundancy subset, then DAIC is not measuring causal attribution beyond frequency and the central claim is wrong. Equally fatal: if matched-control and random-control ablations yield indistinguishable effect distributions, the matching operation — the mechanism the whole argument rests on — is doing no work.

## References

1. K2-Horizon: A Frontier Fleet of Open Source Language Models
