# Pixel-Cited Land-Change Claims with Mechanically Verified Evidence Graphs

## Motivation

EarthRecord shows that multimodal agents over four decades of satellite imagery still falter when answers require direct mapping from pixels, because its precomputed-embedding fusion decouples spatial evidence from the reasoning chain: the record moves answers from prior knowledge to pixels, but not yet to the real world. This is the meta-level residual gap across Earth AI, Landsat Collection 2, and CHOICE — pixel-to-answer causal grounding is unestablished whenever the fusion module hides which pixels and timestamps supported a claim, so no downstream check can distinguish a grounded answer from a plausible one.

## Key Insight

A land-change claim is causally grounded exactly when the pixel coordinates and temporal observations it cites are a minimal sufficient witness: removing any cited pixel changes the claim's verifiable value and adding any uncited pixel does not, so mechanical citation verification against raw imagery is equivalent to causal grounding rather than a proxy for it.

## Method

**What it is.** We introduce **CITE-GRAPH** (Citation-Invariant Traceable Evidence Graph), a module that replaces EarthRecord's precomputed-embedding fusion. Given raw multi-sensor Landsat/MODIS/VIIRS/Sentinel-2 records for a site and a land-change question, it outputs an answer only if the language model's emitted claim-to-pixel citations form a *minimal sufficient witness* that passes mechanical verification against the raw imagery.

**Load-bearing assumption (stated explicitly).** CITE-GRAPH assumes that the deterministic detector D is a faithful operationalization of the natural-language land-change claim c: that is, there exists a claim-to-detector-output translation T mapping c to a detector output value T(c), and D(C) = T(c) is equivalent to causal grounding of c by C. This assumption fails when c is a gradual or sub-pixel change outside D's breakpoint vocabulary (Zhu & Woodcock 2014 model such change as continuous harmonic/trend components that never produce a discrete breakpoint), in which case sufficiency reflects detector-vocabulary coverage rather than grounding. We make this assumption testable via a claim-to-detector-output translation table and a measured translation-failure rate (step 3e below), and we report all verified-citation rates separately for claims that pass translation and claims that fail it.

**How it works.**

```pseudocode
INPUT: site record R (raw bands, timestamps, cloud masks), question q
OUTPUT: accepted answer a* with evidence graph G, or REJECT

1. LLM proposes candidate claim c and citation set C = {(pixel_id, t_start, t_end, sensor)}
   with |C| <= K_max = 8 (hyperparameter: citation budget)
2. Build evidence graph G: nodes = cited (pixel, time-window) tuples; edges = temporal
   adjacency within a cited pixel and spatial adjacency between cited pixels at the
   same task-relevant date (adjacency radius r = 1 pixel)
3. MECHANICAL VERIFICATION against R (no embeddings, no learned scorer):
   a. OBSERVE(C): for each node, extract raw reflectance/NDVI at (pixel_id, t) after
      applying the agent's own cloud mask; if any node's observation is cloud-contaminated
      or out of the sensor's valid range, REJECT with reason
   b. SUFFICIENCY: re-run the deterministic land-change detector D (fixed CCDC-style
      breakpoint test, thresholds tau_NDVI = 0.05, tau_NBR = 0.1) on C alone; require
      D(C) = T(c), where T is the claim-to-detector-output translation defined by the
      translation table (step 3e). If not, REJECT
   c. MINIMALITY (counterfactual witness test): for every node n in C, re-run D(C \ {n});
      require D(C \ {n}) != T(c). For every pixel p in the 3x3 neighborhood of C not in C,
      re-run D(C U {p}); require D(C U {p}) = T(c). Additionally, for every pair {n_i, n_j}
      in C, re-run D(C \ {n_i, n_j}); require D(C \ {n_i, n_j}) != T(c) (joint-deletion check
      to catch pairwise-correlated redundancy)
   d. TEMPORAL CONSISTENCY: require every edge in G to connect observations whose
      dates differ by <= Delta_t = 1 acquisition cycle; flag any claim whose cited
      pixels span > 2 sensors without an explicit harmonization citation node
   e. TRANSLATION VALIDATION: apply the claim-to-detector-output translation table T
      (a fixed lookup from claim predicates to detector output fields: e.g., 'forest loss'
      -> D's breakpoint type = negative NDVI breakpoint; 'burn' -> NBR breakpoint;
      'stable' -> no breakpoint) to map c to T(c). If c has no entry in T (i.e., c is
      outside D's vocabulary), record a translation failure and REJECT with reason
      'outside detector vocabulary'; report the translation-failure rate per task
4. If all checks pass: emit a* = c with G serialized as a citation table; else return
   REJECT plus the failing check and the minimal counterexample found in 3b/3c
5. Calibration probe: maintain a held-out set of 40 verified (claim, citation) pairs per
   task; if the LLM's self-reported citation confidence and the probe's pass rate diverge
   by > 0.15, down-weight the LLM's proposed citations and re-propose once (max 1 retry)
```

**Claim-to-detector-output translation table T (step 3e).** The table is a fixed, hand-authored mapping from the claim predicates the benchmark uses to detector output fields, applied before sufficiency and minimality checks:

| Claim predicate (from c) | Detector output field T(c) |
|---|---|
| forest loss / clearing | negative NDVI breakpoint (CCDC break type = -1) |
| burn / fire scar | negative NBR breakpoint |
| vegetation gain / regrowth | positive NDVI breakpoint |
| stable / no change | no breakpoint in window |
| gradual / sub-pixel change | **no entry** -> translation failure |

Claims whose predicate has no entry in T are counted as translation failures and excluded from the verified-citation numerator; the translation-failure rate is reported per task and per subset.

**Why this design.** We chose a *minimal sufficient witness* over a learned faithfulness head because the underlying detector D is deterministic and the imagery is fixed, so sufficiency and minimality are checkable by re-execution rather than by a learned proxy; the cost is that we can only accept claims expressible through D's breakpoint vocabulary, which excludes gradual qualitative change — and we make this cost measurable via the translation table and failure rate rather than hiding it. We chose *raw-pixel re-execution* over EarthRecord's frozen-embedding fusion because embeddings decouple the evidence from the reasoning chain — the very failure EarthRecord documents — accepting the cost of repeated detector calls (up to |C| + |C|(|C|-1)/2 + 9 per claim with the joint-deletion check) in exchange for a citation that is mechanically falsifiable. We chose a *citation budget K_max = 8* over unbounded citations because minimality becomes vacuous when the model can cite the whole scene; the cost is that some genuine multi-pixel changes require the agent to summarize rather than cite every pixel, which we handle by requiring the summary to name the detector's spatial extent. We chose a *calibration probe* over raw LLM log-probs as the retry trigger because log-prob calibration for task correctness is known to be unreliable (Kadavath et al.); the cost is a fixed per-task probe set that must be refreshed when the sensor mix changes. A domain expert would not describe CITE-GRAPH as a variant of retrieval-augmented generation or of EarthRecord's fusion: RAG retrieves passages, whereas CITE-GRAPH *intervenes* on the cited pixel set and requires the intervention to change the detector's output — a counterfactual operation retrieval cannot perform.

**Why it measures what we claim.** `D(C) = T(c)` (sufficiency) measures *causal grounding* because we assume the deterministic detector D is a faithful operationalization of the land-change claim c via the translation table T, so re-executing D on the cited pixels alone is equivalent to asking whether those pixels are causally responsible for c; this assumption fails when c is a gradual or sub-pixel change outside D's breakpoint vocabulary, in which case sufficiency reflects detector-vocabulary coverage rather than grounding — which is why we report the translation-failure rate separately. `D(C \ {n}) != T(c)` for all n and `D(C \ {n_i, n_j}) != T(c)` for all pairs (minimality) measures *non-redundant evidential necessity* because we assume the 3x3 neighborhood and single-node and pairwise deletion are the relevant counterfactual interventions, so a node whose deletion preserves c is evidentially redundant; this assumption fails when two cited pixels are spatially correlated such that deleting either alone preserves c but deleting both does not, in which case minimality reflects pairwise rather than joint redundancy — which is why we add the joint-deletion check. The cloud-contamination and temporal-consistency checks measure *observational validity* because we assume the agent's own cloud mask and the 1-cycle temporal window are the correct validity conditions, so a citation passing them is an actual observation rather than a hallucinated one; this assumption fails when the agent's cloud mask is itself wrong, in which case validity reflects mask quality rather than observation existence. The calibration probe measures *citation reliability* because we assume the held-out 40-pair set is exchangeable with the test claims, so a probe pass rate tracks the citation's correctness probability; this assumption fails under sensor-mix shift, in which case the probe reflects stale calibration rather than current reliability.

## Contribution

['CITE-GRAPH: a traceable evidence-graph module that replaces precomputed-embedding fusion with a minimal-sufficient-witness verification procedure over raw pixels, including a counterfactual minimality test that retrieval and embedding-fusion methods cannot express.', 'A design principle — pixel-to-answer causal grounding is equivalent to minimal sufficient witness over a deterministic detector — that turns faithfulness from a learned property into a mechanically checkable one, with an explicit statement of when the equivalence fails (gradual change, correlated pixels, mask errors).', "A per-claim citation table format and a calibration-probe protocol for detecting when an LLM's self-reported citation confidence diverges from mechanically verified pass rates, applicable to any Earth-observation agent that emits land-change claims."]

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
|------|--------|-----------|
| Dataset | EarthRecord 56-site, 238-question benchmark | Same testbed as baseline; enables direct comparison |
| Primary metric | Mechanically-verified citation pass rate (fraction of questions whose emitted citation set passes OBSERVE, SUFFICIENCY, MINIMALITY incl. joint-deletion, and TEMPORAL CONSISTENCY) | Directly measures falsifiable grounding, not answer accuracy |
| Secondary metric | Translation-failure rate (fraction of claims with no entry in T) and verified-citation pass rate computed separately on the translation-pass subset and the translation-fail subset | Makes the load-bearing claim-to-detector-output assumption measurable rather than hidden |
| Secondary metric | Runtime and detector-call count per accepted claim (wall-clock seconds, number of D re-executions) | Tests feasibility on the full 238-question benchmark |
| Baseline | EarthRecord frozen-embedding fusion agent | Tests whether raw-pixel re-execution beats embedding decoupling |
| Baseline | Best closed-book MLLM from EarthRecord | Tests whether gains come from evidence access, not model scale |
| Baseline | Instrument-pairs control agent | Tests whether gains come from citation structure, not sensor mix |
| Baseline | Leave-one-out occlusion attribution agent (cites the top-K pixels whose occlusion most changes the frozen-embedding similarity score) | Tests whether minimal-sufficient-witness verification is a relabeling of known attribution |
| Baseline | Learned faithfulness head (a logistic-regression scorer trained on 40 held-out verified pairs per task to predict citation correctness from embedding features) | Tests whether mechanical re-execution beats a learned faithfulness proxy |
| Ablation-of-ours | CITE-GRAPH w/o minimality check | Isolates whether counterfactual witness test drives reliability |
| Ablation-of-ours | CITE-GRAPH w/o joint-deletion check (single-node deletion only) | Isolates whether pairwise-correlated redundancy matters |
| Ablation-of-ours | CITE-GRAPH with K_max = 4 and r = 0 (reduced budget and no spatial neighborhood) | Pre-registered reduced-budget configuration to demonstrate the method runs within budget |

### Why this setup validates the claim

The central claim is that a claim-to-pixel citation that survives mechanical re-execution is causally grounded, whereas EarthRecord's frozen-embedding fusion produces citations that are decoupled from the reasoning chain. The EarthRecord benchmark is the right testbed because it supplies raw multi-sensor records and withheld references, so a verified citation can be checked against the actual imagery rather than against a learned faithfulness proxy. The primary metric — mechanically-verified citation pass rate — is chosen because it is the only observable that directly instantiates the claim: an answer can be right for the wrong reason, but a citation that passes sufficiency, minimality (including joint-deletion), cloud-validity, and temporal-consistency checks cannot. The translation-failure rate is reported alongside it because the load-bearing assumption is that D faithfully operationalizes c via T; if a large fraction of claims have no entry in T, the verified-citation rate applies only to the subset inside D's vocabulary, and we report both subsets so the reader can see the scope of the claim. The EarthRecord fusion baseline tests the sub-claim that raw-pixel re-execution beats embedding fusion; the closed-book baseline tests that gains are not merely model scale; the instrument-pairs control tests that gains are not an artifact of sensor pairing; the leave-one-out occlusion and learned-faithfulness-head baselines test that the minimal-sufficient-witness partition is strictly different from and better-calibrated than known attribution methods. The minimality and joint-deletion ablations are the load-bearing tests: if removing the counterfactual witness check (or the joint-deletion check) does not degrade verified-citation rate, then the minimal-sufficient-witness mechanism is not what produces the effect, and the central claim collapses to ordinary retrieval.

### Expected outcome and causal chain

**vs. EarthRecord frozen-embedding fusion agent** — On a case where a land-change claim depends on a subtle NDVI breakpoint at a single pixel (e.g., a gradual clearing that only crosses tau_NDVI in one acquisition), the baseline produces a plausible citation set drawn from embedding-nearest neighbors that may include the right pixel but also redundant neighbors, because its frozen embeddings decouple the cited evidence from the reasoning chain and cannot test whether deleting a node changes the detector output. Our method instead re-executes the deterministic CCDC-style detector on the cited pixel set and rejects any citation whose node deletion preserves the claim, so we expect a pronounced gap in verified-citation pass rate on single-pixel-breakpoint questions, with parity on coarse multi-pixel changes where embedding neighbors happen to be sufficient.

**vs. Best closed-book MLLM from EarthRecord** — On a case where the question asks about a change at a specific site and date, the closed-book baseline produces an answer with no pixel citations at all (or hallucinated pixel IDs), because it has no access to the raw record and must rely on parametric memory. Our method instead requires every claim to carry a citation set that passes OBSERVE and SUFFICIENCY against the raw bands, so the closed-book baseline cannot even enter the verified-citation metric; we expect it to score near zero on verified citations while our method scores substantially above zero, with the gap largest on questions whose answers are not inferable from site metadata alone.

**vs. Instrument-pairs control agent** — On a case where the sensor mix includes a VIIRS night-lights band paired with Landsat optical, the instrument-pairs control produces citations that cross sensors without an explicit harmonization node, because its pairing mechanism treats sensor identity as a nuisance rather than a validity condition. Our method instead flags any claim whose cited pixels span more than two sensors without a harmonization citation node, and rejects cloud-contaminated or out-of-range observations, so we expect our verified-citation rate to exceed the control specifically on multi-sensor questions, with the two converging on single-sensor questions where temporal consistency is trivially satisfied.

**vs. Leave-one-out occlusion attribution agent** — On a case where two spatially adjacent pixels both show the breakpoint, the occlusion baseline cites both because occluding either one alone reduces the embedding similarity score, so its citation set is not minimal under joint deletion. Our method's joint-deletion check rejects the pair when D(C \ {n_i, n_j}) still equals T(c), so we expect our verified-citation rate to exceed the occlusion baseline specifically on questions with spatially correlated breakpoints, with parity on isolated single-pixel changes where occlusion and minimality coincide.

**vs. Learned faithfulness head** — On a case where the held-out 40-pair probe set is drawn from a different sensor mix than the test questions, the learned faithfulness head's accept/reject partition drifts because its logistic-regression weights are fit to stale embedding features, whereas our method's re-execution checks are sensor-agnostic. We expect our verified-citation rate to exceed the learned head on sensor-mix-shifted questions, with the two converging on in-distribution questions where the probe set is exchangeable.

### What would falsify this idea

If the minimality ablation (or the joint-deletion ablation) matches full CITE-GRAPH on verified-citation pass rate, or if our gain over the EarthRecord fusion baseline is uniform across single-pixel-breakpoint and coarse-change subsets rather than concentrated where embedding decoupling is predicted to fail, then the counterfactual witness mechanism is not what produces the effect and the central claim is wrong. Additionally, if the translation-failure rate is high enough that the verified-citation rate on the translation-pass subset is computed over a small and unrepresentative fraction of questions, the claim that CITE-GRAPH closes the EarthRecord residual gap is not supported for the benchmark as a whole.

## References

1. EarthRecord: From Pixels to Land Change with Multimodal Agents over Four Decades of Satellite Imagery
