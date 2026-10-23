# Drift-Aware Evidence Validation and Staleness Monitoring for Open-Web Deep Search

## Motivation

OpenDS-World and similar open-web benchmarks rely on independent human verification of evidence validity and answer uniqueness, making the correctness oracle costly, non-reproducible at scale, and unable to account for dynamic web drift or answer staleness. This structural limitation recurs across open-web evaluation efforts: DeepResearch Bench uses adaptive reference-based evaluation but still depends on static reference reports and human-curated criteria, while BrowseComp-Plus sidesteps live-web variability by freezing a corpus, sacrificing realism. The field's trajectory toward live-web agents demands an automated, drift-aware oracle that can certify evidence validity and answer uniqueness without human bottlenecks and remain reproducible as the web changes.

## Key Insight

A correctness oracle for open-web evidence can be made scalable and drift-aware by treating evidence validity as a property of a dynamically maintained entailment graph over source snapshots and answer uniqueness as a property of an evidence-coverage closure, where drift is detected as a distribution shift in the support set rather than as a change in surface text.

## Method

**What it is** — DRIFT-ORACLE (Drift-Responsive Inference for Factual Truth) is an automated pipeline that takes as input a query-answer pair (q, a) produced by a web agent and outputs a tri-state verdict (Valid, Invalid, Stale) together with a drift score and a timestamped evidence certificate. It replaces human verification of evidence validity and answer uniqueness by combining LLM-based entailment checking with a statistical drift monitor over dynamically snapshotted web pages.

**How it works** — The pipeline operates in three phases:

```python
# Phase 1: Snapshot & Seed Evidence
crawl_result = web_crawler.search_and_snapshot(q, max_pages=20, snapshot_interval='24h')
# Each snapshot s_i stores raw HTML, extracted text, and a content hash h_i.
# Calibrated support gate (replaces fixed tau_entail=0.7):
#   - entailment_score(s, a) returns a logit; temperature T is fit on a held-out
#     calibration set of 512 (snippet, answer, human_label) triples per query type
#     (factoid / list / comparison), so the calibrated probability is
#     p_entail = sigmoid(logit / T).
#   - tau_entail is set per query type as the threshold achieving 90% precision
#     on the calibration set (reported in the certificate).
#   - Abstention: if p_entail in [tau_entail - 0.1, tau_entail + 0.1], the snippet
#     is marked 'uncertain' and excluded from seed_evidence; abstention rate is
#     logged.
#   - Entity-swap consistency check: for each snippet, generate a counterfactual
#     by swapping the answer entity with a distractor of the same type; require
#     entailment_score(s, a) - entailment_score(s, a_swap) > 0.2 (margin).
seed_evidence = [s for s in crawl_result
                 if p_entail(s, a) > tau_entail
                 and margin(s, a, a_swap) > 0.2
                 and s not in abstained]

# Phase 2: Build Entailment Graph & Check Answer Uniqueness
G = build_entailment_graph(seed_evidence, q, a, llm=GPT-4o)
# Nodes = evidence snippets; edges = bidirectional entailment between snippets about a.
# For each candidate answer a' != a, compute coverage(a') = fraction of G's nodes that contradict a'.
unique = all(coverage(a') > tau_cover for a' in candidate_answers)  # tau_cover = 0.85

# Phase 3: Drift Monitor & Staleness Verdict
for each snapshot s_i in seed_evidence:
    recheck = web_crawler.refetch(s_i.url)
    # Boilerplate filter: extract main content region via readability/trafilatura
    # before computing drift, so ads/nav churn does not inflate JS divergence.
    drift_i = distribution_shift(main_content(s_i.text), main_content(recheck.text))  # Jensen-Shannon divergence on token n-grams
    if drift_i > tau_drift:  # tau_drift = 0.3
        # Re-apply the same calibrated gate at refetch time.
        new_p_entail = p_entail(recheck, a)
        new_margin = margin(recheck, a, a_swap)
        # Agreement requirement: mark stale only if BOTH seed-time and refetch-time
        # support decisions disagree (seed supported, refetch does not).
        if new_p_entail < tau_entail or new_margin <= 0.2:
            mark_stale(s_i)
verdict = 'Valid' if unique and not any_stale else ('Stale' if any_stale else 'Invalid')
```

A schematic flow:

```
query-answer (q,a)
      |
      v
[Web Crawler + Snapshotter] --> snapshots {s_i}
      |
      v
[Calibrated Entailment Scorer + Entity-Swap Check] --> seed evidence E
      |
      v
[Entailment Graph Builder] --> G (nodes: snippets, edges: entailment)
      |                         |
      v                         v
[Uniqueness Checker]       [Drift Monitor]
(coverage over candidate a')   (JS divergence on refetch, boilerplate-filtered)
      |                         |
      +------------+------------+
                   v
           [Verdict Aggregator] --> (verdict, drift_score, certificate)
```

**Why this design** — We chose an entailment graph over a flat list of evidence because answer uniqueness requires reasoning about the *joint* support of multiple sources, not independent per-source checks; this costs additional LLM calls for edge construction but yields a coverage-based uniqueness signal that is robust to redundant sources. We chose Jensen-Shannon divergence on token n-grams over embedding-based drift because n-gram shift is directly attributable to content change and avoids embedding-model staleness, accepting that it may miss semantic drift under paraphrase. We chose a tri-state verdict (Valid/Invalid/Stale) over binary because staleness is a distinct failure mode from incorrectness and must be surfaced for downstream retraining; this increases annotation complexity but enables targeted refresh policies. We chose a fixed refetch interval of 24 hours over adaptive scheduling because it provides a consistent temporal baseline for drift comparison, at the cost of delayed detection for fast-changing pages.

**Why this measures what we claim** — The entailment score `entailment_score(s, a)` measures evidence validity because we assume that a high calibrated score from an LLM entailment checker (temperature-scaled on a held-out set, with entity-swap margin) is equivalent to the snippet logically supporting the answer; this assumption fails when the LLM hallucinates entailment for plausible but unsupported claims, in which case the score reflects surface-level lexical overlap rather than logical support. The coverage statistic `coverage(a')` measures answer uniqueness because we assume that if every node in the entailment graph contradicts a given alternative answer a', then a' is not supported by the evidence set; this assumption fails when the graph is incomplete (missing sources that support a'), in which case coverage reflects search bias rather than true uniqueness. The drift score `distribution_shift(s_i.text, recheck.text)` measures staleness because we assume that a large JS divergence between the original and refetched content implies the evidence for the answer has materially changed; this assumption fails when the page changes in irrelevant boilerplate (e.g., ads), in which case drift reflects page churn rather than answer-relevant staleness.

## Contribution

['A novel automated correctness oracle, DRIFT-ORACLE, that combines LLM-based entailment graph construction with a statistical drift monitor to produce tri-state verdicts (Valid/Invalid/Stale) for open-web evidence, replacing human verification of evidence validity and answer uniqueness.', 'A design principle for drift-aware evaluation: answer staleness can be operationalized as distribution shift in the support set of an entailment graph, enabling reproducible detection of when live-web changes invalidate previously correct answers.', 'An open-source implementation of the oracle pipeline and a protocol for integrating it into existing deep-search benchmarks (e.g., OpenDS-World) to provide scalable, reproducible correctness certificates.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | OpenDS-World + BrowseComp-Plus fixed-corpus queries | Live-web drift plus frozen corpus for controlled refetch |
| Primary metric | Verdict accuracy against human-annotated Valid/Invalid/Stale labels | Directly tests tri-state claim, not downstream QA |
| Baseline | Deep-research agent self-reported confidence | Tests whether internal confidence tracks evidence validity |
| Baseline | LLM-as-judge entailment-only checker | Tests whether uniqueness and drift add signal beyond entailment |
| Baseline | Citation-count / source-agreement heuristic | Tests whether redundancy proxies uniqueness and freshness |
| Ablation-of-ours | DRIFT-ORACLE minus drift monitor (binary Valid/Invalid) | Isolates staleness detection as the source of gain |
| Calibration metric | Reliability diagram of calibrated entailment scores vs. human labels on a held-out set of 512 snippets | Tests the load-bearing assumption that the scorer is calibrated |
| Reproducibility metric | Inter-run verdict agreement across two independent crawls of the same query | Tests the scalability/reproducibility claim |

### Why this setup validates the claim

The central claim is that evidence validity, answer uniqueness, and temporal staleness are three separable failure modes that a single verdict must disentangle, and that entailment-graph coverage plus n-gram drift monitoring detects each where cheaper proxies fail. The dataset pairing is deliberate: OpenDS-World supplies genuinely live pages where drift actually occurs, while BrowseComp-Plus's fixed corpus lets us replay refetches deterministically so that any drift signal is attributable to content change rather than crawl variance. The entailment-only baseline isolates the uniqueness sub-claim — if coverage over the graph adds nothing beyond per-snippet entailment, our uniqueness mechanism is unmotivated. The confidence and citation-count baselines test whether the problem is already solved by agent self-assessment or redundancy heuristics. The drift ablation is the sharpest test: it holds everything constant except staleness detection, so any accuracy gap concentrated on the Stale subset is direct causal evidence that the drift monitor, not some incidental pipeline effect, produces the gain. Verdict accuracy against human tri-state labels is the right metric because it penalizes the exact confusions we claim matter — calling stale evidence valid, or calling a genuinely contested answer invalid. The calibration metric directly tests the load-bearing assumption that the entailment scorer is reliable; the reproducibility metric tests the claim that the oracle is stable across independent crawls.

### Expected outcome and causal chain

**vs. Deep-research agent self-reported confidence** — On a case where a page's answer-bearing sentence is silently revised after the agent's crawl, the baseline reports high confidence because its confidence was computed at answer time and never re-grounded against the changed page. Our method instead refetches and computes JS divergence, flagging the snapshot stale when refetched text no longer entails the answer. We expect a pronounced gap on the Stale subset with near-parity on the Valid subset, since confidence and entailment agree when nothing has drifted.

**vs. LLM-as-judge entailment-only checker** — On a case where a single authoritative-looking snippet supports the answer but several other sources contradict it, the baseline marks the answer valid because it checks each snippet independently and never aggregates joint support. Our method instead builds the entailment graph and computes coverage of alternative answers across nodes, surfacing the contradiction as low uniqueness. We expect the gap to concentrate on Invalid cases with redundant-but-conflicting evidence, while both methods agree on clean single-source Valid cases.

**vs. Citation-count / source-agreement heuristic** — On a case where many pages repeat the same stale fact (syndicated content), the baseline treats high citation agreement as strong support and never notices the shared source has drifted. Our method instead detects that the refetched pages no longer entail the answer, marking Stale despite high redundancy. We expect the baseline to be systematically overconfident on syndicated-stale instances, with our accuracy advantage largest exactly there and shrinking to parity on independently-sourced Valid cases.

### What would falsify this idea

If the drift ablation's accuracy deficit were uniform across Valid, Invalid, and Stale subsets rather than concentrated on Stale instances, the claim that drift monitoring specifically captures staleness would be wrong. Likewise, if entailment-only matched full DRIFT-ORACLE on the conflicting-evidence Invalid subset, the entailment-graph uniqueness mechanism would be unmotivated. Additionally, if the calibration reliability diagram shows systematic miscalibration (e.g., expected calibration error > 0.1), the load-bearing assumption that the entailment scorer is reliable would be falsified.

## References

1. Deep Search in Real-World Web Environments
2. DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents
3. BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent
4. HelloBench: Evaluating Long Text Generation Capabilities of Large Language Models
5. LAB-Bench: Measuring Capabilities of Language Models for Biology Research
6. ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery
7. BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval
8. GPQA: A Graduate-Level Google-Proof Q&A Benchmark
