# Composition-Closure Verification with Residual-Driven Primitive Discovery for Semantic Action Models

## Motivation

Semantic-action VLA models such as nanoAstra (A Glimpse of Training VLMs as Foundation Models for Robotic Manipulation) represent behavior as sequences of reusable, parameterized action primitives, but the primitive vocabulary is fixed a priori and nothing verifies that novel task compositions are expressible within it; CP-SFT's structured action-space constraint can only emit sequences over that frozen set, so any composition requiring an absent primitive is silently inexpressible and the policy fails, which is the structural reason zero-shot success stalls at 17.8% on RoboTwin 2.0 rather than a data or capacity shortfall. This is a recurring property across the VLA tree: LLaRA and OTTER likewise inherit a fixed action interface and never test whether the interface itself is closed under the compositions they are asked to perform.

## Key Insight

Expressibility of a task composition is a decidable property of the CP-SFT action grammar against the primitive vocabulary, and every inexpressible composition leaves a residual action trace whose structure identifies exactly which new parameterized primitive must be added, so action-vocabulary expansion becomes a verifiable, residual-driven decision rather than an open-ended search.

## Method

**What it is.** We introduce CLOSE-VLA (Composition-cLosure with residual-driven primitive Expansion), a training-and-verification loop wrapped around nanoAstra's semantic-action CP-SFT model. Inputs: a trained nanoAstra policy with primitive vocabulary $\mathcal{P}=\{p_1,\dots,p_K\}$ and CP-SFT system prompt $S$ defining the action grammar; a held-out robot dataset $\mathcal{D}_{ho}$. Outputs: an expanded vocabulary $\mathcal{P}'\supseteq\mathcal{P}$ with new parameterized primitives, a re-trained policy, and a closure certificate (the set of tested compositions and their expressibility verdicts).

**Load-bearing assumption (stated explicitly).** The closure verdict is only meaningful if the CP-SFT system-prompt grammar is a faithful, tight characterization of the policy's realized action distribution: every sequence that parses under $S$ is emittable by the trained policy (soundness), and every sequence the policy can emit parses under $S$ (completeness). We do not assume this holds a priori; we calibrate it empirically in Phase 1b below and report the calibration as part of the closure certificate. When the grammar is looser than the learned distribution (parseable but never emitted) or tighter (emittable but unparseable), the verdict reflects grammar permissiveness or over-constraint rather than realized capability, and the residual signal is correspondingly contaminated.

**How it works.**

```pseudocode
# Phase 1: Composition-closure test (self-supervised, no new labels)
for each held-out episode e in D_ho:
    decompose e into progress descriptions g_1..g_T and action segments a_1..a_T
    # ask the grammar: can S express this composition over P?
    for each segment a_t:
        parse a_t against grammar(S, P)  ->  matched primitive p* or UNPARSED
    parse_verdict[e] = ALL segments matched

# Phase 1b: Emit-ability calibration (makes the load-bearing assumption testable)
# For each held-out segment a_t, test whether the trained policy can actually
# produce a matching action sequence under the grammar.
for each held-out segment a_t:
    sample M = 16 action sequences from the policy under constrained decoding
    against grammar(S, P) (temperature 0.7, top-p 0.95, max length = len(a_t))
    match_rate[a_t] = fraction of samples whose parsed primitive sequence equals
                      the parse of a_t (segment-level exact match on primitive IDs)
    emit_verdict[a_t] = (match_rate[a_t] >= 0.5)
closure_verdict[e] = parse_verdict[e] AND (ALL emit_verdict[a_t] for a_t in e)
# Report calibration: P(emit | parse) and P(parse | emit) on D_ho.
# If P(emit | parse) < 0.9 or P(parse | emit) < 0.9, flag grammar-tightness
# violation in the certificate and report closure verdicts both with and
# without the emit-ability conjunct.

# Phase 2: Residual extraction from inexpressible compositions
for each e with closure_verdict[e] == False:
    R_e = concat of UNPARSED action segments (residual traces)
    cluster R_e across episodes with DTW k-medoids (k chosen by silhouette, k<=8)
    for each cluster C with support >= N_min (N_min = 20 episodes):
        fit a parameterized primitive p_new(theta) by least-squares
        over the cluster's action traces, theta = (start_pose,
        end_pose, approach_dir, gripper_profile)
        accept p_new iff held-out reconstruction error on C drops
        below tau (tau = 5% of segment action range)

# Phase 3: Vocabulary augmentation + re-verification
P' = P union {accepted p_new}
regenerate the CP-SFT system prompt S' to declare P'
re-run Phase 1 and Phase 1b on D_ho with (S', P')
if closure rate increases: commit; else: revert and log
```

The loop terminates when either all held-out compositions parse and are emit-able (closure achieved) or no cluster passes the support and reconstruction thresholds.

**Why this design.** We chose a *grammar-parse closure test* over a learned classifier that predicts expressibility because the CP-SFT system prompt is an explicit symbolic grammar, so parse success is an exact property of the artifact rather than a proxy, accepting the cost that hand-written grammar coverage must be maintained. We pair it with an *emit-ability calibration* (Phase 1b) because the grammar-parse verdict alone is only as trustworthy as the assumption that the grammar is tight against the policy's learned distribution, and a 16-sample constrained-decoding match rate is a direct, cheap test of that assumption on the same held-out segments. We chose *residual clustering with least-squares primitive fitting* over end-to-end learning of new tokens because the residual traces are the only signal that isolates what the vocabulary lacks, accepting the cost that discovered primitives are restricted to the parametric family we fit. We chose a *support threshold $N_{min}=20$ plus reconstruction gate $\tau=5\%$* over accepting every cluster because a spurious primitive corrupts the grammar for all downstream compositions, accepting the cost that rare-but-real primitives are missed. We chose *re-verification after augmentation* over a single expansion pass because adding a primitive can shadow existing parses, accepting the cost of an extra closure pass. A domain expert would not describe this as retrieval over a primitive library (the nearest-neighbor shape of Toolformer-style tool selection): the mechanism *intervenes on the library itself* by fitting and admitting new primitives, which retrieval cannot do, and the closure verdict is computed from the grammar plus the policy's emit-ability, not from descriptor similarity.

**Why it measures what we claim.** The closure verdict `closure_verdict[e]` measures *expressibility of a composition under the deployed action interface* because it conjoins a parse against the CP-SFT system prompt (the exact grammar the policy is constrained to emit) with an empirical emit-ability check (16 constrained-decoding samples per segment, match rate $\ge 0.5$); this assumption fails when the grammar is looser than the policy's learned distribution (the policy could parse but never emit) or tighter (emittable but unparseable), in which case the verdict reflects grammar permissiveness or over-constraint rather than realized capability — which is why Phase 1b reports $P(\text{emit}\mid\text{parse})$ and $P(\text{parse}\mid\text{emit})$ on $\mathcal{D}_{ho}$ and flags violations below 0.9. The residual cluster support $|C|$ measures *whether a missing primitive is a systematic vocabulary gap rather than episode-specific noise* because repeated co-occurring unparsed segments across independent episodes are assumed to share a generating primitive; this assumption fails when the same residual shape arises from sensor noise or operator idiosyncrasy, in which case the cluster reflects nuisance variation. The reconstruction error drop on held-out members of $C$ measures *whether the fitted primitive actually covers the gap* because low error on unseen cluster members is assumed to require the primitive to generalize beyond the fitting traces; this assumption fails when the cluster is near-degenerate, in which case the error reflects interpolation within a single trajectory.

**Hyperparameters.** $N_{min}=20$ episodes; $\tau=5\%$ of segment action range; $k\le 8$ residual clusters; silhouette-based $k$ selection; emit-ability sample count $M=16$ per segment; emit-ability match threshold $0.5$; grammar-tightness flag threshold $0.9$ for $P(\text{emit}\mid\text{parse})$ and $P(\text{parse}\mid\text{emit})$.

## Contribution

(1) CLOSE-VLA, a composition-closure verification and residual-driven primitive-expansion loop that makes the expressibility of task compositions under a semantic-action VLA's CP-SFT grammar an explicitly tested and repairable property rather than a fixed assumption. (2) A residual-trace primitive-discovery procedure that fits new parameterized primitives from unparsed action segments and admits them only under support and held-out reconstruction gates, yielding a design principle for when action-vocabulary expansion is justified. (3) A closure certificate artifact recording tested compositions and expressibility verdicts, enabling diagnosis of whether VLA failures are vocabulary gaps or policy errors.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | RoboTwin 2.0 randomized benchmark | Zero-shot compositional stress test with held-out episodes |
| Primary metric | Held-out composition closure rate (parse AND emit-ability) | Directly measures expressibility under deployed grammar and policy |
| Diagnostic metric | Failure decomposition: UNPARSED vs parsed-but-misexecuted | Tests binding-constraint premise before main loop |
| Diagnostic metric | Grammar-tightness calibration: P(emit\|parse), P(parse\|emit) | Detects when grammar is looser/tighter than policy |
| Baseline | nanoAstra (fixed vocabulary) | Isolates gain from primitive expansion alone |
| Baseline | OTTER (frozen VLM, text-aware features) | Tests whether semantic alignment substitutes for vocabulary |
| Baseline | CoT-VLA (visual chain-of-thought) | Tests whether reasoning substitutes for primitive coverage |
| Ablation-of-ours | CLOSE-VLA without re-verification pass | Isolates grammar-shadowing effect of augmentation |
| Ablation-of-ours | CLOSE-VLA with parse-only closure verdict (no Phase 1b) | Isolates contribution of emit-ability calibration |

### Why this setup validates the claim

The central claim is that a fixed primitive vocabulary is the binding constraint on compositional generalization in semantic-action VLAs, and that residual-driven expansion relieves it. RoboTwin 2.0's randomized benchmark is the right arena because its zero-shot splits contain compositions the training distribution never covered, so a policy that only recombines known primitives will fail on exactly the compositions the grammar cannot parse — the failure mode we predict. Before crediting the expansion loop, we run a standalone failure-decomposition diagnostic on held-out episodes: we classify each failure as UNPARSED (no primitive in $\mathcal{P}$ matches the segment) versus parsed-but-misexecuted (a primitive parses but execution fails), and report the fraction of failures attributable to each. If UNPARSED failures are a minority, the binding-constraint premise is falsified and the paper must be reframed. We also report grammar-tightness calibration $P(\text{emit}\mid\text{parse})$ and $P(\text{parse}\mid\text{emit})$ from Phase 1b, so the closure verdict is explicitly conditional on the grammar being a faithful characterization of the policy's emission distribution. The fixed-vocabulary nanoAstra baseline tests the sub-claim that vocabulary, not data scale or VLM quality, is the limiter: it shares our backbone and CP-SFT prompt, so any gap is attributable to the primitive set. OTTER and CoT-VLA test competing sub-claims — that frozen semantic alignment or explicit visual reasoning can substitute for expressibility — and should show parity on in-vocabulary compositions while lagging on unparseable ones, which is the dissociation we need. The no-re-verification ablation is the sharpest test: if adding primitives shadows existing parses, the ablation should show closure gains that reverse on re-check, proving the verification loop does real work rather than decorating a monotone improvement. The parse-only ablation isolates whether the emit-ability conjunct changes the verdict distribution on $\mathcal{D}_{ho}$, which is the direct test of the load-bearing assumption. Closure rate, not task success, is the primary metric because it reads the causal variable directly: success conflates execution quality with expressibility, whereas parse-plus-emit success is a property of the deployed grammar and policy and is falsifiable at the artifact level.

### Expected outcome and causal chain

**vs. nanoAstra (fixed vocabulary)** — On a case where a held-out episode requires a bimanual regrasp whose approach direction is not in the parametric family, the baseline produces a nearest-primitive substitution that parses but executes the wrong approach, because its vocabulary has no parameterization for that direction and CP-SFT forces emission into the fixed grammar. Our method instead clusters the unparsed residual traces across episodes, fits a new primitive with an approach-direction parameter, and admits it after the reconstruction gate, so we expect a concentrated closure-rate gap on the unparseable subset with near-parity on in-vocabulary compositions. The failure-decomposition diagnostic should show UNPARSED failures are a non-trivial fraction (we predict at least 30%) of nanoAstra's zero-shot failures; if they are below 10%, the premise is falsified.

**vs. OTTER (frozen VLM, text-aware features)** — On a case where the language instruction names a compositional sequence the pretrained alignment never saw paired, the baseline produces a semantically plausible but action-incoherent plan, because text-aware feature extraction selects task-relevant visual features but cannot synthesize an action primitive outside the downstream policy's fixed set. Our method instead treats the unparsed segment as evidence of a vocabulary gap and fits the missing primitive from residual traces, so we expect OTTER to match us on familiar compositions and fall behind specifically on episodes whose closure verdict flips from false to true after expansion. The emit-ability calibration should show OTTER's grammar-tightness is comparable to nanoAstra's, so the dissociation is attributable to vocabulary coverage rather than to a looser grammar.

**vs. CoT-VLA (visual chain-of-thought)** — On a case where the visual goal is correctly predicted but the intermediate motion requires a gripper profile absent from the vocabulary, the baseline produces a correct-looking subgoal image followed by an action sequence that cannot realize it, because visual CoT supplies planning but the action head still emits from a fixed primitive set. Our method instead closes the gap at the vocabulary level, so we expect CoT-VLA to show strong subgoal prediction yet weak execution on those episodes, and our gain to concentrate where residual clusters passed the support threshold rather than uniformly across the benchmark. The parse-only ablation should show a higher closure rate than the full method on episodes where the grammar is permissive, confirming that the emit-ability conjunct is doing real filtering work.

### What would falsify this idea

If closure-rate gains after expansion are uniform across in-vocabulary and unparseable subsets rather than concentrated on episodes whose verdict flips, the vocabulary-gap explanation is wrong and the loop is merely adding capacity. Equally diagnostic: if the no-re-verification ablation matches the full method, then grammar shadowing is not occurring and the re-verification pass is not doing the work we claim; and if the failure-decomposition diagnostic shows UNPARSED failures are a small minority of nanoAstra's zero-shot errors, the binding-constraint premise is falsified and the paper must be reframed.

## References

1. A Glimpse of Training VLMs as Foundation Models for Robotic Manipulation
