# Verifier-Gated Continual Tool Adaptation for VLM Robot Coding Agents

## Motivation

Across VLM-based robot coding agents such as HexaShell, the tool library is developed on a fixed set of ten development layouts and then frozen before evaluation on novel layouts, so any residual generalization gap becomes permanent and uncorrectable at test time; π0.5 and GR00T N1 share the same structural limitation because their skills are also fixed after training. The root cause is architectural: there is no mechanism that lets the agent reopen tool refinement on the novel distribution while still guaranteeing that adapted tools do not regress on the development distribution, which is exactly why the 'frozen tool library' gap recurs across paper-trees.

## Key Insight

A tool adaptation is admissible exactly when it is accepted by an independent verifier that replays the tool on the original development layouts and on the triggering novel layout, so the accepted tool library is a monotone sequence of sets whose development-layout success is non-decreasing and whose novel-layout coverage strictly increases.

## Method

## VG-CTA: Verifier-Gated Continual Tool Adaptation

**(A) What it is.** VG-CTA is a wrapper around a VLM coding agent (e.g., the HexaShell agent) that periodically re-opens tool refinement whenever the agent fails on a novel layout, and only commits an adapted tool to the persistent library if an independent verifier confirms that the adapted tool preserves success on a held-out certification suite drawn from the development distribution and succeeds on the triggering novel layout. Inputs: the current tool library $\mathcal{T}$, a novel layout $\ell^\star$, the agent's failure log, a proposal set of development layouts $\mathcal{D}_{\text{prop}}=\{\ell_1,\dots,\ell_{10}\}$ (the ten layouts HexaShell was built on), and a disjoint certification suite $\mathcal{C}$ of $|\mathcal{C}|=30$ layouts sampled from the same development distribution but never seen by the agent during tool construction. Outputs: an updated library $\mathcal{T}'$ (or unchanged $\mathcal{T}$ if rejected) and a per-tool acceptance record.

**(B) How it works.**

```pseudocode
# Called once per novel layout after the frozen library fails
function ADAPT(lib T, layout l_star, log L, prop_layouts D_prop, cert_suite C, verifier V):
    # Phase 1: diagnose and propose a minimal edit
    cause   = AGENT.diagnose(L)                     # e.g. "plane assumption violated"
    tool_t  = AGENT.propose_edit(T, cause, l_star)  # new or patched tool code
    # Phase 2: independent verification on two disjoint suites
    #   dev_ok now evaluated on held-out certification suite C (30 layouts), not on D_prop
    dev_ok  = all(V.run(tool_t, l) for l in C)      # 30 certification layouts
    nov_ok  = V.run(tool_t, l_star)                 # triggering novel layout
    # Phase 3: gate
    if dev_ok and nov_ok:
        T = T + {tool_t}                            # commit
        record(tool_t, cause, dev_ok, nov_ok)
    else:
        record(tool_t, cause, dev_ok, nov_ok, rejected=True)
    return T

# Outer loop over an evaluation stream of novel layouts
T = HEXASHELL.build_tools(D_prop)     # frozen baseline library built on the ten proposal layouts
for l_star in NOVEL_STREAM:           # 20 held-out layouts
    if not V.run(T, l_star):          # frozen library fails
        T = ADAPT(T, l_star, AGENT.last_log, D_prop, C, V)
    evaluate(T, l_star)
```

Hyperparameters inline: proposal set size $|\mathcal{D}_{\text{prop}}|=10$ (fixed by HexaShell); certification suite size $|\mathcal{C}|=30$ (held-out layouts from the development distribution, disjoint from $\mathcal{D}_{\text{prop}}$); verifier replay budget = 1 run per layout per candidate; adaptation trigger = first failure of the frozen library on a novel layout; candidate edit scope = one tool per adaptation round (minimal-edit constraint).

**(C) Why this design.** We chose an *independent* verifier that replays a held-out certification suite $\mathcal{C}$ over the agent's own self-reported success because the agent's log is exactly the signal that is biased toward the novel layout it just saw, and because replaying only the ten proposal layouts $\mathcal{D}_{\text{prop}}$ would certify non-regression on the same points the agent was built on, accepting the cost that each adaptation round requires $|\mathcal{C}|+1=31$ verifier rollouts rather than a single self-check. We chose a *minimal one-tool edit* over wholesale tool-library rewriting because wholesale rewriting destroys the monotonicity property that makes the library auditable, accepting the cost that some novel layouts require several sequential adaptation rounds. We chose a *strict conjunction gate* ($dev\_ok \land nov\_ok$) over a weighted score because a weighted score would let a tool that succeeds on the novel layout but breaks a certification layout slip through, accepting the cost that the gate rejects some partially useful tools. This is not a variant of Toolformer, Gorilla, or standard RAG tool retrieval: those systems select from a fixed library by descriptor similarity, whereas VG-CTA *intervenes on the library itself* and admits a new tool only when a counterfactual replay on the certification suite certifies non-regression — a quantity retrieval cannot compute because retrieval never writes a candidate tool.

**(D) Why it measures what we claim.** The computational quantity `dev_ok` — the verifier's success on all 30 certification layouts in $\mathcal{C}$ — measures *non-regression of the tool library* because the assumption that the 30 certification layouts are a representative sample of the development distribution makes `dev_ok` an estimate of distribution-level non-regression; this assumption fails when the certification suite is not representative (e.g., it omits a capability present in $\mathcal{D}_{\text{prop}}$), in which case `dev_ok` reflects certification-set coverage rather than capability preservation. The computational quantity `nov_ok` — verifier success on the triggering novel layout — measures *coverage extension* because the assumption that the novel layout is drawn from the same task family as the development layouts makes `nov_ok` equivalent to 'the adapted tool generalizes to the novel distribution'; this assumption fails when the novel layout is an outlier (e.g., a different object category), in which case `nov_ok` reflects overfitting to one layout rather than distribution extension. The conjunction `dev_ok ∧ nov_ok` measures *admissible continual adaptation* because the two quantities are computed by the same verifier on disjoint suites, so their conjunction is equivalent to 'the library grew without shrinking' under the assumption that the verifier is deterministic across runs; this assumption fails when the verifier is stochastic (e.g., camera noise), in which case the gate reflects verifier variance rather than a true capability change. **Explicit load-bearing assumption:** the ten development layouts $\mathcal{D}_{\text{prop}}$ are not an exhaustive proxy for the development distribution; instead, the 30-layout certification suite $\mathcal{C}$ is assumed to be a representative sample of that distribution, so that `dev_ok` estimates non-regression on the distribution rather than on the ten points the agent was built on. This assumption fails when $\mathcal{C}$ is itself under-specified or non-representative, in which case `dev_ok` reflects certification-set replay luck rather than capability preservation.

## Contribution

['VG-CTA, a continual tool-adaptation mechanism that re-opens tool refinement on novel layouts and gates each candidate tool through an independent verifier replaying the original development suite, yielding a monotonically non-regressing tool library.', 'A design principle: admissibility of a robot-coding-agent tool should be certified by counterfactual replay on the development distribution rather than by self-reported success, because self-reports are biased toward the layout that triggered adaptation.', 'A per-tool acceptance record schema that logs cause, development-suite result, and novel-layout result, enabling post-hoc audit of which novel layouts caused which tool adaptations.']

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
| --- | --- | --- |
| Dataset | HexaShell novel-layout stream (20 held-out layouts) | Matches the frozen-tool generalization setting the claim targets |
| Certification suite | 30 held-out development-distribution layouts (disjoint from the 10 proposal layouts) | Tests non-regression on distribution, not on the 10 build layouts |
| Primary metric | Success rate on held-out novel layouts after adaptation | Directly measures coverage extension without capability loss |
| Secondary metric | Success rate on the 30-layout certification suite | Measures non-regression on the development distribution |
| Baseline | Frozen HexaShell library (no adaptation) | Tests whether adaptation is needed at all |
| Baseline | Ungated self-adaptation (agent commits own edits) | Tests whether the verifier gate is what prevents regression |
| Baseline | Wholesale tool-library rewrite on failure | Tests whether minimal-edit monotonicity matters |
| Baseline | Verifier gate with multi-tool edit (non-minimal) | Separates gate contribution from minimal-edit contribution |
| Ablation-of-ours | VG-CTA with dev-suite gate removed (nov_ok only) | Isolates the non-regression half of the conjunction |
| Ablation-of-ours | VG-CTA with minimal-edit constraint removed (gate kept) | Isolates the minimal-edit half of the conjunction |

### Why this setup validates the claim

The central claim is that an independent verifier gating a minimal one-tool edit yields a library that grows without shrinking. The frozen-library baseline establishes the floor: if adaptation were unnecessary, the frozen library would already succeed on novel layouts, and our gains would vanish. The ungated self-adaptation baseline tests the specific sub-claim that the agent's own failure log is biased toward the triggering layout — if committing self-reported edits already preserves development performance, the verifier is redundant. The wholesale-rewrite baseline tests the sub-claim that minimal-edit scope preserves auditability and monotonicity; if rewriting matches us, the minimal-edit constraint is doing no work. The multi-tool-edit baseline with the same gate separates the gate's contribution from the minimal-edit constraint: if it matches us, the minimal-edit constraint is not load-bearing. The ablation (nov_ok only) isolates the conjunction: it should recover novel-layout successes but lose certification-suite layouts, which is exactly the regression the full gate is designed to block. The ablation removing the minimal-edit constraint (gate kept) tests whether the gate alone suffices or whether minimal edit is also necessary. Success rate on held-out novel layouts is the right primary metric because the claim is about coverage extension under non-regression; success rate on the 30-layout certification suite is the right secondary metric because it directly measures non-regression on the development distribution rather than on the ten build layouts. Together these choices form a falsifiable test: the predicted pattern is a gain concentrated on layouts where the frozen library fails, parity on layouts it already handles, and a measurable certification-suite regression for the ablation.

### Expected outcome and causal chain

**vs. Frozen HexaShell library** — On a novel layout where the frozen library's plane assumption is violated, the baseline produces repeated grasp failures because its tools were built only on the ten development layouts and never revisited. Our method instead diagnoses the violated assumption, proposes a patched tool, and commits it only after replaying the 30-layout certification suite, so we expect a clear success-rate gap on the subset of novel layouts that trigger frozen-library failures, with near-parity on layouts the frozen library already solves.

**vs. Ungated self-adaptation** — On a novel layout that admits a layout-specific shortcut (e.g., a tool that hardcodes the triggering layout's object pose), the baseline commits the agent's self-reported edit and produces high novel-layout success but a visible drop on the certification suite because the shortcut overfits. Our method instead rejects that candidate at the certification-suite replay and keeps searching, so we expect the ungated baseline to show a higher novel-layout spike but a marked certification-suite regression, while ours shows slightly lower novel-layout gains with certification-suite parity.

**vs. Wholesale tool-library rewrite** — On a novel layout requiring a small patch, the baseline rewrites the whole library and produces collateral breakage on unrelated certification-suite layouts because the rewrite discards tools that were never implicated in the failure. Our method instead edits exactly one tool and gates it, so we expect the rewrite baseline to lose success on certification-suite layouts it previously handled, concentrated on tools unrelated to the triggering failure, while ours retains those successes.

**vs. Verifier gate with multi-tool edit (non-minimal)** — On a novel layout requiring a small patch, the baseline edits multiple tools at once and gates the bundle; because the gate only accepts or rejects the whole bundle, a single bad edit causes rejection of an otherwise useful patch, leading to lower novel-layout success. Our method edits one tool per round and gates it, so we expect the multi-tool baseline to show lower novel-layout success (more rejections) while both maintain certification-suite parity, isolating the minimal-edit constraint as the source of novel-layout gains.

**vs. VG-CTA without dev-suite gate** — On a novel layout whose fix trades away a development capability, the ablation commits the candidate on novel-layout success alone and produces a library that solves the new layout but regresses on at least one certification-suite layout. Our method instead requires the conjunction and rejects the candidate, so we expect the ablation to match us on novel-layout success but diverge sharply on certification-suite retention, isolating the gate as the causal mechanism.

**vs. VG-CTA without minimal-edit constraint (gate kept)** — On a novel layout requiring a small patch, the ablation edits multiple tools but still gates on the certification suite; because the gate rejects bundles with any regression, it may reject useful patches, leading to lower novel-layout success than ours. We expect this ablation to match us on certification-suite retention but underperform on novel-layout success, confirming that minimal edit is necessary for coverage extension under the gate.

### What would falsify this idea

If our gain over the frozen library is uniform across all novel layouts rather than concentrated on the subset where the frozen library fails, or if the ablation and the full method show indistinguishable certification-suite retention, then the verifier gate is not doing the causal work the claim assigns it and the central claim is wrong. Additionally, if the multi-tool-edit baseline matches our novel-layout success, the minimal-edit constraint is not load-bearing.

## References

1. HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents
2. Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents
3. π0.5: a Vision-Language-Action Model with Open-World Generalization
