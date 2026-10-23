# Failure-Closed Routing: Online Mode Invention with Identity-Anchored Advantage Decomposition

## Motivation

Existing LLM routers such as RouteWeaver treat the routing mode set as a fixed, pre-declared library (single-round, multi-round, agentic), so the system is not closed under failure substitution: when every available mode repeatedly fails on a query family, the router can only re-select among modes that are already known to fail, and its COMET-GRPO credit decomposition has no slot in which an unseen execution strategy could even be represented. This is a structural gap rather than a tuning issue — the mode vocabulary is baked into the policy head and the advantage decomposition before training begins, so no amount of RL can instantiate a strategy that was never in the action space, and Hybrid LLM-style difficulty routers inherit the same closure violation because their action set is two fixed models.

## Key Insight

Closure under failure substitution holds exactly when mode identity is a first-class, extendable coordinate of the credit-assignment decomposition rather than a fixed output dimension, so that adding a mode is a coordinate extension that preserves the semantic meaning of every existing advantage term.

## Method

**What it is.** FCR (Failure-Closed Router) is a routing policy whose mode set is a growable registry rather than a fixed head. Inputs: a query, the current mode registry with per-mode identity embeddings, and recent per-mode failure statistics. Outputs: a selected mode, a model within that mode, and — when a failure trigger fires — a newly instantiated mode with its own identity embedding and a fresh credit-assignment slot.

**How it works.**

```pseudocode
registry R = {single_round, multi_round, agentic}   # each m has identity embedding e_m (frozen SBERT all-MiniLM-L6-v2, d=256)
policy pi_theta(query q, R) -> (mode m, model a within m)
for each query batch B:
    for q in B:
        m, a = pi_theta(q, R)
        trajectory tau = execute(q, m, a)
        r = task_reward(tau)
        update failure stats F[m] over a sliding window W=64
        # failure trigger: sequential probability ratio test (SPRT) on per-mode success
        # H0: p_success >= 0.3 (mode competent), H1: p_success < 0.3 (mode failing)
        # SPRT with alpha=0.05 (false-positive rate), beta=0.10 (false-negative rate)
        # log-likelihood ratio Lambda_m = sum_{i in W} log( p1(x_i) / p0(x_i) )
        # where p0 = Bernoulli(0.3), p1 = Bernoulli(0.1)
        # trigger fires when Lambda_m > log((1-beta)/alpha) AND m failed on last k=3 draws of this query family
        if SPRT_trigger(F[m], alpha=0.05, beta=0.10) and m failed on last k=3 draws of this query family:
            m_new = META_CONTROLLER(q, tau, R)      # LLM (GPT-4o, temperature=0.7) proposes a new execution strategy
            e_new = IDENTITY_ENCODE(m_new)            # frozen SBERT all-MiniLM-L6-v2 -> d=256
            R = R + {(m_new, e_new)}                  # coordinate extension
            re-decompose advantages (below) with the extended registry
    COMET-style decomposition with identity anchoring:
        A_mode(q, m)   = r - V_exec(q, m)             # did choosing m help?
        A_exec(q, m, a) = r - V_mode(q)               # did execution within m help?
        # behavioral novelty test replaces embedding-cosine discount:
        # execute m_new and m_failed on the same query family (paired rollout, n=16 queries)
        # behavioral_overlap = mean( 1[ r_new(q_i) == r_failed(q_i) ] ) over the paired queries
        # lambda = 0.3 (fixed; bounds expected credit a pure-rename mode can harvest to <= 0.3 * (1 - overlap))
        A_new(q, m_new) = r_new - max(V_mode(q), V_exec(q, m_failed)) + lambda * (1 - behavioral_overlap)
                          # lambda=0.3: novelty credit discounted by measured outcome divergence
                          # from the failed mode, so a renamed failure (overlap=1) earns 0 novelty credit
    GRPO update of pi_theta and META_CONTROLLER over the extended registry
```

**Why this design.** We chose a *growable registry with identity embeddings* over a fixed policy head because a fixed head cannot represent an action it was not trained on, accepting the cost that the registry grows and the softmax must be recomputed over an expanding action set. We chose *failure-triggered meta-controller invocation* over always-on mode generation because unconditional generation floods the registry with redundant modes and destabilizes credit assignment, accepting the cost that a genuinely novel need which never crosses the failure threshold is missed. We chose *behavioral novelty credit* (the paired-rollout overlap term) over granting a new mode full advantage because a meta-controller that merely renames a failed strategy would otherwise harvest free credit, accepting the cost that a legitimately novel mode that happens to produce similar outcomes to a failed one on the paired sample is under-credited early on. This differs from RouteWeaver, whose COMET-GRPO decomposition is defined over a static mode vocabulary and whose progressive exploration only shifts the *distribution* over pre-existing modes; a domain expert would not describe FCR as a variant of RouteWeaver because the action space itself is a learned, extendable object and the advantage decomposition is re-derived on coordinate extension rather than fixed at initialization.

**Why it measures what we claim.** The registry-extension event measures *closure under failure substitution* because the assumption is that an SPRT crossing (Lambda_m > log(9) with alpha=0.05, beta=0.10 over W=64) is a sufficient indicator that the query family is outside every existing mode's competence; this assumption fails when the failure is caused by a transient model-side outage rather than a genuine mode-capability gap, in which case the trigger reflects infrastructure noise instead of a closure violation. The behavioral_overlap term measures *mode novelty* because the assumption is that paired rollouts on the same query family reveal the fraction of the new mode's reward attributable to renaming; this assumption fails when the paired sample (n=16) is too small to distinguish a genuinely novel mode from a renamed one, in which case the overlap estimate is noisy and the discount is unreliable. The similarity-discounted advantage A_new measures *genuine credit for invention* because the assumption is that behavioral overlap with the failed mode upper-bounds how much of the new mode's reward is attributable to renaming; this assumption fails when a novel mode is legitimately close in outcome to a failed one on the paired sample, in which case A_new under-credits a real improvement rather than over-crediting a fake one.

## Contribution

(1) FCR, a routing framework in which the mode set is a growable registry and mode identity is a first-class coordinate of the credit-assignment decomposition, enabling failure-triggered mode instantiation without invalidating existing advantage terms. (2) A similarity-discounted novelty-credit mechanism that separates genuine strategy invention from renaming a failed mode, together with the failure-trigger statistic that gates meta-controller invocation. (3) An analysis of when registry extension preserves the semantic meaning of mode-selection versus execution advantages, expressed as an operational condition on identity-embedding separation.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale |
| --- | --- | --- |
| Dataset | Two-domain routing suite: coding (HumanEval, MBPP) and recall (Natural Questions, TriviaQA) | Scoped to two domains per feasibility review; forces heterogeneous mode-competence gaps across families |
| Primary metric | Accuracy on failure-eligible query families (families where the SPRT trigger fires for at least one baseline mode) | Directly probes closure-under-failure, not average quality |
| Baseline | RouteWeaver (static mode vocabulary) | Tests whether growable action space beats fixed modes |
| Baseline | Hybrid LLM (large/small static router) | Tests whether mode extension beats model-size routing |
| Baseline | ReTool-style RL reasoning model | Tests whether mode invention beats single fixed strategy |
| Baseline | Meta-controller-only (LLM proposes new modes but they are executed without registry integration or re-decomposition) | Attributes significance specifically to closure rather than to LLM-generated strategies |
| Ablation-of-ours | FCR without registry extension (frozen registry) | Isolates the contribution of coordinate extension |
| Ablation-of-ours | FCR with embedding-cosine discount (original A_new) | Isolates the contribution of behavioral novelty test over embedding geometry |

### Why this setup validates the claim

The central claim is that a growable registry closes over failure families that no fixed mode set can represent, and that behavioral novelty credit prevents fake novelty from being rewarded. RouteWeaver tests the first sub-claim directly: it has progressive exploration but a static mode vocabulary, so any gain we show on failure-eligible families that RouteWeaver cannot match must come from coordinate extension rather than from better exploration over existing modes. Hybrid LLM tests whether the effect is merely model-size selection — if our gain survives against a quality-aware size router, the mechanism is mode-level, not scale-level. ReTool tests whether a strong single fixed strategy (RL code-augmented reasoning) already absorbs the cases we claim need new modes; parity there would undercut the need for extension. The meta-controller-only baseline tests whether the gain comes from LLM-generated strategies alone or from registry integration and re-decomposition; if meta-controller-only matches full FCR, the closure mechanism is not doing the work. The frozen-registry ablation is the load-bearing control: it shares our policy, credit decomposition, and meta-controller plumbing but cannot add modes, so any gap between it and full FCR isolates extension as the cause. The embedding-cosine ablation tests whether the behavioral novelty test is necessary for anti-gaming; if both yield similar false-credit rates, the simpler embedding discount suffices. Accuracy restricted to failure-eligible families is the right metric because average accuracy dilutes the predicted effect across families where all methods already succeed; the claim is about substitution under failure, so the metric must concentrate where failures actually occur. We pre-register the failure-eligible family selection rule: a family is failure-eligible if the SPRT trigger (alpha=0.05, beta=0.10, W=64) fires for at least one baseline mode on a held-out calibration set of 512 queries per family.

### Expected outcome and causal chain

**vs. RouteWeaver** — On a coding query family where the best fixed mode (say agentic) fails repeatedly because the family needs interleaved retrieval-then-verify that no predefined mode expresses, RouteWeaver keeps reweighting its static softmax toward whichever existing mode is least bad, so it plateaus at the ceiling of its vocabulary. Our method crosses the SPRT failure threshold, invokes the meta-controller, and instantiates a retrieval-verify mode with its own identity embedding, so we expect a clear accuracy gap concentrated on failure-eligible families and near-parity on families already served by existing modes.

**vs. Hybrid LLM** — On a case where the small model fails a multi-step coding query and the large model also fails because the failure is strategic (no tool use, no decomposition) rather than capacity-bound, the hybrid router escalates to the large model and still produces a wrong answer, since it only chooses among fixed execution styles. Our method instead treats the repeated failure as a mode-capability gap and grows an execution strategy, so we expect the hybrid to show cost reduction but no accuracy recovery on these families, while FCR recovers accuracy without proportional large-model escalation.

**vs. ReTool-style RL reasoning model** — On a factual-recall-plus-arithmetic family where a single code-augmented reasoning strategy is mismatched (it over-invokes execution when a direct lookup suffices), the ReTool-style model applies its one learned policy and wastes turns or mis-executes. Our method routes that family to a lighter existing mode or grows a lookup-first mode, so we expect ReTool to be competitive on math-heavy families but visibly behind on recall and logic families, with the gap tracking the mode-mismatch pattern rather than raw difficulty.

**vs. Meta-controller-only** — On a failure-eligible coding family, the meta-controller-only baseline generates a new strategy but executes it without registry integration or re-decomposition, so the policy cannot learn to prefer it and the credit assignment remains anchored to the failed mode. We expect meta-controller-only to show some improvement from the generated strategy but a smaller gain than full FCR, with the gap concentrated on families where the new mode is repeatedly selected after extension.

### What would falsify this idea

If FCR's gain over the frozen-registry ablation is uniform across all families rather than concentrated on failure-eligible families where the trigger fires, the registry-extension mechanism is not doing the work and the claim of closure under failure substitution is wrong. Likewise, if the behavioral novelty test yields no measurable separation between genuinely novel modes and renamed failures (measured by false-credit rate on injected renamed-failure modes), the anti-gaming argument collapses.

## References

1. RouteWeaver: Weaving Mode Selection and Execution into Unified LLM Routing
2. Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing
3. ReTool: Reinforcement Learning for Strategic Tool Use in LLMs
