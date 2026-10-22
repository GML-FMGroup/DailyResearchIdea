# Closed-Loop Streaming Video Evaluation via Model-Derived Closure Operators

## Motivation

Existing streaming/interactive video benchmarks—StreamEval, WorldScore, TC-Bench—all treat the request stream as an exogenous, fully pre-specified script: StreamEval's 300 cases ship with fixed incremental multimodal requests and explicit state-change requirements, WorldScore conditions each next-scene on a pre-authored camera trajectory, and TC-Bench fixes initial/final states in advance. Because the evaluation input never depends on the model's realized output, these protocols are structurally open-loop: they can measure whether a model executes a script, but cannot measure whether a model can sustain a self-consistent request loop, which is the actual capability implied by 'interactive' streaming generation. The root cause is not a per-benchmark bug but the shared meta-assumption that evaluation inputs are exogenous, which a closure operator over the request stream directly removes.

## Key Insight

If the next evaluation request is defined as a deterministic re-projection of the model's own realized output through the model's generative factorization, then a stable evaluation trajectory is exactly a fixed point of the model's output distribution, so loop-sustention becomes measurable without any new human-authored requests.

## Method

**What it is.** We introduce CLOVE (CLOsure-operator Video Evaluation), a protocol that replaces the exogenous request script with a model-derived closure map. Inputs: a streaming video model M with a per-step conditioning interface, an initial asset a₀, and a seed request r₀. Output: a trajectory of realized outputs {o₁,…,o_T}, a closure-stability profile (per-step drift, fixed-point residual, escape rate), and a per-slot faithfulness floor (round-trip error of M's own extractors) used to residualize drift. The load-bearing assumption, stated plainly: M's conditioning extractors (reference-frame encoder, layout/camera head, instruction-token decoder) are faithful enough inverses of its conditioning encoders that re-projecting a realized output back into the conditioning slots yields a signal whose drift reflects the model's genuine loop (in)stability rather than extractor quantization, lossiness, or sampling noise. CLOVE measures loop-sustention *above the extractor's own noise floor*; the faithfulness floor is measured, not asserted.

**How it works.** The closure map C is *derived from M's own conditional factorization*, not authored. Concretely, M factors the next chunk as p(o_t | c_t) where c_t is a conditioning bundle (reference frame/identity embedding, layout/camera token, edit instruction token). C re-projects the realized output o_t back into each conditioning slot by deterministic, model-internal extractors that M already uses at generation time (its reference-frame encoder, its layout/camera head, its instruction-token decoder).

```pseudocode
INPUT: model M (factorization p(o_t|c_t)), initial asset a0, seed request r0, horizon T, tolerance eps
# Step 0: derive the closure map from M's own factorization
C = {  extract_ref  : o -> reference-frame slot   (M's ref encoder)
       extract_layout: o -> layout/camera token   (M's layout head)
       extract_instr : o -> instruction token      (M's instr decoder) }
# Step 0b: measure per-slot faithfulness floor on held-out real clips
#   (N_calib = 256 real clips, disjoint from evaluation seeds)
for each slot s in {ref, layout, instr}:
    floor_s = mean over clips of d_s( extract_s( encode_s(x) ), encode_s(x) )
#   floor_s is the extractor's own round-trip quantization error; drift_t is
#   reported residualized: drift_t^net = max(0, drift_t - floor_s)
# Step 1: seed the loop with the exogenous request only at t=0
c_0 = bundle(a0, r0)
o_1 ~ M( . | c_0 )
for t = 1 .. T:
    # Step 2: CLOSE the loop -- next request is a function of realized output
    c_{t+1} = C(o_t)                      # no human-authored r_t for t>=1
    o_{t+1} ~ M( . | c_{t+1} )
    # Step 3: measure closure trajectory
    drift_t      = d( c_{t+1}, c_t )       # request-space displacement
    drift_t^net  = max(0, drift_t - floor_s)  # residualized against extractor floor
# global fixed-point residual  R = || c_T - c_1 ||
#   R^net = max(0, R - floor_s)  (reported alongside raw R)
escape flag   = 1[ drift_t^net > eps  for any t ]
OUTPUT: {o_t}, closure-stability profile {drift_t, drift_t^net}, R, R^net, escape rate, floor_s per slot
```

Hyperparameters: horizon T (default 15–60s of generated video), tolerance eps (set per task as the 90th percentile of within-model drift on a calibration clip), the deterministic distance d (L2 on layout tokens, cosine on reference embeddings, token-edit distance on instruction tokens), and the faithfulness calibration set (N_calib = 256 real clips, disjoint from evaluation seeds).

**Why this design.** We chose a *derived* closure map over a *learned* or *human-authored* one because a learned C would introduce its own distribution and contaminate the measurement of M's loop behavior, accepting the cost that C is only as expressive as M's existing conditioning slots. We chose deterministic extractors already present in M's factorization over a separate perception model because using an external extractor would make the closure depend on a second model's errors, accepting the cost that models with weak conditioning interfaces yield coarse closures. We chose a fixed-point residual R over per-step task-execution scores because R directly operationalizes loop-sustention rather than script adherence, accepting the cost that R is not comparable across models with different conditioning granularities (we report within-model profiles and normalize across models by calibration drift). We chose to measure and subtract a per-slot faithfulness floor floor_s over a learned or assumed one because the extractors are trained as encoders for generation, not as calibrated decoders of realized outputs, and the layout/camera head in particular is a coarse quantizer; residualizing drift_t against floor_s makes the reported quantity 'loop-sustention above the extractor's own noise floor,' which is defensible regardless of head coarseness, accepting the cost that floor_s must be estimated on held-out real clips and that a very coarse head yields a large floor and a correspondingly conservative drift signal. A domain expert would not describe CLOVE as a variant of StreamEval-style scripted evaluation or of WorldScore's camera-trajectory conditioning, because in both the conditioning stream is exogenous by construction, whereas here the conditioning stream is a deterministic function of the model's realized output—the structural property prior protocols lack.

**Why it measures what we claim.** The per-step drift_t^net measures *loop-sustention* because the assumption that M's conditioning extractors are faithful inverses of its conditioning encoders makes drift_t equal to the output-space displacement the model itself would perceive; this assumption fails when the extractors are lossy (e.g., a layout head that quantizes camera tokens coarsely), in which case drift_t reflects extractor quantization rather than genuine instability—which is why we report drift_t^net = max(0, drift_t - floor_s) with floor_s measured as the extractor's round-trip error on 256 held-out real clips, so the reported quantity is loop-sustention above the extractor's own noise floor. The fixed-point residual R^net measures *whether the evaluation trajectory is a fixed point of M's own output distribution* because the assumption that the closure map is deterministic and M's sampling is the only stochasticity makes a zero residual equivalent to a stationary output distribution; this assumption fails when M's sampling temperature is high enough that stochastic variation mimics drift, in which case R reflects sampling variance rather than non-stationarity (we control this by reporting R at fixed low temperature and averaging over seeds). The escape flag measures *the model's inability to close an open-ended request loop* because the assumption that a model capable of loop closure will keep drift_t^net below the calibration percentile makes an escape a genuine breakdown; this assumption fails when the initial seed r₀ is out-of-distribution for M, in which case the escape reflects seed difficulty rather than loop incapacity (we report seed-conditional escape rates).

## Contribution

(1) CLOVE, a closure-operator evaluation protocol that replaces exogenous request scripts with a deterministic map derived from the model's own generative factorization, turning streaming evaluation into a fixed-point measurement. (2) A demonstration that loop-sustention (fixed-point residual, escape rate) is empirically dissociable from scripted task execution: models that score well on StreamEval-style Following can still fail to close the loop, establishing loop closure as a distinct capability axis. (3) A model-agnostic recipe for deriving closure maps from existing per-step conditioning interfaces, released as an evaluation harness compatible with StreamEval-style incremental request APIs.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|------------------------|
| Dataset | StreamEval's 300 scripted cases, replayed as open-ended seeds | Provides exogenous seed r₀; scripted r_t removed after t=0 |
| Faithfulness calibration set | 256 real clips, disjoint from evaluation seeds | Measures per-slot round-trip floor floor_s |
| Primary metric | Fixed-point residual R^net = max(0, R − floor_s), normalized by calibration drift | Operationalizes loop-sustention above extractor noise floor |
| Secondary metric | Per-step drift_t^net and seed-conditional escape rate | Reports trajectory shape and seed-difficulty confound |
| Baseline | StreamEval scripted protocol | Exogenous conditioning stream as reference |
| Baseline | WorldScore camera-trajectory conditioning | Exogenous layout stream as reference |
| Baseline | TC-Bench interpolation scoring | Measures composition, not loop closure |
| Ablation-of-ours | CLOVE with randomized C (shuffled extractors) | Tests whether derived closure map is essential |
| Ablation-of-ours | CLOVE with raw R (no floor subtraction) | Tests whether floor subtraction changes conclusions |

### Why this setup validates the claim

The central claim is that replacing an exogenous request script with a model-derived closure map yields a diagnostic of loop-sustention that scripted protocols cannot see. The dataset supplies initial assets and seed requests from StreamEval's scripted cases, but we discard the human-authored r_t for t≥1, so the only way to sustain the trajectory is through CLOVE's closure map. StreamEval's scripted protocol tests whether a model follows a human script; if CLOVE's residual R^net is low on cases where the scripted score is high, that shows script adherence and loop-sustention are separable, which is the claim. WorldScore's camera-trajectory conditioning is exogenous by construction, so it cannot reveal whether a model can close its own loop; a divergence between WorldScore controllability and CLOVE R^net on the same clips would confirm the structural distinction. TC-Bench's interpolation scoring measures temporal composition between fixed endpoints, not sustained closure; parity there with divergence in R^net would show CLOVE is measuring something orthogonal. The shuffled-C ablation is the key falsifier of the mechanism: if randomizing the closure map leaves R^net unchanged, then the derived map is not doing the work and the measured stability is an artifact of the model's sampling, not its conditioning factorization. The raw-R ablation is the faithfulness control: if R and R^net order models identically, the floor subtraction is not load-bearing; if they diverge, the floor is doing measurable work and must be reported. The primary metric R^net is chosen because it is the only quantity in the setup that is defined relative to M's own output distribution *and* corrected for the extractor's own quantization lattice; task scores and controllability scores are defined relative to an external target, so they cannot detect whether the trajectory is a fixed point of the model's own factorization.

### Expected outcome and causal chain

**vs. StreamEval scripted protocol** — On a case where the human script asks for a slow camera pan with an object entering frame, the StreamEval protocol keeps feeding the authored pan token regardless of what the model actually renders, so a model that drifts off the pan still receives the correct next instruction and its scripted score stays high. Our method instead re-extracts the layout token from the realized frame, so the next request is whatever the model actually produced; a model that cannot hold the pan will show drift_t^net rising above the calibration percentile within a few steps. We expect a noticeable gap on open-ended pan cases where StreamEval scores are high but CLOVE escape rate is elevated, with parity on short clips where drift has not yet accumulated. The raw-R vs R^net comparison on these cases will show whether the gap survives floor subtraction; we expect it to survive on reference-embedding and instruction-token slots (finer-grained extractors) and to shrink on the layout slot (coarser quantizer), which is the predicted faithfulness signature.

**vs. WorldScore camera-trajectory conditioning** — On a case where the camera trajectory is specified as a sequence of poses, WorldScore feeds each pose exogenously and scores controllability against the specified path, so a model that renders a plausible but off-path scene still receives the correct next pose and its controllability score reflects only per-scene fidelity. Our method instead derives the next layout token from the realized frame's own layout head, so an off-path render propagates into the next request and the trajectory either converges to a fixed point or escapes. We expect WorldScore controllability to be roughly flat across models that differ sharply in CLOVE R^net, with the divergence concentrated on long horizons where exogenous conditioning masks accumulation, and with the layout-slot floor_s reported alongside so the reader can see how much of the divergence is extractor quantization.

**vs. TC-Bench interpolation scoring** — On a case where a model interpolates between an initial and final frame, TC-Bench scores the composition against ground-truth intermediates, so a model that produces a visually smooth but non-stationary sequence can score well because the endpoints anchor the evaluation. Our method instead has no final frame to anchor to; the trajectory must close on itself, so a model whose intermediate frames drift in identity or layout will show rising drift_t^net even though its TC-Bench composition score is high. We expect TC-Bench scores to be near-parity across models that separate clearly on CLOVE fixed-point residual R^net, with the gap appearing on the subset where the seed request is open-ended rather than endpoint-anchored.

### What would falsify this idea

If the shuffled-C ablation produces fixed-point residuals and escape rates statistically indistinguishable from the derived-C condition, then the closure map is not doing the work and the measured stability reflects sampling variance rather than the model's conditional factorization. Equally, if CLOVE's residual ordering across models is perfectly predicted by StreamEval scripted scores, then loop-sustention is not separable from script adherence and the structural claim collapses; and if R^net ordering is identical to raw-R ordering on every slot, the faithfulness-floor correction is not load-bearing and the load-bearing assumption remains untested.

## References

1. StreamEval: Evaluating Interactive Streaming Video Generation Models
2. TC-Bench: Benchmarking Temporal Compositionality in Text-to-Video and Image-to-Video Generation
3. WorldScore: A Unified Evaluation Benchmark for World Generation
4. Davidsonian Scene Graph: Improving Reliability in Fine-grained Evaluation for Text-to-Image Generation
5. FETV: A Benchmark for Fine-Grained Evaluation of Open-Domain Text-to-Video Generation
6. Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation
7. VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models
8. All You May Need for VQA are Image Captions
