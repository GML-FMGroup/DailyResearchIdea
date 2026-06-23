# CERT-Bench: Certificate-Guaranteed Benchmark Generation via Trace-Embedded Proofs

## Motivation

The ABC checklist (Establishing Best Practices for Building Rigorous Agentic Benchmarks) provides a manual approach for ensuring benchmark validity but requires human application and cannot scale. BeTaL (Automating Benchmark Design) automates generation but relies on human-designed templates and does not provide automated verification of the generated benchmarks' correctness. The root cause is that existing methods treat validity as external to the generation process, requiring human judgment or separate verification post-generation. We propose to embed verification into the generation trace itself, so that each task is produced alongside a formal certificate that attests to its solvability, diversity, and representativeness, enabling fully autonomous benchmark creation.

## Key Insight

An LLM conditioned on a formal specification of validity properties can generate a task and a structured proof trace that, by construction, contains the evidence needed to verify those properties, eliminating the need for external human-provided templates or checklists.

## Method

### CERT-Bench: Certificate-Guaranteed Benchmark Generation via Trace-Embedded Proofs

**Key assumption**: The LLM can reliably generate accurate certificates. To mitigate hallucination risk, we incorporate an iterative self-consistency verification loop: generate multiple certificate candidates per task and only accept tasks with consistent certificates across samples (e.g., at least 2 out of 3 pass verification).

(A) **What it is**: CERT-Bench is a framework that takes a domain specification (e.g., a set of APIs, policies, and a domain distribution) and generates a benchmark of tasks, each accompanied by a formal certificate of validity derived from the generation trace. The certificate is a structured JSON object containing sub-proofs for solvability, diversity, and representativeness, which are automatically verifiable by a set of verifiers.

(B) **How it works**:
```python
# Pseudocode for CERT-Bench generation and verification with iterative consistency

def generate_task_with_certificate(D, num_candidates=3, consistency_threshold=2):
    prompt = f"""Generate a benchmark task for domain D. A valid task must include:
- Task description and success criteria
- A solvability proof: a sequence of actions using the APIs that achieves the success criteria (simulate if needed).
- A diversity proof: show that the task is dissimilar to each reference task in reference_tasks (use embedding distance > threshold tau=0.8, embedder='sentence-transformers/all-MiniLM-L6-v2').
- A representativeness proof: show that the task's parameter vector has high likelihood under the domain_model (domain_model='GaussianMixture(n_components=5)' with likelihood > epsilon=0.05).
Output a JSON with fields: "task", "certificate"."""
    candidates = []
    for _ in range(num_candidates):
        response = llm.generate(prompt, temperature=0.7, max_tokens=2048)
        output = json.loads(response)
        cert = output["certificate"]
        if verify_certificate(cert, D):
            candidates.append(output)
    if len(candidates) >= consistency_threshold:
        # Select the task with highest consensus (e.g., most common task description)
        return candidates[0]["task"], candidates[0]["certificate"]
    else:
        return None, None  # generation failed; optionally retry

def verify_certificate(cert, D):
    # Solvability check
    solv_proof = cert["solvability"]
    if not simulate_steps(solv_proof, D["APIs"]):
        return False
    # Diversity check
    div_proof = cert["diversity"]
    task_embed = embed_task(task, model='sentence-transformers/all-MiniLM-L6-v2')
    for ref_task, ref_embed in zip(D["reference_tasks"], div_proof["ref_embeds"]):
        if cosine_similarity(task_embed, ref_embed) > div_proof["threshold"]:
            return False
    # Representativeness check
    rep_proof = cert["representativeness"]
    task_params = extract_params(task)  # e.g., number of steps, tool types
    if domain_model.log_likelihood(task_params) < rep_proof["likelihood"]:
        return False
    return True
```

(C) **Why this design**: We chose to generate the certificate as part of the LLM's output trace rather than as a separate post-hoc verification step because this forces the LLM to reason about validity during generation, which increases the chance that the task is actually valid and reduces the risk of hallucinating a fake certificate. We require the LLM to produce explicit proof steps (e.g., action sequences for solvability) rather than a simple claim, allowing the verifier to check each step independently using a simulator. This design accepts the cost that generation is slower due to the longer prompt and output, but gains provable correctness. We use a fixed similarity threshold (0.8) for diversity rather than an adaptive one because it provides a clear, reproducible standard; the trade-off is that tasks may be slightly less diverse if the threshold is too high, but this can be adjusted per domain. The likelihood threshold (0.05) for representativeness ensures that tasks are within the domain distribution, but may exclude rare but valid edge cases; this is acceptable because the goal is to generate representative benchmarks, not exhaustive ones. By embedding the verification logic into the generation prompt, we avoid needing a separate classifier model or reinforcement learning, which would introduce additional complexity and potential for reward hacking. To address hallucination risk, we added an iterative self-consistency loop: we generate multiple certificate candidates and only accept tasks that have consistent certificates across samples, ensuring that the final certificate is reliable.

(D) **Why it measures what we claim**: The solvability proof as a sequence of API calls measures the computational quantity of 'existence of a solution path'; this is equivalent to the motivation-level concept 'solvability' under the assumption that the simulator correctly reflects the real-world API behavior and that the provided actions are sufficient to achieve the goal. This assumption fails when the simulator abstracts away real-world constraints (e.g., latency, concurrency), in which case the metric reflects only simulated solvability. The diversity proof's cosine similarity between task embeddings measures 'dissimilarity to reference tasks'; this is equivalent to 'diversity' under the assumption that the embedding space captures task-level variation and that the threshold correctly separates 'new' from 'redundant' tasks; this assumption fails when the embedding is insensitive to critical differences (e.g., only surface-level text varies), in which case the metric reflects text similarity rather than functional diversity. The representativeness proof's log-likelihood under the domain model measures 'typicality within the domain'; this equals 'representativeness' under the assumption that the domain model accurately captures the true distribution of tasks; this assumption fails when the model is misspecified (e.g., Gaussian for multimodal distribution), in which case the metric reflects likelihood under the wrong model.

## Contribution

(1) CERT-Bench, a framework for generating benchmark tasks with formal certificates of validity derived from generation traces, enabling autonomous verification. (2) A technique for conditioning LLMs to produce structured proof traces alongside tasks, using a formal specification of validity properties (solvability, diversity, representativeness). (3) A set of verifiers that automatically check certificates against domain simulators and reference sets, ensuring correctness without human intervention.

## Experiment

### Evaluation Setup

| Role | Choice | Rationale (≤12 words) |
|------|--------|----------------------|
| Dataset | Airline booking domain specification | Realistic multi-tool domain. |
| Primary metric | Task validity rate | Directly measures certificate correctness. |
| Baseline 1 | Random generation | No validity constraints baseline. |
| Baseline 2 | LLM-only generation | Without certificate prompting. |
| Ablation of ours | CERT-Bench (no verification) | Tests necessity of verification step. |

### Why this setup validates the claim
This experimental design tests whether CERT-Bench's certificate generation and verification improve benchmark quality. Using a realistic domain specification ensures generalizability. The primary metric, task validity rate, directly captures the core claim: generated tasks are solvable, diverse, and representative. Comparing against random and LLM-only baselines isolates the effect of our certificate framework. The ablation (no verification) tests whether the verification step itself contributes to validity, or if mere prompting suffices. If CERT-Bench outperforms both baselines and the ablation on validity rate, the claim is supported; otherwise, the mechanism is not functioning as intended.

### Expected outcome and causal chain

**vs. Random generation** — On a case where the domain has many invalid task configurations (e.g., conflicting API states), random generation produces mostly invalid tasks because it has no mechanism to ensure solvability, diversity, or representativeness. Our method explicitly generates and verifies certificates, so it produces only tasks that pass all checks. We expect CERT-Bench to have a markedly higher task validity rate (e.g., >80%) compared to random (<10%).

**vs. LLM-only generation** — On a case where the LLM hallucinates a plausible but unsolvable task (e.g., a sequence of API calls that seems right but fails in simulation), LLM-only generation accepts it because it lacks explicit proof steps. Our method requires the LLM to output a concrete action sequence as part of the certificate, which the verifier then simulates; any hallucination is caught. We expect CERT-Bench to achieve >90% validity, while LLM-only may drop to 50-70% due to unsolvable tasks.

**vs. Ablation (no verification)** — On a case where the LLM outputs a certificate that looks valid but contains errors (e.g., threshold manipulation), the no-verification version accepts it, producing an invalid task. With verification, such failures are detected and regenerated or rejected. We expect the full CERT-Bench to have >90% validity, while the ablation may be around 70-80%, confirming that verification is critical.

### What would falsify this idea
If CERT-Bench's task validity rate is not significantly higher than the ablation (no verification), then the verification step adds no value, contradicting the claim that certificates ensure validity.

## References

1. Establishing Best Practices for Building Rigorous Agentic Benchmarks
2. Automating Benchmark Design
3. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains
4. LLM-POET: Evolving Complex Environments using Large Language Models
5. APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets
6. Automatic Generation of Benchmarks and Reliable LLM Judgment for Code Tasks
7. MarioGPT: Open-Ended Text2Level Generation through Large Language Models
8. Evolution Gym: A Large-Scale Benchmark for Evolving Soft Robots
