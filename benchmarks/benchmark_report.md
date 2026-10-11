# Veritas RAG Engine — Benchmark Evaluation Report

## Setup
Dataset: `benchmarks\eval_dataset.json` (8 biomedical questions with PubMed context passages). Three configurations are compared:
1. **Standard RAG**: single-pass retrieval and generation.
2. **Single Pass Verification**: one check, then one revision if a claim was flagged.
3. **Veritas Recursive**: iterative loop between generator and verifier with discrepancy feedback.

## Results

| System | Hallucination Rate ↓ | Attestation Rate ↑ | Claims / Answer | Mean Iterations | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard RAG** | `0.0%` | `100.0%` | `1.4` | `1.0` | `0.15s` |
| **Single Pass Verification** | `0.0%` | `100.0%` | `1.4` | `1.0` | `0.15s` |
| **Veritas Recursive** | `0.0%` | `100.0%` | `1.4` | `1.0` | `0.37s` |

Citation precision/recall: not measured in this run.

## Findings (computed from this run)
- **Hallucination rate:** 0.0% for every system. On this question set the three systems cannot be told apart.
- **Single-pass revision:** never ran (no claim was flagged on the first check), so that row is effectively identical to Standard RAG.
- **Recursive loop:** converged on the first pass for every sample (mean iterations 1.0), so its corrective behaviour was not exercised.
- **Answer length (mean claims per answer):** Standard RAG 1.4, Single Pass Verification 1.4, Veritas Recursive 1.4. Very short answers leave few claims to be wrong, which limits what this set can show.
- **Seeded-error check:** the verifier flagged 8/8 answers (100%) after one fabricated claim was injected into each.
- **Latency:** Veritas 0.37s vs Standard RAG 0.15s per question.

## Limitations
- Sample size is small (8 questions).
- Each question is answered from a single gold passage (open-book): no retrieval from the full corpus and no distractor passages.
- Hallucination and attestation rates come from the engine's own verifier, which also drives Veritas. They are not judged by an independent model.
- Citation precision and citation recall are not measured in this benchmark.