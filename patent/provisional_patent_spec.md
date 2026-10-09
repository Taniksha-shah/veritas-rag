# UNITED STATES PATENT AND TRADEMARK OFFICE (USPTO)
## PROVISIONAL PATENT APPLICATION SPECIFICATION

### TITLE OF THE INVENTION
**SYSTEMS AND METHODS FOR HALLUCINATION-FREE RETRIEVAL-AUGMENTED GENERATION USING RECURSIVE CROSS-VERIFICATION AND SENTENCE-LEVEL CITATION ATTRIBUTION**

---

### INVENTORS
Taniksha Shah et al.

---

### FIELD OF THE INVENTION
The present invention relates generally to natural language processing, machine learning, and information retrieval. More particularly, the invention relates to domain-specific retrieval-augmented generation (RAG) systems that eliminate factual hallucinations in high-stakes environments through recursive multi-model cross-verification feedback loops and fine-grained, sentence-to-source-span citation attribution.

---

### BACKGROUND OF THE INVENTION
Large language models (LLMs) have achieved state-of-the-art capabilities in natural language synthesis across numerous domains. However, in high-stakes domains such as pharmaceutical research and development (R&D), clinical decision support, and legal analysis, LLMs suffer from a fatal flaw: "hallucination," defined as the generation of fluent, syntactically plausible assertions that are factually unsupported by, or directly contradictory to, underlying ground truth.

Standard retrieval-augmented generation (RAG) frameworks attempt to mitigate hallucination by retrieving passages from a document repository and conditioning the LLM's prompt on the retrieved context. Nonetheless, standard RAG pipelines frequently fail because:
1. The generator model frequently ignores or misinterprets technical context (e.g., dosage numbers, statistical p-values, or exclusionary legal clauses);
2. Prior knowledge encoded in model weights leaks into generated answers as unverified extrapolations;
3. Existing critique mechanisms (e.g., Self-RAG or RARR) are single-pass; when an LLM is prompted to edit an error in a single pass, it frequently introduces novel secondary hallucinations during the revision pass;
4. Prior art systems lack deterministic, fine-grained citation mechanisms capable of linking individual assertions directly to specific, auditable character spans in the underlying source documents.

Consequently, there is an urgent and unmet technological need for an automated generative architecture that guarantees factual attestation through recursive cross-verification and provides exact sentence-to-span attribution.

---

### SUMMARY OF THE INVENTION
The present invention addresses these deficiencies by providing a novel RAG architecture termed **Veritas**. Veritas implements:
1. **A Recursive Cross-Verification Pipeline**: An iterative control loop operating between a primary generator model and an independent verifier model. The verifier decomposes draft text into atomic factual propositions, retrieves candidate supporting passages, and evaluates textual entailment. Assertions that fail entailment are compiled into an explicit discrepancy report specifying the exact contradictory or missing evidence. The generator iteratively refines the text conditioned on the discrepancy feedback until every assertion is proven or safety-pruned.
2. **A Sentence-to-Source-Span Citation Attribution Module**: A post-verification alignment engine that segments the attested output into constituent sentences, identifies maximal contiguous semantic and lexical matches within the underlying document repository, and annotates each sentence with character offset coordinates `[start_char, end_char]` and clickable citation identifiers.
3. **An On-Premises, Sovereign Knowledge Appliance**: A self-hosted system that maintains private enterprise data boundaries, generates cryptographic audit records for every verification decision, and satisfies strict compliance mandates (HIPAA, SOC 2).

---

### DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENTS

#### 1. System Architecture
The system comprises the following interconnected functional modules:
- **Hybrid Retrieval Subsystem**: Combines sparse term-frequency indexing (Okapi BM25) with dense vector representations (transformer embeddings) fused via Reciprocal Rank Fusion (RRF):
  $$RRF\_score(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
- **Primary Generator Subsystem**: Produces initial drafts $y^{(0)}$ and subsequent iterative refinements $y^{(t)}$ conditioned on retrieved passages $\mathcal{P}$ and discrepancy feedback $\Delta^{(t)}$.
- **Atomic Claim Decomposition Subsystem**: Decomposes multi-sentence candidate answers into discrete, self-contained atomic propositions $\{c_1, \dots, c_m\}$ by resolving anaphoric references, splitting compound clauses, and preserving numerical qualifiers.
- **Textual Entailment Verification Subsystem**: Executes natural language inference (NLI) for each atomic claim against candidate evidence passages, assigning a classification selected from `SUPPORTED`, `CONTRADICTED`, and `UNVERIFIED`.
- **Discrepancy Feedback Generator**: Synthesizes specific corrective directives for every unverified or contradicted claim, identifying the conflict and specifying corrective constraints.
- **Sentence-to-Source-Span Attribution Subsystem**: Calculates character-level boundary coordinates $[s_{\text{start}}, s_{\text{end}}]$ in source documents corresponding to each verified sentence, formatting inline citation markers and audit tables.
- **Audit & Compliance Logger**: Generates persistent, immutable audit records detailing prompt hashes, intermediate drafts, decomposed claims, verification statuses, and timestamps.

#### 2. The Recursive Cross-Verification Loop Method
1. Receive a domain query $q$ from a client device.
2. Retrieve top-$k$ evidence passages $\mathcal{P}$ from an indexed corpus using hybrid retrieval.
3. Generate an initial candidate draft $y^{(0)} = G(q, \mathcal{P})$.
4. For iteration $t = 1$ to $T_{\text{max}}$:
   a. Decompose $y^{(t-1)}$ into atomic claims $\{c_1, \dots, c_m\}$.
   b. For each claim $c_i$, retrieve best-matching evidence $p^* \in \mathcal{P}$ and compute entailment status $V(c_i, p^*)$.
   c. If all claims are `SUPPORTED`, exit loop with verified text $y^* = y^{(t-1)}$.
   d. If any claim is `CONTRADICTED` or `UNVERIFIED`:
      - If $t < T_{\text{max}}$, compile discrepancy report $\Delta^{(t)}$ and generate revised draft $y^{(t)} = G(q, \mathcal{P}, y^{(t-1)}, \Delta^{(t)})$.
      - If $t = T_{\text{max}}$, execute safety pruning by removing or qualifying sentences containing unverified claims.
5. Partition verified text into sentences and align each sentence to exact character spans in $\mathcal{P}$.
6. Deliver attributed output with inline citations and audit tables to client.

---

### PATENT CLAIMS

**What is claimed is:**

1. A computer-implemented system for generating hallucination-free, citation-attributed natural language text, comprising:
   - one or more processors; and
   - a memory storing instructions that, when executed by the one or more processors, cause the system to:
     - receive an input query;
     - retrieve a plurality of candidate evidence passages from a document repository;
     - generate, via a primary generator model, a candidate answer conditioned on the candidate evidence passages;
     - execute a recursive cross-verification loop between the generator model and a verification subsystem, wherein the verification subsystem decomposes the candidate answer into a plurality of atomic factual claims, evaluates textual entailment for each atomic claim against the candidate evidence passages, and synthesizes a discrepancy report comprising corrective feedback for any claim that fails entailment;
     - iteratively refine the candidate answer using the generator model conditioned on the discrepancy report until all atomic claims are verified or a maximum iteration threshold is reached; and
     - generate sentence-to-source-span citations mapping constituent sentences of the refined answer to specific character spans within the candidate evidence passages.

2. The system of claim 1, wherein the verification subsystem decomposes the candidate answer into atomic claims by resolving pronouns and implicit references into explicit entities and preserving quantitative metrics and epistemic qualifiers.

3. The system of claim 1, wherein the textual entailment evaluation assigns each atomic claim a status selected from the group consisting of: SUPPORTED, CONTRADICTED, and UNVERIFIED.

4. The system of claim 3, wherein the discrepancy report identifies the specific contradictory evidence for any claim assigned the status CONTRADICTED, and specifies that the claim must be excised or corrected.

5. The system of claim 3, wherein the discrepancy report identifies any claim assigned the status UNVERIFIED as lacking evidentiary proof, and instructs the generator model to remove the unverified assertion.

6. The system of claim 1, wherein upon reaching the maximum iteration threshold with one or more remaining unverified claims, the system executes safety pruning to remove sentences containing the unverified claims from the final answer.

7. The system of claim 1, wherein retrieving the candidate evidence passages comprises executing hybrid retrieval combining sparse BM25 indexing and dense transformer vector embeddings via Reciprocal Rank Fusion (RRF).

8. The system of claim 1, wherein the sentence-to-source-span citation comprises:
   - a document identifier;
   - an integer character start offset;
   - an integer character end offset;
   - an exact quoted snippet from the source document; and
   - an attestation confidence score.

9. The system of claim 1, further comprising an audit logging module configured to persist an immutable execution record comprising the input query, intermediate iterations, decomposed claims, entailment statuses, and citation spans.

10. The system of claim 1, configured to operate within a sovereign, self-hosted on-premises computing appliance without transmitting query data to external public networks.

11. A computer-implemented method for verified retrieval-augmented generation, comprising:
    - receiving an input query;
    - retrieving, via hybrid dense and sparse ranking, a set of evidence passages from a curated corpus;
    - generating, via a first language model, an initial draft response conditioned on the evidence passages;
    - decomposing, via a claim decomposition model, the draft response into a set of atomic factual claims;
    - determining, for each atomic factual claim, whether the claim is entailed by at least one evidence passage;
    - upon determining that at least one atomic claim is not entailed:
      - constructing an explicit discrepancy feedback prompt detailing failed claims and corresponding evidence contradictions;
      - generating, via the first language model, a refined draft response conditioned on the evidence passages and the discrepancy feedback prompt; and
      - repeating the decomposing, determining, and constructing steps until all atomic claims are entailed or a threshold number of iterations has elapsed; and
    - aligning verified sentences of the response to specific character offset ranges of the evidence passages.

12. The method of claim 11, wherein decomposing the draft response comprises formatting input text with a task-specific prompt and extracting a structured JSON array of discrete propositions.

13. The method of claim 11, wherein determining entailment comprises computing a directional Natural Language Inference score using a cross-encoder model or a secondary language model verifier.

14. The method of claim 11, wherein the discrepancy feedback prompt instructs the first language model to retain all supported claims while removing ungrounded propositions.

15. The method of claim 11, wherein the threshold number of iterations is an integer between two and five.

16. The method of claim 11, further comprising appending clickable citation indicators adjacent to each verified sentence, linked to a display modal revealing the aligned evidence passage.

17. The method of claim 11, further comprising compiling a tabular citation audit report comprising source document identifiers, character offsets, and attestation confidence metrics.

18. The method of claim 11, executed in an on-premises enterprise environment compliant with HIPAA and SOC 2 security standards.

19. The method of claim 11, wherein retrieving evidence passages comprises applying Reciprocal Rank Fusion with a smoothing constant $k=60$.

20. A non-transitory computer-readable storage medium storing instructions that, when executed by one or more processors, cause the one or more processors to perform operations comprising:
    - retrieving evidence passages relevant to a domain query from an indexed knowledge base;
    - producing an initial draft answer conditioned on the evidence passages;
    - iteratively executing a recursive verification loop wherein an independent verifier extracts atomic propositions from the draft answer, evaluates textual entailment against the evidence passages, and outputs discrepancy feedback driving generator refinement; and
    - mapping each verified sentence in the refined answer to exact character span coordinates in the evidence passages to generate an auditable, citation-grounded response.
