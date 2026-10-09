"""
Veritas RAG Engine — Interactive Web Appliance.
Features:
- Live Precision Querying with Real-Time Recursive Verification Stepper
- Iteration-by-Iteration Claim Decomposition & Entailment Trace Inspector
- Sentence-to-Source-Span Citation Badges & Evidence Viewer
- Knowledge Base Corpus Explorer & Dynamic PDF/Text Document Ingestion
- Benchmark Performance Dashboard & Comparative Visualizations
- Enterprise Compliance Audit Trail Exporter
"""

import json
import time
from pathlib import Path
import streamlit as st
import pandas as pd

from veritas.config import VeritasConfig
from veritas.pipeline.engine import VeritasEngine, VeritasResponse
from veritas.retrieval.indexer import CorpusIndexer

# Page configuration
st.set_page_config(
    page_title="Veritas RAG Engine — Self-Hosted Knowledge Appliance",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for high-stakes enterprise UI
st.markdown(
    """
    <style>
    .main-header {
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
    }
    .badge-supported {
        background-color: #DCFCE7;
        color: #166534;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-contradicted {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-unverified {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine():
    config = VeritasConfig()
    return VeritasEngine(config=config)


engine = get_engine()

# -------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("Veritas Appliance")
st.sidebar.caption("Recursive Citation Verification v1.0.0")

st.sidebar.subheader("Engine Parameters")
max_iterations = st.sidebar.slider("Max Verification Iterations", min_value=1, max_value=5, value=3)
top_k = st.sidebar.slider("Top-k Evidence Passages", min_value=1, max_value=10, value=5)
confidence_thresh = st.sidebar.slider("Entailment Threshold", min_value=0.5, max_value=0.95, value=0.75, step=0.05)
strict_pruning = st.sidebar.toggle("Strict Safety Pruning", value=True, help="Prune or qualify unverified claims at max iterations")

# Corpus stats
raw_papers_path = engine.config.raw_papers_dir
paper_count = len(list(raw_papers_path.glob("*.txt"))) if raw_papers_path.exists() else 0
st.sidebar.divider()
st.sidebar.metric("Indexed Knowledge Base", f"{paper_count} Papers", "Europe PMC / PubMed")

# Sample Prompts
st.sidebar.subheader("Domain Sample Queries")
samples = [
    "What is the relationship between cumulative clozapine exposure and the risk of hematologic malignancy?",
    "What are the recommended treatments and monitoring protocols for cardiac irAEs?",
    "What evidence exists regarding non-invasive respiratory support in interstitial lung diseases with acute respiratory failure?",
    "Can a normal ECG rule out myocarditis, and what diagnostic limitations exist for troponin?",
]

selected_sample = None
for s in samples:
    if st.sidebar.button(s[:65] + "...", key=s):
        selected_sample = s

# -------------------------------------------------------------
# Main Header
# -------------------------------------------------------------
st.markdown("<h1 class='main-header'>⚖️ Veritas RAG Engine</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-header'>Hallucination-Free Domain-Specific RAG with Recursive Cross-Verification & Span Attribution</p>",
    unsafe_allow_html=True,
)

# Tabs
tab_query, tab_corpus, tab_benchmarks, tab_audit, tab_docs = st.tabs([
    "🔬 Precision Query & Verifier",
    "📚 Corpus Explorer & Ingestion",
    "📊 Benchmark Dashboard",
    "🛡️ Compliance Audit Trail",
    "📄 Paper & Patent Drafts",
])

# -------------------------------------------------------------
# TAB 1: Query & Live Verification
# -------------------------------------------------------------
with tab_query:
    st.subheader("High-Stakes Scientific & Legal Query Interface")

    query_input = st.text_area(
        "Enter your query:",
        value=selected_sample or "What is the relationship between cumulative clozapine exposure and the risk of hematologic malignancy?",
        height=90,
    )

    col_btn, col_clear = st.columns([1, 6])
    with col_btn:
        run_query = st.button("🚀 Run Veritas", type="primary", use_container_width=True)

    if run_query and query_input:
        engine.config.max_verification_iterations = max_iterations
        engine.config.top_k_retrieval = top_k
        engine.config.entailment_confidence_threshold = confidence_thresh
        engine.config.strict_safety_pruning = strict_pruning

        with st.status("Executing Veritas Verification Pipeline...", expanded=True) as status:
            st.write("🔍 Step 1: Hybrid Retrieval (BM25 + Dense embeddings + RRF)...")
            time.sleep(0.3)
            st.write("✍️ Step 2: Primary LLM initial draft generation...")
            time.sleep(0.3)
            st.write("🔄 Step 3: Executing Recursive Cross-Verification Loop...")

            # Run query
            response: VeritasResponse = engine.query(query_input)

            st.write(f"✅ Step 4: Iterations completed ({response.iterations_run}). Attestation rate: {response.final_attestation_rate * 100:.1f}%.")
            st.write("🏷️ Step 5: Sentence-to-source-span citation attribution applied.")
            status.update(label="Query Verification Complete!", state="complete", expanded=False)

        # Performance summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final Hallucination Rate", f"{response.final_hallucination_rate * 100:.1f}%", delta=f"{(response.final_hallucination_rate - response.initial_hallucination_rate) * 100:.1f}%", delta_color="inverse")
        m2.metric("Claim Attestation Rate", f"{response.final_attestation_rate * 100:.1f}%", delta="Proven")
        m3.metric("Iterations Run", f"{response.iterations_run} rounds", "Converged" if response.converged else "Pruned")
        m4.metric("Latency", f"{response.execution_time_seconds:.2f}s", f"{len(response.citations)} Citations")

        st.divider()

        # Final Verified Output
        st.markdown("### 📜 Verified Attested Output")
        st.info(response.annotated_answer)

        # Evidence Citation Table
        st.markdown(response.audit_table_markdown)

        st.divider()

        # Detailed Verification Drawer / Traces
        st.subheader("🔍 Recursive Verification Trace & Audit Drawer")
        trace_tabs = st.tabs([f"Iteration {t['iteration']}" for t in response.iteration_traces])

        for t_idx, trace in enumerate(response.iteration_traces):
            with trace_tabs[t_idx]:
                st.markdown(f"**Draft at Iteration {trace['iteration']}:**")
                st.code(trace["draft"], language="text")

                col_stat1, col_stat2, col_stat3 = st.columns(3)
                col_stat1.write(f"🟢 **Supported Claims:** {trace['supported_count']}")
                col_stat2.write(f"🔴 **Contradicted Claims:** {trace['contradicted_count']}")
                col_stat3.write(f"🟡 **Unverified Claims:** {trace['unverified_count']}")

                st.markdown("**Decomposed Atomic Claims & Entailment Status:**")
                claims_data = []
                for c in trace["claims"]:
                    status_pill = f"🟢 {c['status']}" if c["status"] == "SUPPORTED" else (f"🔴 {c['status']}" if c["status"] == "CONTRADICTED" else f"🟡 {c['status']}")
                    claims_data.append({
                        "ID": c["claim_id"],
                        "Atomic Claim": c["text"],
                        "Entailment Status": status_pill,
                        "Confidence": f"{c.get('confidence', 0.0) * 100:.1f}%",
                        "Source Document": c.get("evidence_doc_id", "N/A"),
                        "Evidence Snippet": c.get("evidence_snippet", "")[:100] + "...",
                    })
                st.dataframe(pd.DataFrame(claims_data), use_container_width=True)

                if trace.get("feedback_prompt"):
                    st.markdown("**Discrepancy Feedback Injected into Generator:**")
                    st.warning(trace["feedback_prompt"])

        # Retrieved passages inspected
        with st.expander("📚 Inspect Raw Retrieved Evidence Passages (Top-k)"):
            for i, p in enumerate(response.retrieved_passages, start=1):
                st.markdown(f"**Passage {i} — [{p.get('doc_id')}]** ({p.get('title', 'Unknown')})")
                st.caption(f"RRF Rank: {p.get('final_rank', i)} | Sparse Score: {p.get('sparse_score', 0):.2f} | Dense Score: {p.get('dense_score', 0):.2f}")
                st.text(p.get("text", "")[:600] + "...")

# -------------------------------------------------------------
# TAB 2: Corpus Explorer & Ingestion
# -------------------------------------------------------------
with tab_corpus:
    st.subheader("📚 Knowledge Base Management & Ingestion")

    col_up, col_list = st.columns([1, 1])

    with col_up:
        st.markdown("#### Upload New Domain Documents")
        uploaded_file = st.file_uploader("Upload scientific paper or legal contract (.txt or .pdf)", type=["txt", "pdf"])
        if uploaded_file is not None:
            save_path = engine.config.raw_papers_dir / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded and stored: `{uploaded_file.name}`")
            if st.button("Index Uploaded Document"):
                indexer = CorpusIndexer()
                indexer.index_directory(engine.config.raw_papers_dir)
                engine.retriever = indexer.build_and_save_index(engine.config.index_dir)
                st.success("Successfully rebuilt hybrid index with new document!")

    with col_list:
        st.markdown("#### Indexed PubMed Corpus Documents")
        if engine.config.raw_papers_dir.exists():
            paper_files = sorted(list(engine.config.raw_papers_dir.glob("*.txt")))[:30]
            st.write(f"Displaying {len(paper_files)} available source documents:")
            for pf in paper_files:
                with st.expander(f"📄 {pf.name}"):
                    try:
                        with open(pf, "r", encoding="utf-8", errors="ignore") as f:
                            preview = f.read(500)
                        st.text(preview + "...")
                    except Exception as e:
                        st.error(f"Error reading {pf.name}: {e}")

# -------------------------------------------------------------
# TAB 3: Benchmark Dashboard
# -------------------------------------------------------------
with tab_benchmarks:
    st.subheader("📊 Comparative Benchmark Evaluation (Phase 1)")
    st.markdown("Direct comparison against baseline RAG and single-pass revision on PubMed evaluation set.")

    bench_file = Path("benchmarks/benchmark_results.json")
    if bench_file.exists():
        with open(bench_file, "r", encoding="utf-8") as f:
            b_data = json.load(f)

        df_bench = pd.DataFrame([
            {
                "System": k.replace("_", " "),
                "Hallucination Rate (%)": v["mean_hallucination_rate"] * 100,
                "Attestation Rate (%)": v["mean_attestation_rate"] * 100,
                "Citation Precision (%)": v["mean_citation_precision"] * 100,
                "Citation Recall (%)": v["mean_citation_recall"] * 100,
                "Mean Iterations": v["mean_iterations"],
                "Latency (s)": v["mean_latency_seconds"],
            }
            for k, v in b_data.items()
        ])

        st.dataframe(df_bench.style.highlight_min(subset=["Hallucination Rate (%)"], color="#dcfce7").highlight_max(subset=["Attestation Rate (%)"], color="#dcfce7"), use_container_width=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Hallucination Rate Comparison (Lower is Better)")
            st.bar_chart(df_bench.set_index("System")["Hallucination Rate (%)"])
        with col_c2:
            st.markdown("#### Claim Attestation Rate (Higher is Better)")
            st.bar_chart(df_bench.set_index("System")["Attestation Rate (%)"])
    else:
        st.info("Run `python benchmarks/run_benchmarks.py` to generate live benchmark metrics.")

# -------------------------------------------------------------
# TAB 4: Compliance Audit Trail
# -------------------------------------------------------------
with tab_audit:
    st.subheader("🛡️ Enterprise Compliance Audit Trail")
    st.markdown("Immutable audit log records for regulatory verification (HIPAA / SOC 2 / GDPR).")

    records = engine.audit_logger.get_recent_records(limit=20)
    if records:
        audit_rows = []
        for r in reversed(records):
            audit_rows.append({
                "Execution ID": r.get("execution_id", "")[:8],
                "Timestamp": r.get("timestamp"),
                "Query": r.get("query"),
                "Iterations": r.get("iterations_run"),
                "Converged": "✅ Yes" if r.get("converged") else "⚠️ Pruned",
                "Final Hallucination Rate": f"{r.get('final_hallucination_rate', 0) * 100:.1f}%",
                "Citations Count": r.get("citations_count", 0),
                "Latency": f"{r.get('execution_time_seconds', 0):.2f}s",
            })
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)

        json_str = json.dumps(records, indent=2)
        st.download_button(
            label="📥 Download Full Audit Log (JSON)",
            data=json_str,
            file_name="veritas_compliance_audit_log.json",
            mime="application/json",
        )
    else:
        st.info("No audit records logged yet. Run a query in the Query tab to generate audit logs.")

# -------------------------------------------------------------
# TAB 5: Paper & Patent Drafts
# -------------------------------------------------------------
with tab_docs:
    st.subheader("📄 Publications & IP Deliverables (Phase 2)")
    doc_sel = st.radio("Select Document:", ["ACL Research Paper Draft", "Provisional Patent Application"], horizontal=True)

    if doc_sel == "ACL Research Paper Draft":
        paper_path = Path("paper/ACL_paper_draft.md")
        if paper_path.exists():
            st.markdown(paper_path.read_text(encoding="utf-8"))
    else:
        patent_path = Path("patent/provisional_patent_spec.md")
        if patent_path.exists():
            st.markdown(patent_path.read_text(encoding="utf-8"))
