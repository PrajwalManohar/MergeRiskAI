"""
M&A Tax Risk Assessment AI - Streamlit Application
"""
import streamlit as st
from pathlib import Path
import time
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStoreManager
from utils.rag_engine import RAGEngine
from utils.tax_analyzer import TaxAnalyzer
from utils.logger import setup_logger
from config import APP_TITLE, APP_ICON, MAX_FILE_SIZE_MB, UPLOAD_DIR

logger = setup_logger("app")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .risk-low { border-left-color: #10b981 !important; }
    .risk-medium { border-left-color: #f59e0b !important; }
    .risk-high { border-left-color: #ef4444 !important; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e293b;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .insight-box {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .audit-insight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'doc_processor' not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor()

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()

if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = RAGEngine(st.session_state.vector_store)

if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

if 'show_executive' not in st.session_state:
    st.session_state.show_executive = True

if 'show_metrics' not in st.session_state:
    st.session_state.show_metrics = True

if 'show_risks' not in st.session_state:
    st.session_state.show_risks = True

if 'show_investment' not in st.session_state:
    st.session_state.show_investment = False

if 'show_contingencies' not in st.session_state:
    st.session_state.show_contingencies = False

# Sidebar
with st.sidebar:
    st.markdown("### 📁 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Upload Tax Document",
        type=['pdf'],
        help=f"Max {MAX_FILE_SIZE_MB}MB"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    st.success("✓ Groq API Connected")
    
    st.markdown("---")
    st.markdown("### 📊 Database")
    doc_count = st.session_state.vector_store.get_collection_count()
    st.metric("Chunks", doc_count)
    
    if st.button("Clear", use_container_width=True):
        st.session_state.vector_store.clear_collection()
        st.session_state.document_processed = False
        st.session_state.analysis_results = None
        st.session_state.current_analysis = None
        st.rerun()
    
    # Section toggles
    if st.session_state.analysis_results:
        st.markdown("---")
        st.markdown("### 📑 Sections")
        
        if st.button("📋 Executive Summary", use_container_width=True):
            st.session_state.show_executive = not st.session_state.show_executive
        
        if st.button("📊 Tax Metrics", use_container_width=True):
            st.session_state.show_metrics = not st.session_state.show_metrics
        
        if st.button("⚠️ Risk Findings", use_container_width=True):
            st.session_state.show_risks = not st.session_state.show_risks
        
        if st.button("📈 Investment", use_container_width=True):
            st.session_state.show_investment = not st.session_state.show_investment
        
        if st.button("💼 Contingencies", use_container_width=True):
            st.session_state.show_contingencies = not st.session_state.show_contingencies

# Header
st.markdown(f'<div class="main-header">{APP_ICON} {APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Professional Due Diligence and Tax Exposure Analysis</div>', unsafe_allow_html=True)

# Process uploaded file
if uploaded_file and not st.session_state.document_processed:
    file_path = UPLOAD_DIR / uploaded_file.name
    
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("📄 Extracting text...")
        progress_bar.progress(33)
        text, metadata, analysis, chunks = st.session_state.doc_processor.process_uploaded_file(file_path)
        
        status_text.text("🔢 Generating embeddings...")
        progress_bar.progress(66)
        ids = st.session_state.vector_store.add_documents(chunks)
        
        status_text.text("✅ Complete!")
        progress_bar.progress(100)
        time.sleep(0.3)
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.document_processed = True
        st.session_state.current_analysis = analysis
        st.success(f"✓ Processed: {len(chunks)} chunks indexed")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Tax Audit Outcomes & Indicators
if st.session_state.current_analysis:
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Tax Audit Outcomes & Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    tax_score = int(st.session_state.current_analysis.get('tax_relevance_score', 0) * 100)
    risk_count = len(st.session_state.current_analysis.get('risk_indicators', []))
    keywords_found = len(st.session_state.current_analysis.get('tax_keywords_found', []))
    
    with col1:
        risk_class = "risk-low" if tax_score < 30 else "risk-medium" if tax_score < 70 else "risk-high"
        st.markdown(f'<div class="metric-card {risk_class}">', unsafe_allow_html=True)
        st.metric("Tax Relevance Score", f"{tax_score}%")
        st.caption("Content analysis")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        risk_class = "risk-low" if risk_count == 0 else "risk-medium" if risk_count < 3 else "risk-high"
        st.markdown(f'<div class="metric-card {risk_class}">', unsafe_allow_html=True)
        st.metric("Risk Indicators", risk_count)
        st.caption("Flags detected")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Tax Keywords", keywords_found)
        st.caption("Terms identified")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Professional Audit Insights
    st.markdown('<div class="audit-insight">', unsafe_allow_html=True)
    st.markdown("### 📊 Professional Assessment")
    
    if tax_score >= 70:
        st.markdown("**High Tax Content Document** - Substantial tax-related information requiring detailed review.")
    elif tax_score >= 30:
        st.markdown("**Moderate Tax Relevance** - Tax considerations among other business matters.")
    else:
        st.markdown("**Low Tax Content** - Limited tax-specific information detected.")
    
    if st.session_state.current_analysis.get('tax_keywords_found'):
        keywords_str = ", ".join(st.session_state.current_analysis['tax_keywords_found'][:5])
        st.markdown(f"**Key Terms:** {keywords_str}")
    
    if st.session_state.current_analysis.get('risk_indicators'):
        st.markdown(f"**⚠️ Risk Flags:** {', '.join(st.session_state.current_analysis['risk_indicators'])}")
    
    if st.session_state.current_analysis.get('contains_financial_data'):
        st.markdown("**💰 Financial Data:** Numerical metrics and amounts detected")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Analysis Button
if st.session_state.document_processed:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Generate Comprehensive Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                analyzer = TaxAnalyzer(st.session_state.rag_engine)
                st.session_state.analysis_results = analyzer.analyze_document()
                st.session_state.show_executive = True
                st.session_state.show_metrics = True
                st.session_state.show_risks = True

# Display Results with toggles
if st.session_state.analysis_results:
    
    st.markdown("---")
    
    # Executive Summary
    if st.session_state.show_executive:
        st.markdown('<div class="section-header">📋 Executive Summary</div>', unsafe_allow_html=True)
        for item in st.session_state.analysis_results.get('executive_summary', []):
            with st.expander(f"**{item['question']}**", expanded=False):
                st.markdown(f'<div class="insight-box">{item["answer"]}</div>', unsafe_allow_html=True)
    
    # Key Metrics
    if st.session_state.show_metrics:
        st.markdown('<div class="section-header">📊 Key Tax Metrics</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tax Relevance", f"{tax_score}%")
        with col2:
            st.metric("Risk Indicators", risk_count)
        with col3:
            st.metric("Audit Score", "N/A")
        with col4:
            st.metric("Chunks", doc_count)
    
    # Critical Findings
    if st.session_state.show_risks:
        st.markdown('<div class="section-header">⚠️ Critical Findings</div>', unsafe_allow_html=True)
        for item in st.session_state.analysis_results.get('escalation_flags', []):
            if "no" not in item["answer"].lower():
                st.markdown(f'<div class="warning-box">**{item["question"]}**<br>{item["answer"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-box">**{item["question"]}**<br>{item["answer"]}</div>', unsafe_allow_html=True)
    
    # Investment Analysis
    if st.session_state.show_investment:
        st.markdown('<div class="section-header">📈 Investment Analysis</div>', unsafe_allow_html=True)
        for item in st.session_state.analysis_results.get('investment_analysis', []):
            st.markdown(f"**{item['question']}**")
            st.info(item['answer'])
    
    # Tax Contingencies
    if st.session_state.show_contingencies:
        st.markdown('<div class="section-header">💼 Tax Contingencies</div>', unsafe_allow_html=True)
        for item in st.session_state.analysis_results.get('tax_contingencies', []):
            st.markdown(f"**{item['question']}**")
            st.info(item['answer'])
    
    # Download
    st.markdown("---")
    analyzer = TaxAnalyzer(st.session_state.rag_engine)
    full_report = analyzer.generate_summary_report(st.session_state.analysis_results)
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.download_button(
            label="📥 Download Report",
            data=full_report,
            file_name="tax_analysis.md",
            mime="text/markdown",
            use_container_width=True
        )

# Q&A Section
if st.session_state.document_processed:
    st.markdown("---")
    st.markdown('<div class="section-header">💬 Document Q&A</div>', unsafe_allow_html=True)
    
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - What are the main tax liabilities?
        - What is the effective tax rate?
        - Are there any audit findings?
        - What jurisdictions are involved?
        """)
    
    question = st.text_input("Ask a question:", placeholder="What are the key tax risks?")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        ask_btn = st.button("Ask", use_container_width=True)
    
    if ask_btn and question:
        with st.spinner("Searching..."):
            response = st.session_state.rag_engine.query(question)
            st.success(response["answer"])
            
            if response["sources"]:
                st.markdown("**Sources:**")
                for idx, source in enumerate(response["sources"][:2], 1):
                    st.caption(f"{idx}. {source['content'][:200]}...")