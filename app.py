"""
M&A Tax Risk Assessment - Streamlit Application
Main application file for document upload, analysis, and RAG-based Q&A using Groq
"""
import streamlit as st
from pathlib import Path
import time
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStoreManager
from utils.rag_engine import RAGEngine
from utils.logger import setup_logger
from config import (
    APP_TITLE, APP_SUBTITLE, UPLOAD_DIR, 
    MAX_FILE_SIZE_MB, GROQ_API_KEY, LLM_MODEL
)

# Initialize logger
logger = setup_logger("streamlit_app")

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F1F5F9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #60A5FA;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #ECFDF5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
    }
    .answer-box {
        background-color: #F9FAFB;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
        logger.info("DocumentProcessor initialized in session state")
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStoreManager()
        logger.info("VectorStoreManager initialized in session state")
    
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = RAGEngine(st.session_state.vector_store)
        logger.info("RAGEngine initialized in session state")
    
    if 'processed_document' not in st.session_state:
        st.session_state.processed_document = None
    
    if 'document_metadata' not in st.session_state:
        st.session_state.document_metadata = None
    
    if 'document_analysis' not in st.session_state:
        st.session_state.document_analysis = None
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

def display_header():
    """Display application header"""
    st.markdown(f'<div class="main-header">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{APP_SUBTITLE}</div>', unsafe_allow_html=True)

def display_document_metadata(metadata: dict, analysis: dict):
    """Display document metadata and analysis"""
    st.markdown("### 📊 Document Preview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Pages", metadata.get('pages', 'N/A'))
    
    with col2:
        st.metric("Word Count", f"{analysis.get('total_words', 0):,}")
    
    with col3:
        st.metric("Characters", f"{analysis.get('total_characters', 0):,}")
    
    with col4:
        reading_time = analysis.get('estimated_reading_time_minutes', 0)
        st.metric("Est. Reading Time", f"{reading_time} min")
    
    # Document Text Preview
    st.markdown("#### Document Content Preview")
    with st.expander("📄 View Document Text (First 2000 characters)", expanded=False):
        if st.session_state.processed_document:
            preview_text = st.session_state.processed_document[:2000]
            st.text_area("Content", preview_text, height=300, disabled=True, label_visibility="collapsed")

def display_tax_analysis(analysis: dict):
    """Display tax-specific analysis"""
    st.markdown("### 🔍 Tax Audit Outcomes & Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tax relevance
        relevance_score = analysis.get('tax_relevance_score', 0) * 100
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**Tax Relevance Score**")
        st.progress(relevance_score / 100)
        st.caption(f"{relevance_score:.1f}% tax-related content")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Keywords found
        keywords = analysis.get('tax_keywords_found', [])
        if keywords:
            st.markdown("**Tax Keywords Detected:**")
            st.write(", ".join(keywords[:10]))
    
    with col2:
        # Risk indicators
        risks = analysis.get('risk_indicators', [])
        if risks:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("**⚠️ Risk Indicators Found:**")
            for risk in risks:
                st.markdown(f"• {risk.title()}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("**✅ No Hard Audit Indicators Found**")
            st.markdown("No specific risk indicators detected in document")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Financial data detection
    if analysis.get('contains_financial_data'):
        st.info("💰 Document contains financial data and numerical metrics")

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Document Upload & Analysis")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Tax Documents",
            type=['pdf'],
            help=f"Supported formats: PDF (Max size: {MAX_FILE_SIZE_MB}MB)"
        )
        
        st.markdown("---")
        
        # API Key status
        st.markdown("### ⚙️ System Configuration")
        if GROQ_API_KEY:
            st.success(f"✅ Groq API Connected")
            st.caption(f"Model: {LLM_MODEL}")
        else:
            st.error("❌ Groq API Key Missing")
            st.caption("Add GROQ_API_KEY to .env file")
            st.link_button("Get API Key", "https://console.groq.com")
        
        st.markdown("---")
        
        # Collection info
        st.markdown("### 📚 Vector Database")
        doc_count = st.session_state.vector_store.get_collection_count()
        st.metric("Chunks Stored", doc_count)
        
        if st.button("🗑️ Clear Database", use_container_width=True):
            with st.spinner("Clearing database..."):
                st.session_state.vector_store.clear_collection()
                st.session_state.processed_document = None
                st.session_state.document_metadata = None
                st.session_state.document_analysis = None
                st.session_state.query_history = []
                st.success("Database cleared!")
                st.rerun()
    
    # Main content area
    if uploaded_file is not None:
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"❌ File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB}MB)")
            return
        
        # Save uploaded file
        upload_path = UPLOAD_DIR / uploaded_file.name
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"File uploaded: {uploaded_file.name} ({file_size_mb:.2f}MB)")
        
        # Process document
        if st.session_state.processed_document is None or \
           st.session_state.document_metadata.get('filename') != uploaded_file.name:
            
            with st.spinner("🔄 Analyzing document..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Extract text
                    status_text.text("Extracting text from PDF...")
                    progress_bar.progress(25)
                    text, metadata, analysis, chunks = st.session_state.document_processor.process_uploaded_file(upload_path)
                    
                    # Step 2: Analyze content
                    status_text.text("Analyzing content...")
                    progress_bar.progress(50)
                    time.sleep(0.5)
                    
                    # Step 3: Create embeddings
                    status_text.text("Creating vector embeddings...")
                    progress_bar.progress(75)
                    st.session_state.vector_store.add_documents(chunks)
                    
                    # Step 4: Complete
                    status_text.text("Processing complete!")
                    progress_bar.progress(100)
                    time.sleep(0.5)
                    
                    # Store in session state
                    st.session_state.processed_document = text
                    st.session_state.document_metadata = metadata
                    st.session_state.document_analysis = analysis
                    st.session_state.query_history = []
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    st.success(f"✅ Successfully processed: **{uploaded_file.name}**")
                    logger.info(f"Document processed successfully: {uploaded_file.name}")
                    
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")
                    logger.error(f"Document processing error: {str(e)}")
                    return
        
        # Display document information
        if st.session_state.document_metadata and st.session_state.document_analysis:
            display_document_metadata(
                st.session_state.document_metadata,
                st.session_state.document_analysis
            )
            
            display_tax_analysis(st.session_state.document_analysis)
        
        st.markdown("---")
        
        # Q&A Section
        st.markdown("### 💬 Document Q&A (RAG System)")
        st.caption("Ask questions about the uploaded tax document using Groq AI")
        
        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What are the main tax liabilities identified?",
            key="query_input"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
        
        if ask_button and query:
            with st.spinner("🤔 Analyzing document and generating answer..."):
                # Get answer from RAG
                result = st.session_state.rag_engine.query(query)
                
                # Add to history
                st.session_state.query_history.append({
                    "question": query,
                    "answer": result["answer"],
                    "sources": result.get("sources", [])
                })
                
                # Display answer
                st.markdown("#### Answer:")
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                
                # Display sources
                if result.get("sources"):
                    with st.expander(f"📚 View {len(result['sources'])} Source Documents"):
                        for i, source in enumerate(result['sources'], 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(source['content'])
                            if source.get('metadata'):
                                st.caption(f"Chunk: {source['metadata'].get('chunk_index', 'N/A')}")
                            st.markdown("---")
                
                logger.info(f"Query answered: {query[:50]}...")
        
        # Query history
        if st.session_state.query_history:
            with st.expander(f"📝 Query History ({len(st.session_state.query_history)} questions)"):
                for i, item in enumerate(reversed(st.session_state.query_history), 1):
                    st.markdown(f"**Q{i}: {item['question']}**")
                    st.caption(item['answer'][:200] + "..." if len(item['answer']) > 200 else item['answer'])
                    st.markdown("---")
        
        # Example queries
        with st.expander("💡 Example Questions"):
            st.markdown("""
            **Tax Liabilities & Assessments:**
            - What are the total tax liabilities identified in this document?
            - Are there any pending tax audits or examinations mentioned?
            - What is the company's effective tax rate?
            
            **Compliance & Risk:**
            - What tax compliance issues are highlighted?
            - Are there any IRS findings or adjustments?
            - What penalties or interest charges are mentioned?
            
            **Deductions & Credits:**
            - What deductions or exemptions are discussed?
            - Are there any tax credits claimed?
            
            **Jurisdictions:**
            - What tax jurisdictions are involved?
            - Are there any international tax considerations?
            """)
    
    else:
        # No document uploaded - show instructions
        st.info("👆 Please upload a tax document (PDF) using the sidebar to begin analysis")
        
        st.markdown("### 🎯 How It Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>1️⃣ Upload Document</h4>
                <p>Upload your tax-related PDF documents for analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>2️⃣ Automatic Analysis</h4>
                <p>AI extracts key information and identifies tax risks</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>3️⃣ Ask Questions</h4>
                <p>Use Groq-powered RAG to get specific insights</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🔧 Features")
        st.markdown("""
        - **Document Processing**: Automatic text extraction and parsing
        - **Metadata Analysis**: Page count, word count, reading time
        - **Tax Intelligence**: Keyword detection and risk assessment
        - **Vector Search**: Semantic search through document content  
        - **RAG Q&A (Groq)**: Fast AI-powered answers using Llama 3.1
        - **Source Citation**: View exact document sections used for answers
        - **Query History**: Track all questions and answers
        """)
        
        st.markdown("### 🚀 Powered by Groq")
        st.info(f"Using **{LLM_MODEL}** for ultra-fast inference")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")