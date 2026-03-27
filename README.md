# 📊 MergeRiskAI

**AI-powered M&A tax risk assessment tool for due diligence professionals.**

MergeRiskAI automates the analysis of tax documents during mergers and acquisitions. Upload a tax filing or due diligence document, and the system extracts key tax risks, liabilities, audit flags, and investment implications — all powered by a RAG pipeline with Groq-hosted LLaMA 3.3 70B.

---

## Problem

Tax due diligence in M&A deals is manual, slow, and error-prone. Analysts spend hours reading through dense PDFs to identify tax exposures, contingencies, and compliance risks. MergeRiskAI reduces that process from hours to minutes.

---

## How it works

```
PDF Upload → Text Extraction → Chunking → Embeddings → ChromaDB → RAG Query → LLM Analysis → Report
```

1. **Document ingestion** — Upload PDF tax documents through the Streamlit UI. Text is extracted via PyPDF2 and split into chunks (1000 tokens, 200 overlap) using LangChain text splitters.

2. **Vector indexing** — Chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in a ChromaDB collection for semantic retrieval.

3. **RAG-powered analysis** — The system runs a structured set of due diligence queries against the indexed document, retrieving the top 5 most relevant chunks per query and passing them to LLaMA 3.3 70B via Groq for analysis.

4. **Risk assessment** — Results are organized into five sections: executive summary, key tax metrics, critical risk findings, investment analysis, and tax contingencies.

5. **Interactive Q&A** — Ask follow-up questions about the document in natural language, with source attribution from the original text.

---

## Features

- **Tax relevance scoring** — Automatically scores documents on tax content density and flags risk indicators
- **Structured due diligence report** — Executive summary, risk findings, investment analysis, and contingencies
- **Interactive document Q&A** — Ask natural language questions with cited source passages
- **Downloadable reports** — Export the full analysis as a Markdown report
- **Real-time processing pipeline** — Progress tracking for document ingestion and analysis

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | LLaMA 3.3 70B via Groq API |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| Orchestration | LangChain |
| PDF Processing | PyPDF2 |
| Language | Python |

---

## Project structure

```
MergeRiskAI/
├── app.py                  # Main Streamlit application
├── config.py               # Configuration and environment settings
├── requirements.txt        # Python dependencies
├── utils/
│   ├── document_processor.py   # PDF extraction and chunking
│   ├── vector_store.py         # ChromaDB vector store management
│   ├── rag_engine.py           # RAG query pipeline
│   ├── tax_analyzer.py         # Structured tax analysis engine
│   └── logger.py               # Logging configuration
├── data/
│   ├── uploads/            # Uploaded documents
│   └── vectordb/           # ChromaDB persistence
└── logs/                   # Application logs
```

---

## Getting started

### Prerequisites

- Python 3.9+
- [Groq API key](https://console.groq.com/) (free tier available)

### Installation

```bash
# Clone the repository
git clone https://github.com/PrajwalManohar/MergeRiskAI.git
cd MergeRiskAI

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Run

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Usage

1. Upload a tax document (PDF, up to 100MB) via the sidebar
2. Review the **Tax Audit Outcomes** — relevance score, risk indicators, and keyword analysis
3. Click **Generate Comprehensive Analysis** for the full due diligence report
4. Toggle report sections (Executive Summary, Tax Metrics, Risk Findings, Investment, Contingencies) from the sidebar
5. Use the **Document Q&A** section to ask follow-up questions
6. Download the report as Markdown

---

## Example questions

- *What are the main tax liabilities disclosed?*
- *What is the effective tax rate?*
- *Are there any pending audit findings or disputes?*
- *What jurisdictions are involved?*
- *What are the key tax contingencies?*

---
