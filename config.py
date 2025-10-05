"""
Configuration settings for MergerRiskAI application
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTORDB_DIR = DATA_DIR / "vectordb"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, UPLOAD_DIR, VECTORDB_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Document Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_FILE_SIZE_MB = 100

# Vector Database Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "tax_documents"

# RAG Configuration - Groq Models
TOP_K_RESULTS = 5
LLM_MODEL = "llama-3.3-70b-versatile"  # Options: llama-3.1-70b-versatile, mixtral-8x7b-32768
TEMPERATURE = 0.3
MAX_TOKENS = 2000

# UI Configuration
APP_TITLE = "M&A Tax Risk Assessment Model"
APP_SUBTITLE = "Professional Due Diligence and Tax Exposure Analysis"