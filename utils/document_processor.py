"""
Document processing utilities for PDF parsing and text extraction
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from utils.logger import setup_logger
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = setup_logger("document_processor")

class DocumentProcessor:
    """Process and analyze PDF documents for tax risk assessment"""
    
    def __init__(self):
        """Initialize document processor"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        logger.info("DocumentProcessor initialized")
    
    def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text content from PDF file"""
        try:
            logger.info(f"Extracting text from PDF: {file_path.name}")
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
                
                full_text = "\n\n".join(text_content)
                logger.info(f"Successfully extracted {len(full_text)} characters from {len(pdf_reader.pages)} pages")
                return full_text
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def extract_metadata(self, file_path: Path) -> Dict[str, any]:
        """Extract metadata from PDF document"""
        try:
            logger.info(f"Extracting metadata from: {file_path.name}")
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                metadata = {
                    "filename": file_path.name,
                    "pages": len(pdf_reader.pages),
                    "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                }
                
                # Extract PDF metadata - convert all to strings
                if pdf_reader.metadata:
                    for key in ['creator', 'producer', 'creation_date', 'modification_date']:
                        value = getattr(pdf_reader.metadata, key, None)
                        if value is not None:
                            metadata[key] = str(value)
                
                logger.info(f"Metadata extracted: {metadata}")
                return metadata
                
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return {"filename": file_path.name}
    
    def analyze_document_content(self, text: str) -> Dict[str, any]:
        """Analyze document content for key information"""
        try:
            logger.info("Analyzing document content")
            
            analysis = {
                "total_characters": len(text),
                "total_words": len(text.split()),
                "estimated_reading_time_minutes": round(len(text.split()) / 200, 1),
                "contains_financial_data": bool(re.search(r'\$[\d,]+|\d+%|revenue|profit|loss|tax|liability', text, re.IGNORECASE))
            }
            
            tax_keywords = ['tax', 'audit', 'liability', 'deduction', 'irs', 'revenue', 'assessment', 'compliance', 'return', 'withholding', 'exemption']
            found_keywords = [kw for kw in tax_keywords if kw.lower() in text.lower()]
            analysis["tax_keywords_found"] = found_keywords
            analysis["tax_relevance_score"] = min(len(found_keywords) / len(tax_keywords), 1.0)
            
            risk_indicators = ['penalty', 'non-compliance', 'dispute', 'assessment', 'adjustment', 'deficiency', 'examination']
            found_risks = [ind for ind in risk_indicators if ind.lower() in text.lower()]
            analysis["risk_indicators"] = found_risks
            
            logger.info(f"Document analysis complete")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing document: {str(e)}")
            return {"error": str(e)}
    
    def split_into_chunks(self, text: str, metadata: Dict = None) -> List[Document]:
        """Split text into chunks for vector storage"""
        try:
            logger.info(f"Splitting text into chunks (size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP})")
            
            chunks = self.text_splitter.split_text(text)
            
            # Filter metadata to only include primitive types
            clean_metadata = {}
            if metadata:
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_metadata[k] = v
            
            documents = [
                Document(
                    page_content=chunk,
                    metadata={**clean_metadata, "chunk_index": i}
                )
                for i, chunk in enumerate(chunks)
            ]
            
            logger.info(f"Created {len(documents)} document chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error splitting document: {str(e)}")
            raise
    
    def process_uploaded_file(self, file_path: Path) -> Tuple[str, Dict, Dict, List[Document]]:
        """Complete processing pipeline for uploaded file"""
        try:
            logger.info(f"Starting complete processing for: {file_path.name}")
            
            text = self.extract_text_from_pdf(file_path)
            metadata = self.extract_metadata(file_path)
            analysis = self.analyze_document_content(text)
            chunks = self.split_into_chunks(text, metadata)
            
            logger.info(f"Processing complete for {file_path.name}")
            return text, metadata, analysis, chunks
            
        except Exception as e:
            logger.error(f"Error in processing pipeline: {str(e)}")
            raise