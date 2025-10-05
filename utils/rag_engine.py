"""
RAG engine using direct Groq API calls
"""
from typing import List, Dict, Optional
import requests
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForLLMRun
from utils.logger import setup_logger
from utils.vector_store import VectorStoreManager
from config import LLM_MODEL, TEMPERATURE, MAX_TOKENS, TOP_K_RESULTS, GROQ_API_KEY

logger = setup_logger("rag_engine")

class GroqHTTP(LLM):
    """Direct Groq API via HTTP"""
    
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        """Call Groq API with proper error handling"""
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            # Log the error details
            if response.status_code != 200:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return f"API Error: {response.status_code} - {response.text}"
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return f"Request failed: {str(e)}"
        except (KeyError, IndexError) as e:
            logger.error(f"Invalid response format: {str(e)}")
            return f"Invalid response: {str(e)}"

class RAGEngine:
    """RAG-based question answering system"""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        try:
            logger.info("Initializing RAGEngine")
            
            if not GROQ_API_KEY or GROQ_API_KEY == "":
                raise ValueError("GROQ_API_KEY not configured in .env file")
            
            self.vector_store = vector_store_manager
            self.llm = GroqHTTP()
            
            prompt_template = """You are an expert M&A tax analyst. Use the following context to answer the question.

Context:
{context}

Question: {question}

Provide a detailed answer. If the information is not in the context, state "I cannot find this information in the provided document."

Answer:"""
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.vector_store.as_retriever(
                    search_kwargs={"k": TOP_K_RESULTS}
                ),
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "question"]
                    )
                },
                return_source_documents=True
            )
            
            logger.info("RAGEngine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAGEngine: {str(e)}")
            raise
    
    def query(self, question: str) -> Dict:
        """Query the RAG system"""
        try:
            logger.info(f"Processing query: '{question}'")
            
            if not GROQ_API_KEY:
                return {
                    "answer": "⚠️ Groq API key not configured. Set GROQ_API_KEY in .env file.",
                    "sources": []
                }
            
            result = self.qa_chain({"query": question})
            
            response = {
                "answer": result["result"],
                "sources": [
                    {
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ]
            }
            
            logger.info(f"Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": f"❌ Error: {str(e)}",
                "sources": []
            }