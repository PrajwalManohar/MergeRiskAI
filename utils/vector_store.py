"""
Vector database management using ChromaDB
"""
from typing import List, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from utils.logger import setup_logger
from config import VECTORDB_DIR, COLLECTION_NAME

logger = setup_logger("vector_store")

class ChromaDBEmbeddings(Embeddings):
    """Wrapper for ChromaDB default embeddings to work with Langchain"""
    
    def __init__(self):
        from chromadb.utils import embedding_functions
        self.ef = embedding_functions.DefaultEmbeddingFunction()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return self.ef(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.ef([text])[0]

class VectorStoreManager:
    """Manage vector database operations"""
    
    def __init__(self):
        """Initialize vector store with embeddings"""
        try:
            logger.info("Initializing VectorStoreManager")
            
            # Use custom wrapper for ChromaDB embeddings
            self.embeddings = ChromaDBEmbeddings()
            logger.info("Using ChromaDB default embeddings")
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(VECTORDB_DIR),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Initialize vector store
            self.vector_store = Chroma(
                client=self.client,
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(VECTORDB_DIR)
            )
            
            logger.info("VectorStoreManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing VectorStoreManager: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to vector store"""
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            ids = self.vector_store.add_documents(documents)
            logger.info(f"Successfully added {len(ids)} documents")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            self.client.delete_collection(COLLECTION_NAME)
            self.vector_store = Chroma(
                client=self.client,
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(VECTORDB_DIR)
            )
            logger.info("Collection cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            raise
    
    def get_collection_count(self) -> int:
        """Get number of documents in collection"""
        try:
            collection = self.client.get_collection(COLLECTION_NAME)
            count = collection.count()
            return count
        except Exception as e:
            logger.error(f"Error getting collection count: {str(e)}")
            return 0