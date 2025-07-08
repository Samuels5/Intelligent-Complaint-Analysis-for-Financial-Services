"""
Vector store setup and persistence utilities for complaint analysis.

This module provides functions to create, save, and load FAISS vector stores and associated metadata.
"""

import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Any

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Create a FAISS index for similarity search (cosine similarity).
    """
    dimension = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    return index

def save_faiss_index(index: faiss.IndexFlatIP, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)

def load_faiss_index(path: str) -> faiss.IndexFlatIP:
    return faiss.read_index(path)

def save_metadata(metadata: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(metadata, f)

def load_metadata(path: str) -> List[Dict[str, Any]]:
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_chunks(chunks: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(chunks, f)

def load_chunks(path: str) -> List[str]:
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_embeddings(embeddings: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeddings)

def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)

class VectorStoreManager:
    """
    Comprehensive manager for vector store operations.
    """
    
    def __init__(self, base_path: str = "../vector_store"):
        self.base_path = base_path
        self.index_path = os.path.join(base_path, "faiss_index.bin")
        self.chunks_path = os.path.join(base_path, "chunks.pkl")
        self.metadata_path = os.path.join(base_path, "metadata.pkl")
        self.embeddings_path = os.path.join(base_path, "embeddings.npy")
    
    def save_all(self, index: faiss.IndexFlatIP, chunks: List[str], 
                 metadata: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Save all vector store components."""
        save_faiss_index(index, self.index_path)
        save_chunks(chunks, self.chunks_path)
        save_metadata(metadata, self.metadata_path)
        save_embeddings(embeddings, self.embeddings_path)
        print(f"✅ All components saved to {self.base_path}")
    
    def load_all(self) -> tuple:
        """Load all vector store components."""
        index = load_faiss_index(self.index_path)
        chunks = load_chunks(self.chunks_path)
        metadata = load_metadata(self.metadata_path)
        embeddings = load_embeddings(self.embeddings_path)
        print(f"✅ All components loaded from {self.base_path}")
        return index, chunks, metadata, embeddings
    
    def verify_files(self) -> Dict[str, bool]:
        """Verify that all required files exist."""
        files = {
            'index': os.path.exists(self.index_path),
            'chunks': os.path.exists(self.chunks_path),
            'metadata': os.path.exists(self.metadata_path),
            'embeddings': os.path.exists(self.embeddings_path)
        }
        return files
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the vector store."""
        if not all(self.verify_files().values()):
            return {"error": "Some vector store files are missing"}
        
        try:
            index, chunks, metadata, embeddings = self.load_all()
            
            info = {
                'total_vectors': index.ntotal,
                'vector_dimension': index.d,
                'total_chunks': len(chunks),
                'total_metadata': len(metadata),
                'embeddings_shape': embeddings.shape,
                'products': list(set(meta['product'] for meta in metadata)),
                'chunk_length_stats': {
                    'min': min(len(chunk) for chunk in chunks),
                    'max': max(len(chunk) for chunk in chunks),
                    'avg': sum(len(chunk) for chunk in chunks) / len(chunks)
                }
            }
            return info
        except Exception as e:
            return {"error": f"Failed to load vector store: {str(e)}"}


def create_vector_store_from_chunks(chunks: List[str], metadata: List[Dict[str, Any]], 
                                   embedding_model, base_path: str = "../vector_store") -> VectorStoreManager:
    """
    Create a complete vector store from chunks and metadata.
    
    Args:
        chunks: List of text chunks
        metadata: List of metadata dictionaries
        embedding_model: Embedding model for generating vectors
        base_path: Base path to save vector store
        
    Returns:
        VectorStoreManager instance
    """
    print("Creating vector store from chunks...")
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    
    # Create FAISS index
    print("Creating FAISS index...")
    index = create_faiss_index(embeddings.copy())
    
    # Save everything
    manager = VectorStoreManager(base_path)
    manager.save_all(index, chunks, metadata, embeddings)
    
    return manager


if __name__ == "__main__":
    # Example usage
    manager = VectorStoreManager()
    files_status = manager.verify_files()
    print("Vector store files status:", files_status)
    
    if all(files_status.values()):
        info = manager.get_info()
        print("Vector store info:", info)
