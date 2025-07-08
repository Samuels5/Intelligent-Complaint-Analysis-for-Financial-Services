"""
Text chunking utilities for breaking long complaint narratives into smaller pieces.

This module implements various text splitting strategies optimized for
embedding generation and retrieval.
"""

from typing import List, Dict, Any
import re


class RecursiveCharacterTextSplitter:
    """
    A recursive character text splitter that breaks text into chunks
    while trying to preserve semantic boundaries.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, separators: List[str] = None):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators in order of preference
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ".", "!", "?", ";", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using recursive character splitting.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # Try each separator in order of preference
        for separator in self.separators:
            if separator in text:
                chunks = self._split_by_separator(text, separator)
                if chunks:
                    return self._add_overlap(chunks)
        
        # If no separator worked, split by character count
        return self._split_by_character_count(text)
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """Split text by a specific separator."""
        parts = text.split(separator)
        chunks = []
        current_chunk = ""
        
        for i, part in enumerate(parts):
            potential_chunk = current_chunk + part
            if i < len(parts) - 1:  # Add separator back except for last part
                potential_chunk += separator
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part
                if i < len(parts) - 1:
                    current_chunk += separator
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _split_by_character_count(self, text: str) -> List[str]:
        """Split text by character count as fallback."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                # Try to end at a word boundary
                while end > start and text[end] != ' ':
                    end -= 1
                if end == start:  # No space found, use original end
                    end = start + self.chunk_size
            else:
                end = len(text)
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = max(start + 1, end - self.chunk_overlap)
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # Get overlap from previous chunk
            if len(prev_chunk) > self.chunk_overlap:
                overlap = prev_chunk[-self.chunk_overlap:]
            else:
                overlap = prev_chunk
            
            # Combine overlap with current chunk
            overlapped_chunk = overlap + " " + current_chunk
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks


class ComplaintTextChunker:
    """
    Specialized text chunker for complaint narratives with metadata tracking.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the complaint text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def chunk_complaints(self, complaints_df) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Chunk all complaint narratives and create metadata.
        
        Args:
            complaints_df: DataFrame with complaint data
            
        Returns:
            Tuple of (chunks, metadata) lists
        """
        all_chunks = []
        chunk_metadata = []
        
        for idx, row in complaints_df.iterrows():
            narrative = row['Consumer_complaint_narrative']
            chunks = self.splitter.split_text(narrative)
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'complaint_id': row.get('Complaint ID', idx),
                    'chunk_id': f"{idx}_{chunk_idx}",
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'product': row['Product'],
                    'issue': row['Issue'],
                    'company': row.get('Company', 'Unknown'),
                    'date_received': row.get('Date received', ''),
                    'original_text_length': len(narrative),
                    'chunk_length': len(chunk)
                })
        
        return all_chunks, chunk_metadata
    
    def get_chunking_stats(self, chunks: List[str], metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the chunking process.
        
        Args:
            chunks: List of text chunks
            metadata: List of chunk metadata
            
        Returns:
            Dictionary with chunking statistics
        """
        chunk_lengths = [len(chunk) for chunk in chunks]
        complaints_count = len(set(meta['complaint_id'] for meta in metadata))
        
        stats = {
            'total_chunks': len(chunks),
            'total_complaints': complaints_count,
            'avg_chunks_per_complaint': len(chunks) / complaints_count,
            'chunk_length_stats': {
                'min': min(chunk_lengths),
                'max': max(chunk_lengths),
                'mean': sum(chunk_lengths) / len(chunk_lengths),
                'median': sorted(chunk_lengths)[len(chunk_lengths) // 2]
            },
            'product_distribution': {}
        }
        
        # Product distribution
        product_counts = {}
        for meta in metadata:
            product = meta['product']
            product_counts[product] = product_counts.get(product, 0) + 1
        
        stats['product_distribution'] = product_counts
        
        return stats


def optimize_chunk_size(sample_texts: List[str], embedding_model, target_sizes: List[int] = None) -> Dict[str, Any]:
    """
    Optimize chunk size by testing different configurations.
    
    Args:
        sample_texts: Sample texts to test chunking on
        embedding_model: Embedding model for testing
        target_sizes: List of chunk sizes to test
        
    Returns:
        Dictionary with optimization results
    """
    if target_sizes is None:
        target_sizes = [200, 300, 400, 500, 600, 800]
    
    results = {}
    
    for chunk_size in target_sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 5  # 20% overlap
        )
        
        all_chunks = []
        for text in sample_texts:
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
        
        # Test embedding generation (if model provided)
        try:
            if hasattr(embedding_model, 'encode'):
                sample_chunks = all_chunks[:10]  # Test with first 10 chunks
                embeddings = embedding_model.encode(sample_chunks)
                embedding_success = True
            else:
                embedding_success = False
        except Exception:
            embedding_success = False
        
        chunk_lengths = [len(chunk) for chunk in all_chunks]
        
        results[chunk_size] = {
            'total_chunks': len(all_chunks),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'chunk_length_std': (sum((x - sum(chunk_lengths) / len(chunk_lengths))**2 for x in chunk_lengths) / len(chunk_lengths))**0.5,
            'embedding_success': embedding_success,
            'efficiency_score': len(all_chunks) / sum(chunk_lengths) * 1000  # chunks per 1000 chars
        }
    
    return results


if __name__ == "__main__":
    # Example usage
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    sample_text = """
    I am writing to file a complaint about my credit card billing. 
    Last month I was charged a fee that I believe was incorrect. 
    When I called customer service, they were unhelpful and rude. 
    I have been a customer for over 5 years and this treatment is unacceptable.
    I would like this fee reversed and an apology for the poor service.
    """
    
    chunks = splitter.split_text(sample_text)
    print(f"Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}")
