"""
RAG pipeline core logic for complaint analysis chatbot.

This module implements the retriever, prompt template, generator, and the complete RAG pipeline
for analyzing customer complaints in financial services.

Classes:
    ComplaintRetriever: Semantic search and retrieval of relevant complaint chunks
    PromptTemplate: Template management for LLM prompts
    ComplaintGenerator: Text generation using transformer models
    ComplaintRAG: Complete RAG pipeline orchestration
    RAGPipelineManager: High-level management and initialization
"""

import os
import sys
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime
import json
import pickle
import numpy as np
import pandas as pd

# Core ML/AI libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import torch

# Transformers for text generation
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. Generator will use fallback mode.")

# Local imports
try:
    from .text_chunking import ComplaintTextChunker
except ImportError:
    # Handle direct execution
    sys.path.append(os.path.dirname(__file__))
    from text_chunking import ComplaintTextChunker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplaintRetriever:
    """
    Retriever class for finding relevant complaint chunks based on semantic similarity.
    
    This class handles the retrieval component of the RAG pipeline, using FAISS
    for efficient similarity search and providing various filtering options.
    """
    
    def __init__(self, faiss_index, chunks: List[str], metadata: List[Dict[str, Any]], 
                 embedding_model: SentenceTransformer):
        """
        Initialize the retriever with pre-built components.
        
        Args:
            faiss_index: FAISS index for similarity search
            chunks: List of text chunks
            metadata: List of metadata dictionaries for each chunk
            embedding_model: SentenceTransformer model for encoding queries
        """
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.metadata = metadata
        self.embedding_model = embedding_model
        
        # Validate inputs
        if len(chunks) != len(metadata):
            raise ValueError("Chunks and metadata must have the same length")
        
        logger.info(f"Retriever initialized with {len(chunks)} chunks")
    
    def retrieve(self, query: str, k: int = 5, filter_product: str = None, 
                 filter_company: str = None, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant chunks for a given query.
        
        Args:
            query: User question/query
            k: Number of chunks to retrieve
            filter_product: Optional product filter
            filter_company: Optional company filter  
            min_score: Minimum similarity score threshold
            
        Returns:
            List of retrieved chunks with metadata and scores
        """
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index (get more results for filtering)
            search_k = min(k * 3, self.faiss_index.ntotal)
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), search_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= len(self.chunks):  # Safety check
                    continue
                    
                if score < min_score:  # Score threshold
                    continue
                    
                chunk = self.chunks[idx]
                meta = self.metadata[idx]
                
                # Apply product filter if specified
                if filter_product and meta.get('product', '').lower() != filter_product.lower():
                    continue
                
                # Apply company filter if specified
                if filter_company and meta.get('company', '').lower() != filter_company.lower():
                    continue
                
                results.append({
                    'chunk': chunk,
                    'score': float(score),
                    'metadata': meta,
                    'chunk_index': idx
                })
                
                if len(results) >= k:  # Stop when we have enough results
                    break
            
            logger.debug(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return []
    
    def retrieve_with_context(self, query: str, k: int = 5, **kwargs) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve chunks and format them as context for LLM.
        
        Args:
            query: User question/query
            k: Number of chunks to retrieve
            **kwargs: Additional arguments passed to retrieve()
            
        Returns:
            Tuple of (formatted context string, list of retrieved chunks)
        """
        retrieved_chunks = self.retrieve(query, k, **kwargs)
        
        if not retrieved_chunks:
            return "No relevant information found.", []
        
        # Format context for LLM
        context_parts = []
        for i, result in enumerate(retrieved_chunks, 1):
            chunk = result['chunk']
            meta = result['metadata']
            
            context_part = f"""[Source {i}]
Product: {meta.get('product', 'Unknown')}
Issue: {meta.get('issue', 'Unknown')}
Company: {meta.get('company', 'Unknown')}
Content: {chunk}"""
            
            context_parts.append(context_part)
        
        context = "\n\n".join(context_parts)
        return context, retrieved_chunks
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval corpus."""
        products = [meta.get('product', 'Unknown') for meta in self.metadata]
        companies = [meta.get('company', 'Unknown') for meta in self.metadata]
        
        return {
            'total_chunks': len(self.chunks),
            'unique_products': len(set(products)),
            'unique_companies': len(set(companies)),
            'product_distribution': pd.Series(products).value_counts().to_dict(),
            'avg_chunk_length': np.mean([len(chunk) for chunk in self.chunks])
        }

class PromptTemplate:
    """
    Prompt template class for generating structured prompts for the LLM.
    
    This class manages different types of prompts for various use cases and
    stakeholder needs in the financial services complaint analysis system.
    """
    
    def __init__(self, template_type: str = "analyst"):
        """
        Initialize prompt template with specified type.
        
        Args:
            template_type: Type of template ("analyst", "manager", "support")
        """
        self.template_type = template_type
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, str]:
        """Load different prompt templates for different use cases."""
        templates = {
            "analyst": """You are a financial analyst assistant for CrediTrust Financial, a digital finance company. 
Your task is to analyze customer complaint data and provide helpful, accurate insights to internal stakeholders.

Instructions:
1. Use ONLY the provided complaint excerpts to formulate your answer
2. Be specific and cite the sources when possible
3. If the context doesn't contain enough information to answer the question, clearly state this
4. Focus on actionable insights for product managers and support teams
5. Maintain a professional, analytical tone
6. Summarize key themes and patterns when multiple complaints are relevant
7. Provide quantitative insights when possible (e.g., frequency of issues)""",

            "manager": """You are a senior product manager assistant for CrediTrust Financial. 
Your role is to provide strategic insights from customer complaint data to support decision-making.

Instructions:
1. Analyze the provided complaint data to extract strategic insights
2. Focus on business impact and prioritization
3. Identify trends that require immediate attention
4. Suggest actionable next steps for product teams
5. Highlight risks and opportunities
6. Use data to support recommendations
7. Keep responses concise but comprehensive""",

            "support": """You are a customer support specialist assistant for CrediTrust Financial.
Your role is to help support teams understand complaint patterns and improve service quality.

Instructions:
1. Focus on customer experience insights from the complaint data
2. Identify common pain points and resolution strategies
3. Suggest improvements to support processes
4. Highlight training opportunities for support staff
5. Provide practical, actionable recommendations
6. Use empathetic tone while maintaining professionalism
7. Consider customer satisfaction implications"""
        }
        
        return templates
    
    def create_prompt(self, context: str, question: str, template_type: str = None) -> str:
        """
        Create a complete prompt with system message, context, and question.
        
        Args:
            context: Retrieved complaint excerpts
            question: User's question
            template_type: Override default template type
            
        Returns:
            Formatted prompt string
        """
        template_key = template_type or self.template_type
        system_prompt = self.templates.get(template_key, self.templates["analyst"])
        
        prompt = f"""{system_prompt}

Context - Customer Complaint Excerpts:
{context}

Question: {question}

Analysis:"""
        
        return prompt
    
    def create_conversation_prompt(self, context: str, question: str, 
                                 conversation_history: List[Dict] = None,
                                 template_type: str = None) -> str:
        """
        Create a prompt that includes conversation history for follow-up questions.
        
        Args:
            context: Retrieved complaint excerpts
            question: Current question
            conversation_history: Previous Q&A pairs
            template_type: Override default template type
            
        Returns:
            Formatted prompt with conversation history
        """
        base_prompt = self.create_prompt(context, question, template_type)
        
        if conversation_history:
            history_text = "\n\nPrevious Conversation:\n"
            for turn in conversation_history[-3:]:  # Include last 3 turns
                history_text += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"
            
            # Insert history before the current question
            base_prompt = base_prompt.replace("Question:", f"{history_text}Current Question:")
        
        return base_prompt
    
    def create_evaluation_prompt(self, context: str, question: str, 
                               expected_topics: List[str] = None) -> str:
        """
        Create a prompt specifically for evaluation purposes.
        
        Args:
            context: Retrieved complaint excerpts
            question: Question to evaluate
            expected_topics: Topics that should be covered in the response
            
        Returns:
            Evaluation-focused prompt
        """
        eval_instructions = """
Additional Evaluation Instructions:
- Provide specific examples from the complaint data
- Quantify findings where possible
- Rate the confidence level of your analysis (High/Medium/Low)
- Identify any limitations in the available data"""
        
        if expected_topics:
            eval_instructions += f"\n- Address these key topics if relevant: {', '.join(expected_topics)}"
        
        base_prompt = self.create_prompt(context, question)
        return base_prompt.replace("Analysis:", f"Analysis:{eval_instructions}\n\nResponse:")
    
    def get_available_templates(self) -> List[str]:
        """Get list of available template types."""
        return list(self.templates.keys())

class ComplaintGenerator:
    def __init__(self, model_name: str = "distilgpt2"):
        try:
            from transformers import pipeline
            import torch
            self.generator = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1,
                return_full_text=False,
                pad_token_id=50256
            )
        except Exception as e:
            print(f"Error initializing generator: {e}")
            self.generator = None

    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        if self.generator is None:
            return "Error: Generator not properly initialized."
        try:
            result = self.generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=50256,
                eos_token_id=50256,
                num_return_sequences=1
            )
            response = result[0]['generated_text'].strip()
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

class ComplaintRAG:
    def __init__(self, retriever, generator, prompt_template):
        self.retriever = retriever
        self.generator = generator
        self.prompt_template = prompt_template
        self.conversation_history = []

    def answer_question(self, question: str, k: int = 5, include_sources: bool = True) -> Dict[str, Any]:
        context, retrieved_chunks = self.retriever.retrieve_with_context(question, k)
        prompt = self.prompt_template.create_prompt(context, question)
        if self.generator.generator is not None:
            answer = self.generator.generate_response(prompt, max_length=300)
        else:
            answer = self._create_fallback_response(question, retrieved_chunks)
        result = {
            'question': question,
            'answer': answer,
            'context': context,
            'sources': retrieved_chunks if include_sources else [],
            'num_sources': len(retrieved_chunks),
            'timestamp': datetime.now().isoformat()
        }
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })
        return result

    def _create_fallback_response(self, question: str, retrieved_chunks: List[Dict]) -> str:
        if not retrieved_chunks:
            return "I don't have enough information to answer your question based on the available complaint data."
        products = [chunk['metadata']['product'] for chunk in retrieved_chunks]
        issues = [chunk['metadata']['issue'] for chunk in retrieved_chunks]
        product_counts = {}
        issue_counts = {}
        for product in products:
            product_counts[product] = product_counts.get(product, 0) + 1
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        response = f"Based on {len(retrieved_chunks)} relevant complaint(s):\n\n"
        top_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        response += f"Main products involved: {', '.join([f'{p[0]} ({p[1]} complaints)' for p in top_products])}\n\n"
        top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        response += f"Primary issues: {', '.join([f'{i[0]} ({i[1]} complaints)' for i in top_issues])}\n\n"
        response += "Key complaint details:\n"
        for i, chunk in enumerate(retrieved_chunks[:3], 1):
            content = chunk['chunk'][:150] + "..." if len(chunk['chunk']) > 150 else chunk['chunk']
            response += f"{i}. {content}\n"
        return response

    def clear_history(self):
        self.conversation_history = []

    def get_conversation_summary(self) -> Dict[str, Any]:
        return {
            'total_questions': len(self.conversation_history),
            'questions': [entry['question'] for entry in self.conversation_history],
            'last_question_time': self.conversation_history[-1]['timestamp'] if self.conversation_history else None
        }

class RAGPipelineManager:
    """
    High-level manager for initializing and running the RAG pipeline.
    Handles loading vector store, embedding model, and all pipeline components.
    """
    def __init__(self, vector_store_path: str = "../vector_store",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 generator_model_name: str = "distilgpt2",
                 prompt_template_type: str = "analyst"):
        self.vector_store_path = vector_store_path
        self.embedding_model_name = embedding_model_name
        self.generator_model_name = generator_model_name
        self.prompt_template_type = prompt_template_type
        self.vector_manager = VectorStoreManager(vector_store_path)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.generator = ComplaintGenerator(model_name=generator_model_name)
        self.prompt_template = PromptTemplate(template_type=prompt_template_type)
        self.retriever = None
        self.rag_pipeline = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        files_status = self.vector_manager.verify_files()
        if not all(files_status.values()):
            raise FileNotFoundError(f"Missing vector store files: {files_status}")
        index, chunks, metadata, _ = self.vector_manager.load_all()
        self.retriever = ComplaintRetriever(index, chunks, metadata, self.embedding_model)
        self.rag_pipeline = ComplaintRAG(self.retriever, self.generator, self.prompt_template)
        logger.info("RAG pipeline initialized and ready.")

    def answer(self, question: str, k: int = 5, include_sources: bool = True) -> Dict[str, Any]:
        return self.rag_pipeline.answer_question(question, k=k, include_sources=include_sources)

    def get_stats(self) -> Dict[str, Any]:
        if self.retriever:
            return self.retriever.get_retrieval_stats()
        return {}

    def clear_conversation(self):
        if self.rag_pipeline:
            self.rag_pipeline.clear_history()

    def get_conversation_summary(self):
        if self.rag_pipeline:
            return self.rag_pipeline.get_conversation_summary()
        return {}

# Utility function for quick pipeline loading

def load_rag_pipeline(vector_store_path: str = "../vector_store",
                      embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                      generator_model_name: str = "distilgpt2",
                      prompt_template_type: str = "analyst") -> RAGPipelineManager:
    """
    Load and initialize the RAG pipeline manager for interactive use.
    """
    return RAGPipelineManager(
        vector_store_path=vector_store_path,
        embedding_model_name=embedding_model_name,
        generator_model_name=generator_model_name,
        prompt_template_type=prompt_template_type
    )

if __name__ == "__main__":
    # Example usage for testing
    try:
        rag_manager = load_rag_pipeline()
        print("RAG pipeline loaded successfully.")
        print("Pipeline stats:", rag_manager.get_stats())
        # Example question
        result = rag_manager.answer("What are the main issues with credit cards?", k=3)
        print("\nSample Answer:")
        print(result['answer'])
        print("\nSources:")
        for src in result['sources']:
            print(f"- {src['metadata']['product']}: {src['metadata']['issue']}")
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
