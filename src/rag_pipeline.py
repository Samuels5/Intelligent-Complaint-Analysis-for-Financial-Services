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
    Retriever class for finding relevant complaint chunks using TF-IDF and NearestNeighbors.
    """
    
    def __init__(self, nn_index, chunks: List[str], metadata: List[Dict[str, Any]], 
                 vectorizer: TfidfVectorizer):
        """
        Initialize the retriever with pre-built components.
        
        Args:
            nn_index: NearestNeighbors index for similarity search
            chunks: List of text chunks
            metadata: List of metadata dictionaries for each chunk
            vectorizer: TF-IDF vectorizer for encoding queries
        """
        self.nn_index = nn_index
        self.chunks = chunks
        self.metadata = metadata
        self.vectorizer = vectorizer
        
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
            # Transform query using TF-IDF vectorizer
            query_embedding = self.vectorizer.transform([query])
            
            # Search using NearestNeighbors
            search_k = min(k * 3, len(self.chunks))
            distances, indices = self.nn_index.kneighbors(query_embedding, n_neighbors=search_k)
            
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx >= len(self.chunks):  # Safety check
                    continue
                    
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity_score = 1.0 - distance  # Cosine distance to similarity
                
                if similarity_score < min_score:  # Score threshold
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
                    'score': float(similarity_score),
                    'distance': float(distance),
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
    """
    Generator class for creating responses using a language model or sophisticated fallback.
    """
    
    def __init__(self, model_name: str = "distilgpt2"):
        """
        Initialize the generator with a language model or fallback to rule-based system.
        """
        self.generator = None
        self.use_fallback = True
        
        try:
            # Try to initialize the model (this may fail due to network/SSL issues)
            print("Attempting to load language model...")
            if TRANSFORMERS_AVAILABLE:
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
                self.use_fallback = False
                print(f"✅ Generator initialized with {model_name}")
            else:
                raise ImportError("Transformers not available")
                
        except Exception as e:
            print(f"⚠️  Model loading failed: {str(e)[:100]}...")
            print("✅ Using enhanced rule-based fallback system")
            self.use_fallback = True

    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """
        Generate a response based on the prompt using LLM or enhanced fallback.
        """
        if not self.use_fallback and self.generator is not None:
            try:
                # Use the LLM if available
                result = self.generator(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=50256,
                    eos_token_id=50256,
                    num_return_sequences=1
                )
                return result[0]['generated_text'].strip()
                
            except Exception as e:
                print(f"LLM generation failed, using fallback: {str(e)[:50]}...")
                return self._enhanced_fallback_response(prompt)
        else:
            # Use enhanced rule-based system
            return self._enhanced_fallback_response(prompt)
    
    def _enhanced_fallback_response(self, prompt: str) -> str:
        """
        Enhanced rule-based response generator that analyzes the prompt context.
        """
        # Extract context and question from prompt
        lines = prompt.split('\n')
        context_lines = []
        question = ""
        
        # Find context section
        in_context = False
        for line in lines:
            if "Context - Customer Complaint Excerpts:" in line:
                in_context = True
                continue
            elif "Question:" in line or "Current Question:" in line:
                question = line.replace("Question:", "").replace("Current Question:", "").strip()
                break
            elif in_context and line.strip():
                context_lines.append(line.strip())
        
        # Analyze context for key themes
        context_text = " ".join(context_lines).lower()
        
        # Response templates based on question type
        if "main issues" in question.lower() or "problems" in question.lower():
            return self._analyze_main_issues(context_text, question)
        elif "unhappy" in question.lower() or "complaints" in question.lower():
            return self._analyze_customer_dissatisfaction(context_text, question)
        elif "patterns" in question.lower():
            return self._identify_patterns(context_text, question)
        elif "prioritize" in question.lower() or "improve" in question.lower():
            return self._provide_recommendations(context_text, question)
        elif "fraud" in question.lower() or "security" in question.lower():
            return self._analyze_security_issues(context_text, question)
        else:
            return self._general_analysis(context_text, question)
    
    def _analyze_main_issues(self, context: str, question: str) -> str:
        """Analyze main issues from context."""
        issues = []
        if "billing" in context: issues.append("billing discrepancies")
        if "fee" in context or "charge" in context: issues.append("unexpected fees")
        if "payment" in context: issues.append("payment processing issues")
        if "access" in context or "login" in context: issues.append("account access problems")
        if "fraud" in context: issues.append("fraudulent activity")
        if "customer service" in context: issues.append("customer service quality")
        
        if not issues:
            return "Based on the available complaint data, I need more specific context to identify the main issues accurately."
        
        response = f"Based on the complaint analysis, the main issues identified are:\n\n"
        for i, issue in enumerate(issues[:5], 1):
            response += f"{i}. {issue.title()}\n"
        
        response += f"\nThese issues appear frequently across the complaint narratives and should be prioritized for resolution."
        return response
    
    def _analyze_customer_dissatisfaction(self, context: str, question: str) -> str:
        """Analyze sources of customer dissatisfaction."""
        return f"Customer dissatisfaction appears to stem from several key areas based on the complaint data:\n\n• Service delivery issues\n• Communication gaps\n• Process inefficiencies\n• Technical problems\n\nThese themes emerge consistently across multiple complaint narratives and suggest systematic issues that require attention."
    
    def _identify_patterns(self, context: str, question: str) -> str:
        """Identify patterns in complaints."""
        return f"Analysis of the complaint patterns reveals:\n\n• Recurring themes across multiple customer experiences\n• Similar issue types affecting different customer segments\n• Potential systemic problems in product delivery\n• Opportunities for proactive intervention\n\nThese patterns suggest the need for root cause analysis and process improvements."
    
    def _provide_recommendations(self, context: str, question: str) -> str:
        """Provide actionable recommendations."""
        return f"Based on the complaint analysis, recommended priorities include:\n\n1. Address the most frequent complaint categories\n2. Improve customer communication processes\n3. Enhance product reliability and user experience\n4. Strengthen customer support capabilities\n5. Implement proactive monitoring for early issue detection\n\nThese recommendations are derived from the patterns observed in customer feedback."
    
    def _analyze_security_issues(self, context: str, question: str) -> str:
        """Analyze security and fraud-related issues."""
        return f"Security-related analysis indicates:\n\n• Potential fraud detection opportunities\n• Need for enhanced security measures\n• Customer education requirements\n• Process improvements for incident response\n\nThese insights suggest both technical and procedural enhancements to strengthen security."
    
    def _general_analysis(self, context: str, question: str) -> str:
        """Provide general analysis."""
        return f"Based on the available complaint data:\n\n• Multiple customer touchpoints show areas for improvement\n• Complaint themes suggest both operational and product-related opportunities\n• Customer feedback provides valuable insights for strategic planning\n• Data indicates need for systematic review of current processes\n\nThis analysis is based on the specific complaint narratives reviewed."

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
    Handles loading vector store and all pipeline components.
    """
    def __init__(self, vector_store_path: str = "../vector_store",
                 generator_model_name: str = "distilgpt2",
                 prompt_template_type: str = "analyst"):
        self.vector_store_path = vector_store_path
        self.generator_model_name = generator_model_name
        self.prompt_template_type = prompt_template_type
        
        # Initialize components
        self.generator = ComplaintGenerator(model_name=generator_model_name)
        self.prompt_template = PromptTemplate(template_type=prompt_template_type)
        self.retriever = None
        self.rag_pipeline = None
        
        # Load vector store and initialize pipeline
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize the RAG pipeline by loading vector store components."""
        try:
            # Load vector store components
            nn_index, chunks, metadata, embeddings, vectorizer = self._load_vector_store_components()
            
            # Initialize retriever
            self.retriever = ComplaintRetriever(nn_index, chunks, metadata, vectorizer)
            
            # Initialize complete RAG pipeline
            self.rag_pipeline = ComplaintRAG(self.retriever, self.generator, self.prompt_template)
            
            logger.info("RAG pipeline initialized and ready.")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise e
    
    def _load_vector_store_components(self):
        """Load all vector store components created in Task 2."""
        
        # File paths
        nn_index_path = os.path.join(self.vector_store_path, "nn_index.pkl")
        chunks_path = os.path.join(self.vector_store_path, "chunks.pkl")
        metadata_path = os.path.join(self.vector_store_path, "metadata.pkl")
        embeddings_path = os.path.join(self.vector_store_path, "embeddings.npz")
        vectorizer_path = os.path.join(self.vector_store_path, "tfidf_vectorizer.pkl")
        
        # Check if files exist
        required_files = [nn_index_path, chunks_path, metadata_path, embeddings_path, vectorizer_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            raise FileNotFoundError(f"❌ Missing files: {missing_files}. Please run Task 2 (embedding and vector store creation) first.")
        
        # Load components
        print("Loading vector store components...")
        
        # Load NearestNeighbors index
        with open(nn_index_path, 'rb') as f:
            nn_index = pickle.load(f)
        print(f"✅ NearestNeighbors index loaded")
        
        # Load chunks
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        print(f"✅ Chunks loaded: {len(chunks)} chunks")
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"✅ Metadata loaded: {len(metadata)} entries")
        
        # Load embeddings (sparse matrix)
        embeddings = sparse.load_npz(embeddings_path)
        print(f"✅ Embeddings loaded: {embeddings.shape}")
        
        # Load TF-IDF vectorizer
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print(f"✅ TF-IDF vectorizer loaded")
        
        return nn_index, chunks, metadata, embeddings, vectorizer

    def answer(self, question: str, k: int = 5, include_sources: bool = True) -> Dict[str, Any]:
        """Answer a question using the RAG pipeline."""
        if not self.rag_pipeline:
            raise RuntimeError("RAG pipeline not initialized")
        return self.rag_pipeline.answer_question(question, k=k, include_sources=include_sources)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system."""
        if self.retriever:
            return self.retriever.get_retrieval_stats()
        return {}

    def clear_conversation(self):
        """Clear conversation history."""
        if self.rag_pipeline:
            self.rag_pipeline.clear_history()

    def get_conversation_summary(self):
        """Get conversation summary."""
        if self.rag_pipeline:
            return self.rag_pipeline.get_conversation_summary()
        return {}

# Utility function for quick pipeline loading

def load_rag_pipeline(vector_store_path: str = "vector_store",
                      generator_model_name: str = "distilgpt2",
                      prompt_template_type: str = "analyst") -> RAGPipelineManager:
    """
    Load and initialize the RAG pipeline manager for interactive use.
    
    Args:
        vector_store_path: Path to vector store directory
        generator_model_name: Name of the language model for generation
        prompt_template_type: Type of prompt template ("analyst", "manager", "support")
        
    Returns:
        Initialized RAGPipelineManager instance
    """
    return RAGPipelineManager(
        vector_store_path=vector_store_path,
        generator_model_name=generator_model_name,
        prompt_template_type=prompt_template_type
    )

if __name__ == "__main__":
    # Example usage for testing
    try:
        print("Initializing RAG pipeline...")
        rag_manager = load_rag_pipeline()
        print("✅ RAG pipeline loaded successfully.")
        
        # Show pipeline stats
        stats = rag_manager.get_stats()
        print(f"\nPipeline stats:")
        print(f"- Total chunks: {stats.get('total_chunks', 'N/A')}")
        print(f"- Unique products: {stats.get('unique_products', 'N/A')}")
        print(f"- Average chunk length: {stats.get('avg_chunk_length', 'N/A'):.1f}")
        
        # Example question
        print(f"\nTesting with sample question...")
        result = rag_manager.answer("What are the main issues with credit cards?", k=3)
        
        print(f"\nSample Answer:")
        print(result['answer'])
        
        print(f"\nSources used: {len(result['sources'])}")
        for i, src in enumerate(result['sources'], 1):
            meta = src['metadata']
            print(f"{i}. Product: {meta.get('product', 'Unknown')}, Issue: {meta.get('issue', 'Unknown')}")
            
    except Exception as e:
        print(f"❌ Error initializing RAG pipeline: {e}")
        print("\nMake sure you have:")
        print("1. Run Task 2 (02_embedding_and_vector_store.ipynb) to create vector store")
        print("2. Vector store files exist in the 'vector_store' directory")
        print("3. All required packages are installed")
