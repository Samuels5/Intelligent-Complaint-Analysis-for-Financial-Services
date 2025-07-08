# Intelligent Complaint Analysis for Financial Services

## Overview

This project implements a Retrieval-Augmented Generation (RAG) powered chatbot for CrediTrust Financial, enabling internal teams to turn unstructured customer complaint data into actionable insights. The system leverages semantic search, vector databases, and large language models to answer natural language questions about customer pain points across multiple financial products.

## Features

- **Exploratory Data Analysis & Preprocessing**: Clean and filter CFPB complaint data for target products.
- **Text Chunking & Embedding**: Break long narratives into chunks and generate embeddings using sentence-transformers.
- **Vector Store Indexing**: Store embeddings and metadata in a FAISS vector database for efficient semantic search.
- **RAG Core Logic**: Retrieve relevant complaint excerpts and generate evidence-backed answers using an LLM.
- **Interactive Chat Interface**: User-friendly Gradio/Streamlit app for non-technical users, with source traceability.

## Project Structure

```
├── app.py                        # Interactive web app (Gradio/Streamlit)
├── requirements.txt              # Python dependencies
├── data/                         # Raw and processed datasets
│   └── filtered_complaints.csv   # Cleaned dataset for RAG
├── notebooks/                    # Jupyter notebooks for each task
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_embedding_and_vector_store.ipynb
│   └── 03_rag_core_logic.ipynb
├── src/                         # Source code modules
│   ├── data_preprocessing.py
│   ├── text_chunking.py
│   ├── vector_store_setup.py
│   └── rag_pipeline.py
├── vector_store/                 # Persisted FAISS index, embeddings, metadata
└── README.md
```

## Setup & Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run EDA & Preprocessing**:
   - See `notebooks/01_eda_and_preprocessing.ipynb` to clean and filter the data.
3. **Chunk, Embed, and Index**:
   - See `notebooks/02_embedding_and_vector_store.ipynb` to create the vector store.
4. **RAG Core Logic & Evaluation**:
   - See `notebooks/03_rag_core_logic.ipynb` for pipeline and evaluation.
5. **Launch the Chatbot App**:
   - Run `app.py` to start the interactive interface.

## Deliverables

- Cleaned dataset: `data/filtered_complaints.csv`
- Vector store: `vector_store/`
- Modular source code: `src/`
- Jupyter notebooks for each task
- Interactive app: `app.py`
- Evaluation results and analysis

## How it Works

- **Ask a question** (e.g., "Why are people unhappy with BNPL?")
- The system retrieves the most relevant complaint excerpts using semantic search.
- An LLM generates a concise, evidence-backed answer, citing sources.
- The app displays both the answer and the supporting complaint excerpts.

