# Intelligent Complaint Analysis for Financial Services

## Overview

This project implements an end-to-end Retrieval-Augmented Generation (RAG) system for a fictional financial institution, CrediTrust Financial. It enables internal teams, such as product managers and compliance officers, to transform unstructured customer complaint data into actionable insights. The system uses a combination of classical information retrieval techniques (TF-IDF) and a nearest-neighbors approach for efficient document retrieval, coupled with a Large Language Model (LLM) for response generation. This provides a powerful tool for answering natural language questions about customer pain points across various financial products.

A key focus of this project is ensuring compatibility with standard enterprise environments, including Windows, and the ability to function offline without requiring specialized GPU hardware or proprietary model APIs.

## Features

- **End-to-End Data Processing**: Includes scripts for cleaning and preprocessing raw complaint data from the CFPB.
- **Classical Text Representation**: Uses TF-IDF for text vectorization, a robust and interpretable industry standard.
- **Efficient Retrieval**: Implements a `NearestNeighbors` index for fast and scalable semantic search over the complaint database.
- **Modular RAG Pipeline**: A flexible RAG core logic that retrieves relevant complaint excerpts and generates evidence-backed answers using an LLM.
- **Robust Fallback**: If a powerful LLM (like one requiring `torch`) is unavailable, the system gracefully falls back to a simpler model, ensuring the application always runs.
- **Interactive Chat Interface**: A user-friendly Streamlit application allows non-technical users to interact with the system, ask questions, and view retrieved source documents for full traceability.
- **Comprehensive Evaluation**: Jupyter notebooks are provided for each stage, from data processing to a full evaluation of the RAG pipeline's performance.

## Project Structure

```
├── app.py                        # Interactive Streamlit web app
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── data/                         # Raw and processed datasets
│   ├── complaints.csv            # Raw dataset from CFPB
│   ├── filtered_complaints.csv   # Cleaned dataset for RAG
│   └── evaluation_results.json   # Results from the evaluation notebook
├── notebooks/                    # Jupyter notebooks for each development stage
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_embedding_and_vector_store.ipynb
│   └── 03_rag_core_logic.ipynb
├── src/                          # Source code modules
│   ├── data_preprocessing.py     # Data cleaning and preparation
│   ├── text_chunking.py          # Logic for splitting text into manageable chunks
│   ├── rag_pipeline.py           # Core RAG pipeline, retriever, and generator
│   └── __init__.py
└── vector_store/                 # Persisted vector store artifacts
    ├── chunks.pkl                # Pickled text chunks
    ├── metadata.pkl              # Pickled metadata for each chunk
    ├── nn_index.pkl              # Pickled scikit-learn NearestNeighbors index
    └── tfidf_vectorizer.pkl      # Pickled TF-IDF vectorizer
```

## Setup & Usage

### Prerequisites

- Python 3.9+
- A virtual environment (e.g., `venv` or `conda`) is highly recommended.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Samuels5/Intelligent-Complaint-Analysis-for-Financial-Services.git
    cd Intelligent-Complaint-Analysis-for-Financial-Services
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    # source venv/bin/activate # On macOS/Linux
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    _Note: If you encounter issues with `torch`, you can install a CPU-only version, which is sufficient for the fallback LLM:_
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```

### Running the Project

The project is designed to be run sequentially through the Jupyter notebooks, which build the necessary data and vector store artifacts.

1.  **Run EDA & Preprocessing**:

    - Open and run the cells in `notebooks/01_eda_and_preprocessing.ipynb`.
    - This will generate the cleaned `data/filtered_complaints.csv` file.

2.  **Create the Vector Store**:

    - Open and run the cells in `notebooks/02_embedding_and_vector_store.ipynb`.
    - This will perform text chunking and create all the necessary artifacts in the `vector_store/` directory (TF-IDF vectorizer, chunks, metadata, and the nearest neighbors index).

3.  **Evaluate the RAG Pipeline**:

    - Open and run the cells in `notebooks/03_rag_core_logic.ipynb`.
    - This notebook loads the vector store, runs a series of test queries, and saves the output to `data/evaluation_results.json`. It's a great way to understand how the RAG pipeline works under the hood.

4.  **Launch the Interactive Chatbot App**:
    - Once the vector store is created, you can launch the Streamlit application.
    - From the root directory of the project, run:
      ```bash
      streamlit run app.py
      ```
    - This will open a new tab in your browser with the chat interface.

## How It Works

1.  **User Query**: The user asks a question in the Streamlit app (e.g., "Why are people unhappy with their credit reports?").
2.  **Vectorization**: The user's query is transformed into a TF-IDF vector using the pre-fitted `TfidfVectorizer`.
3.  **Retrieval**: The system uses the `NearestNeighbors` index to find the text chunks from the complaint database that are most semantically similar to the query vector.
4.  **Context Augmentation**: The retrieved text chunks are formatted and combined into a context block.
5.  **Generation**: This context, along with the original query, is passed to an LLM via a structured prompt. The LLM's task is to synthesize the information from the context to generate a concise, evidence-backed answer.
6.  **Fallback Mechanism**: If a sophisticated LLM (like `google/flan-t5-base`) fails to load (e.g., due to missing `torch` or internet issues), the system automatically switches to a basic string-matching generator. This ensures the app remains functional, providing retrieved contexts even if it cannot generate a summary.
7.  **Display**: The final answer and the source complaint excerpts are displayed in the user interface, ensuring transparency and allowing for deeper investigation.
