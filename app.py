import streamlit as st
from src.rag_pipeline import load_rag_pipeline

st.set_page_config(page_title="Complaint Analysis Chatbot", layout="wide")
st.title("💬 Intelligent Complaint Analysis for Financial Services")
st.markdown("""
Ask any question about customer complaints across Credit Cards, Personal Loans, BNPL, Savings Accounts, and Money Transfers. 
The AI will answer using real complaint data and show the sources for transparency.
""")

# Load RAG pipeline (cached)
@st.cache_resource(show_spinner=True)
def get_rag():
    return load_rag_pipeline()

rag = get_rag()

# Session state for conversation
if 'history' not in st.session_state:
    st.session_state['history'] = []

# User input
with st.form(key="chat_form"):
    user_question = st.text_input("Type your question:", "Why are people unhappy with BNPL?", key="input")
    submit = st.form_submit_button("Ask")
    clear = st.form_submit_button("Clear Conversation")

if clear:
    st.session_state['history'] = []
    rag.clear_conversation()
    st.experimental_rerun()

if submit and user_question.strip():
    with st.spinner("Generating answer..."):
        result = rag.answer(user_question, k=5, include_sources=True)
        st.session_state['history'].append({
            'question': user_question,
            'answer': result['answer'],
            'sources': result['sources']
        })

# Display conversation history
for i, turn in enumerate(st.session_state['history']):
    st.markdown(f"**Q{i+1}: {turn['question']}**")
    st.markdown(f"> {turn['answer']}")
    if turn['sources']:
        with st.expander("Show Sources"):
            for j, src in enumerate(turn['sources'], 1):
                meta = src['metadata']
                st.markdown(f"**Source {j}:**")
                st.markdown(f"- Product: {meta.get('product', 'Unknown')}")
                st.markdown(f"- Issue: {meta.get('issue', 'Unknown')}")
                st.markdown(f"- Company: {meta.get('company', 'Unknown')}")
                st.code(src['chunk'][:500] + ("..." if len(src['chunk']) > 500 else ""), language="text")
    st.markdown("---")

st.markdown("<sub>Powered by Retrieval-Augmented Generation (RAG) and real CFPB complaint data.</sub>", unsafe_allow_html=True)
