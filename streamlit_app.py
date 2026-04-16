import os
import ssl
import certifi
import asyncio
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# 1. THE DEFINITIVE macOS SSL BYPASS
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from urllib3.exceptions import InsecureRequestWarning
import requests
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

# 2. CORE IMPORTS
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_cohere import CohereRerank
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

# RAGAS Evaluation Imports
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

# --- STREAMLIT UI CONFIG ---
st.set_page_config(page_title="HDFC Card Advisor", page_icon="💳", layout="wide")
st.title("💳 HDFC Credit Card RAG Advisor")
st.markdown("Compare benefits, lounge access, and rewards using Gemini 2.5 & Cohere Reranking.")

# --- STEP 1: CACHED DATA & INDEXING ---
@st.cache_resource
def initialize_rag_system():
    if not os.path.exists("data/hdfc_cards.csv"):
        st.error("CSV Data not found in data/hdfc_cards.csv")
        return None

    df = pd.read_csv("data/hdfc_cards.csv")
    
    # Metadata Enrichment
    raw_docs = [
        Document(
            page_content=f"Card Name: {row['card_name']}\nFeatures: {row['features']}", 
            metadata={"name": row['card_name']}
        ) 
        for _, row in df.iterrows()
    ]

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(raw_docs)

    # Embedding & Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Hybrid Retrieval (BM25 + Vector)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=[0.4, 0.6]
    )

    # Cohere Reranking
    compressor = CohereRerank(model="rerank-english-v3.0", top_n=5)
    rerank_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )

    # LLM & QA Chain
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=rerank_retriever,
        return_source_documents=True
    )

    return qa_chain, llm, embeddings

# Initialize System
with st.spinner("🚀 Booting HDFC Financial Engine..."):
    system = initialize_rag_system()
    if system:
        qa_chain, llm, embeddings = system

# --- STEP 2: CHAT INTERFACE ---
query = st.text_input("Ask a question about HDFC Cards:", placeholder="e.g., Which card has better lounge access, Regalia or Millennia?")

if query:
    with st.spinner("Analyzing data..."):
        # Execution
        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        source_docs = result["source_documents"]

        # UI Display
        st.subheader("💡 Advisor Response")
        st.write(answer)

        with st.expander("🔍 View Grounded Sources (Verified Context)"):
            for i, doc in enumerate(source_docs):
                st.info(f"**Source {i+1}:** {doc.page_content}")

        # --- STEP 3: ASYNC RAGAS EVALUATION ---
        st.subheader("📊 Automated Quality Audit (RAGAS)")
        
        async def run_eval():
            try:
                eval_llm = LangchainLLMWrapper(llm)
                eval_embeddings = LangchainEmbeddingsWrapper(embeddings)
                sample = SingleTurnSample(
                    user_input=query,
                    response=answer,
                    retrieved_contexts=[doc.page_content for doc in source_docs]
                )
                
                f_metric = Faithfulness(llm=eval_llm)
                r_metric = AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings)
                
                f_score = await f_metric.single_turn_ascore(sample)
                r_score = await r_metric.single_turn_ascore(sample)
                
                col1, col2 = st.columns(2)
                col1.metric("Faithfulness (Grounding)", f"{f_score:.2f}")
                col2.metric("Answer Relevancy", f"{r_score:.2f}")
                
                if f_score > 0.8:
                    st.success("✅ This response is highly grounded in official data.")
            except Exception as e:
                st.warning(f"Evaluation pending or error: {e}")

        asyncio.run(run_eval())

# --- FOOTER ---
st.divider()
st.caption("Developed by Yasar | Lead Analyst | HDFC RAG Prototype 2026")