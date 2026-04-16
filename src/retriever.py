from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank

def get_advanced_retriever(chunks):
    # 1. Semantic Search (Gemini Embeddings)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # 2. Keyword Search (BM25) - Essential for exact card name matches
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 10
    
    # 3. Hybrid Ensemble (RRF)
    ensemble = EnsembleRetriever(
        retrievers=[bm25, vector_retriever], 
        weights=[0.4, 0.6]
    )
    
    # 4. Reranking Layer (Cohere)
    # This filters the top 10 results down to the best 5 for the LLM
    compressor = CohereRerank(model="rerank-english-v3.0", top_n=5)
    return ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble
    )