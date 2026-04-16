import os
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from ingestion import process_hdfc_data
from retriever import get_advanced_retriever

# RAGAS Imports
from ragas.metrics import faithfulness, answer_relevancy
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper

load_dotenv()

async def run_hdfc_rag():
    # 1. Initialize Components
    chunks = process_hdfc_data("data/hdfc_cards.csv")
    retriever = get_advanced_retriever(chunks)
    
    # Use Gemini 1.5 Flash (Fast and Cost-Effective)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True
    )

    # 2. Execute a Sample Query
    query = "Compare HDFC Regalia and Millennia lounge access features."
    result = qa_chain.invoke({"query": query})
    
    print("\n--- AI RESPONSE ---")
    print(result["result"])

    # 3. RAGAS Evaluation (Grounding Check)
    # Using Gemini as the judge model for evaluation
    evaluator_llm = LangchainLLMWrapper(llm) 
    
    sample = SingleTurnSample(
        user_input=query,
        response=result["result"],
        retrieved_contexts=[doc.page_content for doc in result["source_documents"]]
    )
    
    print("\n--- PERFORMANCE EVALUATION ---")
    f_score = await faithfulness.single_turn_ascore(sample, llm=evaluator_llm)
    r_score = await answer_relevancy.single_turn_ascore(sample, llm=evaluator_llm)
    
    print(f"Grounding (Faithfulness): {f_score:.2f}")
    print(f"Answer Relevancy: {r_score:.2f}")

if __name__ == "__main__":
    # Python 3.11+ async execution
    asyncio.run(run_hdfc_rag())