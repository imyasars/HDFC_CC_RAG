import asyncio
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

async def evaluate_rag_performance(query, response, context_documents):
    sample = SingleTurnSample(
        user_input=query,
        response=response,
        retrieved_contexts=[doc.page_content for doc in context_documents]
    )
    
    # Calculate key financial RAG metrics
    f_score = await faithfulness.single_turn_ascore(sample, llm=evaluator_llm)
    r_score = await answer_relevancy.single_turn_ascore(sample, llm=evaluator_llm)
    p_score = await context_precision.single_turn_ascore(sample, llm=evaluator_llm)
    
    return {
        "Grounding (Faithfulness)": f_score,
        "Answer Relevancy": r_score,
        "Context Precision": p_score
    }