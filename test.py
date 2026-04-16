import os
import ssl
import certifi
import asyncio
import warnings
from dotenv import load_dotenv

# 1. THE DEFINITIVE macOS SSL BYPASS
# This forces every library to ignore local certificate verification
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Patch the global SSL context
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Disable InsecureRequestWarnings for the console output
from urllib3.exceptions import InsecureRequestWarning
import requests
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

# 2. IMPORTS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_cohere import CohereRerank
from langchain_core.documents import Document

load_dotenv()

async def test_all_connections():
    print("🚀 Running Diagnostics (Gemini 2.5 + Cohere SSL Bypass)...\n")

    # --- Test 1: Gemini 2.5 LLM ---
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        response = llm.invoke("Confirm connection.")
        print(f"✅ Gemini LLM: Success! (Response: {response.content[:20]}...)")
    except Exception as e:
        print(f"❌ Gemini LLM: Failed. Error: {e}")

    # --- Test 2: Gemini Embeddings ---
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
        vector = embeddings.embed_query("HDFC Bank")
        print(f"✅ Gemini Embeddings: Success! (Vector: {len(vector)} dims)")
    except Exception as e:
        print(f"❌ Gemini Embeddings: Failed. Error: {e}")

    # --- Test 3: Cohere Reranker ---
    try:
        # The SSL bypass above will now allow this to connect
        reranker = CohereRerank(model="rerank-english-v3.0", top_n=1)
        test_docs = [Document(page_content="HDFC Card"), Document(page_content="Weather")]
        compressed = reranker.compress_documents(test_docs, "HDFC")
        print(f"✅ Cohere Rerank: Success! (Match: {compressed[0].page_content})")
    except Exception as e:
        print(f"❌ Cohere Rerank: Failed. Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_all_connections())