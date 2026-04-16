import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_hdfc_data(file_path):
    # Load scraped HDFC data (assuming columns: 'card_name', 'features')
    df = pd.read_csv(file_path)
    loader = DataFrameLoader(df, page_content_column="features")
    documents = loader.load()
    
    # Financial data splitting: Keep features/eligibility together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)