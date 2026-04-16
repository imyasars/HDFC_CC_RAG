import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_hdfc_data(file_path):
    # Load your scraped HDFC data
    df = pd.read_csv(file_path)
    
    # Use the "features" column as the main text content
    loader = DataFrameLoader(df, page_content_column="features")
    documents = loader.load()
    
    # Financial data requires precise splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)