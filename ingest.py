import os
import time
from PyPDF2 import PdfReader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("API Key not found in .env")
    exit(1)

genai.configure(api_key=api_key)

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store_batched(text_chunks, batch_size=3):
    print("Initializing FAISS index generation...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = None
    
    total_chunks = len(text_chunks)
    
    for i in range(0, total_chunks, batch_size):
        batch = text_chunks[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}...")
        
        # Retry mechanism for each batch
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if vector_store is None:
                    vector_store = FAISS.from_texts(batch, embedding=embeddings)
                else:
                    new_batch_store = FAISS.from_texts(batch, embedding=embeddings)
                    vector_store.merge_from(new_batch_store)
                
                # Save incrementally after every successful batch
                try:
                    vector_store.save_local("faiss_index")
                    print(f"Index updated and saved (Chunk {min(i + batch_size, total_chunks)}/{total_chunks})")
                except Exception as save_err:
                    print(f"Warning: Could not save intermediate index: {save_err}")
                
                break # Success
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = 10 * (attempt + 1)
                    print(f"Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time) 
                else:
                    print(f"Error: {e}")
                    # Continue if possible, or decide to fail. For now, we skip bad batches to keep going.
                    break 
        
        time.sleep(1) # Reduced sleep

    if vector_store:
        vector_store.save_local("faiss_index")
        print("Final Index saved to 'faiss_index'.")
    return vector_store

if __name__ == "__main__":
    pdf_path = "complete data.pdf"
    if os.path.exists(pdf_path):
        raw_text = get_pdf_text(pdf_path)
        chunks = get_text_chunks(raw_text)
        create_vector_store_batched(chunks)
    else:
        print("PDF not found")
