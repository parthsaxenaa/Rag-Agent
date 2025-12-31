import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def quick_start():
    print("Quick starting...")
    pdf_path = "complete data.pdf"
    if not os.path.exists(pdf_path):
        print("PDF not found")
        return

    # Read ONLY the first page for instant access
    pdf_reader = PdfReader(pdf_path)
    text = ""
    if len(pdf_reader.pages) > 0:
        text = pdf_reader.pages[0].extract_text()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    
    # Create small index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    print("Quick index created! App should be ready.")

if __name__ == "__main__":
    quick_start()
