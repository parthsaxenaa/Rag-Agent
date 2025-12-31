import streamlit as st
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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Vellicate Bot", layout="wide")

# Check for API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or api_key == "your_api_key_here":
    # Fallback to sidebar if .env is not set
    with st.sidebar:
        st.title("Settings")
        api_key = st.text_input("Enter your Gemini API Key", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            genai.configure(api_key=api_key)
    
    if not api_key:
        st.warning("⚠️ Google API Key not found in .env file. Please enter it in the sidebar.")
else:
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

def create_vector_store_batched(text_chunks, batch_size=5):
    """
    Creates a FAISS vector store by processing chunks in batches to avoid rate limits.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = None
    
    total_chunks = len(text_chunks)
    progress_text = "Generating embeddings... Please wait. (Handling API Rate Limits)"
    my_bar = st.progress(0, text=progress_text)

    for i in range(0, total_chunks, batch_size):
        batch = text_chunks[i:i+batch_size]
        
        # Retry mechanism for each batch
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if vector_store is None:
                    vector_store = FAISS.from_texts(batch, embedding=embeddings)
                else:
                    new_batch_store = FAISS.from_texts(batch, embedding=embeddings)
                    vector_store.merge_from(new_batch_store)
                break # Success, exit retry loop
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = 10 * (attempt + 1)
                    st.warning(f"Rate limit hit. Waiting {wait_time}s before retrying...")
                    time.sleep(wait_time) 
                else:
                    st.error(f"Error processing batch: {e}")
                    return None
        
        # Update progress
        progress = min((i + batch_size) / total_chunks, 1.0)
        my_bar.progress(progress, text=f"Processed {min(i + batch_size, total_chunks)}/{total_chunks} chunks")
        
        # Small delay between successful batches
        time.sleep(2)

    if vector_store:
        vector_store.save_local("faiss_index")
    
    my_bar.empty()
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def handle_userinput(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        return f"Error: {e}"

# --- Main Interface ---
st.title("🤖 Chat with Vellicate Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialization logic
pdf_file_path = "complete data.pdf"

# Initialize session state for text cache if needed
if "pdf_text_cache" not in st.session_state and os.path.exists(pdf_file_path):
    # Pre-load text for fallback mode
    try:
        st.session_state.pdf_text_cache = get_pdf_text(pdf_file_path)
    except:
        st.session_state.pdf_text_cache = ""

if api_key:
    # Always show chat interface immediately
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate response
        with st.spinner("Thinking..."):
            # Hybrid Search Strategy
            if os.path.exists("faiss_index"):
                # Full RAG Mode
                response = handle_userinput(prompt)
            else:
                # Fallback: simple context mode (pass first 40k chars)
                # This gets called if index is still building or checking
                try:
                    raw_context = st.session_state.pdf_text_cache[:40000] # Safe limit for Gemini Pro
                    
                    fallback_prompt = f"""
                    You are a helpful assistant. The user is asking a question about a document.
                    The document is currently being indexed for deep search. 
                    For now, here is the beginning of the document text to help you answer.
                    
                    Context:
                    {raw_context}...
                    
                    Question: {prompt}
                    Answer:
                    """
                    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
                    res = model.invoke(fallback_prompt)
                    response = res.content + "\n\n*(Note: Deep search index is still processing. Answering based on initial text preview.)*"
                except Exception as e:
                    response = f"I'm still setting up. Please give me a moment. (Error: {e})"
        
        # Display assistant message
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Please set your GOOGLE_API_KEY in the .env file or enter it above to start.")
