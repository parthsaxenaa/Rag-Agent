# Vellicate Bot 🤖

Vellicate Bot is a powerful **RAG (Retrieval-Augmented Generation)** chatbot built with **Streamlit** and **Google Gemini**. It allows you to chat with your PDF documents accurately and efficiently.

## 🚀 Features

- **Hybrid Search Architecture**: 
  - **Fast Preview Mode**: Starts answering immediately using the document's initial context while the full index builds.
  - **Deep Search Mode**: Automatically switches to a full vector search (FAISS) once indexing is complete for high-accuracy answers across the entire document.
- **Smart Background Processing**: Processes large PDFs in the background without blocking the UI, handling Google API rate limits automatically.
- **Powered by Gemini 2.5**: Uses Google's latest `gemini-2.5-flash` model for fast and intelligent responses.
- **Persistent Memory**: Remembers your conversation history within the session.

## 🛠️ Prerequisites

- Python 3.10 or higher
- A Google Cloud Gemini API Key

## 📦 Installation

1. **Clone or Download** this repository.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure API Key**:
   - Open the `.env` file in the root directory.
   - Add your API key:
     ```env
     GOOGLE_API_KEY=your_api_key_here
     ```

## 🏃‍♂️ Running the App

1. **Start the Chatbot**:
   ```bash
   streamlit run app.py
   ```
2. **Open in Browser**:
   - The app will open automatically at `http://localhost:8501`.
   - If you haven't processed the PDF before, it will start indexing in the background. You can chat immediately!

## 📂 Project Structure

- `app.py`: Main application interface and logic.
- `ingest.py`: Background script for robust PDF processing and indexing.
- `requirements.txt`: List of Python dependencies.
- `.env`: Configuration file for API keys.
- `faiss_index/`: (Created automatically) Stores the vector embeddings of your PDF.

## ⚠️ Troubleshooting

- **Rate Limits (429 Error)**: The app is designed to handle this. If you see delays, it's just the background processor waiting for the API quota to reset. It will resume automatically.
- **Model Not Found**: Ensure you are using a valid API key with access to `gemini-2.5-flash`.

---
*Built with ❤️ using LangChain and Streamlit*
