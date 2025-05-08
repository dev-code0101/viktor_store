import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER", "documents/")
LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL_NAME", "gpt-3.5-turbo"
)  # Using a chat model for better performance
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")
CHUNKS_DATA_PATH = os.getenv("CHUNKS_DATA_PATH", "chunks_data.json")
