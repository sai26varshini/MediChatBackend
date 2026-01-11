import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
VECTOR_DB_PATH = "vectorstore"

# Create embeddings once (faster)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def extract_text_from_pdf(file_path: str) -> str:
    """Loads PDF using PyMuPDFLoader (supports OCR automatically)."""
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()  # returns list of Documents
    return "\n".join([doc.page_content for doc in docs])


def create_faiss_index(texts: List[str]):
    return FAISS.from_texts(texts, embeddings)


def load_or_create_index():
    """Loads saved index if available, otherwise returns None."""
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return None


def update_faiss_index(file_path: str):
    """Extracts text from PDF → Adds to existing FAISS or creates new."""
    text = extract_text_from_pdf(file_path)
    existing = load_or_create_index()

    if existing:
        existing.add_texts([text])
        return existing
    else:
        return create_faiss_index([text])
