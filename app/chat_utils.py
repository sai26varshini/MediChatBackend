from app.vectorstore_utils import load_or_create_index
from app.config import Groq_Api_Key
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


_vectordb = None

def get_vectordb():
    global _vectordb
    if _vectordb is None:
        _vectordb = load_or_create_index()
    return _vectordb

def save_vectordb(vectordb):
    vectordb.save_local("vectorstore")

# NEW REPLACEMENT FOR RetrievalQA
def ask_question(query: str) -> str:
    vectordb = get_vectordb()
    if vectordb is None:
        return "No documents uploaded yet. Please upload PDFs."

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        groq_api_key=Groq_Api_Key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

    prompt = PromptTemplate.from_template("""
You are a professional medical assistant.
Use ONLY the provided context to answer.

⚠️ IMPORTANT RULES:
- Answer in 2–3 concise sentences.
- Do NOT add extra medical assumptions.
- If the context is unclear, say so briefly.

Context:
{context}

Question:
{question}

Short answer:
""")


    def combine_docs(docs):
        return "\n".join([doc.page_content for doc in docs])

    chain = (
        {
            "context": retriever | combine_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return chain.invoke(query).content
