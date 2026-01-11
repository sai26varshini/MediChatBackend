import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.vectorstore_utils import update_faiss_index, load_or_create_index
from app.chat_utils import save_vectordb, ask_question

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class Query(BaseModel):
    question: str


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    vectordb = update_faiss_index(file_path)
    save_vectordb(vectordb)

    return {"status": "success", "message": f"{file.filename} added to knowledge base!"}


@app.post("/chat")
async def chat(query: Query):
    answer = ask_question(query.question)
    return {"answer": answer}
