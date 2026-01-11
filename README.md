# 🩺 MediChat Backend (FastAPI + LangChain + FAISS)
This backend processes uploaded PDF medical reports, extracts text, embeds the content into a FAISS vector database, and answers user questions using an LLM.

---
## 🚀 Features
- Upload medical report PDFs
- Extracts text (supports scanned PDFs via PyMuPDF)
- Embeds & stores all PDF data in FAISS
- Asks natural-language questions using an LLM
- Stores knowledge across multiple uploads

---
## 🛠️ Tech Stack
- FastAPI
- LangChain + HuggingFace Embeddings
- FAISS Vector Store
- PyMuPDF PDF Loader
- Python 3.11

---
## 📁 Folder Structure
backend/
│── main.py
│── requirements.txt
│── app/
│ ├── chat_utils.py
│ ├── vectorstore_utils.py
│── uploaded_pdfs/
│── vectorstore/ (created automatically)


---
## ▶️ Setup

### 1️⃣ Create environment
```bash
python -m venv medi
medi\Scripts\activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Optional: remove old knowledge
rmdir /s /q vectorstore

4️⃣ Start server
uvicorn main:app --reload

Backend runs at:
http://127.0.0.1:8000

RESPONSE:-
POST
/upload_pdf
File Uploaded

POST
/chat
Example:-
JSON body:-
{
  "question": "What does my report say?"
}
