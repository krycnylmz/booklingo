import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
import chromadb
from dotenv import load_dotenv

# ── config
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

CHROMA_PATH = "book_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection("book_knowledge")

embedding_model = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

# ── functions
def embed_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=120)
    chunks = splitter.split_text(text)
    print(f"📚 {len(chunks)} created text chunk")

    # # store first 100 chunks for demo you can change it
    limit = 100
    stored = 0
    for i, chunk in enumerate(chunks[:limit]):  # store first 100 chunks for demo you can change it
        emb = genai.embed_content(model=embedding_model, content=chunk)["embedding"]
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[emb],
            documents=[chunk],
            metadatas=[{"source": os.path.basename(pdf_path), "page": i}]
        )
        stored += 1

    print("✅ Embedding and storage complete.")

    return {
        "file": os.path.basename(pdf_path),
        "chunks": len(chunks),
        "added": stored,
        "collection": "book_knowledge"
    }

# ── main
if __name__ == "__main__":
    embed_pdf("data/english_book.pdf") # replace with your pdf file path if needed