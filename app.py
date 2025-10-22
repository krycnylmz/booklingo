from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
from src.pdf_embedder import embed_pdf 
import os, chromadb, google.generativeai as genai

# ──────────────────────────────────────────────────────────────
# 1️⃣ ENV + CONFIG
# ──────────────────────────────────────────────────────────────
load_dotenv()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
CHROMA_PATH = os.getenv("CHROMA_PATH", "book_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "book_knowledge")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "models/gemini-2.5-flash")

os.makedirs(UPLOAD_DIR, exist_ok=True)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# FLASK + CHROMA + GEMINI

app = Flask(__name__)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# Initialize Gemini Chat Model
model = genai.GenerativeModel(
    model_name=GENERATION_MODEL,
    system_instruction=(
        "Sen bir İngilizce öğretmenisin. "
        "Her şeyi olabildiğince sade ve anlaşılır bir şekilde açıkla ve ekstra bonus kazan."
        "Kullanıcı bir kelime veya kalıp sorduğunda önce anlamını kısaca açıkla, "
        "ardından kitapta geçen benzer kullanımlardan  en fazla 2 tane kısa örnek cümleler sun. "
        "Yanıtlarını Türkçe yaz, ancak İngilizce örnekleri koru."
        "Asla Bir İngilizce öğretmeni oldupunu söyleme."
    )
)
chat = model.start_chat(history=[])

# Routes
@app.route("/", methods=["GET"])
def index():
    return render_template("chat.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "files" not in request.files:
        return jsonify({"error": "no files field; use form-data 'files'"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "no files uploaded"}), 400

    results = []
    for f in files:
        filename = f.filename
        if not filename.lower().endswith(".pdf"):
            results.append({"file": filename, "status": "skipped", "reason": "not a pdf"})
            continue

        save_path = os.path.join(UPLOAD_DIR, filename)
        f.save(save_path)

        try:
            stats = embed_pdf(save_path)   # dict: {file, chunks, added, collection}
            results.append(stats)
        except Exception as e:
            results.append({"file": filename, "status": "error", "reason": str(e)})

    return jsonify({"ok": True, "model": os.getenv("EMBED_MODEL", "models/text-embedding-004"), "results": results})

@app.route("/chat", methods=["POST"])
def chat_query():
    data = request.get_json()
    query = data.get("query", "").strip()
    
    if not query:
        return jsonify({"error": "query is required"}), 400

    # Create embedding for the query
    q_emb = embed(query)

    # Get relevant documents from ChromaDB
    results = collection.query(query_embeddings=[q_emb], n_results=5)
    docs = results.get("documents", [[]])[0]
    context = "\n\n".join(docs) if docs else "Bağlam bulunamadı."

    # Create prompt for Gemini
    prompt = f"""
Kullanıcı, kitapta geçen "{query}" ifadesinin anlamını sordu.

=== BAĞLAM ===
{context}

=== GÖREV ===
1. "{query}" ifadesinin anlamını sade ve anlaşılır bir Türkçe ile açıkla.
   - Kısa, net ve doğal bir açıklama yap.
   - Kelime kelime çeviri yapma; anlatılmak istenen duyguyu ya da durumu açıkla.
2. Bu cümlede yazarın neyi kastettiğini ve bağlamı kısaca açıkla.
3. Cevabını Türkçe yaz. İngilizce örnekleri orijinal haliyle koru.
4. Yanıtın aşağıdaki örnek formatta olmalı:

=== ÖRNEK ÇIKTI ===
**Anlamı:** "Disheartened by the failure" ifadesi, bir başarısızlık yüzünden cesareti kırılmış, umudu azalmış veya hayal kırıklığına uğramış olmak anlamına gelir.

**Kitaptaki Kullanımı:**
"I had been disheartened by the failure of my Drawing Number One and my Drawing Number Two."
→ Burada yazar, yaptığı iki çizimin başarısız olmasından dolayı moralinin bozulduğunu ifade eder.

Cevap:
"""
    response = chat.send_message(prompt)

    # Extract full text from response
    full_text = getattr(response, "text", "")
    if not full_text and hasattr(response, "candidates"):
        full_text = "\n".join(
            p.text for c in response.candidates if hasattr(c, "content") for p in getattr(c.content, "parts", []) if hasattr(p, "text")
        )

    return jsonify({"query": query, "response": full_text})

# Utility
def embed(text: str):
    return genai.embed_content(model=EMBED_MODEL, content=text)["embedding"]

if __name__ == "__main__":
    app.run(port=8000, debug=True)