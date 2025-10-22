from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
from src.pdf_embedder import embed_pdf 

load_dotenv()

app = Flask(__name__)
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---- ROUTES ----
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

if __name__ == "__main__":
    app.run(port=8000, debug=True)