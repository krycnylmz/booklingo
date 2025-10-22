# BookLingo – RAG-Based English Reading Assistant
[Click for demo](https://www.booklingo.koraydev.com)
## Project Summary  
**BookLingo** is an English learning assistant developed as part of the Akbank GenAI Bootcamp,  
based on **Retrieval-Augmented Generation (RAG)**.  

The system allows users to upload **English books in PDF format**.  
Uploaded book texts are automatically **chunked**, **converted to vectors**  
and stored in the **ChromaDB** database.  

When a user asks about a word or phrase in the book:  
-  The RAG structure finds the contexts (example sentences) in which that word appears in the book,  
-  The Gemini model explains its meaning using this context,  
-  The web interface provides the user with a rich response including meaning + examples.  

This project was developed to demonstrate the potential of the **LLM + semantic search** combination in the field of education.

## Project Objectives
- Transform English books from “passive reading” to “active learning”  
- Teach each word or phrase **directly in the context of the author's usage**  
- Provide **contextual explanations** instead of a traditional dictionary  
- To enable users to upload their own PDF books to the system and use them as a **personal learning assistant**  

---

## About the Data Set  

The project creates its own data set from **books uploaded by the user**.  
In other words, the system **dynamically generates** the data set:

1. The user uploads an English book (PDF).  
2. Text is extracted from the PDF (`pypdfium2`).  
3. The text is split into 500-character chunks (`LangChain TextSplitter`).  
4. Each chunk is converted into numerical vectors using the **Google Gemini Embedding** model.  
5. These vectors are stored in **ChromaDB**.  

With this method, each user creates their own “mini vector database.”  

---

## Technologies Used

| Component | Technology / Library | Description |
|----------|------------------------|-----------|
| **LLM / Generation Model** | [Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs) | Response generation, context analysis |
| **Embedding Model** | `models/text-embedding-004` | Converts sentences into 768-dimensional vectors |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) | Where embeddings are stored and queried |
| **RAG Pipeline** | Manual (LangChain TextSplitter + Gemini + Chroma) | LangChain-like custom pipeline |
| **Web Framework** | Flask | API and web interface |
| **Frontend** | HTML + JS + CSS + [Marked.js](https://marked.js.org) | Markdown-supported Chat UI |
| **PDF Processing** | pypdfium2 | Extract text from PDF |

---

# Setup
1. Clone this repo
```
git clone https://github.com/krycnylmz/booklingo.git
cd booklingo
```
2. Create virtual environment 
```
python -m venv venv
```
3. Install dependencies
```
pip install -r requirements.txt
```
3. Run on terminal to set your environments
```
mv .env.example .env
# Then write your own GEMINI_API_KEY
```
4. Run the app
```
python app.py
```