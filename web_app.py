#!/usr/bin/env python3
"""
=============================================================================
 RAG Chatbot — Web Interface (Flask)
=============================================================================
Menyediakan antarmuka web untuk RAG chatbot.
Menggunakan ulang logika RAG dari rag_chatbot.py

Endpoint API:
  GET  /             → Halaman utama chatbot
  GET  /api/status   → Cek status index & jumlah chunk
  POST /api/chat     → Kirim pertanyaan, terima jawaban + sumber
=============================================================================
"""

import os
import sys
import json
import warnings
from pathlib import Path
from threading import Lock

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from rag_chatbot import (
    load_and_index_pdfs,
    load_existing_index,
    setup_rag_chain,
    BASE_DIR,
    DATA_DIR,
    FAISS_DIR,
    FAISS_INDEX_PATH,
)

warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Inisialisasi Flask ──────────────────────────────────────────────────
app = Flask(__name__)

# ─── Global state: RAG components ────────────────────────────────────────
# Disimpan sekali di awal, digunakan ulang untuk semua request
rag_components = None
rag_lock = Lock()
index_ready = False


def initialize_rag():
    """
    Memuat atau membuat index FAISS, lalu setup RAG chain.
    Dipanggil sekali saat server start.
    """
    global rag_components, index_ready

    print("=" * 60)
    print("  RAG CHATBOT - WEB INTERFACE")
    print("  LLM: Groq (Llama 3)  |  Embedding: all-MiniLM-L6-v2")
    print("  Vector Store: FAISS")
    print("=" * 60)

    with rag_lock:
        try:
            if FAISS_INDEX_PATH.exists():
                print("\nMemuat index FAISS yang sudah ada...")
                vector_store = load_existing_index()
            else:
                print("\nIndex FAISS belum ada. Membuat dari PDF...")
                vector_store = load_and_index_pdfs()

            rag = setup_rag_chain(vector_store)
            rag_components = rag
            index_ready = True

            # Hitung jumlah chunk dari index
            try:
                num_chunks = vector_store.index.ntotal
            except Exception:
                num_chunks = "?"

            print(f"\nOK RAG Chatbot siap! ({num_chunks} chunk terindex)")
            print(f"   PDF di folder: {DATA_DIR}")
            print(f"   Buka http://127.0.0.1:5000 di browser")
            print("=" * 60)

        except Exception as e:
            print(f"\n[ERROR] saat inisialisasi: {e}")
            print("   Server tetap jalan, tapi index belum tersedia.")
            print("   Upload PDF ke folder data/ lalu restart server.")
            index_ready = False


# ─── Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Halaman utama chatbot."""
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Cek status index RAG."""
    global index_ready

    with rag_lock:
        ready = index_ready

    info = {
        "ready": ready,
        "pdf_files": [f.name for f in DATA_DIR.glob("*.pdf")],
        "index_exists": FAISS_INDEX_PATH.exists(),
    }
    return jsonify(info)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Endpoint utama chat.
    Request body: {"question": "Apa isi dokumen?"}
    Response: {
      "answer": "Jawaban...",
      "sources": [{"file": "...", "page": 1, "snippet": "..."}],
      "grounded": true/false
    }
    """
    global rag_components, index_ready

    # Validasi input
    data = request.get_json(silent=True)
    if not data or "question" not in data:
        return jsonify({"error": "Field 'question' wajib diisi"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Pertanyaan tidak boleh kosong"}), 400

    # Cek apakah index sudah siap
    if not index_ready:
        return jsonify({
            "answer": "Index dokumen belum tersedia. Pastikan ada file PDF di folder data/ dan restart server.",
            "sources": [],
            "grounded": False,
        })

    with rag_lock:
        chain = rag_components["chain"]
        retriever = rag_components["retriever"]
        fmt_docs = rag_components["format_docs"]

    try:
        # --- Retrieval + Generate ---
        # Retrieval tanpa threshold (k=3). Strict grounding ditangani
        # oleh prompt LLM, bukan filter angka.
        retrieved_docs = retriever.invoke(question)

        # --- Generate Jawaban ---
        context = fmt_docs(retrieved_docs)
        response = chain.invoke({"context": context, "question": question})
        answer = response.content if hasattr(response, "content") else str(response)

        # --- Format Sumber untuk dikirim sebagai JSON ---
        sources = []
        for doc in retrieved_docs:
            metadata = doc.metadata
            sources.append({
                "file": metadata.get("source", "Tidak diketahui"),
                "page": metadata.get("page", 0) + 1,  # 0-indexed → 1-indexed
                "snippet": doc.page_content[:200].replace("\n", " ").strip(),
            })

        return jsonify({
            "answer": answer,
            "sources": sources,
            "grounded": True,
        })

    except Exception as e:
        return jsonify({
            "answer": f"Terjadi kesalahan saat memproses pertanyaan: {str(e)}",
            "sources": [],
            "grounded": False,
        })


if __name__ == "__main__":
    # Inisialisasi RAG sebelum server jalan
    initialize_rag()

    app.run(host="0.0.0.0", port=5000, debug=False)
